"""Hourly-average, top-k soft demand charge (GridEnforcer fork, ge-g65f).

Swedish effect tariffs bill the average of the k highest clock-hour averages
of grid import in the billing month. With ``capacity_charge_hourly_average``
the LP prices ``sum_largest([horizon hourly averages] ∪ [existing top-k], k)``
instead of the per-timestep ``peak_import`` epigraph of #623, so the solver
itself decides when raising the month's peak is cheaper than serving the
energy another way. Default off must be byte-identical to #623.
"""

import logging
import pathlib
import unittest

import numpy as np
import pandas as pd

from emhass.command_line import OptimizationCache
from emhass.optimization import Optimization

TEST_ROOT = pathlib.Path(__file__).resolve().parents[1]
TZ = "Europe/Stockholm"


def build_optimization(
    optim_overrides=None,
    plant_overrides=None,
    step_minutes=15,
    opt_time_delta=5,
    costfun="cost",
) -> Optimization:
    """Single-battery Optimization with a configurable step (15 min default)."""
    logger = logging.getLogger("ge_soft_peak_test")
    logger.handlers = []
    logger.addHandler(logging.NullHandler())
    retrieve_hass_conf = {
        "optimization_time_step": pd.to_timedelta(step_minutes, "minutes"),
        "time_zone": TZ,
        "sensor_power_photovoltaics": "pv",
        "sensor_power_load_no_var_loads": "load",
    }
    optim_conf = {
        "delta_forecast_daily": pd.Timedelta(hours=5),
        "num_threads": 0,
        "set_use_battery": True,
        "set_use_pv": True,
        "set_total_pv_sell": False,
        "set_nocharge_from_grid": False,
        "set_nodischarge_to_grid": True,
        "set_battery_dynamic": False,
        "set_battery_first_priority": False,
        "battery_dynamic_max": 0.9,
        "battery_dynamic_min": -0.9,
        # A small cycle cost so the battery only moves when it pays; with a flat
        # price and no cost every zero-net schedule is equally optimal.
        "weight_battery_discharge": 0.1,
        "weight_battery_charge": 0.1,
        "battery_soc_deficit_threshold": 0.2,
        "battery_soc_deficit_cost": 0.0,
        "battery_soc_surplus_threshold": 0.9,
        "battery_soc_surplus_cost": 0.0,
        "number_of_deferrable_loads": 0,
        "nominal_power_of_deferrable_loads": [],
        "treat_deferrable_load_as_semi_cont": [],
        "set_deferrable_load_single_constant": [],
        "set_deferrable_startup_penalty": [],
        "operating_hours_of_each_deferrable_load": [],
        "start_timesteps_of_each_deferrable_load": [],
        "end_timesteps_of_each_deferrable_load": [],
        "lp_solver_timeout": 45,
        "lp_solver_mip_rel_gap": 0,
    }
    if optim_overrides:
        optim_conf.update(optim_overrides)
    plant_conf = {
        "inverter_is_hybrid": False,
        "compute_curtailment": False,
        "maximum_power_from_grid": 50000,
        "maximum_power_to_grid": 50000,
        "battery_discharge_power_max": 5000,
        "battery_charge_power_max": 5000,
        "battery_minimum_state_of_charge": 0.2,
        "battery_maximum_state_of_charge": 0.9,
        "battery_target_state_of_charge": 0.5,
        "battery_nominal_energy_capacity": 10000,
        "battery_discharge_efficiency": 1.0,
        "battery_charge_efficiency": 1.0,
        "battery_stress_cost": 0.0,
        "battery_stress_segments": 10,
    }
    if plant_overrides:
        plant_conf.update(plant_overrides)
    emhass_conf = {
        "root_path": TEST_ROOT / "src" / "emhass",
        "data_path": TEST_ROOT / "data",
    }
    return Optimization(
        retrieve_hass_conf,
        optim_conf,
        plant_conf,
        "unit_load_cost",
        "unit_prod_price",
        costfun,
        emhass_conf,
        logger,
        opt_time_delta=opt_time_delta,
    )


def scenario(start="2026-01-14 07:15", n=8, step_minutes=15, load=1000.0, pv=0.0, price=0.20):
    index = pd.date_range(start, periods=n, freq=f"{step_minutes}min", tz=TZ)
    df = pd.DataFrame(index=index)
    df["unit_load_cost"] = price
    df["unit_prod_price"] = 0.05
    p_pv = pd.Series(np.full(n, pv), index=index)
    p_load = pd.Series(np.full(n, load), index=index)
    return df, p_pv, p_load


def run_mpc(opt, df, p_pv, p_load, **kw):
    n = len(df)
    return opt.perform_naive_mpc_optim(df, p_pv, p_load, n, soc_init=0.5, soc_final=0.5, **kw)


HOURLY = {"capacity_cost_per_kw": 40.0, "capacity_charge_hourly_average": True}


class TestHourAggregation(unittest.TestCase):
    def test_hour_agg_groups_clock_hours_and_weights_energy(self):
        opt = build_optimization({**HOURLY, "capacity_charge_top_k": 3})
        df, pv, load = scenario(start="2026-01-14 14:15", n=8)
        run_mpc(opt, df, pv, load, capacity_charge_window=[1, 1, 1, 0.5, 0.5, 1, 1, 1])
        agg = opt.param_hour_agg.value
        # 14:15, 14:30, 14:45 -> row 0; 15:00..15:45 -> row 1; 16:00 -> row 2
        np.testing.assert_allclose(agg[0, :3], [0.25, 0.25, 0.25])
        np.testing.assert_allclose(agg[0, 3:], 0.0)
        np.testing.assert_allclose(agg[1, 3:7], [0.125, 0.125, 0.25, 0.25])
        np.testing.assert_allclose(agg[2, 7], 0.25)
        self.assertEqual(agg.shape, (opt._capacity_hour_rows(), 8))

    def test_hour_agg_elapsed_clips_step0_and_offset_carries_imported_energy(self):
        opt = build_optimization(HOURLY)
        df, pv, load = scenario(start="2026-01-14 14:15", n=8)
        run_mpc(
            opt,
            df,
            pv,
            load,
            current_hour_imported_wh=1200.0,
            current_hour_elapsed_h=20 / 60,  # now = 14:20 -> step 14:15-14:30 keeps 10 min
        )
        agg = opt.param_hour_agg.value
        self.assertAlmostEqual(agg[0, 0], 10 / 60, places=6)
        self.assertAlmostEqual(agg[0, 1], 0.25, places=6)
        self.assertAlmostEqual(opt.param_hour_offset.value[0], 1200.0)
        self.assertEqual(opt.param_hour_offset.value[1:].sum(), 0.0)

    def test_hour_agg_groups_across_dst_change(self):
        opt = build_optimization(HOURLY)
        # DST ends 2026-10-25 03:00 CEST -> 02:00 CET. Ten 15-min steps from
        # 01:30 CEST pass through the repeated 02:xx hour: 23:30-01:45 UTC, i.e.
        # three real clock hours, none of them double-counted.
        index = pd.date_range("2026-10-25 01:30", periods=10, freq="15min", tz=TZ)
        df = pd.DataFrame(index=index)
        df["unit_load_cost"] = 0.2
        df["unit_prod_price"] = 0.05
        pv = pd.Series(0.0, index=index)
        load = pd.Series(1000.0, index=index)
        run_mpc(opt, df, pv, load)
        agg = opt.param_hour_agg.value
        rows_used = int(np.count_nonzero(agg.sum(axis=1) > 0))
        self.assertEqual(rows_used, 3)
        np.testing.assert_allclose(agg.sum(axis=0), 0.25)
        np.testing.assert_allclose(agg[:3].sum(axis=1), [0.5, 1.0, 1.0])

    def test_existing_peaks_padded_and_truncated_never_unbounded(self):
        opt = build_optimization({**HOURLY, "capacity_charge_top_k": 3})
        df, pv, load = scenario(n=8)
        res = run_mpc(opt, df, pv, load, current_period_peaks=[5000.0])
        self.assertEqual(res["optim_status"].iloc[0], "Optimal")
        np.testing.assert_allclose(opt.param_existing_peaks.value, [5000.0, 0.0, 0.0])
        res = run_mpc(opt, df, pv, load, current_period_peaks=[1.0, 9000.0, 3000.0, 7000.0, 500.0])
        self.assertEqual(res["optim_status"].iloc[0], "Optimal")
        np.testing.assert_allclose(
            np.sort(opt.param_existing_peaks.value), [3000.0, 7000.0, 9000.0]
        )
        res = run_mpc(opt, df, pv, load, current_period_peaks=None)
        self.assertEqual(res["optim_status"].iloc[0], "Optimal")

    def test_disabled_leaves_params_zero_and_peak_import_path_intact(self):
        opt = build_optimization({"capacity_cost_per_kw": 40.0})
        df, pv, load = scenario(n=8)
        res = run_mpc(opt, df, pv, load, current_period_peaks=[5000.0])
        self.assertIn("peak_import", opt.vars)
        self.assertNotIn("planned_peak_w", res.columns)
        self.assertEqual(opt.param_hour_agg.value.sum(), 0.0)


class TestSoftPeakEconomics(unittest.TestCase):
    """A 10 kWh battery at 5 kW; 15-min steps from 07:00 so hours are whole."""

    def _spiky(self, spike_w, n=8):
        df, pv, load = scenario(start="2026-01-14 07:00", n=n, load=1000.0)
        load.iloc[4:8] = spike_w  # hour 08:00-09:00 is the spike hour
        return df, pv, load

    def test_soft_peak_shaves_hour_above_existing_topk(self):
        df, pv, load = self._spiky(5000.0)
        opt = build_optimization({**HOURLY, "capacity_charge_top_k": 3})
        res = run_mpc(opt, df, pv, load, current_period_peaks=[4000.0] * 3)
        spike_hour_grid = res["P_grid_pos"].iloc[4:8].mean()
        self.assertLess(spike_hour_grid, 4100.0)
        self.assertAlmostEqual(res["planned_peak_w"].iloc[0], spike_hour_grid, delta=1.0)
        self.assertAlmostEqual(res["planned_topk_avg_w"].iloc[0], 4000.0, delta=100.0)

    def test_no_price_means_no_shaving(self):
        df, pv, load = self._spiky(5000.0)
        opt = build_optimization(
            {"capacity_cost_per_kw": 0.0, "capacity_charge_hourly_average": True}
        )
        res = run_mpc(opt, df, pv, load, current_period_peaks=[4000.0] * 3)
        self.assertGreater(res["P_grid_pos"].iloc[4:8].mean(), 4900.0)

    def test_hour_below_kth_existing_peak_is_free(self):
        df, pv, load = self._spiky(4500.0)
        opt = build_optimization({**HOURLY, "capacity_charge_top_k": 3})
        res = run_mpc(opt, df, pv, load, current_period_peaks=[50000.0] * 3)
        self.assertGreater(res["P_grid_pos"].iloc[4:8].mean(), 4400.0)
        self.assertAlmostEqual(res["planned_topk_avg_w"].iloc[0], 50000.0, delta=1.0)

    def test_base_load_above_peak_stays_feasible_and_reports_planned_peak(self):
        """Car-dealer 2026-09-23: 72 kW base load vs a 40 kW quota. With a hard
        limit this was infeasible every cycle; the soft peak simply raises."""
        df, pv, load = scenario(start="2026-01-14 07:00", n=8, load=72000.0)
        opt = build_optimization(
            {**HOURLY, "capacity_charge_top_k": 3},
            {
                "maximum_power_from_grid": 139000,
                "battery_discharge_power_max": 1,
                "battery_charge_power_max": 1,
            },
        )
        res = run_mpc(opt, df, pv, load, current_period_peaks=[40000.0] * 3)
        self.assertEqual(res["optim_status"].iloc[0], "Optimal")
        self.assertAlmostEqual(res["planned_peak_w"].iloc[0], 72000.0, delta=50.0)
        # two whole horizon hours at 72 kW join the existing 40 kW: (72+72+40)/3
        self.assertAlmostEqual(res["planned_topk_avg_w"].iloc[0], 184000.0 / 3, delta=50.0)
        self.assertAlmostEqual(res["planned_hour0_w"].iloc[0], 72000.0, delta=50.0)

    def test_off_window_zero_multipliers_leave_plan_unchanged(self):
        df, pv, load = self._spiky(5000.0)
        base = build_optimization({"capacity_cost_per_kw": 0.0})
        ref = run_mpc(base, df, pv, load)
        opt = build_optimization({**HOURLY, "capacity_charge_top_k": 3})
        res = run_mpc(
            opt, df, pv, load, capacity_charge_window=[0] * 8, current_period_peaks=[0, 0, 0]
        )
        np.testing.assert_allclose(
            res["P_grid_pos"].to_numpy(), ref["P_grid_pos"].to_numpy(), atol=1.0
        )
        self.assertEqual(res["planned_peak_w"].iloc[0], 0.0)

    def test_planned_hour0_includes_already_imported_energy(self):
        df, pv, load = scenario(start="2026-01-14 07:30", n=8, load=2000.0)
        opt = build_optimization(
            HOURLY, {"battery_discharge_power_max": 1, "battery_charge_power_max": 1}
        )
        res = run_mpc(
            opt, df, pv, load, current_hour_imported_wh=1500.0, current_hour_elapsed_h=0.5
        )
        # 1500 Wh already + 2 kW for the remaining 30 min = 2500 Wh -> 2500 W average
        self.assertAlmostEqual(res["planned_hour0_w"].iloc[0], 2500.0, delta=5.0)


class TestStructure(unittest.TestCase):
    def test_resize_recreates_hourly_params(self):
        opt = build_optimization({**HOURLY, "capacity_charge_top_k": 2})
        df, pv, load = scenario(n=8)
        run_mpc(opt, df, pv, load)
        self.assertEqual(opt.param_hour_agg.shape[1], 8)
        df, pv, load = scenario(n=12)
        run_mpc(opt, df, pv, load)
        self.assertEqual(opt.param_hour_agg.shape[1], 12)
        self.assertEqual(opt.param_hour_agg.shape[0], opt._capacity_hour_rows())
        self.assertEqual(opt.param_existing_peaks.shape, (2,))

    def test_relaxed_retry_keeps_soft_peak_term(self):
        df, pv, load = scenario(start="2026-01-14 07:00", n=8, load=1000.0)
        load.iloc[4:8] = 5000.0
        opt = build_optimization({**HOURLY, "capacity_charge_top_k": 3})
        run_mpc(opt, df, pv, load, current_period_peaks=[4000.0] * 3)

        class _Stub:
            status = "infeasible"
            value = None

            def solve(self, *a, **k):
                return None

        real = opt.prob
        opt.prob = _Stub()
        try:
            res = run_mpc(opt, df, pv, load, current_period_peaks=[4000.0] * 3)
        finally:
            opt.prob = real
        self.assertEqual(res["optim_status"].iloc[0], "Optimal (Relaxed)")
        self.assertLess(res["P_grid_pos"].iloc[4:8].mean(), 4100.0)

    def test_cache_key_changes_on_hourly_flag_and_top_k(self):
        base = build_optimization({})
        key0 = OptimizationCache._compute_cache_key(
            base.optim_conf, base.plant_conf, "cost", base.retrieve_hass_conf
        )
        key1 = OptimizationCache._compute_cache_key(
            {**base.optim_conf, "capacity_charge_hourly_average": True},
            base.plant_conf,
            "cost",
            base.retrieve_hass_conf,
        )
        key2 = OptimizationCache._compute_cache_key(
            {**base.optim_conf, "capacity_charge_top_k": 3},
            base.plant_conf,
            "cost",
            base.retrieve_hass_conf,
        )
        self.assertNotEqual(key0, key1)
        self.assertNotEqual(key0, key2)

    def test_top_k_coercion(self):
        opt = build_optimization({**HOURLY, "capacity_charge_top_k": "3"})
        self.assertEqual(opt._capacity_top_k(), 3)
        opt = build_optimization({**HOURLY, "capacity_charge_top_k": "junk"})
        self.assertEqual(opt._capacity_top_k(), 1)
        opt = build_optimization({**HOURLY, "capacity_charge_top_k": 0})
        self.assertEqual(opt._capacity_top_k(), 1)
        self.assertTrue(
            build_optimization(
                {**HOURLY, "capacity_charge_hourly_average": "true"}
            )._capacity_hourly_enabled()
        )
        self.assertFalse(
            build_optimization(
                {"capacity_cost_per_kw": 0.0, "capacity_charge_hourly_average": True}
            )._capacity_hourly_enabled()
        )


class TestSoftPeakRuntimeRouting(unittest.IsolatedAsyncioTestCase):
    """ge-mo3z lesson: a runtime key without its associations.csv row (or its
    passed_data slot) is silently dropped by treat_runtimeparams."""

    @staticmethod
    def _emhass_conf() -> dict:
        from emhass import utils

        root = pathlib.Path(utils.get_root(__file__, num_parent=2))
        root_path = root / "src/emhass/"
        return {
            "data_path": root / "data/",
            "root_path": root_path,
            "options_path": root / "options.json",
            "config_path": root / "config.json",
            "secrets_path": root / "secrets_emhass(example).yaml",
            "legacy_config_path": pathlib.Path(utils.get_root(__file__, num_parent=1))
            / "config_emhass.yaml",
            "defaults_path": root_path / "data/config_defaults.json",
            "associations_path": root_path / "data/associations.csv",
        }

    async def _route(self, runtime_extra: dict, action: str = "naive-mpc-optim"):
        import orjson

        from emhass import utils

        emhass_conf = self._emhass_conf()
        logger, _ = utils.get_logger(__name__, emhass_conf, save_to_file=False)
        config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
        _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
        params = await utils.build_params(emhass_conf, secrets, config, logger)
        params["optim_conf"]["set_use_battery"] = True
        runtimeparams = {
            "pv_power_forecast": [100.0] * 48,
            "load_power_forecast": [100.0] * 48,
            "load_cost_forecast": [1.0] * 48,
            "prod_price_forecast": [0.5] * 48,
            "prediction_horizon": 48,
            **runtime_extra,
        }
        params_json = orjson.dumps(params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
        params_out, _, optim_conf_out, _ = await utils.treat_runtimeparams(
            runtimeparams,
            params_json,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            action,
            logger,
            emhass_conf,
        )
        if isinstance(params_out, str):
            params_out = orjson.loads(params_out)
        return params_out, optim_conf_out

    async def test_structural_options_reach_optim_conf(self):
        _, optim_conf = await self._route(
            {"capacity_charge_hourly_average": True, "capacity_charge_top_k": 3}
        )
        self.assertTrue(optim_conf["capacity_charge_hourly_average"])
        self.assertEqual(optim_conf["capacity_charge_top_k"], 3)

    async def test_structural_options_default_off(self):
        _, optim_conf = await self._route({})
        self.assertFalse(optim_conf["capacity_charge_hourly_average"])
        self.assertEqual(optim_conf["capacity_charge_top_k"], 1)

    async def test_per_call_values_reach_passed_data_for_naive_mpc(self):
        params, _ = await self._route(
            {
                "current_period_peaks": [40000.0, 38000.0, 0.0],
                "current_hour_imported_wh": 12500.0,
                "current_hour_elapsed_h": 0.35,
            }
        )
        pd_ = params["passed_data"]
        self.assertEqual(pd_["current_period_peaks"], [40000.0, 38000.0, 0.0])
        self.assertEqual(pd_["current_hour_imported_wh"], 12500.0)
        self.assertEqual(pd_["current_hour_elapsed_h"], 0.35)

    async def test_per_call_values_are_none_for_dayahead(self):
        params, _ = await self._route(
            {"current_period_peaks": [1.0], "current_hour_imported_wh": 5.0},
            action="dayahead-optim",
        )
        self.assertIsNone(params["passed_data"]["current_period_peaks"])
        self.assertIsNone(params["passed_data"]["current_hour_imported_wh"])


if __name__ == "__main__":
    unittest.main()
