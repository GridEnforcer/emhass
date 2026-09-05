"""Terminal salvage value (GridEnforcer ge-zues, xdm option B).

A battery with ``battery_salvage_price`` > 0 is excluded from the terminal
soc_final constraint; its end-of-horizon stored energy is instead rewarded in
the objective at that price. The LP then discharges only when the price beats
the salvage value and never round-trips just to satisfy a terminal target.
"""

import logging
import unittest

import numpy as np
import pandas as pd

from tests.test_multi_battery_optimization import build_optimization

logger = logging.getLogger(__name__)


def _scenario(prices, n=10, pv=0, load=1000):
    index = pd.date_range("2026-02-01", periods=n, freq="30min", tz="Europe/Tallinn")
    p_pv = pd.Series([pv] * n, index=index)
    p_load = pd.Series([load] * n, index=index)
    df_input = pd.DataFrame(index=index)
    df_input["unit_load_cost"] = prices
    df_input["unit_prod_price"] = [p * 0.9 for p in prices]
    return index, p_pv, p_load, df_input


def _run(opt, df_input, p_pv, p_load, soc_init, soc_final):
    res = opt.perform_dayahead_forecast_optim(
        df_input, p_pv, p_load, soc_init=soc_init, soc_final=soc_final
    )
    assert opt.optim_status in ("Optimal", "Optimal (Relaxed)"), opt.optim_status
    return res


class TestSalvageValue(unittest.TestCase):
    def test_salvage_battery_discharges_into_high_prices_without_refill(self):
        """Prices well above salvage everywhere: the battery discharges to
        its floor and is NOT forced back to soc_final."""
        opt = build_optimization(
            optim_overrides={"battery_salvage_price": 0.10},
        )
        # 0.30-0.40 currency/kWh... unit_load_cost is per kWh in scenario terms
        _, p_pv, p_load, df_input = _scenario([0.35] * 10)
        res = _run(opt, df_input, p_pv, p_load, soc_init=0.8, soc_final=0.8)
        # Net discharge happened (SOC ends near the floor, far below soc_final)
        final_soc = res["SOC_opt"].to_numpy()[-1]
        self.assertLess(final_soc, 0.45)
        self.assertGreaterEqual(final_soc, 0.3 - 1e-6)  # min SoC respected
        self.assertGreater(res["P_batt"].to_numpy().sum(), 0)  # net discharge

    def test_salvage_far_above_prices_hoards_cheap_energy(self):
        """Salvage far above the buy price: storing cheap grid energy for
        the horizon edge is profitable arbitrage, so the battery CHARGES
        toward its ceiling — and the soc_final target (0.3) is ignored."""
        opt = build_optimization(
            optim_overrides={"battery_salvage_price": 1.0},
        )
        _, p_pv, p_load, df_input = _scenario([0.10] * 10)
        res = _run(opt, df_input, p_pv, p_load, soc_init=0.5, soc_final=0.3)
        final_soc = res["SOC_opt"].to_numpy()[-1]
        self.assertGreater(final_soc, 0.75)  # toward max (0.8)
        self.assertLess(res["P_batt"].to_numpy().sum(), 0)  # net charge

    def test_salvage_inside_efficiency_deadband_holds(self):
        """Salvage between price*eff_chg and price/eff_dis: neither selling
        nor hoarding clears the round-trip losses — the battery holds.
        (In production the cycle weights widen this deadband further.)"""
        opt = build_optimization(
            optim_overrides={"battery_salvage_price": 0.31},
            plant_overrides={
                "battery_discharge_efficiency": 0.9,
                "battery_charge_efficiency": 0.9,
            },
        )
        _, p_pv, p_load, df_input = _scenario([0.30] * 10)
        res = _run(opt, df_input, p_pv, p_load, soc_init=0.6, soc_final=0.3)
        final_soc = res["SOC_opt"].to_numpy()[-1]
        self.assertAlmostEqual(final_soc, 0.6, delta=0.01)
        np.testing.assert_allclose(res["P_batt"].to_numpy(), 0.0, atol=1e-6)

    def test_pinned_battery_still_honours_soc_final(self):
        """salvage price 0 (default): terminal target semantics unchanged."""
        opt = build_optimization()
        _, p_pv, p_load, df_input = _scenario([0.35] * 10)
        res = _run(opt, df_input, p_pv, p_load, soc_init=0.8, soc_final=0.8)
        final_soc = res["SOC_opt"].to_numpy()[-1]
        self.assertAlmostEqual(final_soc, 0.8, delta=0.01)

    def test_mixed_fleet_pinned_and_salvage(self):
        """Battery 0 pinned, battery 1 salvage: 0 round-trips to its target,
        1 net-discharges into the high prices."""
        opt = build_optimization(
            optim_overrides={
                "battery_salvage_price": [0.0, 0.10],
            },
            plant_overrides={
                "battery_discharge_power_max": [5000, 5000],
                "battery_charge_power_max": [5000, 5000],
                "battery_minimum_state_of_charge": [0.3, 0.3],
                "battery_maximum_state_of_charge": [0.9, 0.9],
                "battery_target_state_of_charge": [0.6, 0.6],
                "battery_nominal_energy_capacity": [10000, 10000],
                "battery_discharge_efficiency": [1.0, 1.0],
                "battery_charge_efficiency": [1.0, 1.0],
                "battery_stress_cost": [0.0, 0.0],
                "number_of_batteries": 2,
            },
        )
        _, p_pv, p_load, df_input = _scenario([0.35] * 10, load=2000)
        res = _run(opt, df_input, p_pv, p_load, soc_init=[0.8, 0.8], soc_final=[0.8, 0.8])
        soc0 = res["SOC_opt_0"].to_numpy()[-1]
        soc1 = res["SOC_opt_1"].to_numpy()[-1]
        self.assertAlmostEqual(soc0, 0.8, delta=0.01)  # pinned round-trips
        self.assertLess(soc1, 0.45)  # salvage sells down toward its floor

    def test_price_list_normalisation(self):
        opt = build_optimization(
            optim_overrides={"battery_salvage_price": 0.25},
        )
        self.assertEqual(opt._battery_salvage_price_list(), [0.25])
        self.assertEqual(opt._battery_salvage_mask(), [True])
        opt2 = build_optimization()
        self.assertEqual(opt2._battery_salvage_mask(), [False])


if __name__ == "__main__":
    unittest.main()


class TestSalvagePriceRuntimeRouting(unittest.IsolatedAsyncioTestCase):
    """ge-mo3z: the salvage price arrives as a RUNTIME param over the REST
    API (GridEnforcer plugin sends ``battery_salvage_price`` next to the other
    per-battery arrays). treat_runtimeparams routes runtime keys into
    optim_conf only through associations.csv; without a row there the key is
    silently dropped, check_batt_params fills the 0.0 default, the salvage
    mask is all-False and the terminal soc_final pin is built as upstream.
    The tests above inject via optim_conf directly and never exercised this.
    """

    @staticmethod
    def _emhass_conf() -> dict:
        import pathlib

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

    async def _route(self, runtime_extra: dict, num_batteries: int = 1) -> dict:
        import orjson

        from emhass import utils

        emhass_conf = self._emhass_conf()
        logger, _ = utils.get_logger(__name__, emhass_conf, save_to_file=False)
        config = await utils.build_config(emhass_conf, logger, emhass_conf["defaults_path"])
        _, secrets = await utils.build_secrets(emhass_conf, logger, no_response=True)
        params = await utils.build_params(emhass_conf, secrets, config, logger)
        params["plant_conf"]["number_of_batteries"] = num_batteries
        params["optim_conf"]["set_use_battery"] = True
        runtimeparams = {
            "pv_power_forecast": [100.0] * 48,
            "load_power_forecast": [100.0] * 48,
            "load_cost_forecast": [1.0] * 48,
            "prod_price_forecast": [0.5] * 48,
            "prediction_horizon": 48,
            "number_of_batteries": num_batteries,
            **runtime_extra,
        }
        params_json = orjson.dumps(params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = utils.get_yaml_parse(params_json, logger)
        _, _, optim_conf_out, _ = await utils.treat_runtimeparams(
            runtimeparams,
            params_json,
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            "naive-mpc-optim",
            logger,
            emhass_conf,
        )
        return optim_conf_out

    async def test_scalar_salvage_price_reaches_optim_conf(self):
        optim_conf = await self._route({"battery_salvage_price": 2.4163})
        self.assertEqual(optim_conf["battery_salvage_price"], 2.4163)

    async def test_per_battery_salvage_list_reaches_optim_conf(self):
        optim_conf = await self._route({"battery_salvage_price": [0.0, 1.7]}, num_batteries=2)
        self.assertEqual(optim_conf["battery_salvage_price"], [0.0, 1.7])

    async def test_absent_salvage_price_defaults_to_off(self):
        optim_conf = await self._route({})
        self.assertEqual(optim_conf["battery_salvage_price"], 0.0)

    async def test_startup_penalty_routes_the_same_way(self):
        """Regression guard for the sibling fork param that IS routed — if
        this passes and the salvage tests fail, the gap is the CSV row."""
        optim_conf = await self._route({"set_battery_startup_penalty": 1.0})
        self.assertEqual(optim_conf["set_battery_startup_penalty"], 1.0)
