#!/usr/bin/env python
"""Tests for the GridEnforcer fork extensions on top of upstream multi-battery.

Two extensions (see the fork's CLAUDE.md):

1. Per-battery availability windows — runtime params ``batt_start_timestep``
   / ``batt_end_timestep`` (naive-MPC only). An EV modelled as a battery is
   physically present only between plug-in and departure; outside its
   half-open window ``[start, end)`` it must not charge or discharge.
   Implemented as per-battery availability power-ceiling Parameters
   (``mask * power_max``) so windows are per-call value updates —
   OptimizationCache/warm-start safe, all-ones default byte-identical to
   the unwindowed problem.

2. Per-battery ``battery_is_dc_coupled`` (plant_conf, scalar or list,
   default True). An AC-coupled battery (e.g. a V2G EV charger with its own
   AC connection) enters the main AC balance directly instead of transiting
   the hybrid inverter's DC bus, caps, and conversion efficiencies.

Harness mirrors tests/test_multi_battery_optimization.py (synthetic,
self-contained, fast).
"""

import unittest

import numpy as np
import pandas as pd

from tests.test_multi_battery_optimization import (
    VALID_OPTIMAL_STATUSES,
    build_optimization,
)


def _two_battery_overrides(**extra):
    overrides = {
        "number_of_batteries": 2,
        "battery_nominal_energy_capacity": [10000, 10000],
        "battery_discharge_power_max": [3000, 3000],
        "battery_charge_power_max": [3000, 3000],
        "battery_discharge_efficiency": [1.0, 1.0],
        "battery_charge_efficiency": [1.0, 1.0],
        "battery_minimum_state_of_charge": [0.1, 0.1],
        "battery_maximum_state_of_charge": [0.9, 0.9],
        "battery_target_state_of_charge": [0.5, 0.5],
    }
    overrides.update(extra)
    return overrides


def _arbitrage_scenario(n=8):
    """Cheap early, expensive late — every battery wants to charge early and
    discharge late, so a window restriction visibly bites."""
    index = pd.date_range("2026-03-01", periods=n, freq="30min", tz="Europe/Tallinn")
    p_pv = pd.Series([0.0] * n, index=index)
    p_load = pd.Series([1000.0] * n, index=index)
    df_input = pd.DataFrame(index=index)
    df_input["unit_load_cost"] = [0.05] * (n // 2) + [0.60] * (n - n // 2)
    df_input["unit_prod_price"] = [0.01] * n
    # Load exceeds a single battery's 3 kW power max in the expensive half,
    # so BOTH batteries must participate — otherwise the index tie-break
    # sends all the work to battery 0 and window tests prove nothing.
    p_load = pd.Series([1000.0] * (n // 2) + [5000.0] * (n - n // 2), index=index)
    return index, p_pv, p_load, df_input


def _run_mpc(opt, n=8, soc_init=None, soc_final=None, **kwargs):
    index, p_pv, p_load, df_input = _arbitrage_scenario(n)
    return opt.perform_naive_mpc_optim(
        df_input,
        p_pv,
        p_load,
        n,
        soc_init=soc_init,
        soc_final=soc_final,
        def_total_hours=[],
        def_total_timestep=[],
        def_start_timestep=[],
        def_end_timestep=[],
        **kwargs,
    )


class TestBatteryAvailabilityWindows(unittest.TestCase):
    def test_window_pins_battery_to_zero_outside(self):
        opt = build_optimization(plant_overrides=_two_battery_overrides())
        opt_res = _run_mpc(
            opt,
            soc_init=[0.5, 0.5],
            soc_final=[0.5, 0.5],
            batt_start_timestep=[0, 2],
            batt_end_timestep=[0, 6],
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        p1 = opt_res["P_batt_1"].to_numpy()
        # Battery 1 window is [2, 6): pinned to zero outside it.
        np.testing.assert_allclose(p1[[0, 1, 6, 7]], 0.0, atol=1e-6)
        # Inside the window it actually acts (arbitrage makes idling costly).
        self.assertGreater(np.abs(p1[2:6]).max(), 1.0)
        # Battery 0 (no window) is unrestricted and does the late discharge.
        self.assertGreater(np.abs(opt_res["P_batt_0"].to_numpy()).max(), 1.0)

    def test_end_zero_means_horizon_end(self):
        opt = build_optimization(plant_overrides=_two_battery_overrides())
        opt_res = _run_mpc(
            opt,
            soc_init=[0.5, 0.5],
            soc_final=[0.5, 0.5],
            batt_start_timestep=[0, 3],
            batt_end_timestep=[0, 0],
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        p1 = opt_res["P_batt_1"].to_numpy()
        np.testing.assert_allclose(p1[:3], 0.0, atol=1e-6)
        self.assertGreater(np.abs(p1[3:]).max(), 1.0)

    def test_windowless_call_resets_previous_window(self):
        """A cached/reused optimizer must not leak the previous call's
        window into a windowless call."""
        opt = build_optimization(plant_overrides=_two_battery_overrides())
        _run_mpc(
            opt,
            soc_init=[0.5, 0.5],
            soc_final=[0.5, 0.5],
            batt_start_timestep=[0, 2],
            batt_end_timestep=[0, 6],
        )
        # Second call without windows: masks reset to all-ones.
        opt_res = _run_mpc(opt, soc_init=[0.5, 0.5], soc_final=[0.5, 0.5])
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        for k in range(2):
            np.testing.assert_allclose(
                opt.param_batt_avail_dis_max[k].value, 3000.0, atol=1e-6
            )
            np.testing.assert_allclose(
                opt.param_batt_avail_chg_max[k].value, 3000.0, atol=1e-6
            )
        self.assertIsNotNone(opt_res)

    def test_invalid_window_entries_ignored_with_full_availability(self):
        opt = build_optimization(plant_overrides=_two_battery_overrides())
        opt_res = _run_mpc(
            opt,
            soc_init=[0.5, 0.5],
            soc_final=[0.5, 0.5],
            batt_start_timestep=[None, "bogus"],
            batt_end_timestep=[None, "bogus"],
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        for k in range(2):
            np.testing.assert_allclose(
                opt.param_batt_avail_dis_max[k].value, 3000.0, atol=1e-6
            )
        self.assertIsNotNone(opt_res)

    def test_short_lists_pad_with_no_window(self):
        opt = build_optimization(plant_overrides=_two_battery_overrides())
        _run_mpc(
            opt,
            soc_init=[0.5, 0.5],
            soc_final=[0.5, 0.5],
            batt_start_timestep=[2],
            batt_end_timestep=[6],
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        # Battery 0 got the window; battery 1 padded to fully available.
        self.assertEqual(float(opt.param_batt_avail_dis_max[0].value[0]), 0.0)
        np.testing.assert_allclose(
            opt.param_batt_avail_dis_max[1].value, 3000.0, atol=1e-6
        )


class TestBatteryDcCoupling(unittest.TestCase):
    def _hybrid_overrides(self, is_dc_coupled):
        return _two_battery_overrides(
            inverter_is_hybrid=True,
            inverter_ac_output_max=100,  # DC bus can export almost nothing
            inverter_ac_input_max=100,
            inverter_efficiency_dc_ac=1.0,
            inverter_efficiency_ac_dc=1.0,
            battery_is_dc_coupled=is_dc_coupled,
        )

    def _load_only_scenario(self, n=6):
        index = pd.date_range("2026-03-02", periods=n, freq="30min", tz="Europe/Tallinn")
        p_pv = pd.Series([0.0] * n, index=index)
        p_load = pd.Series([1500.0] * n, index=index)
        df_input = pd.DataFrame(index=index)
        df_input["unit_load_cost"] = [0.60] * n
        df_input["unit_prod_price"] = [0.01] * n
        return index, p_pv, p_load, df_input

    def test_ac_coupled_battery_bypasses_inverter_caps(self):
        """An AC-coupled battery serves the load directly, unrestricted by
        the hybrid inverter's (here nearly zero) AC output cap; the hybrid
        balance only carries the DC-coupled battery."""
        opt = build_optimization(
            plant_overrides=self._hybrid_overrides([True, False])
        )
        index, p_pv, p_load, df_input = self._load_only_scenario()
        opt_res = opt.perform_naive_mpc_optim(
            df_input,
            p_pv,
            p_load,
            len(index),
            soc_init=[0.9, 0.9],
            soc_final=[0.1, 0.1],
            def_total_hours=[],
            def_total_timestep=[],
            def_start_timestep=[],
            def_end_timestep=[],
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        # The AC battery discharges well beyond the 100 W inverter cap.
        self.assertGreater(opt_res["P_batt_1"].to_numpy().max(), 500.0)
        # DC-bus identity: the hybrid inverter carries PV + DC battery ONLY.
        np.testing.assert_allclose(
            opt_res["P_hybrid_inverter"].to_numpy(),
            (opt_res["P_PV"] + opt_res["P_batt_0"]).to_numpy(),
            atol=1e-6,
        )

    def test_all_dc_default_keeps_upstream_balance(self):
        """Scalar default (True) reproduces the upstream all-DC hybrid
        balance: P_hybrid == P_PV + fleet P_batt. Complements upstream's own
        test_n2_hybrid_dc_bus_balance_holds pin."""
        opt = build_optimization(
            plant_overrides=self._hybrid_overrides(True) | {
                "inverter_ac_output_max": 8000,
                "inverter_ac_input_max": 8000,
            }
        )
        index, p_pv, p_load, df_input = self._load_only_scenario()
        opt_res = opt.perform_naive_mpc_optim(
            df_input,
            p_pv,
            p_load,
            len(index),
            soc_init=[0.9, 0.9],
            soc_final=[0.3, 0.3],
            def_total_hours=[],
            def_total_timestep=[],
            def_start_timestep=[],
            def_end_timestep=[],
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        np.testing.assert_allclose(
            opt_res["P_hybrid_inverter"].to_numpy(),
            (opt_res["P_PV"] + opt_res["P_batt"]).to_numpy(),
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()


class TestBatteryStartupPenalty(unittest.TestCase):
    """GridEnforcer ge-jeh: set_battery_startup_penalty — the LP-side fix
    for EV session churn (ge-i2f). Each 0→active transition costs
    penalty * battery_charge_power_max * unit_load_cost⁺ * timestep, with
    battery_initial_active exempting a session that is already running."""

    def _sliver_scenario(self, n=8):
        """Two SEPARATED marginally-profitable windows: without a penalty
        the solver happily starts two short sessions; with it, lone
        slivers stop paying for themselves."""
        index = pd.date_range(
            "2026-03-03", periods=n, freq="30min", tz="Europe/Tallinn"
        )
        p_pv = pd.Series([0.0] * n, index=index)
        # 2 kW load over 8x30min = 8 kWh; one 30-min 5 kW charge sliver
        # holds only 2.5 kWh, so serving the expensive steps from the
        # battery genuinely needs BOTH cheap slivers.
        p_load = pd.Series([2000.0] * n, index=index)
        df_input = pd.DataFrame(index=index)
        # Cheap slivers at steps 1 and 5, moderately expensive elsewhere —
        # charging in each cheap sliver and discharging later is only
        # marginally profitable.
        df_input["unit_load_cost"] = [0.30, 0.10, 0.30, 0.30, 0.30, 0.10, 0.30, 0.30]
        df_input["unit_prod_price"] = [0.02] * n
        return index, p_pv, p_load, df_input

    def _run(self, opt, initial_active=None, n=8):
        index, p_pv, p_load, df_input = self._sliver_scenario(n)
        return opt.perform_naive_mpc_optim(
            df_input,
            p_pv,
            p_load,
            n,
            soc_init=0.5,
            soc_final=0.5,
            battery_initial_active=initial_active,
            def_total_hours=[],
            def_total_timestep=[],
            def_start_timestep=[],
            def_end_timestep=[],
        )

    def _active_blocks(self, p, tol=1.0):
        """Count contiguous nonzero-power blocks."""
        active = [abs(v) > tol for v in p]
        blocks = 0
        prev = False
        for a in active:
            if a and not prev:
                blocks += 1
            prev = a
        return blocks

    def test_default_zero_penalty_is_noop(self):
        """Default 0.0 = disabled: no activity binaries, no start vars —
        structurally byte-identical (the N=1 regression pins in
        test_multi_battery_optimization.py prove the numbers)."""
        opt = build_optimization()
        self.assertFalse(opt._battery_startup_penalties_enabled())
        opt_res = self._run(opt)
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        self.assertNotIn("batt_active", opt.vars)
        self.assertNotIn("batt_start", opt.vars)
        self.assertIsNotNone(opt_res)

    def test_penalty_reduces_session_starts(self):
        opt = build_optimization(
            optim_overrides={"set_battery_startup_penalty": 5.0}
        )
        self.assertTrue(opt._battery_startup_penalties_enabled())
        opt_res = self._run(opt)
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        baseline = build_optimization()
        base_res = self._run(baseline)
        self.assertLess(
            self._active_blocks(opt_res["P_batt"].to_numpy()),
            self._active_blocks(base_res["P_batt"].to_numpy()),
        )

    def test_initial_active_makes_continuation_free(self):
        """With a session already running, an immediate first block incurs
        no start cost — the plan may keep using the battery from step 0,
        and the objective must be at least as good as the cold-start case."""
        opt_cold = build_optimization(
            optim_overrides={"set_battery_startup_penalty": 2.0}
        )
        self._run(opt_cold, initial_active=[0])
        cold_obj = opt_cold.prob.value

        opt_warm = build_optimization(
            optim_overrides={"set_battery_startup_penalty": 2.0}
        )
        self._run(opt_warm, initial_active=[1])
        warm_obj = opt_warm.prob.value
        # Objective is maximized profit (negative cost); warm start can only
        # help or equal.
        self.assertGreaterEqual(warm_obj, cold_obj - 1e-9)

    def test_per_battery_penalty_only_hits_flagged_battery(self):
        opt = build_optimization(
            plant_overrides=_two_battery_overrides(),
            optim_overrides={"set_battery_startup_penalty": [0.0, 5.0]},
        )
        index, p_pv, p_load, df_input = self._sliver_scenario()
        opt_res = opt.perform_naive_mpc_optim(
            df_input,
            p_pv,
            p_load,
            8,
            soc_init=[0.5, 0.5],
            soc_final=[0.5, 0.5],
            def_total_hours=[],
            def_total_timestep=[],
            def_start_timestep=[],
            def_end_timestep=[],
        )
        self.assertIn(opt.optim_status, VALID_OPTIMAL_STATUSES)
        # Battery 1 (heavy penalty) stays out; battery 0 does the arbitrage.
        self.assertEqual(self._active_blocks(opt_res["P_batt_1"].to_numpy()), 0)
        self.assertGreaterEqual(
            self._active_blocks(opt_res["P_batt_0"].to_numpy()), 1
        )
