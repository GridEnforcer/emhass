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
        res = _run(
            opt, df_input, p_pv, p_load, soc_init=[0.8, 0.8], soc_final=[0.8, 0.8]
        )
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
