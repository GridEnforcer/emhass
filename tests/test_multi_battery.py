"""Tests for multi-battery support in EMHASS optimization."""

import json
import pathlib
import pickle
import unittest

import numpy as np
import pandas as pd

from emhass.forecast import Forecast
from emhass.optimization import Optimization
from emhass.utils import (
    build_config,
    build_params,
    build_secrets,
    get_logger,
    get_root,
    get_yaml_parse,
)

root = pathlib.Path(get_root(__file__, num_parent=2))
emhass_conf = {
    "data_path": root / "data/",
    "root_path": root / "src/emhass/",
    "defaults_path": root / "src/emhass/data/config_defaults.json",
    "associations_path": root / "src/emhass/data/associations.csv",
}
logger, ch = get_logger(__name__, emhass_conf, save_to_file=False)


class TestMultiBattery(unittest.TestCase):
    """Multi-battery optimization tests."""

    def setUp(self):
        config = build_config(emhass_conf, logger, emhass_conf["defaults_path"])
        _, secrets = build_secrets(emhass_conf, logger, no_response=True)
        params = build_params(emhass_conf, secrets, config, logger)
        params["optim_conf"]["set_use_pv"] = True

        retrieve_hass_conf, optim_conf, plant_conf = get_yaml_parse(
            json.dumps(params), logger
        )
        self.retrieve_hass_conf = retrieve_hass_conf
        self.optim_conf = optim_conf
        self.plant_conf = plant_conf

        from emhass.retrieve_hass import RetrieveHass

        rh = RetrieveHass(
            retrieve_hass_conf["hass_url"],
            retrieve_hass_conf["long_lived_token"],
            retrieve_hass_conf["optimization_time_step"],
            retrieve_hass_conf["time_zone"],
            params,
            emhass_conf,
            logger,
        )
        with open(emhass_conf["data_path"] / "test_df_final.pkl", "rb") as inp:
            rh.df_final, self.days_list, self.var_list, rh.ha_config = pickle.load(inp)
            rh.var_list = self.var_list

        self.retrieve_hass_conf["sensor_power_load_no_var_loads"] = str(self.var_list[0])
        self.retrieve_hass_conf["sensor_power_photovoltaics"] = str(self.var_list[1])
        self.retrieve_hass_conf["sensor_linear_interp"] = [
            retrieve_hass_conf["sensor_power_photovoltaics"],
            retrieve_hass_conf["sensor_power_load_no_var_loads"],
        ]
        self.retrieve_hass_conf["sensor_replace_zero"] = [
            retrieve_hass_conf["sensor_power_photovoltaics"]
        ]

        rh.prepare_data(
            retrieve_hass_conf["sensor_power_load_no_var_loads"],
            load_negative=retrieve_hass_conf["load_negative"],
            set_zero_min=retrieve_hass_conf["set_zero_min"],
            var_replace_zero=retrieve_hass_conf["sensor_replace_zero"],
            var_interp=retrieve_hass_conf["sensor_linear_interp"],
        )
        self.df_input_data = rh.df_final.copy()

        fcst = Forecast(
            retrieve_hass_conf,
            optim_conf,
            plant_conf,
            params,
            emhass_conf,
            logger,
            get_data_from_file=True,
        )
        self.fcst = fcst
        df_weather = fcst.get_weather_forecast(method="csv")
        self.P_PV_forecast = fcst.get_power_from_weather(df_weather)
        self.P_load_forecast = fcst.get_load_forecast(
            method=optim_conf["load_forecast_method"]
        )
        self.df_input_data_dayahead = pd.concat(
            [self.P_PV_forecast, self.P_load_forecast], axis=1
        )
        self.df_input_data_dayahead.columns = ["P_PV_forecast", "P_load_forecast"]
        self.df_input_data_dayahead = fcst.get_load_cost_forecast(
            self.df_input_data_dayahead
        )
        self.df_input_data_dayahead = fcst.get_prod_price_forecast(
            self.df_input_data_dayahead
        )

    def _make_opt(self, optim_conf=None, plant_conf=None):
        """Build an Optimization object with given config overrides."""
        oc = optim_conf or self.optim_conf
        pc = plant_conf or self.plant_conf
        return Optimization(
            self.retrieve_hass_conf,
            oc,
            pc,
            self.fcst.var_load_cost,
            self.fcst.var_prod_price,
            "profit",
            emhass_conf,
            logger,
        )

    # ── Backwards compatibility: single battery with scalar soc_init ──

    def test_single_battery_scalar_soc_init(self):
        """Single battery with scalar soc_init produces P_batt and SOC_opt aliases."""
        self.optim_conf.update({"set_use_battery": True, "number_of_batteries": 1})
        opt = self._make_opt()
        prediction_horizon = 10
        res = opt.perform_naive_mpc_optim(
            self.df_input_data_dayahead,
            self.P_PV_forecast,
            self.P_load_forecast,
            prediction_horizon,
            soc_init=0.5,
            soc_final=0.5,
        )
        self.assertIn("P_batt", res.columns)
        self.assertIn("SOC_opt", res.columns)
        self.assertIn("P_batt0", res.columns)
        self.assertIn("SOC_opt0", res.columns)
        # P_batt should equal P_batt0
        pd.testing.assert_series_equal(res["P_batt"], res["P_batt0"], check_names=False)

    # ── Two batteries with different SOC init → different SOC trajectories ──

    def test_two_batteries_different_soc_init(self):
        """Two batteries with different soc_init should have different SOC trajectories."""
        self.optim_conf.update({
            "set_use_battery": True,
            "number_of_batteries": 2,
            "set_nocharge_from_grid_list": [False, False],
            "set_nodischarge_to_grid_list": [True, True],
        })
        self.plant_conf.update({
            "battery_discharge_power_max_list": [1000, 500],
            "battery_charge_power_max_list": [1000, 500],
            "battery_discharge_efficiency_list": [0.95, 0.90],
            "battery_charge_efficiency_list": [0.95, 0.90],
            "battery_nominal_energy_capacity_list": [5000, 3000],
            "battery_minimum_state_of_charge_list": [0.2, 0.1],
            "battery_maximum_state_of_charge_list": [0.9, 0.9],
            "battery_target_state_of_charge_list": [0.5, 0.5],
        })
        opt = self._make_opt()
        prediction_horizon = 10
        res = opt.perform_naive_mpc_optim(
            self.df_input_data_dayahead,
            self.P_PV_forecast,
            self.P_load_forecast,
            prediction_horizon,
            soc_init=[0.8, 0.3],
            soc_final=[0.5, 0.5],
        )
        self.assertIn("P_batt0", res.columns)
        self.assertIn("P_batt1", res.columns)
        self.assertIn("SOC_opt0", res.columns)
        self.assertIn("SOC_opt1", res.columns)
        # No backwards compat aliases when N>1
        self.assertNotIn("P_batt", res.columns)
        self.assertNotIn("SOC_opt", res.columns)
        # SOC trajectories should start at different values
        self.assertAlmostEqual(res["SOC_opt0"].iloc[0], 0.8, delta=0.1)
        self.assertAlmostEqual(res["SOC_opt1"].iloc[0], 0.3, delta=0.1)
        # Final SOC should be close to target
        self.assertAlmostEqual(res["SOC_opt0"].iloc[-1], 0.5, places=2)
        self.assertAlmostEqual(res["SOC_opt1"].iloc[-1], 0.5, places=2)

    # ── DC-coupled + AC-coupled battery with hybrid inverter ──

    def test_dc_and_ac_coupled_batteries_hybrid(self):
        """With a hybrid inverter, DC battery should be on DC bus,
        AC battery should bypass the inverter."""
        self.optim_conf.update({
            "set_use_battery": True,
            "number_of_batteries": 2,
            "set_nocharge_from_grid_list": [False, False],
            "set_nodischarge_to_grid_list": [True, True],
        })
        self.plant_conf.update({
            "inverter_is_hybrid": True,
            "compute_curtailment": True,
            "battery_discharge_power_max_list": [1000, 1000],
            "battery_charge_power_max_list": [1000, 1000],
            "battery_discharge_efficiency_list": [0.95, 0.95],
            "battery_charge_efficiency_list": [0.95, 0.95],
            "battery_nominal_energy_capacity_list": [5000, 5000],
            "battery_minimum_state_of_charge_list": [0.2, 0.2],
            "battery_maximum_state_of_charge_list": [0.9, 0.9],
            "battery_target_state_of_charge_list": [0.5, 0.5],
            "battery_is_dc_coupled_list": [True, False],
        })
        opt = self._make_opt()
        prediction_horizon = 10
        res = opt.perform_naive_mpc_optim(
            self.df_input_data_dayahead,
            self.P_PV_forecast,
            self.P_load_forecast,
            prediction_horizon,
            soc_init=[0.5, 0.5],
            soc_final=[0.5, 0.5],
        )
        # Both batteries should have output columns
        self.assertIn("P_batt0", res.columns)
        self.assertIn("P_batt1", res.columns)
        self.assertIn("P_hybrid_inverter", res.columns)
        # Optimization should be feasible
        self.assertTrue(len(res) == prediction_horizon)

    # ── Battery availability window ──

    def test_battery_availability_window(self):
        """Battery with restricted availability should have zero power outside window."""
        self.optim_conf.update({
            "set_use_battery": True,
            "number_of_batteries": 2,
            "set_nocharge_from_grid_list": [False, False],
            "set_nodischarge_to_grid_list": [True, True],
        })
        self.plant_conf.update({
            "battery_discharge_power_max_list": [1000, 500],
            "battery_charge_power_max_list": [1000, 500],
            "battery_discharge_efficiency_list": [0.95, 0.95],
            "battery_charge_efficiency_list": [0.95, 0.95],
            "battery_nominal_energy_capacity_list": [5000, 3000],
            "battery_minimum_state_of_charge_list": [0.2, 0.2],
            "battery_maximum_state_of_charge_list": [0.9, 0.9],
            "battery_target_state_of_charge_list": [0.5, 0.5],
        })
        opt = self._make_opt()
        prediction_horizon = 10
        # Battery 1 is only available in timesteps [3, 8)
        res = opt.perform_naive_mpc_optim(
            self.df_input_data_dayahead,
            self.P_PV_forecast,
            self.P_load_forecast,
            prediction_horizon,
            soc_init=[0.5, 0.5],
            soc_final=[0.5, 0.5],
            batt_start_timestep=[0, 3],
            batt_end_timestep=[0, 8],
        )
        self.assertIn("P_batt1", res.columns)
        # Battery 1 should be zero outside its availability window
        batt1_vals = res["P_batt1"].values
        for i in range(prediction_horizon):
            if i < 3 or i >= 8:
                self.assertAlmostEqual(
                    batt1_vals[i], 0.0,
                    msg=f"Battery 1 should be zero at timestep {i} (outside window)"
                )
        # Battery 0 has no window restriction (full horizon)
        # Just check it ran successfully
        self.assertTrue(len(res) == prediction_horizon)

    # ── soc_init as list through MPC ──

    def test_soc_init_list_backwards_compat(self):
        """Scalar soc_init should work identically for multi-battery (expanded to all)."""
        self.optim_conf.update({
            "set_use_battery": True,
            "number_of_batteries": 2,
            "set_nocharge_from_grid_list": [False, False],
            "set_nodischarge_to_grid_list": [True, True],
        })
        self.plant_conf.update({
            "battery_discharge_power_max_list": [1000, 1000],
            "battery_charge_power_max_list": [1000, 1000],
            "battery_discharge_efficiency_list": [0.95, 0.95],
            "battery_charge_efficiency_list": [0.95, 0.95],
            "battery_nominal_energy_capacity_list": [5000, 5000],
            "battery_minimum_state_of_charge_list": [0.2, 0.2],
            "battery_maximum_state_of_charge_list": [0.9, 0.9],
            "battery_target_state_of_charge_list": [0.5, 0.5],
        })
        opt = self._make_opt()
        prediction_horizon = 10
        # Scalar soc_init should be expanded to both batteries
        res = opt.perform_naive_mpc_optim(
            self.df_input_data_dayahead,
            self.P_PV_forecast,
            self.P_load_forecast,
            prediction_horizon,
            soc_init=0.6,
            soc_final=0.5,
        )
        self.assertIn("P_batt0", res.columns)
        self.assertIn("P_batt1", res.columns)
        # Both batteries start from same SOC, so first SOC values should be similar
        self.assertAlmostEqual(res["SOC_opt0"].iloc[0], res["SOC_opt1"].iloc[0], places=2)


if __name__ == "__main__":
    unittest.main()
    ch.close()
    logger.removeHandler(ch)
