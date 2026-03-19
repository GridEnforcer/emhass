#!/usr/bin/env python
"""Tests for multi-battery optimization support (CVXPY)."""

import copy
import pathlib
import pickle
import unittest

import aiofiles
import numpy as np
import orjson
import pandas as pd

from emhass.optimization import Optimization
from emhass.forecast import Forecast
from emhass.retrieve_hass import RetrieveHass
from emhass.utils import (
    build_config,
    build_params,
    build_secrets,
    get_logger,
    get_root,
    get_yaml_parse,
)

# The root folder
root = pathlib.Path(get_root(__file__, num_parent=2))
emhass_conf = {}
emhass_conf["data_path"] = root / "data/"
emhass_conf["root_path"] = root / "src/emhass/"
emhass_conf["defaults_path"] = emhass_conf["root_path"] / "data/config_defaults.json"
emhass_conf["associations_path"] = emhass_conf["root_path"] / "data/associations.csv"

logger, ch = get_logger(__name__, emhass_conf, save_to_file=False)


class TestMultiBattery(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        config = await build_config(emhass_conf, logger, emhass_conf["defaults_path"])
        _, secrets = await build_secrets(emhass_conf, logger, no_response=True)
        params = await build_params(emhass_conf, secrets, config, logger)
        params["optim_conf"]["set_use_pv"] = True
        params_json = orjson.dumps(params).decode("utf-8")
        retrieve_hass_conf, optim_conf, plant_conf = get_yaml_parse(params_json, logger)
        self.retrieve_hass_conf = retrieve_hass_conf
        self.optim_conf = optim_conf
        self.plant_conf = plant_conf

        # Load test data from pickle
        self.rh = RetrieveHass(
            retrieve_hass_conf["hass_url"],
            retrieve_hass_conf["long_lived_token"],
            retrieve_hass_conf["optimization_time_step"],
            retrieve_hass_conf["time_zone"],
            params_json,
            emhass_conf,
            logger,
        )
        async with aiofiles.open(emhass_conf["data_path"] / "test_df_final.pkl", "rb") as inp:
            contents = await inp.read()
            self.rh.df_final, self.days_list, self.var_list, self.rh.ha_config = pickle.loads(contents)
            self.rh.var_list = self.var_list
        self.retrieve_hass_conf["sensor_power_load_no_var_loads"] = str(self.var_list[0])
        self.retrieve_hass_conf["sensor_power_photovoltaics"] = str(self.var_list[1])
        self.retrieve_hass_conf["sensor_linear_interp"] = [
            retrieve_hass_conf["sensor_power_photovoltaics"],
            retrieve_hass_conf["sensor_power_load_no_var_loads"],
        ]
        self.retrieve_hass_conf["sensor_replace_zero"] = [
            retrieve_hass_conf["sensor_power_photovoltaics"]
        ]

        self.rh.prepare_data(
            self.retrieve_hass_conf["sensor_power_load_no_var_loads"],
            load_negative=self.retrieve_hass_conf["load_negative"],
            set_zero_min=self.retrieve_hass_conf["set_zero_min"],
            var_replace_zero=self.retrieve_hass_conf["sensor_replace_zero"],
            var_interp=self.retrieve_hass_conf["sensor_linear_interp"],
        )
        self.df_input_data = self.rh.df_final.copy()

        self.fcst = Forecast(
            self.retrieve_hass_conf,
            self.optim_conf,
            self.plant_conf,
            params_json,
            emhass_conf,
            logger,
            get_data_from_file=True,
        )
        self.df_weather = await self.fcst.get_weather_forecast(method="csv")
        self.p_pv_forecast = self.fcst.get_power_from_weather(self.df_weather)
        self.p_load_forecast = await self.fcst.get_load_forecast(
            method=optim_conf["load_forecast_method"]
        )
        self.df_input_data_dayahead = pd.concat(
            [self.p_pv_forecast, self.p_load_forecast], axis=1
        )
        self.df_input_data_dayahead.columns = ["p_pv_forecast", "p_load_forecast"]

    def _make_opt(self, optim_conf=None, plant_conf=None):
        """Create an Optimization instance with overridden config."""
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

    def _prepare_mpc_data(self, horizon=10):
        """Prepare data for MPC-style optimization with given horizon."""
        df = self.df_input_data_dayahead.copy()
        df = self.fcst.get_load_cost_forecast(df)
        df = self.fcst.get_prod_price_forecast(df)
        return df.iloc[:horizon]

    async def test_single_battery_scalar_soc_backwards_compat(self):
        """Single battery with scalar soc_init → P_batt and SOC_opt aliases present."""
        oc = copy.deepcopy(self.optim_conf)
        oc["set_use_battery"] = True
        oc["number_of_batteries"] = 1
        pc = copy.deepcopy(self.plant_conf)
        pc["battery_discharge_power_max"] = 1000
        pc["battery_charge_power_max"] = 1000
        pc["battery_nominal_energy_capacity"] = 5000
        pc["battery_discharge_efficiency"] = 0.95
        pc["battery_charge_efficiency"] = 0.95

        opt = self._make_opt(optim_conf=oc, plant_conf=pc)
        df = self._prepare_mpc_data(horizon=10)
        p_pv = df.iloc[:, 0].values
        p_load = df.iloc[:, 1].values
        unit_cost = df[self.fcst.var_load_cost].values
        unit_price = df[self.fcst.var_prod_price].values

        result = opt.perform_optimization(
            df, p_pv, p_load, unit_cost, unit_price,
            soc_init=0.5, soc_final=0.5,
        )

        # Must have both indexed (P_batt0, SOC_opt0) and alias (P_batt, SOC_opt)
        self.assertIn("P_batt0", result.columns)
        self.assertIn("SOC_opt0", result.columns)
        self.assertIn("P_batt", result.columns)
        self.assertIn("SOC_opt", result.columns)
        # Aliases should be equal
        np.testing.assert_array_equal(result["P_batt"].values, result["P_batt0"].values)
        np.testing.assert_array_equal(result["SOC_opt"].values, result["SOC_opt0"].values)

    async def test_two_batteries_different_soc(self):
        """Two batteries with different soc_init produce separate trajectories."""
        oc = copy.deepcopy(self.optim_conf)
        oc["set_use_battery"] = True
        oc["number_of_batteries"] = 2
        pc = copy.deepcopy(self.plant_conf)
        pc["battery_discharge_power_max_list"] = [1000, 800]
        pc["battery_charge_power_max_list"] = [1000, 800]
        pc["battery_nominal_energy_capacity_list"] = [5000, 3000]
        pc["battery_discharge_efficiency_list"] = [0.95, 0.90]
        pc["battery_charge_efficiency_list"] = [0.95, 0.90]
        pc["battery_minimum_state_of_charge_list"] = [0.2, 0.2]
        pc["battery_maximum_state_of_charge_list"] = [0.9, 0.9]

        opt = self._make_opt(optim_conf=oc, plant_conf=pc)
        df = self._prepare_mpc_data(horizon=10)
        p_pv = df.iloc[:, 0].values
        p_load = df.iloc[:, 1].values
        unit_cost = df[self.fcst.var_load_cost].values
        unit_price = df[self.fcst.var_prod_price].values

        result = opt.perform_optimization(
            df, p_pv, p_load, unit_cost, unit_price,
            soc_init=[0.8, 0.3], soc_final=[0.5, 0.5],
        )

        # Both batteries should have columns
        self.assertIn("P_batt0", result.columns)
        self.assertIn("P_batt1", result.columns)
        self.assertIn("SOC_opt0", result.columns)
        self.assertIn("SOC_opt1", result.columns)
        # No aliases for multi-battery
        self.assertNotIn("P_batt", result.columns)
        self.assertNotIn("SOC_opt", result.columns)
        # SOC trajectories should start at different values
        self.assertAlmostEqual(result["SOC_opt0"].iloc[0], 0.8, delta=0.15)
        self.assertAlmostEqual(result["SOC_opt1"].iloc[0], 0.3, delta=0.15)

    async def test_dc_ac_coupled_hybrid(self):
        """DC-coupled + AC-coupled batteries with hybrid inverter converge."""
        oc = copy.deepcopy(self.optim_conf)
        oc["set_use_battery"] = True
        oc["number_of_batteries"] = 2
        pc = copy.deepcopy(self.plant_conf)
        pc["inverter_is_hybrid"] = True
        pc["battery_discharge_power_max_list"] = [1000, 500]
        pc["battery_charge_power_max_list"] = [1000, 500]
        pc["battery_nominal_energy_capacity_list"] = [5000, 3000]
        pc["battery_discharge_efficiency_list"] = [0.95, 0.95]
        pc["battery_charge_efficiency_list"] = [0.95, 0.95]
        pc["battery_minimum_state_of_charge_list"] = [0.2, 0.2]
        pc["battery_maximum_state_of_charge_list"] = [0.9, 0.9]
        pc["battery_is_dc_coupled_list"] = [True, False]  # DC + AC

        opt = self._make_opt(optim_conf=oc, plant_conf=pc)
        df = self._prepare_mpc_data(horizon=10)
        p_pv = df.iloc[:, 0].values
        p_load = df.iloc[:, 1].values
        unit_cost = df[self.fcst.var_load_cost].values
        unit_price = df[self.fcst.var_prod_price].values

        result = opt.perform_optimization(
            df, p_pv, p_load, unit_cost, unit_price,
            soc_init=[0.5, 0.5], soc_final=[0.5, 0.5],
        )

        self.assertIn("P_batt0", result.columns)
        self.assertIn("P_batt1", result.columns)
        self.assertIn("P_hybrid_inverter", result.columns)
        # Verify optimization converged
        self.assertIn("optim_status", result.columns)
        self.assertIn("ptimal", result["optim_status"].iloc[0])

    async def test_battery_availability_window(self):
        """Battery 1 forced to zero outside [3, 8) window."""
        oc = copy.deepcopy(self.optim_conf)
        oc["set_use_battery"] = True
        oc["number_of_batteries"] = 2
        pc = copy.deepcopy(self.plant_conf)
        pc["battery_discharge_power_max_list"] = [1000, 1000]
        pc["battery_charge_power_max_list"] = [1000, 1000]
        pc["battery_nominal_energy_capacity_list"] = [5000, 5000]
        pc["battery_discharge_efficiency_list"] = [0.95, 0.95]
        pc["battery_charge_efficiency_list"] = [0.95, 0.95]
        pc["battery_minimum_state_of_charge_list"] = [0.2, 0.2]
        pc["battery_maximum_state_of_charge_list"] = [0.9, 0.9]

        opt = self._make_opt(optim_conf=oc, plant_conf=pc)
        df = self._prepare_mpc_data(horizon=10)
        p_pv = df.iloc[:, 0].values
        p_load = df.iloc[:, 1].values
        unit_cost = df[self.fcst.var_load_cost].values
        unit_price = df[self.fcst.var_prod_price].values

        result = opt.perform_optimization(
            df, p_pv, p_load, unit_cost, unit_price,
            soc_init=[0.5, 0.5], soc_final=[0.5, 0.5],
            batt_start_timestep=[0, 3],
            batt_end_timestep=[0, 8],
        )

        self.assertIn("P_batt1", result.columns)
        p_batt1 = result["P_batt1"].values
        # Outside [3, 8): timesteps 0, 1, 2, 8, 9 should be zero
        for i in [0, 1, 2, 8, 9]:
            self.assertAlmostEqual(p_batt1[i], 0.0, places=1,
                                   msg=f"Battery 1 should be zero at timestep {i}")

    async def test_scalar_soc_broadcast_two_batteries(self):
        """Scalar soc_init=0.6 should broadcast to [0.6, 0.6] for 2 batteries."""
        oc = copy.deepcopy(self.optim_conf)
        oc["set_use_battery"] = True
        oc["number_of_batteries"] = 2
        pc = copy.deepcopy(self.plant_conf)
        pc["battery_discharge_power_max_list"] = [1000, 1000]
        pc["battery_charge_power_max_list"] = [1000, 1000]
        pc["battery_nominal_energy_capacity_list"] = [5000, 5000]
        pc["battery_discharge_efficiency_list"] = [0.95, 0.95]
        pc["battery_charge_efficiency_list"] = [0.95, 0.95]
        pc["battery_minimum_state_of_charge_list"] = [0.2, 0.2]
        pc["battery_maximum_state_of_charge_list"] = [0.9, 0.9]

        opt = self._make_opt(optim_conf=oc, plant_conf=pc)
        df = self._prepare_mpc_data(horizon=10)
        p_pv = df.iloc[:, 0].values
        p_load = df.iloc[:, 1].values
        unit_cost = df[self.fcst.var_load_cost].values
        unit_price = df[self.fcst.var_prod_price].values

        # Pass scalar — should be broadcast
        result = opt.perform_optimization(
            df, p_pv, p_load, unit_cost, unit_price,
            soc_init=0.6, soc_final=0.6,
        )

        self.assertIn("SOC_opt0", result.columns)
        self.assertIn("SOC_opt1", result.columns)
        # Both should start near 0.6
        self.assertAlmostEqual(result["SOC_opt0"].iloc[0], 0.6, delta=0.15)
        self.assertAlmostEqual(result["SOC_opt1"].iloc[0], 0.6, delta=0.15)


if __name__ == "__main__":
    unittest.main()
