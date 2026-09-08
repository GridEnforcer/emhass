"""Fit metrics + seasonal-naive baseline published for REST callers (GridEnforcer ge-56k0).

A planner that trains via /action/forecast-model-fit only sees an HTTP 200;
the fit's quality lived in the log. The fit now scores itself against
"same time yesterday" on its own test window, the numbers are written to
<data_path>/ml_fit_<model_type>.json and served by
GET /api/v1/ml-fit/<model_type>, so the caller can keep its fallback in
front of a model that does not beat it.
"""

import logging
import pathlib
import tempfile
import unittest

import numpy as np
import orjson
import pandas as pd

from emhass import command_line, utils, web_server
from emhass.machine_learning_forecaster import MLForecaster

logging.getLogger("quart.app").setLevel(logging.CRITICAL)

root = pathlib.Path(utils.get_root(__file__, num_parent=2))
emhass_conf = {
    "data_path": root / "data/",
    "root_path": root / "src/emhass/",
    "config_path": root / "config.json",
}
logger, _ = utils.get_logger(__name__, emhass_conf, save_to_file=False)

VAR = "sensor.gridenforcer_core_base_load_power"


def _daily_series(days: int, noise: float, freq: str = "30min") -> pd.DataFrame:
    idx = pd.date_range("2026-06-01", periods=days * 48, freq=freq, tz="Europe/Stockholm")
    rng = np.random.default_rng(3)
    t = np.arange(len(idx))
    load = 800 + 400 * np.sin(2 * np.pi * t / 48) + rng.normal(0, noise, len(idx))
    return pd.DataFrame({VAR: load}, index=idx)


def _mlf(data: pd.DataFrame) -> MLForecaster:
    return MLForecaster(data, "load_forecast", VAR, "LinearRegression", 48, emhass_conf, logger)


class TestFitMetrics(unittest.IsolatedAsyncioTestCase):
    async def test_fit_publishes_metrics_with_naive_baseline(self):
        mlf = _mlf(_daily_series(days=14, noise=20))
        await mlf.fit(split_date_delta="24h", perform_backtest=False)
        m = mlf.fit_metrics_
        self.assertIsNotNone(m)
        self.assertEqual(m["var_model"], VAR)
        self.assertEqual(m["num_lags"], 48)
        self.assertEqual(m["freq_minutes"], 30)
        self.assertEqual(m["n_test_rows"], 48)
        self.assertEqual(m["n_train_rows"], 14 * 48 - 48)
        self.assertEqual(m["n_compared_rows"], 48)  # whole test window has a yesterday
        self.assertEqual(m["leading_gap_rows"], 0)
        for key in ("test_r2", "test_mae", "test_mae_vs_naive", "naive_mae", "naive_r2"):
            self.assertIsInstance(m[key], float, key)
        # A clean daily sine: both the model and yesterday's value are close.
        self.assertGreater(m["test_r2"], 0.9)
        self.assertGreater(m["naive_r2"], 0.9)

    async def test_naive_baseline_needs_a_yesterday(self):
        """Test window inside the first day of history: no baseline, fields None."""
        # 3 days total, 24 h test -> train = 2 days; force a 1-day series so the
        # test rows have no t-24h partner: 1 day of data, 12 h test window.
        data = _daily_series(days=1, noise=20)
        mlf = MLForecaster(data, "load_forecast", VAR, "LinearRegression", 4, emhass_conf, logger)
        await mlf.fit(split_date_delta="12h", perform_backtest=False)
        m = mlf.fit_metrics_
        self.assertEqual(m["n_compared_rows"], 0)
        self.assertIsNone(m["naive_mae"])
        self.assertIsNone(m["naive_r2"])
        self.assertIsInstance(m["test_mae"], float)

    async def test_sidecar_round_trip(self):
        mlf = _mlf(_daily_series(days=6, noise=20))
        await mlf.fit(split_date_delta="24h", perform_backtest=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp)
            self.assertIsNone(command_line.read_ml_fit_metrics(path, "load_forecast"))
            await command_line.write_ml_fit_metrics(path, "load_forecast", mlf, logger)
            back = command_line.read_ml_fit_metrics(path, "load_forecast")
        self.assertEqual(back["model_type"], "load_forecast")
        self.assertTrue(back["fitted_at"].endswith("Z"))
        self.assertEqual(back["test_mae"], mlf.fit_metrics_["test_mae"])
        self.assertEqual(back["naive_mae"], mlf.fit_metrics_["naive_mae"])


class TestMlFitEndpoint(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = web_server.app.test_client()
        self.tmp = tempfile.TemporaryDirectory()
        self._saved = dict(web_server.emhass_conf)
        web_server.emhass_conf["data_path"] = pathlib.Path(self.tmp.name)

    async def asyncTearDown(self):
        web_server.emhass_conf.clear()
        web_server.emhass_conf.update(self._saved)
        self.tmp.cleanup()

    async def test_404_when_never_fitted(self):
        r = await self.client.get("/api/v1/ml-fit/load_forecast")
        self.assertEqual(r.status_code, 404)
        body = orjson.loads(await r.get_data())
        self.assertEqual(body["status"], "no-fit")
        self.assertEqual(r.headers["Cache-Control"], "no-store")

    async def test_serves_the_sidecar(self):
        payload = {
            "model_type": "load_forecast",
            "test_r2": 0.03,
            "test_mae": 210.0,
            "naive_mae": 180.0,
        }
        (pathlib.Path(self.tmp.name) / "ml_fit_load_forecast.json").write_bytes(
            orjson.dumps(payload)
        )
        r = await self.client.get("/api/v1/ml-fit/load_forecast")
        self.assertEqual(r.status_code, 200)
        body = orjson.loads(await r.get_data())
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["naive_mae"], 180.0)

    async def test_rejects_bad_model_type(self):
        r = await self.client.get("/api/v1/ml-fit/..%2Fetc")
        self.assertIn(r.status_code, (400, 404))
