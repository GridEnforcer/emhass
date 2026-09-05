"""MLForecaster.fit on a sensor younger than the retrieval window (GridEnforcer ge-jfe7).

The websocket/statistics path returns NaN from the window start to the
sensor's first sample. EMHASS's forward linear interpolation fills interior
and trailing gaps but never leading ones, and skforecast then rejects ``y``
("has missing values") - so a young sensor could never train, on every
retry, forever. The fit now drops the leading gap and trains on the span
that exists, and says clearly when that span is too short.
"""

import logging
import pathlib
import unittest

import numpy as np
import pandas as pd

from emhass import utils
from emhass.machine_learning_forecaster import MLForecaster

root = pathlib.Path(utils.get_root(__file__, num_parent=2))
emhass_conf = {
    "data_path": root / "data/",
    "root_path": root / "src/emhass/",
    "config_path": root / "config.json",
}
logger, _ = utils.get_logger(__name__, emhass_conf, save_to_file=False)

VAR = "sensor.gridenforcer_core_base_load_power"
NUM_LAGS = 48


def _series(total_hours: int, leading_nan_hours: int) -> pd.DataFrame:
    """Hourly frame over ``total_hours`` whose first ``leading_nan_hours`` are NaN."""
    idx = pd.date_range("2026-06-01", periods=total_hours, freq="h", tz="Europe/Stockholm")
    rng = np.random.default_rng(7)
    hour = idx.hour.to_numpy()
    load = 800 + 400 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 30, total_hours)
    load[:leading_nan_hours] = np.nan
    return pd.DataFrame({VAR: load}, index=idx)


def _mlf(data: pd.DataFrame) -> MLForecaster:
    return MLForecaster(
        data, "load_forecast", VAR, "LinearRegression", NUM_LAGS, emhass_conf, logger
    )


class TestLeadingGaps(unittest.IsolatedAsyncioTestCase):
    async def test_leading_gap_is_dropped_and_fit_succeeds(self):
        """80 of 90 days NaN, 10 days real: the fit trains on the 10 days."""
        data = _series(total_hours=90 * 24, leading_nan_hours=80 * 24)
        mlf = _mlf(data)
        with self.assertLogs(logger, level=logging.WARNING) as cm:
            df_pred, _ = await mlf.fit(split_date_delta="24h", perform_backtest=False)
        self.assertIsInstance(df_pred, pd.DataFrame)
        self.assertEqual(len(mlf.data_exo), 10 * 24)
        self.assertFalse(mlf.data_train[VAR].isna().any())
        self.assertTrue(any("dropping 1920 leading rows" in m for m in cm.output))

    async def test_no_leading_gap_is_unchanged(self):
        """The default path (upstream behaviour) is untouched: no warning, same rows."""
        data = _series(total_hours=10 * 24, leading_nan_hours=0)
        mlf = _mlf(data)
        with self.assertNoLogs(logger, level=logging.WARNING):
            await mlf.fit(split_date_delta="24h", perform_backtest=False)
        self.assertEqual(len(mlf.data_exo), 10 * 24)

    async def test_all_nan_names_the_sensor(self):
        data = _series(total_hours=5 * 24, leading_nan_hours=5 * 24)
        with self.assertRaises(ValueError) as cm:
            await _mlf(data).fit(split_date_delta="24h", perform_backtest=False)
        self.assertIn("no valid samples", str(cm.exception))
        self.assertIn(VAR, str(cm.exception))

    async def test_too_short_after_gap_says_how_much_is_missing(self):
        """4 days of real data - 24 h test = 72 training rows < 2 x 48 lags.

        Customer #1 2026-09-05 shape: enough rows to fit, far too few to
        fit anything better than a constant (R2 -1.78). Must refuse."""
        data = _series(total_hours=30 * 24, leading_nan_hours=26 * 24)
        with self.assertRaises(ValueError) as cm:
            await _mlf(data).fit(split_date_delta="24h", perform_backtest=False)
        msg = str(cm.exception)
        self.assertIn("training rows", msg)
        self.assertIn(f"num_lags={NUM_LAGS} needs at least {2 * NUM_LAGS}", msg)
        self.assertIn(VAR, msg)
