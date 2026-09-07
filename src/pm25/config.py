"""Paths, column groups, and competition constants.

This module is pure specification — it encodes facts measured from the provided
data and the competition rules. Import from here rather than re-deriving
constants, so that a change to the schema surfaces in exactly one place.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths ------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "Data"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

CONFIG_DIR = ROOT / "configs"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
OOF_DIR = ARTIFACT_DIR / "oof"
LOG_DIR = ARTIFACT_DIR / "logs"
SUBMISSION_DIR = ROOT / "submissions"
REPORT_DIR = ROOT / "reports"

# --- Schema -----------------------------------------------------------------

ID_COL = "id"
TIME_COL = "observation_timestamp"
STATION_COL = "station"
TARGET = "PM2_5_next_hour"

#: The persistence anchor. Predicting this column verbatim scores RMSE 19.67 on
#: train; every model is measured against that.
ANCHOR = "current_PM2_5"

POLLUTANT_COLS = ["current_PM2_5", "PM10", "SO2", "NO2", "CO", "O3"]
WEATHER_COLS = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
CALENDAR_COLS = ["year", "month", "day", "hour"]
CATEGORICAL_COLS = ["station", "wd"]

NUMERIC_COLS = POLLUTANT_COLS + WEATHER_COLS

STATIONS = [
    "Aotizhongxin",
    "Changping",
    "Dingling",
    "Dongsi",
    "Guanyuan",
    "Gucheng",
    "Huairou",
    "Nongzhanguan",
    "Shunyi",
    "Tiantan",
    "Wanliu",
    "Wanshouxigong",
]

#: 16 compass points, ordered clockwise from north. Blank `wd` is kept as its own
#: category (missing 0.22% in train but 1.97% in test — do not impute to modal).
# fmt: off
WIND_DIRECTIONS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]
# fmt: on
WD_MISSING = "UNKNOWN"

# --- Split boundaries (measured from the provided files) --------------------

TRAIN_START = "2013-03-01 00:00:00"
TRAIN_END = "2016-08-31 22:00:00"
TEST_START = "2016-08-31 23:00:00"
TEST_END = "2017-02-28 22:00:00"

N_TRAIN_ROWS = 360_954
N_TEST_ROWS = 51_063

#: Test covers September-February only, and is materially more polluted than the
#: training average (mean current_PM2_5 92.3 vs 78.0). Validation folds that
#: include spring/summer will read optimistically.
TEST_MONTHS = [9, 10, 11, 12, 1, 2]
HEATING_SEASON_MONTHS = TEST_MONTHS

# --- Reference scores -------------------------------------------------------

#: RMSE of predicting the global mean, on full train.
BASELINE_MEAN_RMSE = 77.78
#: RMSE of predicting `current_PM2_5` verbatim, on full train.
BASELINE_PERSISTENCE_RMSE = 19.67
#: The same persistence baseline restricted to 2016-03-01 onward.
BASELINE_PERSISTENCE_RECENT_RMSE = 15.24

#: A validation RMSE below this is implausible for an honest model and almost
#: certainly indicates target leakage. Investigate before trusting it.
LEAKAGE_SUSPICION_RMSE = 8.0

SEED = 42
