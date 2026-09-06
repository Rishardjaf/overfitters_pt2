"""Project-wide constants. No logic lives here."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"
ML_ALGORITHMS_DIR = PROJECT_ROOT / "ML_Algorithms"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_PATH = RAW_DIR / "train.csv"
TEST_PATH = RAW_DIR / "test.csv"

# Engineered feature frames, cached so six algorithm runs need one build.
TRAIN_PARQUET = PROCESSED_DIR / "train.parquet"
TEST_PARQUET = PROCESSED_DIR / "test.parquet"

ID_COL = "id"
TIME_COL = "observation_timestamp"
STATION_COL = "station"
TARGET_COL = "PM2_5_next_hour"
# PM10 replaces current_PM2_5 as the anchor: the revised release deletes the
# PM2.5 column, and PM10 at the observation hour carries r = +0.848 with the
# target -- the strongest remaining predictor by a wide margin.
ANCHOR_COL = "PM10"

POLLUTANT_COLS = ["PM10", "SO2", "NO2", "CO", "O3"]
MET_COLS = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
WIND_DIR_COL = "wd"
NUMERIC_PREDICTORS = POLLUTANT_COLS + MET_COLS

CALENDAR_COLS = ["year", "month", "day", "hour"]

STATIONS = [
    "Aotizhongxin", "Changping", "Dingling", "Dongsi",
    "Guanyuan", "Gucheng", "Huairou", "Nongzhanguan",
    "Shunyi", "Tiantan", "Wanliu", "Wanshouxigong",
]

# Dingling, Huairou and Changping sit outside the urban core.
SUBURBAN_STATIONS = ["Dingling", "Huairou", "Changping"]

# Feature-construction parameters.
LAG_HOURS = (1, 2, 3, 6, 12, 24)
ROLL_WINDOWS = (3, 6, 24)
LAGGED_COLS = POLLUTANT_COLS + ["TEMP", "PRES", "DEWP", "WSPM"]
# NO2 takes the slot current_PM2_5 vacated: r = +0.643, third-strongest.
ROLLED_COLS = ["PM10", "CO", "NO2", "WSPM"]
RAIN_WINDOWS = (3, 24)

# Forward-fill may carry a measurement at most this many hours forward.
FFILL_LIMIT_HOURS = 3

# Beijing municipal heating season, as (month, day) bounds. Wraps the new year.
HEATING_START = (11, 15)
HEATING_END = (3, 15)

# Season-matched validation blocks: (holdout_start, holdout_end_exclusive).
VALIDATION_BLOCKS = [
    ("2015-09-01", "2016-03-01"),
    ("2014-09-01", "2015-03-01"),
]

# Seed for every rung that randomises, so a run is reproducible.
RANDOM_SEED = 33854602

# 16-point compass to degrees, measured clockwise from north.
WIND_DIR_DEGREES = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5,
    "E": 90.0, "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
}
