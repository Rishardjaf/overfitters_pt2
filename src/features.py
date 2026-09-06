"""The four feature families, all strictly backward-looking.

Every window ends at the observation hour and none extends past it. Cross-station
features use other stations at the same hour, which is legal — those readings
genuinely exist at time t. See docs/adr/0001; src/leakage.py enforces it.
"""

import numpy as np
import pandas as pd

from src import config
from src.grid import ABSENT_FLAG
from src.impute import IMPUTED_SUFFIX, STALENESS_COL

NON_FEATURE_COLS = {
    config.ID_COL,
    config.TIME_COL,
    config.STATION_COL,
    config.TARGET_COL,
    config.WIND_DIR_COL,
    *config.CALENDAR_COLS,
}


def _by_station(frame: pd.DataFrame):
    return frame.groupby(config.STATION_COL, sort=False)


def _concat_new(frame: pd.DataFrame, columns: dict[str, pd.Series]) -> pd.DataFrame:
    """Attach many new columns in one allocation.

    Repeated `assign` calls fragment the block manager badly once the frame is
    over a hundred columns wide; one concat keeps it consolidated.
    """
    return pd.concat([frame, pd.DataFrame(columns, index=frame.index)], axis=1)


def add_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Values from 1, 2, 3, 6, 12 and 24 hours earlier at the same station."""
    grouped = _by_station(frame)
    lags = {
        f"{column}_lag{hours}": grouped[column].shift(hours)
        for column in config.LAGGED_COLS
        for hours in config.LAG_HOURS
    }
    return _concat_new(frame, lags)


def add_rolling_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Trailing mean and standard deviation over 3, 6 and 24 hours."""
    grouped = _by_station(frame)
    rolls = {}
    for column in config.ROLLED_COLS:
        for window in config.ROLL_WINDOWS:
            series = grouped[column]
            rolls[f"{column}_mean{window}"] = series.transform(
                lambda s, w=window: s.rolling(w, min_periods=1).mean()
            )
            rolls[f"{column}_sd{window}"] = series.transform(
                lambda s, w=window: s.rolling(w, min_periods=2).std()
            )
    return _concat_new(frame, rolls)


def add_delta_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Rate of change — the signal that separates a model from persistence.

    At a one-hour horizon the useful question is not how polluted it is but
    whether it is rising or falling, and how fast.
    """
    deltas = {}
    for column in config.ROLLED_COLS:
        deltas[f"{column}_delta1"] = frame[column] - frame[f"{column}_lag1"]
        deltas[f"{column}_delta3"] = frame[column] - frame[f"{column}_lag3"]
        deltas[f"{column}_accel"] = (frame[column] - frame[f"{column}_lag1"]) - (
            frame[f"{column}_lag1"] - frame[f"{column}_lag2"]
        )
    return _concat_new(frame, deltas)


def add_wind_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Decompose the compass wind direction into u and v components.

    Raw `wd` is a 16-point string, so "NNW" and "NW" are adjacent physically but
    arbitrary as categories. Combining direction with speed gives two continuous
    columns a model can actually use. Meteorological convention: the direction is
    where the wind comes from, hence the negative signs.
    """
    radians = np.deg2rad(frame[config.WIND_DIR_COL].map(config.WIND_DIR_DEGREES))
    speed = frame["WSPM"]
    grouped = _by_station(frame)
    return _concat_new(
        frame,
        {
            "wind_u": -speed * np.sin(radians),
            "wind_v": -speed * np.cos(radians),
            "wind_speed_mean6": grouped["WSPM"].transform(
                lambda s: s.rolling(6, min_periods=1).mean()
            ),
        },
    )


def add_stability_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Dew-point depression and rain scavenging.

    A small temperature-minus-dew-point gap means humid, stagnant, inversion-prone
    air in which particulates accumulate. Rain washes them out.
    """
    grouped = _by_station(frame)
    rain_flags = {
        f"rain_last{hours}h": grouped["RAIN"].transform(
            lambda s, w=hours: (s.rolling(w, min_periods=1).sum() > 0).astype(float)
        )
        for hours in config.RAIN_WINDOWS
    }
    return _concat_new(
        frame,
        {
            "dew_point_depression": frame["TEMP"] - frame["DEWP"],
            "pressure_delta3": frame["PRES"] - frame["PRES_lag3"],
            **rain_flags,
        },
    )


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Cyclical time encodings plus the heating-season flag.

    Hour and day-of-year are encoded as sine/cosine pairs so hour 23 sits next to
    hour 0 rather than 23 units away.
    """
    timestamps = frame[config.TIME_COL]
    hour = timestamps.dt.hour
    day_of_year = timestamps.dt.dayofyear
    month, day = timestamps.dt.month, timestamps.dt.day

    after_start = (month > config.HEATING_START[0]) | (
        (month == config.HEATING_START[0]) & (day >= config.HEATING_START[1])
    )
    before_end = (month < config.HEATING_END[0]) | (
        (month == config.HEATING_END[0]) & (day <= config.HEATING_END[1])
    )
    return _concat_new(
        frame,
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "doy_sin": np.sin(2 * np.pi * day_of_year / 365.25),
            "doy_cos": np.cos(2 * np.pi * day_of_year / 365.25),
            "day_of_week": timestamps.dt.dayofweek,
            "month_number": month,
            "is_heating_season": (after_start | before_end).astype(float),
        },
    )


def add_cross_station_features(frame: pd.DataFrame) -> pd.DataFrame:
    """City-wide context at the same hour.

    One station rising while the city is flat means something different from the
    whole city rising together.
    """
    by_hour = frame.groupby(config.TIME_COL, sort=False)[config.ANCHOR_COL]
    city_mean = by_hour.transform("mean")
    stations = pd.Categorical(frame[config.STATION_COL], categories=config.STATIONS)
    return _concat_new(
        frame,
        {
            "city_anchor_mean": city_mean,
            "city_anchor_spread": by_hour.transform("std"),
            "station_minus_city": frame[config.ANCHOR_COL] - city_mean,
            "station_code": pd.Series(stations.codes.astype(float), index=frame.index),
            "is_suburban": frame[config.STATION_COL]
            .isin(config.SUBURBAN_STATIONS)
            .astype(float),
        },
    )


def build_features(gridded: pd.DataFrame) -> pd.DataFrame:
    """Run every family in dependency order and return a new frame.

    Deltas need lags, and stability needs the lagged pressure, so the order here
    is load-bearing.
    """
    return (
        gridded.pipe(add_lag_features)
        .pipe(add_rolling_features)
        .pipe(add_delta_features)
        .pipe(add_wind_features)
        .pipe(add_stability_features)
        .pipe(add_calendar_features)
        .pipe(add_cross_station_features)
    )


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Model input columns: everything numeric that is not an identifier or target."""
    return [
        column
        for column in frame.columns
        if column not in NON_FEATURE_COLS
        and column != ABSENT_FLAG
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def dense_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Feature columns for the rungs that cannot see NaN — drops the imputation flags."""
    return [
        column
        for column in feature_columns(frame)
        if not column.endswith(IMPUTED_SUFFIX) and column != STALENESS_COL
    ]
