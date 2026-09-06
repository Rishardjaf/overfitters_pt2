"""Synthetic frames that exercise every case the real data contains.

`synthetic_raw` is sixty hours with a deliberate hole in the hourly grid and a
deliberate blank field in a row that does exist — the two conditions that must
never be confused.

`synthetic_labelled` spans two years so the season-matched folds have blocks to
land on, and carries a next-hour target so the model rungs can actually be fitted.
"""

import numpy as np
import pandas as pd
import pytest

from src import config

HOURS = 60
HOLE_START, HOLE_LENGTH = 20, 4
BLANK_ROW = 35

# The labelled fixture: long enough for one season-matched fold, small enough to
# stay a unit test.
# April to November over three stations is ~17.5k station-hours, of which the
# pre-September training half clears `season_matched_folds`' real 10,000-row
# floor — so the folds under test are built the same way the real ones are.
LABELLED_START = "2015-04-01"
LABELLED_END = "2015-11-30 23:00"
LABELLED_STATIONS = 3
LABELLED_BLOCKS = [("2015-09-01", "2015-12-01")]
BLANK_SHARE = 0.02


@pytest.fixture
def synthetic_raw() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    timestamps = pd.date_range("2015-01-01", periods=HOURS, freq="h")
    frames = []
    for offset, station in enumerate(config.STATIONS[:2]):
        frame = pd.DataFrame({config.TIME_COL: timestamps})
        frame[config.STATION_COL] = station
        frame[config.ID_COL] = [f"AQ_{station[:3]}_{i:03d}" for i in range(HOURS)]
        for column in config.NUMERIC_PREDICTORS:
            frame[column] = rng.normal(50 + offset * 10, 5, HOURS).round(2)
        frame[config.WIND_DIR_COL] = rng.choice(list(config.WIND_DIR_DEGREES), HOURS)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    # An absent station-hour: the rows are removed entirely, for one station.
    hole = combined[config.STATION_COL] == config.STATIONS[0]
    hole &= combined[config.TIME_COL].isin(timestamps[HOLE_START : HOLE_START + HOLE_LENGTH])
    combined = combined.loc[~hole].reset_index(drop=True)

    # A blank predictor: the row exists, one field is empty.
    blank = combined[config.STATION_COL] == config.STATIONS[1]
    blank &= combined[config.TIME_COL] == timestamps[BLANK_ROW]
    combined.loc[blank, config.ANCHOR_COL] = np.nan

    return combined.sort_values([config.STATION_COL, config.TIME_COL]).reset_index(drop=True)


def _autoregressive_pollution(rng, hours: int, level: float) -> np.ndarray:
    """A persistent, right-skewed series with a daily cycle — PM2.5 in miniature.

    Persistence has to be a genuinely strong baseline for the ladder tests to mean
    anything, so the series is built to have real hour-to-hour momentum rather
    than being noise around a mean.
    """
    daily = 8 * np.sin(2 * np.pi * np.arange(hours) / 24)
    series = np.empty(hours)
    series[0] = level
    for step in range(1, hours):
        series[step] = 0.92 * series[step - 1] + 0.08 * level + rng.normal(0, 4)
    return np.clip(series + daily, 1, None)


@pytest.fixture
def synthetic_labelled() -> pd.DataFrame:
    """Three stations across a Jun-Nov window, carrying a next-hour target.

    The target is the same station's PM2.5 one hour later, exactly as the
    competition defines it. Rows whose next hour does not exist are dropped, the
    way the organisers dropped theirs.
    """
    rng = np.random.default_rng(7)
    timestamps = pd.date_range(LABELLED_START, LABELLED_END, freq="h")
    hours = len(timestamps)

    frames = []
    for offset, station in enumerate(config.STATIONS[:LABELLED_STATIONS]):
        frame = pd.DataFrame({config.TIME_COL: timestamps})
        frame[config.STATION_COL] = station
        frame[config.ID_COL] = [f"AQ_{station[:3]}_{i:05d}" for i in range(hours)]

        pollution = _autoregressive_pollution(rng, hours, 45 + offset * 15)
        frame[config.ANCHOR_COL] = pollution.round(1)
        for column in config.POLLUTANT_COLS[1:]:
            frame[column] = (pollution * rng.uniform(0.4, 1.6) + rng.normal(0, 5, hours)).round(1)
        for column in config.MET_COLS:
            frame[column] = rng.normal(15, 6, hours).round(1)
        frame["RAIN"] = (rng.random(hours) > 0.95).astype(float)
        frame[config.WIND_DIR_COL] = rng.choice(list(config.WIND_DIR_DEGREES), hours)

        # The target: this station's PM2.5 one hour later.
        frame[config.TARGET_COL] = frame[config.ANCHOR_COL].shift(-1)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    # Blank predictors, scattered the way the real data carries them.
    for column in (config.ANCHOR_COL, "CO", "TEMP"):
        blanks = rng.random(len(combined)) < BLANK_SHARE
        combined.loc[blanks, column] = np.nan

    # Excluded rows: no valid next-hour target means the row was never supplied.
    combined = combined.dropna(subset=[config.TARGET_COL])
    return combined.sort_values([config.STATION_COL, config.TIME_COL]).reset_index(drop=True)
