"""Make absent station-hours explicit before anything is lagged.

An absent station-hour is a hole in the hourly grid, not a blank field in a row.
If lags are taken before the grid is completed, "one row back" silently becomes
"three hours back" for any station that has a hole — see docs/adr/0001.
"""

import numpy as np
import pandas as pd

from src import config

ABSENT_FLAG = "is_absent_station_hour"


def complete_hourly_grid(frame: pd.DataFrame) -> pd.DataFrame:
    """Reindex to every (station, hour) in the frame's overall window.

    Rows that did not exist are inserted with NaN predictors and flagged. The
    returned frame is new; the input is untouched.
    """
    hours = pd.date_range(
        frame[config.TIME_COL].min(), frame[config.TIME_COL].max(), freq="h"
    )
    stations = sorted(frame[config.STATION_COL].unique())
    full_index = pd.MultiIndex.from_product(
        [stations, hours], names=[config.STATION_COL, config.TIME_COL]
    )

    indexed = frame.set_index([config.STATION_COL, config.TIME_COL])
    reindexed = indexed.reindex(full_index)
    present = indexed.assign(_present=True)["_present"].reindex(full_index).fillna(False)

    return (
        reindexed.assign(**{ABSENT_FLAG: ~present.to_numpy(dtype=bool)})
        .reset_index()
        .sort_values([config.STATION_COL, config.TIME_COL])
        .reset_index(drop=True)
    )


def absent_summary(gridded: pd.DataFrame) -> pd.DataFrame:
    """Per-station count and share of absent station-hours."""
    grouped = gridded.groupby(config.STATION_COL)[ABSENT_FLAG]
    return pd.DataFrame({"absent": grouped.sum(), "slots": grouped.size()}).assign(
        share=lambda d: (d["absent"] / d["slots"]).round(4)
    )


def absent_run_lengths(gridded: pd.DataFrame) -> pd.DataFrame:
    """Lengths of consecutive runs of absent station-hours, one row per run.

    Run length drives how deep lag features can usefully go: a lag-24 feature is
    worthless wherever a 30-hour hole precedes it.
    """
    runs = []
    for station, group in gridded.groupby(config.STATION_COL, sort=True):
        flags = group[ABSENT_FLAG].to_numpy(dtype=bool)
        if not flags.any():
            continue
        boundaries = np.diff(np.concatenate(([0], flags.view(np.int8), [0])))
        starts = np.flatnonzero(boundaries == 1)
        ends = np.flatnonzero(boundaries == -1)
        times = group[config.TIME_COL].to_numpy()
        runs.extend(
            {"station": station, "start": times[start], "length_hours": int(end - start)}
            for start, end in zip(starts, ends)
        )
    return pd.DataFrame(runs, columns=["station", "start", "length_hours"])


def blank_predictor_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Count and share of blank predictors per column, ignoring absent rows."""
    real_rows = frame.loc[~frame.get(ABSENT_FLAG, pd.Series(False, index=frame.index))]
    columns = config.NUMERIC_PREDICTORS + [config.WIND_DIR_COL]
    blanks = real_rows[columns].isna().sum()
    return pd.DataFrame({"blank": blanks, "rows": len(real_rows)}).assign(
        share=lambda d: (d["blank"] / d["rows"]).round(5)
    ).sort_values("blank", ascending=False)
