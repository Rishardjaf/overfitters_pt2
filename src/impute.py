"""Fill blank predictors without ever reading across a gap.

Forward-fill only: it carries a real earlier measurement forward, which is legal
under docs/adr/0001. Interpolation and backward fill both read the value after
the gap and are therefore never used.

Two stages, per docs/adr/0003. Every rung gets `forward_fill_with_flags`. Only
ridge and SARIMAX get the median fill on top, and its medians are fitted on
training rows alone so the transform carries no information from the holdout.
"""

import pandas as pd

from src import config

IMPUTED_SUFFIX = "_is_imputed"
STALENESS_COL = "hours_since_anchor_observation"


def forward_fill_with_flags(
    gridded: pd.DataFrame,
    columns: list[str] | None = None,
    limit: int = config.FFILL_LIMIT_HOURS,
) -> pd.DataFrame:
    """Forward-fill within each station, capped at `limit` hours of staleness.

    Adds one `<column>_is_imputed` flag per filled column so no fabricated value
    is ever indistinguishable from a measurement. Returns a new frame.
    """
    target_columns = columns or config.NUMERIC_PREDICTORS
    grouped = gridded.groupby(config.STATION_COL, sort=False)

    flags = {
        f"{column}{IMPUTED_SUFFIX}": gridded[column].isna()
        & grouped[column].ffill(limit=limit).notna()
        for column in target_columns
    }
    filled = {
        column: grouped[column].ffill(limit=limit) for column in target_columns
    }
    wind = {
        config.WIND_DIR_COL: grouped[config.WIND_DIR_COL].ffill(limit=limit)
    }
    return gridded.assign(**filled, **wind, **flags)


def add_staleness(gridded: pd.DataFrame) -> pd.DataFrame:
    """Hours since the anchor column last carried a real measurement.

    Lets a model discount a lag that is merely a stale copy of an older reading.
    """
    observed = gridded[config.ANCHOR_COL].notna()
    grouped = gridded.groupby(config.STATION_COL, sort=False)
    hours = grouped.cumcount()
    last_observed_hour = hours.where(observed).groupby(
        gridded[config.STATION_COL], sort=False
    ).ffill()
    return gridded.assign(**{STALENESS_COL: (hours - last_observed_hour)})


def fit_median_fill(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.Series:
    """Learn station-by-hour-of-day medians from training rows only.

    Fitting on the holdout would leak its distribution into the transform, so the
    caller must pass a frame restricted to the fold's training rows.
    """
    target_columns = columns or config.NUMERIC_PREDICTORS
    keys = [frame[config.STATION_COL], frame[config.TIME_COL].dt.hour]
    return frame.groupby(keys, sort=True)[target_columns].median()


def apply_median_fill(
    frame: pd.DataFrame, medians: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """Fill remaining blanks from fitted medians, then from each column's median.

    The second pass covers station-hour combinations absent from the training
    rows; without it a ridge or SARIMAX fit would still see NaN and fail.
    """
    target_columns = columns or config.NUMERIC_PREDICTORS
    keys = pd.MultiIndex.from_arrays(
        [frame[config.STATION_COL], frame[config.TIME_COL].dt.hour]
    )
    lookup = medians.reindex(keys)
    lookup.index = frame.index

    filled = {
        column: frame[column]
        .fillna(lookup[column])
        .fillna(medians[column].median())
        for column in target_columns
    }
    return frame.assign(**filled)
