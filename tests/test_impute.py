"""Forward-fill only, capped, and never without a flag."""

import numpy as np
import pandas as pd

from src import config
from src.grid import complete_hourly_grid
from src.impute import (
    IMPUTED_SUFFIX,
    STALENESS_COL,
    add_staleness,
    apply_median_fill,
    fit_median_fill,
    forward_fill_with_flags,
)


def test_no_filled_value_exists_without_a_flag(synthetic_raw):
    gridded = complete_hourly_grid(synthetic_raw)
    filled = forward_fill_with_flags(gridded)

    for column in config.NUMERIC_PREDICTORS:
        newly_filled = gridded[column].isna() & filled[column].notna()
        assert (newly_filled == filled[f"{column}{IMPUTED_SUFFIX}"]).all()


def test_forward_fill_respects_the_staleness_cap(synthetic_raw):
    gridded = complete_hourly_grid(synthetic_raw)

    filled = forward_fill_with_flags(gridded, limit=1)

    still_blank = filled[config.ANCHOR_COL].isna().sum()
    assert still_blank > 0, "a 4-hour hole must not be fully filled at limit=1"


def test_fill_never_reads_backwards(synthetic_raw):
    """The first hour of each station has nothing before it, so it stays blank."""
    gridded = complete_hourly_grid(synthetic_raw)
    gridded.loc[gridded.groupby(config.STATION_COL).head(1).index, "PM10"] = np.nan

    filled = forward_fill_with_flags(gridded)

    first_rows = filled.groupby(config.STATION_COL).head(1)
    assert first_rows["PM10"].isna().all()


def test_staleness_counts_hours_since_a_real_reading(synthetic_raw):
    gridded = complete_hourly_grid(synthetic_raw)

    staleness = add_staleness(gridded)[STALENESS_COL]

    assert staleness.min() == 0
    assert staleness.max() >= 1


def test_median_fill_is_fitted_on_training_rows_only(synthetic_raw):
    gridded = complete_hourly_grid(synthetic_raw)
    train = gridded.iloc[:60]
    holdout = gridded.iloc[60:]

    medians = fit_median_fill(train)
    filled = apply_median_fill(holdout, medians)

    assert filled[config.NUMERIC_PREDICTORS].notna().all().all()
    assert len(medians) <= train[config.STATION_COL].nunique() * 24
