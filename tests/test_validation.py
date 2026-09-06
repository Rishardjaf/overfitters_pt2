"""Folds must be chronological and season-matched."""

import pandas as pd
import pytest

from src import config
from src.validation import (
    assert_folds_are_chronological,
    assert_folds_are_season_matched,
    describe_folds,
    season_matched_folds,
)


@pytest.fixture
def multi_year_frame() -> pd.DataFrame:
    hours = pd.date_range("2013-03-01", "2016-08-31", freq="h")
    return pd.DataFrame({config.TIME_COL: hours, config.STATION_COL: config.STATIONS[0]})


def test_folds_are_chronological_and_season_matched(multi_year_frame):
    folds = season_matched_folds(multi_year_frame)

    assert len(folds) == 2
    assert_folds_are_chronological(multi_year_frame, folds)
    assert_folds_are_season_matched(multi_year_frame, folds)


def test_holdout_months_are_september_to_february(multi_year_frame):
    folds = season_matched_folds(multi_year_frame)

    months = set(multi_year_frame.loc[folds[0].holdout_mask, config.TIME_COL].dt.month)
    assert months == {9, 10, 11, 12, 1, 2}


def test_a_tail_split_would_be_rejected(multi_year_frame):
    """Guards the rule itself: a spring/summer holdout must not pass the check."""
    folds = season_matched_folds(multi_year_frame, blocks=[("2016-03-01", "2016-08-01")])

    with pytest.raises(AssertionError, match="not season-matched"):
        assert_folds_are_season_matched(multi_year_frame, folds)


def test_blocks_outside_the_window_are_skipped(multi_year_frame):
    with pytest.raises(ValueError, match="No usable validation folds"):
        season_matched_folds(multi_year_frame, blocks=[("2020-09-01", "2021-03-01")])


def test_describe_folds_reports_boundaries(multi_year_frame):
    table = describe_folds(multi_year_frame, season_matched_folds(multi_year_frame))

    assert list(table.columns) == [
        "fold", "train_rows", "holdout_rows", "train_end", "holdout_start", "holdout_end",
    ]
    assert (table["train_end"] < table["holdout_start"]).all()
