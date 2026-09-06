"""The parquet cache must round-trip exactly, and the joined build must help."""

import pandas as pd
import pytest

from src import cache, config
from src.pipeline import build_frame


@pytest.fixture
def split_raw(synthetic_labelled):
    """Cut the labelled fixture into a train half and a later test half."""
    boundary = pd.Timestamp("2015-10-01")
    train = synthetic_labelled.loc[synthetic_labelled[config.TIME_COL] < boundary]
    test = synthetic_labelled.loc[synthetic_labelled[config.TIME_COL] >= boundary]
    return train.reset_index(drop=True), test.drop(columns=[config.TARGET_COL]).reset_index(
        drop=True
    ), boundary


def test_parquet_round_trip_preserves_the_frame(synthetic_labelled, tmp_path):
    built = build_frame(synthetic_labelled)
    path = tmp_path / "round_trip.parquet"

    built.to_parquet(path, index=False)

    pd.testing.assert_frame_equal(built, pd.read_parquet(path))


def test_joined_build_fills_lags_at_the_boundary(split_raw):
    """The first hours of the test window should gain lags from the train half."""
    train_raw, test_raw, boundary = split_raw

    separate = build_frame(test_raw)
    joined = build_frame(cache.join_raw(train_raw, test_raw))
    _, joined_test = cache.split_on_boundary(joined, boundary)

    lag = "PM10_lag24"
    assert joined_test[lag].notna().sum() > separate[lag].notna().sum()


def test_joined_build_agrees_wherever_both_are_defined(split_raw):
    """Extra history may fill a blank lag; it must never change a defined one."""
    train_raw, test_raw, boundary = split_raw
    keys = [config.STATION_COL, config.TIME_COL]

    separate = build_frame(test_raw).set_index(keys).sort_index()
    joined, joined_test = build_frame(cache.join_raw(train_raw, test_raw)), None
    joined_test = cache.split_on_boundary(joined, boundary)[1].set_index(keys).sort_index()

    lag = "PM10_lag24"
    both = separate[lag].notna() & joined_test[lag].notna()
    pd.testing.assert_series_equal(separate.loc[both, lag], joined_test.loc[both, lag])


def test_split_puts_the_boundary_hour_in_the_test_half(split_raw):
    train_raw, test_raw, boundary = split_raw

    built = build_frame(cache.join_raw(train_raw, test_raw))
    train_built, test_built = cache.split_on_boundary(built, boundary)

    assert train_built[config.TIME_COL].max() < boundary
    assert test_built[config.TIME_COL].min() == boundary
    assert len(train_built) + len(test_built) == len(built)


def test_overlapping_windows_are_rejected(synthetic_labelled):
    """A chronological split is an assumption, so it is checked rather than trusted."""
    with pytest.raises(ValueError, match="overlap"):
        cache.join_raw(synthetic_labelled, synthetic_labelled)
