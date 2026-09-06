"""The no-lookahead check must pass on honest features and fail on a leak."""

import pandas as pd
import pytest

from src import config
from src.leakage import LookaheadError, assert_no_lookahead, default_cutoffs
from src.pipeline import build_frame


def test_honest_pipeline_passes(synthetic_raw):
    cutoffs = default_cutoffs(synthetic_raw, count=2)

    report = assert_no_lookahead(synthetic_raw, build_frame, cutoffs)

    assert report.to_numpy().max() == 0


def test_a_deliberate_leak_is_caught(synthetic_raw):
    """A negative shift is the exact mistake the rule exists to prevent."""

    def leaking_build(frame: pd.DataFrame) -> pd.DataFrame:
        built = build_frame(frame)
        return built.assign(
            next_hour_peek=built.groupby(config.STATION_COL)[config.ANCHOR_COL].shift(-1)
        )

    with pytest.raises(LookaheadError, match="next_hour_peek"):
        assert_no_lookahead(synthetic_raw, leaking_build, default_cutoffs(synthetic_raw, 2))


def test_centred_window_is_caught(synthetic_raw):
    """A centred rolling mean reads half its window from the future."""

    def centred_build(frame: pd.DataFrame) -> pd.DataFrame:
        built = build_frame(frame)
        return built.assign(
            centred_mean=built.groupby(config.STATION_COL)[config.ANCHOR_COL].transform(
                lambda s: s.rolling(5, center=True, min_periods=1).mean()
            )
        )

    with pytest.raises(LookaheadError, match="centred_mean"):
        assert_no_lookahead(synthetic_raw, centred_build, default_cutoffs(synthetic_raw, 2))
