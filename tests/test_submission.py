"""A submission is checked before it ever leaves the machine."""

import pandas as pd
import pytest

from src import config
from src.submission import SubmissionError, build, validate


@pytest.fixture
def test_frame() -> pd.DataFrame:
    return pd.DataFrame({config.ID_COL: ["AQ_1", "AQ_2", None], config.STATION_COL: "Dongsi"})


def test_build_drops_invented_rows(test_frame):
    submission = build(test_frame, pd.Series([10.0, 20.0, 30.0]))

    assert len(submission) == 2
    assert list(submission[config.ID_COL]) == ["AQ_1", "AQ_2"]


def test_validate_accepts_a_matching_submission(test_frame):
    submission = build(test_frame, pd.Series([10.0, 20.0, 30.0]))

    validate(submission, pd.Series(["AQ_1", "AQ_2"]))


def test_validate_rejects_a_row_count_mismatch(test_frame):
    submission = build(test_frame, pd.Series([10.0, 20.0, 30.0]))

    with pytest.raises(SubmissionError, match="expected 3"):
        validate(submission, pd.Series(["AQ_1", "AQ_2", "AQ_3"]))


def test_validate_rejects_missing_and_negative_predictions(test_frame):
    with pytest.raises(SubmissionError, match="missing"):
        validate(build(test_frame, pd.Series([10.0, None, 30.0])), pd.Series(["AQ_1", "AQ_2"]))

    with pytest.raises(SubmissionError, match="Negative"):
        validate(build(test_frame, pd.Series([-1.0, 20.0, 30.0])), pd.Series(["AQ_1", "AQ_2"]))
