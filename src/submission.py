"""Write and validate a submission file.

Two things have sunk more datathon entries than any model choice: a row count
that does not match, and ids that silently reorder. Both are checked here before
anything leaves the machine.
"""

from pathlib import Path

import pandas as pd

from src import config

EXPECTED_TEST_ROWS = 51_063

# The prediction column header is taken from the brief's target name. Confirm it
# against the competition's sample-submission file before the first upload.
PREDICTION_COL = config.TARGET_COL


class SubmissionError(ValueError):
    """Raised when a submission would not be accepted by the grader."""


def build(test_frame: pd.DataFrame, predictions: pd.Series) -> pd.DataFrame:
    """Assemble id and prediction columns, dropping the rows the grid invented."""
    submission = pd.DataFrame(
        {
            config.ID_COL: test_frame[config.ID_COL],
            PREDICTION_COL: predictions.to_numpy(),
        }
    )
    return submission.loc[submission[config.ID_COL].notna()].reset_index(drop=True)


def validate(submission: pd.DataFrame, reference_ids: pd.Series) -> None:
    """Check the row count, the id set and the absence of missing predictions."""
    if len(submission) != len(reference_ids):
        raise SubmissionError(
            f"{len(submission)} rows, expected {len(reference_ids)}."
        )
    if set(submission[config.ID_COL]) != set(reference_ids):
        raise SubmissionError("Submission ids do not match the test set ids.")
    if submission[PREDICTION_COL].isna().any():
        missing = int(submission[PREDICTION_COL].isna().sum())
        raise SubmissionError(f"{missing} predictions are missing.")
    if (submission[PREDICTION_COL] < 0).any():
        raise SubmissionError("Negative concentrations are not physical.")


def write(submission: pd.DataFrame, name: str, directory: Path | None = None) -> Path:
    """Validate nothing here — call `validate` first — then write the CSV.

    Defaults to outputs/submissions/; the experiment runner passes each test's
    own folder so a result and the file it produced stay together.
    """
    target = directory or config.SUBMISSION_DIR
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{name}.csv"
    submission.to_csv(path, index=False)
    return path
