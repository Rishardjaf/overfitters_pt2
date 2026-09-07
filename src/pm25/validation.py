"""Time-aware validation folds.

Not yet implemented — see the roadmap in PROJECT.md.

Random K-fold is forbidden on this dataset: adjacent hours are near-duplicates,
so a shuffled split leaks them across folds and reports a score the leaderboard
will not honour. Every boundary below is a timestamp, never a row index.
"""

from __future__ import annotations

import pandas as pd

#: Expanding-window (walk-forward) folds. Each entry is
#: (train_end, valid_start, valid_end), all inclusive of the stated hour.
#: Folds 0 and 2 are heating-season windows and mirror the test period; their
#: mean RMSE is the headline validation number.
EXPANDING_FOLDS: list[tuple[str, str, str]] = [
    ("2014-08-31 23:00", "2014-09-01 00:00", "2015-02-28 23:00"),
    ("2015-02-28 23:00", "2015-03-01 00:00", "2015-08-31 23:00"),
    ("2015-08-31 23:00", "2015-09-01 00:00", "2016-02-29 23:00"),
    ("2016-02-29 23:00", "2016-03-01 00:00", "2016-08-31 23:00"),
]

#: Indices into EXPANDING_FOLDS whose validation window is September-February.
HEATING_SEASON_FOLDS = [0, 2]

#: Held back from all tuning; touched once, before the final submission.
FINAL_HOLDOUT = ("2016-03-01 00:00", "2016-08-31 23:00")


def time_folds(df: pd.DataFrame) -> list[tuple[pd.Index, pd.Index]]:
    """Materialise EXPANDING_FOLDS as (train_idx, valid_idx) pairs."""
    raise NotImplementedError


def assert_no_temporal_overlap(
    df: pd.DataFrame,
    train_idx: pd.Index,
    valid_idx: pd.Index,
) -> None:
    """Raise if any training timestamp is at or after the validation start."""
    raise NotImplementedError
