"""Loading the competition CSVs and building the complete hourly grid.

Not yet implemented — see the roadmap in PROJECT.md.

Design notes for whoever implements this:

* `Data/*.csv` are read-only. Cache derived frames to `artifacts/` as parquet.
* Roughly 2,466 places across the 12 stations have consecutive retained rows
  more than one hour apart. Lag features must therefore be built against a
  reindexed *complete* hourly grid per station, so that "lag 1" is always truly
  `t-1` and never "the previous surviving row", which may be hours earlier.
* `load_panel()` concatenates train and test on one timeline so that test rows
  can carry their own backward lags. This is allowed and necessary; it is only
  leakage if a window looks forward. See CLAUDE.md.
"""

from __future__ import annotations

import pandas as pd


def load_train() -> pd.DataFrame:
    """Read `Data/train.csv` with parsed timestamps and correct dtypes."""
    raise NotImplementedError


def load_test() -> pd.DataFrame:
    """Read `Data/test.csv` with parsed timestamps and correct dtypes."""
    raise NotImplementedError


def load_panel() -> pd.DataFrame:
    """Train and test concatenated on one timeline, with an `is_test` flag.

    Sorted by (station, timestamp) so that grouped backward shifts are valid.
    The target is NaN for test rows.
    """
    raise NotImplementedError


def complete_hourly_grid(panel: pd.DataFrame) -> pd.DataFrame:
    """Reindex each station onto a gap-free hourly index.

    Inserted rows carry NaN predictors and are excluded from training; they exist
    so that shift-based lags measure real elapsed hours.
    """
    raise NotImplementedError
