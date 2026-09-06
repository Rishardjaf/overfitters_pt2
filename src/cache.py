"""Build the engineered feature frames once and cache them as parquet.

Features are built over the *joined* timeline — train and test concatenated and
sorted — then split back apart on the timestamp boundary. Train ends 2016-08-31
22:00 and test begins at 23:00, so the two are contiguous and disjoint; building
them separately would leave the first day of the test window with empty lags for
no reason other than which file the earlier hours happened to live in.

Every feature still reads only times at or before its own observation hour, so
docs/adr/0001 holds. `verify_no_lookahead` proves it on the joined frame rather
than asserting it. See docs/adr/0004.
"""

import time

import pandas as pd

from src import config, leakage, load
from src.pipeline import build_frame


def join_raw(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> pd.DataFrame:
    """One chronological frame covering both windows.

    The test rows have no target column; concatenation gives them NaN there,
    which is what `scored_rows` and every rung already expect of unlabelled rows.
    """
    if train_raw[config.TIME_COL].max() >= test_raw[config.TIME_COL].min():
        raise ValueError(
            "Train and test windows overlap. The joined build assumes a "
            "chronological split — check the supplied CSVs."
        )
    joined = pd.concat([train_raw, test_raw], ignore_index=True)
    return joined.sort_values([config.STATION_COL, config.TIME_COL]).reset_index(drop=True)


def split_on_boundary(
    built: pd.DataFrame, boundary: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cut the built frame back into its train and test halves.

    `boundary` is the first test hour, so it belongs to the test half.
    """
    is_test = built[config.TIME_COL] >= boundary
    return (
        built.loc[~is_test].reset_index(drop=True),
        built.loc[is_test].reset_index(drop=True),
    )


def build_caches(verify: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both CSVs, build features once, write both parquet files.

    Raises `load.DataValidationError` with download instructions if either CSV is
    missing.
    """
    train_raw, test_raw = load.load_train(), load.load_test()
    boundary = test_raw[config.TIME_COL].min()

    joined = join_raw(train_raw, test_raw)
    built = build_frame(joined)
    train_built, test_built = split_on_boundary(built, boundary)

    if verify:
        verify_no_lookahead(joined)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_built.to_parquet(config.TRAIN_PARQUET, index=False)
    test_built.to_parquet(config.TEST_PARQUET, index=False)
    return train_built, test_built


def verify_no_lookahead(joined_raw: pd.DataFrame) -> pd.DataFrame:
    """Rebuild from truncated views of the joined frame and assert nothing moved."""
    return leakage.assert_no_lookahead(
        joined_raw, build_frame, leakage.default_cutoffs(joined_raw)
    )


def _load_or_build(path, index: int) -> pd.DataFrame:
    if not path.exists():
        return build_caches()[index]
    return pd.read_parquet(path)


def load_train_features() -> pd.DataFrame:
    """The engineered training frame, built on first use."""
    return _load_or_build(config.TRAIN_PARQUET, 0)


def load_test_features() -> pd.DataFrame:
    """The engineered test frame, built on first use."""
    return _load_or_build(config.TEST_PARQUET, 1)


if __name__ == "__main__":
    started = time.perf_counter()
    train_built, test_built = build_caches()
    elapsed = time.perf_counter() - started

    for name, frame, path in (
        ("train", train_built, config.TRAIN_PARQUET),
        ("test", test_built, config.TEST_PARQUET),
    ):
        size_mb = path.stat().st_size / 1e6
        print(
            f"{name:5s} {frame.shape[0]:>7,} rows x {frame.shape[1]:>3} cols  "
            f"{frame[config.TIME_COL].min()} to {frame[config.TIME_COL].max()}  "
            f"-> {path.name} ({size_mb:.1f} MB)"
        )
    print(f"\nBuilt and verified in {elapsed:.1f}s. No-lookahead check passed.")
