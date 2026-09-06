"""Read the competition CSVs into typed frames.

Boundary validation lives here: anything the rest of the pipeline assumes about
shape, columns and dtypes is checked once, at the point the data enters.
"""

from pathlib import Path

import pandas as pd

from src import config


class DataValidationError(ValueError):
    """Raised when a supplied CSV does not match the documented schema."""


def _require_columns(frame: pd.DataFrame, required: list[str], source: Path) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataValidationError(
            f"{source.name} is missing required columns: {', '.join(missing)}"
        )


def load_raw(path: Path, *, expect_target: bool) -> pd.DataFrame:
    """Load one competition CSV, parse its timestamp and coerce numeric columns.

    Blank fields stay as NaN — they are meaningful and are never filled here.
    """
    if not path.exists():
        raise DataValidationError(
            f"{path} not found. Download it from the competition Data tab into {config.DATA_DIR}."
        )

    frame = pd.read_csv(path)
    required = [config.ID_COL, config.TIME_COL, config.STATION_COL]
    required += config.NUMERIC_PREDICTORS + [config.WIND_DIR_COL]
    if expect_target:
        required += [config.TARGET_COL]
    _require_columns(frame, required, path)

    numeric = config.NUMERIC_PREDICTORS + ([config.TARGET_COL] if expect_target else [])
    typed = frame.assign(
        **{config.TIME_COL: pd.to_datetime(frame[config.TIME_COL])},
        **{column: pd.to_numeric(frame[column], errors="coerce") for column in numeric},
    )

    unknown_stations = set(typed[config.STATION_COL]) - set(config.STATIONS)
    if unknown_stations:
        raise DataValidationError(f"Unexpected stations in {path.name}: {sorted(unknown_stations)}")

    return typed.sort_values([config.STATION_COL, config.TIME_COL]).reset_index(drop=True)


def load_test() -> pd.DataFrame:
    """The hidden-test predictors. Has no target column."""
    return load_raw(config.TEST_PATH, expect_target=False)


def load_train() -> pd.DataFrame:
    """The labelled training station-hours."""
    return load_raw(config.TRAIN_PATH, expect_target=True)


def describe_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-station row counts and observation windows — the first EDA sanity check."""
    grouped = frame.groupby(config.STATION_COL)[config.TIME_COL]
    span_hours = (grouped.max() - grouped.min()).dt.total_seconds() / 3600 + 1
    return pd.DataFrame(
        {
            "rows": grouped.size(),
            "first_hour": grouped.min(),
            "last_hour": grouped.max(),
            "grid_slots": span_hours.astype(int),
        }
    ).assign(coverage=lambda d: (d["rows"] / d["grid_slots"]).round(4))
