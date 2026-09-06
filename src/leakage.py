"""The check that makes the no-lookahead rule real.

Discipline is not enforcement. This rebuilds the feature frame from a view of the
data truncated at a cutoff hour and asserts that every feature on the surviving
rows is unchanged. A feature that reads forward loses its input when the future
is removed, its value shifts, and the check fails. See docs/adr/0001.
"""

import numpy as np
import pandas as pd

from src import config
from src.features import feature_columns

KEY_COLS = [config.STATION_COL, config.TIME_COL]


class LookaheadError(AssertionError):
    """Raised when a feature's value depends on data after its observation hour."""


def _aligned(built: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return built.set_index(KEY_COLS)[columns].astype("float64").sort_index()


def compare_at_cutoff(
    raw: pd.DataFrame, build_fn, cutoff: pd.Timestamp, full: pd.DataFrame | None = None
) -> pd.Series:
    """Max absolute difference per feature between truncated and full builds.

    NaN matching NaN counts as no difference; NaN appearing on only one side is
    reported as infinite, because that is a real change in the feature.
    """
    truncated_raw = raw.loc[raw[config.TIME_COL] <= cutoff]
    if truncated_raw.empty:
        raise ValueError(f"cutoff {cutoff} precedes the start of the data")

    truncated = build_fn(truncated_raw)
    full = build_fn(raw) if full is None else full
    columns = feature_columns(truncated)

    left = _aligned(truncated, columns)
    right = _aligned(full, columns).reindex(left.index)

    both_nan = left.isna() & right.isna()
    one_nan = left.isna() ^ right.isna()
    difference = (left - right).abs().where(~both_nan, 0.0)
    return difference.mask(one_nan, np.inf).max().sort_values(ascending=False)


def assert_no_lookahead(
    raw: pd.DataFrame, build_fn, cutoffs: list[pd.Timestamp], tolerance: float = 1e-9
) -> pd.DataFrame:
    """Run the comparison at several cutoffs and raise if any feature moved.

    Returns the per-cutoff, per-feature maximum difference so the notebook can
    show the evidence rather than just a green tick.
    """
    full = build_fn(raw)
    report = pd.DataFrame(
        {
            f"{cutoff:%Y-%m-%d %H:00}": compare_at_cutoff(raw, build_fn, cutoff, full)
            for cutoff in cutoffs
        }
    )
    offenders = report.max(axis=1)
    failed = offenders[offenders > tolerance]
    if not failed.empty:
        listing = ", ".join(f"{name} (max diff {value:g})" for name, value in failed.items())
        raise LookaheadError(f"Features read data from after their observation hour: {listing}")
    return report


def default_cutoffs(raw: pd.DataFrame, count: int = 4) -> list[pd.Timestamp]:
    """Evenly spaced cutoffs across the middle of the observed window."""
    start, end = raw[config.TIME_COL].min(), raw[config.TIME_COL].max()
    fractions = np.linspace(0.4, 0.9, count)
    return [start + (end - start) * fraction for fraction in fractions]


if __name__ == "__main__":
    from src import load
    from src.pipeline import build_frame

    frame = load.load_test()
    result = assert_no_lookahead(frame, build_frame, default_cutoffs(frame))
    print(f"No-lookahead check passed for {len(result)} features "
          f"at {len(result.columns)} cutoffs.")
    print(f"Largest observed difference: {result.to_numpy().max():g}")
