"""Paired day-block bootstrap for RMSE differences.

Station-hours are not independent: a pollution episode spans many consecutive
hours across all twelve stations at once, so the effective sample size behind a
holdout RMSE is nearer the number of days than the number of rows. Resampling
individual rows would understate the interval badly; resampling whole calendar
days keeps each episode's hours together and is honest about how little
independent evidence a fold really contains.

The comparison is paired — both prediction vectors are scored on identical rows
— so the day-to-day variation that dominates each model's own RMSE cancels in
the difference, and the interval speaks only to which model is better.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config

DEFAULT_DRAWS = 2_000
DEFAULT_INTERVAL = 0.90


@dataclass(frozen=True)
class BootstrapComparison:
    """The paired interval for RMSE(variant) − RMSE(reference)."""

    rmse_reference: float
    rmse_variant: float
    delta_rmse: float
    lower: float
    upper: float
    interval: float
    draws: int
    blocks: int

    @property
    def is_conclusive(self) -> bool:
        """True when the interval excludes zero — the difference is real."""
        return self.upper < 0 or self.lower > 0

    def describe(self) -> str:
        percent = round(self.interval * 100)
        verdict = "conclusive" if self.is_conclusive else "not conclusive"
        return (
            f"ΔRMSE {self.delta_rmse:+.4f} "
            f"[{self.lower:+.4f}, {self.upper:+.4f}] {percent}% — {verdict} "
            f"({self.blocks} day blocks, {self.draws} draws)"
        )


def _validated(name: str, values, length: int) -> np.ndarray:
    array = np.asarray(values, dtype="float64")
    if len(array) != length:
        raise ValueError(
            f"{name} has length {len(array)}, expected {length} — every input "
            "must cover the same holdout rows."
        )
    if np.isnan(array).any():
        raise ValueError(
            f"{name} contains blank values; score only rows where both models "
            "predicted and the actual is known."
        )
    return array


def compare_rmse(
    actual: pd.Series,
    reference: pd.Series,
    variant: pd.Series,
    timestamps: pd.Series,
    draws: int = DEFAULT_DRAWS,
    interval: float = DEFAULT_INTERVAL,
    seed: int = config.RANDOM_SEED,
) -> BootstrapComparison:
    """Bootstrap the RMSE difference by resampling whole days with replacement.

    Per-day sums of squared errors are precomputed once, so each draw is a sum
    over day blocks rather than a pass over the rows — 2,000 draws over a
    six-month fold take well under a second.
    """
    length = len(actual)
    truth = _validated("actual", actual, length)
    first = _validated("reference predictions", reference, length)
    second = _validated("variant predictions", variant, length)

    days = pd.Series(pd.DatetimeIndex(timestamps).floor("D"))
    if len(days) != length:
        raise ValueError(f"timestamps has length {len(days)}, expected {length}.")

    squared_reference = pd.Series((truth - first) ** 2)
    squared_variant = pd.Series((truth - second) ** 2)
    by_day = pd.DataFrame(
        {
            "reference": squared_reference.groupby(days.values).sum(),
            "variant": squared_variant.groupby(days.values).sum(),
            "rows": squared_reference.groupby(days.values).size(),
        }
    )

    rmse_reference = float(np.sqrt(squared_reference.mean()))
    rmse_variant = float(np.sqrt(squared_variant.mean()))

    rng = np.random.default_rng(seed)
    block_count = len(by_day)
    samples = rng.integers(0, block_count, size=(draws, block_count))

    reference_sums = by_day["reference"].to_numpy()[samples].sum(axis=1)
    variant_sums = by_day["variant"].to_numpy()[samples].sum(axis=1)
    row_counts = by_day["rows"].to_numpy()[samples].sum(axis=1)
    deltas = np.sqrt(variant_sums / row_counts) - np.sqrt(reference_sums / row_counts)

    tail = (1 - interval) / 2
    return BootstrapComparison(
        rmse_reference=rmse_reference,
        rmse_variant=rmse_variant,
        delta_rmse=rmse_variant - rmse_reference,
        lower=float(np.quantile(deltas, tail)),
        upper=float(np.quantile(deltas, 1 - tail)),
        interval=interval,
        draws=draws,
        blocks=block_count,
    )
