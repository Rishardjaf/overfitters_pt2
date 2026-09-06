"""Season-matched blocked validation.

The hidden test window is September to February. A chronological tail split would
hand us a spring/summer holdout — clean air, low variance, and a score that says
nothing about the test period. So the holdout blocks are the same months from
earlier years instead. See docs/adr/0002.
"""

from dataclasses import dataclass

import pandas as pd

from src import config


@dataclass(frozen=True)
class Fold:
    """One season-matched block and everything that precedes it."""

    name: str
    train_mask: pd.Series
    holdout_mask: pd.Series

    @property
    def sizes(self) -> tuple[int, int]:
        return int(self.train_mask.sum()), int(self.holdout_mask.sum())


def season_matched_folds(
    frame: pd.DataFrame,
    blocks: list[tuple[str, str]] | None = None,
    min_train_rows: int = 10_000,
) -> list[Fold]:
    """Build one fold per block, training only on data preceding that block.

    Blocks that fall outside the frame's window, or that leave too little
    training data, are skipped rather than silently producing a degenerate fold.
    """
    timestamps = frame[config.TIME_COL]
    folds = []
    for start_text, end_text in blocks or config.VALIDATION_BLOCKS:
        start, end = pd.Timestamp(start_text), pd.Timestamp(end_text)
        holdout_mask = (timestamps >= start) & (timestamps < end)
        train_mask = timestamps < start
        if holdout_mask.sum() == 0 or train_mask.sum() < min_train_rows:
            continue
        folds.append(
            Fold(f"{start:%Y-%m}_to_{end:%Y-%m}", train_mask, holdout_mask)
        )
    if not folds:
        raise ValueError(
            "No usable validation folds. Check that the training data actually "
            f"covers the blocks in config.VALIDATION_BLOCKS: {config.VALIDATION_BLOCKS}"
        )
    return folds


def describe_folds(frame: pd.DataFrame, folds: list[Fold]) -> pd.DataFrame:
    """A table of fold boundaries and sizes, for the notebook to display."""
    timestamps = frame[config.TIME_COL]
    return pd.DataFrame(
        [
            {
                "fold": fold.name,
                "train_rows": fold.sizes[0],
                "holdout_rows": fold.sizes[1],
                "train_end": timestamps[fold.train_mask].max(),
                "holdout_start": timestamps[fold.holdout_mask].min(),
                "holdout_end": timestamps[fold.holdout_mask].max(),
            }
            for fold in folds
        ]
    )


def assert_folds_are_chronological(frame: pd.DataFrame, folds: list[Fold]) -> None:
    """Every training hour must strictly precede its holdout block."""
    timestamps = frame[config.TIME_COL]
    for fold in folds:
        latest_train = timestamps[fold.train_mask].max()
        earliest_holdout = timestamps[fold.holdout_mask].min()
        if latest_train >= earliest_holdout:
            raise AssertionError(
                f"Fold {fold.name} trains on {latest_train}, at or after its "
                f"holdout starts at {earliest_holdout}."
            )


def assert_folds_are_season_matched(frame: pd.DataFrame, folds: list[Fold]) -> None:
    """Holdout blocks must sit in the September-to-February window."""
    expected = {9, 10, 11, 12, 1, 2}
    timestamps = frame[config.TIME_COL]
    for fold in folds:
        months = set(timestamps[fold.holdout_mask].dt.month.unique())
        if not months <= expected:
            raise AssertionError(
                f"Fold {fold.name} holdout covers months {sorted(months)}, which "
                "is not season-matched to the Sep-Feb test window."
            )
