"""Score one rung on the season-matched holdout, then refit and predict the test set.

Two things make a comparison honest here. Persistence is re-scored on the *same*
holdout rows as the rung under test, so the improvement figure is like-for-like
rather than against a number computed somewhere else. And the model that produces
the submission is refit on every labelled training row, which is the final refit
docs/adr/0002 calls for — the fold exists to estimate the score, not to produce
the artefact.
"""

import time
from dataclasses import dataclass, field

import pandas as pd

from src import config, evaluate, load, submission, validation
from src.experiments.registry import Rung
from src.models import pm10_anchor

PRIMARY_FOLD_ONLY = 1


@dataclass(frozen=True)
class FoldScore:
    """One rung's result on one season-matched holdout block."""

    fold: str
    train_rows: int
    holdout_rows: int
    rmse: float
    mae: float
    baseline_rmse: float
    baseline_mae: float
    rmse_vs_baseline_pct: float
    beats_baseline: bool
    fit_seconds: float


@dataclass(frozen=True)
class Diagnostics:
    """The primary fold's holdout, kept so the figures can be drawn from it."""

    timestamps: pd.Series
    actual: pd.Series
    predicted: pd.Series


@dataclass(frozen=True)
class RunOutcome:
    """Everything one numbered test produced."""

    rung: Rung
    fold_scores: list[FoldScore]
    diagnostics: Diagnostics
    importance: pd.DataFrame | None = None
    importance_column: str | None = None
    submission_rows: int = 0
    gap_filled_predictions: int = 0
    total_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def primary(self) -> FoldScore:
        return self.fold_scores[0]


def _labelled(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.dropna(subset=[config.TARGET_COL])


def _fill_gaps(
    predictions: pd.Series, train: pd.DataFrame, frame: pd.DataFrame
) -> tuple[pd.Series, int]:
    """Fall back to the PM10 anchor wherever a rung declined to predict.

    A submission with a hole in it is rejected by the grader, so the gap is filled
    with the baseline and the count is reported rather than hidden. The anchor is
    refitted on the training rows passed in, never on the frame being filled.
    """
    missing = predictions.isna()
    if not missing.any():
        return predictions, 0
    filled = pm10_anchor.predict(pm10_anchor.fit(train), frame)
    return predictions.fillna(filled), int(missing.sum())


def _score_fold(
    rung: Rung, train_rows: pd.DataFrame, holdout_rows: pd.DataFrame, fold_name: str
) -> tuple[FoldScore, Diagnostics]:
    started = time.perf_counter()
    result = rung.adapter(train_rows, holdout_rows)
    predictions, _ = _fill_gaps(result.predictions, train_rows, holdout_rows)
    fit_seconds = time.perf_counter() - started

    scored = _labelled(holdout_rows)
    actual = scored[config.TARGET_COL]
    predicted = predictions.reindex(scored.index)
    # The baseline sees exactly the frame the rung saw, then is narrowed the same
    # way. Scoring it on the already-narrowed frame would shift the anchor's own
    # lag fallback and stop the baseline test scoring 0.00% against itself. The
    # line is fitted on the fold's training rows only.
    baseline_predicted = pm10_anchor.predict(
        pm10_anchor.fit(train_rows), holdout_rows
    ).reindex(scored.index)

    rmse = evaluate.rmse(actual, predicted)
    baseline_rmse = evaluate.rmse(actual, baseline_predicted)

    score = FoldScore(
        fold=fold_name,
        train_rows=len(_labelled(train_rows)),
        holdout_rows=len(scored),
        rmse=rmse,
        mae=evaluate.mae(actual, predicted),
        baseline_rmse=baseline_rmse,
        baseline_mae=evaluate.mae(actual, baseline_predicted),
        rmse_vs_baseline_pct=round((baseline_rmse - rmse) / baseline_rmse * 100, 3),
        beats_baseline=bool(rmse < baseline_rmse),
        fit_seconds=round(fit_seconds, 1),
    )
    return score, Diagnostics(scored[config.TIME_COL], actual, predicted)


def _build_submission(
    rung: Rung, train_features: pd.DataFrame, test_features: pd.DataFrame, directory
) -> tuple[int, int, pd.DataFrame | None, str | None]:
    """Refit on every labelled training row, predict the test set, write the CSV."""
    result = rung.adapter(train_features, test_features)
    predictions, gap_filled = _fill_gaps(result.predictions, train_features, test_features)

    entry = submission.build(test_features, predictions)
    submission.validate(entry, load.load_test()[config.ID_COL])
    submission.write(entry, "submission", directory=directory)
    return len(entry), gap_filled, result.importance, result.importance_column


def run(
    rung: Rung,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    directory,
    all_folds: bool = False,
) -> RunOutcome:
    """Score the rung on its folds, then refit and write its submission."""
    started = time.perf_counter()
    folds = validation.season_matched_folds(train_features)
    validation.assert_folds_are_chronological(train_features, folds)
    validation.assert_folds_are_season_matched(train_features, folds)
    if not all_folds:
        folds = folds[:PRIMARY_FOLD_ONLY]

    scores, diagnostics = [], None
    for fold in folds:
        score, fold_diagnostics = _score_fold(
            rung,
            train_features.loc[fold.train_mask],
            train_features.loc[fold.holdout_mask],
            fold.name,
        )
        scores.append(score)
        diagnostics = diagnostics or fold_diagnostics

    rows, gap_filled, importance, importance_column = _build_submission(
        rung, train_features, test_features, directory
    )

    warnings = []
    if gap_filled:
        warnings.append(
            f"{gap_filled:,} test predictions were empty and fell back to the PM10 anchor."
        )
    if not scores[0].beats_baseline and not rung.is_baseline:
        warnings.append(
            "This rung does not beat the PM10 anchor on the primary fold. Recorded as a "
            "finding, not dropped."
        )

    return RunOutcome(
        rung=rung,
        fold_scores=scores,
        diagnostics=diagnostics,
        importance=importance,
        importance_column=importance_column,
        submission_rows=rows,
        gap_filled_predictions=gap_filled,
        total_seconds=round(time.perf_counter() - started, 1),
        warnings=warnings,
    )
