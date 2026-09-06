"""The rungs, behind one calling convention.

The model modules deliberately expose different shapes — the anchor baseline
fits three numbers, ridge carries a fitted median fill alongside the estimator,
LightGBM hands back a booster. Rather than flatten working code into a lowest
common denominator, each rung gets a small adapter here.

Ordering is by complexity: the anchor baseline, the linear model, then boosting.
The old ladder's random forest, histogram boosting and SARIMAX rungs are not
carried over: on the previous release the forest lost to ridge and SARIMAX lost
to the baseline itself, and neither is worth its fitting time on a short clock.

Model modules are imported inside their adapters rather than at the top. LightGBM
loads a native library that needs `libomp` present on macOS, and importing it
here would mean one missing dylib stops the other rungs from running at all.
"""

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class RungResult:
    """Predictions for one holdout, plus whatever the model can say about why."""

    predictions: pd.Series
    importance: pd.DataFrame | None = None
    importance_column: str | None = None


@dataclass(frozen=True)
class Rung:
    """One numbered test: what it runs, where it writes, what it is called."""

    slug: str
    algorithm: str
    label: str
    summary: str
    adapter: Callable[[pd.DataFrame, pd.DataFrame], RungResult]
    is_baseline: bool = False

    @property
    def number(self) -> int:
        return int(self.slug.split("_")[0])


def _run_pm10_anchor(train: pd.DataFrame, holdout: pd.DataFrame) -> RungResult:
    """Three fitted numbers — but they are fitted, so the training rows matter."""
    from src.models import pm10_anchor

    return RungResult(pm10_anchor.predict(pm10_anchor.fit(train), holdout))


def _run_ridge(train: pd.DataFrame, holdout: pd.DataFrame) -> RungResult:
    from src.models import ridge

    fitted = ridge.fit(train)
    return RungResult(
        ridge.predict(fitted, holdout), ridge.importance_table(fitted), "magnitude"
    )


def _run_lgbm(train: pd.DataFrame, holdout: pd.DataFrame) -> RungResult:
    """No early stopping: the only validation block available is the holdout.

    Stopping on it would tune the model against the very rows used to score it.
    A fixed round budget is the honest alternative.
    """
    from src.models import lgbm

    booster = lgbm.fit(train)
    return RungResult(
        lgbm.predict(booster, holdout), lgbm.importance_table(booster), "gain"
    )


def _run_delta_lgbm(train: pd.DataFrame, holdout: pd.DataFrame) -> RungResult:
    """Boosting on the deviation from the anchor rather than on the level.

    A tree cannot predict past the largest leaf mean it fitted, and the target's
    tail reaches 999 µg/m³ against a median of 55, so the levels that decide the
    squared error are the ones the trees are worst at. Handing them an exact
    offset and asking only for the residual moves the extrapolation into the
    linear part, where it is unbounded.
    """
    from src.models import delta_lgbm

    fitted = delta_lgbm.fit(train)
    return RungResult(
        delta_lgbm.predict(fitted, holdout), delta_lgbm.importance_table(fitted), "gain"
    )


RUNGS: tuple[Rung, ...] = (
    Rung(
        slug="01_baseline_pm10_anchor",
        algorithm="pm10_anchor",
        label="BASELINE TEST",
        summary="Next-hour PM2.5 as a fitted line through PM10 at the observation hour. The bar every rung is measured against.",
        adapter=_run_pm10_anchor,
        is_baseline=True,
    ),
    Rung(
        slug="02_ridge",
        algorithm="ridge",
        label="TEST 02",
        summary="Regularised linear regression on the median-filled frame. How much of the signal is linear?",
        adapter=_run_ridge,
    ),
    Rung(
        slug="03_lightgbm",
        algorithm="lightgbm",
        label="TEST 03",
        summary="LightGBM, blanks left intact so absence itself is learnable. The strongest rung that predicts the concentration itself.",
        adapter=_run_lgbm,
    ),
    Rung(
        slug="04_delta_lightgbm",
        algorithm="delta_lightgbm",
        label="TEST 04",
        summary="LightGBM on the deviation from the PM10 anchor. Removes the tree extrapolation ceiling that decides the tail.",
        adapter=_run_delta_lgbm,
    ),
)

BY_SLUG = {rung.slug: rung for rung in RUNGS}
BY_ALGORITHM = {rung.algorithm: rung for rung in RUNGS}


def resolve(name: str) -> Rung:
    """Look a rung up by slug or by algorithm name, with a listing on failure."""
    rung = BY_SLUG.get(name) or BY_ALGORITHM.get(name)
    if rung is None:
        known = ", ".join(sorted(BY_SLUG))
        raise KeyError(f"Unknown rung {name!r}. Known rungs: {known}")
    return rung


def baseline() -> Rung:
    """The rung every other one is scored against."""
    return next(rung for rung in RUNGS if rung.is_baseline)
