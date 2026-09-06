"""Rung 4 — boosting on the deviation from the PM10 anchor, not on the level.

A gradient-boosted tree predicts a weighted mean of leaf values, so it cannot
return anything outside the range of leaf means it fitted. The target's median is
55 µg/m³ and its maximum is 999, and squared error is decided almost entirely by
the episodes in that tail — precisely the region a tree is structurally worst at.

Fitting instead on Δ = target − anchor(t) moves the level into the linear part,
which is unbounded, and leaves the trees to model only the correction. The old
release established this on a persistence anchor (ADR-0006, 19.998 against
21.635 for the same booster on the level). That anchor is gone; the argument is
not, and `pm10_anchor` supplies its replacement.

The anchor is refitted inside `fit` on the training rows it is given, so the
offset a prediction is added back onto never carries information from the block
being scored.
"""

from dataclasses import dataclass

import lightgbm as lgb
import pandas as pd

from src import config
from src.features import feature_columns
from src.models import lgbm, pm10_anchor

MIN_CONCENTRATION = 0.0


@dataclass(frozen=True)
class DeltaFit:
    """The booster and the anchor whose output it corrects."""

    booster: lgb.Booster
    anchor: pm10_anchor.AnchorFit


def fit(train: pd.DataFrame, params: dict | None = None) -> DeltaFit:
    """Fit the anchor on the training rows, then boost on what it gets wrong."""
    anchor = pm10_anchor.fit(train)
    labelled = train.dropna(subset=[config.TARGET_COL])
    columns = feature_columns(labelled)
    delta = labelled[config.TARGET_COL] - pm10_anchor.predict(anchor, labelled)

    booster = lgb.train(
        {**lgbm.DEFAULT_PARAMS, **(params or {})},
        lgb.Dataset(labelled[columns], delta),
        num_boost_round=lgbm.DEFAULT_ROUNDS,
    )
    return DeltaFit(booster=booster, anchor=anchor)


def predict(fitted: DeltaFit, frame: pd.DataFrame) -> pd.Series:
    """Anchor plus predicted correction, clipped at zero."""
    columns = fitted.booster.feature_name()
    correction = fitted.booster.predict(frame[columns])
    level = pm10_anchor.predict(fitted.anchor, frame)
    return (level + correction).clip(lower=MIN_CONCENTRATION).rename("delta_lightgbm")


def importance_table(fitted: DeltaFit, top: int = 30) -> pd.DataFrame:
    """Gain-based importance over the correction, not over the level."""
    return lgbm.importance_table(fitted.booster, top=top)
