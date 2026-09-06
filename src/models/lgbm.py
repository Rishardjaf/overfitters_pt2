"""Rung 3 — the submission candidate.

LightGBM handles missing values natively and can learn from the pattern of
absence, so blank predictors are passed straight through rather than imputed.
See docs/adr/0003.
"""

import lightgbm as lgb
import pandas as pd

from src import config
from src.features import feature_columns

DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "seed": 33854602,
}
DEFAULT_ROUNDS = 800
EARLY_STOPPING_ROUNDS = 50


def fit(train: pd.DataFrame, valid: pd.DataFrame | None = None, params: dict | None = None):
    """Train on the fold's rows, optionally early-stopping on a validation block."""
    columns = feature_columns(train)
    labelled = train.dropna(subset=[config.TARGET_COL])
    train_set = lgb.Dataset(labelled[columns], labelled[config.TARGET_COL])

    valid_sets, callbacks = None, []
    if valid is not None:
        valid_labelled = valid.dropna(subset=[config.TARGET_COL])
        valid_sets = [lgb.Dataset(valid_labelled[columns], valid_labelled[config.TARGET_COL])]
        callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]

    return lgb.train(
        {**DEFAULT_PARAMS, **(params or {})},
        train_set,
        num_boost_round=DEFAULT_ROUNDS,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )


def predict(booster, frame: pd.DataFrame) -> pd.Series:
    """Predictions clipped at zero — a negative concentration is not physical."""
    columns = booster.feature_name()
    values = booster.predict(frame[columns], num_iteration=booster.best_iteration)
    return pd.Series(values, index=frame.index, name="lightgbm").clip(lower=0)


def importance_table(booster, top: int = 30) -> pd.DataFrame:
    """Gain-based feature importance, for the EDA writeup."""
    return (
        pd.DataFrame(
            {
                "feature": booster.feature_name(),
                "gain": booster.feature_importance("gain"),
                "splits": booster.feature_importance("split"),
            }
        )
        .sort_values("gain", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )
