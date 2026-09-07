"""Model factories.

Not yet implemented — see the roadmap in PROJECT.md.

LightGBM is the primary model: fast, native NaN handling, native categoricals.
XGBoost and CatBoost exist for blend diversity later.

Objective note: the metric is RMSE on the raw scale, so train on the raw target
with an L2 objective by default. Training on log1p(y) optimises relative error
and systematically under-predicts spikes — exactly what RMSE punishes. The
target transform is a config switch so the claim can be tested rather than
assumed.

Consider predicting the residual `PM2_5_next_hour - current_PM2_5` and adding
the anchor back at prediction time. Equivalent under RMSE, but better
conditioned, and it makes a model that adds nothing immediately obvious.
"""

from __future__ import annotations

from typing import Any, Protocol


class Regressor(Protocol):
    """Minimal interface the training loop needs from any model."""

    def fit(self, X: Any, y: Any, **kwargs: Any) -> Any: ...
    def predict(self, X: Any) -> Any: ...


def build_model(name: str, params: dict) -> Regressor:
    """Instantiate a model by config name: 'lightgbm', 'xgboost', 'catboost'."""
    raise NotImplementedError


def persistence_predict(X: Any) -> Any:
    """The baseline: return `current_PM2_5` unchanged.

    Needs a fallback for the ~0.7% of rows where `current_PM2_5` is itself
    missing — last valid observation carried forward, or a cross-station
    estimate for the same hour.
    """
    raise NotImplementedError
