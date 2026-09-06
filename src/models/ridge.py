"""Rung 2 — how much of the signal is linear?

Ridge cannot see a NaN, so it receives the median-filled frame described in
docs/adr/0003, prepared by `dense_frame`. Its value is diagnostic: the gap
between this rung and LightGBM is the part of the problem that needs
non-linearity.
"""

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models import dense_frame

DEFAULT_ALPHA = 1.0


def fit(train: pd.DataFrame, alpha: float = DEFAULT_ALPHA):
    """Fit the scaler-and-ridge pipeline on the fold's training rows.

    Returns the fitted pipeline alongside the `DenseFit` that produced its
    matrix, because predicting anything else requires the same fitted fill.
    """
    prepared = dense_frame.prepare(train)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(prepared.features, prepared.target)
    return model, prepared


def predict(fitted, frame: pd.DataFrame) -> pd.Series:
    """Predictions clipped at zero — a negative concentration is not physical."""
    model, prepared = fitted
    features = dense_frame.transform(prepared, frame)
    return pd.Series(
        model.predict(features), index=frame.index, name="ridge"
    ).clip(lower=0)


def fit_predict(
    train: pd.DataFrame, holdout: pd.DataFrame, alpha: float = DEFAULT_ALPHA
) -> pd.Series:
    """Fit on the fold's training rows, predict its holdout."""
    return predict(fit(train, alpha), holdout)


def importance_table(fitted, top: int = 30) -> pd.DataFrame:
    """Coefficient magnitudes on the standardised scale.

    The scaler puts every predictor in the same units, so the coefficients are
    directly comparable as a measure of influence — which raw ones would not be.
    """
    model, prepared = fitted
    coefficients = model.named_steps["ridge"].coef_
    return (
        pd.DataFrame({"feature": prepared.columns, "weight": coefficients})
        .assign(magnitude=lambda d: d["weight"].abs())
        .sort_values("magnitude", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )
