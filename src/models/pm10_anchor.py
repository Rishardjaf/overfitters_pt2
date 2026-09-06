"""Rung 1 — scale PM10 at the observation hour into a next-hour PM2.5 level.

Persistence was the old baseline and the revised release deletes the column it
stood on. PM10 is the natural replacement: at the observation hour it carries
r = +0.848 with the target, against +0.762 for CO and +0.643 for NO2, and a
single global line through it scores 41.08 RMSE on the primary fold where a
station x hour-of-day climatology scores 101.40 against a constant's 102.04.
Diurnal and seasonal means are worthless here because PM2.5 is set by weather
regime, not by the clock; PM10 is the only cheap thing that knows the regime.

Per-station coefficients were measured and buy 0.06 RMSE. That is far inside the
day-block noise floor, so the fit is global and the baseline stays one number.

The slope, intercept and the median used to fill a blank anchor are all fitted on
training rows alone. A median taken across the holdout would be train/test
contamination that `src.leakage` cannot see, because it is not a lookahead within
the frame.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config

MIN_CONCENTRATION = 0.0


@dataclass(frozen=True)
class AnchorFit:
    """The three numbers the baseline learns, all from training rows."""

    intercept: float
    slope: float
    anchor_median: float


def fit(train: pd.DataFrame) -> AnchorFit:
    """Least squares of the target on the anchor, over rows where both are real."""
    anchor = train[config.ANCHOR_COL]
    target = train[config.TARGET_COL]
    usable = anchor.notna() & target.notna()
    if not usable.any():
        raise ValueError("No training row carries both the anchor and the target.")
    slope, intercept = np.polyfit(anchor[usable], target[usable], 1)
    return AnchorFit(
        intercept=float(intercept),
        slope=float(slope),
        anchor_median=float(anchor[usable].median()),
    )


def predict(fitted: AnchorFit, frame: pd.DataFrame) -> pd.Series:
    """Apply the fitted line, filling a blank anchor with the training median.

    Falling back to the most recent lag before the median keeps a station that has
    simply missed one reading on its own recent level rather than the city's.
    """
    anchor = frame[config.ANCHOR_COL]
    for fallback in (f"{config.ANCHOR_COL}_lag1", f"{config.ANCHOR_COL}_lag2"):
        if fallback in frame.columns:
            anchor = anchor.fillna(frame[fallback])
    anchor = anchor.fillna(fitted.anchor_median)
    prediction = fitted.intercept + fitted.slope * anchor
    return prediction.clip(lower=MIN_CONCENTRATION).rename("pm10_anchor")
