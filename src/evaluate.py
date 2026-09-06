"""Scoring and the ladder comparison table.

RMSE = sqrt( (1/n) · Σᵢ (yᵢ − ŷᵢ)² )

  yᵢ  — actual next-hour PM2.5 for station-hour i, µg/m³
  ŷᵢ  — predicted next-hour PM2.5 for the same station-hour, µg/m³
  n   — number of scored station-hours

Read as: square each error, average them, take the square root. A single
100 µg/m³ miss during a pollution episode costs as much as a hundred 10 µg/m³
misses on ordinary hours, so the tail dominates the score.
"""

import numpy as np
import pandas as pd


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    """Root mean squared error over the rows where both values are present."""
    frame = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if frame.empty:
        return float("nan")
    return float(np.sqrt(np.mean((frame["actual"] - frame["predicted"]) ** 2)))


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    """Mean absolute error — reported alongside RMSE to show the tail's influence."""
    frame = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if frame.empty:
        return float("nan")
    return float(np.mean((frame["actual"] - frame["predicted"]).abs()))


def ladder_table(scores: dict[str, dict[str, float]], baseline: str = "pm10_anchor") -> pd.DataFrame:
    """One row per rung, with each rung's improvement over the baseline.

    A rung that fails to beat the baseline is reported as such rather than quietly
    dropped — that is a finding, not an embarrassment.
    """
    table = pd.DataFrame(scores).T
    if baseline in table.index:
        reference = table.loc[baseline, "rmse"]
        table["rmse_vs_baseline"] = (
            (reference - table["rmse"]) / reference * 100
        ).round(2)
        table["beats_baseline"] = table["rmse"] < reference
    return table.round(4)
