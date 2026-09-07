"""RMSE and the diagnostic breakdowns that make an RMSE number interpretable.

Not yet implemented — see the roadmap in PROJECT.md.

The competition metric is RMSE on the raw PM2.5 scale. Because RMSE is
tail-dominated, a single averaged number hides the failure mode that decides the
leaderboard: rapid-onset pollution events. Always report the breakdowns
alongside it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """The competition metric."""
    raise NotImplementedError


def score_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """RMSE broken down by station, month, hour-of-day, and concentration decile.

    `meta` supplies the grouping columns for the same rows as `y_true`.
    """
    raise NotImplementedError


def vs_persistence(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    anchor: np.ndarray,
) -> dict[str, float]:
    """Model RMSE, persistence RMSE, and the delta between them.

    `anchor` is `current_PM2_5`. Define delta as model RMSE minus persistence
    RMSE, so a negative delta means the model is better than doing nothing.
    """
    raise NotImplementedError
