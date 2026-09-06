"""Shared preparation for the rungs that cannot see a missing value.

Ridge and random forest both need a NaN-free matrix; LightGBM and histogram
gradient boosting do not, and are deliberately fed the frame with its blanks
intact. See docs/adr/0003.

The median-fill transform is fitted on the fold's training rows alone. Fitting it
on the holdout would leak that block's distribution into the features, which is
the whole reason this is a fitted object rather than a function of both frames.
"""

from dataclasses import dataclass

import pandas as pd

from src import config, impute
from src.features import dense_feature_columns


@dataclass(frozen=True)
class DenseFit:
    """Everything learned from the training rows, and nothing from the holdout."""

    columns: list[str]
    medians: pd.DataFrame
    fallback: pd.Series
    features: pd.DataFrame
    target: pd.Series

    @property
    def rows(self) -> int:
        return len(self.features)


def prepare(train: pd.DataFrame) -> DenseFit:
    """Fit the median fill and assemble the NaN-free training matrix.

    `apply_median_fill` only covers the raw predictors, so features derived from
    them — lags, rolling windows, deltas — can still be blank where a gap was too
    long to forward-fill. Those rows are dropped rather than fabricated; on the
    real training data about 91% survive.
    """
    medians = impute.fit_median_fill(train)
    filled = impute.apply_median_fill(train, medians)
    columns = dense_feature_columns(filled)

    usable = filled.dropna(subset=columns + [config.TARGET_COL])
    if usable.empty:
        raise ValueError(
            "No training rows survive the dense-frame filter. Every row has at "
            "least one blank derived feature — check the forward-fill limit."
        )
    return DenseFit(
        columns=columns,
        medians=medians,
        fallback=usable[columns].median(),
        features=usable[columns],
        target=usable[config.TARGET_COL],
    )


def transform(fit: DenseFit, frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the fitted fill to any frame and return its NaN-free feature matrix.

    Holdout rows are never dropped — every station-hour has to receive a
    prediction — so anything still blank after the fitted fill falls back to the
    training column median.
    """
    filled = impute.apply_median_fill(frame, fit.medians)
    return filled[fit.columns].fillna(fit.fallback)
