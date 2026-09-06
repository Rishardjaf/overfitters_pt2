"""The one shared path from a raw CSV frame to a model-ready feature frame.

Both the notebook and the leakage check call this, so there is a single
definition of what a feature is.
"""

import pandas as pd

from src import features, grid, impute


def build_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Complete the hourly grid, fill forward with flags, then build features."""
    return (
        raw.pipe(grid.complete_hourly_grid)
        .pipe(impute.forward_fill_with_flags)
        .pipe(impute.add_staleness)
        .pipe(features.build_features)
    )


def scored_rows(built: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """Drop the rows the grid invented — only real station-hours are ever scored."""
    return built.loc[built[id_column].notna()].reset_index(drop=True)
