"""Decide between two rungs with the paired day-block bootstrap.

A holdout RMSE difference here is routinely smaller than its own uncertainty. A
pollution episode is one event spanning many consecutive hours across all twelve
stations, so residuals are correlated within a day and the independent sample is
nearer a fold's ~182 day blocks than its ~51,000 rows. Naive row-wise SE is about
0.07; the honest one is 0.2-0.4. Ranking two models on the point estimate alone
would promote noise about as often as signal.

Both rungs are scored on identical rows so the difference is paired.

    .venv/bin/python -m scripts.modeling.gate 03_lightgbm 04_delta_lightgbm
"""

import sys

import pandas as pd

from src import cache, config, uncertainty, validation
from src.experiments import registry

INTERVAL = 0.90


def _labelled(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.dropna(subset=[config.TARGET_COL])


def compare(reference_name: str, variant_name: str) -> None:
    reference, variant = registry.resolve(reference_name), registry.resolve(variant_name)
    train_features = cache.load_train_features()

    for fold in validation.season_matched_folds(train_features):
        train_rows = train_features.loc[fold.train_mask]
        holdout_rows = train_features.loc[fold.holdout_mask]
        scored = _labelled(holdout_rows)

        predictions = {
            rung.algorithm: rung.adapter(train_rows, holdout_rows)
            .predictions.reindex(scored.index)
            for rung in (reference, variant)
        }

        result = uncertainty.compare_rmse(
            actual=scored[config.TARGET_COL],
            reference=predictions[reference.algorithm],
            variant=predictions[variant.algorithm],
            timestamps=scored[config.TIME_COL],
            interval=INTERVAL,
        )
        verdict = "CONCLUSIVE" if result.is_conclusive else "INCONCLUSIVE"
        print(f"\n{fold.name}  ({len(scored):,} rows, {result.blocks} day blocks)")
        print(f"  {reference.algorithm:16s} RMSE {result.rmse_reference:.4f}")
        print(f"  {variant.algorithm:16s} RMSE {result.rmse_variant:.4f}")
        print(f"  delta {result.delta_rmse:+.4f}   "
              f"{int(INTERVAL * 100)}% interval [{result.lower:+.4f}, {result.upper:+.4f}]"
              f"   {verdict}")


if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2])
