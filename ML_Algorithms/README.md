# ML Algorithms — the ladder

One folder per test. Each holds a `RESULTS.md`, a `metrics.json`, the diagnostic figures, and
the `submission.csv` that test produced.

Every rung is scored on the same season-matched holdout — **2015-09_to_2016-03**, trained only on the data
preceding it ([ADR-0002](../docs/adr/0002-season-matched-validation.md)) — and the anchor is
re-scored on the same rows so the comparison is like-for-like. A rung that fails to beat
the PM10 anchor is reported as failing rather than quietly dropped.

| Test | Algorithm | RMSE | MAE | vs baseline | Beats baseline | Fit (s) |
|---|---|---|---|---|---|---|
| [01_baseline_pm10_anchor](01_baseline_pm10_anchor/RESULTS.md) | pm10_anchor | 40.664 | 24.650 | baseline | — | 0.0 |
| [02_ridge](02_ridge/RESULTS.md) | ridge | 34.396 | 20.261 | +15.41% | yes | 4.7 |
| [03_lightgbm](03_lightgbm/RESULTS.md) | lightgbm | 29.614 | 14.783 | +27.17% | yes | 31.9 |
| [04_delta_lightgbm](04_delta_lightgbm/RESULTS.md) | delta_lightgbm | 29.184 | 14.649 | +28.23% | yes | 33.8 |

**delta_lightgbm** leads on both RMSE (29.184) and MAE (14.649). Leading the table is not promotion: a rung is the submission only once the paired day-block bootstrap puts its primary-fold 90% interval entirely below zero.

## How blanks are treated

Not uniformly, and deliberately so ([ADR-0003](../docs/adr/0003-per-rung-imputation.md)).
Ridge cannot see a missing value, so it receives a station-by-hour-of-day median fill fitted
on training rows alone. LightGBM handles blanks natively and is fed the frame intact, because
the pattern of absence is itself information.

## Reproduce

```bash
.venv/bin/python -m src.cache            # data/processed/train.parquet and test.parquet
.venv/bin/python -m src.experiments all  # every test, then this table
```
