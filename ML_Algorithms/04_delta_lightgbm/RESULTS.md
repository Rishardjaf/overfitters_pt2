# TEST 04 — delta_lightgbm

LightGBM on the deviation from the PM10 anchor. Removes the tree extrapolation ceiling that decides the tail.


## Scores

Season-matched holdout blocks, each trained only on the data preceding it
([ADR-0002](../../docs/adr/0002-season-matched-validation.md)). Persistence is re-scored on the
same holdout rows, so the comparison is like-for-like.

| Fold | Train rows | Holdout rows | RMSE | MAE | Persistence RMSE | vs persistence | Fit (s) |
|---|---|---|---|---|---|---|---|
| 2015-09_to_2016-03 | 257,927 | 51,349 | 29.184 | 14.649 | 40.664 | +28.23% | 33.8 |
| 2014-09_to_2015-03 | 155,190 | 50,799 | 26.955 | 16.773 | 43.425 | +37.93% | 23.9 |

**Beats the baseline** by 28.23% RMSE on the primary fold (29.184 against persistence's 40.664).

## Submission

`submission.csv` — 51,063 rows, produced by refitting on every labelled
training station-hour and predicting the hidden test window. It passed `submission.validate`:
row count, id set, no blanks, no negative concentrations.

## Reproduce

```bash
.venv/bin/python -m src.cache                      # build data/train.parquet if absent
.venv/bin/python -m src.experiments 04_delta_lightgbm
```

Total run time: 92.4s.

## Figures

### Predicted vs actual

Predicted against actual next-hour PM2.5 on the primary holdout.

![predicted_vs_actual](figures/predicted_vs_actual.png)

### Error by level

RMSE within each air-quality band, and how many hours sit in each.

![error_by_level](figures/error_by_level.png)

### Residuals over time

Daily RMSE and daily bias across the holdout block.

![residuals_over_time](figures/residuals_over_time.png)

### Top drivers

The features the model leaned on hardest.

![top_drivers](figures/top_drivers.png)
