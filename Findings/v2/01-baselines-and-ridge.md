# Finding v2-01 — Baselines, and why the notebook's Ridge lost to persistence

**Plan:** `plans v2/01-harness-and-baselines.md` · **Features:** `jihad` (124 columns, faithful
re-implementation of `notebooks/jihad/feature eng & selection.ipynb`, built on the complete
hourly grid) · **Seed:** 42 · **Git SHA:** 3cc4cc5 (+ uncommitted v2 harness)

## Persistence per fold — the yardstick for everything else

| fold | window | rows | persistence RMSE | fall RMSE | calm RMSE | rise RMSE |
|---|---|---:|---:|---:|---:|---:|
| 0 **(heating)** | 2014-09 → 2015-02 | 50,799 | **20.385** | 70.0 | 9.83 | 61.3 |
| 1 | 2015-03 → 2015-08 | 51,938 | 15.706 | 58.1 | 9.31 | 60.1 |
| 2 **(heating)** | 2015-09 → 2016-02 | 51,349 | **22.247** | 77.1 | 9.24 | 72.8 |
| 3 | 2016-03 → 2016-08 | 51,678 | 15.242 | 60.6 | 9.21 | 53.9 |

- **Heating-season headline for persistence: 21.316.** Fold 3 reproduces PROJECT.md's 15.24, so
  the harness scores correctly.
- Winter is ~40% harder than summer for persistence. The hidden test set (persistence ≈ 19.7)
  sits between the two, so quote *relative* gains.
- Rapid falls and rises have persistence RMSE 60–77 against ~9.5 on calm hours. The score is
  decided in the tails, as the EDA said.

## The Ridge arms

All four arms: median imputation, robust scaling, one-hot `station`, all fitted **inside** the
training fold; alpha ∈ {0.1, 1, 10, 100, 1000} chosen on the previous-season inner window.

| arm | fold 0 | fold 1 | fold 2 | fold 3 | **headline** | Δ vs persistence |
|---|---:|---:|---:|---:|---:|---:|
| persistence | 20.385 | 15.706 | 22.247 | 15.242 | 21.316 | — |
| Ridge, raw level | 17.957 | 14.167 | 20.428 | 13.714 | 19.193 | −2.12 |
| Ridge, raw level, predictors clipped | 18.347 | 14.127 | **27.007** | 14.070 | 22.677 | **+1.36** |
| Ridge, Δ target | 17.664 | 14.022 | 20.115 | 13.630 | **18.890** | **−2.43** |
| Ridge, Δ target, predictors clipped | 17.772 | 13.942 | 19.906 | 13.559 | 18.839 | −2.48 |

Commands: `scripts/v2/run_experiment.py --model ridge --features jihad --target {level|delta} [--params '{"clip":true}']`

## What this says, in plain terms

1. **Ridge itself is fine.** With the preprocessing done properly (fit inside the fold, robust
   scaling, indicators kept) the *same 124 features* beat persistence by 2.1–2.4 RMSE on every
   fold. The 19.918 leaderboard score therefore did not come from the model class or the
   features — it came from whatever happened between the notebook and the CSV (unscaled inputs,
   a NaN fill, test lags computed on the test file alone so its first hours have no history, or
   a row-order mismatch). That pipeline is not in the repo, so this cannot be pinned further.
2. **Predicting Δ beats predicting the level, by 0.3 RMSE, on all four folds.** Moving
   `current_PM2_5` to the left-hand side stops the L2 penalty from shrinking the one coefficient
   that must stay at 1.
3. **Clipping predictors is catastrophic for a level model** (fold 2: 27.0). Winter 2015-16
   contains readings above the training 99.5th percentile; clipping the anchor caps the
   prediction exactly where the squared error is largest. For the Δ model clipping is neutral
   (−0.05, noise), because the anchor is added back unclipped. **Do not clip.**
4. **Missing-anchor rows are a hidden cost.** 0.7% of rows have no `current_PM2_5`; on them
   Ridge scores RMSE 40–80. They alone move the operational headline from 18.89 to 19.63
   (+0.74). The test set has 371 such rows. This deserves its own experiment (see plan 07).
5. **What carries linear signal** (largest |coef| in the Δ model): `PM10`, `PM10_lag_1h`,
   `PM25_x_WSPM`, `NO2`, `CO_lag_1h`, `pm25_gt_pm10`, `PM25_vs_network_mean`, `PM25_x_ddd`,
   `hour_cos/sin`. Near-zero: `CO_x_pm25_floor`, `NO2_x_pm25_floor`, `PM10_x_pm25_floor`,
   `WSPM_lag_24h`, `dayofweek_cos`, `PM25_roll_min_48h`, `PM10_missing`. The floor-flag
   interactions and the deepest weather lags are dead weight for a linear model.

## Verdict

- Persistence heating-season yardstick: **21.316**.
- Best Ridge: **18.890** (Δ target, unclipped), −2.43 vs persistence, same sign on all folds.
  This is the strongest Ridge can do on these features and is kept only as a blend candidate.
- Unblocks plan 02 (LightGBM) with Δ as the default target; feeds plan 06 (blend).
