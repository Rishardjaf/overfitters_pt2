# Experiment log

One row per run. Append, never rewrite — a log you edit is a log you cannot trust.

**Headline** is the mean RMSE over the heating-season folds (0 and 2), which mirror
the test period's season. **Δ persistence** is headline minus the persistence RMSE
on the same folds; negative means the model is better than doing nothing.

| Date | Config | Git SHA | Fold RMSEs (f0 / f1 / f2 / f3) | Headline | Δ persistence | Notes |
|---|---|---|---|---:|---:|---|
| 2026-09-06 | `scripts/v2/run_experiment.py --model persistence` | 3cc4cc5+ | 20.385 / 15.706 / 22.247 / 15.242 | 21.316 | 0 | yardstick; fold 3 reproduces PROJECT.md's 15.24 |
| 2026-09-06 | ridge, jihad (124), level | 3cc4cc5+ | 17.957 / 14.167 / 20.428 / 13.714 | 19.193 | −2.12 | in-fold impute+robust-scale+one-hot; alpha on inner window |
| 2026-09-06 | ridge, jihad, Δ | 3cc4cc5+ | 17.664 / 14.022 / 20.115 / 13.630 | 18.890 | −2.43 | Δ beats level on every fold (Findings/v2/01) |
| 2026-09-06 | ridge, jihad, level, predictors clipped | 3cc4cc5+ | 18.347 / 14.127 / 27.007 / 14.070 | 22.677 | +1.36 | **clipping the anchor is catastrophic on a level model** |
| 2026-09-06 | lightgbm, jihad, level | 3cc4cc5+ | 17.469 / 13.455 / 21.333 / 13.226 | 19.401 | −1.92 | Δ gate control arm |
| 2026-09-06 | lightgbm, jihad, Δ | 3cc4cc5+ | 17.181 / 13.254 / 19.401 / 13.234 | 18.291 | −3.03 | **Δ wins the gate** (Findings/v2/02) |
| 2026-09-06 | lightgbm, jihad, Δ, fixed 600 rounds | 3cc4cc5+ | 17.479 / 13.228 / 19.700 / — | 17.59* | | *partial; more rounds hurt — previous-season early stopping is right |
| 2026-09-06 | lightgbm, v2 (169), Δ | 3cc4cc5+ | 16.364 / 12.849 / 18.593 / 12.718 | 17.478 | −3.84 | motion/regime/quality/meteorology features (Findings/v2/03) |
| 2026-09-06 | lightgbm, lean (141), Δ | 3cc4cc5+ | 16.314 / 12.884 / 18.621 / 12.764 | 17.468 | −3.85 | v2 minus products and ratios: dead weight confirmed |
| 2026-09-06 | lightgbm, v2 − motion group | 3cc4cc5+ | 16.980 / 13.219 / 19.263 / 13.406 | 18.122 | −3.19 | ablation: motion is worth 0.64 |
| 2026-09-06 | lightgbm, v2 − regime/quality/met | 3cc4cc5+ | 16.427 / 12.905 / 18.766 / 12.736 | 17.596 | −3.72 | ablation: +0.12 |
| 2026-09-06 | lightgbm, v2 − weather lags / − pollutant lags / − rolling / − ratios | 3cc4cc5+ | — | 17.442 / 17.507 / 17.507 / 17.452 | | ablation: all within noise |
| 2026-09-06 | lightgbm, lean_nb (169), Δ | 3cc4cc5+ | 16.252 / 12.812 / 18.550 / 12.436 | 17.401 | −3.92 | per-neighbour station columns; better on all 4 folds |
| 2026-09-06 | lightgbm, slim_nb (129), Δ | 3cc4cc5+ | 16.272 / 12.814 / 18.500 / 12.548 | 17.386 | −3.93 | lean_nb minus lag groups; tie |
| 2026-09-06 | lightgbm, lean, Δ — weights heating / both / mdl300 / leaves31 / reg / ff0.5 | 3cc4cc5+ | — | 17.456 / 17.430 / 17.495 / 17.521 / 17.494 / 17.481 | | tuning plateau (Findings/v2/05) |
| 2026-09-06 | lightgbm, lean_nb, Δ, seeds 1 / 2 | 3cc4cc5+ | — | 17.374 / 17.450 | | seed noise band ±0.04 |
| 2026-09-06 | xgboost, lean, Δ | 3cc4cc5+ | 16.333 / 12.901 / 18.578 / 12.900 | 17.456 | −3.86 | model zoo (Findings/v2/04) |
| 2026-09-06 | xgboost, lean_nb, Δ (seed 42 / 1) | 3cc4cc5+ | 16.255 / 12.801 / 18.401 / 12.743 | 17.328 / 17.381 | −3.99 | best single model |
| 2026-09-06 | catboost, lean, Δ | 3cc4cc5+ | 16.381 / 12.913 / 18.475 / 12.665 | 17.428 | −3.89 | most decorrelated tree (0.978) |
| 2026-09-06 | histgb, lean, Δ | 3cc4cc5+ | 16.420 / 12.871 / 18.588 / 12.682 | 17.504 | −3.81 | |
| 2026-09-06 | extratrees, lean, Δ | 3cc4cc5+ | 16.827 / stopped | — | −3.56 (f0) | dropped after fold 0 |
| 2026-09-06 | ridge, lean, Δ | 3cc4cc5+ | 17.310 / 13.918 / 19.467 / 13.159 | 18.389 | −2.93 | motion features help the linear model too |
| 2026-09-06 | equal-weight average: lightgbm ×2 seeds + xgboost ×2 seeds, lean_nb | 3cc4cc5+ | 16.178 / — / 18.393 / — | 17.286 | −4.03 | 2-model recipe; `v2_final_lgbm_xgb_seedavg.csv` with 3 seeds and lr 0.02 × 700 |
| 2026-09-06 | catboost, lean_nb, Δ | 3cc4cc5+ | 16.341 / 12.825 / 18.498 / 12.729 | 17.419 | −3.90 | lowest residual correlation with the others (0.977) |
| 2026-09-06 | equal-weight average: lightgbm ×3 + xgboost ×2 + catboost ×1, lean_nb | 3cc4cc5+ | 16.170 / — / 18.379 / — | 17.275 | −4.04 | **final recipe** (Findings/v2/06); `v2_final_3model_seedavg.csv`, 3 seeds per model |

## Reference points

| Baseline | Train RMSE |
|---|---:|
| Global mean | 77.78 |
| Persistence (`current_PM2_5`) | 19.67 |
| Persistence, 2016-03-01 onward | 15.24 |

## Conventions

- `Δ persistence = model RMSE - persistence RMSE`; negative is better.
- A change worth under ~0.05 RMSE is noise unless it reproduces across folds.
- Record the failure too. A negative result that stops the team re-trying an idea
  is worth as much as a positive one.
- If a validation RMSE comes in below ~8, assume leakage and find it before
  recording the run as a success.
