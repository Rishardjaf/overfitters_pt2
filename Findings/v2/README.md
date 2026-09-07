# Findings v2

Measured results from executing the documents in `plans v2/`. One file per plan, written in
plain terms so they can be reviewed later without re-running anything.

Every number here comes from the time-aware expanding-window folds in
`src/pm25/validation.py`. **Headline = mean RMSE over the heating-season folds (0 and 2)**,
on rows where `current_PM2_5` is present, next to persistence on the same rows. The hidden
test set is easier than those folds (persistence ≈ 19.7 on test vs 21.32 on the heating
folds), so read *relative* improvement, not absolute numbers.

| # | File | One-line result (headline RMSE on heating folds; persistence = 21.32) |
|---|---|---|
| 01 | [01-baselines-and-ridge.md](01-baselines-and-ridge.md) | Ridge on the notebook's own 124 features, preprocessed inside the fold, scores **18.89** (Δ target) — the 19.9 leaderboard score was a pipeline problem, not a model problem. Clipping predictors wrecks a level model (+4.8 on fold 2). |
| 02 | [02-lightgbm-delta-gate.md](02-lightgbm-delta-gate.md) | LightGBM on the same features: **18.29**. Δ beats the raw level by 1.1 (all of it on the extreme winter fold). Gains are mostly on rapid falls; calm hours pay a little. |
| 03 | [03-feature-set-v2.md](03-feature-set-v2.md) | **Motion features** (other stations' 1h change, co-pollutant changes, acceleration) are worth **−0.81 → 17.48**; level×level products, ratios and weather lags are dead weight; per-neighbour columns another −0.07 → **17.40**. |
| 04 | [04-model-zoo.md](04-model-zoo.md) | LightGBM / XGBoost / CatBoost / HistGB all within 0.08 (17.43–17.50 on `lean`); ExtraTrees 0.5 worse, Ridge 0.9 worse. No better algorithm — the target and features did the work. |
| 05 | [05-weighting-and-tuning.md](05-weighting-and-tuning.md) | Every hyper-parameter arm within ±0.05 — a plateau. Seed noise alone is ±0.04. Seed-averaging LightGBM+XGBoost is worth **−0.10 → 17.29**. |
| 06 | [06-blend-and-submission.md](06-blend-and-submission.md) | Fitted blends ≈ equal weights; Ridge adds nothing. Round count measured on a curve (lr 0.02 × 700). **Submit `submissions/v2_final_3model_seedavg.csv`** (LightGBM + XGBoost + CatBoost, 3 seeds each; CV ≈ 17.23); `v2_final_lgbm_xgb_seedavg.csv` is the 2-model backup. Expected leaderboard ≈ 16.3–17.0. |
| 07 | [07-missing-anchor.md](07-missing-anchor.md) | The 0.7% rows with no `current_PM2_5` are 10× noisier and no fallback wins consistently; hedge with the mean of network-mean and level model (F). Worth ~−0.25 operational. |

## The whole story in one table

| step | headline | Δ vs previous | why |
|---|---:|---:|---|
| persistence | 21.32 | — | the yardstick |
| Ridge, level, notebook features | 19.19 | −2.13 | proper in-fold preprocessing |
| Ridge, Δ target | 18.89 | −0.30 | anchor moved to the left-hand side |
| LightGBM, Δ, notebook features | 18.29 | −0.60 | non-linear, regime-dependent structure |
| + motion / regime / quality features (v2) | 17.48 | −0.81 | *what moved in the last hour*, city-wide |
| − products, ratios (lean) | 17.47 | −0.01 | dead weight removed |
| + per-neighbour station columns (lean_nb) | 17.40 | −0.07 | which station leads which |
| seed-averaged LightGBM + XGBoost | 17.29 | −0.11 | variance reduction |
| + CatBoost as a third member | 17.28 | −0.02 | the least-correlated tree |
| lr 0.02 with the measured round count | ≈17.23 | ≈−0.04 | round curve |

**−19% against persistence, every step in the same direction on both heating folds.**

Reproduce any row with `scripts/v2/run_experiment.py` — the exact command is in each file.
Raw per-fold JSON and out-of-fold predictions are in `artifacts/v2/` (gitignored, regenerable).

## Integrity checks performed before any result was recorded

- `make audit` (token scan for negative shifts / centred windows / backfills): clean.
- `scripts/v2/test_causality.py`: the panel rebuilt on data truncated at 2015-06-15 is
  identical, for every one of 207 features on 240,912 rows, to the full build. Nothing reads
  the future.
- Persistence through the harness reproduces the PROJECT.md figures (fold 3 = 15.24).
- No fold RMSE anywhere in these findings is below 8.
