# Plans v2 — beating 19.7 on the hidden test set

**Starting point.** The Ridge model built from `notebooks/jihad/feature eng & selection.ipynb`
(124 features, raw-level target) scored **19.918** on the hidden test set. Persistence
(`ŷ = current_PM2_5`) scores ~19.7 there. So the current model is *worse than doing nothing*.
The target for this round is **18.0**.

**What is different from `plans/`.** The v1 plans are design documents that were never run and
none of `src/pm25/` is implemented. v2 is execution-first: one shared, tested harness
(`scripts/v2/`), then a short sequence of measured experiments, each recorded in
`Findings/v2/`. Every number quoted is from the time-aware folds in
`src/pm25/validation.py`; nothing is tuned on the leaderboard.

## Order of execution

| # | Plan | Question it answers | Status → finding |
|---|---|---|---|
| 01 | [Harness and baselines](01-harness-and-baselines.md) | What does persistence score per fold, and *why* did Ridge lose to it? | done — persistence 21.32; Ridge Δ 18.89; the notebook's pipeline, not the model, was the problem |
| 02 | [LightGBM Δ gate](02-lightgbm-delta-gate.md) | Does a tree on the jihad features beat persistence? Δ target or raw level? | done — Δ wins by 1.1; LightGBM 18.29 |
| 03 | [Feature set v2](03-feature-set-v2.md) | Which untested features add signal; which existing ones are dead weight? | done — motion features −0.81; products/ratios/weather lags dead; neighbours −0.07 → 17.40 |
| 04 | [Model zoo](04-model-zoo.md) | Does XGBoost / CatBoost / ExtraTrees / HistGB beat LightGBM on the same features? | done — all boosters within 0.08; ExtraTrees and Ridge dropped; CatBoost most decorrelated |
| 05 | [Weighting and tuning](05-weighting-and-tuning.md) | Does weighting toward the heating season or tuning tree depth move the headline? | done — plateau; seed noise ±0.04; seed-averaging −0.10 |
| 06 | [Blend and submission](06-blend-and-submission.md) | Does a blend of OOF predictions beat the best single model? Final retrain and CSV. | done — equal-weight seed average; `submissions/v2_final_lgbm_xgb_seedavg.csv` |
| 07 | [Missing-anchor rows](07-missing-anchor.md) | Can a dedicated fallback cut the cost of the 0.7% rows with no `current_PM2_5`? | done — no stable winner; hedge F (mean of network mean and level model) |

Each plan is gated on the one before it: there is no point running four models on the
wrong target or on a feature set with a leak.

## Shared harness (`scripts/v2/`)

- `features_v2.py` — builds the panel: train and test concatenated on one timeline (test rows
  need their own backward lags — legitimate), reindexed to a **complete hourly grid per
  station** so `lag 1` is always genuinely `t−1`, then every feature group. All windows are
  strictly backward: `shift(k>0)`, `rolling(..., closed="left")`, `groupby.transform` at the
  same hour only. Padding rows are dropped before fitting or scoring.
- `cv.py` — materialises `EXPANDING_FOLDS`, computes persistence per fold, RMSE, the signed-tail
  split (fall ≤ −30 / calm / rise ≥ 30), and writes a markdown table.
- `run_experiment.py` — one CLI, one experiment name, one results markdown + OOF parquet under
  `artifacts/v2/`.

## Rules honoured in every plan

1. `Data/*.csv` are never written to.
2. No negative shift, no centred window, no backfill. `make audit` scans `scripts/`, so the
   harness is covered by the leakage guard.
3. Folds are timestamp boundaries only. Never a random split.
4. Any fitted statistic (imputer median, scaler, encoder) is fitted inside the training fold.
5. Every result is quoted as `model RMSE − persistence RMSE` on the same fold, and the headline
   is the mean over the two heating-season folds (0 and 2). Moves under ~0.05 are noise.
6. A fold RMSE below ~8 is leakage, not a result.
7. No external data. The test set's target is never reconstructed (no next-row self-join).

## How early stopping is kept honest

Choosing the round count on the fold being reported inflates the score. For each outer fold the
round count is chosen on the **same season one year earlier** (which lies wholly inside the
outer training window), then the model is refitted on the full outer training window for that
many rounds and scored once on the untouched validation window.

## Missing-anchor rows

`current_PM2_5` is missing on 0.7% of rows; Δ is undefined there. Δ-models are fitted on
anchor-present rows only. At prediction time those rows take the raw-level model's prediction
(the "operational" score). Comparisons between targets are made on the common anchor-present
population (the "gate" score). Both are reported.
