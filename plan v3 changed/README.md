# Plan v3 changed — modelling without the PM2.5 anchor

**Read `new data/migration.md` first**, then `notebooks/eda_v2/01_eda_new_data.ipynb`. The
migration doc explains what changed (PM2.5 removed from both train and test, everything else
byte-identical) and the safety rules. The EDA measured what signal is left. This file is the
index and the shared context for the plans that act on both.

## The bar

| baseline (no feature engineering) | RMSE |
|---|---:|
| Ridge | 31.99 |
| LightGBM | 27.60 |
| **Target for this round** | **17.0** |

Persistence (`ŷ = current_PM2_5`) cannot be computed any more. Every result is quoted against
the two baselines above, measured on the same time-aware folds as before
(`src/pm25/validation.py`; headline = mean of the two heating-season folds).

**Honesty note on the 17.0 target.** The deleted anchor alone scored 19.67 by doing nothing.
Reaching 17.0 *without* it means out-predicting the old easy baseline using only correlated
proxies — the EDA puts the best remaining single signal at r = 0.85 vs the anchor's 0.97. That
may or may not be reachable honestly; the plans below go as deep as the data allows and the
findings will say plainly how far they got. What is *not* on the table: reconstructing PM2.5
from chained self-predictions on the test set (plan 05, still gated), or any feature that
reads a row later than `t`.

## What the EDA changed about the plan

- **Motion features are ~3× weaker than last round** (best 1h-change r = 0.11 vs 0.33 before;
  cross-station *changes* agree at 0.28 vs *levels* at 0.84). Plan 02 stays, but as a
  secondary correction, and is measured with first *and* second derivatives so the question is
  settled rather than assumed.
- **Levels are the backbone now**, the reverse of last round. Lags/rolling stats of PM10, CO,
  NO2 and the cross-station *levels* are the first things to build, not the last.
- **PM10 is a loose proxy for PM2.5** (median ratio 0.77, wide spread). That makes the
  proxy-anchor idea (plan 03) the more promising of the two structural tricks, and it opens
  a new angle: model the *fine fraction* (PM2.5 ÷ PM10) rather than the level (plan 06).

## Plans

| # | Plan | Question | Status |
|---|---|---|---|
| 01 | [Baselines and harness](01-new-baselines-and-signal-scan.md) | Reproduce 31.99 / 27.60 through the harness; establish the diagnostic yardsticks | done → `Findings/v3/01` |
| 02 | [Feature build-up: levels, lags, motion (1st + 2nd derivative), network, neighbours](02-copollutant-motion-features.md) | Which feature groups earn their place, built cumulatively and then ablated | done → `Findings/v3/02` (27.90) |
| 03 | [Proxy-anchor residual](03-proxy-anchor-residual.md) | Does modelling `target − k·PM10` beat modelling the level? | done → −0.3; superseded by the ratio target (`Findings/v3/06`) |
| 04 | [Algorithms and combinations](04-model-recipe-transfer.md) | LightGBM / XGBoost / CatBoost / Ridge / HistGB on the settled set; seed averaging; blends | **in progress** — runs queued; see `task/HANDOFF.md` |
| 05 | [Gated: recursive reconstruction](05-optional-recursive-forecast.md) | Off unless explicitly approved | **not started** |
| 06 | [Different angles: target transforms and the fine fraction](06-target-angles.md) | log target, ratio target, in-fold climatology encoding, heating weights | done → **ratio target −2.1 → 25.94** (`Findings/v3/06`) |
| 07 | [Feature selection: pairwise redundancy and ablation](07-feature-selection.md) | Prune the winning set to what actually matters | done → `Findings/v3/07` |

## Order

01 → 02 (forward build) → 07 (prune) → 03 and 06 on the pruned set (targets) → 04 (models,
seeds, blend) → submission. 02 and the target experiments both use LightGBM with the same fixed
parameters so every comparison changes one thing at a time.

## Shared harness

`scripts/v3/` — `features_v3.py` (panel builder; no PM2.5 anywhere in the feature groups, hard
assertion), `run_experiment.py` (one CLI, targets: `level`, `proxy`, `log1p`, `ratio`),
`test_causality.py`. Folds, fitters, weighting and scoring are imported from `scripts/v2/` so
the methodology is identical to the v2 findings.

**Diagnostic-only anchor.** Validation rows are training rows, so their true "PM2.5 now" is
known from the previous row's label. The harness reconstructs it as `diag_pm25_now` **for
scoring only** — it lets each result be broken into rapid falls / calm / rapid rises and quoted
next to what persistence *would* have scored, exactly as in v2. A hard assertion refuses to
run if that column, or anything named like it, appears in a feature list.

## What is carried over from v2 without re-testing

The complete-hourly-grid construction, the causal feature-building pattern, the time-aware
folds, the leakage audit, the previous-season early-stopping rule, and the v2 conclusions that
boosted trees plateau within ~0.1 of each other and that hyperparameter tuning is noise.
