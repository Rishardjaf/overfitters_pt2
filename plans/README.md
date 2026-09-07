# Modelling plans — team overfitters

Seven plans, one per candidate algorithm. Each is a design document: rationale,
model-specific data preparation, validation protocol, success criteria and risks.
**No plan has been executed.** Nothing here is a measured result.

Read this file first — it holds the context every plan assumes rather than repeating.

| # | Plan | Role | Target | Status |
|---|---|---|---|---|
| 01 | [LightGBM](01-lightgbm.md) | Primary workhorse; sets the Δ decision gate | **Δ (change)** | Not run |
| 02 | [CatBoost](02-catboost.md) | Strongest categorical handling | Gated on 01 | Not run |
| 03 | [XGBoost](03-xgboost.md) | Ensemble diversity | Gated on 01 | Not run |
| 04 | [Ridge / ElasticNet](04-ridge-elasticnet.md) | Decorrelated blend member | Gated on 01 | Not run |
| 05 | [RandomForest / ExtraTrees](05-randomforest-extratrees.md) | Bagged counterweight to boosting | Gated on 01 | Not run |
| 06 | [Two-stage spike model](06-two-stage-spike.md) | Targets the rows that carry the score | Gated on 01 | Not run |
| 07 | [Blend](07-blend.md) | Combines the above | Inherits members' targets | Not run |

## The Δ decision gate

Plan 01 is the only plan that commits to training on **Δ = `PM2_5_next_hour` − `current_PM2_5`**
rather than on the raw level. It runs first and is measured both ways.

The result is recorded in `Findings/` (see the template there). Plans 02–06 each carry a
**Target** section that stays open until that finding lands. Then:

- If Δ wins by more than 0.05 RMSE on the heating-season folds and the win reproduces across
  folds, every remaining plan adopts Δ.
- If it is inside ±0.05 RMSE, the difference is noise. Prefer Δ anyway, on the grounds given
  in plan 01 (persistence is its embedded zero-residual baseline), but stop treating it as a
  measured advantage.
- If the raw level wins, plans 02–06 revert to the level and plan 01's reasoning is recorded
  as refuted.

Do not fan out plans 02–07 before this gate resolves. Running six models against the wrong
target wastes the hackathon's scarcest resource, which is measured comparisons.

## Shared context

### The task

Predict PM2.5 one hour ahead at 12 urban stations. Scored by RMSE on a hidden,
chronologically later test set. Train: 360,954 rows, 2013-03-01 → 2016-08-31. Test: 51,063
rows, 2016-08-31 23:00 → 2017-02-28, **September–February only**.

### Baselines every plan is quoted against

| Baseline | Train RMSE |
|---|---:|
| Global mean | 77.78 |
| **Persistence (ŷ = `current_PM2_5`)** | **19.67** |
| Persistence, 2016-03-01 onward | 15.24 |

A plan that does not beat persistence on the heating-season folds has failed, regardless of
how it compares to the other plans.

### EDA findings the plans reference

| Ref | Finding |
|---|---|
| **F1** | Persistence RMSE 19.67 equals the standard deviation of Δ (mean Δ = 0.11 — unbiased). |
| **F2** | The worst 1% of rows carry 47% of total squared error; the worst 10% carry 83%. |
| **F3** | Spikes (Δ ≥ 30) are 3.04% of rows but 31.4% of squared error. Persistence RMSE 63.0 on them vs 16.5 elsewhere. |
| **F4** | Δ autocorrelation is 0.184 at lag 1, ≈0.00 at lag 2, 0.025 at lag 24 — one hour of momentum, no daily echo. |
| **F5** | Level autocorrelation is 0.97 at lag 1 with a visible lag-24 echo. |
| **F6** | Cross-station Δ correlation averages 0.30 off-diagonal — weak but real, and onset is city-wide. |
| **F7** | Every raw column's linear correlation with Δ is near zero. Signal is non-linear and regime-dependent. |
| **F8** | Test is 18% more polluted than train (mean level 92.3 vs 78.0) and covers only Sep–Feb. |
| **F9** | Monthly mean Δ is flat (0.01–0.21) while monthly mean level swings 53.5–96.2. Δ's *scale* still swings (std 13.7–26.8). |
| **F10** | Spikes are **two-sided and asymmetric**. Rising events (Δ ≥ 30) are 3.04% of rows and 31.4% of squared error; **falling events (Δ ≤ −30) are 3.73% of rows and 45.6%** — the larger half. Together \|Δ\| ≥ 30 is 6.78% of rows and **77.0%** of all squared error, against a calm-hour persistence RMSE of just 9.77. |

### Data problems the plans reference

| Ref | Problem | Scale |
|---|---|---|
| **P1** | Missing `current_PM2_5`. **Structural**: the 2,466 affected train rows are exactly the rows following a timeline gap, so the previous hour is missing by construction. 371 test rows are affected and must still be predicted. | 0.68% train / 0.73% test |
| **P2** | Timeline gaps against a complete hourly calendar. Steps of 2h (1,459), 3h (544), up to 8h+. | 7,674 missing slots |
| **P3** | `current_PM2_5` > `PM10`, physically impossible. Varies by station (3.88%–9.66%). Median violation 19 µg/m³. | 21,012 rows (5.82%) |
| **P4** | `CO` is coarsely quantised: 131 distinct values, 99% multiples of 100, range 100–10,000. | 35% consecutive-identical |
| **P5** | Stuck sensors. Longest flat runs: `CO` 76h, `current_PM2_5` 39h, `PM10` 24h, `TEMP` 19h. | — |
| **P6** | Blank `wd`, 9× more common in test than train. Blank rows average ~92 µg/m³ against a 78 overall mean. | 0.22% train / 1.97% test |
| **P7** | Test years (2016–17) extend past train (2013–16), and the test months are the polluted half of the year. | — |
| **P8** | `RAIN` is 95.9% zeros with a tail to 72.5mm. | — |
| **P9** | Heavy-tailed target. Max 999 (once in 360k rows — not a censoring ceiling). | — |

Other pollutant missingness: `CO` 4.39%, `O3` 2.35%, `NO2` 2.08%, `SO2` 1.29%, `PM10` 0.53%,
meteorology ~0.05%.

### Preparation shared by every plan

This layer is model-independent and must be built once, not per model:

1. Concatenate train and test on one timeline. Legitimate and necessary — test rows need
   their own backward lags. Only the *direction* of a window can leak.
2. Reindex each station onto a gap-free hourly calendar (**P2**), marking inserted rows as
   padding. Without this, a one-hour lag silently means "the previous surviving row", which is
   sometimes five hours earlier.
3. Build backward-only features on that grid: lags, rolling means, rolling standard
   deviations, same-hour cross-station aggregates (**F6**).
4. Add missingness indicators, gap-length, flat-run counters (**P5**) and the `pm_inconsistent`
   flag (**P3**).
5. Filter back to real rows before fitting or predicting.
6. Never repair **P3** by clipping either column, and never winsorise the target (**P9**,
   **F2**) — the tails are where the score is decided.

What each plan then specifies is the **representation layer**: how that one clean panel is
adapted to the model's requirements. That layer is genuinely model-specific and is the bulk
of each document.

### Validation protocol, common to all plans

Expanding-window folds split on timestamps. The headline number is the mean RMSE over the
**heating-season folds**, because the test period is Sep–Feb and materially more polluted
(**F8**). Never a random split; never tuning against the public leaderboard.

Every plan reports: the headline CV RMSE, the per-fold breakdown, the delta against *that
fold's* persistence score, and separate breakdowns for rapid falls, calm hours and rapid rises
(**F10**). A single averaged number hides the failure mode that costs the most. Define delta
against persistence as `model_rmse - persistence_rmse`; negative is better.

Moves smaller than ~0.05 RMSE are noise unless they reproduce across folds.

### Guard rails

- Run the leakage guard before generating any submission.
- Any statistic used in preparation — medians, category lists, scaler parameters — is fitted
  on the training fold only. Fitting on train+test leaks future statistics backwards.
- A validation RMSE below ~8 means leakage, not success. Stop and hunt for it.
- Fix seeds everywhere. Every run records its config, git SHA, CV score and feature list.
