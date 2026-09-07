# Plan 01 — LightGBM

**Role:** primary workhorse and the first real model. Also the plan that resolves the Δ
decision gate for every other plan.

**Status:** not run.

**Prerequisites:** the shared preparation layer in `plans/README.md` must exist first
(hourly grid, backward-only features, flags). This plan does not restate it.

---

## 1. Purpose

Produce the first model that beats persistence on the heating-season folds, and settle
whether training on Δ (the hour-over-hour change) is better than training on the raw level.

Two questions, one plan, because they need the same infrastructure and answering them
separately would cost a day.

## 2. Hypothesis

A gradient-boosted tree ensemble on backward lag, rolling and cross-station features, trained
on Δ with an L2 objective, beats persistence's 19.67 by a margin that survives across folds.

The margin will be modest. Persistence already removes ~75% of the mean baseline's error, and
every raw column's linear correlation with Δ is near zero (**F7**). Anyone promising a large
improvement has either leaked or not measured on time-aware folds.

## 3. Why LightGBM first

**The signal shape matches the model.** **F7** says the relationship between the predictors
and Δ is weak, non-linear and regime-dependent — dispersion behaves one way in stagnant air
and another in wind. Boosted trees find that kind of conditional structure without being told
where to look. A linear model cannot.

**Native NaN feature handling is not a convenience here, it is a requirement.** **P1** is
structural: 371 test rows have no `current_PM2_5`, and you must submit a prediction for every
one. LightGBM can route a missing *feature* down a learned branch. It cannot train on a missing
target, however, so the residual arm must exclude rows whose anchor makes Δ undefined and use
the raw-level arm as its production fallback. Section 4 specifies that path exactly.

**Native categorical handling.** `station` (12 levels) and `wd` (17 with blank as its own
category, **P6**) go in directly. No one-hot expansion, no ordinal encoding that implies a
false ordering on compass directions.

**Iteration speed decides hackathons.** 360k rows by roughly 80 features trains in seconds.
Being able to run twenty feature experiments in an afternoon is worth more than any single
model's marginal accuracy.

**It is the honest baseline for the other six plans.** If CatBoost or a blend cannot beat a
tuned LightGBM, that is a finding, and it is cheaper to learn it here.

## 4. Target — the Δ decision

### The case for Δ

**Embedded persistence baseline.** Define Δ as `PM2_5_next_hour` minus
`current_PM2_5`. Predicting zero reproduces persistence exactly on rows with a valid anchor,
so the baseline is the residual model's null output and every tree split can focus on change
rather than reconstructing the identity mapping. This is not a guaranteed performance floor:
a fitted model can predict a worse-than-zero residual and must still beat persistence out of
sample.

**It reduces sensitivity to the location shift into the test period.** **F8** says test is 18%
more polluted than the all-month training average. **F9** quantifies the seasonal component:
monthly mean level swings from 53.5 to 96.2, while monthly mean Δ stays flat between 0.01 and
0.21. A raw-level model still sees the current level and may adapt perfectly well, so this is
a hypothesis rather than proof that Δ must win. The matched A/B below decides it.

**The metric aligns.** **F1**: persistence RMSE of 19.67 *is* the standard deviation of Δ,
because mean Δ is 0.11 — essentially unbiased. Modelling Δ under L2 is therefore directly
modelling the quantity the score measures.

### The honest caveat

Δ fixes the *location* shift, not the *scale* shift. **F9** also shows Δ's standard deviation
swings from 13.7 in August to 26.8 in February — a 1.95× spread, almost identical to the
level's 1.8×. Δ remains strongly heteroskedastic across the year.

The mitigation is feature-side, not objective-side: backward rolling standard deviations tell
the model which regime it is in. They improve an L2 point forecast only when volatility is
associated with a change in the conditional mean or interacts with direction-driving features;
variance by itself is not predictable signed Δ. Do not change the objective merely to model
variance — RMSE is plain L2 and a different loss optimises something the leaderboard does not
score.

### Missing-anchor protocol — fixed before execution

The residual label and inverse transform both require the anchor:

`Δ = PM2_5_next_hour - current_PM2_5`

For the 2,466 training rows and 371 test rows where `current_PM2_5` is missing, Δ is undefined.
LightGBM's native NaN handling applies to input features, not labels, and cannot solve this.
Adding a predicted one-hour Δ to the last valid reading is also invalid: that reading is at
least two hours old, so the unobserved change between it and hour `t` would be omitted.

Use one deterministic production path:

1. Train the **residual model** only on real training rows with a non-missing anchor.
2. Train the **raw-level model** on every real training row; its target is always present.
3. If the prediction row has an anchor, return
   `max(0, current_PM2_5 + predicted_delta)`.
4. If the prediction row has no anchor, return `max(0, raw_level_prediction)`.

The raw arm is already required by the A/B, so this does not introduce another model in Plan
01. It becomes the retained fallback if Δ wins the gate. Never forward-fill an anchor and add
a one-hour residual to it.

For a complete-row persistence reference, use `current_PM2_5` where present. Where it is
missing, use the same-hour leave-one-station-out network median plus that station's historical
median offset from the network median. Fit the offset on the outer-training fold only. This is
a baseline fallback, not an input imputation: the model's `current_PM2_5` remains NaN.

### The A/B this plan runs

Two arms use identical features, folds, seeds and fixed hyperparameters, differing only in
target. Their fitting populations necessarily differ: the residual arm cannot use rows where
the anchor is missing, while the raw arm can.

Report two comparisons:

- **Target gate:** raw versus Δ on the identical anchor-present validation rows. This isolates
  the target representation decision.
- **Operational score:** raw on every validation row versus the residual/raw hybrid above on
  every validation row. This measures the actual submission path.

Both are scored in raw PM2.5 units. On anchor-present rows, report rapid falls (`Δ <= -30`), calm rows
(`abs(Δ) < 30`) and rapid rises (`Δ >= 30`) separately. **F10** shows falling events carry
45.6% of persistence squared error, more than the 31.4% carried by rises; a positive-spike-only
report misses the larger failure mode. Result goes to `Findings/`.

## 5. Data preparation specific to LightGBM

The shared layer produces one clean panel. LightGBM's adapter is the thinnest of any plan in
this folder — which is most of its appeal.

| Problem | Treatment for LightGBM | Rationale |
|---|---|---|
| **P1** missing anchor | Keep the feature NaN and add `anchor_is_missing` plus hours since the last valid anchor. Exclude the row only from residual-target fitting; retain it for raw-level fitting and validation. | LightGBM learns a missing-feature branch, but cannot accept the undefined residual label. At prediction time the raw-level arm handles this row. |
| **P2** timeline gaps | Handled upstream by the hourly grid. Padding rows excluded from fit and predict. | Without it, a "1-hour lag" is sometimes a five-hour lag. |
| **P3** PM2.5 > PM10 | **Do not repair.** Add the ratio as a feature and a boolean inconsistency flag. | Clipping either column corrupts the anchor, the most predictive input. The disagreement is plausibly informative about measurement conditions; let the splits decide. |
| **P4** `CO` quantisation | Pass through raw. Do not difference it finely. Add a flat-run counter. | With 131 distinct values 99% of which are multiples of 100, fine differences are quantisation noise. Trees split on thresholds and are indifferent to the coarse grid. |
| **P5** stuck sensors | Flat-run-length counters for `current_PM2_5`, `PM10` and `CO`. Do not delete the rows. | A 39-hour constant PM2.5 reading is a frozen instrument, and it is also exactly when the persistence anchor is stale. The flag lets the model distrust its own anchor. |
| **P6** blank `wd` | Fill with an explicit `(blank)` category, then declare `wd` categorical. | Blank rows average ~92 µg/m³ against a 78 mean — nowhere near any modal direction. Modal imputation would inject a systematic falsehood into 2% of test rows. |
| **P7** `year` / seasonality | **Drop raw `year`.** Replace with cyclical hour and day-of-year encodings, plus hours-since-series-start if a trend term is wanted. | Test years lie outside the training range and a tree cannot split past its last observed boundary. |
| **P8** `RAIN` zeros | Pass through raw. | 95.9% zeros with a long tail is a non-problem for threshold splits. Adding a rain indicator is optional and low value here. |
| **P9** heavy tail | **No winsorising, no log transform, no outlier removal.** | **F2**: the worst 1% of rows carry 47% of the squared error. Capping the target caps the achievable score. |
| Scaling | Not applied. | Trees are invariant to monotone rescaling. |
| One-hot | Not applied. | Native categorical support is better here than one-hot dilution. |

## 6. Feature representation

Grouped by the finding that justifies them.

- **Lags of `current_PM2_5`** at 1, 2, 3, 6, 12 and 24 hours (**F5** — level autocorrelation
  0.97 with a lag-24 echo). These carry the memory that matters.
- **Δ momentum**, one hour only (**F4** — Δ autocorrelation is 0.184 at lag 1 and collapses to
  zero by lag 2, with no daily echo). Deeper Δ lags are expected to be dead weight; include
  lag 2 once purely to confirm the EDA prediction, then drop it.
- **Backward rolling means and standard deviations** over 3, 6, 12 and 24 hours, accompanied
  by a non-missing observation count for each window. The standard deviations identify the
  current regime; counts prevent a sparse window from masquerading as a complete one.
- **Leave-one-station-out same-hour network levels**: mean, median, standard deviation,
  availability count, and this station's deviation from the other-station median. Excluding
  the focal station prevents its own anchor from dominating its network feature.
- **Leave-one-station-out network momentum**: mean and median across other stations of
  `current_PM2_5(t) - current_PM2_5(t-1)`. **F6** establishes shared movement, but its 0.30
  correlation is calculated from future target deltas and is not itself a usable feature.
  Current network momentum is causal and directly tests the claimed advance-warning signal.
  Missing station-hours remain NaN and the availability count tells LightGBM how much of the
  network was observed.
- **Meteorology**: wind speed, wind direction as a category plus its sine and cosine,
  dew-point depression, pressure, rain.
- **Calendar**: cyclical hour and day-of-year, day-of-week, and a heating-season indicator.
- **Data-quality flags**: the missingness indicators, gap length, flat-run counters and the
  PM inconsistency flag described in section 5.

Feature count should land around 70–90. Resist going wider before the first measurement;
**F7** says the marginal feature is weak, and a wide matrix makes the early-stopping bias in
section 9 worse.

## 7. Hyperparameter approach

Objective is plain L2, matching RMSE. Do not substitute MAE, Huber or a quantile loss — they
optimise a different quantity than the leaderboard scores. One Huber run is worth doing once
as a diagnostic of how strongly the tails drive the fit, then discarding.

Starting posture, to be tuned only after the feature set is settled:

| Knob | Starting point | Reasoning |
|---|---|---|
| Learning rate | Low (~0.02–0.05) with many rounds | Standard accuracy-for-time trade. Iteration speed here is cheap. |
| Number of leaves | Moderate (~63–255) | Wide enough for interactions; the fold-level early stopping constrains the rest. |
| Minimum data per leaf | **The contentious one.** Start moderate, tune deliberately. | High values smooth noise, which suits **F7**'s weak signal — but two-sided events are only 6.78% of rows (**F10**), and too high a floor can erase them into their neighbours. Tune against overall RMSE while inspecting rise/fall/calm diagnostics. |
| Feature and bagging fraction | ~0.7–0.9 | Decorrelates trees, and matters later for the blend in plan 07. |
| L2 regularisation | Non-zero | Lag features are near-collinear by construction (**F5**). |
| Categorical smoothing / min data per group | Start at LightGBM defaults | All declared categories have substantial support. Tune smoothing only if fold diagnostics show unstable category splits; do not add unmeasured regularisation to the A/B. |
| Categorical-to-one-hot threshold | Consider forcing one-hot | LightGBM's optimal categorical split is powerful and correspondingly prone to overfitting on small categories. Worth measuring both ways. |

Seed everything, including bagging, feature-fraction and data seeds, and record it. Use
LightGBM's deterministic mode with a fixed row/column histogram strategy. Treat early stopping
as training-loop control implemented with LightGBM callbacks; do not pass
`early_stopping_rounds` through as an unchecked booster parameter.

## 8. Validation

As specified in `plans/README.md`: expanding-window folds on timestamps, headline is the mean
RMSE over heating-season folds, every result quoted as a delta against that fold's own
persistence score. Define `delta_vs_persistence = model_rmse - persistence_rmse`, so a
negative number is an improvement.

Additionally required for this plan, since it sets the gate:

- Per-fold table and pooled out-of-fold RMSE, not just an unweighted mean.
- Rapid-fall, calm and rapid-rise RMSE, row count and squared-error share on anchor-present
  rows (**F10**); Δ-based regimes are undefined on missing-anchor rows.
- Both target arms scored in raw units on the common anchor-present population.
- Full-row operational scores for the raw arm, the residual/raw hybrid and complete-row
  persistence with its declared network fallback.
- A missing-anchor slice with row count and RMSE. In the heating folds there are only hundreds
  of such rows, so retain the raw per-row errors for inspection.
- RMSE by station, month, hour and current-concentration decile.

All target-independent causal features may be built across the raw panel before splitting.
Any learned statistic, including the station offsets used by the complete-row baseline, is fit
inside the outer-training fold only.

### Early stopping without optimistic outer-fold scores

For each outer fold, choose the number of boosting rounds on the most recent earlier complete
window with the same season as that outer validation window. That inner window must lie wholly
inside the outer-training period. Then refit on the entire outer-training fold for the selected
round count and score the untouched outer validation fold. Apply the same procedure separately
to the raw and residual arms; do not early-stop on the outer fold being reported.

### Uncertainty and the 0.05 rule

Compare the arms with paired per-row squared-error differences, not two unrelated scores.
Because errors and pollution events are serially clustered, calculate a moving-block bootstrap
confidence interval using seven-day timestamp blocks. Treat 0.05 RMSE as a practical effect-size
threshold, not proof of statistical significance. The gate should reproduce in direction on
both heating folds, with the bootstrap interval reported alongside it.

## 9. Success criteria

- **Must:** beat the comparable persistence score on both heating-season folds. There are only
  two such folds, so "a majority" would also mean both and is needlessly ambiguous.
- **Must:** no fold scores below ~8 RMSE. That indicates leakage, not success, and the run
  should be stopped and audited rather than celebrated.
- **Target:** an improvement over persistence exceeding 0.05 RMSE that has the same sign on
  both heating folds. Report the paired seven-day-block bootstrap interval; do not declare a
  sub-0.05 movement meaningful from a point estimate alone.
- **Diagnostic, not a second metric:** flag regressions in either rapid-rise or rapid-fall
  RMSE. The competition decision is still made by overall RMSE; a tail regression can coexist
  with a genuine overall RMSE improvement and must be described accurately rather than called
  mathematically worse.
- **Gate output:** a clear Δ-versus-level verdict written to `Findings/`.

## 10. Risks and failure modes

**Early-stopping bias.** Choosing the number of rounds on the same fold that is then reported
inflates the score. Either fix the round count in advance from a preliminary fold, or nest the
early-stopping split inside the training fold. This is the most likely source of a CV number
that fails to transfer, and it is easy to do accidentally.

**Missing-anchor routing.** Accidentally fitting an undefined residual label, adding a
one-hour residual to a stale multi-hour anchor, or scoring only the easy anchor-present rows
invalidates the experiment. The explicit raw-level fallback and dual-population reporting in
sections 4 and 8 are mandatory.

**Categorical overfit on `station`.** Twelve levels over 360k rows is not obviously dangerous,
but combined with `wd` the interaction space is large and the optimal-split algorithm is
aggressive.

**Smoothing away the tails.** Rapid rises and falls together carry 77.0% of persistence
squared error (**F10**), with falls carrying the larger share. Inspect both sides whenever a
regularisation change moves the headline.

**Feature-count creep.** Adding features until the CV moves is how you overfit the folds. Each
addition should be justified by a finding and measured on its own.

## 11. What this plan unblocks

Plans 02 through 06 are blocked on the Δ verdict. Plan 07 additionally needs this model's
out-of-fold predictions retained in a stable format, so store them from the first run rather
than regenerating later.

## 12. Effort estimate

The shared preparation layer dominates. Once it exists, both arms of this plan are roughly a
day including the validation harness and missing-anchor routing. The A/B itself is cheap — the
same feature panel fitted twice with a different target specification.

## 13. Execution contract

Plan 01 is ready to run only when all of the following are true:

1. The shared hourly-grid and causal feature layer is implemented and tested.
2. `configs/lgbm_v1.yaml` runs the residual arm with anchor-present fit rows and raw-level
   fallback; `configs/lgbm_v1_level.yaml` runs the raw arm.
3. The training adapter passes native pandas categorical columns to LightGBM with identical
   category domains across each fold, and uses callbacks for early stopping.
4. The validation report contains common-population target-gate metrics, full-row operational
   metrics, the three signed-change regimes, and the missing-anchor slice.
5. Predictions are finite for every required row and lower-bounded at zero. There is no upper
   clipping, target winsorisation or log-target inverse transform.
6. The leakage audit passes and out-of-fold predictions for both arms are stored with `id`,
   timestamp, station, target, anchor-presence flag and fold.
