# Methodology Report — Team overfitters
## Next-Hour PM2.5 Forecasting

**Final leaderboard submission:** `v3_lgbm_xgb_cat_ratio_800x3.csv` — an equal-weight blend of
LightGBM, XGBoost and CatBoost, each predicting the PM2.5 fine fraction (PM2.5 ÷ PM10) rather
than the raw level, trained three times with different seeds and averaged.

---

## 1. Problem statement

For each labelled station-hour across 12 urban air-quality monitoring stations, predict
`PM2_5_next_hour` — the PM2.5 concentration at the same station exactly one hour after the
observation timestamp. This is a supervised tabular regression problem over a panel
(multi-series) time structure, not a multi-step forecast: the horizon is fixed at +1 hour and
every predictor is observed at hour `t`. Scored by RMSE (lower is better) on a hidden,
chronologically later test set.

RMSE weights large errors quadratically, so the leaderboard is decided by the tail — rapid
pollution-episode onsets — not by the many calm hours. This shaped two decisions described
below: we did not use a log-target (it under-predicts spikes) and we did not clip predictions
to a low ceiling.

### A mid-project change to the problem

The competition data was provided in two forms during this project:

- An **earlier version** (`Data/train.csv`, `Data/test.csv`) that included a `current_PM2_5`
  column — the PM2.5 reading at hour `t` itself. That column alone explained ~94% of the
  variance in the target (r ≈ 0.968) purely through hour-to-hour persistence.
- The **final version** used for this submission (`new data/train_new.csv`,
  `new data/test_new.csv`) — verified byte-for-byte identical to the earlier files except that
  `current_PM2_5` has been removed from both. Every other column, row, id and target value is
  unchanged.

This is a deliberate difficulty change by the organisers: it forces the model to estimate
PM2.5 from correlated-but-imperfect signals (the other five pollutants, weather, time of year,
neighbouring stations) instead of leaning on a near-perfect anchor. **This report and the final
submission are built entirely on the final, anchor-free version.** The removed column, and
anything computed from it, is never used as a model input (see §2.3 and §7 for how this is
enforced and verified).

---

## 2. Data cleaning and preprocessing

### 2.1 Inputs

Only two files are read: `new data/train_new.csv` (360,954 rows) and `new data/test_new.csv`
(51,063 rows), covering 12 stations at hourly resolution. No external data, weather reanalysis,
public air-quality archives, or satellite products were used at any point — verified by design
(the feature code only ever opens these two files) and by inspection.

### 2.2 Complete hourly grid

Train and test are concatenated on one timeline (tagged `is_test`) and each station is
reindexed to a **complete, gap-free hourly grid** from its first to its last real observation.
Missing hours become explicit padded rows (`is_pad=True`) rather than being silently skipped,
so that lag and rolling-window features have a correct notion of elapsed time and do not
silently bridge a multi-hour sensor outage as if it were one hour. Padded rows are dropped
again before scoring or training; the real row count is unchanged (360,954 train + 51,063 test
= 412,017 real rows out of 420,756 grid rows).

### 2.3 What the deleted column means for preprocessing

Because `current_PM2_5` is absent from both files (not just held out), any code path that could
read it, impute it, or reconstruct it as a usable feature is a leakage risk. Two specific traps
were identified and guarded against:

1. **The test-set self-join.** Test is contiguous hourly data, so for most test rows the true
   answer is the target of the *next* test row at the same station. This is never computed or
   used.
2. **Chained reconstruction.** For training rows only, `PM2_5_next_hour(t) == old
   current_PM2_5(t+1)` for 100% of non-gap rows, meaning the entire historical PM2.5 series
   could technically be rebuilt for training by chaining the target backwards one row at a
   time. This reconstruction cannot be extended past the first test row per station without
   substituting the model's own predictions for missing ground truth (a recursive/autoregressive
   forecast). That approach was explicitly identified as a large but risky lever and was **left
   out of the final submission by design** — it was gated behind explicit sign-off that was
   never given, precisely because one bad early prediction would compound forward through
   roughly six months of hourly test data per station, and because it reintroduces the PM2.5
   dependency the harder problem was designed to remove.

A helper column, `diag_pm25_now` (the previous row's own target, shifted forward by one row),
is computed **only** for local scoring diagnostics — to check a trained model's error against
where the old, easier persistence baseline would have scored. It is hard-refused from ever
reaching a feature list by an assertion (`assert_no_pm25`) that scans every feature-set name for
`PM2_5`, `pm25`, `PM25`, or `diag` substrings.

### 2.4 Missing data

No blanket imputation. Gradient-boosted trees handle `NaN` natively and were preferred over
imputing, since missingness itself carries information here (e.g. a station's PM10 sensor
being offline changes which fallback features apply). Two exceptions, both explicit and
auditable:
- `PM10_filled`: PM10 with a same-hour, leave-one-out cross-station mean substituted only where
  the station's own PM10 is missing — used solely as the denominator for the ratio target
  (§4), not as a general-purpose feature.
- Explicit `*_missing` indicator flags (PM10, SO2, NO2, CO, O3, weather columns, wind
  direction) were engineered as separate binary features rather than silently filled, so the
  tree can use missingness as a signal in its own right.
- Blank wind direction is treated as its own category (`"UNKNOWN"`), never coerced to the modal
  direction — it is nine times more common in test than in train, so treating it as noise would
  bias the model.

---

## 3. Feature engineering

Every feature is built causally: only `shift(k>0)`, trailing (`closed='left'` / shift-then-roll)
windows, and same-hour cross-station aggregates. Nothing looks forward in time. This was
enforced by a leakage audit (`src/pm25/audit.py`, tokenises source files and flags negative
shifts, centred windows, and backfills) and a **causality test**
(`scripts/v3/test_causality.py`): the entire feature panel is rebuilt twice — once on the full
timeline, once truncated at an earlier cutoff — and every one of 292 feature columns (plus the
diagnostic anchor) is asserted numerically identical wherever both versions have data. Any
feature that had read the future would differ and fail this test; none did.

Feature groups were added incrementally and measured one group at a time against the
time-aware validation folds (§5), in this order:

| step | group added | effect on heading RMSE |
|---|---|---:|
| f0 | raw pollutant/weather columns, calendar, wind, quality flags | baseline 30.80 |
| f1 | pollutant lags (1–24h) | −0.58 |
| f2 | rolling mean/std/min/max (3/6/12/24h) | +0.30 (hurts alone; kept later, see below) |
| f3 | first-derivative motion (hour-over-hour change) | −1.10 |
| f4 | second-derivative motion (acceleration) | +0.43 — **dropped**, too noisy |
| f5 | same-hour cross-station levels (PM10/CO/etc. network mean/median/std/max) | **−1.59, the single biggest lever of the build-up** |
| f6 | cross-station momentum | −0.12 |
| f7 | per-neighbour station columns | −0.23 |
| f8 | extra weather lags/meteorology | +0.13 — **dropped**, dead weight |
| f9 (deep) | fine-fraction physics: PM10 crossed with each combustion tracer (NO2, SO2, O3, temperature, dew point), city-level counterparts, season/diurnal interactions | see §4 — only pays off once the target itself changes |

A backward ablation from the f7 set confirmed that acceleration and the missing/quality flags
(12 zero-gain columns) could be removed with no cost, and that rolling statistics — which hurt
when added in isolation — became a small net positive once the network-level features existed,
i.e. **a feature's value depends on what else the model can already see**, so ablation was done
from the full set rather than trusting build-order alone. A pairwise redundancy scan found 119
column pairs with |r| > 0.95 among 282 numeric columns (documented, not removed — dropping them
costs trees ~0 to +0.1 RMSE and was judged not worth the added engineering risk this close to
deadline).

**Where the signal actually lives** (gain share on the f7 set): `PM10` alone accounts for 36% of
total split gain, `PM10 × CO` for 26%, and the city-wide `net_PM10_median` for 13% — three
columns carry three-quarters of the model's discriminative power. The model is, in essence,
*"PM10, corrected by combustion intensity and by what the rest of the city looks like right
now."* The final feature set (`f9_deep`, 284 columns) adds explicit versions of that correction:
PM10 × NO2/SO2/O3/dew-point/temperature, log-log and per-particle combustion ratios, city-level
counterparts of those same products, and PM10 × season/hour-of-day interactions.

A further round of feature ideas (fine-fraction climatology encoding, humidity via the Magnus
formula, episode-age/stagnation duration, coarse-vs-combustion balance, 72h/168h long memory,
and a learned wind-direction transport adjacency) was explored after this submission was
generated. Two of those ideas combined to a further, later-confirmed improvement over the
model reported here; see §6 "Limitations and further work" — that later result was not used for
this leaderboard submission and is out of scope for this package, which documents the pipeline
that actually produced `v3_lgbm_xgb_cat_ratio_800x3.csv`.

---

## 4. The single biggest modelling decision: predicting a ratio, not a level

Rather than training the model to output `PM2_5_next_hour` directly (the "level" target), the
final model predicts the **fine fraction**, `y ÷ PM10`, with each row weighted by `PM10²` in
the training loss. Because squared error on `(y/PM10 − ŷ/PM10) × PM10` is algebraically the
same quantity as squared error on `(y − ŷ)` when weighted by `PM10²`, this is a
re-parameterisation of the same L2 objective, not a different loss function — but it changes
what each tree leaf represents: a leaf now holds a *fraction that scales with PM10*, so the
model's output is no longer capped by the highest PM2.5 level it saw during training. This
matters specifically because the test period (September–February) is materially more polluted
than the training average (mean PM10-linked target 92.3 in test vs. 78.0 in train) — a level
model has no way to extrapolate past its training range, while a ratio model can.

Measured effect, same features, same fold, same everything else changed:

| features | target | heading RMSE | vs. level target |
|---|---|---:|---:|
| f7 | level | 27.90 | — |
| f7 | ratio | 26.26 | **−1.64** |
| f9 (deep) | level | 28.03 | — |
| f9 (deep) | **ratio** | **25.94** | **−2.09** |

This was, by a wide margin, the largest single lever found in this phase of the project — larger
than any feature group, and larger than any difference between algorithms (§5). The gain was
concentrated exactly where it was needed: fold 2 (the hardest, most extreme winter in
validation) improved from 29.9 to 27.0. Two weaker alternatives were also tested and rejected:
an additive proxy (`y − k_station · PM10`, −0.3 only) and `log1p(y)` (+2.0 — it optimises
relative error and under-predicts exactly the spikes RMSE penalises, confirming the concern
raised in the project's own risk notes before any number was measured).

---

## 5. Validation strategy

**Expanding-window, time-aware folds** (`src/pm25/validation.py`), never a random split — a
random K-fold on hourly panel data leaks near-identical adjacent hours between train and
validation and would report a score the leaderboard will not honour. Four folds, each trained
on all data up to a cutoff and validated on the following six-month window:

| fold | train ends | validates on | season |
|---|---|---|---|
| 0 | 2014-08-31 | 2014-09-01 – 2015-02-28 | heating |
| 1 | 2015-02-28 | 2015-03-01 – 2015-08-31 | summer |
| 2 | 2015-08-31 | 2015-09-01 – 2016-02-29 | heating |
| 3 | 2016-02-29 | 2016-03-01 – 2016-08-31 | summer |

**Headline metric = mean RMSE over the two heating-season folds (0 and 2)**, because the actual
hidden test period is September–February and is materially more polluted than the training
average — the summer folds are reported for completeness but do not decide model choices.
Within each outer fold, an inner previous-season window provides early-stopping validation
without touching the true validation fold.

Every fitted statistic (target encodings, the fine-fraction denominator's cross-station fill,
per-station medians used in proxy targets) is fit on the training portion of each fold only,
never on validation or test rows, and never on the concatenated full dataset before splitting.

---

## 6. Models tested and final model selection

Boosted-tree algorithms were compared on the identical feature set (f9_deep, 284 columns) and
identical ratio target:

| model | fold 0 | fold 1 | fold 2 | fold 3 | **headline** | notes |
|---|---:|---:|---:|---:|---:|---|
| **LightGBM**, seed 42 | 24.90 | 19.80 | 26.99 | 18.95 | **25.94** | best single model |
| **XGBoost**, tuned | 25.11 | 20.63 | **26.87** | 19.06 | **25.99** | see below — required a specific fix |
| **CatBoost** | 25.67 | 21.81 | 27.60 | 19.16 | 26.64 | ~0.7 behind, kept as a blend member for its lower residual correlation |
| HistGB | 25.24 | 19.82 | 28.34 | 18.86 | 26.79 | not used further |
| Ridge (regularisation path to α=10,000) | 33.22 | 26.59 | 30.43 | 23.10 | 31.82 | linear model unsuited to a non-linear fine-fraction target |

**XGBoost required one specific fix to be competitive.** Its `min_child_weight` parameter is a
sum of per-leaf *Hessians*, not a row count. Under `PM10²` sample weights (which range from
~10² to ~10⁶ across rows), the library default of 50 constrained almost nothing, so XGBoost grew
unregularised, overly deep trees and initially scored 29.64 — 3.7 points worse than LightGBM,
worst specifically on calm hours. Scaling `min_child_weight` to ~1,000,000 (matching the
sample-weight magnitude) and reducing max depth from 8 to 7 closed almost the entire gap
(29.64 → 25.99) and made XGBoost the single best model on fold 2, the hardest validation
window. This generalises: **any tree library with a Hessian-based leaf-size constraint needs
that constraint re-scaled to match the sample weights** whenever a weighted objective like the
ratio target is used.

**Final model: an equal-weight blend of all three tuned tree models**, not a single algorithm.
Justification, computed from out-of-fold predictions on the heating folds:

- Residual correlation between the three models is high but not 1: LightGBM–XGBoost 0.969,
  LightGBM–CatBoost 0.932, XGBoost–CatBoost 0.923 — enough independent error to be worth
  combining.
- A simple **equal-weight average** of the three models' out-of-fold predictions scores
  **24.56 on fold 0 and 26.77 on fold 2 — a heading RMSE of 25.66**, better than any single
  model (best single: 25.94).
- An NNLS-fit blend (weights optimised on one heating fold, evaluated on the other) gave
  weights in the region of 0.18–0.46 per model depending on which fold it was fit on — too
  fold-dependent to trust over the deadline window, so the **simpler, fold-stable equal-weight
  average was adopted** for the submission rather than a fitted weight vector.

### Final hyperparameters

| model | key parameters | rounds | seeds |
|---|---|---:|---|
| LightGBM | `objective=regression`, `learning_rate=0.05`, `num_leaves=63`, `min_data_in_leaf=100`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=1`, `lambda_l2=1.0` | 800 | 42, 1, 2 |
| XGBoost | `objective=reg:squarederror`, `tree_method=hist`, `learning_rate=0.05`, `max_depth=7`, `min_child_weight=1,000,000`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0` | 800 | 42, 1, 2 |
| CatBoost | `loss_function=RMSE`, `learning_rate=0.08`, `depth=8`, `l2_leaf_reg=3.0`, `border_count=254` | 800 | 42, 1, 2 |

Every model is trained three times with different seeds on the ratio target with `PM10²`
sample weights; the three seed-runs per model are averaged first, then the three models'
seed-averaged predictions are averaged with equal weight (1/3 each) to produce the final
prediction, which is then multiplied back by the same-hour-filled PM10 to return to the raw
PM2.5 scale.

---

## 7. Ensembling and post-processing

- **Ensembling**: equal-weight (1/3, 1/3, 1/3) average of LightGBM, XGBoost and CatBoost
  predictions, each itself a 3-seed average, as justified in §6.
- **Post-processing**: predictions are converted from the fine-fraction scale back to the raw
  PM2.5 scale by multiplying by the same-hour PM10 (with a same-hour cross-station fill where a
  station's own PM10 reading is missing), then clipped at zero (`max(pred, 0)`) since PM2.5
  cannot be negative. No other post-processing (no smoothing, no manual adjustment, no
  leaderboard-informed correction) is applied.
- Before the file is written, the pipeline re-runs the leakage audit and asserts: every row's
  id is unique, id count matches the known test-set size exactly, no `NaN` or negative values
  remain, and the output row order matches the original `test_new.csv` id order exactly.

---

## 8. Key results, observations, and limitations

**Result.** Best single model (LightGBM, f9_deep, ratio target): heading RMSE **25.94**.
Equal-weight three-model blend, out-of-fold: heading RMSE **25.66**. Both are measured against
the harder, anchor-free version of the problem; the deleted `current_PM2_5` anchor's own
persistence baseline would score **21.3** on the same folds for reference — the honest gap
between "the easy version of this problem" and where this model landed.

**Observations.**
- The gain from feature engineering (30.80 → 27.90, level target) was smaller than the gain
  from choosing the right prediction target (level → ratio, a further −2.1 on the same
  features). *What* the model is asked to predict mattered more than what it was allowed to
  see.
- Boosted-tree algorithms plateaued within ~0.1–0.7 RMSE of each other once the feature set and
  target were fixed — the remaining gains came from ensembling distinct algorithms, not from
  further algorithm search.
- Errors are not uniform: a residual audit (out-of-fold, no retraining) found a 1.4× RMSE
  spread across the 12 stations with systematic (not random) per-station bias of up to ±4.4,
  the worst hours are overnight (00:00–02:00, RMSE 30–36 vs. ~22 in the best hours), and the
  worst months are November–December. Rapid pollution-episode onsets remain the hardest regime
  under RMSE by construction (they are the tail RMSE punishes most).

**Limitations.**
- The model does not reach the round's aspirational target of RMSE 17.0 — reaching that would
  require beating a baseline (21.3) that had access to a column (`current_PM2_5`) this version
  of the problem does not provide.
- Round-count selection under the ratio target is fold-dependent in a way that is not fully
  resolved: fold 0 keeps improving to ~1,500 rounds while fold 2 (the most extreme winter)
  peaks near 100 rounds and degrades after — the 800-round setting used for the final
  submission is a compromise, not the individually optimal count for either regime.
- A recursive/autoregressive reconstruction of the deleted PM2.5 signal from the model's own
  chained predictions was identified as the single largest remaining lever but was deliberately
  **not used** in this submission (§2.3) — it was assessed as high-risk (compounding error over
  ~6 months per station) and was gated behind an explicit decision that was not made in time for
  this deadline.
- A subsequent, brief round of feature exploration (documented separately, not part of this
  package) found that combining an episode-duration/stagnation feature group with a
  fine-fraction climatology encoding improved the single LightGBM model to a heading RMSE of
  25.77 across all four folds, improving or holding flat on every fold. That result was
  produced after this leaderboard submission was generated and is not part of the pipeline
  described here.
