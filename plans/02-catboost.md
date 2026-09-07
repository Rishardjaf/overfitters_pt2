# Plan 02 — CatBoost

**Role:** the model most likely to win outright on this data shape, and the strongest handler
of the `station` × `wd` categorical structure.

**Status:** not run. **Blocked on the Δ gate in plan 01.**

---

## 1. Purpose

Test whether CatBoost's ordered target statistics and ordered boosting extract more from the
categorical structure than LightGBM's split-based categorical handling, and whether its
regularisation posture holds up better on the rare regimes that carry the score.

## 2. Why CatBoost is worth a plan of its own

**Ordered target statistics.** CatBoost encodes categorical values using target statistics
computed over a prefix of the data rather than the whole set. On this dataset the categorical
structure is not decoration: `station` (12 levels) carries genuine per-site behaviour —
**P3** violation rates range from 3.88% at Aotizhongxin to 9.66% at Dingling, which is
per-instrument character, not noise — and `wd` (17 levels with blank) interacts with it,
because which direction is dirty depends on where a station sits relative to the sources.
LightGBM must discover that interaction through splits. CatBoost encodes it directly.

**Ordered boosting resists the overfit this data invites.** **F7** says the signal is weak, so
the gap between fitting signal and fitting noise is narrow. **F3** says 3.04% of rows carry
31.4% of the squared error, so the model is under pressure to chase a small number of extreme
rows. Ordered boosting exists precisely to reduce the target leakage that drives that kind of
overfitting.

**Empirical prior.** On tabular panels with meaningful categoricals and a few hundred thousand
rows, CatBoost is frequently the strongest single model with the least tuning. That is a prior,
not a finding, and this plan exists to test it.

## 3. Target

**Open — gated on plan 01.** See "The Δ decision gate" in `plans/README.md`.

If Δ is adopted, one CatBoost-specific note applies: the Δ target is near-symmetric around
zero with heavy two-sided tails (**F1**), and CatBoost's default handling is unremarkable for
that shape. No special treatment expected. The anchor-reconstruction edge case from plan 01
section 4 applies identically and should reuse whatever handling plan 01 settles on rather
than inventing a second one.

## 4. Data preparation specific to CatBoost

Two hard requirements distinguish this adapter from LightGBM's. Both were verified against the
installed version rather than assumed.

**Categorical columns must not contain missing values.** CatBoost raises an error on a NaN
inside a declared categorical feature — it does not silently handle it. So for **P6**, filling
blank `wd` with an explicit `(blank)` category is not a modelling preference here, it is a
precondition for the model running at all. Convenient, since it is the correct treatment
regardless: blank rows average ~92 µg/m³ against a 78 mean, so they are a real category and
never a modal-imputation candidate.

**Categorical values must be strings or integers, not floats.** A silent source of type errors
if `station` or any flag arrives as a float from an upstream merge.

**Rows must be sorted chronologically and the model told that they are.** CatBoost's ordered
statistics use random permutations of the data by default. On a time series that is wrong in a
specific and subtle way: it lets target information from later hours inform the encoding of
earlier ones. CatBoost provides a setting that makes it respect existing row order instead,
and it must be enabled. **This is the single highest-risk detail in this plan** — it produces
no error, no warning, and a validation score that is quietly too good. If a CatBoost fold ever
scores conspicuously better than the equivalent LightGBM fold, check this before believing it.

| Problem | Treatment for CatBoost | Difference from plan 01 |
|---|---|---|
| **P1** missing anchor | Leave the NaN, but the **explicit missing-anchor flag matters more here**. | CatBoost substitutes missing numeric values with a sentinel below or above all observed values rather than learning a per-split default direction. That is a blunter mechanism, so the model gets less from the NaN itself and more from the explicit flag. |
| **P6** blank `wd` | Fill with `(blank)`. **Mandatory**, not optional. | LightGBM tolerates a missing categorical; CatBoost errors. |
| Row order | Chronological, with ordered-statistics mode enabled. | LightGBM is order-independent. This requirement does not exist in plan 01. |
| Categorical dtype | String or integer. | LightGBM accepts a pandas categorical directly. |
| **P3**, **P4**, **P5**, **P7**, **P8**, **P9** | Identical to plan 01. | No difference. |
| Scaling, one-hot | Not applied. | Same as plan 01. CatBoost one-hots small-cardinality features internally up to a configurable size; worth testing whether `station` and `wd` fall above or below that threshold. |

## 5. Feature representation

Identical to plan 01 section 6. Deliberately so — the comparison is only meaningful if the
feature set is held constant.

One CatBoost-specific option worth one experiment: it can construct feature combinations of
categorical variables automatically. Given the `station` × `wd` interaction argued for in
section 2, allowing a greater combination depth is the most plausible place CatBoost pulls
ahead. Test it as a single deliberate change, not bundled with other tuning.

## 6. Hyperparameter approach

L2 objective, matching RMSE, as everywhere in this folder.

The knobs that matter differ from LightGBM's. CatBoost grows symmetric (oblivious) trees by
default, so depth is a much stronger lever than it is in a leaf-wise learner and the usual
leaf-count intuitions do not transfer. Start moderate on depth and tune it before anything
else. The L2 leaf regularisation and the categorical-combination depth are the next two.

Expect training to be materially slower than LightGBM at equivalent quality. Budget for that:
if the tuning loop is too slow to run a dozen configurations, this plan will lose to plan 01
on iteration count regardless of its per-model merit.

## 7. Validation

As specified in `plans/README.md`. Because this plan's main claim is about categorical
handling, additionally report the per-station RMSE breakdown, not only the per-fold and
spike-vs-calm ones. If CatBoost's advantage is real, it should appear as a narrowing of the
spread across stations — particularly at the stations with the most anomalous **P3** rates.

## 8. Success criteria

- Beats persistence on a majority of heating-season folds. Non-negotiable.
- Beats plan 01's LightGBM by more than 0.05 RMSE, reproducing across folds, to justify being
  a submission candidate on its own.
- Fails to beat LightGBM but produces **decorrelated errors** — still a success, in which case
  it graduates to plan 07 as a blend member rather than a standalone.
- Per-station spread narrows relative to plan 01, supporting the categorical hypothesis.

## 9. Risks and failure modes

**Silent time leakage through unordered target statistics** (section 4). Highest-severity risk
in this plan because it fails invisibly and in the flattering direction.

**Slower iteration.** The realistic failure mode in a hackathon is not that CatBoost scores
worse but that it consumes the time that plan 07 needed.

**Ordered boosting's variance cost.** It reduces overfitting at the cost of using less data
per estimate. With **F7**'s weak signal that trade may go the wrong way. Measure rather than
assume.

**Categorical combinations exploding.** Raising combination depth expands the search space
quickly and interacts badly with 3% spike rows. Change one thing at a time.

## 10. Effort estimate

Low, if plan 01 has built the shared layer and the validation harness. The adapter is a fill,
a dtype cast, a sort and one configuration flag. Most of the cost is training time, not
engineering time.
