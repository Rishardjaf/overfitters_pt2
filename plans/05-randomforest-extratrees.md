# Plan 05 — RandomForest / ExtraTrees

**Role:** a **bagged** ensemble as a counterweight to three boosted ones. Variance reduction
rather than bias reduction, and therefore a different error profile.

**Status:** not run. **Blocked on the Δ gate in plan 01.**

---

## 1. Purpose

Every other tree plan in this folder boosts: each tree corrects its predecessors' residuals,
which drives bias down and leaves the ensemble able to chase noise. A random forest averages
independently grown trees instead, driving variance down while leaving bias largely intact.

Those are opposite failure modes. That is the entire justification for this plan, and it is a
blend argument, not a standalone one.

## 2. Expected outcome, stated up front

**This model will underperform the boosted plans on the metric that matters, and the reason is
specific rather than generic.**

Averaging shrinks predictions toward the mean of the leaf population. **F3** says spikes are
3.04% of rows carrying 31.4% of the squared error, and **F2** says the worst 1% of rows carry
47%. A rare extreme value averaged across hundreds of trees — most of which saw leaves
dominated by ordinary hours — comes out substantially attenuated. Shrinkage toward the mean is
precisely the wrong behaviour under a squared-error metric whose mass sits in the tails.

Write this prediction down and check it against the spike-vs-calm breakdown. If the forest does
*not* attenuate the spikes, the reasoning above is wrong and that is worth knowing.

## 3. Target

**Open — gated on plan 01.** One property specific to bagging is worth noting, and it is
favourable:

If the Δ target is adopted, over-smoothing degrades gracefully. Mean Δ is 0.11 (**F1**), so a
forest that shrinks its predictions toward the population mean converges toward predicting
approximately zero change — which *is* persistence. The failure mode of this model on the Δ
target is therefore "reverts to the baseline", not "goes somewhere wrong".

On the raw level target there is no such safety property: shrinkage pulls predictions toward
the global mean of roughly 78 µg/m³, which is catastrophic on a polluted hour. **If plan 01
returns a marginal Δ verdict, this plan is a reason to favour Δ anyway.**

## 4. Data preparation specific to RandomForest / ExtraTrees

**Native missing-value support, but no categorical support.** That combination is unique in
this folder and defines the adapter.

**Missing values are handled natively.** sklearn's forests gained this relatively recently;
verified working on the installed version (1.9.0). So **P1** needs no imputation, and the
missing-anchor NaN can be left in place exactly as in plan 01. Retain the explicit indicator
regardless.

**Categoricals must be encoded manually.** There is no native support, so a decision is
required for `station` and `wd`:

- **`station` — ordinal encode.** The integer codes carry no real ordering, but a tree can
  carve any subset out of an ordinal axis given enough splits, and 12 levels is small enough
  that it will. One column instead of twelve.
- **`wd` — one-hot.** Seventeen levels including `(blank)` (**P6**). Ordinal encoding here is
  worse than for `station` because the compass ordering is *nearly* meaningful — adjacent codes
  are adjacent directions — which tempts the model into treating a wrap-around boundary as a
  real discontinuity. One-hot avoids inventing that. Pair it with the sine and cosine encoding
  so genuine directional continuity is still available.

This asymmetry is deliberate and should be defended in review rather than smoothed over.

**One-hot dilution is a real cost here.** Random forests sample a subset of features at each
split. Every one-hot column added lowers the probability that any *particular* informative
feature is offered at a given split. With roughly 17 additional columns against a base of
70–90, the dilution is meaningful and argues for raising the per-split feature sample fraction
above the regression default.

| Problem | Treatment | Contrast with plan 01 |
|---|---|---|
| **P1** missing anchor | Leave the NaN, keep the indicator. | Same as plan 01. |
| **P3**, **P4**, **P5**, **P7**, **P8**, **P9** | Identical to plan 01. | No difference. |
| **P6** blank `wd` | `(blank)` as a one-hot column, plus sine/cosine. | Plan 01 uses a native category; no expansion. |
| `station` | Ordinal encoded. | Plan 01 uses a native category. |
| Scaling | Not applied. | Same as plan 01. |
| Feature sample fraction | Raise above default to offset one-hot dilution. | No equivalent concern in plan 01. |

## 5. Model choice within the family

**RandomForest first.** The standard bagged baseline and the more predictable of the two.

**ExtraTrees second, and it may well be the better member.** It selects split thresholds at
random rather than optimally, which adds variance to each tree and decorrelates them further.
Two consequences here: greater decorrelation is exactly what plan 07 wants, and the training
cost drops substantially because no threshold search is performed. The cost is even more
shrinkage, which section 2 already identifies as this plan's weakness.

Run both. They are cheap once the adapter exists, and the comparison is informative about how
much of any gain is decorrelation rather than fit.

## 6. Resource constraints — a real consideration here

Unlike the boosted plans, memory is a genuine limit. Fully grown trees on 360k rows, replicated
across several hundred estimators, are large. Impose a minimum-samples-per-leaf floor and a
depth cap from the start rather than discovering the limit by exhausting memory.

The minimum-leaf floor carries the same tension flagged in plan 01: it smooths noise, which
suits **F7**'s weak signal, but it also erases the 3% of rows that carry the score. Tune it
against the spike-vs-calm breakdown, not the headline.

## 7. Validation

As specified in `plans/README.md`, plus:

- **Spike-vs-calm breakdown is the primary diagnostic for this plan**, not a secondary check.
  It is the direct test of the section 2 prediction.
- Out-of-fold residual correlation against plan 01, as in plans 03 and 04.

Do **not** use out-of-bag scoring as a substitute for the time-aware folds. Out-of-bag samples
are drawn randomly across the whole training period, so an OOB estimate is a random-split score
wearing a different name, and it will leak adjacent hours exactly as a random KFold would. It
may be reported as a curiosity; it is not a validation number.

## 8. Success criteria

- Beats persistence on the heating-season folds.
- Residuals meaningfully decorrelated from plan 01's — expected to be the *least* correlated of
  the four tree plans, since it is the only bagged one.
- Improves plan 07's blend.
- Confirms or refutes the section 2 shrinkage prediction on the spike breakdown. Either result
  is a useful finding.

## 9. Risks and failure modes

**Spike attenuation** (section 2). Predicted, not merely possible.

**Accidental random-split validation via out-of-bag scoring** (section 7). Easy to reach for
because sklearn offers it for free, and it silently violates the project's core validation rule.

**Memory exhaustion** on an unconstrained fit.

**Opportunity cost.** Alongside plan 03, the most cuttable plan in the folder. If time is
short, keep plan 03 over this one — a decorrelated *boosted* model is a safer blend member than
a bagged one that attenuates the rows carrying the score.

## 10. Effort estimate

Low. The adapter is an encoding decision and two resource caps. Training time is the main cost,
and ExtraTrees substantially reduces it.
