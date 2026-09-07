# Plan 03 — XGBoost

**Role:** third boosted-tree implementation, included primarily for **error decorrelation in
the blend**, not because it is expected to win outright.

**Status:** not run. **Blocked on the Δ gate in plan 01.**

---

## 1. Purpose

Provide a third tree ensemble whose errors are structurally different from LightGBM's and
CatBoost's, so that plan 07 has something to gain from averaging.

This plan should be judged on whether it improves the blend, not on whether it beats plan 01.
If it is scored only on standalone RMSE it will probably look redundant and be cut for the
wrong reason.

## 2. Why include it at all

**Different growth policy, different bias.** XGBoost grows depth-wise by default while
LightGBM grows leaf-wise. On the same features and the same folds that produces genuinely
different partitions of the space, and therefore errors that are correlated with LightGBM's
but not identically. That is the entire argument, and for plan 07 it is sufficient.

**Different missing-value mechanism.** XGBoost learns a default branch direction per split for
missing values. Given **P1** is structural and affects rows that must be predicted, having two
models that disagree about how to route those rows is useful diversity rather than redundancy.

**Low marginal cost.** The adapter is nearly identical to plan 01's. This is the cheapest
additional model in the folder.

## 3. Target

**Open — gated on plan 01.** See `plans/README.md`. Whatever plan 01 concludes, this plan
follows without further argument; its purpose is diversity within a fixed setup, so deviating
on target would defeat the comparison.

## 4. Data preparation specific to XGBoost

Nearly identical to plan 01's adapter. Three differences, one of which is a real trap.

**Categorical support requires two settings together.** XGBoost handles pandas categorical
columns natively only when categorical support is explicitly enabled *and* the histogram tree
method is selected. Enabling one without the other fails or silently falls back. Verified
working on the installed version (3.4.1).

**The sparse-matrix trap.** If `station` and `wd` are one-hot encoded into a scipy sparse
matrix instead of passed as native categoricals, XGBoost treats structural zeros in that
matrix as *missing* by default. Every "this row is not station Dongsi" becomes "unknown
station" — a silent corruption of the entire categorical block, producing a model that trains
without error and scores badly for reasons that are hard to trace. Either use native
categorical support, or one-hot into a dense frame, and never into sparse without overriding
the missing-value sentinel.

**Explicit missing sentinel.** Pass the missing marker explicitly rather than relying on the
default. Cheap insurance against the previous point.

| Problem | Treatment for XGBoost | Difference from plan 01 |
|---|---|---|
| **P1** missing anchor | Leave the NaN. Retain the explicit flag. | Mechanism differs (learned default branch, same as LightGBM in spirit) but the treatment is the same. |
| **P6** blank `wd` | Fill with `(blank)`, declare categorical. | Same as plan 01. Not a hard requirement as it is in plan 02, but keep it consistent so the three tree models see identical inputs. |
| Categorical encoding | Native, via the two settings above. **Never sparse one-hot.** | This trap does not exist in plan 01. |
| **P3**, **P4**, **P5**, **P7**, **P8**, **P9** | Identical to plan 01. | No difference. |
| Scaling | Not applied. | Same. |

## 5. Feature representation

Identical to plan 01 section 6, unchanged. The diversity this plan contributes must come from
the learner, not from a different feature set — otherwise the blend cannot attribute the gain
and the comparison is uninterpretable.

## 6. Hyperparameter approach

L2 objective. Do not tune this model to parity with LightGBM by making it behave like
LightGBM — that is, resist switching it to leaf-wise growth and matching its leaf count. The
value here is the difference. Tune depth, minimum child weight, subsampling and the L1/L2
regularisation pair within a depth-wise posture.

Budget modest tuning effort. A well-configured but not exhaustively tuned XGBoost that
decorrelates well is worth more to plan 07 than a heavily tuned one that has converged toward
LightGBM's solution.

## 7. Validation

As specified in `plans/README.md`, plus one addition specific to this plan's purpose: report
the **correlation between this model's out-of-fold residuals and plan 01's**. That number, not
the standalone RMSE, is what determines whether this plan earns its place.

A residual correlation near 1.0 means the model is redundant and should be cut. Meaningfully
below that means it will contribute to the blend even if its own RMSE is slightly worse.

## 8. Success criteria

- Beats persistence on a majority of heating-season folds. Table stakes.
- Out-of-fold residuals are visibly less than perfectly correlated with plan 01's.
- Improves plan 07's blended score when added. **This is the criterion that matters.**
- Standalone parity with plan 01 is welcome but explicitly not required.

## 9. Risks and failure modes

**The sparse one-hot trap** (section 4). Silent, and the resulting model is bad rather than
broken, which makes it hard to diagnose.

**Convergent tuning.** Tuned aggressively enough against the same folds, three boosted-tree
implementations converge on similar solutions and the blend gains nothing. Tune this one less,
deliberately.

**Opportunity cost.** This is the most cuttable plan in the folder. If time is short, drop it
before dropping plan 07, since a two-model blend still works.

## 10. Effort estimate

Lowest of the three tree plans, assuming plan 01 has built the shared layer. Hours, not days —
provided the sparse trap is avoided rather than debugged.
