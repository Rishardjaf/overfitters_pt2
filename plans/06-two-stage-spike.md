# Plan 06 — Two-stage spike model

**Role:** an experiment that attacks the rows carrying the score directly, instead of hoping a
single model happens to handle them.

**Status:** not run. **Blocked on the Δ gate in plan 01.** Also blocked on plan 01 producing a
tuned single-model score to beat — this plan is meaningless without that reference.

---

## 1. Purpose

**F10** is the sharpest number in the whole EDA: hours where PM2.5 moves by 30 µg/m³ or more
in either direction are **6.78% of rows and 77.0% of all squared error**. The remaining 93% of
hours carry a persistence RMSE of just 9.77 and are very nearly solved already.

A single model trained on all rows spends most of its capacity on the 93% that do not matter.
This plan tests whether splitting the problem — one model for *whether* a large move happens,
another for *how large* — extracts more from the 6.78% that decide the leaderboard.

## 2. The design mistake to avoid first

The obvious implementation is wrong: train a classifier, and if it predicts "spike", use the
spike model, otherwise use the calm model. **A hard switch is strictly worse than the
alternative below and should not be built.**

Under squared error the optimal point prediction is the *conditional mean*. So the two stages
must combine as an expectation — the probability of a large move, multiplying the expected
magnitude given a large move, plus the complementary probability multiplying the expected
magnitude given a calm hour. Every prediction blends both heads, weighted by the probability.

This has a consequence that governs the whole plan: **the classifier's calibration matters more
than its accuracy.** A classifier that ranks perfectly but outputs probabilities that are
systematically too high will damage RMSE across every row it touches. A well-calibrated but
mediocre discriminator will not. Design and evaluate for calibration.

## 3. Spikes are two-sided, and the falling half is bigger

**F10** in detail, which reshapes the naive design:

| Event | Share of rows | Share of squared error | Persistence RMSE |
|---|---:|---:|---:|
| Rising, Δ ≥ 30 | 3.04% | 31.4% | 63.0 |
| **Falling, Δ ≤ −30** | **3.73%** | **45.6%** | **68.5** |
| Calm, \|Δ\| < 30 | 93.2% | 23.0% | 9.77 |

Dissipation events are more common *and* individually larger than onset events. A one-sided
"will pollution spike upward" classifier — the intuitive framing, and the one the notebook's
section 11 emphasises — addresses the **smaller** half of the problem.

So the first stage must be **three-class** (falling / calm / rising), or two separate
one-versus-rest classifiers, and the expectation combines three conditional means rather than
two. The three-class formulation is preferred: it keeps the probabilities normalised, which the
expectation combination requires.

This is the main reason this plan is worth running at all. It is not obvious from the notebook
narrative, and a naive one-sided implementation would leave most of the available gain on the
table.

## 4. Target

**Open — gated on plan 01** for the magnitude heads.

Note that the *classifier* stage is defined on Δ regardless of what plan 01 concludes, since
the event being classified is a change. If plan 01 rejects Δ for the regression target, this
plan becomes internally inconsistent — a Δ-defined classifier feeding level-target regressors —
and should be reconsidered rather than forced.

## 5. The threshold is a free parameter and must be justified

30 µg/m³ is inherited from the notebook, not derived. Measured sensitivity:

| Threshold | Rising rows / SE | Falling rows / SE |
|---|---|---|
| 20 | 6.46% / 36.5% | 6.87% / 50.3% |
| **30** | 3.04% / 31.4% | 3.73% / 45.6% |
| 50 | 0.99% / 24.0% | 1.49% / 37.2% |

The trade is legible: a lower threshold captures more of the squared error but dilutes the
event class toward ordinary hours, weakening what the specialist head is specialising in. A
higher one gives a purer class with too few examples to fit — under 1% of rows at 50.

30 is a defensible starting point. Run 20 and 50 as an explicit sensitivity check and report
all three, rather than presenting 30 as though it were derived.

## 6. Data preparation

**Identical to plan 01.** Both stages are gradient-boosted trees over the same feature matrix,
so the adapter is reused unchanged. No new cleaning decisions.

Two additions specific to this plan, neither of them cleaning:

**Do not resample to fix class imbalance.** At roughly 3–4% per event class the temptation is
oversampling or synthetic minority generation. Both destroy probability calibration, and
section 2 established that calibration is the thing this design depends on. Class weighting has
the same problem in milder form. Prefer leaving the imbalance alone and letting a
properly-fitted probabilistic classifier report genuinely small probabilities.

**Calibration must be fitted chronologically.** If a calibration step is added, its held-out
data must be a later time slice, not a random sample. A randomly-drawn calibration set leaks
adjacent hours in exactly the way the project's validation rules forbid — and it would do so
inside a component whose entire purpose is producing trustworthy probabilities.

## 7. Architecture

Three components over one shared feature matrix:

- **Stage one:** a three-class classifier over falling / calm / rising.
- **Stage two:** conditional magnitude estimators for the event classes, fitted on event rows
  only, plus a calm-hour regressor fitted on calm rows.
- **Combination:** the probability-weighted expectation described in section 2.

Fitting each magnitude head on its own subset is what makes this different from a single model:
the rising head never sees the 93% of calm hours that would otherwise dominate its loss. Fitted
on 3.04% of 360k rows it still has roughly 11,000 examples, which is thin but workable.

## 8. Validation

As specified in `plans/README.md`, plus three additions that are the point of the plan:

- **RMSE decomposed over the three regimes separately.** The headline alone cannot show whether
  the specialisation worked.
- **Classifier calibration**, reported as a reliability curve or equivalent, on the
  heating-season folds. Not just an AUC — ranking quality is not what the combination consumes.
- **Direct comparison against plan 01's single model on the event rows only.** That is the
  claim being tested. If the two-stage model matches plan 01 on calm hours and beats it on
  event hours, it wins even at equal headline RMSE.

## 9. Success criteria

- Beats plan 01's single tuned model on **event-row RMSE** without degrading calm-row RMSE.
- Classifier probabilities are calibrated, not merely discriminative.
- Headline heating-season RMSE at least matches plan 01. Given events carry 77% of the squared
  error, a genuine event-row gain should be visible in the headline; if it is not, suspect that
  the combination is leaking accuracy on the calm rows.
- Threshold sensitivity reported across 20 / 30 / 50.

## 10. Risks and failure modes

**Miscalibration contaminating every row.** The combination applies the classifier's
probabilities everywhere, so a systematic probability bias degrades all 51,063 predictions, not
only the event ones. This is the reason a single well-tuned GBM usually wins: it has no such
failure mode. **Highest-severity risk in this plan.**

**Three models to tune instead of one.** Three times the hyperparameter surface, three times
the overfitting exposure, on the same folds. In a hackathon this is the realistic reason the
plan fails — not that the idea is wrong, but that it consumes the time plan 07 needed.

**Thin event classes.** Roughly 11,000 rising and 13,000 falling examples, drawn from
seasonally clustered episodes rather than independent hours. The effective sample size is
smaller than the row count suggests, and the folds will disagree.

**The one-sided trap** (section 3). Anyone implementing from the notebook narrative rather than
from this document will build a rising-only classifier and address the smaller half.

## 11. Recommendation

Worth running **only after plans 01 and 07 have produced a submittable result.** The expected
value is real but the variance is high and the implementation surface is three times the size
of any other plan here. It is an upside experiment, not a foundation to build on.

## 12. Effort estimate

Highest of the folder. Three models, a combination rule, a calibration procedure, and a
three-way validation breakdown. Do not start it with less than a clear day remaining.
