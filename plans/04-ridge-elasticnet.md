# Plan 04 — Ridge / ElasticNet

**Role:** a linear model whose errors are structurally unlike any tree's, included for blend
decorrelation and as a sanity check. **Not expected to be competitive standalone.**

**Status:** not run. **Blocked on the Δ gate in plan 01.**

---

## 1. Purpose

Two things a tree ensemble cannot give you:

**A decorrelated blend member.** Plans 01–03 and 05 are all axis-aligned partitioners. Their
errors will be highly correlated no matter how differently they are tuned. A linear model fails
in a different direction — smoothly, and by extrapolating rather than by flattening at the
boundary of the training data. That difference is the whole reason this plan exists.

**A sanity check.** If a regularised linear model on well-engineered features lands anywhere
near a tuned LightGBM, the tree models are underperforming and something is wrong with the
feature set or the tuning. It is a cheap instrument for detecting that.

## 2. Expected outcome, stated up front

This model will probably not beat plan 01, and **F7** says why: every raw column's linear
correlation with Δ is near zero. The signal here is non-linear and regime-dependent — the
effect of wind speed depends on humidity, the effect of a lag depends on the volatility
regime — and a linear model can only represent interactions you hand it explicitly.

Writing that expectation down before running the plan is the point. If this model *does* win,
that is a surprising and important finding about the feature set, not a reason to celebrate a
linear baseline.

## 3. Target

**Open — gated on plan 01.** One observation specific to this model family: a linear model
fitted on the raw level with `current_PM2_5` among the features will simply learn a
coefficient close to 1 on that column and express everything else as a correction. That is
approximately persistence-plus-a-linear-term already, so the Δ and level formulations are far
closer to equivalent here than they are for a tree. If plan 01's Δ verdict is marginal, this
plan is a poor place to look for confirmation either way.

## 4. Data preparation specific to Ridge / ElasticNet

**This is by far the heaviest adapter in the folder, and the only one that requires fitted
preprocessing.** That matters for more than effort: fitted transforms are a leakage surface
that the tree plans simply do not have.

### Imputation is mandatory

Ridge raises an error on any missing value — verified, not assumed. So every problem the tree
plans solve by leaving a NaN in place must be solved explicitly here:

- **P1** (missing anchor, 2,466 train / 371 test rows): impute with the station's training-fold
  median, **and always add the missing-anchor indicator alongside**. Without the indicator the
  model cannot distinguish an imputed value from a real one, and these rows are structurally
  different — the previous hour is absent by construction.
- **P4** (`CO`, 4.39% missing), and the remaining pollutants and meteorology: same treatment,
  median plus indicator per column.
- Indicators are not optional decoration here. They are what stops imputation from silently
  fabricating signal.

### Every fitted statistic is a leakage surface

Imputation medians, scaler centres and scales, and the one-hot category list are all *fitted*
objects. Each must be fitted on the training fold alone and then applied to the validation
fold. Fitting any of them on the full dataset before splitting is Trap 2 — it leaks future
statistics backwards, and it will produce a validation score that the leaderboard does not
honour.

Plans 01–03 and 05 have no equivalent exposure, because trees need no fitted preprocessing at
all. **If a leakage bug appears anywhere in this project, this plan is the most likely place it
lives.**

### Scaling

Required — the coefficients are otherwise incomparable and the regularisation penalty is
applied inconsistently across features. Prefer a **robust** scaler (median and interquartile
range) over a standard mean-and-variance one, because **P9** and **F2** mean the distributions
are heavy-tailed and a variance-based scale estimate is dominated by the same extreme rows that
carry the score.

### Categorical encoding

One-hot for `station` (12) and `wd` (17 with `(blank)`, **P6**), giving roughly 27–29 columns
after dropping a reference level. Ordinal encoding is not acceptable here as it is for trees: a
linear model would read the integer codes as a genuine ordering and fit a monotone effect
across compass directions that has no physical meaning.

### Winsorising inputs — the one place it is allowed

The tree plans forbid winsorising anything. For a linear model the rule splits:

- **The target stays untouched**, exactly as everywhere else. **F2** — the worst 1% of rows
  carry 47% of the squared error. Capping the target caps the score.
- **Extreme predictor values may be clipped**, at a high percentile, because a linear model
  gives high-leverage points disproportionate influence over the fitted coefficients in a way
  a tree does not. A single 999 reading can visibly move a coefficient.

This asymmetry is deliberate and should be called out in review, because it looks inconsistent
with the rest of the folder until the leverage argument is stated.

### Explicit interactions and non-linearities

The model cannot discover these, so they must be supplied: cyclical hour and day-of-year (a
linear model given raw `hour` would fit a straight line across midnight), wind speed crossed
with dew-point depression, lag values crossed with the rolling volatility, and station crossed
with season. Keep the list short and justified — each addition worsens the collinearity
problem below.

| Problem | Treatment | Contrast with plan 01 |
|---|---|---|
| **P1** missing anchor | Median-impute **plus indicator**. | Plan 01 leaves the NaN. |
| **P2** gaps | Same upstream grid. | No difference. |
| **P3** PM2.5 > PM10 | Ratio and flag, as elsewhere. Consider clipping extreme ratio values. | Clipping is permitted here, not in plan 01. |
| **P4** `CO` | Impute plus indicator. Treat as continuous despite the 100-unit grid. | Plan 01 passes it through raw. |
| **P5** flat runs | Same counters. | No difference. |
| **P6** blank `wd` | `(blank)` as an explicit one-hot column. | Plan 01 uses a native category. |
| **P7** seasonality | Cyclical encodings **essential**, not merely preferred. | A tree can approximate the cycle with splits; a linear model cannot. |
| **P8** `RAIN` | Binary rain indicator plus a log-transformed magnitude. | Plan 01 passes it through raw; 95.9% zeros is fine for a threshold split and poor for a linear coefficient. |
| **P9** heavy tail | Target untouched. Predictors may be clipped. | Plan 01 clips nothing. |
| Scaling | **Required**, robust scaler. | Not applied in plan 01. |
| One-hot | **Required.** | Not applied in plan 01. |

## 5. Model choice within the family

**Ridge first.** The lag features are near-collinear by construction — **F5** puts level
autocorrelation at 0.97, so lag 1 and lag 2 are almost the same column, and the rolling means
are linear combinations of them. The design matrix will be severely ill-conditioned. L2
regularisation is the textbook response and will produce stable coefficients where an
unregularised fit would not.

**ElasticNet second**, for a different reason: its L1 component performs selection, and the
resulting sparse coefficient set is a readable answer to "which of these engineered features
actually carry linear signal". That has diagnostic value for the tree plans even if the model
itself is never submitted.

Plain unregularised least squares is not worth running given the conditioning.

Report the design matrix condition number. If it is extreme, that is context the reviewer
needs before reading any coefficient.

## 6. Validation

As specified in `plans/README.md`, with the preprocessing pipeline fitted **inside** each fold
rather than once beforehand. Report the out-of-fold residual correlation against plan 01, as in
plan 03 — the same criterion applies.

## 7. Success criteria

- Beats persistence on the heating-season folds. If a regularised linear model on these
  features cannot manage that, the feature set is wrong and every other plan is affected.
- Residuals meaningfully decorrelated from plan 01's.
- Improves plan 07's blend. This is the real bar.
- Standalone parity with the tree models is **not** expected and not required.

## 8. Risks and failure modes

**Leakage through fitted preprocessing** (section 4). Highest-probability leakage site in the
project. Every transform fitted inside the fold, no exceptions.

**Collinearity making coefficients uninterpretable.** Manageable for prediction, but do not
read feature importance off Ridge coefficients on a near-singular design matrix and present it
as insight.

**Interaction sprawl.** Each hand-built interaction worsens conditioning and adds a leakage
surface. Keep the list short and justified.

**Being cut for the wrong reason.** If judged on standalone RMSE it will look like a failure.
Judge it on residual decorrelation and blend contribution, as designed.

## 9. Effort estimate

Highest engineering cost per unit of expected standalone score in the folder, because of the
imputation, scaling and encoding pipeline and the discipline required to keep all of it inside
the fold. Justified only by the blend and the sanity check — if plan 07 is cut, cut this too.
