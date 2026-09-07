# Plan v2-02 — LightGBM on the jihad features: the Δ gate

**Question.** Does a gradient-boosted tree beat persistence on the heating-season folds using
the feature set that already exists, and should it predict Δ or the raw level?

## Why this before any new features

The jihad notebook built a sensible feature set and then handed it to a linear model. F7 in the
EDA (every raw column's linear correlation with Δ is ≈0) says the signal is non-linear and
regime-dependent — wind speed matters differently in damp and dry air, a lag matters differently
in a volatile and a calm regime. That is the shape a tree finds and a line cannot. Before
inventing new features, measure how much of the gap to 18.0 is closed by the model class alone.

## Method

Same panel as plan 01 (`--set jihad`, ~120 features; `station` as a native categorical;
NaNs left in place — no imputation, no scaling). Two arms, identical hyper-parameters, seeds,
folds:

| Arm | Target | Fit rows |
|---|---|---|
| level | `PM2_5_next_hour` | all |
| delta | `PM2_5_next_hour − current_PM2_5`, anchor added back | anchor present |

Fixed starting parameters (not tuned in this plan): `learning_rate 0.05`, `num_leaves 63`,
`min_data_in_leaf 100`, `feature_fraction 0.8`, `bagging 0.8`, `lambda_l2 1.0`, L2 objective.
Round count chosen on the previous-season inner window, then refit on the full training fold.

## Reports

- Fold table, both arms, with Δ vs persistence.
- Gate comparison on anchor-present rows.
- Operational score with the level arm as the missing-anchor fallback.
- Fall / calm / rise split for both arms.
- Top-30 gain importances from the delta arm (input to plan 03's ablation).

## Decision rule

- Δ better by > 0.05 on both heating folds → every later plan uses Δ.
- Inside ±0.05 → treat as noise; use Δ anyway (persistence is its zero output).
- Level wins by > 0.05 → later plans use level and this is recorded as a refuted hypothesis.

## Output

`Findings/v2/02-lightgbm-delta-gate.md`
