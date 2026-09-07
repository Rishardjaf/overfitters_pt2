# Plan v2-05 — Sample weighting and hyper-parameter tuning

**Question.** The test period is Sep–Feb and 18% more polluted than the train average (F8). Does
telling the model to care more about those months, or tuning tree capacity, move the headline?

## Experiments

**Weighting** (LightGBM, Δ, winning features):
1. Uniform (control).
2. Heating-season months weight 2.0, others 1.0.
3. Recency: weight grows linearly from 0.5 at series start to 1.5 at the fold's train end.
4. Heating × recency.

**Capacity** (one knob at a time, others at plan-02 defaults):
- `num_leaves` ∈ {31, 63, 127, 255}
- `min_data_in_leaf` ∈ {20, 50, 100, 300} — the contentious knob: high values smooth noise, but
  two-sided events are only ~7% of rows and can be smoothed away.
- `learning_rate` 0.02 with proportionally more rounds, for the final model only.

**Diagnostic, not a candidate**: one Huber run to see how much the tails drive the fit. Not
submitted — the metric is L2.

## Method

Every arm on all four folds; headline is folds 0 and 2. Round count chosen on the inner
previous-season window as always. Record fall/calm/rise for every arm — a weighting that helps
the mean by flattening spikes shows up here.

## Success criteria

- A change is adopted only if it improves the headline by > 0.05 and has the same sign on both
  heating folds.

## Output

`Findings/v2/05-weighting-and-tuning.md`
