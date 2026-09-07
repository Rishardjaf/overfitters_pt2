# Finding v2-05 — Weighting and tuning: LightGBM is on a plateau

**Plan:** `plans v2/05-weighting-and-tuning.md` · **Base:** `lgbm_lean_delta` (lean, 141 cols,
Δ target, `num_leaves 63`, `min_data_in_leaf 100`, `lr 0.05`, `feature_fraction 0.8`,
`lambda_l2 1`) · **Seed:** 42 · Persistence headline 21.316.

## Result — one knob at a time

| arm | fold 0 | fold 1 | fold 2 | fold 3 | **headline** | vs base | rounds chosen (f0/f2) |
|---|---:|---:|---:|---:|---:|---:|---|
| base | 16.314 | 12.884 | 18.621 | 12.764 | 17.468 | — | 86 / 145 |
| weights: heating months ×2 | 16.315 | 12.877 | 18.596 | 12.732 | 17.456 | −0.01 | 86 / 211 |
| weights: heating ×2 and recency 0.5→1.5 | 16.284 | 12.870 | 18.575 | 12.803 | **17.430** | −0.04 | 101 / 172 |
| `min_data_in_leaf` 300 | 16.429 | 12.892 | 18.561 | 12.597 | 17.495 | +0.03 | 170 / 271 |
| `num_leaves` 31 | 16.332 | 12.884 | 18.710 | 12.820 | 17.521 | +0.05 | 108 / 224 |
| regularised: mdl 300, `lambda_l2` 10, ff 0.6 | 16.409 | 12.895 | 18.578 | 12.614 | 17.494 | +0.03 | 172 / 267 |
| `feature_fraction` 0.5 | 16.287 | 12.867 | 18.674 | 12.721 | 17.481 | +0.01 | 101 / 161 |
| fixed 600 rounds (jihad features, finding 02) | — | — | — | — | | +0.32 | 600 / 600 |

Signed-tail split barely moves either: falls 46.5–47.3 / 52.6–53.5, calm 9.44–9.82 / 9.23–9.46,
rises 52.6–52.9 / 66.2–67.2 across all arms.

## What this says, in plain terms

1. **Nothing here is worth more than noise.** The whole table spans 17.43–17.52; the 0.05 rule
   says none of it is a real effect. The heating+recency weighting is the only arm that
   improves both heating folds, by 0.03 and 0.05 — real or not, it costs nothing and points the
   model at the test season, so it is kept for the final model.
2. **Regularising harder (bigger leaves, fewer leaves, more L2) does not help**, and neither does
   more capacity. The early-stopped round count already adapts: every regularised arm simply
   chose more rounds to compensate and landed in the same place.
3. **The plateau is a feature-set property.** Finding 03 moved the headline by 0.8 with
   features; the entire hyper-parameter sweep moved it by 0.09. Further gains must come from new
   information (finding 03b: neighbour columns) or from averaging (seeds / models), not tuning.
4. Not run: `learning_rate 0.02`. Reserved for the final model, where it is a free small gain
   for 2.5× the training time.

## Seed variance — how big is "noise" here, measured

Same configuration (`lean_nb`, Δ), three LightGBM seeds and two XGBoost seeds:

| run | fold 0 | fold 2 | headline |
|---|---:|---:|---:|
| LightGBM seed 42 | 16.252 | 18.550 | 17.401 |
| LightGBM seed 1 | 16.234 | 18.513 | 17.374 |
| LightGBM seed 2 | 16.293 | 18.607 | 17.450 |
| XGBoost seed 42 | 16.255 | 18.401 | 17.328 |
| XGBoost seed 1 | 16.297 | 18.465 | 17.381 |
| **equal-weight average of all five** | **16.184** | **18.413** | **17.299** |

Changing nothing but the seed moves the headline by up to 0.08. That is the same size as every
"effect" in the tuning table above, so none of them is real, and the 0.05 rule in CLAUDE.md is
if anything generous. Averaging seeds and the two implementations is worth a genuine −0.10
against the single-seed LightGBM (17.40 → 17.30), better on both heating folds, and the OOF
residuals of any two runs correlate at 0.98–0.99 — so this is variance reduction, not a blend.

## Verdict

- Keep base parameters; add `--weights both` for the final model.
- Stop tuning LightGBM. The final model is a **seed-averaged LightGBM + XGBoost pair** on
  `lean_nb` (finding 06).
