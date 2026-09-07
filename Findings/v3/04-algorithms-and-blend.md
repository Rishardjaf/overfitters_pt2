# Finding v3-04 — Algorithms on the ratio target

**Plan:** `plan v3 changed/04-model-recipe-transfer.md` · **Config for every row:** deep set
(284 cols), ratio target (`y ÷ PM10`, weight `PM10²`), same folds, previous-season early
stopping.

## Result

| model | fold 0 | fold 1 | fold 2 | fold 3 | **headline** | notes |
|---|---:|---:|---:|---:|---:|---|
| LightGBM, seed 42 | 24.90 | 19.80 | 26.99 | 18.95 | **25.94** | best |
| LightGBM, seed 1 | 24.98 | 19.53 | 27.12 | 18.84 | 26.05 | seed band ±0.1 |
| **XGBoost, `min_child_weight` 1e6, depth 7** | 25.11 | 20.63 | **26.87** | 19.06 | **25.99** | fixed — see below; beats LightGBM on fold 2 |
| CatBoost | 25.67 | 21.81 | 27.60 | 19.16 | 26.64 | 0.7 behind; still a decorrelation candidate |
| HistGB | 25.24 | 19.82 | 28.34 | 18.86 | 26.79 | |
| Ridge, alphas to 10,000 | 33.22 | 26.59 | 30.43 | 23.10 | 31.82 | linear model unsuited to the ratio target |
| XGBoost, v2 defaults (broken) | 27.43 | 25.07 | 31.84 | 19.66 | 29.64 | superseded by the fix above |
| LightGBM regularised (31 leaves, min-leaf 300, L2 10, ff 0.6) | _pending_ | | | | | motivated by the fold-2 round curve |

## What this says, in plain terms

1. **XGBoost's original 3.7-point loss was a parameter interaction, not an algorithm verdict —
   confirmed by the fix.** Its `min_child_weight` is a sum of *hessians*, and under `PM10²`
   sample weights (10²–10⁶ per row) the default of 50 constrained nothing — it grew
   unregularised depth-8 trees and lost on every fold, worst on calm hours (23.2 vs 18.7).
   Scaling `min_child_weight` to ~1e6 (matching the weight magnitude) and reducing depth to 7
   brought it to **25.99 — within 0.05 of LightGBM, and actually better on fold 2 (26.87 vs
   26.99)**, the hardest window. **Any model with a hessian-based leaf constraint needs it
   scaled to the sample weights**; this generalises beyond XGBoost — watch for it if CatBoost's
   `min_data_in_leaf`-equivalent is ever tuned.
2. **Ridge is clearly unsuited to the ratio target** (31.82, worse than its own level-target
   run on f0_base at 30.95). The fine fraction's dependence on humidity, wind and combustion
   intensity is non-linear enough that a linear model loses more from the harder target than it
   gains from the stronger single predictor (PM10). Not a blend candidate.
3. **CatBoost and HistGB are both ~0.7–0.9 behind** — real trees, just not tuned as far as
   LightGBM/XGBoost were. CatBoost stays a decorrelation candidate given its lower residual
   correlation with LightGBM in v2 (0.977); confirm with `blend.py`.
4. Seed noise is ±0.1 — smaller than in v2 relative to the score, so effects ≥ 0.2 are
   readable.

## Verdict

**Two models are now essentially tied for best: LightGBM (25.94) and XGBoost (25.99)**, and
they were tuned independently, so their errors are likely to be usefully different — a strong
blend candidate pair. Next: `scripts/v3/blend.py lgbm_f9_deep_ratio xgb_f9_ratio_mcw
cat_f9_ratio` to check residual correlation and NNLS blend weights (HANDOFF Step 1).
