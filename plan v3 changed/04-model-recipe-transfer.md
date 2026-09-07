# Plan v3-04 — Algorithms and combinations

**Question.** On the settled feature set and target, which model families are competitive, and
does combining them beat the best single one?

## What v2 already established (not re-litigated)

Boosted trees (LightGBM, XGBoost, CatBoost, HistGB) landed within 0.08 of each other;
ExtraTrees and Ridge were clearly behind; hyperparameter tuning moved the headline by less than
seed noise (±0.04); seed-averaging 2–3 model types was a real −0.10. None of that depended on
which column played the anchor, so the recipe is *applied*, and only re-opened if the new data
behaves differently.

## Runs

1. LightGBM, XGBoost, CatBoost, HistGB, Ridge — same feature set, same target, same folds, same
   honest round-count rule. Ridge is kept for one reason: on a level target with a strong linear
   proxy (PM10) it may be closer to the trees than it was on the Δ problem, and if so it is a
   more decorrelated blend member than before.
2. Residual correlation between every pair; NNLS blend fit on one heating fold and scored on
   the other, both ways; equal-weight average for reference.
3. Two extra seeds for the top two models.
4. One "different angle" model with no v2 precedent: **LightGBM with a different objective**
   (`huber` as a diagnostic of tail influence; `tweedie` since the target is positive and
   skewed). Scored on raw-scale RMSE like everything else.

## Success criteria

- Anything within 0.1 of the best single model with residual correlation < 0.98 is a blend
  candidate.
- A blend is adopted only if it wins on both heating folds when its weights were fitted on the
  other one.

## Output

`Findings/v3/04-algorithms-and-blend.md`
