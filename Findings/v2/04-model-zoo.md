# Finding v2-04 — Model zoo: every boosted tree lands in the same place

**Plan:** `plans v2/04-model-zoo.md` · **Features:** `lean` (141) for the like-for-like
comparison; `lean_nb` (169) for the models carried into the final · **Target:** Δ with fallback
F · **Seed:** 42 · Persistence headline 21.316.

## Result on `lean`

| model | fold 0 | fold 1 | fold 2 | fold 3 | **headline** | Δ vs persistence | residual corr. with LightGBM |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | 16.314 | 12.884 | 18.621 | 12.764 | 17.468 | −3.85 | 1.000 |
| XGBoost (hist, depth 8) | 16.333 | 12.901 | 18.578 | 12.900 | 17.456 | −3.86 | 0.990 |
| **CatBoost** (depth 8) | 16.381 | 12.913 | **18.475** | 12.665 | **17.428** | −3.89 | **0.978** |
| sklearn HistGB | 16.420 | 12.871 | 18.588 | 12.682 | 17.504 | −3.81 | 0.982 |
| ExtraTrees (200 trees, leaf 20) | 16.827 | — | — | — | *stopped* | −3.56 (f0) | — |
| Ridge (Δ, robust-scaled, one-hot) | 17.310 | 13.918 | 19.467 | 13.159 | 18.389 | −2.93 | 0.937 |

ExtraTrees was stopped after fold 0: 0.51 behind LightGBM on the fold that matters, with three
folds of CPU still to spend and no route to a blend weight. Ridge is 0.9 behind the trees, as
finding 01 predicted.

## Result on `lean_nb` (the final feature set)

| model | fold 0 | fold 1 | fold 2 | fold 3 | headline |
|---|---:|---:|---:|---:|---:|
| LightGBM (seed 42 / 1 / 2) | 16.25 / 16.23 / 16.29 | 12.81 / 12.78 / 12.79 | 18.55 / 18.51 / 18.61 | 12.44 / 12.48 / 12.55 | 17.40 / 17.37 / 17.45 |
| XGBoost (seed 42 / 1) | 16.26 / 16.30 | 12.80 / 12.83 | 18.40 / 18.47 | 12.74 / 12.79 | 17.33 / 17.38 |
| CatBoost | 16.341 | 12.825 | 18.498 | 12.729 | 17.419 |

Equal-weight average of all six `lean_nb` runs (3 LightGBM + 2 XGBoost + 1 CatBoost): fold 0
**16.170**, fold 2 **18.379**, headline **17.275** — better than the five-run average (17.299) on
both folds. CatBoost's residual correlation with every other run is 0.976–0.977, the lowest in
the table, which is why one CatBoost is worth as much as an extra seed of the others.

## What this says, in plain terms

1. **The four boosting implementations are within 0.08 of each other**, which is the seed-noise
   band measured in finding 05. There is no "better algorithm" to find here; the gain came from
   the target (Δ) and the features, and any of these models delivers it.
2. **XGBoost's lead over LightGBM on fold 2 (18.40 vs 18.55) is consistent across two seeds
   each**, so it is probably a small real advantage on the hardest winter, but it is also within
   what one more seed could erase. Both go into the final.
3. **CatBoost is the most *different* tree** (residual correlation 0.976–0.978 vs 0.99 for
   XGBoost) and its equal-weight addition to the `lean` average improves both heating folds
   (four-model average 16.21 / 18.46 = 17.33 vs best single 17.43). That is why a `lean_nb`
   CatBoost run was added for the final.
4. **Ridge is not a useful blend member.** NNLS gives it 6–9% weight and the blend moves by
   < 0.01. Its residuals are the least correlated (0.94), but its errors are too large for the
   decorrelation to pay.
5. **The submission-format sanity check held everywhere:** no fold below 8 for any model, no
   model worse than persistence on any fold.

## Verdict

- Final average: **LightGBM + XGBoost (+ CatBoost if its `lean_nb` run is within 0.1)**, equal
  weights, several seeds each. Drop ExtraTrees and Ridge.
- Blend weights fitted by NNLS were not adopted: the fitted weights swing between folds
  (LightGBM 0.05–0.31, CatBoost 0.27–0.58) and beat equal weights by < 0.02 on the fold they
  were not fitted on.
