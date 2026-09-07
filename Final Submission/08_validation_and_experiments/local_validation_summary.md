# Local Validation Scores and Experiment Comparison

All numbers below are read directly from `scripts/v3/summarize.py`, which prints every recorded
experiment from its saved JSON in `artifacts/v3/results/`. Raw JSON for the three models used
in the final blend is included alongside this file
(`lgbm_f9_deep_ratio.json`, `xgb_f9_ratio_mcw.json`, `cat_f9_ratio.json`), and the blend
evaluation script's console output is in `blend_evaluation_output.txt`.

**Validation method:** four expanding-window, time-aware folds (`src/pm25/validation.py`) —
never a random split. Headline = mean RMSE over the two heating-season folds (0 and 2), which
match the actual test period's season.

## Models used in the final submission

| run | model | features | target | fold 0 | fold 1 | fold 2 | fold 3 | **headline** | boosting rounds selected per fold (early stopping) |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| lgbm_f9_deep_ratio | LightGBM | f9_deep (284 cols) | ratio | 24.90 | 19.80 | 26.99 | 18.95 | **25.942** | 1146 / 3378 / 114 / 697 |
| xgb_f9_ratio_mcw | XGBoost | f9_deep (284 cols) | ratio | 25.11 | 20.63 | 26.87 | 19.06 | **25.989** | 10371 / 8469 / 128 / 577 |
| cat_f9_ratio | CatBoost | f9_deep (284 cols) | ratio | 25.67 | 21.81 | 27.60 | 19.16 | **26.635** | 8918 / 4199 / 536 / 1690 |

## Ensemble evaluation (out-of-fold, `scripts/v3/blend.py`)

```
205,764 OOF rows shared by ['lgbm_f9_deep_ratio', 'xgb_f9_ratio_mcw', 'cat_f9_ratio']

  lgbm_f9_deep_ratio       f0=24.896  f1=19.800  f2=26.988  f3=18.954   headline=25.942
  xgb_f9_ratio_mcw         f0=25.105  f1=20.625  f2=26.872  f3=19.064   headline=25.989
  cat_f9_ratio             f0=25.670  f1=21.809  f2=27.600  f3=19.163   headline=26.635

residual correlation (heating folds):
                    lgbm_f9_deep_ratio  xgb_f9_ratio_mcw  cat_f9_ratio
lgbm_f9_deep_ratio               1.000             0.969         0.932
xgb_f9_ratio_mcw                 0.969             1.000         0.923
cat_f9_ratio                     0.932             0.923         1.000

NNLS blend, fit on one heating fold -> score on the other:
  fit f0 -> f2: blend 26.753 vs best single 26.872   weights lgbm=0.448 xgb=0.358 cat=0.194
  fit f2 -> f0: blend 24.635 vs best single 24.896   weights lgbm=0.182 xgb=0.358 cat=0.460
  equal-weight f0: 24.556
  equal-weight f2: 26.769
```

**Decision:** the NNLS-fitted weights swing considerably depending on which fold they are fit
on (LightGBM's weight alone ranges 0.18–0.45), which is a sign of fold-dependent overfitting of
the blend weights rather than a stable combination. The plain **equal-weight average — heading
RMSE 25.663 — beat every single model and both NNLS fits**, so it was adopted for the final
submission (`--weights` omitted in `make_submission.py`, which defaults to equal weighting).

## Selected prior comparisons, for context on the modelling decisions in §3/§4 of the report

| run | model | features | target | headline | note |
|---|---|---|---|---:|---|
| lgbm_f0 | LightGBM | f0_base (55 cols, raw columns only) | level | 30.801 | starting point before any feature engineering |
| lgbm_f7 | LightGBM | f7_neighbors (264 cols) | level | 27.902 | best feature set on the level target, before switching to the ratio target |
| ridge_f0 | Ridge | f0_base (55 cols) | level | 30.949 | linear baseline |
| **lgbm_f9_deep_ratio** | LightGBM | f9_deep (284 cols) | **ratio** | **25.942** | best single model — the biggest single gain across this project came from changing the target (level → ratio), not from any one feature or algorithm |

Full experiment history (36 recorded runs spanning feature-set ablations, target comparisons,
and algorithm comparisons) is reproducible by re-running the commands in
`05_README_Reproduction_Instructions.md` and printed in full by
`uv run python scripts/v3/summarize.py` once `artifacts/v3/results/` is populated.
