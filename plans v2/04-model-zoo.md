# Plan v2-04 — Model zoo on the winning feature set

**Question.** On the feature set chosen in plan 03, does any other model family beat LightGBM,
and are its errors different enough to be worth blending?

## Candidates and why

| Model | Why it is worth one run | Expected |
|---|---|---|
| XGBoost (hist) | Different split-finding and regularisation; standard blend partner | ≈ LightGBM |
| CatBoost | Ordered boosting + strongest native categorical handling for `station × wd` | ≈ LightGBM, possibly better on `wd` |
| sklearn HistGradientBoosting | Independent implementation; cheap sanity check | slightly worse |
| ExtraTrees | Bagged, not boosted — least-correlated tree errors | worse standalone, useful in blend |
| Ridge (Δ, from plan 01) | The only non-tree; most decorrelated | clearly worse standalone |

Not run: RandomForest (ExtraTrees covers the bagged case faster), neural nets (no time for the
tuning they need on 360k rows), any log-target model (PROJECT.md: optimises relative error;
under-predicts spikes).

## Method

- Same panel, same folds, same Δ target with level fallback, same honest early stopping.
- Trees receive NaNs natively. ExtraTrees needs imputation: train-fold median plus the existing
  `*_missing` indicators.
- Store OOF predictions per model in `artifacts/v2/oof/<model>.parquet` for plan 06.
- Report per-model fold table and the pairwise correlation of OOF *residuals* against LightGBM.

## Success criteria

- Any model within 0.1 of LightGBM on the headline with residual correlation < 0.9 is a blend
  candidate.
- A model worse than persistence on either heating fold is dropped.

## Output

`Findings/v2/04-model-zoo.md`
