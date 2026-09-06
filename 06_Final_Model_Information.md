# Final Model Information

## Model / ensemble identity

**Equal-weight (1/3, 1/3, 1/3) blend of three gradient-boosted tree models** — LightGBM,
XGBoost, and CatBoost — each independently trained to predict the PM2.5 fine fraction
(`PM2_5_next_hour ÷ PM10`, weighted by `PM10²` in the training loss) on the same 284-column
feature set, then converted back to the raw PM2.5 scale by multiplying by same-hour PM10
before blending. No neural network, AutoML system, or pretrained model is involved.

All three libraries are standard open-source gradient-boosting implementations
(`lightgbm`, `xgboost`, `catboost` on PyPI) — no custom or modified library code, and no
model weights or checkpoints obtained from outside this project.

## Feature set

`f9_deep`, 284 columns, produced by `scripts/v3/features_v3.py` (`build_panel_from` →
`feature_sets()["f9_deep"]`). Composition: raw pollutant/weather columns, calendar and wind
encodings, quality flags, physical ratios, pollutant lags (1–24h), rolling statistics,
first-derivative motion features, same-hour cross-station levels and momentum, per-neighbour
station columns, and a "fine-fraction physics" group (PM10 crossed with combustion tracers,
city-level counterparts, season/diurnal interactions). Full derivation and the incremental
ablation that selected this set are in §3 of `01_Methodology_Report.md`.

## Target

Ratio target: `y = PM2_5_next_hour / max(PM10_filled, 1)`, sample weight `PM10_filled²`.
Predictions are multiplied back by the same denominator before being written to the submission
file. See §4 of the methodology report for why this outperforms predicting the raw level.

## Hyperparameters

### LightGBM (`lightgbm==4.7.0`)

| parameter | value |
|---|---|
| objective | regression |
| metric | rmse |
| learning_rate | 0.05 |
| num_leaves | 63 |
| min_data_in_leaf | 100 |
| feature_fraction | 0.8 |
| bagging_fraction | 0.8 |
| bagging_freq | 1 |
| lambda_l2 | 1.0 |
| num_boost_round | 800 |
| seed | 42, 1, 2 (averaged) |

### XGBoost (`xgboost==3.4.1`)

| parameter | value |
|---|---|
| objective | reg:squarederror |
| tree_method | hist |
| learning_rate | 0.05 |
| max_depth | **7** (reduced from the library default of 6/8 used elsewhere in exploration) |
| min_child_weight | **1,000,000** (see note below — this is not a default value) |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| reg_lambda | 1.0 |
| num_boost_round | 800 |
| seed | 42, 1, 2 (averaged) |

**Note on `min_child_weight`:** this parameter is a sum of per-leaf Hessians in XGBoost, not a
row count. Because the ratio target is trained with `PM10²` sample weights (ranging roughly
10²–10⁶ across rows), the library default of 50 imposes almost no real constraint and produces
overfit trees. It was rescaled to ~1,000,000 to match the sample-weight magnitude; this
single change improved XGBoost's cross-validated heading RMSE from 29.64 to 25.99. See §6 of
the methodology report.

### CatBoost (`catboost==1.2.10`)

| parameter | value |
|---|---|
| loss_function | RMSE |
| learning_rate | 0.08 |
| depth | 8 |
| l2_leaf_reg | 3.0 |
| border_count | 254 |
| iterations | 800 |
| random_seed | 42, 1, 2 (averaged) |

## Ensemble weights

Equal weight across the three models: **0.333 / 0.333 / 0.333** (the `--weights` flag was not
passed to `make_submission.py`, which defaults to equal weighting across the models listed in
`--models`). This was chosen over an NNLS-fitted weight vector because the fitted weights were
unstable across which validation fold they were fit on (ranging roughly 0.18–0.46 per model
depending on fit direction — see `08_validation_and_experiments/blend_evaluation_output.txt`),
while the simple equal-weight average scored better than any single model and was stable
across both fit directions.

Within each model, the three seeds (42, 1, 2) are averaged with equal weight before the
cross-model blend is computed — i.e. the final prediction is the mean of nine trained models
(3 algorithms × 3 seeds), with all nine weighted equally in aggregate.

## Local validation score

Heading RMSE (mean over the two heating-season expanding-window folds), out-of-fold:

| | fold 0 | fold 2 | headline |
|---|---:|---:|---:|
| LightGBM (single) | 24.896 | 26.988 | 25.942 |
| XGBoost (single) | 25.105 | 26.872 | 25.989 |
| CatBoost (single) | 25.670 | 27.600 | 26.635 |
| **Equal-weight blend (final model)** | **24.556** | **26.769** | **25.663** |

Source: `08_validation_and_experiments/blend_evaluation_output.txt`, produced by
`scripts/v3/blend.py lgbm_f9_deep_ratio xgb_f9_ratio_mcw cat_f9_ratio`.

## Leaderboard score

The prediction file at `03_final_prediction/v3_lgbm_xgb_cat_ratio_800x3.csv` is the exact file
submitted for finalist-selection consideration. Its own leaderboard score should be read from
the competition platform's records for this team, since it is assigned by the platform rather
than computed locally.
