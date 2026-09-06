# Final Submission — Team overfitters

**Model under review:** `v3_lgbm_xgb_cat_ratio_800x3` — equal-weight LightGBM + XGBoost +
CatBoost blend, ratio target, f9_deep feature set. This folder contains only the materials for
this model, matching the PDF checklist item by item.

| # | Required material | Location in this folder |
|---|---|---|
| 1 | Final Notebook / Methodology Report | [`01_Methodology_Report.md`](01_Methodology_Report.md) |
| 2 | Complete Source Code | [`02_source_code/`](02_source_code/) — data loading, preprocessing, feature generation, model training, validation, test inference, post-processing, and final-submission generation are all covered; see the execution order in item 5 |
| 3 | Final Submission / Prediction File | [`03_final_prediction/v3_lgbm_xgb_cat_ratio_800x3.csv`](03_final_prediction/v3_lgbm_xgb_cat_ratio_800x3.csv) — the exact file, unmodified, plus its companion summary-statistics JSON |
| 4 | Processed / Cleaned Dataset(s) | [`04_processed_data/README.md`](04_processed_data/README.md) — not duplicated as a file (it is a ~218 MB deterministic cache); the exact regeneration command is given instead, per the option the instructions explicitly allow |
| 5 | README / Reproduction Instructions | [`05_README_Reproduction_Instructions.md`](05_README_Reproduction_Instructions.md) |
| 6 | Final Model Information | [`06_Final_Model_Information.md`](06_Final_Model_Information.md) — hyperparameters and ensemble weights |
| — | Required Disclosure | [`07_Disclosure.md`](07_Disclosure.md) |
| — | Local validation scores / experiment comparisons (strongly recommended) | [`08_validation_and_experiments/`](08_validation_and_experiments/) |
| — | Package/environment file (strongly recommended) | [`09_environment/`](09_environment/) — `pyproject.toml` and `uv.lock` |

## Note on saved model files

Trained model binaries (LightGBM/XGBoost/CatBoost files for the nine models — 3 algorithms ×
3 seeds — that make up the final blend) are **not included** in this package. Training is not
expensive (each full-data fit completes in well under 30 minutes on a standard machine; the
whole final submission was generated in under 4 minutes for the single-model case measured
during this project, and the three-model version completed in the time available before the
submission deadline) and every seed, hyperparameter, and script needed to regenerate them
exactly is included in `02_source_code/`, `06_Final_Model_Information.md`, and
`05_README_Reproduction_Instructions.md`. They can be produced on request if a reviewer would
prefer not to retrain.

## Reading order

For a reviewer verifying the leaderboard result: `01_Methodology_Report.md` →
`06_Final_Model_Information.md` → `05_README_Reproduction_Instructions.md` → run the commands
in the README against `02_source_code/` → compare the regenerated CSV to
`03_final_prediction/v3_lgbm_xgb_cat_ratio_800x3.csv`.
