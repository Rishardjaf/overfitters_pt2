# Required Disclosure

## External datasets used

None. Only the two competition-provided files (`new data/train_new.csv`,
`new data/test_new.csv`) were read by any part of this pipeline. No weather reanalysis
products, public air-quality archives, satellite data, or any other external dataset was used,
and no attempt was made to identify the original source of the competition data.

## External code, notebooks, repositories, or public solutions consulted or adapted

None consulted or adapted. The feature engineering, validation, and modelling code was written
specifically for this project. The modelling libraries used (LightGBM, XGBoost, CatBoost,
scikit-learn, pandas, numpy) are standard open-source packages installed from PyPI via
`pyproject.toml` / `uv.lock` (included in `09_environment/`) and used through their public
APIs with no modification to library source code.

## Pretrained models used

None. All three models (LightGBM, XGBoost, CatBoost) are trained from scratch on the
competition training data for this submission.

## AI tools or coding agents used

**Claude Code (Anthropic), using Claude models, was used throughout this project** as a coding
and analysis assistant — for writing and editing the feature-engineering, validation, and
experiment-runner code in `02_source_code/`, for running and reading the results of the
cross-validation experiments described in this package, and for drafting this documentation set.
All modelling decisions reported in `01_Methodology_Report.md` (feature selection, the
level-vs-ratio target comparison, per-model hyperparameter choices, and the ensembling
approach) reflect measured cross-validation results rather than being asserted without
supporting numbers; those measurements are included in `08_validation_and_experiments/` so
they can be independently checked. No part of the pipeline calls out to an external AI model
or API at training or inference time — the trained models are ordinary LightGBM/XGBoost/
CatBoost models, and prediction is a standard `model.predict()` call.

## Manual modification or post-processing of predictions

None beyond the two automated post-processing steps documented in §7 of
`01_Methodology_Report.md`: converting the fine-fraction predictions back to the raw PM2.5
scale by multiplying by the same-hour PM10 value, and clipping the result at zero (PM2.5
cannot be negative). No prediction value was manually edited, smoothed, or adjusted based on
leaderboard feedback.

## Additional information used beyond the competition-provided files

None. Every feature is derived exclusively from columns present in
`new data/train_new.csv` / `new data/test_new.csv` (the 12-station identifier, timestamp, five
other pollutants, five weather variables, and the target column). The `current_PM2_5` column
present in an earlier version of the data (`Data/train.csv`, `Data/test.csv`) is not present in
the files this submission is built from and is not used, reconstructed, or approximated as a
model input anywhere in this pipeline (see §2.3 of the methodology report for how this is
verified). A within-project diagnostic column (`diag_pm25_now`) exists solely to compare model
error against a historical baseline during local development and is programmatically excluded
from every feature list used for training or inference.

---

**Use of AI tools is disclosed above in the interest of transparency and reproducibility, per
the organisers' stated policy that such use is not automatically considered misconduct.** The
provenance of every reported number in this package is a script in `02_source_code/`, runnable
against the two competition-provided files, producing the results archived in
`08_validation_and_experiments/`.
