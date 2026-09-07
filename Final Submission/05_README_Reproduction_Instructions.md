# README — Reproducing the Final Submission

This package reproduces `03_final_prediction/v3_lgbm_xgb_cat_ratio_800x3.csv`, the file
associated with the leaderboard submission under review.

## Required files (not included in this package)

- The competition-provided data, placed at `new data/train_new.csv` and
  `new data/test_new.csv` relative to the project root described below. These are the
  organiser-provided files with the `current_PM2_5` column removed; they are not modified by
  our pipeline and are not duplicated in this package since they are competition-owned inputs
  the reviewer already has.

## Directory layout expected at run time

Unpack `02_source_code/` so it sits next to a `new data/` folder holding the two CSVs above:

```
<project root>/
  new data/
    train_new.csv
    test_new.csv
  src/
    pm25/
      config.py
      validation.py
      audit.py
  scripts/
    v2/
      cv.py
      features_v2.py
      run_experiment.py
    v3/
      features_v3.py
      run_experiment.py
      make_submission.py
      blend.py
      summarize.py
      test_causality.py
      queue.sh
```

(`02_source_code/src/...` and `02_source_code/scripts/...` in this package map directly onto
`src/...` and `scripts/...` above — copy their contents in, or point `PYTHONPATH` at them, or
simply run commands from inside `02_source_code/`.)

## Dependencies

Python **3.12** exactly (LightGBM/XGBoost wheels used here are not available for newer
Pythons at time of writing). Install with `uv` (recommended, matches the locked environment in
`09_environment/`) or `pip`:

```bash
cd 02_source_code
uv sync   # uses 09_environment/pyproject.toml and uv.lock — copy both into this directory first
```

Key package versions (see `09_environment/pyproject.toml` / `uv.lock` for the full pinned set):

| package | version |
|---|---|
| Python | 3.12.13 |
| pandas | 3.0.5 |
| numpy | 2.5.2 |
| scikit-learn | 1.9.0 |
| lightgbm | 4.7.0 |
| xgboost | 3.4.1 |
| catboost | 1.2.10 |

No pretrained models, AutoML systems, or external packages beyond these standard open-source
ML libraries are used.

## Execution order

All commands are run from the project root (the directory containing `new data/`, `src/`,
`scripts/`), using the project's Python environment (`uv run python ...` or an activated venv
with the packages above installed).

1. **Build the feature panel** (data loading, cleaning, complete-hourly-grid construction,
   feature generation — see `04_processed_data/README.md` for what this produces):
   ```bash
   uv run python scripts/v3/features_v3.py --rebuild
   ```
   This reads the two raw CSVs and writes a cached panel to `artifacts/v3/panel.parquet`
   (created automatically). Takes under a minute.

2. **Verify no leakage** (required before trusting any downstream result):
   ```bash
   uv run python src/pm25/audit.py
   uv run python scripts/v3/test_causality.py
   ```
   Both must report success. The audit tokenises the feature/experiment scripts for negative
   shifts, centred windows, and backfills; the causality test rebuilds the panel on a truncated
   timeline and asserts every feature column is bit-identical to the full-timeline build
   wherever both have data.

3. **(Optional) Re-run the cross-validated model comparison** that justified the final model
   choice (§6 of `01_Methodology_Report.md`). Each command trains and validates on the four
   time-aware expanding-window folds; see `08_validation_and_experiments/` for the saved output
   of these exact runs.
   ```bash
   uv run python scripts/v3/run_experiment.py --name lgbm_f9_deep_ratio --model lightgbm \
       --features f9_deep --target ratio
   uv run python scripts/v3/run_experiment.py --name xgb_f9_ratio_mcw --model xgboost \
       --features f9_deep --target ratio --params '{"min_child_weight":1000000,"max_depth":7}'
   uv run python scripts/v3/run_experiment.py --name cat_f9_ratio --model catboost \
       --features f9_deep --target ratio
   uv run python scripts/v3/blend.py lgbm_f9_deep_ratio xgb_f9_ratio_mcw cat_f9_ratio
   ```
   Each `run_experiment.py` call writes `artifacts/v3/results/<name>.json` (per-fold RMSE) and
   `artifacts/v3/oof/<name>.parquet` (out-of-fold predictions, used by `blend.py`). Random seed
   is fixed at 42 by default (`src/pm25/config.py::SEED`) for every fold's inner early-stopping
   split and every model's default seed; the three submission seeds (42, 1, 2) are passed
   explicitly to step 4 below, not left at the default.

4. **Generate the final submission file** — this is the script that produces the exact file in
   `03_final_prediction/`:
   ```bash
   uv run python scripts/v3/make_submission.py \
       --name v3_lgbm_xgb_cat_ratio_800x3 \
       --models lightgbm:800 xgboost:800 catboost:800 \
       --features f9_deep --target ratio \
       --seeds 42 1 2 \
       --params '{"xgboost": {"min_child_weight": 1000000, "max_depth": 7}}'
   ```
   This step performs test inference and post-processing end to end: it re-runs the leakage
   audit, retrains all three models (three seeds each) on the **full** training set, predicts
   the fine fraction on the test rows, converts back to the raw PM2.5 scale by multiplying by
   same-hour-filled PM10, clips at zero, and writes `submissions/v3_lgbm_xgb_cat_ratio_800x3.csv`
   plus a companion `.json` with summary statistics. No `--weights` flag is passed, so the three
   models are blended with equal weight (the default), matching §6/§7 of the methodology report.

## Key random seeds and settings

- `SEED = 42` (`src/pm25/config.py`) — default seed for CV folds' inner early-stopping split.
- Final submission seeds: **42, 1, 2**, passed via `--seeds 42 1 2`, averaged with equal
  weight per model before the three models are themselves equal-weight blended.
- `--target ratio` — see §4 of the methodology report; this is not the default value in some
  of the CLI tools' help text and must be passed explicitly.
- `--features f9_deep` — the 284-column feature set; also not the default in every script and
  must be passed explicitly where shown above.
- XGBoost `min_child_weight=1000000, max_depth=7` — required override; the library default of
  50 is far too small once `PM10²` sample weights are in play (§6 of the methodology report).

## Which script generates the final prediction file

`scripts/v3/make_submission.py`, invoked exactly as in step 4 above.
