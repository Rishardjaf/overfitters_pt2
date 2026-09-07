# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

Hackathon entry (team **overfitters**) forecasting PM2.5 one hour ahead across 12
urban air-quality monitoring stations. Tabular regression on a panel time series.
Scored by **RMSE** on a hidden, chronologically later test set. Lower is better.

Read [PROJECT.md](PROJECT.md) for the full problem statement, EDA findings, and
modelling plan. This file is the working agreement: the rules that are easy to
violate by accident and expensive to discover late.

## The five rules

1. **Never modify `Data/train.csv` or `Data/test.csv`.** They are the provided
   competition data. Read them; write derived artefacts elsewhere.
2. **Never use a negative shift or a centred window.** Every feature must be
   computable from information at or before hour `t`. See "Leakage" below.
3. **Never validate with a random split.** Time-aware folds only. A random KFold
   on this data leaks near-identical adjacent hours and will report a score that
   the leaderboard will not honour.
4. **Never use external data**, and do not try to identify the source dataset.
   Only `Data/train.csv` and `Data/test.csv`. No weather reanalysis, no public
   AQI archives, no satellite products, no searching for the original source.
5. **Always compare against persistence.** A model that does not beat
   `ŷ = current_PM2_5` is not a model.

## Baselines to beat

| Baseline | Train RMSE |
|---|---:|
| Global mean | 77.78 |
| **Persistence (`ŷ = current_PM2_5`)** | **19.67** |
| Persistence, evaluated on 2016-03-01 onward | 15.24 |

Persistence already removes ~75% of the achievable error. The real task is
predicting the *hour-over-hour change*. Quote any new result against 19.67 (or
against the relevant fold's persistence score), never against the mean baseline —
that flatters every model and tells you nothing.

If a change moves the CV score by less than ~0.05 RMSE, treat it as noise unless
it reproduces across folds.

## Leakage

This dataset makes leakage easy and catastrophic. Two specific traps:

**Trap 1 — the test-set self-join.** The test set is contiguous hourly data, so
for most test rows the true answer is the `current_PM2_5` of the *next* test row
at the same station. Using it would score near-perfectly and is **prohibited** —
the rules forbid recovering hidden test targets. Do not build it, do not submit
it. If you notice a validation score that looks impossibly good (RMSE below ~8),
assume leakage and hunt for it before celebrating.

**Trap 2 — accidental future information.** Fitting an imputer, scaler, or target
encoding on the full dataset before splitting leaks future statistics backwards.
Fit on the training fold only.

```
ALLOWED                              FORBIDDEN
  .shift(1), .shift(24)                .shift(-1)  or any negative shift
  .rolling(k).mean()                   .rolling(k, center=True)
  cross-station values at hour t       .bfill()  /  fillna(method='bfill')
  expanding stats over the past        .interpolate() across a fold boundary
                                       fitting transforms on train+test together
```

Before generating any submission, run the leakage guard:

```bash
make audit          # scans src/, scripts/ and notebooks/ for negative shifts,
                    # centred windows, and backfills
```

The checker lives in `src/pm25/audit.py` and is shared with
`tests/test_leakage.py`, so the command line and the test suite cannot disagree.
It tokenizes each file and ignores comments and string literals — prose that
*documents* a forbidden operation is fine, real code is not.

Building features over the concatenated train+test timeline is fine and often
desirable — test rows need their own lags — **provided every window looks strictly
backwards**. The concatenation is not the problem; the direction of the window is.

## Validation

Expanding-window folds split on timestamps, defined in `src/pm25/validation.py`.
The headline number is the mean RMSE over the **heating-season folds**, because
the test period is September–February only and materially more polluted than the
train average (mean `current_PM2_5` 92.3 in test vs 78.0 in train).

Never tune against the public leaderboard. Submissions are limited and the
leaderboard is a small sample; trust the time-aware CV.

## Commands

```bash
make setup        # create the venv and install dependencies (uv, Python 3.12)
make eda          # regenerate the data profile in reports/
make baseline     # persistence and mean baselines through the real pipeline
make train        # time-aware CV training, config via CONFIG=configs/<name>.yaml
make submit       # generate submissions/<name>.csv from a trained model
make audit        # leakage guard
make fmt          # ruff format + check
make test         # pytest
```

Run Python through the project environment: `uv run python ...`, or
`uv run pytest`. Do not install into the system Python — this machine's default
is Python 3.14, which has no wheels for LightGBM/XGBoost; the project pins 3.12.

Resolved stack: pandas 3.0.5, numpy 2.5.2, scikit-learn 1.9.0, LightGBM 4.7.0,
XGBoost 3.4.1, CatBoost 1.2.10. **pandas 3.x**, so copy-on-write is the default
and the legacy `fillna(method=...)` / `df.append` spellings are gone — write
against the 3.x API, and prefer `.ffill()` over `fillna(method='ffill')`.
`uv.lock` is committed; use `uv sync` rather than editing the environment.

## Code conventions

- **Reusable logic lives in `src/pm25/`.** Notebooks are for exploration; once a
  finding is real, graduate it into the package. Do not let a notebook become the
  source of truth for a feature.
- **Every experiment is a config.** New idea → new YAML in `configs/`, not an
  edited constant. Runs record their config, git SHA, CV score, and feature list
  to `reports/experiments.md`.
- Set `random_state` / `seed` everywhere. Results must reproduce.
- Prefer native NaN handling in gradient-boosted trees over imputation. Where you
  do impute, add an explicit `*_is_missing` indicator — missingness here is
  plausibly informative.
- Treat blank `wd` as its own category, never as the modal direction. It is 9×
  more common in test than train.
- Avoid raw `year` as a numeric feature: test years lie outside the training
  range and trees extrapolate badly. Use cyclical and relative encodings.
- Type-hint public functions. Docstrings say *why*, not *what*.

## Working style here

- This is a hackathon: prefer a working end-to-end pipeline over a perfect
  component. Get raw CSV → submission working, then improve pieces.
- When reporting a result, give the CV score, the fold breakdown, and the delta
  against persistence. A single averaged number hides the failure modes that
  matter under RMSE.
- Do not claim an improvement that has not been measured on the time-aware folds.
- The interesting residuals are the two-sided tail events. Report rapid falls,
  calm hours and rapid rises separately, but make the competition decision on
  overall RMSE; a tail-slice regression is a diagnostic, not a second metric.
