# Findings v3 — the new data (no PM2.5 anchor)

Measured results from executing `plan v3 changed/`. Plain language, one file per plan.

Every number is from the time-aware expanding-window folds in `src/pm25/validation.py`.
**Headline = mean RMSE over the heating-season folds (0 and 2), all rows.** Persistence cannot
be computed on this data; the yardsticks are the two no-feature-engineering baselines
(Ridge 31.99, LightGBM 27.60, as given) reproduced through the harness in finding 01, and the
**target for the round is 17.0**.

Each result also shows, for diagnosis only, what the *old* persistence baseline would have
scored on the same rows (reconstructed from the previous row's label — never a feature) and
the falls / calm / rises split, so results stay comparable with `Findings/v2/`.

| # | File | One-line result (heating-fold RMSE; raw-column baseline 30.80) |
|---|---|---|
| 01 | [01-baselines.md](01-baselines.md) | Harness reproduces your 27.6 (all-fold 26.6). Ridge ≈ LightGBM on raw columns; Ridge *wins* the extreme winter (trees can't extrapolate). |
| 02 | [02-feature-buildup.md](02-feature-buildup.md) | Lags −0.6, first-derivative motion −1.1, **cross-station levels −1.6**, neighbours −0.2 → **27.90**. Acceleration (+0.4) and weather lags hurt. |
| 07 | [07-feature-selection.md](07-feature-selection.md) | Drop acceleration and zero-gain flags, keep rolling; 119 redundant pairs documented; final set **deep (284 cols)**. |
| 03/06 | [06-target-angles.md](06-target-angles.md) | **Predict the fine fraction (y ÷ PM10, weight PM10²): −2.1 → 25.94**, the biggest lever. log target +2.0. Round count is fold-dependent (winter 2 wants ~100 rounds). |
| 04 | [04-algorithms-and-blend.md](04-algorithms-and-blend.md) | *In progress at hand-off.* XGBoost's loss was a weight/`min_child_weight` interaction; CatBoost, Ridge, HistGB, the fix and a regularised arm were queued. |

## The story so far

| step | headline | Δ |
|---|---:|---:|
| LightGBM, raw columns | 30.80 | — |
| + lags, motion, network levels, neighbours | 27.90 | −2.90 |
| + fine-fraction features (deep set) on the level target | 28.03 | +0.13 |
| **ratio target** on f7 | 26.26 | −1.64 |
| **ratio target on the deep set** | **25.94** | −0.32 |
| *target for the round* | *17.0* | *−8.9 still to find* |

Submitted: `submissions/v3_lgbm_deep_ratio_800x3.csv` (LightGBM × 3 seeds, deep set, ratio
target). Expected leaderboard range 23–26. The old anchor's persistence would score 21.3 on
these folds; 17.0 is 4.3 below a baseline that had a column we do not have — the remaining
honest levers are listed in `task/HANDOFF.md`.

Feature exploration and selection is tracked step by step in
`notebooks/features new/01_feature_exploration.ipynb`. Raw per-fold JSON and out-of-fold
predictions are in `artifacts/v3/` (gitignored, regenerable via `scripts/v3/`).

## Integrity checks performed before any result was recorded

- `make audit`: clean.
- `scripts/v3/test_causality.py`: all 292 panel columns (every feature plus the diagnostic
  anchor) identical on 240,912 rows whether or not later rows exist.
- `features_v3.assert_no_pm25()`: every feature set is refused if any column name contains
  `PM2_5`, `pm25`, `PM25` or `diag`.
- Naive yardsticks through the harness: predicting the training mean scores 94.8 on the
  heating folds; predicting `station_ratio × PM10` scores 39.8.
