# Plan v3-01 — New baselines and signal scan

**Question.** Does the harness reproduce the two given baselines (Ridge 31.99, LightGBM 27.599)
end-to-end on `new data/train_new.csv`? And with `current_PM2_5` gone, which of the remaining
raw columns and simple derived features actually carry signal about the target now?

## Why this comes first

Every later plan assumes the panel-building code has been re-pointed at the new files and that
`current_PM2_5` has been fully removed from the pipeline (per migration.md §3 — a stray
reference that silently returns NaN is worse than one that crashes). This plan is where that
gets proven, on the same time-aware folds used throughout the project, before any new feature
family is built on top.

## Method

1. Re-point `scripts/v2/features_v2.py` (or a new `features_v3.py`) at `new data/train_new.csv`
   / `new data/test_new.csv`. Delete every PM2.5-anchor feature group listed in migration.md §6
   from the builder rather than leaving it in and hoping it's unused — grep for
   `current_PM2_5`/`ANCHOR` afterward and confirm zero hits outside the target column itself.
2. Score plain Ridge and plain LightGBM (raw level target — there is no Δ without an anchor) on
   the remaining raw columns only (PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, wd, WSPM,
   station, calendar), on the same expanding-window folds as before. These should land close to
   31.99 / 27.599 — if they don't, something in the migration broke before any new feature work
   starts, and that's the thing to fix first.
3. Correlation scan: every remaining raw column, its own lags (1, 3, 6, 24h), and its rolling
   mean/std against the raw target directly (not a Δ — there's nothing to subtract it from).
   The table in migration.md §7 (PM10 0.848, CO 0.762, NO2 0.643, SO2 0.499, WSPM −0.280, ...)
   is the starting point, not the finish — it doesn't yet include any lag or rolling feature.
4. Multicollinearity check among the five remaining pollutants (already known to be substantial
   from the old correlation matrix — PM10–CO 0.68, CO–NO2 0.69 — re-confirm since Ridge's
   conditioning depends on it).

## Success criteria

- Harness reproduces 31.99 / 27.599 within noise (a config/environment difference, not a
  modelling one, would explain a large gap here).
- A ranked list of which raw-column-derived features are worth carrying into plan 02, and which
  (if any) are already close to dead weight, the same way finding 03 pruned the old set.

## Output

`Findings/v3/01-new-baselines-and-signal-scan.md`
