# Plan v2-01 — Harness and baselines

**Question.** What does persistence score on each time-aware fold, and why did the jihad Ridge
model (19.918 on test) lose to it?

## Hypotheses

- **H1.** Persistence on the heating-season folds is materially worse than the 19.67 full-train
  figure (winter is more volatile), so the per-fold persistence number is the correct yardstick
  for every later plan.
- **H2.** Ridge on the raw level with 124 collinear features shrinks the coefficient on
  `current_PM2_5` below 1 and ends up *worse* than persistence. Refitting the same Ridge on
  Δ = `target − current_PM2_5` (anchor added back) cannot be worse than persistence by more than
  noise and should be slightly better.
- **H3.** The unscaled ratio and product features (`PM10_PM25_ratio` max 213, `PM25_x_CO` in the
  millions) are high-leverage points that further damage the linear fit; clipping them at the
  training-fold 99.5th percentile helps Ridge.

## Method

1. Build the panel with `features_v2.py --set jihad` — a faithful re-implementation of the
   notebook's 124 features on the complete hourly grid.
2. Score persistence on every fold; also the calm/fall/rise split.
3. Ridge, inside each fold: median-impute (fit on train fold), robust-scale (fit on train fold),
   one-hot `station`, clip predictors at train-fold percentiles. Four arms:
   - raw level, no clipping
   - raw level, clipped
   - Δ, no clipping
   - Δ, clipped
   Alpha chosen from {0.1, 1, 10, 100} on the inner (previous-season) window.
4. Report the fold table for all arms.

## Success criteria

- Persistence reproduces the 19.67 full-train figure when pooled — proves the harness scores
  correctly.
- The Ridge raw-level arm reproduces the "at or worse than persistence" behaviour seen on the
  leaderboard. If it doesn't, the leaderboard failure had a different cause and this plan says so.

## Output

`Findings/v2/01-baselines-and-ridge.md`
