# Finding v3-01 — Baselines and harness on the new data

**Plan:** `plan v3 changed/01-new-baselines-and-signal-scan.md` · **Harness:** `scripts/v3/`
(panel of 291 features + diagnostic anchor, all backward-looking) · **Folds:** the same four
expanding-window folds as v2 · **Seed:** 42

## Integrity first

- Causality test: all 292 panel columns identical on 240,912 rows whether or not later rows
  exist. Leakage audit clean. `assert_no_pm25()` refuses any feature list with a PM2.5-derived
  name.
- The diagnostic anchor (`diag_pm25_now` = previous row's label) reproduces the **old
  persistence scores exactly** on every fold (20.385 / 15.706 / 22.247 / 15.242), which proves
  the reconstruction is right and gives every v3 result a familiar yardstick — for reading
  only, never as a feature.

## The yardsticks

| model | fold 0 | fold 1 | fold 2 | fold 3 | **heating headline** | all-fold mean |
|---|---:|---:|---:|---:|---:|---:|
| predict the training mean | 87.6 | 57.3 | 102.0 | 61.9 | 94.8 | 77.2 |
| `station_ratio × PM10` (no model) | 43.4 | 44.3 | 36.3 | 40.7 | 39.8 | 41.2 |
| Ridge, base set (55 cols) | 31.15 | 25.56 | **30.75** | 26.21 | 30.95 | 28.42 |
| LightGBM, base set (55 cols) | 28.22 | 22.48 | 33.39 | 22.24 | 30.80 | **26.58** |
| *old persistence, for scale (diagnostic)* | *20.39* | *15.71* | *22.25* | *15.24* | *21.32* | *18.40* |

The base set is the raw pollutant/weather columns plus cyclical calendar, wind vectors,
missing flags, quality flags and a few physical ratios — no lags, no rolling, no network.

## What this says, in plain terms

1. **The harness matches your numbers.** LightGBM's all-fold mean of 26.58 lines up with the
   27.60 you measured; Ridge's 28.4 / 31.0 brackets your 31.99. The heating-season headline is
   higher than either because winter is genuinely harder — the same pattern as v2, where
   persistence scored 19.67 overall but 21.32 on the heating folds.
2. **Without the anchor, even the naive PM10 ratio is a respectable 39.8** — PM10 really is the
   backbone. But the gap from 39.8 to 30.8 shows a model adds a lot on top of it.
3. **LightGBM barely beats Ridge on the base set (30.80 vs 30.95), and on the extreme winter
   fold Ridge is clearly better (30.75 vs 33.39).** A tree cannot predict above the highest
   level in its training window, and winter 2015-16 breaks that ceiling; a linear model
   extrapolates. This is the same failure the Δ target fixed in v2, and it is why the proxy
   target (plan 03) is tested early rather than late — and why Ridge is a real blend member
   this round, not just a sanity check.
4. **How far 17.0 is:** the old easy baseline would score 21.3 on these folds; the target is
   4 points *below* that, without the column that made it possible. Every later finding quotes
   its distance to both.
5. Ridge chose the largest alpha offered (1000) on every fold — the grid is extended to 10,000
   for the plan-04 runs.

## Verdict

Harness validated; yardsticks fixed at **30.80 (LightGBM) / 30.95 (Ridge)** on the heating
folds, **26.58 / 28.42** all-fold. Unblocks plan 02.
