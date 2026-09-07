# Plan v2-07 — The missing-anchor rows

**Added after findings 01 and 02.** Not in the original sequence; inserted because two
independent measurements pointed at the same lever.

**Question.** 0.7% of rows have no `current_PM2_5`. On them every model so far scores RMSE
40–80, and they alone move the heating-season headline from 18.29 to 19.06 (+0.77). The test
set has 371 such rows (0.73%). Can a dedicated fallback cut that cost?

## Why they are hard

P1 in the EDA: these rows are exactly the rows following a timeline gap, so *the previous hour is
missing by construction*. Own lags, own changes and own rolling stats are all NaN or stale. What
remains at hour `t`: other stations at the same hour (the network is almost always fully
observed), the station's own `PM10`/`CO`/`NO2`/`SO2`/`O3` at `t` (often present even when
PM2.5 is not), weather, station identity, and how long the gap was (`gap_prev_hours`,
`anchor_age_hours`).

## Candidates

| # | Fallback | Idea |
|---|---|---|
| A | LOO network mean (current baseline for persistence) | What the city is doing right now |
| B | General level model (current fallback: LightGBM level, 400 rounds, all rows) | Already measured: 40–80 |
| C | **Level model trained on the missing-anchor-like population only** — rows where the anchor is missing *plus* rows re-labelled by masking their anchor and own-history features, so the model learns to predict the level from network + own PM10 + weather | Trains on the actual input distribution the fallback sees |
| D | `PM10 × station-specific fine fraction` — PM2.5 is usually 0.5–0.9 of PM10 at the same station; fit the ratio on the train fold | Uses the one same-station pollutant that is usually present |
| E | Blend of C and D | |

## Method

- Score only on validation rows with a missing anchor (fold 0: 443, fold 2: 420 rows). Report
  RMSE on those rows and the resulting change in the operational headline.
- For C, the masked-augmentation training set: take every anchor-present training row, set
  `current_PM2_5` and every feature derived from it (lags, changes, rolling, network deviation,
  products) to NaN, keep the level target. Combine with the genuinely missing rows. Fit
  LightGBM level.

## Success criterion

- Missing-anchor RMSE below 40 on both heating folds (from ~40 and ~79), which would recover
  roughly 0.3–0.5 of the operational headline.

## Output

`Findings/v2/07-missing-anchor.md`
