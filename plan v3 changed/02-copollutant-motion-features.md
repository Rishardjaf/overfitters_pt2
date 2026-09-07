# Plan v3-02 — Feature build-up: levels, lags, motion (1st and 2nd derivative), network, neighbours

**Question.** Built one group at a time on top of the previous, which feature groups earn their
place on the new data — and does the old lesson "motion beats state" survive when the thing in
motion is PM10/CO/NO2 rather than PM2.5?

## What the EDA already told us (so this plan measures, not hopes)

- Levels lead: PM10 0.85, CO 0.76, NO2 0.64 with the raw target.
- Lags decay steadily; PM10 keeps 0.62 at 6h and 0.33 at 24h. Build 1/2/3/6/12/24h.
- Cross-station **levels** agree at 0.84; cross-station **changes** only at 0.28.
- Best first-derivative feature: CO 1h change, r = 0.11 — a third of last round's best.

So the expectation is: levels + lags + network levels do most of the work; motion is a
secondary correction. Second derivatives (acceleration) have never been measured on this data
at all.

## The build order (each step = one LightGBM run, same fixed parameters, level target)

| step | set | adds |
|---|---|---|
| f0 | base | raw pollutants + weather, calendar (cyclical), wind vectors + `wd`/`station` categoricals, missing flags, quality flags, physical ratios (dew-point depression, NO2/CO, SO2/CO, PM10×humidity, PM10×wind) |
| f1 | + lags | PM10/CO/NO2/SO2/O3 at 1, 2, 3, 6, 12, 24h |
| f2 | + rolling | trailing mean/std/min/max over 3/6/12/24h for PM10, CO, NO2 (excluding the current hour), plus trailing means *including* the current hour for PM10 and CO |
| f3 | + motion, 1st derivative | 1/3/6/24h change of each pollutant; wind, temperature, pressure tendencies; dew-point-depression change |
| f4 | + motion, 2nd derivative | acceleration (change of the 1h change; change of the 3h change) for PM10, CO, NO2 |
| f5 | + network levels | leave-one-station-out same-hour mean/median/std/max per pollutant; this station's deviation from the network |
| f6 | + network momentum | LOO mean/median of other stations' 1h and 3h change; count of other stations rising/falling; network acceleration |
| f7 | + neighbours | each other station's PM10 level and 1h change as its own column (24 cols); CO levels (12 cols) |
| f8 | + weather lags and extra meteorology | TEMP/PRES/DEWP/WSPM/RAIN lags; rain rolling sum; wind × humidity, wind × deviation-from-network |

Then plan 07 ablates backward from the best step and prunes pairwise-redundant columns.

## What "worked best on the old data" and how it is adapted

| v2 winner | v2 gain | v3 adaptation |
|---|---|---|
| network momentum of PM2.5 (`net_loo_change_1h_mean`) | strongest single feature, r = 0.33 | same construction on PM10/CO/NO2 (f6); EDA predicts ≈ 0.09 — measured, not assumed |
| station − network deviation × wind (`WSPM_x_PM25_vs_net`) | top gain-importance feature | `WSPM_x_PM10_vs_net` (f8) |
| per-neighbour columns (`nb_chg_<station>`) | −0.07, all folds | `nb_PM10_level/chg_<station>` (f7) |
| acceleration (`PM25_accel`) | r = 0.15 | `PM10/CO/NO2_accel` (f4) — the second-derivative test |
| regime features (rolling std, position in 24h range) | +0.12 group | rolling std of PM10/CO (f2) |
| level×level products, ratios, weather lags | dead weight | tested last (f8) so they cannot mask anything, expected to be pruned in plan 07 |

## Success criteria

- Each step reported as a fold table with Δ vs the previous step; a step is kept if it moves
  the heating-fold headline by > 0.05 in the same direction on both heating folds.
- A clear verdict on first vs second derivatives.
- A ranked gain-importance list from the best step, to seed plan 07.

## Output

`Findings/v3/02-feature-buildup.md` and the running log in
`notebooks/features new/01_feature_exploration.ipynb`.
