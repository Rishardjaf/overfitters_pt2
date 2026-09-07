# Plan v2-03 — Feature set v2: what to add, what to drop

**Question.** Which features not yet tried carry signal about Δ, and which of the existing 124
are dead weight for the tree model?

## Angle that has not been tried

The jihad set describes the **state** at hour `t` very thoroughly (levels, lags, same-hour
network levels, level×level products) but says almost nothing about **motion** — and the
correlation scan in this session showed motion is where the signal lives:

| feature | r with Δ | in jihad set? |
|---|---:|---|
| network mean change over the last hour | **0.325** | no |
| station minus network mean | −0.239 | yes |
| own 1h change | 0.186 | yes |
| acceleration (change of the change) | 0.148 | no |

Every raw pollutant level is below 0.08. The products of two levels (`PM25_x_CO`, `CO_x_NO2`,
…) are products of two near-zero-signal columns; for a tree they add nothing a split on each
parent cannot express.

## Candidate additions (each justified by an EDA finding)

**Motion / momentum**
- Leave-one-station-out network momentum: mean and median of other stations' 1h and 3h change
  (F6 — onset is city-wide). LOO so the station's own anchor does not dominate.
- Number of other stations currently rising by > 10 µg/m³ and falling by > 10 (a count is
  robust to one outlier station).
- Acceleration: `Δ₁(t) − Δ₁(t−1)`; second difference.
- 1h change of PM10, CO, NO2 (co-moving pollutants confirm a real front vs a sensor blip).
- Change in station-minus-network deviation over 1h (is this station converging to the network
  or diverging?).

**Regime / volatility**
- Rolling std of the *change* series over 6h and 24h (not just std of the level).
- Position of the current value inside its trailing 24h range: `(x − min) / (max − min)`.
- Current value relative to the trailing 24h mean (ratio).

**Data quality (P3, P5)**
- Flat-run length for `current_PM2_5`, `PM10`, `CO` (stuck-sensor counter; a frozen anchor is a
  stale anchor).
- Hours since the last observed row (gap length) — P1 says missing anchors are exactly the rows
  after a gap.
- `pm_inconsistent = current_PM2_5 > PM10` already exists; keep.

**Meteorology**
- Wind vector `u = WSPM·sin(wd)`, `v = WSPM·cos(wd)`; blank `wd` stays NaN.
- Pressure tendency over 3h and 24h (front passage).
- Dew-point depression change over 3h (air drying out = dispersion).
- `wd` as a native categorical with blank as its own level (P6: 9× more common in test).

**Calendar**
- Day-of-year sin/cos, is-weekend, heating-season indicator. Drop raw `year`, `day`,
  `weekofyear`, `dayofyear`.

## Candidate removals (measured, not assumed)

Group ablation with the plan-02 LightGBM configuration on the Δ target: remove one group,
re-run all four folds, record the change in headline RMSE.

| Group | Contents |
|---|---|
| products | all `*_x_*` level×level interactions (~30 columns) |
| ratios | `PM10_PM25_ratio`, `PM25_over_PM10`, `NO2_CO_ratio`, `SO2_CO_ratio` |
| weather_lags | `TEMP/PRES/DEWP/RAIN/WSPM_lag_*` |
| pollutant_lags | `PM10/SO2/NO2/CO/O3_lag_*` |
| long_pm_lags | `current_PM2_5_lag_48h`, `_72h` |
| rolling_minmax | `PM25_roll_min/max_*` |
| calendar_raw | `year`, `day`, `dayofyear`, `weekofyear`, `month` |

A group whose removal changes the headline by less than 0.05 is dead weight and is dropped for
speed and to reduce early-stopping noise. A group whose removal *improves* the headline was
hurting.

## Pairwise redundancy

Report |r| > 0.95 pairs among the final feature set (train only). For a tree this is mostly
harmless, but it inflates feature count and dilutes `feature_fraction`; keep one of each pair.

## Method

1. `features_v2.py --set v2` builds jihad ∪ additions.
2. Run plan-02 LightGBM (Δ) on: jihad, jihad + additions, v2 minus each ablation group.
3. Record the fold table for each; gain importances for the winning set.

## Success criteria

- The additions move the headline by > 0.05 in the right direction on both heating folds.
- At least one removal group is shown to be dead weight.

## Output

`Findings/v2/03-feature-set-v2.md`
