# Finding v3-02 — Feature build-up: levels and the city are the backbone; acceleration hurts

**Plan:** `plan v3 changed/02-copollutant-motion-features.md` · **Model for every row:**
LightGBM, level target, v2 default parameters, previous-season early stopping · **Seed:** 42 ·
Yardstick: LightGBM base set **30.80** (heating headline). Target: 17.0.

## The build-up, one group at a time

| step | adds | cols | fold 0 | fold 1 | fold 2 | fold 3 | **headline** | Δ | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| f0 | base (raw, calendar, wind, flags, physical ratios) | 55 | 28.22 | 22.48 | 33.39 | 22.24 | 30.80 | — | |
| f1 | + pollutant lags 1–24h | 85 | 27.42 | 21.71 | 33.02 | 21.59 | 30.22 | −0.58 | keep |
| f2 | + rolling mean/std/min/max | 143 | 27.59 | 20.79 | 33.44 | 21.07 | 30.52 | +0.30 | hurts both heating folds, helps summer |
| f3 | + motion, **1st derivative** | 175 | 27.00 | 20.19 | 31.83 | 20.43 | 29.42 | −1.10 | keep |
| f4 | + motion, **2nd derivative** | 181 | 27.15 | 20.20 | 32.55 | 20.51 | 29.85 | **+0.43** | hurts both heating folds |
| f5 | + **network levels** | 208 | 26.59 | 19.50 | 29.93 | 19.40 | 28.26 | **−1.59** | biggest win |
| f6 | + network momentum | 228 | 26.41 | 19.20 | 29.86 | 19.25 | 28.14 | −0.12 | small, both folds |
| **f7** | + per-neighbour columns | 264 | **25.91** | 19.60 | **29.89** | 19.37 | **27.90** | −0.23 | **best** |
| f8 | + weather lags, extra meteorology | 291 | 25.78 | 19.67 | 30.27 | 19.30 | 28.03 | +0.13 | dead weight |

Signed-tail split at f7 (old persistence in brackets): fold 0 falls 55 (70), calm 21.0 (9.8),
rises 61 (61); fold 2 falls 61 (77), calm 20.4 (9.2), **rises 97 (73)**.

## What this says, in plain terms

1. **From 30.80 to 27.90 — a 9% gain from features alone**, and 27.90 is ~1 better than the
   27.6 all-fold number you started from *on the harder winter-only folds* (the all-fold mean
   is 23.69).
2. **The city is the backbone.** Cross-station same-hour *levels* of PM10/CO (f5) were worth
   −1.6 — more than everything else combined after lags. Without your own PM2.5 reading, "what
   does the whole city look like right now" is the best substitute, and it is robust because 11
   other sensors average out any one station's noise.
3. **First derivatives help (−1.1); second derivatives hurt (+0.4).** Acceleration features are
   too noisy on this data — the co-pollutants' hour-to-hour changes are already weak (EDA:
   r ≈ 0.1), and differencing them again is mostly differencing noise. Dropped in plan 07.
4. **Rolling statistics and weather lags are dead weight** (+0.3 and +0.1, both hurting the
   heating folds) — same verdict as v2 for weather lags; rolling stats are new dead weight
   because the tree already has the raw lags.
5. **Where the gain actually lives** (gain share, f7): `PM10` 36%, `PM10 × CO` 26%,
   `net_PM10_median` 13%, `net_CO_loo_mean` 3%, `log_PM10` 3%. Three features carry 75%. The
   model is essentially *PM10, corrected by combustion intensity (CO) and by the city-wide
   level*. Whole groups with ~0% gain: missing flags, quality flags, weather lags, meteo_v2,
   second derivatives.
6. **Rapid rises are the open wound**: RMSE 97 on fold 2 against 73 for the old anchor. The
   onset hour of a spike is where a co-pollutant model has the least to go on, and RMSE weights
   it most.
7. **A caution on noise.** The early-stopped round count on fold 2 swings between runs (262,
   471, 837, 1465) because the previous-season inner window is a poor proxy for the extreme
   winter. Effects under ~0.3 on fold 2 should be read as suggestive until confirmed by the
   ablation and by a second seed.

## What it changes

- `f7` is the base for plans 03/06/07.
- The importance list sets the "go deep" direction: features that predict the **fine fraction**
  (PM2.5 ÷ PM10) — more PM10 × combustion products, per-particle combustion ratios, and a
  season-aware proxy — rather than more columns of any other kind. Run as `f9_deep`.
