# PROJECT.md — Next-Hour PM2.5 Forecasting

Team **overfitters** · Machine Learning Hackathon · one-hour-ahead PM2.5 forecasting
across a 12-station urban air-quality monitoring network.

---

## 1. Problem statement

Accurate short-term air-quality forecasts support public warnings, pollution
management, transport planning, and health-protection decisions. Fine particulate
matter (PM2.5) matters especially because its concentration can change rapidly
during pollution events.

**Task.** For each labelled station-hour, predict `PM2_5_next_hour` — the PM2.5
concentration recorded at the *same station* exactly one hour after the
observation timestamp.

**Framing.** This is a supervised tabular regression problem over a panel
(multi-series) time structure: 12 stations × hourly timestamps. It is *not* a
multi-step forecasting problem — the horizon is fixed at exactly +1 hour, and
every predictor is observed at hour `t`.

### Evaluation

Submissions are scored by **Root Mean Squared Error (RMSE)** on the hidden test set:

```
RMSE = sqrt( (1/N) * Σ (y_i - ŷ_i)² )
```

**Lower is better.** RMSE weights large errors quadratically, which is deliberate:
substantially under- or over-estimating a severe pollution event has real
operational consequences for public warnings. Practically, this means **the
leaderboard is decided by the tail** — the handful of rapid-onset spike hours,
not the many calm hours.

Direct consequences for how we model:
- Optimise squared error directly (L2 objective). Do **not** train on `log1p(y)`
  and submit `expm1` predictions without checking — that optimises relative error
  and systematically under-predicts spikes, which is exactly what RMSE punishes.
- Do not clip predictions to a low ceiling.
- A single badly-missed spike hour costs more than a hundred small calm-hour errors.

---

## 2. Data

Location: [Data/](Data/) — `train.csv`, `test.csv`. **Never modify these files.**

| Split | Rows | Period (observation timestamp) | Target column |
|---|---:|---|---|
| `train.csv` | 360,954 | 2013-03-01 00:00 → 2016-08-31 22:00 | `PM2_5_next_hour` present |
| `test.csv`  | 51,063  | 2016-08-31 23:00 → 2017-02-28 22:00 | withheld |

The split is **chronological and contiguous** — test begins the very next hour
after train ends. There is no overlap and no gap.

### Data dictionary

| Column | Description |
|---|---|
| `id` | Unique competition identifier for a station and forecast target time. |
| `observation_timestamp` | Hour at which predictors are observed. |
| `station` | Monitoring-station name (12 levels). |
| `year`, `month`, `day`, `hour` | Components of the observation timestamp. |
| `current_PM2_5` | PM2.5 concentration at the observation hour (µg/m³). |
| `PM10`, `SO2`, `NO2`, `CO`, `O3` | Contemporaneous pollutant measurements. |
| `TEMP` | Temperature (°C). |
| `PRES` | Atmospheric pressure (hPa). |
| `DEWP` | Dew-point temperature (°C). |
| `RAIN` | Precipitation (mm). |
| `wd` | Wind direction (16 compass points). |
| `WSPM` | Wind speed (m/s). |
| `PM2_5_next_hour` | **Target** — PM2.5 one hour after the observation. |

Missing predictor values are preserved as blank CSV fields. Rows without a valid
next-hour target were excluded by the organisers.

### Stations (12)

`Aotizhongxin`, `Changping`, `Dingling`, `Dongsi`, `Guanyuan`, `Gucheng`,
`Huairou`, `Nongzhanguan`, `Shunyi`, `Tiantan`, `Wanliu`, `Wanshouxigong`

Several stations share identical weather readings hour-by-hour (they are served
by the same meteorological source), while pollutant readings differ per station.
This is worth verifying before designing cross-station weather features.

---

## 3. EDA findings

Measured directly from the provided files. Reproduce with `make eda`.

### 3.1 The target is dominated by persistence

`current_PM2_5` is an extremely strong predictor of `PM2_5_next_hour`:

| Baseline (train) | RMSE |
|---|---:|
| Predict the global mean | 77.78 |
| **Predict `current_PM2_5` (persistence)** | **19.67** |
| Persistence, last 6 months of train only | 15.24 |

Persistence removes ~75% of the RMSE a mean-predictor leaves. **This is the number
to beat.** Any model scoring worse than persistence on a time-aware holdout is
broken, not merely weak. The entire competition is the residual: predicting the
*hour-over-hour change* in PM2.5.

A useful reframing: model `Δ = PM2_5_next_hour − current_PM2_5` and add
`current_PM2_5` back at prediction time. This is mathematically equivalent under
RMSE but makes the learning problem better-conditioned and makes it obvious when
a model is adding nothing.

### 3.2 Target distribution

| min | p25 | median | p75 | p95 | p99 | max | mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 21 | 55 | 108 | 235 | 354 | 999 | 78.04 |

Strong right skew. Values are floored at 2 µg/m³ (7,701 rows sit at ≤3) and a
single row reaches 999. The skew is real signal, not corruption — pollution
events genuinely span two orders of magnitude.

### 3.3 Distribution shift: test is a winter set

Test covers **September → February only**; train covers all twelve months.

```
train rows by month: Jan 26.4k  Feb 24.0k  Mar 35.3k  Apr 33.7k  May 34.7k  Jun 33.7k
                     Jul 34.7k  Aug 35.2k  Sep 25.5k  Oct 26.2k  Nov 25.6k  Dec 25.9k
test  rows by month: Aug 0.01k  Sep 8.2k   Oct 8.7k   Nov 8.6k   Dec 8.9k   Jan 8.9k   Feb 7.8k
```

Mean `current_PM2_5` is **77.99 in train vs 92.32 in test** — the test period is
materially more polluted. This is the heating season, when PM2.5 is both higher
and more volatile.

**Implications.**
- A validation split that includes spring/summer will mislead. Validate on
  autumn/winter windows.
- Train contains three prior heating seasons (2013-14, 2014-15, 2015-16), so
  seasonal analogues exist — use them, e.g. via sample weighting or a
  season-matched validation fold.
- Do not let the model lean on `year` as a raw numeric feature: test years
  (2016/2017) sit outside the training range and tree models will extrapolate
  badly. Prefer cyclical and relative time encodings.

### 3.4 Missingness

Missing predictors are retained and must be handled. Rates differ between splits:

| Column | Train | Test |
|---|---:|---:|
| `CO` | 4.39% | 1.51% |
| `O3` | 2.35% | 2.00% |
| `NO2` | 2.08% | 1.31% |
| `SO2` | 1.29% | 0.92% |
| `current_PM2_5` | 0.68% | 0.73% |
| `PM10` | 0.53% | 0.58% |
| `wd` | 0.22% | 1.97% |
| `TEMP`/`PRES`/`DEWP`/`RAIN`/`WSPM` | ~0.05% | ~0.4% |

Notes:
- **`current_PM2_5` is missing on ~0.7% of rows in both splits.** Persistence has
  no answer for these; they need a genuine fallback (last valid observation
  carried forward, or a cross-station estimate for the same hour).
- `wd` missingness is **9× higher in test** than train. Treat blank `wd` as its own
  category rather than imputing to the modal direction.
- Gradient-boosted trees handle NaN natively — prefer that to mean-imputation,
  and add explicit `*_is_missing` indicator columns where missingness may itself
  be informative (instrument outages often coincide with extreme conditions).

### 3.5 Timeline continuity

Each station's hourly series is essentially complete: across all 12 stations there
are only 2,466 places where consecutive retained rows are not exactly one hour
apart. **Lag and rolling-window features are therefore viable** — and they remain
viable at inference time, because the test set is itself a contiguous hourly
series per station.

Gaps must still be handled honestly: build features against a reindexed complete
hourly grid so that a "lag 1" is always genuinely `t−1`, never "the previous
surviving row", which could be hours earlier.

---

## 4. Modelling plan

### Stage 0 — Baselines (must exist before anything else)
1. Persistence: `ŷ = current_PM2_5`. Target to beat: **RMSE 19.67** full-train,
   **15.24** on a recent window.
2. Global mean (sanity floor, RMSE 77.78).
3. Persistence + linear correction on a handful of features.

### Stage 1 — Gradient-boosted trees on tabular features
LightGBM as the primary model (fast, native NaN handling, native categoricals),
XGBoost and CatBoost as diversity for later blending.

Feature groups:
- **Raw predictors** — all pollutants and weather at hour `t`.
- **Lags** (per station, on a complete hourly grid) — `current_PM2_5` at
  t−1, t−2, t−3, t−6, t−12, t−24; the same for `PM10`, `CO`, `NO2`.
- **Deltas / velocity** — `pm25_t − pm25_{t−1}`, `t−1 → t−2`, and second
  differences. Rate of change is the single most informative signal for the
  next hour.
- **Rolling statistics** over trailing windows (3h, 6h, 12h, 24h): mean, std,
  min, max, and the current value's position within that window. **Trailing
  only — never centred.**
- **Ratios** — `PM2_5 / PM10` (fine fraction), `NO2 / CO`.
- **Meteorology derived** — dew-point depression `TEMP − DEWP` (proxy for
  relative humidity, which drives particle growth), wind `u`/`v` components from
  `wd` + `WSPM`, pressure tendency over 3h/24h.
- **Ventilation** — wind speed and its interaction with the PM2.5 level; strong
  wind after a still period is the dominant clearing mechanism.
- **Calendar** — cyclical `sin`/`cos` of hour and day-of-year, day-of-week,
  is-weekend. Avoid raw `year`.
- **Cross-station** — the network-wide mean/median PM2.5 at hour `t`, and this
  station's deviation from it. Pollution fronts move across the network, so
  neighbouring stations carry advance warning. All 12 stations are present in
  test at each hour, so these features are available at inference.

### Stage 2 — Validation design
See §5. Non-negotiable: time-aware only.

### Stage 3 — Refinement
- Tune on the time-aware CV, not the leaderboard.
- Sample weighting toward recent data and toward heating-season months.
- Residual analysis by station, by hour-of-day, and by concentration decile —
  find *where* the model loses to persistence.
- Blend/stack diverse models; consider a separate specialist for high-concentration
  regimes given RMSE's tail sensitivity.

### Stage 4 — Submission
Reproducible end-to-end run from raw CSV to `submissions/*.csv`, with the
validation score, config, and git SHA recorded in [reports/](reports/).

---

## 5. Validation protocol

Because the split is chronological, **random K-fold cross-validation will produce
optimistic, misleading scores** — adjacent hours are nearly identical, so a random
split leaks near-duplicate rows across folds.

**Primary scheme — expanding-window (walk-forward) split by time:**

| Fold | Train up to | Validate on |
|---|---|---|
| 1 | 2014-08-31 | 2014-09-01 → 2015-02-28 |
| 2 | 2015-02-28 | 2015-03-01 → 2015-08-31 |
| 3 | 2015-08-31 | 2015-09-01 → 2016-02-29 |
| 4 | 2016-02-29 | 2016-03-01 → 2016-08-31 |

**Headline metric — the heating-season folds (1 and 3).** They mirror the test
period's season and are the honest estimate of leaderboard performance.

**Final holdout.** Keep 2016-03-01 → 2016-08-31 untouched for a last sanity check
before submitting, and retrain on all data for the final model.

Rules:
- Every fold boundary is a timestamp, never a row index or a shuffle.
- Feature engineering must be causal: any lag, rolling window, or aggregate uses
  data at or before hour `t` only.
- Target encodings and imputation statistics are fitted on the training fold and
  applied to the validation fold — never fitted on the full dataset.
- Report RMSE as headline; also log MAE and per-season/per-station breakdowns to
  catch a model that wins on average while failing on spikes.

---

## 6. Integrity rules

The organisers' constraints, and how we honour them:

- **Do not attempt to identify the original dataset.** No reverse-searching
  station names, value patterns, or timestamps against public sources. Source and
  attribution are disclosed only after the competition.
- **Do not use external copies of the data**, or any external dataset joined on
  time and location (weather reanalysis, satellite AOD, public AQI archives).
  Only `Data/train.csv` and `Data/test.csv`.
- **Do not recover hidden test targets.**

On that last rule there is a specific trap worth naming, because it is easy to
hit by accident. The test set is a contiguous hourly series, so for most test rows
the true answer is literally the `current_PM2_5` of the *next* test row at the
same station. Joining forward in time on the test set would score near-perfectly
and is **prohibited** — it recovers hidden test targets rather than forecasting
them.

The line we hold:

```
ALLOWED  ── information at or before hour t
  lag_1, lag_2, lag_24 and trailing rolling windows over PAST rows
  cross-station readings at the SAME hour t
  station identity, calendar features, weather at t
  computing these across the concatenated train+test timeline,
    provided every window looks strictly backwards

FORBIDDEN ── information after hour t
  next-row current_PM2_5 self-join on the test set
  forward-fill, centred or forward-looking rolling windows
  .shift(-1), any negative shift, on any column
```

Any feature built with a negative shift or a centred window is a bug. Guard for it
in code, and grep the feature pipeline for `shift(-` before every submission.

---

## 7. Repository layout

```
overfitters/
├── CLAUDE.md              # working agreement for AI sessions in this repo
├── PROJECT.md             # this file
├── README.md
├── pyproject.toml         # dependencies, managed with uv
├── Makefile               # make setup / eda / baseline / train / submit
├── Data/                  # provided competition data — READ ONLY
│   ├── train.csv
│   └── test.csv
├── configs/               # YAML experiment configs, one per run
│   └── baseline.yaml
├── src/pm25/              # the package — all reusable logic lives here
│   ├── config.py          # paths, column groups, constants
│   ├── data.py            # loading, dtypes, the complete hourly grid
│   ├── features.py        # causal feature engineering
│   ├── validation.py      # time-aware folds
│   ├── metrics.py         # RMSE and the diagnostic breakdowns
│   ├── models.py          # model factories
│   ├── train.py           # CV training entry point
│   └── predict.py         # submission generation
├── scripts/               # thin CLI wrappers over the package
├── notebooks/             # exploration only — findings graduate into src/
├── artifacts/             # models, OOF predictions, logs (gitignored)
├── submissions/           # generated submission CSVs
└── reports/               # experiment log, result write-ups
```

---

## 8. Roadmap

- [x] Inspect data; confirm schema, split boundaries, missingness, shift
- [x] Establish persistence baseline (RMSE 19.67 / 15.24 recent)
- [x] Write `CLAUDE.md` and `PROJECT.md`
- [x] Scaffold repository and dependency environment
- [ ] Implement `data.py` — loading and the complete hourly grid
- [ ] Implement `validation.py` — expanding-window folds
- [ ] Implement `metrics.py` — RMSE plus per-season/station/decile breakdowns
- [ ] Reproduce the persistence baseline through the real pipeline
- [ ] Implement `features.py` — lags, deltas, rolling windows, meteorology
- [ ] First LightGBM model; confirm it beats persistence on heating-season folds
- [ ] Leakage audit of the feature pipeline
- [ ] Feature iteration guided by residual analysis
- [ ] Hyperparameter tuning on time-aware CV
- [ ] Model diversity and blending
- [ ] Final retrain on all data; submit
