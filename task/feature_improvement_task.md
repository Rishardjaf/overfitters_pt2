# Feature improvement task — round 2 features for the ratio model

For the model executing this (Sonnet): every step is written so you can act without
re-reading the project. Do the steps **in order**, **one experiment at a time**, and write
results as you go. Read `task/HANDOFF.md` §1 (rules) and §4 (things that bite) before
starting; this file does not repeat them except where a new step creates a new trap.

---

## 0. Context in ten lines

- Task: predict `PM2_5_next_hour` for 12 stations from `new data/train_new.csv` and
  `new data/test_new.csv`. There is **no** current-hour PM2.5 column. Metric: RMSE.
- Local scoring: `scripts/v3/run_experiment.py`. **Headline = mean RMSE over folds 0 and 2**
  (the September–February windows). Seed noise is ±0.1.
- Best single CV model: `lgbm_f9_deep_ratio` — LightGBM, `f9_deep` (284 cols), `ratio`
  target: headline **25.94** (fold 0 **24.90**, fold 2 **26.99**).
- **Best Kaggle file: `submissions/v3_lgbm_xgb_cat_ratio_800x3.csv`** — equal-weight blend of
  LightGBM + XGBoost (`min_child_weight` 1e6, `max_depth` 7) + CatBoost, all on `f9_deep`,
  ratio target, 3 seeds × 800 rounds. Its config is in `submissions/v3_lgbm_xgb_cat_ratio_800x3.json`.
- Every model on `f9_deep` + ratio sits between 25.9 and 26.3. The features, not the learner,
  set the ceiling. This task adds **six new feature angles**, tests each one alone, keeps the
  winners, confirms them on all three blend members, and writes a new blend submission.
- Gain today is concentrated: `PM10` 36%, `PM10_x_CO` 26%, `net_PM10_median` 13%. The model
  is "PM10 × a learned fine fraction". Every new feature below is there to sharpen the
  fine-fraction estimate where the error lives: overnight hours 00–02, Nov–Dec, rapid rises,
  and a systematic per-station bias of ±4 (`Findings/v3/08-residuals.md`).
- Goal for this round: headline **< 25.5** for a single LightGBM, and a blend submission that
  beats `v3_lgbm_xgb_cat_ratio_800x3` on the leaderboard. Do not chase 17.0 here.

## 1. Rules specific to this task

1. All of `task/HANDOFF.md` §1 applies unchanged (new-data-only features, nothing looks
   forward, fitted statistics fit inside the fold, plan 05 stays gated, one experiment at a
   time, never quote against 19.67).
2. **The fine-fraction encoding (§2.1) is a fitted statistic and lives in the runner, not in
   `features_v3.py`.** It uses the target, so it must be recomputed inside each fold on the
   fold's training rows only, exactly like the existing `--te` flag. Never put a column derived
   from `PM2_5_next_hour` into the panel parquet.
3. **The transport adjacency (§2.6) is fitted once on the first training year only** (rows
   before fold 0's validation start), which is inside every fold's training window. Read the
   fold definitions in `src/pm25/validation.py` and set the cut-off from there, do not hard-code
   a date from memory.
4. After **every** edit to `scripts/v3/features_v3.py`, in this order:
   ```bash
   uv run --no-sync python scripts/v3/features_v3.py --rebuild
   uv run --no-sync python scripts/v3/test_causality.py
   uv run --no-sync python src/pm25/audit.py
   ```
   All three must pass before any experiment is launched.
5. Screen on the heating folds only (`--folds 0 2`, ~4–8 min per LightGBM run). Confirm
   winners on all four folds. Never run two experiments at once.
6. Keep thresholds honest. A group is a **win** if the screening headline is ≤ **25.80** and
   neither fold 0 nor fold 2 is worse than the baseline by more than 0.10. It is **noise** if
   the headline is within ±0.15 of 25.94. It is a **loss** otherwise. Baseline fold values:
   fold 0 24.90, fold 2 26.99.
7. Stop adding feature groups when **two in a row** are noise or losses, unless a remaining
   group is cheap and already built (then run it, but do not build anything further).

## 2. What to build

Add each angle as its **own group** in `scripts/v3/features_v3.py` so it can be tested with
`--add-groups <name>` on top of `f9_deep` without new feature sets. Register every group in
`build_panel_from()` after `groups["deep2"] = add_deep2(df)`, keeping the order below. Do not
change any existing group; `f9_deep` must still be exactly 284 columns after the rebuild.

Existing helpers you will use (all in `features_v3.py` unless noted):

- `_g(df)` → groupby station (from `features_v2`). `_g(df)[col].shift(k)` is a per-station lag.
- `_roll(df, series, window, stats, min_periods)` → per-station trailing window over `series`.
  Pass `_g(df)[col].shift(1)` as the series when the window must exclude the current hour.
- `_net_stat(df, col, stat)` → same-hour cross-station aggregate (`"median"`, `"max"`, ...).
- `_loo_mean(df, col)` → same-hour cross-station mean excluding the own station.
- `_flat_run(df, col)` → run length of unchanged values (pattern for run-length code).
- Existing columns you can build on: `PM10_filled`, `net_PM10_median`, `net_PM10_loo_mean`,
  `PM10_change_1h`, `PM10_lag_1h`, `TEMP_DEWP_diff`, `CO_per_PM10`, `NO2_per_PM10`,
  `SO2_per_PM10`, `wind_u`, `wind_v`, `wd_cat`, `hour`, `month`, `is_heating_season`,
  `PM10_roll_max_24h`, `PM10_roll_mean_24h`.

The panel is a **complete hourly grid per station**, sorted by station then time, with pad
rows flagged `is_pad`. That means "row position difference within a station == hours". Assert
it once at the top of `build_panel_from` after `complete_grid`:

```python
assert _g(df)[TIME_COL].diff().dropna().eq(pd.Timedelta(hours=1)).all(), "grid is not hourly-contiguous"
```

Add two shared helpers next to `_roll`:

```python
def _hours_since(df: pd.DataFrame, cond: pd.Series, cap: int = 168) -> pd.Series:
    """Hours since `cond` was last True at this station (0 if True now). NaN before the first event."""
    c = cond.fillna(False).astype(bool)
    pos = pd.Series(np.arange(len(df), dtype=np.float64), index=df.index)
    last = pos.where(c).groupby(df[STATION_COL], sort=False).ffill()  # past events only
    return (pos - last).clip(upper=cap).astype(np.float32)


def _run_length(df: pd.DataFrame, cond: pd.Series, cap: int = 72) -> pd.Series:
    """Consecutive hours (ending now) for which `cond` has been True at this station."""
    c = cond.fillna(False).astype(bool)
    block = (~c).groupby(df[STATION_COL], sort=False).cumsum()
    run = c.astype(np.int32).groupby([df[STATION_COL], block]).cumsum()
    return run.clip(upper=cap).astype(np.float32)
```

Both only look backwards (`ffill`, `cumsum`), so the causality test will pass; if it does not,
you introduced a bug, not an acceptable exception.

### 2.1 Fine-fraction climatology encoding — group `te_ratio` (runner flag, not a panel group)

**Why.** The `--te` flag we tested encodes the *level* of the target by station × month ×
hour and was +0.31 on the level target, ≈0 on the ratio target. The model now predicts the
fine fraction, so the matching prior is the mean **fine fraction** by station × month × hour,
by station × wind direction, and by station × hour. It attacks the ±4 per-station bias
directly and gives the tree a season-and-direction-aware starting ratio.

**Where.** `scripts/v3/run_experiment.py` and `scripts/v3/make_submission.py`.

1. Generalise `target_encode` to encode any column, defaulting to the target:

   ```python
   def target_encode(fit, apply, keys, name, m=20.0, col=None):
       col = col or TARGET
       g = fit.groupby(keys, observed=True)[col]
       s, n = g.transform("sum"), g.transform("count")
       mu = fit[col].mean()
       fit[name] = ((s - fit[col]) + m * mu) / ((n - 1) + m)
       tab = fit.groupby(keys, observed=True)[col].agg(["sum", "count"])
       tab["enc"] = (tab["sum"] + m * mu) / (tab["count"] + m)
       for df in apply:
           df[name] = df[keys].merge(tab[["enc"]], left_on=keys, right_index=True, how="left")["enc"].to_numpy()
           df[name] = df[name].fillna(mu)
   ```

2. Add a helper that builds the fine-fraction column on a frame **in place** and then encodes it.
   Use a string key for wind direction (categorical merge keys are fragile):

   ```python
   TE_RATIO_KEYS = {
       "ter_smh": [STATION_COL, "month", "hour"],
       "ter_swd": [STATION_COL, "wd_key"],
       "ter_sh": [STATION_COL, "hour"],
   }

   def add_te_ratio(fit: pd.DataFrame, apply: list[pd.DataFrame]) -> list[str]:
       for d in [fit, *apply]:
           d["wd_key"] = d["wd_cat"].astype(str)
       fit["_ff"] = (fit[TARGET] / np.maximum(fit["PM10_filled"], 1.0)).clip(0, 2.0)
       for name, keys in TE_RATIO_KEYS.items():
           target_encode(fit, apply, keys, name, m=50.0, col="_ff")
       fit.drop(columns="_ff", inplace=True)
       return list(TE_RATIO_KEYS)
   ```

   `m=50` because the fine fraction is a noisy per-row quantity; the smoothing should lean on
   the global mean more than the level encoding did. The `clip(0, 2)` stops a near-zero PM10
   reading from dominating a cell.

3. Add `--te-ratio` to `parse()` and, in `run()`, next to the `--te` block:

   ```python
   if args.te_ratio:
       feats_f += add_te_ratio(tr, [va])
       add_te_ratio(in_tr, [in_va])
   ```

   Record `"te_ratio": args.te_ratio` in the results JSON `out` dict.

4. Mirror it in `make_submission.py` after the `--te` block:

   ```python
   if args.te_ratio:
       feats = feats + _v3.add_te_ratio(train, [test])
   ```

   plus the argparse flag.

**Leakage check.** The encoding is fit on `tr` (or `in_tr`) and only looked up on `va`
(or `in_va`); LOO on the fitting rows. `_ff` is dropped from `fit` before training and never
added to `feats_f`. Confirm with `grep -n "_ff" scripts/v3/*.py` that it never reaches a
feature list. `assert_no_pm25` will not catch `_ff` because its name has no forbidden
substring, so this is on you.

### 2.2 Relative humidity — group `humidity`

**Why.** Fine particles take up water above roughly 80% RH and grow into the PM2.5 size
window, so the fine fraction rises sharply with humidity. `TEMP_DEWP_diff` is a linear proxy;
RH is the physically right quantity and the effect is a threshold, not a slope.

```python
def add_humidity(df: pd.DataFrame) -> list[str]:
    """Magnus-formula RH from TEMP and DEWP (both °C). Hygroscopic growth is a threshold effect."""
    a, b = 17.625, 243.04
    t, td = df["TEMP"], df["DEWP"]
    rh = 100.0 * np.exp(a * td / (b + td)) / np.exp(a * t / (b + t))
    df["RH"] = rh.clip(0, 100).astype(np.float32)
    df["RH_hi"] = (df["RH"] >= 80).astype(np.int8)
    df["RH_x_PM10"] = df["RH"] * df["PM10"]
    df["RH_x_netPM10med"] = df["RH"] * df["net_PM10_median"]
    df["RH_hi_x_PM10"] = df["RH_hi"] * df["PM10"]
    g = _g(df)
    df["RH_change_3h"] = df["RH"] - g["RH"].shift(3)
    df["RH_roll_mean_6h"] = _roll(df, g["RH"].shift(1), 6, ["mean"])["mean"].to_numpy()
    df["net_RH_median"] = _net_stat(df, "RH", "median")
    return ["RH", "RH_hi", "RH_x_PM10", "RH_x_netPM10med", "RH_hi_x_PM10",
            "RH_change_3h", "RH_roll_mean_6h", "net_RH_median"]  # fmt: skip
```

Sanity: on train rows RH should be mostly 10–100 with a median near 50–60; if you see values
above 100 before the clip on more than 1% of rows, TEMP/DEWP are swapped or not in °C — stop
and report.

### 2.3 Episode phase and stagnation duration — group `episode`

**Why.** Rapid rises are the largest error (RMSE 89 vs 50 for falls on fold 2). Nothing in
the panel says how *old* the current episode is, how long the air has been still, or when it
last rained. Episode age separates "onset, more to come" from "mature, about to break".

```python
def add_episode(df: pd.DataFrame) -> list[str]:
    g = _g(df)
    out = []
    # stagnation: how long since the wind last cleared the air, and how long it has been calm
    for thr in (2.0, 4.0):
        df[f"hrs_since_wspm_ge{int(thr)}"] = _hours_since(df, df["WSPM"] >= thr)
        out.append(f"hrs_since_wspm_ge{int(thr)}")
    df["calm_run_hrs"] = _run_length(df, df["WSPM"] < 1.5)
    df["hrs_since_rain"] = _hours_since(df, df["RAIN"] > 0)
    out += ["calm_run_hrs", "hrs_since_rain"]
    # episode age: time since clean air, time since the last upward crossing of 100
    df["hrs_since_pm10_lt50"] = _hours_since(df, df["PM10"] < 50)
    cross_up = (df["PM10"] >= 100) & (g["PM10"].shift(1) < 100)
    df["hrs_since_pm10_cross100"] = _hours_since(df, cross_up)
    out += ["hrs_since_pm10_lt50", "hrs_since_pm10_cross100"]
    # momentum streaks, own station and city
    df["pm10_rise_run"] = _run_length(df, df["PM10_change_1h"] > 0)
    df["pm10_fall_run"] = _run_length(df, df["PM10_change_1h"] < 0)
    net_chg = df["net_PM10_median"] - g["net_PM10_median"].shift(1)
    df["net_pm10_rise_run"] = _run_length(df, net_chg > 0)
    out += ["pm10_rise_run", "pm10_fall_run", "net_pm10_rise_run"]
    # where in the episode: current level against the trailing 72h peak (excl. current hour)
    mx72 = _roll(df, g["PM10"].shift(1), 72, ["max"], min_periods=6)["max"].to_numpy()
    df["PM10_roll_max_72h"] = mx72
    df["PM10_over_max72"] = df["PM10"] / pd.Series(mx72, index=df.index).replace(0, np.nan)
    out += ["PM10_roll_max_72h", "PM10_over_max72"]
    return out
```

Before choosing thresholds, print train-row quantiles of `WSPM` (25/50/75/90%) once. If the
75th percentile is far from 2.0 or the 90th far from 4.0, replace them with those quantiles
and say so in the finding. Check `hrs_since_*` are NaN only at the very start of each station's
history and that `pm10_rise_run` has a sensible max (single digits to ~20, not 72).

### 2.4 Coarse-mode indicators — group `coarse`

**Why.** Dust, construction and re-suspended soil raise PM10 without raising the fine
fraction; combustion raises both. When PM10 is "coarse", it is the wrong anchor and the ratio
should drop. **Two facts to avoid duplication:** the own-station ratios already exist in
`deep` (`CO_per_PM10`, `NO2_per_PM10`, `SO2_per_PM10` — inverse of PM10÷CO, same splits for a
tree), and `add_network_levels` already builds `net_<pollutant>_{loo_mean,median,std,max}` for
all five pollutants including SO2, NO2 and O3. What is **missing** is the *city-wide* view of
the coarse/combustion balance and its momentum.

```python
def add_coarse(df: pd.DataFrame) -> list[str]:
    g = _g(df)
    out = []
    for c in ["CO", "NO2", "SO2"]:
        own = f"{c}_per_PM10"                       # exists (deep)
        df[f"net_{own}_median"] = _net_stat(df, own, "median")
        df[f"{own}_vs_net"] = df[own] - df[f"net_{own}_median"]
        out += [f"net_{own}_median", f"{own}_vs_net"]
    # combustion-per-particle momentum (fine-fraction momentum proxy)
    df["CO_per_PM10_lag1"] = g["CO_per_PM10"].shift(1)
    df["CO_per_PM10_change_1h"] = df["CO_per_PM10"] - df["CO_per_PM10_lag1"]
    df["CO_per_PM10_roll_mean_6h"] = _roll(df, g["CO_per_PM10"].shift(1), 6, ["mean"])["mean"].to_numpy()
    df["net_CO_per_PM10_change_1h"] = df["net_CO_per_PM10_median"] - g["net_CO_per_PM10_median"].shift(1)
    out += ["CO_per_PM10_lag1", "CO_per_PM10_change_1h", "CO_per_PM10_roll_mean_6h", "net_CO_per_PM10_change_1h"]
    # coarse spike: PM10 far above the city while combustion is not
    df["PM10_vs_net_x_CO_per_PM10"] = df["PM10_vs_net_loo"] * df["CO_per_PM10"]
    df["net_PM10_x_net_CO_per_PM10"] = df["net_PM10_median"] * df["net_CO_per_PM10_median"]
    out += ["PM10_vs_net_x_CO_per_PM10", "net_PM10_x_net_CO_per_PM10"]
    return out
```

### 2.5 Longer memory — group `memory`

**Why.** Rolling windows stop at 24h. A winter episode is a departure from the week's
baseline; without a 3-day and 7-day mean the tree cannot see the baseline it departs from.

```python
def add_memory(df: pd.DataFrame) -> list[str]:
    g = _g(df)
    out = []
    for c in ["PM10", "CO"]:
        prev = g[c].shift(1)
        for w in (72, 168):
            st = _roll(df, prev, w, ["mean"], min_periods=24)
            df[f"{c}_roll_mean_{w}h"] = st["mean"].to_numpy()
            out.append(f"{c}_roll_mean_{w}h")
    df["PM10_minus_roll72"] = df["PM10"] - df["PM10_roll_mean_72h"]
    df["PM10_over_roll168"] = df["PM10"] / df["PM10_roll_mean_168h"].replace(0, np.nan)
    df["roll24_over_roll168"] = df["PM10_roll_mean_24h"] / df["PM10_roll_mean_168h"].replace(0, np.nan)
    prevnet = g["net_PM10_median"].shift(1)
    df["net_PM10_roll_mean_72h"] = _roll(df, prevnet, 72, ["mean"], min_periods=24)["mean"].to_numpy()
    df["netPM10_minus_roll72"] = df["net_PM10_median"] - df["net_PM10_roll_mean_72h"]
    out += ["PM10_minus_roll72", "PM10_over_roll168", "roll24_over_roll168",
            "net_PM10_roll_mean_72h", "netPM10_minus_roll72"]  # fmt: skip
    return out
```

`PM10_roll_max_72h` is built in `episode` (§2.3), not here, to avoid a duplicate column name.
If `episode` is dropped later, do not move it; the tree does not need it twice.

### 2.6 Direction-aware network (transport) — group `transport`

**Why.** The neighbour columns give every station's PM10 at hour t, but not *which* neighbour
matters right now. Pollution is transported by wind, so the station upwind an hour ago is the
one whose change predicts yours. There are no coordinates and external data is forbidden, so
the adjacency is **learned from the training data**: for every ordered pair (s, n) and every
wind sector at s, how well does n's PM10 change at t−1 predict s's PM10 change at t.

**Fitting window.** Only rows with timestamp **before fold 0's validation start** (read it
from `src/pm25/validation.py`; pass it into the function). Those rows are in every fold's
training window, so no validation row ever shapes the adjacency. Store the fitted table in
`artifacts/v3/transport_adjacency.json` so the finding can show it.

```python
SECTORS = {"N": ["N", "NNE", "NNW", "NE", "NW"], "E": ["E", "ENE", "ESE"],
           "S": ["S", "SSE", "SSW", "SE", "SW"], "W": ["W", "WNW", "WSW"]}  # fmt: skip


def add_transport(df: pd.DataFrame, fit_end: pd.Timestamp) -> list[str]:
    """Learned lead/lag adjacency: which neighbour's previous-hour change predicts mine, by wind sector."""
    stations = sorted(df[STATION_COL].unique())
    wide = df.pivot_table(index=TIME_COL, columns=STATION_COL, values="PM10", aggfunc="first").sort_index()
    wide = wide.reindex(columns=stations)
    chg = wide.diff()                       # change_1h of every station at hour t (grid is complete)
    chg_prev = chg.shift(1)                 # neighbours' change at t-1
    wdw = df.pivot_table(index=TIME_COL, columns=STATION_COL, values="wd_cat", aggfunc="first").reindex(columns=stations)
    sector_of = {d: sec for sec, ds in SECTORS.items() for d in ds}
    fit_mask = chg.index < fit_end
    lead: dict[str, dict[str, dict[str, float]]] = {}   # lead[s][sector][n]
    for s in stations:
        lead[s] = {}
        s_sec = wdw[s].map(sector_of)
        for sec in [*SECTORS, "ALL"]:
            m = fit_mask if sec == "ALL" else (fit_mask & (s_sec == sec).to_numpy())
            scores = {}
            for n in stations:
                if n == s:
                    continue
                r = chg.loc[m, s].corr(chg_prev.loc[m, n])
                scores[n] = float(0.0 if np.isnan(r) else max(r, 0.0))
            lead[s][sec] = scores
    (CACHE_DIR / "transport_adjacency.json").write_text(json.dumps(lead, indent=1))

    # feature values at hour t use SAME-HOUR neighbour data (allowed): the leading
    # neighbour's current PM10 and current change, chosen by the station's current sector.
    idx = df[TIME_COL].to_numpy()
    lvl_now = wide.reindex(idx).to_numpy(np.float32)            # rows aligned with df
    chg_now = chg.reindex(idx).to_numpy(np.float32)
    col_of = {n: i for i, n in enumerate(stations)}
    st = df[STATION_COL].to_numpy()
    sec = df["wd_cat"].astype(str).map(sector_of).fillna("ALL").to_numpy()
    lead1_lvl = np.full(len(df), np.nan, np.float32)
    lead1_chg = np.full(len(df), np.nan, np.float32)
    w_lvl = np.full(len(df), np.nan, np.float32)
    w_chg = np.full(len(df), np.nan, np.float32)
    for s in stations:
        rows_s = st == s
        for sc in [*SECTORS, "ALL"]:
            rows = rows_s & (sec == sc)
            if not rows.any():
                continue
            scores = lead[s][sc] if any(v > 0 for v in lead[s][sc].values()) else lead[s]["ALL"]
            best = max(scores, key=scores.get)
            j = col_of[best]
            lead1_lvl[rows], lead1_chg[rows] = lvl_now[rows, j], chg_now[rows, j]
            ws = np.array([scores[n] for n in stations if n != s], np.float32)
            js = [col_of[n] for n in stations if n != s]
            if ws.sum() > 0:
                w_lvl[rows] = np.nansum(lvl_now[rows][:, js] * ws, axis=1) / ws.sum()
                w_chg[rows] = np.nansum(chg_now[rows][:, js] * ws, axis=1) / ws.sum()
    df["upwind_PM10"] = lead1_lvl
    df["upwind_PM10_change_1h"] = lead1_chg
    df["leadw_PM10"] = w_lvl
    df["leadw_PM10_change_1h"] = w_chg
    df["upwind_minus_own"] = df["upwind_PM10"] - df["PM10"]
    df["netPM10med_x_wind_u"] = df["net_PM10_median"] * df["wind_u"]   # wind_v version exists in deep2
    return ["upwind_PM10", "upwind_PM10_change_1h", "leadw_PM10", "leadw_PM10_change_1h",
            "upwind_minus_own", "netPM10med_x_wind_u"]  # fmt: skip
```

Notes for this group:

- `np.nansum` treats a missing neighbour as 0, which biases the weighted mean down when a
  neighbour is offline. Acceptable for a screening run; if the group wins, replace with a
  NaN-aware weighted mean (mask the weights where the value is NaN) before confirming.
- Wire it as `groups["transport"] = add_transport(df, FOLD0_VAL_START)` with
  `FOLD0_VAL_START = pd.Timestamp(EXPANDING_FOLDS[0][1])` imported from `pm25.validation`
  (that tuple is `(train_end, valid_start, valid_end)`; the value is 2014-09-01 00:00). The
  causality test rebuilds the panel on data truncated at 2015-06-15, which is after that date,
  so the fitted adjacency is identical on the full and truncated panels and the test will
  pass. Do not move the cut-off later.
- Wind labels are the 16 compass points in `pm25.config.WIND_DIRECTIONS` plus `"UNKNOWN"`;
  `SECTORS` above covers all 16, and `"UNKNOWN"` falls through to the `"ALL"` sector.
- Print the top-2 leading neighbours per station for the "ALL" sector once and include the
  table in the finding; if every station's best neighbour is the same one station, the
  adjacency is measuring the city-wide signal, not transport, and the group is unlikely to add
  anything beyond `net_PM10_median`.

## 3. Fine-fraction EDA (no training, do this right after the rebuild)

Write `scripts/v3/fine_fraction_eda.py`. On **train rows only** (`~is_test & ~is_pad`), build
`ff = clip(PM2_5_next_hour / max(PM10_filled, 1), 0, 2)` and print mean, std and count of `ff`
grouped by: station; `RH` decile; `wd_cat`; `hour` × `is_heating_season`; `PM10` decile;
`CO_per_PM10` decile; `hrs_since_wspm_ge2` bins (0, 1–3, 4–12, 13–48, >48); `pm10_rise_run`
bins (0, 1–2, 3–5, >5). Also print the **overall std of `ff`** and, for each grouping, the
between-group std of the group means — the ratio of the two says how much of the
fine-fraction variance that grouping explains. Write the tables to
`Findings/v3/09-fine-fraction-eda.md` with three sentences on which groupings separate the
fine fraction most. This decides nothing on its own, but it tells you which of §2's groups to
expect to win and gives the findings a physical explanation. `ff` is never written to the
panel or any feature list.

## 4. Screening protocol — one run at a time

Every screening run is LightGBM, `f9_deep`, ratio target, heating folds only. Queue them
**one per `queue.sh` invocation** and wait for `QUEUE_DONE` before the next:

```bash
scripts/v3/queue.sh "s2_te_ratio   --features f9_deep --target ratio --folds 0 2 --te-ratio"
scripts/v3/queue.sh "s2_humidity   --features f9_deep --target ratio --folds 0 2 --add-groups humidity"
scripts/v3/queue.sh "s2_episode    --features f9_deep --target ratio --folds 0 2 --add-groups episode"
scripts/v3/queue.sh "s2_coarse     --features f9_deep --target ratio --folds 0 2 --add-groups coarse"
scripts/v3/queue.sh "s2_memory     --features f9_deep --target ratio --folds 0 2 --add-groups memory"
scripts/v3/queue.sh "s2_transport  --features f9_deep --target ratio --folds 0 2 --add-groups transport"
```

Run them in that order (cheapest and most likely first). Read each result with
`uv run --no-sync python scripts/v3/summarize.py s2_` — never from memory. After each run,
append a row to the table in `Findings/v3/10-feature-round2.md`:

| run | group | cols | fold 0 | fold 2 | headline | Δ vs 25.94 | rounds f0/f2 | verdict |

Verdicts use §1 rule 6. Also note, from the results JSON `importance`, the gain rank of the
best new column in the group; a group whose top column ranks below 100 but still "wins" is
suspicious — confirm with a second seed (`--params '{"seed":1}'`) before believing it.

Apply §1 rule 7: after two consecutive noise/loss verdicts, run only the groups that are
already built and cheap, and build nothing new.

**Reference to compare against:** `lgbm_f9_deep_ratio` — fold 0 24.90, fold 2 26.99, headline
25.94, rounds 1146 / 114. Round counts on fold 2 are erratic (100–400 is normal); a group that
moves fold 2's selected rounds to >1,000 while improving the score is a real signal that the
new columns transfer across winters.

## 5. Combine, confirm, and cross-model check

1. **Combined set.** Add `sets["f12_ff"] = sets["f9_deep"] + <winning panel groups>` in
   `feature_sets()` (the `--te-ratio` flag is not a panel group; carry it as a flag). Run all
   four folds: `scripts/v3/queue.sh "lgbm_f12_ratio --features f12_ff --target ratio [--te-ratio]"`.
   Keep the combined set only if its headline beats the best single-group screening result
   or ties it within 0.1; feature groups that won alone can cancel out together.
2. **Backward ablation** if ≥3 groups won: drop each winner in turn from `f12_ff` (heating
   folds only, `--drop-groups <g>`) and remove any group whose removal changes the headline by
   less than 0.1. Fewer columns, same score, is a win for the blend's CatBoost and XGBoost
   runtimes.
3. **Second seed** on the final LightGBM config: `--params '{"seed":1}'`, all four folds. The
   two seeds must agree in sign on both heating folds against 25.94 / 26.05.
4. **XGBoost and CatBoost on the final set** — the blend's other two members must not get
   worse than their `f9_deep` scores (XGBoost 25.99, CatBoost 26.64):
   ```bash
   scripts/v3/queue.sh "xgb_f12_ratio --model xgboost --features f12_ff --target ratio [--te-ratio] --params '{\"min_child_weight\":1000000,\"max_depth\":7}'"
   scripts/v3/queue.sh "cat_f12_ratio --model catboost --features f12_ff --target ratio [--te-ratio]"
   ```
   XGBoost's `min_child_weight` must stay at 1e6 (hessian-based under `PM10²` weights, see
   `Findings/v3/04`). If one member gets worse on the new set while LightGBM improves, that
   member keeps `f9_deep` in the blend; the blend is allowed to mix feature sets — but note
   `make_submission.py` takes one `--features` for all models, so a mixed-set blend needs two
   submission runs and a manual average of the two CSVs (weights from step 6).

## 6. Blend and submission

1. `uv run --no-sync python scripts/v3/blend.py lgbm_f12_ratio xgb_f12_ratio cat_f12_ratio lgbm_f9_deep_ratio xgb_f9_ratio_mcw cat_f9_ratio`.
   Record the residual correlation matrix and the two "fit on one fold → score on the other"
   lines. Adopt NNLS weights only if the blend beats the best single model on **both** lines;
   otherwise use equal weights over the three f12 members, which is what the current best
   Kaggle file does.
2. Write the submission with the same recipe as the current best, on the new set:
   ```bash
   uv run --no-sync python scripts/v3/make_submission.py --name v3_ff_lgbm_xgb_cat_800x3 \
       --models lightgbm:800 xgboost:800 catboost:800 --features f12_ff --target ratio \
       --seeds 42 1 2 --params '{"xgboost":{"min_child_weight":1000000,"max_depth":7}}' [--te-ratio] [--weights w1 w2 w3]
   ```
   Also write a LightGBM-only file (`--models lightgbm:800`) as a cheaper fallback the user can
   submit if the blend's leaderboard score is ambiguous.
3. Sanity-check the JSON the script prints: 51,063 rows, no NaN, `pred_mean` ≈ 90–93,
   `median_pred_over_pm10` ≈ 0.80–0.83, `corr_pred_pm10` ≈ 0.97, `pred_max` < 1,000. Then
   compare the new CSV with `submissions/v3_lgbm_xgb_cat_ratio_800x3.csv` on the same ids:
   RMS difference per row and correlation. An RMS difference under 3 means the new features
   barely changed the predictions and the leaderboard will not move; say so.
4. Tell the user which file to submit, its CV headline per member, and the expected
   leaderboard range (CV headline minus ~2.6, based on 25.94 → 23.34).

## 7. Write-up

- `Findings/v3/09-fine-fraction-eda.md` (§3) and `Findings/v3/10-feature-round2.md` (§4–§6):
  screening table, combined-set result, ablation, cross-model table, blend result, submission
  name. Plain language, one paragraph of "what this says" per section, numbers only from
  `summarize.py` or the results JSON.
- Add both files to the index table in `Findings/v3/README.md` and extend its "story so far"
  table with the new headline and submission name.
- Update `task/HANDOFF.md` §0 (best model, best file) and §3 (mark this round done, list what
  was kept and what was dropped) so the next session starts from the new state.
- Nothing is committed to git; leave that to the user.

## 8. Things that will bite you in this task

- **`_ff`, `wd_key` and the `ter_*` columns are created on fold copies inside the runner.**
  They must never appear in `feats` for a run without `--te-ratio`, and `_ff` must never
  appear at all. `grep` before the first run.
- **Categorical `wd_cat` as a groupby key** silently produces all 17 categories with count 0
  for absent combinations unless `observed=True` is passed. The code above passes it; keep it.
- **The complete grid includes pad rows.** `_hours_since` and `_run_length` count pad hours as
  elapsed time, which is the intended behaviour (a 5-hour sensor gap is still 5 hours of
  weather). Raw-value conditions on pad rows are NaN → treated as False → do not reset a
  run and do not count as an event. That is correct; do not "fix" it.
- **`add_transport` pivots the whole panel twice**; it takes ~30 s and ~2 GB. Run the rebuild
  once, not per experiment. The panel cache handles this.
- **Do not add groups to `f9_deep` or change `deep2`.** `f11_deep2` and the `ratio_robust`
  target exist but are **out of scope for this task** — the user has not cleared them. Do not
  run them unless the user says so.
- Each LightGBM run on two folds is ~4–8 min, on four folds ~8–17 min; XGBoost and CatBoost on
  four folds ~25–30 min each. Budget: §4 ≈ 40 min machine time, §5 ≈ 1.5–2 h, §6 ≈ 15 min.
- If a screening run's fold 2 improves but fold 0 gets worse (or vice versa), it is not a win;
  the test period spans both kinds of winter.
