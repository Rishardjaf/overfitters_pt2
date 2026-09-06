"""Causal feature panel for the NEW data (no current_PM2_5 anywhere).

Same construction as scripts/v2/features_v2.py — train and test concatenated on
one timeline, reindexed to a complete hourly grid per station, every window
strictly backward — but built from the columns that still exist: PM10, SO2, NO2,
CO, O3, weather, wind direction, station and time.

The one PM2.5-derived column, `diag_pm25_now`, is the previous row's *label*
(the value the deleted anchor used to hold). It exists for scoring diagnostics
only and is refused by `assert_no_pm25()` if it ever reaches a feature list.

Cumulative feature sets f0..f8 implement the build order in
"plan v3 changed/02-copollutant-motion-features.md".

    uv run python scripts/v3/features_v3.py --rebuild
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v2"))

from features_v2 import _flat_run, _g, _loo_mean, add_calendar, add_wind, complete_grid  # noqa: E402

from pm25.config import STATION_COL, TARGET, TIME_COL  # noqa: E402
from pm25.validation import EXPANDING_FOLDS  # noqa: E402

NEWDATA = ROOT / "new data"
TRAIN_CSV = NEWDATA / "train_new.csv"
TEST_CSV = NEWDATA / "test_new.csv"
CACHE_DIR = ROOT / "artifacts" / "v3"
PANEL_PATH = CACHE_DIR / "panel.parquet"
# fold 0's validation start (EXPANDING_FOLDS entries are (train_end, valid_start, valid_end)):
# the transport adjacency is fit only on rows strictly before this, so it is identical inside
# every fold's training window and unaffected by later rows (causality test requirement).
FOLD0_VAL_START = pd.Timestamp(EXPANDING_FOLDS[0][1])

POLL = ["PM10", "SO2", "NO2", "CO", "O3"]
WEATHER = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
DIAG = "diag_pm25_now"
FORBIDDEN_SUBSTRINGS = ("PM2_5", "pm25", "PM25", "diag")


def assert_no_pm25(cols: list[str]) -> None:
    bad = [c for c in cols if any(s in c for s in FORBIDDEN_SUBSTRINGS)]
    if bad:
        raise RuntimeError(f"PM2.5-derived columns in a feature list: {bad}")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_raw() -> pd.DataFrame:
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    assert "current_PM2_5" not in train.columns and "current_PM2_5" not in test.columns
    train["is_test"] = False
    test["is_test"] = True
    test[TARGET] = np.nan
    panel = pd.concat([train, test], ignore_index=True)
    panel[TIME_COL] = pd.to_datetime(panel[TIME_COL])
    return panel.sort_values([STATION_COL, TIME_COL]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _roll(df: pd.DataFrame, s: pd.Series, w: int, stats: list[str], min_periods: int = 1) -> pd.DataFrame:
    r = s.groupby(df[STATION_COL], sort=False).rolling(w, min_periods=min_periods)
    out = r.agg(stats).reset_index(level=0, drop=True).sort_index()
    return out


def _net_stat(df: pd.DataFrame, col: str, stat: str) -> pd.Series:
    return df.groupby(TIME_COL, sort=False)[col].transform(stat)


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


# --------------------------------------------------------------------------- #
# feature groups
# --------------------------------------------------------------------------- #
def add_missing_flags(df: pd.DataFrame) -> list[str]:
    out = []
    for c in POLL + WEATHER + ["wd"]:
        df[f"{c}_missing"] = df[c].isna().astype(np.int8)
        out.append(f"{c}_missing")
    return out


def add_quality(df: pd.DataFrame) -> list[str]:
    df["o3_extreme"] = (df["O3"] > 400).astype(np.int8)
    df["co_floor"] = (df["CO"] == 100).astype(np.int8)
    df["co_ceiling"] = (df["CO"] == 10000).astype(np.int8)
    df["PM10_flat_run"] = _flat_run(df, "PM10")
    df["CO_flat_run"] = _flat_run(df, "CO")
    real_ts = df[TIME_COL].where(~df["is_pad"])
    last = real_ts.groupby(df[STATION_COL], sort=False).shift(1)
    last = last.groupby(df[STATION_COL], sort=False).ffill()  # forward fill = past only
    df["gap_prev_hours"] = ((df[TIME_COL] - last) / pd.Timedelta(hours=1)).astype("float32")
    return ["o3_extreme", "co_floor", "co_ceiling", "PM10_flat_run", "CO_flat_run", "gap_prev_hours"]


def add_physical(df: pd.DataFrame) -> list[str]:
    df["TEMP_DEWP_diff"] = df["TEMP"] - df["DEWP"]
    df["NO2_CO_ratio"] = df["NO2"] / df["CO"].replace(0, np.nan)
    df["SO2_CO_ratio"] = df["SO2"] / df["CO"].replace(0, np.nan)
    df["O3_NO2_ratio"] = df["O3"] / df["NO2"].replace(0, np.nan)
    df["PM10_x_ddd"] = df["PM10"] * df["TEMP_DEWP_diff"]
    df["PM10_x_WSPM"] = df["PM10"] * df["WSPM"]
    df["PM10_x_CO"] = df["PM10"] * df["CO"]
    df["log_PM10"] = np.log1p(df["PM10"])
    df["log_CO"] = np.log1p(df["CO"])
    return ["TEMP_DEWP_diff", "NO2_CO_ratio", "SO2_CO_ratio", "O3_NO2_ratio",
            "PM10_x_ddd", "PM10_x_WSPM", "PM10_x_CO", "log_PM10", "log_CO"]  # fmt: skip


def add_lags(df: pd.DataFrame, cols: list[str], lags: list[int]) -> list[str]:
    g = _g(df)
    out = []
    for c in cols:
        for k in lags:
            df[f"{c}_lag_{k}h"] = g[c].shift(k)
            out.append(f"{c}_lag_{k}h")
    return out


def add_rolling(df: pd.DataFrame) -> list[str]:
    g = _g(df)
    out = []
    for c in ["PM10", "CO", "NO2"]:
        prev = g[c].shift(1)
        for w in [3, 6, 12, 24]:
            st = _roll(df, prev, w, ["mean", "std", "min", "max"])
            for s in ["mean", "std", "min", "max"]:
                df[f"{c}_roll_{s}_{w}h"] = st[s].to_numpy()
                out.append(f"{c}_roll_{s}_{w}h")
    # trailing means INCLUDING the current hour (still causal: window ends at t)
    for c in ["PM10", "CO"]:
        for w in [3, 6, 12, 24]:
            st = _roll(df, df[c], w, ["mean"])
            df[f"{c}_rmean_incl_{w}h"] = st["mean"].to_numpy()
            out.append(f"{c}_rmean_incl_{w}h")
    # position of PM10 inside its trailing 24h range
    rng = (df["PM10_roll_max_24h"] - df["PM10_roll_min_24h"]).replace(0, np.nan)
    df["PM10_pos_in_24h"] = (df["PM10"] - df["PM10_roll_min_24h"]) / rng
    df["PM10_minus_roll24"] = df["PM10"] - df["PM10_roll_mean_24h"]
    out += ["PM10_pos_in_24h", "PM10_minus_roll24"]
    return out


def add_motion1(df: pd.DataFrame) -> list[str]:
    """First derivatives."""
    g = _g(df)
    out = []
    for c in POLL:
        for k in [1, 3, 6, 24]:
            df[f"{c}_change_{k}h"] = df[c] - g[c].shift(k)
            out.append(f"{c}_change_{k}h")
        df[f"{c}_pct_change_1h"] = df[f"{c}_change_1h"] / g[c].shift(1).replace(0, np.nan)
        out.append(f"{c}_pct_change_1h")
    df["WSPM_change_1h"] = df["WSPM"] - g["WSPM"].shift(1)
    df["TEMP_change_3h"] = df["TEMP"] - g["TEMP"].shift(3)
    df["PRES_tend_3h"] = df["PRES"] - g["PRES"].shift(3)
    df["PRES_tend_24h"] = df["PRES"] - g["PRES"].shift(24)
    df["ddd_change_3h"] = df["TEMP_DEWP_diff"] - g["TEMP_DEWP_diff"].shift(3)
    out += ["WSPM_change_1h", "TEMP_change_3h", "PRES_tend_3h", "PRES_tend_24h", "ddd_change_3h"]
    # rolling std of the change (volatility regime)
    for c in ["PM10", "CO"]:
        prev = g[f"{c}_change_1h"].shift(1)
        st = _roll(df, prev, 6, ["std"], min_periods=2)
        df[f"{c}_change_roll_std_6h"] = st["std"].to_numpy()
        out.append(f"{c}_change_roll_std_6h")
    return out


def add_motion2(df: pd.DataFrame) -> list[str]:
    """Second derivatives (acceleration)."""
    g = _g(df)
    out = []
    for c in ["PM10", "CO", "NO2"]:
        df[f"{c}_accel_1h"] = df[f"{c}_change_1h"] - g[f"{c}_change_1h"].shift(1)
        df[f"{c}_accel_3h"] = df[f"{c}_change_3h"] - g[f"{c}_change_3h"].shift(3)
        out += [f"{c}_accel_1h", f"{c}_accel_3h"]
    return out


def add_network_levels(df: pd.DataFrame) -> list[str]:
    out = []
    for c in POLL:
        df[f"net_{c}_loo_mean"] = _loo_mean(df, c)
        df[f"net_{c}_median"] = _net_stat(df, c, "median")
        df[f"net_{c}_std"] = _net_stat(df, c, "std")
        df[f"net_{c}_max"] = _net_stat(df, c, "max")
        df[f"{c}_vs_net_loo"] = df[c] - df[f"net_{c}_loo_mean"]
        out += [f"net_{c}_loo_mean", f"net_{c}_median", f"net_{c}_std", f"net_{c}_max", f"{c}_vs_net_loo"]
    df["net_n_available"] = _net_stat(df, "PM10", "count")
    df["PM10_over_net_loo"] = df["PM10"] / df["net_PM10_loo_mean"].replace(0, np.nan)
    out += ["net_n_available", "PM10_over_net_loo"]
    # PM10 with a same-hour cross-station fill (helper for the proxy / ratio targets)
    df["PM10_filled"] = df["PM10"].where(df["PM10"].notna(), df["net_PM10_loo_mean"])
    return out


def add_network_momentum(df: pd.DataFrame) -> list[str]:
    g = _g(df)
    out = []
    for c in ["PM10", "CO", "NO2"]:
        df[f"net_{c}_loo_change_1h"] = _loo_mean(df, f"{c}_change_1h")
        df[f"net_{c}_loo_change_3h"] = _loo_mean(df, f"{c}_change_3h")
        df[f"net_{c}_change_1h_median"] = _net_stat(df, f"{c}_change_1h", "median")
        out += [f"net_{c}_loo_change_1h", f"net_{c}_loo_change_3h", f"net_{c}_change_1h_median"]
        # network mean's own momentum and acceleration
        df[f"net_{c}_mean_lag1"] = g[f"net_{c}_loo_mean"].shift(1)
        df[f"net_{c}_mean_change_1h"] = df[f"net_{c}_loo_mean"] - df[f"net_{c}_mean_lag1"]
        df[f"net_{c}_accel"] = df[f"net_{c}_mean_change_1h"] - g[f"net_{c}_mean_change_1h"].shift(1)
        out += [f"net_{c}_mean_change_1h", f"net_{c}_accel"]
        # deviation from network: is it converging or diverging?
        df[f"{c}_vs_net_change_1h"] = df[f"{c}_vs_net_loo"] - g[f"{c}_vs_net_loo"].shift(1)
        out.append(f"{c}_vs_net_change_1h")
    thr = 10.0
    ch = df["PM10_change_1h"]
    rising = (ch > thr).astype(float).where(ch.notna())
    falling = (ch < -thr).astype(float).where(ch.notna())
    df["_r"], df["_f"] = rising, falling
    df["net_n_rising_PM10"] = _net_stat(df, "_r", "sum") - rising.fillna(0)
    df["net_n_falling_PM10"] = _net_stat(df, "_f", "sum") - falling.fillna(0)
    df.drop(columns=["_r", "_f"], inplace=True)
    out += ["net_n_rising_PM10", "net_n_falling_PM10"]
    return out


def add_neighbors(df: pd.DataFrame) -> list[str]:
    out = []
    stations = sorted(df[STATION_COL].unique())
    for col, tag in [("PM10", "PM10_level"), ("PM10_change_1h", "PM10_chg"), ("CO", "CO_level")]:
        wide = df.pivot_table(index=TIME_COL, columns=STATION_COL, values=col, aggfunc="first")
        wide = wide.reindex(columns=stations).reindex(df[TIME_COL].to_numpy())
        for s in stations:
            df[f"nb_{tag}_{s}"] = wide[s].to_numpy(dtype=np.float32)
            out.append(f"nb_{tag}_{s}")
    return out


def add_meteo_v2(df: pd.DataFrame) -> list[str]:
    g = _g(df)
    out = []
    df["rain_flag"] = (df["RAIN"] > 0).astype(np.int8)
    df["RAIN_roll_sum_6h"] = _roll(df, g["RAIN"].shift(1), 6, ["sum"])["sum"].to_numpy()
    df["WSPM_roll_mean_3h"] = _roll(df, g["WSPM"].shift(1), 3, ["mean"])["mean"].to_numpy()
    df["WSPM_x_ddd"] = df["WSPM"] * df["TEMP_DEWP_diff"]
    df["WSPM_x_PM10_vs_net"] = df["WSPM"] * df["PM10_vs_net_loo"]
    df["ddd_x_PM10_vs_net"] = df["TEMP_DEWP_diff"] * df["PM10_vs_net_loo"]
    df["WSPM_x_PM10_change_1h"] = df["WSPM"] * df["PM10_change_1h"]
    out += ["rain_flag", "RAIN_roll_sum_6h", "WSPM_roll_mean_3h", "WSPM_x_ddd",
            "WSPM_x_PM10_vs_net", "ddd_x_PM10_vs_net", "WSPM_x_PM10_change_1h"]  # fmt: skip
    return out


def add_deep(df: pd.DataFrame) -> list[str]:
    """Fine-fraction physics: what turns coarse PM10 into fine PM2.5.

    The f7 gain list is dominated by PM10, PM10 x CO and the city-wide PM10
    level, i.e. "PM10 corrected by combustion intensity and by the city". These
    columns give the tree those corrections explicitly: PM10 crossed with each
    combustion tracer and with humidity, combustion *per coarse particle*, and
    the city-level counterparts.
    """
    out = []
    pm = df["PM10"]
    for c in ["NO2", "SO2", "O3", "DEWP", "TEMP"]:
        df[f"PM10_x_{c}"] = pm * df[c]
        out.append(f"PM10_x_{c}")
    df["logPM10_x_logCO"] = df["log_PM10"] * df["log_CO"]
    df["sqrt_PM10_x_CO"] = np.sqrt((pm * df["CO"]).clip(lower=0))
    for c in ["CO", "NO2", "SO2"]:
        df[f"{c}_per_PM10"] = df[c] / pm.replace(0, np.nan)
        out.append(f"{c}_per_PM10")
    out += ["logPM10_x_logCO", "sqrt_PM10_x_CO"]
    # city-level counterparts and own-vs-city products
    df["netPM10med_x_CO"] = df["net_PM10_median"] * df["CO"]
    df["netPM10med_x_netCO"] = df["net_PM10_median"] * df["net_CO_loo_mean"]
    df["PM10_x_netPM10med"] = pm * df["net_PM10_median"]
    df["PM10_x_netCO"] = pm * df["net_CO_loo_mean"]
    out += ["netPM10med_x_CO", "netPM10med_x_netCO", "PM10_x_netPM10med", "PM10_x_netCO"]
    # PM10 x season / diurnal (fine fraction is higher in winter and at night)
    df["PM10_x_doy_cos"] = pm * df["doy_cos"]
    df["PM10_x_hour_cos"] = pm * df["hour_cos"]
    df["PM10_x_heating"] = pm * df["is_heating_season"]
    out += ["PM10_x_doy_cos", "PM10_x_hour_cos", "PM10_x_heating"]
    # lagged products (yesterday's combustion-adjusted level)
    g = _g(df)
    df["PM10_x_CO_lag1"] = g["PM10_x_CO"].shift(1)
    df["PM10_x_CO_lag3"] = g["PM10_x_CO"].shift(3)
    df["PM10_x_CO_change_1h"] = df["PM10_x_CO"] - df["PM10_x_CO_lag1"]
    out += ["PM10_x_CO_lag1", "PM10_x_CO_lag3", "PM10_x_CO_change_1h"]
    return out


def add_deep2(df: pd.DataFrame) -> list[str]:
    """Round 2, motivated by the residual audit (Findings/v3/08).

    (a) City-level x dispersion: the top feature (net_PM10_median) has no explicit
        interaction with the conditions that clear or trap pollution.
    (b) Per-station PM10 products: the residual audit found a 1.4x RMSE spread and
        a systematic (non-zero) bias per station -- station_cat alone is not fully
        capturing a station-specific fine fraction.
    (c) A robust PM10 denominator (own reading blended with the city median) for
        the ratio target, guarding against a single bad PM10 reading blowing up
        the fine-fraction target. See run_experiment.py target "ratio_robust".
    """
    out = []
    net = df["net_PM10_median"]
    for name, col in [("ddd", "TEMP_DEWP_diff"), ("WSPM", "WSPM"), ("wind_v", "wind_v")]:
        df[f"netPM10med_x_{name}"] = net * df[col]
        out.append(f"netPM10med_x_{name}")
    df["hour_x_heating"] = df["hour_cos"] * df["is_heating_season"]
    out.append("hour_x_heating")

    stations = sorted(df[STATION_COL].unique())
    for s in stations:
        name = f"PM10_x_station_{s}"
        df[name] = df["PM10"] * (df[STATION_COL] == s).astype(np.float32)
        out.append(name)

    # robust denominator for the ratio target: own PM10 blended with the network
    # median, so one bad reading cannot dominate the fine-fraction target.
    df["PM10_robust"] = 0.7 * df["PM10_filled"] + 0.3 * net.fillna(df["PM10_filled"])
    out.append("PM10_robust")
    return out


def add_humidity(df: pd.DataFrame) -> list[str]:
    """Magnus-formula RH from TEMP and DEWP (both C). Hygroscopic growth is a threshold effect."""
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


def add_episode(df: pd.DataFrame) -> list[str]:
    """Episode age and stagnation duration -- rapid rises are the largest residual."""
    g = _g(df)
    out = []
    for thr in (2.0, 4.0):
        df[f"hrs_since_wspm_ge{int(thr)}"] = _hours_since(df, df["WSPM"] >= thr)
        out.append(f"hrs_since_wspm_ge{int(thr)}")
    df["calm_run_hrs"] = _run_length(df, df["WSPM"] < 1.5)
    df["hrs_since_rain"] = _hours_since(df, df["RAIN"] > 0)
    out += ["calm_run_hrs", "hrs_since_rain"]
    df["hrs_since_pm10_lt50"] = _hours_since(df, df["PM10"] < 50)
    cross_up = (df["PM10"] >= 100) & (g["PM10"].shift(1) < 100)
    df["hrs_since_pm10_cross100"] = _hours_since(df, cross_up)
    out += ["hrs_since_pm10_lt50", "hrs_since_pm10_cross100"]
    df["pm10_rise_run"] = _run_length(df, df["PM10_change_1h"] > 0)
    df["pm10_fall_run"] = _run_length(df, df["PM10_change_1h"] < 0)
    net_chg = df["net_PM10_median"] - g["net_PM10_median"].shift(1)
    df["net_pm10_rise_run"] = _run_length(df, net_chg > 0)
    out += ["pm10_rise_run", "pm10_fall_run", "net_pm10_rise_run"]
    mx72 = _roll(df, g["PM10"].shift(1), 72, ["max"], min_periods=6)["max"].to_numpy()
    df["PM10_roll_max_72h"] = mx72
    df["PM10_over_max72"] = df["PM10"] / pd.Series(mx72, index=df.index).replace(0, np.nan)
    out += ["PM10_roll_max_72h", "PM10_over_max72"]
    return out


def add_coarse(df: pd.DataFrame) -> list[str]:
    """City-wide coarse-vs-combustion balance and its momentum (own-station ratios live in `deep`)."""
    g = _g(df)
    out = []
    for c in ["CO", "NO2", "SO2"]:
        own = f"{c}_per_PM10"
        df[f"net_{own}_median"] = _net_stat(df, own, "median")
        df[f"{own}_vs_net"] = df[own] - df[f"net_{own}_median"]
        out += [f"net_{own}_median", f"{own}_vs_net"]
    df["CO_per_PM10_lag1"] = g["CO_per_PM10"].shift(1)
    df["CO_per_PM10_change_1h"] = df["CO_per_PM10"] - df["CO_per_PM10_lag1"]
    df["CO_per_PM10_roll_mean_6h"] = _roll(df, g["CO_per_PM10"].shift(1), 6, ["mean"])["mean"].to_numpy()
    df["net_CO_per_PM10_change_1h"] = df["net_CO_per_PM10_median"] - g["net_CO_per_PM10_median"].shift(1)
    out += ["CO_per_PM10_lag1", "CO_per_PM10_change_1h", "CO_per_PM10_roll_mean_6h", "net_CO_per_PM10_change_1h"]
    df["PM10_vs_net_x_CO_per_PM10"] = df["PM10_vs_net_loo"] * df["CO_per_PM10"]
    df["net_PM10_x_net_CO_per_PM10"] = df["net_PM10_median"] * df["net_CO_per_PM10_median"]
    out += ["PM10_vs_net_x_CO_per_PM10", "net_PM10_x_net_CO_per_PM10"]
    return out


def add_memory(df: pd.DataFrame) -> list[str]:
    """72h and 168h baselines -- rolling windows previously stopped at 24h."""
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


SECTORS = {"N": ["N", "NNE", "NNW", "NE", "NW"], "E": ["E", "ENE", "ESE"],
           "S": ["S", "SSE", "SSW", "SE", "SW"], "W": ["W", "WNW", "WSW"]}  # fmt: skip


def add_transport(df: pd.DataFrame, fit_end: pd.Timestamp) -> list[str]:
    """Learned lead/lag adjacency: which neighbour's previous-hour change predicts mine, by wind sector.

    Adjacency is fit only on rows before `fit_end` (fold 0's training window), so it is
    identical whether or not later rows exist -- required for the causality test to pass.
    """
    stations = sorted(df[STATION_COL].unique())
    full_idx = pd.Index(sorted(df[TIME_COL].unique()))
    # dropna=False: pivot_table otherwise drops all-NaN rows differently per value column
    # (PM10 vs wd_cat), which desynchronises the two pivots' indices.
    wide = df.pivot_table(index=TIME_COL, columns=STATION_COL, values="PM10", aggfunc="first", dropna=False)
    wide = wide.reindex(index=full_idx, columns=stations)
    chg = wide.diff()
    chg_prev = chg.shift(1)
    wdw = df.pivot_table(index=TIME_COL, columns=STATION_COL, values="wd_cat", aggfunc="first", dropna=False)
    wdw = wdw.reindex(index=full_idx, columns=stations)
    sector_of = {d: sec for sec, ds in SECTORS.items() for d in ds}
    fit_mask = chg.index < fit_end
    lead: dict[str, dict[str, dict[str, float]]] = {}
    for s in stations:
        lead[s] = {}
        s_sec = wdw[s].astype(str).map(sector_of)
        for sec in [*SECTORS, "ALL"]:
            m = fit_mask if sec == "ALL" else (fit_mask & (s_sec == sec).to_numpy())
            scores = {}
            for n in stations:
                if n == s:
                    continue
                r = chg.loc[m, s].corr(chg_prev.loc[m, n])
                scores[n] = float(0.0 if pd.isna(r) else max(r, 0.0))
            lead[s][sec] = scores
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "transport_adjacency.json").write_text(json.dumps(lead, indent=1))

    idx = df[TIME_COL].to_numpy()
    lvl_now = wide.reindex(idx).to_numpy(np.float32)
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
    df["netPM10med_x_wind_u"] = df["net_PM10_median"] * df["wind_u"]
    return ["upwind_PM10", "upwind_PM10_change_1h", "leadw_PM10", "leadw_PM10_change_1h",
            "upwind_minus_own", "netPM10med_x_wind_u"]  # fmt: skip


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def build_panel_from(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    df = complete_grid(raw)
    assert _g(df)[TIME_COL].diff().dropna().eq(pd.Timedelta(hours=1)).all(), "grid is not hourly-contiguous"
    # diagnostic-only anchor: the previous hour's label. NaN after a gap, NaN on test.
    df[DIAG] = _g(df)[TARGET].shift(1)

    groups: dict[str, list[str]] = {}
    groups["raw"] = POLL + WEATHER
    cal = add_calendar(df)
    groups["calendar"] = [c for c in cal if c not in ("year", "day", "dayofyear", "weekofyear")]
    groups["wind"] = add_wind(df)
    groups["missing"] = add_missing_flags(df)
    groups["quality"] = add_quality(df)
    groups["physical"] = add_physical(df)
    groups["lags"] = add_lags(df, POLL, [1, 2, 3, 6, 12, 24])
    groups["weather_lags"] = add_lags(df, WEATHER, [1, 3, 6, 24])
    groups["rolling"] = add_rolling(df)
    groups["motion1"] = add_motion1(df)
    groups["motion2"] = add_motion2(df)
    groups["network_levels"] = add_network_levels(df)
    groups["network_momentum"] = add_network_momentum(df)
    groups["neighbors"] = add_neighbors(df)
    groups["meteo_v2"] = add_meteo_v2(df)
    groups["deep"] = add_deep(df)
    groups["deep2"] = add_deep2(df)
    groups["humidity"] = add_humidity(df)
    groups["episode"] = add_episode(df)
    groups["coarse"] = add_coarse(df)
    groups["memory"] = add_memory(df)
    groups["transport"] = add_transport(df, FOLD0_VAL_START)

    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    for g_ in groups.values():
        assert_no_pm25(g_)
    return df, groups


def build_panel() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    return build_panel_from(load_raw())


ORDER = ["lags", "rolling", "motion1", "motion2", "network_levels", "network_momentum", "neighbors"]


def feature_sets(groups: dict[str, list[str]]) -> dict[str, list[str]]:
    base = [c for g_ in ["raw", "calendar", "wind", "missing", "quality", "physical"] for c in groups[g_]]
    sets = {"f0_base": list(base)}
    cur = list(base)
    for i, g_ in enumerate(ORDER, start=1):
        cur = cur + groups[g_]
        sets[f"f{i}_{g_}"] = list(cur)
    sets["f8_full"] = cur + groups["meteo_v2"] + groups["weather_lags"]
    # f9: the "go deep" set = best step (f7) + fine-fraction physics
    sets["f9_deep"] = sets["f7_neighbors"] + groups.get("deep", [])
    # lean: f9 minus the groups the build-up and prune runs showed to be dead weight or harmful
    # (acceleration hurt; missing/quality flags had zero gain; rolling is KEPT — dropping it cost +0.13)
    dead = set(groups["motion2"]) | set(groups["missing"]) | set(groups["quality"])
    sets["f10_lean"] = [c for c in sets["f9_deep"] if c not in dead]
    # f11: f9 + round-2 features motivated by the residual audit (dispersion x city
    # level, station x PM10, robust denominator). PM10_robust is a helper column for
    # run_experiment's "ratio_robust" target, not meant to be a raw model feature —
    # excluded here; the target function reads it off the panel directly.
    deep2 = [c for c in groups.get("deep2", []) if c != "PM10_robust"]
    sets["f11_deep2"] = sets["f9_deep"] + deep2
    for cols in sets.values():
        assert_no_pm25(cols)
    return sets


def load_panel(rebuild: bool = False) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    gp = CACHE_DIR / "groups.json"
    if PANEL_PATH.exists() and gp.exists() and not rebuild:
        return pd.read_parquet(PANEL_PATH), json.loads(gp.read_text())
    df, groups = build_panel()
    tmp = PANEL_PATH.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, PANEL_PATH)
    tmp_g = gp.with_suffix(".json.tmp")
    tmp_g.write_text(json.dumps(groups, indent=1))
    os.replace(tmp_g, gp)
    return df, groups


if __name__ == "__main__":
    df, groups = load_panel(rebuild="--rebuild" in sys.argv)
    real = df[~df["is_pad"]]
    print(f"grid rows {len(df):,}  real rows {len(real):,}  train {(~real['is_test']).sum():,}  "
          f"test {real['is_test'].sum():,}")
    print("groups:", {k: len(v) for k, v in groups.items()})
    for name, cols in feature_sets(groups).items():
        print(f"  {name:22s} {len(cols):4d} columns")
