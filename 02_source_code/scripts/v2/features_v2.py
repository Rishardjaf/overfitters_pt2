"""Causal feature panel for the v2 experiments.

Builds train and test on ONE timeline, reindexed to a complete hourly grid per
station, so that `shift(1)` is always genuinely `t-1` and never "the previous
surviving row". Every operation looks strictly backwards:

    shift(k) with k > 0, shift(1).rolling(w), same-hour groupby transforms.

Padding rows inserted by the grid carry NaN predictors and are dropped before
anything is fitted or scored. Building over train+test together is legitimate
(test rows need their own lags); only the direction of a window can leak.

Two named feature sets are exposed:

    jihad  a faithful re-implementation of notebooks/jihad (124 columns)
    v2     jihad plus the motion / regime / quality / meteorology additions
           described in "plans v2/03-feature-set-v2.md"

Usage:
    uv run python scripts/v2/features_v2.py          # build and cache the panel
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pm25.config import (  # noqa: E402
    ANCHOR,
    STATION_COL,
    TARGET,
    TEST_CSV,
    TIME_COL,
    TRAIN_CSV,
    WD_MISSING,
    WIND_DIRECTIONS,
)

CACHE_DIR = ROOT / "artifacts" / "v2"
PANEL_PATH = CACHE_DIR / "panel.parquet"

POLLUTANTS = ["current_PM2_5", "PM10", "SO2", "NO2", "CO", "O3"]
WEATHER = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
MEASUREMENTS = POLLUTANTS + WEATHER + ["wd"]

WIND_ANGLE = {d: i * 22.5 for i, d in enumerate(WIND_DIRECTIONS)}


# --------------------------------------------------------------------------- #
# Loading and the complete hourly grid
# --------------------------------------------------------------------------- #
def load_raw() -> pd.DataFrame:
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    train["is_test"] = False
    test["is_test"] = True
    test[TARGET] = np.nan
    panel = pd.concat([train, test], ignore_index=True)
    panel[TIME_COL] = pd.to_datetime(panel[TIME_COL])
    panel = panel.sort_values([STATION_COL, TIME_COL]).reset_index(drop=True)
    return panel


def complete_grid(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per station per hour from the first to the last timestamp.

    Inserted rows have is_pad=True. They exist so that positional shifts measure
    real elapsed hours.
    """
    hours = pd.date_range(panel[TIME_COL].min(), panel[TIME_COL].max(), freq="h")
    stations = sorted(panel[STATION_COL].unique())
    grid = pd.MultiIndex.from_product([stations, hours], names=[STATION_COL, TIME_COL])
    out = panel.set_index([STATION_COL, TIME_COL]).reindex(grid).reset_index()
    out["is_pad"] = out["id"].isna()
    out["is_test"] = out["is_test"].fillna(False).astype(bool)
    return out


# --------------------------------------------------------------------------- #
# Feature groups. Each function adds columns in place and returns the names.
# --------------------------------------------------------------------------- #
def _g(df: pd.DataFrame):
    return df.groupby(STATION_COL, sort=False, observed=True)


def add_calendar(df: pd.DataFrame) -> list[str]:
    ts = df[TIME_COL]
    df["year"] = ts.dt.year
    df["month"] = ts.dt.month
    df["day"] = ts.dt.day
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["dayofyear"] = ts.dt.dayofyear
    df["weekofyear"] = ts.dt.isocalendar().week.astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    # v2 additions
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)
    df["is_heating_season"] = df["month"].isin([9, 10, 11, 12, 1, 2]).astype(np.int8)
    return [
        "year", "month", "day", "hour", "dayofweek", "dayofyear", "weekofyear",
        "hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek_sin", "dayofweek_cos",
        "doy_sin", "doy_cos", "is_weekend", "is_heating_season",
    ]  # fmt: skip


def add_wind(df: pd.DataFrame) -> list[str]:
    angle = np.deg2rad(df["wd"].map(WIND_ANGLE))
    df["wd_sin"] = np.sin(angle)
    df["wd_cos"] = np.cos(angle)
    df["wind_u"] = df["WSPM"] * df["wd_sin"]
    df["wind_v"] = df["WSPM"] * df["wd_cos"]
    # categorical copy with blank as its own level (P6); pad rows stay NaN
    wd_cat = df["wd"].where(df["is_pad"], df["wd"].fillna(WD_MISSING))
    df["wd_cat"] = pd.Categorical(wd_cat, categories=[*WIND_DIRECTIONS, WD_MISSING])
    df["station_cat"] = pd.Categorical(df[STATION_COL])
    return ["wd_sin", "wd_cos", "wind_u", "wind_v", "wd_cat", "station_cat"]


def add_missing_flags(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in MEASUREMENTS:
        name = f"{c}_missing"
        df[name] = df[c].isna().astype(np.int8)
        cols.append(name)
    return cols


def add_quality_flags(df: pd.DataFrame) -> list[str]:
    df["pm25_floor_flag"] = (df[ANCHOR] == 3).astype(np.int8)
    df["pm10_equals_pm25"] = (df["PM10"] == df[ANCHOR]).astype(np.int8)
    df["pm25_gt_pm10"] = (df[ANCHOR] > df["PM10"]).astype(np.int8)
    df["o3_extreme"] = (df["O3"] > 400).astype(np.int8)
    df["co_floor"] = (df["CO"] == 100).astype(np.int8)
    df["co_ceiling"] = (df["CO"] == 10000).astype(np.int8)
    return [
        "pm25_floor_flag", "pm10_equals_pm25", "pm25_gt_pm10",
        "o3_extreme", "co_floor", "co_ceiling",
    ]  # fmt: skip


def add_physical(df: pd.DataFrame) -> list[str]:
    df["TEMP_DEWP_diff"] = df["TEMP"] - df["DEWP"]
    df["PM10_PM25_ratio"] = df["PM10"] / df[ANCHOR].replace(0, np.nan)
    df["NO2_CO_ratio"] = df["NO2"] / df["CO"].replace(0, np.nan)
    df["SO2_CO_ratio"] = df["SO2"] / df["CO"].replace(0, np.nan)
    df["PM25_over_PM10"] = df[ANCHOR] / df["PM10"].replace(0, np.nan)
    return ["TEMP_DEWP_diff", "PM10_PM25_ratio", "NO2_CO_ratio", "SO2_CO_ratio", "PM25_over_PM10"]


def add_lags(df: pd.DataFrame, cols: list[str], lags: list[int]) -> list[str]:
    g = _g(df)
    out = []
    for c in cols:
        for k in lags:
            name = f"{c}_lag_{k}h"
            df[name] = g[c].shift(k)
            out.append(name)
    return out


def add_changes(df: pd.DataFrame) -> list[str]:
    a = df[ANCHOR]
    out = []
    for k in [1, 3, 6, 24]:
        name = f"PM25_change_{k}h"
        df[name] = a - df[f"{ANCHOR}_lag_{k}h"]
        out.append(name)
    df["PM25_pct_change_1h"] = df["PM25_change_1h"] / df[f"{ANCHOR}_lag_1h"].replace(0, np.nan)
    df["PM25_pct_change_24h"] = df["PM25_change_24h"] / df[f"{ANCHOR}_lag_24h"].replace(0, np.nan)
    return out + ["PM25_pct_change_1h", "PM25_pct_change_24h"]


def add_rolling(df: pd.DataFrame, windows: list[int]) -> list[str]:
    """Trailing window statistics EXCLUDING the current hour (shift(1) first)."""
    g = _g(df)
    prev = g[ANCHOR].shift(1)
    out = []
    for w in windows:
        r = prev.groupby(df[STATION_COL], sort=False).rolling(w, min_periods=1)
        stats = r.agg(["mean", "std", "min", "max"]).reset_index(level=0, drop=True)
        stats = stats.sort_index()
        for s in ["mean", "std", "min", "max"]:
            name = f"PM25_roll_{s}_{w}h"
            df[name] = stats[s].to_numpy()
            out.append(name)
    return out


def add_availability(df: pd.DataFrame) -> list[str]:
    df["PM25_lag_1h_missing"] = df[f"{ANCHOR}_lag_1h"].isna().astype(np.int8)
    df["PM25_lag_24h_missing"] = df[f"{ANCHOR}_lag_24h"].isna().astype(np.int8)
    df["PM25_history_available"] = (
        df[[f"{ANCHOR}_lag_{k}h" for k in [1, 3, 6, 24]]].notna().sum(axis=1).astype(np.int8)
    )
    return ["PM25_lag_1h_missing", "PM25_lag_24h_missing", "PM25_history_available"]


def add_network(df: pd.DataFrame) -> list[str]:
    """Same-hour cross-station aggregates. Uses hour t only — no future rows."""
    g = df.groupby(TIME_COL, sort=False)[ANCHOR]
    df["network_PM25_mean"] = g.transform("mean")
    df["network_PM25_median"] = g.transform("median")
    df["network_PM25_std"] = g.transform("std")
    df["network_PM25_min"] = g.transform("min")
    df["network_PM25_max"] = g.transform("max")
    df["PM25_vs_network_mean"] = df[ANCHOR] - df["network_PM25_mean"]
    return [
        "network_PM25_mean", "network_PM25_median", "network_PM25_std",
        "network_PM25_min", "network_PM25_max", "PM25_vs_network_mean",
    ]  # fmt: skip


def add_interactions(df: pd.DataFrame) -> list[str]:
    a = df[ANCHOR]
    pairs = {
        "PM25_x_PM10": a * df["PM10"],
        "PM25_x_CO": a * df["CO"],
        "PM25_x_NO2": a * df["NO2"],
        "PM25_x_SO2": a * df["SO2"],
        "PM25_x_O3": a * df["O3"],
        "PM25_x_WSPM": a * df["WSPM"],
        "PM25_x_TEMP": a * df["TEMP"],
        "PM25_x_DEWP": a * df["DEWP"],
        "PM25_x_PRES": a * df["PRES"],
        "PM25_x_RAIN": a * df["RAIN"],
        "PM10_x_WSPM": df["PM10"] * df["WSPM"],
        "PM10_x_TEMP": df["PM10"] * df["TEMP"],
        "PM10_x_DEWP": df["PM10"] * df["DEWP"],
        "PM10_x_RAIN": df["PM10"] * df["RAIN"],
        "CO_x_NO2": df["CO"] * df["NO2"],
        "CO_x_SO2": df["CO"] * df["SO2"],
        "NO2_x_SO2": df["NO2"] * df["SO2"],
        "NO2_x_O3": df["NO2"] * df["O3"],
        "PM25_x_network_mean": a * df["network_PM25_mean"],
        "PM25_x_network_std": a * df["network_PM25_std"],
        "PM10_x_pm25_floor": df["PM10"] * df["pm25_floor_flag"],
        "CO_x_pm25_floor": df["CO"] * df["pm25_floor_flag"],
        "NO2_x_pm25_floor": df["NO2"] * df["pm25_floor_flag"],
        "WSPM_x_pm25_floor": df["WSPM"] * df["pm25_floor_flag"],
        "WSPM_x_network_mean": df["WSPM"] * df["network_PM25_mean"],
        "WSPM_x_network_std": df["WSPM"] * df["network_PM25_std"],
        "PM25_x_ddd": a * df["TEMP_DEWP_diff"],
        "PM10_x_ddd": df["PM10"] * df["TEMP_DEWP_diff"],
    }
    for k, v in pairs.items():
        df[k] = v
    return list(pairs)


# ----------------------------- v2 additions --------------------------------- #
def _loo_mean(df: pd.DataFrame, col: str) -> pd.Series:
    """Leave-one-station-out same-hour mean of `col`."""
    g = df.groupby(TIME_COL, sort=False)[col]
    total = g.transform("sum")
    count = g.transform("count")
    own = df[col]
    n_other = count - own.notna().astype(int)
    return (total - own.fillna(0)) / n_other.replace(0, np.nan)


def add_motion(df: pd.DataFrame) -> list[str]:
    g = _g(df)
    out = []

    # own acceleration (second difference)
    df["PM25_change_1h_lag1"] = g["PM25_change_1h"].shift(1)
    df["PM25_accel"] = df["PM25_change_1h"] - df["PM25_change_1h_lag1"]
    out += ["PM25_change_1h_lag1", "PM25_accel"]

    # co-moving pollutants: their own 1h change
    for c in ["PM10", "CO", "NO2", "SO2", "O3"]:
        name = f"{c}_change_1h"
        df[name] = df[c] - df[f"{c}_lag_1h"]
        out.append(name)

    # leave-one-station-out network momentum
    for k in [1, 3]:
        name = f"net_loo_change_{k}h_mean"
        df[name] = _loo_mean(df, f"PM25_change_{k}h")
        out.append(name)
    df["net_change_1h_median"] = df.groupby(TIME_COL, sort=False)["PM25_change_1h"].transform(
        "median"
    )
    out.append("net_change_1h_median")

    # how many OTHER stations are moving sharply right now
    rising = (df["PM25_change_1h"] > 10).astype(float).where(df["PM25_change_1h"].notna())
    falling = (df["PM25_change_1h"] < -10).astype(float).where(df["PM25_change_1h"].notna())
    df["_rising"] = rising
    df["_falling"] = falling
    for src, name in [("_rising", "net_n_rising"), ("_falling", "net_n_falling")]:
        g_t = df.groupby(TIME_COL, sort=False)[src]
        df[name] = g_t.transform("sum") - df[src].fillna(0)
        out.append(name)
    df.drop(columns=["_rising", "_falling"], inplace=True)

    # leave-one-out network level and this station's deviation from it, plus how
    # that deviation is changing (converging to or diverging from the network)
    df["net_loo_mean"] = _loo_mean(df, ANCHOR)
    df["PM25_vs_net_loo"] = df[ANCHOR] - df["net_loo_mean"]
    df["PM25_vs_net_loo_lag1"] = g["PM25_vs_net_loo"].shift(1)
    df["PM25_vs_net_loo_change_1h"] = df["PM25_vs_net_loo"] - df["PM25_vs_net_loo_lag1"]
    df["net_n_available"] = df.groupby(TIME_COL, sort=False)[ANCHOR].transform("count")
    out += ["net_loo_mean", "PM25_vs_net_loo", "PM25_vs_net_loo_lag1",
            "PM25_vs_net_loo_change_1h", "net_n_available"]  # fmt: skip

    # network level one hour ago and the network's own 1h change of the mean
    df["net_mean_lag1"] = g["network_PM25_mean"].shift(1)
    df["net_mean_change_1h"] = df["network_PM25_mean"] - df["net_mean_lag1"]
    out += ["net_mean_lag1", "net_mean_change_1h"]
    return out


def add_regime(df: pd.DataFrame) -> list[str]:
    g = _g(df)
    out = []
    prev_change = g["PM25_change_1h"].shift(1)
    for w in [6, 24]:
        r = prev_change.groupby(df[STATION_COL], sort=False).rolling(w, min_periods=2)
        name = f"PM25_change_roll_std_{w}h"
        df[name] = r.std().reset_index(level=0, drop=True).sort_index().to_numpy()
        out.append(name)
    rng = (df["PM25_roll_max_24h"] - df["PM25_roll_min_24h"]).replace(0, np.nan)
    df["PM25_pos_in_24h_range"] = (df[ANCHOR] - df["PM25_roll_min_24h"]) / rng
    df["PM25_rel_to_roll_mean_24h"] = df[ANCHOR] / df["PM25_roll_mean_24h"].replace(0, np.nan)
    df["PM25_minus_roll_mean_24h"] = df[ANCHOR] - df["PM25_roll_mean_24h"]
    df["PM25_minus_roll_mean_6h"] = df[ANCHOR] - df["PM25_roll_mean_6h"]
    out += ["PM25_pos_in_24h_range", "PM25_rel_to_roll_mean_24h",
            "PM25_minus_roll_mean_24h", "PM25_minus_roll_mean_6h"]  # fmt: skip
    return out


def _flat_run(df: pd.DataFrame, col: str) -> pd.Series:
    """Consecutive hours (ending at t, inclusive) the value has been unchanged."""
    g = _g(df)
    changed = (df[col] != g[col].shift(1)) | df[col].isna()
    run_id = changed.groupby(df[STATION_COL], sort=False).cumsum()
    run_pos = df.groupby([STATION_COL, run_id], sort=False).cumcount()
    return run_pos.where(df[col].notna(), 0).astype(np.int16)


def add_quality_v2(df: pd.DataFrame) -> list[str]:
    out = []
    for c in ["current_PM2_5", "PM10", "CO"]:
        name = f"{c}_flat_run"
        df[name] = _flat_run(df, c)
        out.append(name)

    # hours since the previous REAL row (gap length); 1 means no gap
    # ffill is a *forward* fill: it carries past values forward, never future ones
    real_ts = df[TIME_COL].where(~df["is_pad"])
    last_real_before = real_ts.groupby(df[STATION_COL], sort=False).shift(1)
    last_real_before = last_real_before.groupby(df[STATION_COL], sort=False).ffill()
    df["gap_prev_hours"] = ((df[TIME_COL] - last_real_before) / pd.Timedelta(hours=1)).astype(
        "float32"
    )

    # hours since the anchor was last observed (0 when present)
    anchor_ts = df[TIME_COL].where(df[ANCHOR].notna())
    anchor_ts = anchor_ts.groupby(df[STATION_COL], sort=False).ffill()
    df["anchor_age_hours"] = ((df[TIME_COL] - anchor_ts) / pd.Timedelta(hours=1)).astype("float32")
    out += ["gap_prev_hours", "anchor_age_hours"]
    return out


def add_meteorology_v2(df: pd.DataFrame) -> list[str]:
    g = _g(df)
    df["PRES_tend_3h"] = df["PRES"] - g["PRES"].shift(3)
    df["PRES_tend_24h"] = df["PRES"] - g["PRES"].shift(24)
    df["ddd_change_3h"] = df["TEMP_DEWP_diff"] - g["TEMP_DEWP_diff"].shift(3)
    df["WSPM_change_1h"] = df["WSPM"] - g["WSPM"].shift(1)
    df["WSPM_roll_mean_3h"] = (
        g["WSPM"]
        .shift(1)
        .groupby(df[STATION_COL], sort=False)
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .sort_index()
        .to_numpy()
    )
    df["rain_flag"] = (df["RAIN"] > 0).astype(np.int8)
    df["RAIN_roll_sum_6h"] = (
        g["RAIN"]
        .shift(1)
        .groupby(df[STATION_COL], sort=False)
        .rolling(6, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .sort_index()
        .to_numpy()
    )
    df["WSPM_x_ddd"] = df["WSPM"] * df["TEMP_DEWP_diff"]
    df["WSPM_x_PM25_vs_net"] = df["WSPM"] * df["PM25_vs_net_loo"]
    return [
        "PRES_tend_3h", "PRES_tend_24h", "ddd_change_3h", "WSPM_change_1h",
        "WSPM_roll_mean_3h", "rain_flag", "RAIN_roll_sum_6h", "WSPM_x_ddd", "WSPM_x_PM25_vs_net",
    ]  # fmt: skip


def add_neighbors(df: pd.DataFrame) -> list[str]:
    """Every other station's same-hour level and 1h change as its own column.

    A single network mean hides *which* neighbour is moving. Pollution fronts
    cross the city in a consistent direction for a given wind, so the tree can
    learn, per station, which other stations lead it. Same hour t only.
    """
    out = []
    stations = sorted(df[STATION_COL].unique())
    for col, tag in [(ANCHOR, "level"), ("PM25_change_1h", "chg")]:
        wide = df.pivot_table(index=TIME_COL, columns=STATION_COL, values=col, aggfunc="first")
        wide = wide.reindex(columns=stations)
        aligned = wide.reindex(df[TIME_COL].to_numpy())
        for s in stations:
            name = f"nb_{tag}_{s}"
            df[name] = aligned[s].to_numpy(dtype=np.float32)
            out.append(name)
    # wind x motion: does the wind carry the movement through or disperse it?
    df["WSPM_x_change_1h"] = df["WSPM"] * df["PM25_change_1h"]
    df["WSPM_x_net_change_1h"] = df["WSPM"] * df["net_loo_change_1h_mean"]
    df["ddd_x_PM25_vs_net"] = df["TEMP_DEWP_diff"] * df["PM25_vs_net_loo"]
    df["net_accel"] = df["net_mean_change_1h"] - _g(df)["net_mean_change_1h"].shift(1)
    out += ["WSPM_x_change_1h", "WSPM_x_net_change_1h", "ddd_x_PM25_vs_net", "net_accel"]
    return out


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def build_panel() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    return build_panel_from(load_raw())


def build_panel_from(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    df = complete_grid(raw)
    groups: dict[str, list[str]] = {}
    groups["raw"] = POLLUTANTS + WEATHER
    groups["calendar"] = add_calendar(df)
    groups["wind"] = add_wind(df)
    groups["missing"] = add_missing_flags(df)
    groups["quality"] = add_quality_flags(df)
    groups["physical"] = add_physical(df)
    groups["pm_lags"] = add_lags(df, [ANCHOR], [1, 2, 3, 6, 12, 24, 48, 72])
    groups["pollutant_lags"] = add_lags(df, ["PM10", "SO2", "NO2", "CO", "O3"], [1, 3, 6, 24])
    groups["weather_lags"] = add_lags(df, WEATHER, [1, 3, 6, 24])
    groups["changes"] = add_changes(df)
    groups["rolling"] = add_rolling(df, [3, 6, 12, 24, 48])
    groups["availability"] = add_availability(df)
    groups["network"] = add_network(df)
    groups["interactions"] = add_interactions(df)
    # v2 additions
    groups["motion"] = add_motion(df)
    groups["regime"] = add_regime(df)
    groups["quality_v2"] = add_quality_v2(df)
    groups["meteorology_v2"] = add_meteorology_v2(df)
    groups["neighbors"] = add_neighbors(df)

    # compact dtypes
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    df[TARGET] = df[TARGET].astype(np.float32)
    df[ANCHOR] = df[ANCHOR].astype(np.float32)
    return df, groups


#: Columns the jihad notebook dropped in its final selection step.
JIHAD_DROP = [
    "PM25_x_PRES", "day", "dayofyear", "weekofyear",
    "PRES", "PRES_lag_1h", "PRES_lag_3h", "PRES_lag_6h",
    "TEMP_lag_1h", "TEMP_lag_3h", "TEMP_lag_6h",
    "DEWP_lag_1h", "DEWP_lag_3h", "DEWP_lag_6h",
    "network_PM25_median", "network_PM25_min", "network_PM25_max",
    "current_PM2_5_lag_72h",
    "PM25_roll_max_6h", "PM25_roll_max_12h", "PM25_roll_max_24h", "PM25_roll_max_48h",
    "O3_lag_1h", "O3_lag_3h", "O3_lag_6h", "O3_lag_24h",
    "RAIN_lag_1h", "RAIN_lag_3h", "RAIN_lag_6h", "RAIN_lag_24h",
    "PM25_x_RAIN", "PM10_x_RAIN", "SO2_CO_ratio",
    "TEMP_missing", "PRES_missing", "DEWP_missing", "RAIN_missing",
]  # fmt: skip

V2_ONLY_CALENDAR = ["doy_sin", "doy_cos", "is_weekend", "is_heating_season"]
V2_ONLY_WIND = ["wind_u", "wind_v", "wd_cat", "station_cat"]


def feature_sets(groups: dict[str, list[str]]) -> dict[str, list[str]]:
    jihad_groups = [
        "raw", "calendar", "wind", "missing", "quality", "physical", "pm_lags",
        "pollutant_lags", "weather_lags", "changes", "rolling", "availability",
        "network", "interactions",
    ]  # fmt: skip
    jihad_all = [c for g in jihad_groups for c in groups[g]]
    jihad_all = [c for c in jihad_all if c not in V2_ONLY_CALENDAR and c not in V2_ONLY_WIND]
    jihad = [c for c in jihad_all if c not in JIHAD_DROP] + ["station_cat"]

    v2_extra = [c for g in ["motion", "regime", "quality_v2", "meteorology_v2"] for c in groups[g]]
    v2 = jihad + V2_ONLY_CALENDAR + ["wind_u", "wind_v", "wd_cat"] + v2_extra
    # v2 drops the raw calendar columns a tree extrapolates badly on
    v2 = [c for c in v2 if c not in ["year"]]

    # lean: v2 without the level x level products and the ratio columns
    lean = [c for c in v2 if c not in groups["interactions"] and c not in
            ["PM10_PM25_ratio", "NO2_CO_ratio", "PM25_over_PM10"]]  # fmt: skip
    # lean plus per-station neighbour columns and wind x motion interactions
    lean_nb = lean + groups.get("neighbors", [])
    # lean_nb minus the groups the ablation showed to be dead weight
    slim_nb = [
        c for c in lean_nb if c not in groups["weather_lags"] and c not in groups["pollutant_lags"]
    ]
    return {"jihad": jihad, "v2": v2, "lean": lean, "lean_nb": lean_nb, "slim_nb": slim_nb}


def load_panel(rebuild: bool = False) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    groups_path = CACHE_DIR / "groups.json"
    if PANEL_PATH.exists() and groups_path.exists() and not rebuild:
        import json

        df = pd.read_parquet(PANEL_PATH)
        groups = json.loads(groups_path.read_text())
        return df, groups
    df, groups = build_panel()
    import json
    import os

    # write-then-rename so a concurrent reader never sees a half-written file
    tmp = PANEL_PATH.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, PANEL_PATH)
    tmp_g = groups_path.with_suffix(".json.tmp")
    tmp_g.write_text(json.dumps(groups, indent=1))
    os.replace(tmp_g, groups_path)
    return df, groups


if __name__ == "__main__":
    df, groups = load_panel(rebuild="--rebuild" in sys.argv)
    sets = feature_sets(groups)
    real = df[~df["is_pad"]]
    print(
        f"grid rows {len(df):,}  real rows {len(real):,}  "
        f"train {(~real['is_test']).sum():,}  test {real['is_test'].sum():,}"
    )
    for name, cols in sets.items():
        print(f"feature set {name!r}: {len(cols)} columns")
    print("groups:", {k: len(v) for k, v in groups.items()})
