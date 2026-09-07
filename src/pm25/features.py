"""Causal feature engineering.

Not yet implemented — see the roadmap in PROJECT.md.

EVERY function in this module must be causal: a feature for hour `t` may use
information at or before `t`, never after. Concretely, within this file:

    allowed    .shift(k) for k > 0, .rolling(k).agg(), .expanding().agg(),
               cross-station values at the SAME hour t
    forbidden  .shift(-k), rolling(center=True), .bfill(), .interpolate()

`make audit` greps this module for those forbidden operations and fails the
build. Do not work around it.

Planned feature groups (see PROJECT.md section 4):

  raw          all pollutants and weather at hour t
  lags         current_PM2_5 at t-1, t-2, t-3, t-6, t-12, t-24; same for
               PM10, CO, NO2
  deltas       first and second differences of PM2.5 — rate of change is the
               strongest single signal for the next hour
  rolling      trailing 3/6/12/24h mean, std, min, max, and the current value's
               position within that window
  ratios       PM2_5/PM10 (fine fraction), NO2/CO
  meteorology  dew-point depression (TEMP - DEWP), wind u/v from wd + WSPM,
               pressure tendency over 3h and 24h
  ventilation  wind speed x concentration interactions — strong wind after a
               still period is the dominant clearing mechanism
  calendar     sin/cos of hour and day-of-year, day-of-week, is-weekend.
               NOT raw year: test years lie outside the training range
  network      cross-station mean/median PM2.5 at hour t and this station's
               deviation from it — pollution fronts move across the network
"""

from __future__ import annotations

import pandas as pd


def add_lags(df: pd.DataFrame, cols: list[str], lags: list[int]) -> pd.DataFrame:
    """Backward shifts within each station, on the complete hourly grid."""
    raise NotImplementedError


def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """First and second differences of the pollutant series."""
    raise NotImplementedError


def add_rolling(df: pd.DataFrame, cols: list[str], windows: list[int]) -> pd.DataFrame:
    """Trailing rolling statistics. Never centred."""
    raise NotImplementedError


def add_meteorology(df: pd.DataFrame) -> pd.DataFrame:
    """Dew-point depression, wind vector components, pressure tendency."""
    raise NotImplementedError


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical time encodings. Deliberately excludes raw `year`."""
    raise NotImplementedError


def add_network(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-station aggregates at the same hour, and deviation from them."""
    raise NotImplementedError


def add_missing_indicators(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Explicit `*_is_missing` flags — outages may coincide with extremes."""
    raise NotImplementedError


def build_features(panel: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Compose the feature groups named in a config's `features` block."""
    raise NotImplementedError
