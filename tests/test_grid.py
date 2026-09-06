"""Absent station-hours must become explicit rows before anything is lagged."""

import pandas as pd

from src import config
from src.grid import ABSENT_FLAG, absent_run_lengths, blank_predictor_summary, complete_hourly_grid
from tests.conftest import HOLE_LENGTH, HOURS


def test_grid_is_complete_for_every_station(synthetic_raw):
    gridded = complete_hourly_grid(synthetic_raw)

    for _, group in gridded.groupby(config.STATION_COL):
        steps = group[config.TIME_COL].diff().dropna().dt.total_seconds() / 3600
        assert (steps == 1).all()
    assert len(gridded) == HOURS * 2


def test_absent_station_hours_are_flagged_and_blank_rows_are_not(synthetic_raw):
    gridded = complete_hourly_grid(synthetic_raw)

    assert gridded[ABSENT_FLAG].sum() == HOLE_LENGTH
    invented = gridded.loc[gridded[ABSENT_FLAG]]
    assert invented[config.ID_COL].isna().all()


def test_run_lengths_report_the_hole(synthetic_raw):
    runs = absent_run_lengths(complete_hourly_grid(synthetic_raw))

    assert len(runs) == 1
    assert runs.loc[0, "length_hours"] == HOLE_LENGTH


def test_blank_summary_counts_only_real_rows(synthetic_raw):
    summary = blank_predictor_summary(complete_hourly_grid(synthetic_raw))

    assert summary.loc[config.ANCHOR_COL, "blank"] == 1
    assert summary.loc[config.ANCHOR_COL, "rows"] == HOURS * 2 - HOLE_LENGTH
