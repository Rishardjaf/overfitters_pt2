"""The paired day-block bootstrap must be honest about what it can distinguish.

Station-hours are serially correlated — an episode spans many hours across all
twelve stations — so the bootstrap resamples whole days. These tests pin the
behaviour on synthetic data where the right answer is known.
"""

import numpy as np
import pandas as pd
import pytest

from src import uncertainty


def _synthetic(days: int = 60, rows_per_day: int = 24, seed: int = 7):
    """Actual values plus two prediction vectors, one clearly better."""
    rng = np.random.default_rng(seed)
    n = days * rows_per_day
    timestamps = pd.Series(
        pd.date_range("2015-01-01", periods=n, freq="h"), name="ts"
    )
    actual = pd.Series(50 + 20 * rng.standard_normal(n)).clip(lower=0)
    good = actual + 2 * rng.standard_normal(n)
    bad = actual + 10 * rng.standard_normal(n)
    return actual, good, bad, timestamps


def test_model_compared_with_itself_gives_a_zero_interval():
    actual, good, _, timestamps = _synthetic()

    result = uncertainty.compare_rmse(actual, good, good, timestamps)

    assert result.delta_rmse == 0.0
    assert result.lower == 0.0
    assert result.upper == 0.0
    assert not result.is_conclusive


def test_clearly_better_variant_gets_an_interval_entirely_below_zero():
    actual, good, bad, timestamps = _synthetic()

    result = uncertainty.compare_rmse(actual, bad, good, timestamps)

    assert result.delta_rmse < 0
    assert result.upper < 0
    assert result.is_conclusive


def test_noise_level_differences_are_not_called_conclusive():
    # Two models with the same error scale but independent noise: the true
    # RMSE difference is zero, so the interval must straddle it.
    rng = np.random.default_rng(11)
    actual, _, _, timestamps = _synthetic()
    first = actual + 8 * rng.standard_normal(len(actual))
    second = actual + 8 * rng.standard_normal(len(actual))

    result = uncertainty.compare_rmse(actual, first, second, timestamps)

    assert result.lower < 0 < result.upper
    assert not result.is_conclusive


def test_same_seed_reproduces_the_interval_and_counts_the_days():
    actual, good, bad, timestamps = _synthetic(days=45)

    first = uncertainty.compare_rmse(actual, bad, good, timestamps)
    second = uncertainty.compare_rmse(actual, bad, good, timestamps)

    assert (first.lower, first.upper) == (second.lower, second.upper)
    assert first.blocks == 45


def test_mismatched_lengths_are_rejected():
    actual, good, _, timestamps = _synthetic()

    with pytest.raises(ValueError, match="length"):
        uncertainty.compare_rmse(actual, good.iloc[:-1], good, timestamps)


def test_blank_values_are_rejected_rather_than_silently_dropped():
    actual, good, bad, timestamps = _synthetic()
    with_blank = good.copy()
    with_blank.iloc[3] = np.nan

    with pytest.raises(ValueError, match="blank"):
        uncertainty.compare_rmse(actual, with_blank, bad, timestamps)
