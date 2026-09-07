"""Guards against the two ways this project can silently cheat.

These tests are the safety net behind `make audit`. They are written before the
pipeline exists so that the pipeline has to satisfy them from its first commit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pm25.audit import scan_dirs

ROOT = Path(__file__).resolve().parents[1]


def test_no_future_looking_operations_in_package():
    """No negative shifts, centred windows, or backfills in real code."""
    violations = scan_dirs([ROOT / "src", ROOT / "scripts", ROOT / "notebooks"])
    assert not violations, "future information in the feature pipeline:\n" + "\n".join(violations)


@pytest.mark.skip(reason="enable once validation.time_folds is implemented")
def test_folds_never_train_on_the_future():
    """Every training timestamp must precede its fold's validation window."""
    raise NotImplementedError


@pytest.mark.skip(reason="enable once features.build_features is implemented")
def test_features_are_stable_under_future_truncation():
    """Features for hour t must be identical whether or not rows after t exist.

    The strongest test available: build features on the full panel, then rebuild
    on a panel truncated at t, and assert the row for t is unchanged. Any feature
    that peeks forward fails this.
    """
    raise NotImplementedError
