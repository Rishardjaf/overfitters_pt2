"""Every rung must produce a usable prediction, and the ladder must be honest.

The rungs are exercised with deliberately tiny hyper-parameters — the question
here is whether the contract holds, not whether the model is any good.
"""

import json

import pandas as pd
import pytest

from src import config, validation
from src.experiments import registry, report
from src.experiments.runner import run
from src.models import pm10_anchor
from src.pipeline import build_frame
from tests.conftest import LABELLED_BLOCKS

@pytest.fixture
def built(synthetic_labelled) -> pd.DataFrame:
    return build_frame(synthetic_labelled)


@pytest.fixture
def fixture_blocks(monkeypatch):
    """Point the real fold builder at the fixture's own season-matched block.

    Patching the block list rather than `season_matched_folds` itself keeps the
    production fold logic — including its 10,000-row training floor — under test.
    """
    monkeypatch.setattr(config, "VALIDATION_BLOCKS", LABELLED_BLOCKS)


@pytest.fixture
def fold(built, fixture_blocks) -> validation.Fold:
    return validation.season_matched_folds(built)[0]


@pytest.fixture
def split(built, fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    return built.loc[fold.train_mask], built.loc[fold.holdout_mask]


def _labelled(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.dropna(subset=[config.TARGET_COL])


# --- the rung contract ------------------------------------------------------


@pytest.mark.parametrize(
    "rung", [r for r in registry.RUNGS if not r.is_baseline], ids=lambda r: r.algorithm
)
def test_rung_predicts_every_holdout_row(split, rung):
    """Aligned to the holdout, complete, and never a negative concentration."""
    train, holdout = split

    predictions = rung.adapter(train, holdout).predictions

    assert predictions.index.equals(holdout.index)
    assert predictions.notna().all()
    assert (predictions >= 0).all()


def test_the_anchor_baseline_is_fitted_on_training_rows_only(split):
    """The line and the blank-anchor median must be a function of the train rows.

    `src.leakage` cannot catch this: a statistic fitted across the holdout is
    train/test contamination, not a lookahead within the frame, so nothing about
    the feature values changes when the future is truncated away. Only a test
    like this one pins it.
    """
    train, holdout = split

    on_train = pm10_anchor.fit(train)
    on_holdout = pm10_anchor.fit(holdout)

    assert (on_train.slope, on_train.intercept) != (on_holdout.slope, on_holdout.intercept)
    assert on_train.anchor_median == pytest.approx(
        train.loc[
            train[config.ANCHOR_COL].notna() & train[config.TARGET_COL].notna(),
            config.ANCHOR_COL,
        ].median()
    )


def test_the_delta_rung_is_anchored_and_moves_with_the_anchor(split):
    """Shifting PM10 must move the prediction — otherwise the offset is inert."""
    from src.models import delta_lgbm

    train, holdout = split
    fitted = delta_lgbm.fit(train, {"num_leaves": 7})
    shifted = holdout.assign(**{config.ANCHOR_COL: holdout[config.ANCHOR_COL] + 50})

    moved = delta_lgbm.predict(fitted, shifted) - delta_lgbm.predict(fitted, holdout)

    assert (moved > 0).mean() > 0.9


def test_dense_rungs_never_see_a_missing_value(split):
    from src.models import dense_frame

    train, holdout = split
    prepared = dense_frame.prepare(train)

    assert not prepared.features.isna().to_numpy().any()
    assert not dense_frame.transform(prepared, holdout).isna().to_numpy().any()


def test_median_fill_is_fitted_on_training_rows_only(split):
    """The fitted medians must be a function of the training rows and nothing else.

    Fitting them on the holdout would leak that block's distribution into the
    transform, which is why they are a fitted object rather than a function of
    both frames ([ADR-0003](docs/adr/0003-per-rung-imputation.md)).
    """
    from src import impute
    from src.models import dense_frame

    train, holdout = split
    prepared = dense_frame.prepare(train)

    pd.testing.assert_frame_equal(prepared.medians, impute.fit_median_fill(train))
    assert not prepared.medians.equals(impute.fit_median_fill(holdout))
    assert prepared.rows <= len(train)


# --- the registry -----------------------------------------------------------


def test_every_rung_is_addressable_and_uniquely_numbered():
    numbers = [rung.number for rung in registry.RUNGS]

    assert numbers == sorted(set(numbers))
    for rung in registry.RUNGS:
        assert registry.resolve(rung.slug) is rung
        assert registry.resolve(rung.algorithm) is rung


def test_exactly_one_baseline_and_it_is_first():
    baselines = [rung for rung in registry.RUNGS if rung.is_baseline]

    assert len(baselines) == 1
    assert baselines[0] is registry.RUNGS[0]
    assert registry.baseline() is baselines[0]


def test_an_unknown_rung_lists_the_known_ones():
    with pytest.raises(KeyError, match="01_baseline_pm10_anchor"):
        registry.resolve("gradient_descent_by_hand")


# --- the runner -------------------------------------------------------------


@pytest.fixture
def stub_reference_ids(built, monkeypatch):
    """The submission's reference id set comes from the fixture, not data/test.csv."""
    monkeypatch.setattr(
        "src.experiments.runner.load.load_test",
        lambda: built.dropna(subset=[config.ID_COL]),
    )


@pytest.fixture
def baseline_outcome(built, fixture_blocks, stub_reference_ids, tmp_path):
    """Run the baseline rung against the fixture rather than the real data."""
    return run(registry.baseline(), built, built, tmp_path)


def test_the_baseline_scores_zero_against_itself(baseline_outcome):
    """The baseline is the anchor, so its improvement over the anchor is nil."""
    primary = baseline_outcome.primary

    assert primary.rmse == pytest.approx(primary.baseline_rmse)
    assert primary.rmse_vs_baseline_pct == pytest.approx(0.0)


def test_the_runner_scores_only_labelled_rows(baseline_outcome, built, fold):
    expected = len(_labelled(built.loc[fold.holdout_mask]))

    assert baseline_outcome.primary.holdout_rows == expected


def test_a_losing_rung_is_recorded_as_losing(
    built, fixture_blocks, stub_reference_ids, tmp_path
):
    """A rung that cannot beat the anchor is a finding, not something to drop."""
    hopeless = registry.Rung(
        slug="99_hopeless",
        algorithm="hopeless",
        label="TEST 99",
        summary="Predicts a constant.",
        adapter=lambda train, holdout: registry.RungResult(
            pd.Series(1.0, index=holdout.index)
        ),
    )

    outcome = run(hopeless, built, built, tmp_path)

    assert not outcome.primary.beats_baseline
    assert outcome.primary.rmse_vs_baseline_pct < 0
    assert any("does not beat the PM10 anchor" in warning for warning in outcome.warnings)


def test_empty_predictions_fall_back_to_the_anchor_and_are_counted(
    built, fixture_blocks, stub_reference_ids, tmp_path
):
    """A hole in a submission is rejected by the grader, so it is filled and reported."""
    gappy = registry.Rung(
        slug="98_gappy",
        algorithm="gappy",
        label="TEST 98",
        summary="Declines to predict the first row.",
        adapter=lambda train, holdout: registry.RungResult(
            pd.Series(5.0, index=holdout.index).mask(
                holdout.index == holdout.index[0]
            )
        ),
    )

    outcome = run(gappy, built, built, tmp_path)

    assert outcome.gap_filled_predictions == 1
    assert any("fell back to the PM10 anchor" in warning for warning in outcome.warnings)


def test_the_runner_writes_a_valid_submission(baseline_outcome, tmp_path, built):
    written = pd.read_csv(tmp_path / "submission.csv")

    assert len(written) == len(_labelled(built.dropna(subset=[config.ID_COL])))
    assert written[config.ID_COL].is_unique
    assert written["PM2_5_next_hour"].notna().all()


# --- the report -------------------------------------------------------------


def test_metrics_json_round_trips(baseline_outcome, tmp_path):
    path = report.write_metrics(baseline_outcome, tmp_path)
    metrics = json.loads(path.read_text())

    assert metrics["test"] == "01_baseline_pm10_anchor"
    assert metrics["is_baseline"] is True
    assert metrics["rmse"] > 0
    assert len(metrics["folds"]) == len(baseline_outcome.fold_scores)


def test_results_markdown_names_the_baseline(baseline_outcome, tmp_path):
    body = report.write_results(baseline_outcome, tmp_path, []).read_text()

    assert "BASELINE TEST" in body
    assert "baseline test" in body


def test_the_ladder_refuses_to_invent_an_empty_table(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run a test first"):
        report.write_ladder(tmp_path)


def test_the_ladder_flags_disagreeing_metrics(tmp_path):
    """RMSE and MAE can pick different winners; the ladder must not hide that."""
    entries = [
        {"test": "01_baseline_pm10_anchor", "algorithm": "pm10_anchor", "rmse": 40.7,
         "mae": 10.7, "is_baseline": True, "beats_baseline": False, "fit_seconds": 0.0,
         "primary_fold": "2015-09_to_2016-03"},
        {"test": "02_ridge", "algorithm": "ridge", "rmse": 20.8, "mae": 10.3,
         "is_baseline": False, "beats_baseline": True, "fit_seconds": 2.7,
         "primary_fold": "2015-09_to_2016-03"},
        {"test": "03_lightgbm", "algorithm": "lightgbm", "rmse": 21.6, "mae": 9.7,
         "is_baseline": False, "beats_baseline": True, "fit_seconds": 28.5,
         "primary_fold": "2015-09_to_2016-03"},
    ]
    for entry in entries:
        directory = tmp_path / entry["test"]
        directory.mkdir()
        (directory / report.METRICS_FILE).write_text(json.dumps(entry))

    body = report.write_ladder(tmp_path).read_text()

    assert "two metrics disagree" in body
    assert "`ridge` has the lowest RMSE" in body
    assert "`lightgbm` has the lowest MAE" in body


def test_the_ladder_lists_pending_tests(baseline_outcome, tmp_path):
    slug = baseline_outcome.rung.slug
    report.write_metrics(baseline_outcome, tmp_path / slug)

    body = report.write_ladder(tmp_path).read_text()

    assert slug in body
    assert "Not yet run" in body
    assert "03_lightgbm" in body
