"""Run one numbered test, or the whole ladder.

    python -m src.experiments 01_baseline_pm10_anchor
    python -m src.experiments lightgbm
    python -m src.experiments all --all-folds
"""

import argparse
import sys

from src import cache, config
from src.experiments import figures, report
from src.experiments.registry import RUNGS, resolve
from src.experiments.runner import run


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m src.experiments")
    parser.add_argument(
        "target",
        help="A test slug (01_baseline_pm10_anchor), an algorithm name (lightgbm), or 'all'.",
    )
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="Score both season-matched blocks instead of the primary one only.",
    )
    return parser.parse_args(argv)


def _run_one(rung, train_features, test_features, all_folds: bool) -> None:
    directory = config.ML_ALGORITHMS_DIR / rung.slug
    # Flushed as it goes: the forest and SARIMAX run for minutes, and a silent
    # pipe for that long is indistinguishable from a hang.
    print(f"\n{rung.label} — {rung.algorithm}", flush=True)
    print(f"  {rung.summary}", flush=True)

    outcome = run(rung, train_features, test_features, directory, all_folds=all_folds)
    figure_paths = figures.draw_all(outcome, directory)
    report.write_metrics(outcome, directory)
    report.write_results(outcome, directory, figure_paths)

    primary = outcome.primary
    verdict = (
        "baseline"
        if rung.is_baseline
        else f"{primary.rmse_vs_baseline_pct:+.2f}% vs baseline"
    )
    print(
        f"  RMSE {primary.rmse:.3f} | MAE {primary.mae:.3f} | {verdict} "
        f"| {outcome.total_seconds:.0f}s | {outcome.submission_rows:,} submission rows",
        flush=True,
    )
    for warning in outcome.warnings:
        print(f"  ! {warning}", flush=True)


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_args(argv if argv is not None else sys.argv[1:])
    selected = RUNGS if arguments.target == "all" else (resolve(arguments.target),)

    print("Loading cached feature frames...")
    train_features = cache.load_train_features()
    test_features = cache.load_test_features()
    print(f"  train {train_features.shape} | test {test_features.shape}")

    for rung in selected:
        _run_one(rung, train_features, test_features, arguments.all_folds)

    print(f"\nLadder written to {report.write_ladder()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
