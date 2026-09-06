"""Write each test's record and regenerate the ladder that spans them.

Three artefacts: `metrics.json` so a later run can be compared numerically,
`RESULTS.md` so a human can read what happened without running anything, and
`ML_Algorithms/README.md` rebuilt from every metrics.json on disk so the ladder
never disagrees with the tests it summarises.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src import config, evaluate
from src.experiments.registry import RUNGS, baseline

METRICS_FILE = "metrics.json"
RESULTS_FILE = "RESULTS.md"
LADDER_FILE = "README.md"
ABLATIONS_FILE = "ABLATIONS.md"


def markdown_table(frame: pd.DataFrame) -> str:
    """A GitHub-flavoured table. Avoids a tabulate dependency for a handful of rows."""
    header = list(frame.columns)
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join(lines)


def metrics_of(outcome) -> dict:
    """The machine-readable record of one run."""
    primary = outcome.primary
    return {
        "test": outcome.rung.slug,
        "algorithm": outcome.rung.algorithm,
        "label": outcome.rung.label,
        "is_baseline": outcome.rung.is_baseline,
        "summary": outcome.rung.summary,
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "primary_fold": primary.fold,
        "rmse": round(primary.rmse, 4),
        "mae": round(primary.mae, 4),
        "baseline_rmse": round(primary.baseline_rmse, 4),
        "baseline_mae": round(primary.baseline_mae, 4),
        "rmse_vs_baseline_pct": primary.rmse_vs_baseline_pct,
        "beats_baseline": primary.beats_baseline,
        "fit_seconds": primary.fit_seconds,
        "total_seconds": outcome.total_seconds,
        "submission_rows": outcome.submission_rows,
        "gap_filled_predictions": outcome.gap_filled_predictions,
        "warnings": outcome.warnings,
        "folds": [vars(score) for score in outcome.fold_scores],
    }


def write_metrics(outcome, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / METRICS_FILE
    path.write_text(json.dumps(metrics_of(outcome), indent=2) + "\n")
    return path


def _fold_table(outcome) -> str:
    return markdown_table(
        pd.DataFrame(
            [
                {
                    "Fold": score.fold,
                    "Train rows": f"{score.train_rows:,}",
                    "Holdout rows": f"{score.holdout_rows:,}",
                    "RMSE": f"{score.rmse:.3f}",
                    "MAE": f"{score.mae:.3f}",
                    "Persistence RMSE": f"{score.baseline_rmse:.3f}",
                    "vs baseline": f"{score.rmse_vs_baseline_pct:+.2f}%",
                    "Fit (s)": f"{score.fit_seconds:.1f}",
                }
                for score in outcome.fold_scores
            ]
        )
    )


def _verdict(outcome) -> str:
    if outcome.rung.is_baseline:
        return (
            "This is the **baseline test**. Every other rung in `ML_Algorithms/` is "
            "measured against the number in the table above; it is not measured "
            "against anything itself."
        )
    primary = outcome.primary
    if primary.beats_baseline:
        return (
            f"**Beats the baseline** by {primary.rmse_vs_baseline_pct:.2f}% RMSE on the "
            f"primary fold ({primary.rmse:.3f} against the anchor's "
            f"{primary.baseline_rmse:.3f})."
        )
    return (
        f"**Does not beat the baseline.** {primary.rmse:.3f} RMSE against the anchor's "
        f"{primary.baseline_rmse:.3f}, a {primary.rmse_vs_baseline_pct:.2f}% change. "
        "Recorded as a finding rather than dropped — a rung that cannot beat "
        "the PM10 anchor is not earning its complexity."
    )


def _figure_section(figure_paths: list[Path]) -> str:
    if not figure_paths:
        return ""
    captions = {
        "predicted_vs_actual.png": "Predicted against actual next-hour PM2.5 on the primary holdout.",
        "error_by_level.png": "RMSE within each air-quality band, and how many hours sit in each.",
        "residuals_over_time.png": "Daily RMSE and daily bias across the holdout block.",
        "top_drivers.png": "The features the model leaned on hardest.",
    }
    lines = ["## Figures", ""]
    for path in figure_paths:
        lines += [
            f"### {path.stem.replace('_', ' ').capitalize()}",
            "",
            captions.get(path.name, ""),
            "",
            f"![{path.stem}](figures/{path.name})",
            "",
        ]
    return "\n".join(lines)


def _ablation_link(directory: Path) -> str:
    """Point at the rung's ablation record, for the rungs that have one.

    A rung earns its place by beating the one below it, but the variants it beat
    on the way are evidence too, and they belong somewhere a reader will find
    them. Rungs without an ABLATIONS.md get no link and no empty heading.
    """
    if not (directory / ABLATIONS_FILE).exists():
        return ""
    return (
        "## Further evidence\n\n"
        f"[{ABLATIONS_FILE}]({ABLATIONS_FILE}) — the variants this rung was chosen over, "
        "scored on the same fold.\n\n"
    )


def write_results(outcome, directory: Path, figure_paths: list[Path]) -> Path:
    """The human-readable record of one test."""
    rung = outcome.rung
    warnings = (
        "\n".join(f"> {line}" for line in outcome.warnings) + "\n"
        if outcome.warnings
        else ""
    )
    body = f"""# {rung.label} — {rung.algorithm}

{rung.summary}

{warnings}
## Scores

Season-matched holdout blocks, each trained only on the data preceding it
([ADR-0002](../../docs/adr/0002-season-matched-validation.md)). Persistence is re-scored on the
same holdout rows, so the comparison is like-for-like.

{_fold_table(outcome)}

{_verdict(outcome)}

## Submission

`submission.csv` — {outcome.submission_rows:,} rows, produced by refitting on every labelled
training station-hour and predicting the hidden test window. It passed `submission.validate`:
row count, id set, no blanks, no negative concentrations.

## Reproduce

```bash
.venv/bin/python -m src.cache                      # build data/train.parquet if absent
.venv/bin/python -m src.experiments {rung.slug}
```

Total run time: {outcome.total_seconds:.1f}s.

{_ablation_link(directory)}{_figure_section(figure_paths)}"""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RESULTS_FILE
    path.write_text(body)
    return path


def _collect_metrics(root: Path) -> list[dict]:
    found = []
    for rung in RUNGS:
        path = root / rung.slug / METRICS_FILE
        if path.exists():
            found.append(json.loads(path.read_text()))
    return found


def _metric_split_note(collected: list[dict]) -> str:
    """Say so when the best rung by RMSE is not the best rung by MAE.

    Computed from the metrics rather than written down, so it stays true if the
    ladder is re-run. The split is worth surfacing: RMSE is decided by the
    pollution episodes in the tail, MAE by the ordinary hours in the bulk, and a
    rung can easily win one while losing the other.
    """
    best_rmse = min(collected, key=lambda entry: entry["rmse"])
    best_mae = min(collected, key=lambda entry: entry["mae"])
    if best_rmse["algorithm"] == best_mae["algorithm"]:
        return (
            f"\n**{best_rmse['algorithm']}** leads on both RMSE "
            f"({best_rmse['rmse']:.3f}) and MAE ({best_mae['mae']:.3f}). Leading the table is not "
            "promotion: a rung is the submission only once the paired day-block bootstrap puts "
            "its primary-fold 90% interval entirely below zero.\n"
        )
    return (
        f"\n**The two metrics disagree.** `{best_rmse['algorithm']}` has the lowest RMSE "
        f"({best_rmse['rmse']:.3f}) while `{best_mae['algorithm']}` has the lowest MAE "
        f"({best_mae['mae']:.3f}). RMSE is decided by the pollution episodes in the tail and "
        "MAE by the ordinary hours in the bulk, so a rung that is better on typical hours can "
        "still lose the competition metric. The competition scores RMSE.\n"
    )


def write_ladder(root: Path | None = None) -> Path:
    """Rebuild ML_Algorithms/README.md from every metrics.json on disk."""
    root = root or config.ML_ALGORITHMS_DIR
    collected = _collect_metrics(root)
    if not collected:
        raise FileNotFoundError(f"No metrics.json found under {root}. Run a test first.")

    scores = {
        entry["algorithm"]: {"rmse": entry["rmse"], "mae": entry["mae"]}
        for entry in collected
    }
    ladder = evaluate.ladder_table(scores, baseline=baseline().algorithm)

    table = pd.DataFrame(
        [
            {
                "Test": f"[{entry['test']}]({entry['test']}/{RESULTS_FILE})",
                "Algorithm": entry["algorithm"],
                "RMSE": f"{entry['rmse']:.3f}",
                "MAE": f"{entry['mae']:.3f}",
                "vs baseline": (
                    "baseline"
                    if entry["is_baseline"]
                    else f"{ladder.loc[entry['algorithm'], 'rmse_vs_baseline']:+.2f}%"
                ),
                "Beats baseline": (
                    "—" if entry["is_baseline"] else ("yes" if entry["beats_baseline"] else "no")
                ),
                "Fit (s)": f"{entry['fit_seconds']:.1f}",
            }
            for entry in collected
        ]
    )

    pending = [rung.slug for rung in RUNGS if rung.slug not in {e["test"] for e in collected}]
    split_note = _metric_split_note(collected)
    pending_note = (
        "\nNot yet run: " + ", ".join(f"`{slug}`" for slug in pending) + ".\n"
        if pending
        else ""
    )
    fold = collected[0]["primary_fold"]

    body = f"""# ML Algorithms — the ladder

One folder per test. Each holds a `RESULTS.md`, a `metrics.json`, the diagnostic figures, and
the `submission.csv` that test produced.

Every rung is scored on the same season-matched holdout — **{fold}**, trained only on the data
preceding it ([ADR-0002](../docs/adr/0002-season-matched-validation.md)) — and the anchor is
re-scored on the same rows so the comparison is like-for-like. A rung that fails to beat
the PM10 anchor is reported as failing rather than quietly dropped.

{markdown_table(table)}
{split_note}{pending_note}
## How blanks are treated

Not uniformly, and deliberately so ([ADR-0003](../docs/adr/0003-per-rung-imputation.md)).
Ridge cannot see a missing value, so it receives a station-by-hour-of-day median fill fitted
on training rows alone. LightGBM handles blanks natively and is fed the frame intact, because
the pattern of absence is itself information.

## Reproduce

```bash
.venv/bin/python -m src.cache            # data/processed/train.parquet and test.parquet
.venv/bin/python -m src.experiments all  # every test, then this table
```
"""
    root.mkdir(parents=True, exist_ok=True)
    path = root / LADDER_FILE
    path.write_text(body)
    return path
