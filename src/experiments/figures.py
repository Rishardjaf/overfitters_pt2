"""Three diagnostics per test, plus a top-drivers chart where the model offers one.

The three are chosen to answer the questions the ladder table cannot. A single
RMSE says a rung is better; these say where it is better, whether it is biased,
and whether it holds up during the pollution episodes that decide the score.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import evaluate

FIGURE_DPI = 120
TOP_DRIVERS = 20
UPPER_PERCENTILE = 99.5

INK = "#3d5a80"
ACCENT = "#c1121f"
LIGHT = "#98c1d9"

# China's PM2.5 air-quality bands, in µg/m³. The top two are the pollution
# episodes that dominate a squared-error metric.
LEVEL_EDGES = [0, 35, 75, 150, 250, np.inf]
LEVEL_NAMES = ["0-35", "35-75", "75-150", "150-250", "250+"]


def _save(figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    return path


def predicted_vs_actual(actual: pd.Series, predicted: pd.Series, path: Path) -> Path:
    """Where the rung sits relative to a perfect forecast."""
    limit = float(np.nanpercentile(actual, UPPER_PERCENTILE))
    figure, axis = plt.subplots(figsize=(5.4, 5))
    axis.hexbin(actual, predicted, gridsize=60, bins="log", cmap="Blues",
                extent=(0, limit, 0, limit))
    axis.plot([0, limit], [0, limit], color=ACCENT, lw=1.2, label="perfect forecast")
    axis.set(
        xlim=(0, limit), ylim=(0, limit),
        xlabel="Actual next-hour PM2.5 (µg/m³)",
        ylabel="Predicted next-hour PM2.5 (µg/m³)",
        title="Predicted vs actual",
    )
    axis.legend(loc="upper left", frameon=True)
    return _save(figure, path)


def error_by_level(actual: pd.Series, predicted: pd.Series, path: Path) -> Path:
    """RMSE within each air-quality band — the tail is where the score is decided."""
    bands = pd.cut(actual, bins=LEVEL_EDGES, labels=LEVEL_NAMES, right=False)
    frame = pd.DataFrame({"actual": actual, "predicted": predicted, "band": bands})

    scores = frame.groupby("band", observed=False).apply(
        lambda group: evaluate.rmse(group["actual"], group["predicted"]),
        include_groups=False,
    )
    counts = frame.groupby("band", observed=False).size()

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    axes[0].bar(scores.index.astype(str), scores.to_numpy(), color=INK)
    axes[0].set(xlabel="Actual PM2.5 band (µg/m³)", ylabel="RMSE (µg/m³)",
                title="Error grows with concentration")
    axes[1].bar(counts.index.astype(str), counts.to_numpy(), color=LIGHT)
    axes[1].set(xlabel="Actual PM2.5 band (µg/m³)", ylabel="Station-hours",
                title="Most hours sit in the lowest bands")
    axes[1].set_yscale("log")
    return _save(figure, path)


def residuals_over_time(
    timestamps: pd.Series, actual: pd.Series, predicted: pd.Series, path: Path
) -> Path:
    """Daily error and daily bias across the holdout block."""
    frame = pd.DataFrame(
        {"day": pd.to_datetime(timestamps).dt.floor("D"),
         "actual": actual, "predicted": predicted}
    ).dropna()
    daily = frame.groupby("day").apply(
        lambda group: pd.Series(
            {
                "rmse": evaluate.rmse(group["actual"], group["predicted"]),
                "bias": float((group["predicted"] - group["actual"]).mean()),
            }
        ),
        include_groups=False,
    )

    figure, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    axes[0].plot(daily.index, daily["rmse"], color=INK, lw=1)
    axes[0].set(ylabel="Daily RMSE (µg/m³)", title="Error is concentrated in episodes")
    axes[1].fill_between(daily.index, daily["bias"], color=LIGHT)
    axes[1].axhline(0, color=ACCENT, lw=1)
    axes[1].set(xlabel="Holdout day", ylabel="Daily bias (µg/m³)")
    figure.autofmt_xdate()
    return _save(figure, path)


def top_drivers(importance: pd.DataFrame, column: str, path: Path) -> Path:
    """The features the model leaned on hardest."""
    top = importance.head(TOP_DRIVERS).iloc[::-1]
    figure, axis = plt.subplots(figsize=(7, 0.28 * len(top) + 1.2))
    axis.barh(top["feature"], top[column], color=INK)
    axis.set(xlabel=column.replace("_", " "), title=f"Top {len(top)} drivers")
    axis.tick_params(axis="y", labelsize=8)
    return _save(figure, path)


def draw_all(outcome, directory: Path) -> list[Path]:
    """Every figure this rung can produce, written into `directory/figures`."""
    figure_dir = Path(directory) / "figures"
    diagnostics = outcome.diagnostics
    paths = [
        predicted_vs_actual(
            diagnostics.actual, diagnostics.predicted, figure_dir / "predicted_vs_actual.png"
        ),
        error_by_level(
            diagnostics.actual, diagnostics.predicted, figure_dir / "error_by_level.png"
        ),
        residuals_over_time(
            diagnostics.timestamps, diagnostics.actual, diagnostics.predicted,
            figure_dir / "residuals_over_time.png",
        ),
    ]
    if outcome.importance is not None and outcome.importance_column:
        paths.append(
            top_drivers(
                outcome.importance, outcome.importance_column, figure_dir / "top_drivers.png"
            )
        )
    return paths
