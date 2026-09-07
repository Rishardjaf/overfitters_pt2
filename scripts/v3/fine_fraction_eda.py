"""Fine-fraction EDA: what separates PM2.5 / PM10 on train rows, before any training.

No model is fit here. `ff` is a diagnostic column computed on train rows only and never
written to the panel or a feature list.

    uv run python scripts/v3/fine_fraction_eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "v2"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features_v3 import load_panel  # noqa: E402

from pm25.config import TARGET  # noqa: E402


def explain_ratio(group_means: pd.Series, overall_std: float) -> float:
    between = float(group_means.std(ddof=0))
    return between / overall_std if overall_std else float("nan")


def table(g: pd.core.groupby.generic.SeriesGroupBy, label: str) -> tuple[str, float]:
    agg = g.agg(["mean", "std", "count"]).round(3)
    lines = [f"### {label}", "", "| group | mean ff | std ff | n |", "|---|---:|---:|---:|"]
    for idx, row in agg.iterrows():
        lines.append(f"| {idx} | {row['mean']:.3f} | {row['std']:.3f} | {int(row['count']):,} |")
    return "\n".join(lines), agg["mean"]


def main() -> None:
    panel, _ = load_panel()
    real = panel[~panel["is_pad"]]
    train = real[~real["is_test"]].copy()
    train["ff"] = (train[TARGET] / np.maximum(train["PM10_filled"], 1.0)).clip(0, 2.0)
    overall_std = float(train["ff"].std())
    print(f"overall ff std (train): {overall_std:.4f}")

    train["RH_decile"] = pd.qcut(train["RH"], 10, duplicates="drop")
    train["PM10_decile"] = pd.qcut(train["PM10"], 10, duplicates="drop")
    train["CO_per_PM10_decile"] = pd.qcut(train["CO_per_PM10"], 10, duplicates="drop")
    train["hour_heat"] = train["hour"].astype(str) + "_h" + train["is_heating_season"].astype(str)
    train["wspm2_bin"] = pd.cut(train["hrs_since_wspm_ge2"], [-1, 0, 3, 12, 48, 1e9],
                                 labels=["0", "1-3", "4-12", "13-48", ">48"])  # fmt: skip
    train["rise_run_bin"] = pd.cut(train["pm10_rise_run"], [-1, 0, 2, 5, 1e9],
                                    labels=["0", "1-2", "3-5", ">5"])  # fmt: skip

    groupings = [
        ("station", "station_cat"),
        ("RH decile", "RH_decile"),
        ("wind direction", "wd_cat"),
        ("hour x heating season", "hour_heat"),
        ("PM10 decile", "PM10_decile"),
        ("CO/PM10 decile", "CO_per_PM10_decile"),
        ("hours since WSPM>=2 (stagnation)", "wspm2_bin"),
        ("PM10 rise-run length", "rise_run_bin"),
    ]

    out = ["# Finding v3-09 — Fine-fraction EDA: what separates PM2.5 / PM10",
           "",
           "**Method:** `scripts/v3/fine_fraction_eda.py` · train rows only · "
           "`ff = clip(PM2_5_next_hour / max(PM10_filled, 1), 0, 2)`, a diagnostic column, "
           "never written to the panel or a feature list.",
           "",
           f"Overall std of `ff` across all train rows: **{overall_std:.4f}**.",
           "",
           "For each grouping below, the explanatory ratio is the between-group std of the "
           "group means divided by the overall std — how much of the fine-fraction variance "
           "that grouping explains on its own (no interactions).",
           "",
           "| grouping | between-group std | ratio to overall std |",
           "|---|---:|---:|"]  # fmt: skip
    ratios = []
    detail_sections = []
    for label, col in groupings:
        g = train.groupby(col, observed=True)["ff"]
        means = g.mean()
        between = float(means.std(ddof=0))
        ratio = between / overall_std if overall_std else float("nan")
        ratios.append((label, ratio))
        out.append(f"| {label} | {between:.4f} | {ratio:.3f} |")
        sec, _ = table(g, label)
        detail_sections.append(sec)

    ratios.sort(key=lambda x: -x[1])
    out.append("")
    out.append("## What this says, in plain terms")
    out.append("")
    top3 = ", ".join(f"**{l}** ({r:.2f})" for l, r in ratios[:3])
    out.append(f"1. The groupings that separate the fine fraction most are {top3} — these are "
               f"where a matching feature should pay off first.")
    bottom = ", ".join(f"{l} ({r:.2f})" for l, r in ratios[-2:])
    out.append(f"2. The weakest groupings are {bottom} — features built on these are less likely "
               f"to move the headline.")
    out.append("3. This is a diagnostic only; it does not retrain anything and does not decide "
               "which feature groups to keep — the screening runs in "
               "`Findings/v3/10-feature-round2.md` do that.")
    out.append("")
    out.append("## Detail tables")
    out.append("")
    out.extend(detail_sections)

    dest = ROOT / "Findings" / "v3" / "09-fine-fraction-eda.md"
    dest.write_text("\n\n".join(out) + "\n")
    print(f"wrote {dest}")
    print()
    for label, ratio in ratios:
        print(f"  {label:32s} explanatory ratio {ratio:.3f}")


if __name__ == "__main__":
    main()
