"""Profile the competition data and write reports/data_profile.md.

Deliberately stdlib-only so it runs before `make setup` has built the
environment. Every number quoted in PROJECT.md comes from this script; rerun it
with `make eda` if the data ever changes.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
OUT = ROOT / "reports" / "data_profile.md"

TARGET = "PM2_5_next_hour"
ANCHOR = "current_PM2_5"


def quantile(sorted_vals: list[float], p: float) -> float:
    return sorted_vals[int(p * (len(sorted_vals) - 1))]


def read(path: Path) -> tuple[list[dict], list[str]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def profile(rows: list[dict], cols: list[str], label: str, out: list[str]) -> None:
    n = len(rows)
    stamps = [r["observation_timestamp"] for r in rows]
    stations = Counter(r["station"] for r in rows)
    months = Counter(int(r["month"]) for r in rows)
    missing = Counter()
    for r in rows:
        for c in cols:
            if not r[c]:
                missing[c] += 1

    out.append(f"## {label}\n")
    out.append(f"- rows: **{n:,}**")
    out.append(f"- period: `{min(stamps)}` -> `{max(stamps)}`")
    out.append(f"- stations: {len(stations)}")
    out.append(f"- months present: {sorted(months)}\n")

    out.append("### Rows by month\n")
    out.append("| " + " | ".join(str(m) for m in sorted(months)) + " |")
    out.append("|" + "---:|" * len(months))
    out.append("| " + " | ".join(f"{months[m]:,}" for m in sorted(months)) + " |\n")

    if missing:
        out.append("### Missingness\n")
        out.append("| column | missing | % |")
        out.append("|---|---:|---:|")
        for c in cols:
            if missing[c]:
                out.append(f"| `{c}` | {missing[c]:,} | {100 * missing[c] / n:.2f}% |")
        out.append("")

    anchor = sorted(float(r[ANCHOR]) for r in rows if r[ANCHOR])
    out.append(
        f"- `{ANCHOR}` mean **{statistics.mean(anchor):.2f}**, "
        f"median {quantile(anchor, 0.5):.0f}, p95 {quantile(anchor, 0.95):.0f}, "
        f"max {anchor[-1]:.0f}\n"
    )

    if TARGET in cols:
        tgt = sorted(float(r[TARGET]) for r in rows if r[TARGET])
        out.append("### Target distribution\n")
        out.append("| min | p25 | median | p75 | p95 | p99 | max | mean |")
        out.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        out.append(
            f"| {tgt[0]:.0f} | {quantile(tgt, 0.25):.0f} | {quantile(tgt, 0.5):.0f} | "
            f"{quantile(tgt, 0.75):.0f} | {quantile(tgt, 0.95):.0f} | "
            f"{quantile(tgt, 0.99):.0f} | {tgt[-1]:.0f} | {statistics.mean(tgt):.2f} |\n"
        )


def baselines(rows: list[dict], out: list[str]) -> None:
    """Persistence and mean baselines — the numbers every model is measured against."""
    pairs = [(float(r[ANCHOR]), float(r[TARGET])) for r in rows if r[ANCHOR] and r[TARGET]]
    se = [(c - t) ** 2 for c, t in pairs]
    ae = [abs(c - t) for c, t in pairs]

    tgt = [float(r[TARGET]) for r in rows if r[TARGET]]
    mean = statistics.mean(tgt)
    mean_rmse = math.sqrt(sum((mean - t) ** 2 for t in tgt) / len(tgt))

    recent = [
        (float(r[ANCHOR]), float(r[TARGET]))
        for r in rows
        if r[ANCHOR] and r[TARGET] and r["observation_timestamp"] >= "2016-03-01"
    ]
    recent_rmse = math.sqrt(sum((c - t) ** 2 for c, t in recent) / len(recent))

    out.append("## Baselines (train)\n")
    out.append("| baseline | RMSE | MAE |")
    out.append("|---|---:|---:|")
    out.append(f"| global mean | {mean_rmse:.2f} | - |")
    out.append(
        f"| **persistence** (`{ANCHOR}`) | **{math.sqrt(sum(se) / len(se)):.2f}** "
        f"| {sum(ae) / len(ae):.2f} |"
    )
    out.append(f"| persistence, 2016-03-01 onward | {recent_rmse:.2f} | - |\n")
    out.append(
        "Persistence removes ~75% of the error the mean baseline leaves. "
        "Quote every model against persistence, never against the mean.\n"
    )


def continuity(rows: list[dict], out: list[str]) -> None:
    """How often consecutive retained rows are not exactly one hour apart."""
    by_station: dict[str, list[datetime]] = defaultdict(list)
    for r in rows:
        by_station[r["station"]].append(datetime.fromisoformat(r["observation_timestamp"]))

    out.append("## Timeline continuity (train)\n")
    out.append("| station | rows | non-1h steps |")
    out.append("|---|---:|---:|")
    total = 0
    for st in sorted(by_station):
        ts = sorted(by_station[st])
        gaps = sum(1 for a, b in zip(ts, ts[1:], strict=False) if b - a != timedelta(hours=1))
        total += gaps
        out.append(f"| {st} | {len(ts):,} | {gaps} |")
    out.append(f"| **total** | | **{total}** |\n")
    out.append(
        "Series are near-complete, so lag and rolling features are viable — but "
        "build them on a reindexed complete hourly grid so a 'lag 1' is genuinely "
        "`t-1` and not merely the previous surviving row.\n"
    )


def main() -> None:
    train, train_cols = read(DATA / "train.csv")
    test, test_cols = read(DATA / "test.csv")

    out: list[str] = [
        "# Data profile\n",
        "Generated by `make eda` (`scripts/profile_data.py`). Do not edit by hand.\n",
    ]
    profile(train, train_cols, "Train", out)
    profile(test, test_cols, "Test", out)
    baselines(train, out)
    continuity(train, out)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(out))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
