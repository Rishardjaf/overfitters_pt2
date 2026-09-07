"""Print every v3 result as one table (cheapest way to see where things stand).

    uv run python scripts/v3/summarize.py            # all runs, sorted by headline
    uv run python scripts/v3/summarize.py lgbm_f9    # only names containing the substring
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "artifacts" / "v3" / "results"

sub = sys.argv[1] if len(sys.argv) > 1 else ""
rows = []
for p in sorted(RES.glob("*.json")):
    if sub and sub not in p.stem:
        continue
    try:
        r = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        continue
    if "folds" not in r:
        continue
    f = r["folds"]
    by = {x["fold"]: x for x in f}
    t2 = by.get(2, {}).get("tails")
    rows.append({
        "run": r["name"], "model": r["model"], "features": r["features"], "cols": r["n_features"],
        "target": r["target"],
        **{f"f{i}": round(by[i]["rmse_all"], 2) if i in by else None for i in range(4)},
        "headline": round(r["summary"]["headline"], 3),
        "all-fold": round(r["summary"]["mean_all_folds"], 2),
        "rounds": [x["fit_info"].get("n_rounds") for x in f],
        "f2 fall/calm/rise": f"{t2['fall']['rmse']:.0f}/{t2['calm']['rmse']:.1f}/{t2['rise']['rmse']:.0f}" if t2 else "",
        "secs": r.get("seconds"),
    })
rows.sort(key=lambda d: d["headline"])
if not rows:
    print("no results")
    sys.exit(0)
cols = list(rows[0].keys())
w = {c: max(len(c), *(len(str(d[c])) for d in rows)) for c in cols}
print(" | ".join(c.ljust(w[c]) for c in cols))
print("-+-".join("-" * w[c] for c in cols))
for d in rows:
    print(" | ".join(str(d[c]).ljust(w[c]) for c in cols))
print(f"\n{len(rows)} runs. headline = mean RMSE over heating folds 0 and 2 (all rows). Yardstick: lgbm_f0 30.80. Target 17.0.")
