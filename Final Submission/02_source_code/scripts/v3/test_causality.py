"""Causality test for the v3 panel: rebuild on truncated data, compare.

Any feature (or the diagnostic anchor) that changes when later rows are added has
read the future.

    uv run python scripts/v3/test_causality.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import features_v3 as fv  # noqa: E402

CUTOFF = pd.Timestamp("2015-06-15 11:00")


def main() -> int:
    raw = fv.load_raw()
    full, groups = fv.build_panel_from(raw)
    trunc, _ = fv.build_panel_from(raw[raw[fv.TIME_COL] <= CUTOFF].copy())
    feats = sorted({c for cols in groups.values() for c in cols} | {fv.DIAG})
    key = [fv.STATION_COL, fv.TIME_COL]
    a = full[full[fv.TIME_COL] <= CUTOFF].set_index(key).sort_index()
    b = trunc.set_index(key).sort_index()
    assert a.index.equals(b.index), "grid differs before the cut-off"
    bad = []
    for c in feats:
        x, y = a[c], b[c]
        if isinstance(x.dtype, pd.CategoricalDtype):
            xs = x.astype(object).where(x.notna(), "<NA>").astype(str).to_numpy()
            ys = y.astype(object).where(y.notna(), "<NA>").astype(str).to_numpy()
            mism = xs != ys
        else:
            mism = ~np.isclose(x.to_numpy(float), y.to_numpy(float), equal_nan=True, atol=1e-4)
        if mism.any():
            bad.append((c, int(mism.sum())))
    if bad:
        print("CAUSALITY TEST FAILED — these features change when future rows are added:")
        for c, n in bad:
            print(f"  {c}: {n} rows differ")
        return 1
    print(f"Causality test passed: {len(feats)} columns identical on {len(a):,} rows before {CUTOFF}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
