"""Blend v3 out-of-fold predictions: NNLS fit on one heating fold, scored on the other.

    uv run python scripts/v3/blend.py lgbm_f9_deep_ratio lgbm_f9_ratio_s1 cat_f9_ratio
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[2]
OOF = ROOT / "artifacts" / "v3" / "oof"
HEATING = [0, 2]


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def main(names: list[str]) -> None:
    df = None
    for n in names:
        d = pd.read_parquet(OOF / f"{n}.parquet")[["id", "fold", "y", "pred"]].rename(columns={"pred": n})
        df = d if df is None else df.merge(d.drop(columns="y"), on=["id", "fold"], how="inner")
    print(f"{len(df):,} OOF rows shared by {names}\n")
    single = {n: {f: rmse(df.loc[df.fold == f, "y"], df.loc[df.fold == f, n]) for f in sorted(df.fold.unique())} for n in names}
    for n in names:
        s = single[n]
        print(f"  {n:24s} " + "  ".join(f"f{f}={v:.3f}" for f, v in s.items()) + f"   headline={np.mean([s[f] for f in HEATING]):.3f}")
    h = df[df.fold.isin(HEATING)]
    print("\nresidual correlation (heating folds):")
    print(pd.DataFrame({n: h["y"] - h[n] for n in names}).corr().round(3).to_string())
    out = {}
    print("\nNNLS blend, fit on one heating fold -> score on the other:")
    for fit_f, ev_f in [(0, 2), (2, 0)]:
        a, b = df[df.fold == fit_f], df[df.fold == ev_f]
        w, _ = nnls(np.column_stack([a[n] for n in names]), a["y"].to_numpy())
        w = w / w.sum() if w.sum() > 0 else w
        p = np.column_stack([b[n] for n in names]) @ w
        best = min(single[n][ev_f] for n in names)
        out[f"fit{fit_f}_eval{ev_f}"] = {"weights": dict(zip(names, w.round(3))), "blend": rmse(b["y"], p), "best_single": best}
        print(f"  fit f{fit_f} -> f{ev_f}: blend {rmse(b['y'], p):.3f} vs best single {best:.3f}   weights {dict(zip(names, w.round(3)))}")
    for f in HEATING:
        b = df[df.fold == f]
        e = rmse(b["y"], np.mean([b[n] for n in names], axis=0))
        out[f"equal_f{f}"] = e
        print(f"  equal-weight f{f}: {e:.3f}")
    (ROOT / "artifacts" / "v3" / "results" / f"blend_{'+'.join(names)}.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
