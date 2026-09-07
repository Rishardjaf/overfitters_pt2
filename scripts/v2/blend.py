"""Plan v2-06 steps 1-3: does a blend of OOF predictions beat the best single model?

Weights are non-negative least squares on one heating fold and evaluated on the
other, both ways, so a blend that only helps where its weights were fitted is
exposed. Also reports the pairwise correlation of the models' residuals.

    uv run python scripts/v2/blend.py lgbm_lean_delta xgb_lean_delta cat_lean_delta ridge_lean_delta
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[2]
OOF = ROOT / "artifacts" / "v2" / "oof"
HEATING = [0, 2]


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def load(names: list[str]) -> pd.DataFrame:
    frames = []
    for n in names:
        df = pd.read_parquet(OOF / f"{n}.parquet")[["id", "fold", "y", "anchor", "pred"]]
        frames.append(df.rename(columns={"pred": n}).set_index(["id", "fold"]))
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f[[c for c in f.columns if c not in ("y", "anchor")]], how="inner")
    return out.reset_index()


def main(names: list[str]) -> None:
    df = load(names)
    df = df[df["anchor"].notna()]  # gate population; the fallback is a separate decision
    print(f"{len(df):,} anchor-present OOF rows shared by {names}")

    print("\n== single-model RMSE per fold ==")
    single = {
        n: {
            f: rmse(df.loc[df.fold == f, "y"], df.loc[df.fold == f, n])
            for f in sorted(df.fold.unique())
        }
        for n in names
    }
    for n in names:
        s = single[n]
        head = np.mean([s[f] for f in HEATING if f in s])
        print(
            f"  {n:22s} "
            + "  ".join(f"f{f}={v:.3f}" for f, v in s.items())
            + f"   headline={head:.3f}"
        )

    print("\n== residual correlation on heating folds ==")
    h = df[df.fold.isin(HEATING)]
    res = pd.DataFrame({n: h["y"] - h[n] for n in names})
    print(res.corr().round(3).to_string())

    print("\n== NNLS blend: fit on one heating fold, score on the other ==")
    out = {}
    for fit_fold, eval_fold in [(0, 2), (2, 0)]:
        a = df[df.fold == fit_fold]
        b = df[df.fold == eval_fold]
        # blend the Δ predictions (pred - anchor) so a zero weight vector is persistence
        A = np.column_stack([a[n] - a["anchor"] for n in names])
        w, _ = nnls(A, (a["y"] - a["anchor"]).to_numpy())
        w = w / w.sum() if w.sum() > 0 else w
        B = np.column_stack([b[n] - b["anchor"] for n in names])
        p = b["anchor"].to_numpy() + B @ w
        best_single = min(single[n][eval_fold] for n in names)
        out[f"fit{fit_fold}_eval{eval_fold}"] = {
            "weights": dict(zip(names, w.round(3))),
            "blend_rmse": rmse(b["y"], p),
            "best_single_rmse": best_single,
            "persistence": rmse(b["y"], b["anchor"]),
        }
        print(
            f"  fit on fold {fit_fold} → eval fold {eval_fold}: blend {rmse(b['y'], p):.3f} vs best single {best_single:.3f}"
            f"   weights {dict(zip(names, w.round(3)))}"
        )
    # equal-weight blend for reference
    for f in HEATING:
        b = df[df.fold == f]
        p = b["anchor"] + np.mean([b[n] - b["anchor"] for n in names], axis=0)
        print(f"  equal-weight blend fold {f}: {rmse(b['y'], p):.3f}")
        out[f"equal_fold{f}"] = rmse(b["y"], p)
    (ROOT / "artifacts" / "v2" / "results" / f"blend_{'+'.join(names)}.json").write_text(
        json.dumps(out, indent=1, default=str)
    )


if __name__ == "__main__":
    main(sys.argv[1:])
