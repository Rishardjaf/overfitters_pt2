"""Plan v2-07: fallback predictions for rows whose `current_PM2_5` is missing.

Δ is undefined on these rows, and the persistence anchor is absent, so they need
their own predictor. Candidates are scored only on the missing-anchor validation
rows of each fold, then combined with the best Δ model's OOF predictions to show
the effect on the operational headline.

    uv run python scripts/v2/missing_anchor.py --delta-oof lgbm_v2_delta
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from cv import ANCHOR, TARGET, TIME_COL, folds, rmse  # noqa: E402
from features_v2 import feature_sets, load_panel  # noqa: E402
from run_experiment import lgbm_default_params  # noqa: E402

from pm25.config import STATION_COL  # noqa: E402

OUT = ROOT / "artifacts" / "v2" / "results" / "missing_anchor.json"


def anchor_derived(cols: list[str]) -> list[str]:
    """Features that are NaN or meaningless when the anchor is missing."""
    keep_prefixes = ("network_", "net_")
    out = []
    for c in cols:
        if c.startswith(keep_prefixes):
            continue
        if c in ("current_PM2_5_missing", "PM25_lag_1h_missing", "PM25_lag_24h_missing",
                 "PM25_history_available"):  # fmt: skip
            continue
        if (
            "current_PM2_5" in c
            or c.startswith("PM25_")
            or c.startswith("pm25_")
            or c in ("pm10_equals_pm25", "PM10_PM25_ratio", "PM25_over_PM10")
        ):
            out.append(c)
    return out


def main(args) -> None:
    panel, groups = load_panel()
    feats = feature_sets(groups)["v2"]
    cat = [c for c in ("station_cat", "wd_cat") if c in feats]
    real = panel[(~panel["is_pad"]) & (~panel["is_test"])].reset_index(drop=True)
    ts = real[TIME_COL]
    miss_all = real[ANCHOR].isna()

    no_anchor_feats = [c for c in feats if c not in anchor_derived(feats)]
    print(f"v2 features {len(feats)} → {len(no_anchor_feats)} usable without the anchor")

    oof = pd.read_parquet(ROOT / "artifacts" / "v2" / "oof" / f"{args.delta_oof}.parquet")

    results = {}
    for f in folds():
        otr, ova, _, _ = f.masks(ts)
        tr = real[otr]
        va = real[ova]
        vm = va[va[ANCHOR].isna()]
        y = vm[TARGET].to_numpy(float)
        n = len(vm)
        res = {"n": int(n), "pm10_present_share": float(vm["PM10"].notna().mean())}

        # A: same-hour leave-one-out network mean, station offset from the train fold
        offset = (tr[ANCHOR] - tr["net_loo_mean"]).groupby(tr[STATION_COL]).median()
        predA = vm["net_loo_mean"].to_numpy(float) + vm[STATION_COL].map(offset).fillna(0).to_numpy(
            float
        )
        predA = np.where(np.isnan(predA), tr[ANCHOR].mean(), predA)
        res["A_network_loo_plus_offset"] = rmse(y, predA)

        # B: general level model on all v2 features (the current fallback)
        p = lgbm_default_params() | {"num_leaves": 31}
        mB = lgb.train(p, lgb.Dataset(tr[feats], tr[TARGET], categorical_feature=cat), 400)
        predB = np.maximum(mB.predict(vm[feats]), 0)
        res["B_level_all_features"] = rmse(y, predB)

        # C: level model trained WITHOUT any anchor-derived feature — forced to learn
        # the level from the network, own PM10/CO/NO2 and weather, like the fallback must
        mC = lgb.train(
            p, lgb.Dataset(tr[no_anchor_feats], tr[TARGET], categorical_feature=cat), 400
        )
        predC = np.maximum(mC.predict(vm[no_anchor_feats]), 0)
        res["C_level_no_anchor_features"] = rmse(y, predC)

        # C2: same, but trained only on heating-season months of the train fold
        trh = tr[tr["is_heating_season"] == 1]
        mC2 = lgb.train(
            p, lgb.Dataset(trh[no_anchor_feats], trh[TARGET], categorical_feature=cat), 400
        )
        predC2 = np.maximum(mC2.predict(vm[no_anchor_feats]), 0)
        res["C2_level_no_anchor_heating_only"] = rmse(y, predC2)

        # D: station fine-fraction x PM10 (train-fold median of PM2.5/PM10), network mean where PM10 missing
        frac = (tr[ANCHOR] / tr["PM10"].replace(0, np.nan)).groupby(tr[STATION_COL]).median()
        predD = vm["PM10"].to_numpy(float) * vm[STATION_COL].map(frac).to_numpy(float)
        predD = np.where(np.isnan(predD), predA, predD)
        res["D_pm10_x_fine_fraction"] = rmse(y, predD)

        # E: mean of A and C
        predE = 0.5 * predA + 0.5 * predC
        res["E_blend_A_C"] = rmse(y, predE)
        # F: mean of A and B
        predF = 0.5 * predA + 0.5 * predB
        res["F_blend_A_B"] = rmse(y, predF)
        # G: B bounded to the network's same-hour spread around A (a station rarely sits
        # more than ~2 network-std away from the rest of the city)
        spread = 2.0 * vm["network_PM25_std"].fillna(vm["network_PM25_std"].median()).to_numpy(
            float
        )
        predG = np.clip(predB, predA - spread, predA + spread)
        res["G_B_bounded_by_network"] = rmse(y, predG)
        # H: median of A, B, D
        predH = np.median(np.vstack([predA, predB, predD]), axis=0)
        res["H_median_A_B_D"] = rmse(y, predH)

        # the rows that carry the error, for inspection
        worst = pd.DataFrame(
            {
                "station": vm[STATION_COL].to_numpy(),
                "ts": vm[TIME_COL].astype(str).to_numpy(),
                "y": y,
                "A": predA.round(1),
                "B": predB.round(1),
                "D": predD.round(1),
                "PM10": vm["PM10"].to_numpy(),
                "net_loo_mean": vm["net_loo_mean"].round(1).to_numpy(),
                "gap_h": vm["gap_prev_hours"].to_numpy(),
            }
        )
        worst["se_B"] = (worst["y"] - worst["B"]) ** 2
        worst = worst.sort_values("se_B", ascending=False)
        res["worst_rows_B"] = worst.head(8).to_dict(orient="records")
        res["share_se_B_top10"] = float(worst["se_B"].head(10).sum() / worst["se_B"].sum())

        # effect on the operational fold score when combined with the Δ model's OOF
        o = oof[oof["fold"] == f.idx].set_index("id")
        present = o[~o["anchor"].isna()]
        se_present = ((present["y"] - present["pred"]) ** 2).sum()
        n_all = len(o)
        op = {}
        for name, pred in [("A", predA), ("B", predB), ("C", predC), ("D", predD), ("E", predE),
                           ("F", predF), ("G", predG), ("H", predH)]:  # fmt: skip
            se_miss = ((y - pred) ** 2).sum()
            op[name] = float(np.sqrt((se_present + se_miss) / n_all))
        res["operational_rmse_with"] = op
        res["gate_rmse_anchor_present"] = float(np.sqrt(se_present / len(present)))
        results[f.idx] = res
        print(
            f"fold {f.idx} (n={n}, PM10 present {res['pm10_present_share']:.0%}, top-10 rows carry "
            f"{res['share_se_B_top10']:.0%} of B's SE): "
            + ", ".join(
                f"{k}={v:.1f}"
                for k, v in res.items()
                if isinstance(v, float)
                and k not in ("pm10_present_share", "gate_rmse_anchor_present", "share_se_B_top10")
            )
        )
        print(
            f"   operational headline with each fallback: {', '.join(f'{k}={v:.3f}' for k, v in op.items())}"
        )
        for w in res["worst_rows_B"][:4]:
            print(f"   worst: {w}")

    OUT.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta-oof", default="lgbm_v2_delta")
    main(ap.parse_args())
