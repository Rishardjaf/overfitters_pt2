"""Plan v2-06 steps 4-5: retrain on all training rows and write the submission.

    uv run python scripts/v2/make_submission.py --name v2_lgbm_lean \
        --models lightgbm --features lean --rounds 500 --fallback F

Δ models are trained on anchor-present training rows; predictions are
`max(0, current_PM2_5 + Δ̂)`. Rows whose anchor is missing take the fallback
chosen in finding 07. Features for test rows come from the same backward-only
panel (test rows carry their own lags from the concatenated timeline). Runs the
leakage audit first and refuses to write if it fails.
"""

from __future__ import annotations

import argparse
import json
import subprocess
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

from cv import ANCHOR, TARGET, TIME_COL  # noqa: E402
from features_v2 import feature_sets, load_panel  # noqa: E402
from run_experiment import lgbm_default_params, make_weights  # noqa: E402

from pm25.config import N_TEST_ROWS, SEED, STATION_COL  # noqa: E402

SUB_DIR = ROOT / "submissions"


def audit_or_die() -> None:
    r = subprocess.run(
        [sys.executable, str(ROOT / "src" / "pm25" / "audit.py")], capture_output=True, text=True
    )
    print(r.stdout.strip())
    if r.returncode != 0:
        raise SystemExit("leakage audit failed — not writing a submission")


def fit_predict_delta(
    model: str, params: dict, rounds: int, Xtr, ytr, Xte, cat, w=None, seeds=(SEED,)
) -> np.ndarray:
    """Average over `seeds` — cheap variance reduction for the boosted models."""
    if model == "lightgbm":
        preds = []
        for s in seeds:
            p = lgbm_default_params() | params | {"seed": int(s)}
            m = lgb.train(
                p, lgb.Dataset(Xtr, ytr, weight=w, categorical_feature=cat), num_boost_round=rounds
            )
            preds.append(m.predict(Xte))
        return np.mean(preds, axis=0)
    if model == "xgboost":
        import xgboost as xgb

        preds = []
        for s in seeds:
            p = {"objective": "reg:squarederror", "tree_method": "hist", "learning_rate": 0.05, "max_depth": 8,
                 "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
                 "seed": int(s), "nthread": 8, "max_cat_to_onehot": 1} | params  # fmt: skip
            m = xgb.train(
                p, xgb.DMatrix(Xtr, ytr, weight=w, enable_categorical=True), num_boost_round=rounds
            )
            preds.append(m.predict(xgb.DMatrix(Xte, enable_categorical=True)))
        return np.mean(preds, axis=0)
    if model == "catboost":
        from catboost import CatBoostRegressor, Pool

        def prep(X):
            X = X.copy()
            for c in cat:
                X[c] = X[c].astype(str)
            return X

        preds = []
        for s in seeds:
            p = {"loss_function": "RMSE", "learning_rate": 0.08, "depth": 8, "l2_leaf_reg": 3.0, "random_seed": int(s),
                 "thread_count": 8, "verbose": False, "allow_writing_files": False, "border_count": 254} | params  # fmt: skip
            m = CatBoostRegressor(iterations=rounds, **p)
            m.fit(Pool(prep(Xtr), ytr, weight=w, cat_features=cat))
            preds.append(m.predict(prep(Xte)))
        return np.mean(preds, axis=0)
    raise ValueError(model)


def main(args) -> None:
    audit_or_die()
    panel, groups = load_panel()
    feats = feature_sets(groups)[args.features]
    cat = [c for c in ("station_cat", "wd_cat") if c in feats]
    real = panel[~panel["is_pad"]]
    train = real[~real["is_test"]]
    test = real[real["is_test"]].copy()
    assert len(test) == N_TEST_ROWS, len(test)
    assert test[TARGET].isna().all()

    # --- Δ model(s), anchor-present training rows only ------------------------
    tr = train[train[ANCHOR].notna()]
    y_delta = (tr[TARGET] - tr[ANCHOR]).to_numpy(np.float32)
    w = make_weights(tr, args.sample_weights, tr[TIME_COL].max())
    preds = []
    for spec in args.models:
        model, _, rounds = spec.partition(":")
        rounds = int(rounds) if rounds else args.rounds
        params = args.params.get(model, {})
        print(
            f"training {model} for {rounds} rounds x {len(args.seeds)} seeds on {len(tr):,} rows x {len(feats)} features "
            f"(sample weights: {args.sample_weights})"
        )
        preds.append(
            fit_predict_delta(
                model, params, rounds, tr[feats], y_delta, test[feats], cat, w, args.seeds
            )
        )
    w = np.asarray(args.weights or [1.0] * len(preds), float)
    w = w / w.sum()
    delta_hat = np.sum([wi * p for wi, p in zip(w, preds)], axis=0)
    pred = test[ANCHOR].to_numpy(float) + delta_hat

    # --- fallback for rows without an anchor -------------------------------------
    miss = test[ANCHOR].isna().to_numpy()
    print(f"{miss.sum()} test rows have no current_PM2_5 → fallback {args.fallback}")
    offset = (train[ANCHOR] - train["net_loo_mean"]).groupby(train[STATION_COL]).median()
    predA = test["net_loo_mean"].to_numpy(float) + test[STATION_COL].map(offset).fillna(0).to_numpy(
        float
    )
    predA = np.where(np.isnan(predA), train[ANCHOR].mean(), predA)
    if args.fallback in ("B", "F", "G", "H"):
        p = lgbm_default_params() | {"num_leaves": 31}
        mB = lgb.train(p, lgb.Dataset(train[feats], train[TARGET], categorical_feature=cat), 400)
        predB = np.maximum(mB.predict(test[feats]), 0)
    if args.fallback == "A":
        fb = predA
    elif args.fallback == "B":
        fb = predB
    elif args.fallback == "F":
        fb = 0.5 * predA + 0.5 * predB
    elif args.fallback == "G":
        spread = 2.0 * test["network_PM25_std"].fillna(train["network_PM25_std"].median()).to_numpy(
            float
        )
        fb = np.clip(predB, predA - spread, predA + spread)
    elif args.fallback == "H":
        frac = (
            (train[ANCHOR] / train["PM10"].replace(0, np.nan)).groupby(train[STATION_COL]).median()
        )
        predD = test["PM10"].to_numpy(float) * test[STATION_COL].map(frac).to_numpy(float)
        predD = np.where(np.isnan(predD), predA, predD)
        fb = np.median(np.vstack([predA, predB, predD]), axis=0)
    else:
        raise ValueError(args.fallback)
    pred[miss] = fb[miss]
    pred = np.maximum(pred, 0.0)

    # --- sanity checks and write -----------------------------------------------------
    sub = pd.DataFrame({"id": test["id"].to_numpy(), TARGET: pred})
    assert sub["id"].is_unique and len(sub) == N_TEST_ROWS
    assert np.isfinite(sub[TARGET]).all() and (sub[TARGET] >= 0).all()
    raw_test_ids = pd.read_csv(ROOT / "Data" / "test.csv", usecols=["id"])["id"]
    assert set(raw_test_ids) == set(sub["id"])
    sub = sub.set_index("id").loc[raw_test_ids].reset_index()  # original test.csv order
    SUB_DIR.mkdir(exist_ok=True)
    path = SUB_DIR / f"{args.name}.csv"
    sub.to_csv(path, index=False)

    anchor = test[ANCHOR].to_numpy(float)
    ok = ~np.isnan(anchor)
    info = {
        "name": args.name,
        "models": args.models,
        "weights": w.round(3).tolist(),
        "features": args.features,
        "n_features": len(feats),
        "rounds": args.rounds,
        "params": args.params,
        "fallback": args.fallback,
        "seeds": list(args.seeds),
        "sample_weights": args.sample_weights,
        "n_rows": int(len(sub)),
        "n_fallback_rows": int(miss.sum()),
        "pred_mean": float(pred.mean()),
        "anchor_mean": float(np.nanmean(anchor)),
        "mean_delta_hat_anchor_present": float((pred[ok] - anchor[ok]).mean()),
        "rmse_pred_vs_anchor_anchor_present": float(np.sqrt(np.mean((pred[ok] - anchor[ok]) ** 2))),
        "pred_max": float(pred.max()),
        "pred_p99": float(np.percentile(pred, 99)),
    }
    (SUB_DIR / f"{args.name}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--models", nargs="+", default=["lightgbm"], help="model[:rounds] ...")
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--features", default="lean")
    ap.add_argument("--rounds", type=int, default=500)
    ap.add_argument(
        "--params", type=json.loads, default={}, help='{"lightgbm": {...}, "xgboost": {...}}'
    )
    ap.add_argument("--fallback", default="F", choices=list("ABFGH"))
    ap.add_argument("--seeds", nargs="*", type=int, default=[SEED])
    ap.add_argument(
        "--sample-weights", default="none", choices=["none", "heating", "recency", "both"]
    )
    main(ap.parse_args())
