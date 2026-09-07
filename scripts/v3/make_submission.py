"""Retrain on all NEW training rows and write a submission (no PM2.5 anchor).

    uv run python scripts/v3/make_submission.py --name v3_lgbm_deep_ratio \
        --models lightgbm:1200 --features f9_deep --target ratio --seeds 42 1 2

Targets as in run_experiment.py. Ratio: the model predicts y / PM10 with PM10^2
sample weights and the prediction is multiplied back by PM10 (same-hour network
fill where PM10 is missing). Features for test rows come from the same
backward-only panel. Runs the leakage audit first.
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
sys.path.insert(0, str(ROOT / "scripts" / "v2"))
sys.path.insert(0, str(HERE))

from cv import TARGET, TIME_COL  # noqa: E402
from features_v3 import DIAG, assert_no_pm25, feature_sets, load_panel  # noqa: E402

from pm25.config import N_TEST_ROWS, SEED, STATION_COL  # noqa: E402

# Both runners are named run_experiment.py; load each by explicit path.
import importlib.util  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_v2 = _load("run_experiment_v2", ROOT / "scripts" / "v2" / "run_experiment.py")
_v3 = _load("run_experiment_v3", HERE / "run_experiment.py")
lgbm_default_params, make_weights = _v2.lgbm_default_params, _v2.make_weights
make_target, back_to_level, target_encode = _v3.make_target, _v3.back_to_level, _v3.target_encode

SUB_DIR = ROOT / "submissions"


def audit_or_die() -> None:
    r = subprocess.run([sys.executable, str(ROOT / "src" / "pm25" / "audit.py")], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        raise SystemExit("leakage audit failed — not writing a submission")


def fit_predict(model: str, params: dict, rounds: int, Xtr, ytr, Xte, cat, w, seeds) -> np.ndarray:
    preds = []
    for s in seeds:
        if model == "lightgbm":
            p = lgbm_default_params() | params | {"seed": int(s)}
            m = lgb.train(p, lgb.Dataset(Xtr, ytr, weight=w, categorical_feature=cat), num_boost_round=rounds)
            preds.append(m.predict(Xte))
        elif model == "xgboost":
            import xgboost as xgb

            p = {"objective": "reg:squarederror", "tree_method": "hist", "learning_rate": 0.05, "max_depth": 8,
                 "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
                 "seed": int(s), "nthread": 8, "max_cat_to_onehot": 1} | params  # fmt: skip
            m = xgb.train(p, xgb.DMatrix(Xtr, ytr, weight=w, enable_categorical=True), num_boost_round=rounds)
            preds.append(m.predict(xgb.DMatrix(Xte, enable_categorical=True)))
        elif model == "catboost":
            from catboost import CatBoostRegressor, Pool

            def prep(X):
                X = X.copy()
                for c in cat:
                    X[c] = X[c].astype(str)
                return X

            p = {"loss_function": "RMSE", "learning_rate": 0.08, "depth": 8, "l2_leaf_reg": 3.0, "random_seed": int(s),
                 "thread_count": 8, "verbose": False, "allow_writing_files": False, "border_count": 254} | params  # fmt: skip
            m = CatBoostRegressor(iterations=rounds, **p)
            m.fit(Pool(prep(Xtr), ytr, weight=w, cat_features=cat))
            preds.append(m.predict(prep(Xte)))
        else:
            raise ValueError(model)
    return np.mean(preds, axis=0)


def main(args) -> None:
    audit_or_die()
    panel, groups = load_panel()
    feats = list(feature_sets(groups)[args.features])
    for g_ in args.add_groups:
        feats += [c for c in groups[g_] if c not in feats]
    assert_no_pm25(feats)
    assert DIAG not in feats and TARGET not in feats
    cat = [c for c in ("station_cat", "wd_cat") if c in feats]
    real = panel[~panel["is_pad"]]
    train = real[~real["is_test"]].copy()
    test = real[real["is_test"]].copy()
    assert len(test) == N_TEST_ROWS and test[TARGET].isna().all()

    if args.te:
        # Same recipe as run_experiment.py's --te: LOO-smoothed encoding on the
        # fitting rows (train), plain fitted lookup on the rows being predicted
        # (test) — fit strictly on train, applied to test, no leakage.
        target_encode(train, [test], [STATION_COL, "month", "hour"], "te_smh")
        target_encode(train, [test], [STATION_COL, "hour"], "te_sh")
        feats = feats + ["te_smh", "te_sh"]
        print(f"climatology target encoding added: te_smh, te_sh ({len(feats)} cols total)")

    if args.te_ratio:
        feats = feats + _v3.add_te_ratio(train, [test])
        print(f"fine-fraction climatology encoding added: {list(_v3.TE_RATIO_KEYS)} ({len(feats)} cols total)")

    y, xw = make_target(args.target, train, train)
    w = make_weights(train, args.sample_weights, train[TIME_COL].max())
    if xw is not None:
        w = xw if w is None else w * xw

    preds = []
    for spec in args.models:
        model, _, rounds = spec.partition(":")
        rounds = int(rounds) if rounds else args.rounds
        params = args.params.get(model, {})
        print(f"training {model} x {len(args.seeds)} seeds, {rounds} rounds, {len(train):,} rows x {len(feats)} cols, "
              f"target={args.target}")
        pt = fit_predict(model, params, rounds, train[feats], y, test[feats], cat, w, args.seeds)
        preds.append(back_to_level(args.target, np.asarray(pt, float), train, test))
    wts = np.asarray(args.weights or [1.0] * len(preds), float)
    wts = wts / wts.sum()
    pred = np.sum([wi * p for wi, p in zip(wts, preds)], axis=0)
    pred = np.where(np.isnan(pred), train[TARGET].mean(), pred)
    pred = np.maximum(pred, 0.0)

    sub = pd.DataFrame({"id": test["id"].to_numpy(), TARGET: pred})
    assert sub["id"].is_unique and len(sub) == N_TEST_ROWS
    assert np.isfinite(sub[TARGET]).all() and (sub[TARGET] >= 0).all()
    raw_ids = pd.read_csv(ROOT / "new data" / "test_new.csv", usecols=["id"])["id"]
    assert set(raw_ids) == set(sub["id"])
    sub = sub.set_index("id").loc[raw_ids].reset_index()
    SUB_DIR.mkdir(exist_ok=True)
    path = SUB_DIR / f"{args.name}.csv"
    sub.to_csv(path, index=False)

    pm10 = test["PM10"].to_numpy(float)
    info = {
        "name": args.name, "models": args.models, "blend_weights": wts.round(3).tolist(), "features": args.features,
        "n_features": len(feats), "target": args.target, "seeds": list(args.seeds), "sample_weights": args.sample_weights,
        "params": args.params, "n_rows": int(len(sub)),
        "pred_mean": float(pred.mean()), "train_target_mean": float(train[TARGET].mean()),
        "pred_median": float(np.median(pred)), "pred_p99": float(np.percentile(pred, 99)), "pred_max": float(pred.max()),
        "median_pred_over_pm10": float(np.nanmedian(pred / np.where(pm10 > 0, pm10, np.nan))),
        "corr_pred_pm10": float(pd.Series(pred).corr(pd.Series(pm10))),
    }
    (SUB_DIR / f"{args.name}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--models", nargs="+", default=["lightgbm"], help="model[:rounds] ...")
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--features", default="f9_deep")
    ap.add_argument("--add-groups", nargs="*", default=[])
    ap.add_argument("--target", default="ratio", choices=["level", "proxy", "proxy_sm", "log1p", "ratio", "ratio_robust"])
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--params", type=json.loads, default={})
    ap.add_argument("--seeds", nargs="*", type=int, default=[SEED])
    ap.add_argument("--sample-weights", default="none", choices=["none", "heating", "recency", "both"])
    ap.add_argument("--te", action="store_true", help="in-fold climatology target encoding (station x month x hour, station x hour)")
    ap.add_argument("--te-ratio", action="store_true", help="fine-fraction climatology encoding (station x month x hour, station x wd, station x hour)")
    main(ap.parse_args())
