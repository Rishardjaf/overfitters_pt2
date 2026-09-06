"""One CLI for every v2 experiment.

    uv run python scripts/v2/run_experiment.py --name lgbm_v2_delta \
        --model lightgbm --features v2 --target delta

Writes artifacts/v2/results/<name>.json, artifacts/v2/oof/<name>.parquet and
prints the fold table. Every fitted statistic (imputer, scaler, encoder, round
count, alpha) is fitted inside the outer training fold.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from cv import (
    ANCHOR,
    TARGET,
    TIME_COL,
    fold_table,
    folds,
    rmse,
    score_fold,
    summarise,
    window_label,
)  # noqa: E402
from features_v2 import feature_sets, load_panel  # noqa: E402

from pm25.config import SEED, STATION_COL  # noqa: E402

RESULTS_DIR = ROOT / "artifacts" / "v2" / "results"
OOF_DIR = ROOT / "artifacts" / "v2" / "oof"
CAT_COLS = ["station_cat", "wd_cat"]


def git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT)
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------- #
# sample weights
# --------------------------------------------------------------------------- #
def make_weights(df: pd.DataFrame, scheme: str, train_end: pd.Timestamp) -> np.ndarray | None:
    if scheme == "none":
        return None
    w = np.ones(len(df))
    if scheme in ("heating", "both"):
        w *= np.where(df["is_heating_season"].to_numpy() == 1, 2.0, 1.0)
    if scheme in ("recency", "both"):
        ts = df[TIME_COL]
        frac = (ts - ts.min()) / (train_end - ts.min())
        w *= 0.5 + frac.clip(0, 1).to_numpy()
    return w


# --------------------------------------------------------------------------- #
# model adapters. Each returns predictions for X_valid (on the target scale).
# --------------------------------------------------------------------------- #
def lgbm_default_params() -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": SEED,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": 8,
    }


def fit_lightgbm(
    Xtr, ytr, wtr, Xin_tr, yin_tr, win_tr, Xin_va, yin_va, Xva, params, max_rounds, cat
):
    import lightgbm as lgb

    params = dict(params)
    fixed_rounds = params.pop("fixed_rounds", None)
    p = lgbm_default_params() | params
    info = {}
    if fixed_rounds:
        # a round count fixed in advance (no data-dependent choice at all)
        n_rounds = int(fixed_rounds)
    else:
        # 1) choose the round count on the inner (previous-season) window
        dtr = lgb.Dataset(
            Xin_tr, yin_tr, weight=win_tr, categorical_feature=cat, free_raw_data=False
        )
        dva = lgb.Dataset(
            Xin_va, yin_va, reference=dtr, categorical_feature=cat, free_raw_data=False
        )
        inner = lgb.train(
            p,
            dtr,
            num_boost_round=max_rounds,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(150, verbose=False)],
        )
        n_rounds = max(50, inner.best_iteration)
        # scale up slightly for the larger outer training set
        n_rounds = int(n_rounds * (len(Xtr) / max(1, len(Xin_tr))) ** 0.5)
        info = {"inner_best": inner.best_iteration,
                "inner_rmse": float(inner.best_score["valid_0"]["rmse"])}  # fmt: skip
    # 2) refit on the full outer training fold
    dfull = lgb.Dataset(Xtr, ytr, weight=wtr, categorical_feature=cat, free_raw_data=False)
    model = lgb.train(p, dfull, num_boost_round=n_rounds)
    pred = model.predict(Xva)
    imp = pd.Series(model.feature_importance("gain"), index=Xtr.columns).sort_values(
        ascending=False
    )
    return pred, {"n_rounds": n_rounds} | info, imp, model


def fit_xgboost(
    Xtr, ytr, wtr, Xin_tr, yin_tr, win_tr, Xin_va, yin_va, Xva, params, max_rounds, cat
):
    import xgboost as xgb

    p = {
        "objective": "reg:squarederror", "eval_metric": "rmse", "tree_method": "hist",
        "learning_rate": 0.05, "max_depth": 8, "min_child_weight": 50, "subsample": 0.8,
        "colsample_bytree": 0.8, "reg_lambda": 1.0, "seed": SEED, "nthread": 8,
        "max_cat_to_onehot": 1,
    } | params  # fmt: skip
    dtr = xgb.DMatrix(Xin_tr, yin_tr, weight=win_tr, enable_categorical=True)
    dva = xgb.DMatrix(Xin_va, yin_va, enable_categorical=True)
    inner = xgb.train(p, dtr, num_boost_round=max_rounds, evals=[(dva, "inner")],
                      early_stopping_rounds=150, verbose_eval=False)  # fmt: skip
    n_rounds = max(50, inner.best_iteration + 1)
    n_rounds = int(n_rounds * (len(Xtr) / max(1, len(Xin_tr))) ** 0.5)
    dfull = xgb.DMatrix(Xtr, ytr, weight=wtr, enable_categorical=True)
    model = xgb.train(p, dfull, num_boost_round=n_rounds)
    pred = model.predict(xgb.DMatrix(Xva, enable_categorical=True))
    gain = model.get_score(importance_type="gain")
    imp = pd.Series({c: gain.get(c, 0.0) for c in Xtr.columns}).sort_values(ascending=False)
    return pred, {"n_rounds": n_rounds, "inner_best": inner.best_iteration}, imp, model


def fit_catboost(
    Xtr, ytr, wtr, Xin_tr, yin_tr, win_tr, Xin_va, yin_va, Xva, params, max_rounds, cat
):
    from catboost import CatBoostRegressor, Pool

    def prep(X):
        X = X.copy()
        for c in cat:
            X[c] = X[c].astype(str)
        return X

    p = {"loss_function": "RMSE", "learning_rate": 0.08, "depth": 8, "l2_leaf_reg": 3.0,
         "random_seed": SEED, "thread_count": 8, "verbose": False, "allow_writing_files": False,
         "border_count": 254} | params  # fmt: skip
    inner = CatBoostRegressor(iterations=max_rounds, **p)
    inner.fit(Pool(prep(Xin_tr), yin_tr, weight=win_tr, cat_features=cat),
              eval_set=Pool(prep(Xin_va), yin_va, cat_features=cat),
              early_stopping_rounds=150)  # fmt: skip
    n_rounds = max(50, inner.get_best_iteration() + 1)
    n_rounds = int(n_rounds * (len(Xtr) / max(1, len(Xin_tr))) ** 0.5)
    model = CatBoostRegressor(iterations=n_rounds, **p)
    model.fit(Pool(prep(Xtr), ytr, weight=wtr, cat_features=cat))
    pred = model.predict(prep(Xva))
    imp = pd.Series(model.get_feature_importance(), index=Xtr.columns).sort_values(ascending=False)
    return pred, {"n_rounds": n_rounds, "inner_best": inner.get_best_iteration()}, imp, model


def fit_histgb(Xtr, ytr, wtr, Xin_tr, yin_tr, win_tr, Xin_va, yin_va, Xva, params, max_rounds, cat):
    from sklearn.ensemble import HistGradientBoostingRegressor

    def prep(X):
        X = X.copy()
        for c in cat:
            X[c] = X[c].cat.codes.replace(-1, np.nan).astype(float)
        return X

    cat_mask = [c in cat for c in Xtr.columns]
    p = {"learning_rate": 0.05, "max_leaf_nodes": 63, "min_samples_leaf": 100,
         "l2_regularization": 1.0, "random_state": SEED} | params  # fmt: skip
    inner = HistGradientBoostingRegressor(
        max_iter=max_rounds,
        early_stopping=True,
        validation_fraction=None,
        n_iter_no_change=100,
        categorical_features=cat_mask,
        **p,
    )
    # sklearn has no external eval set; emulate by fitting on inner train with its own
    # trailing-time validation: pass inner window appended and rely on n_iter chosen there.
    inner.set_params(early_stopping=False)
    best, best_rmse = None, np.inf
    for n in [200, 400, 800, 1200]:
        m = HistGradientBoostingRegressor(
            max_iter=n, early_stopping=False, categorical_features=cat_mask, **p
        )
        m.fit(prep(Xin_tr), yin_tr, sample_weight=win_tr)
        r = rmse(yin_va, m.predict(prep(Xin_va)))
        if r < best_rmse:
            best, best_rmse = n, r
    n_rounds = int(best * (len(Xtr) / max(1, len(Xin_tr))) ** 0.5)
    model = HistGradientBoostingRegressor(
        max_iter=n_rounds, early_stopping=False, categorical_features=cat_mask, **p
    )
    model.fit(prep(Xtr), ytr, sample_weight=wtr)
    pred = model.predict(prep(Xva))
    return pred, {"n_rounds": n_rounds, "inner_best": best}, pd.Series(dtype=float), model


def fit_extratrees(
    Xtr, ytr, wtr, Xin_tr, yin_tr, win_tr, Xin_va, yin_va, Xva, params, max_rounds, cat
):
    from sklearn.ensemble import ExtraTreesRegressor

    def prep(X, med):
        X = X.copy()
        for c in cat:
            X[c] = X[c].cat.codes.astype(float)
        return X.fillna(med)

    med = Xtr.drop(columns=cat).median(numeric_only=True)
    p = {"n_estimators": 300, "min_samples_leaf": 20, "max_features": 0.5, "n_jobs": 8,
         "random_state": SEED} | params  # fmt: skip
    model = ExtraTreesRegressor(**p)
    model.fit(prep(Xtr, med), ytr, sample_weight=wtr)
    pred = model.predict(prep(Xva, med))
    imp = pd.Series(model.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
    return pred, {}, imp, model


def fit_ridge(Xtr, ytr, wtr, Xin_tr, yin_tr, win_tr, Xin_va, yin_va, Xva, params, max_rounds, cat):
    """Ridge with imputation, robust scaling, one-hot categoricals, optional clipping.

    Everything is fitted on the training rows passed in. Alpha is chosen on the
    inner window, then the pipeline is refitted on the outer training fold.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, RobustScaler

    clip = params.get("clip", False)
    alphas = params.get("alphas", [0.1, 1.0, 10.0, 100.0, 1000.0])
    num = [c for c in Xtr.columns if c not in cat]

    from sklearn.base import BaseEstimator, TransformerMixin

    class Clipper(BaseEstimator, TransformerMixin):
        """Clip predictors at train-fold percentiles: linear models are leverage-sensitive."""

        def fit(self, X, y=None):
            X = np.asarray(X, dtype=float)
            self.lo_ = np.nanpercentile(X, 0.5, axis=0)
            self.hi_ = np.nanpercentile(X, 99.5, axis=0)
            self.n_features_in_ = X.shape[1]
            return self

        def transform(self, X):
            return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

        def get_feature_names_out(self, input_features=None):
            return np.asarray(input_features)

    def make(alpha):
        steps = []
        if clip:
            steps.append(("clip", Clipper()))
        steps += [("impute", SimpleImputer(strategy="median")), ("scale", RobustScaler())]
        pre = ColumnTransformer(
            [
                ("num", Pipeline(steps), num),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
            ]
        )
        return Pipeline([("pre", pre), ("ridge", Ridge(alpha=alpha, random_state=SEED))])

    def prep(X):
        X = X.copy()
        for c in cat:
            X[c] = X[c].astype(str)
        X[num] = X[num].astype(np.float64).replace([np.inf, -np.inf], np.nan)
        return X

    Xin_tr_, Xin_va_, Xtr_, Xva_ = prep(Xin_tr), prep(Xin_va), prep(Xtr), prep(Xva)
    scores = {}
    for a in alphas:
        m = make(a).fit(Xin_tr_, yin_tr, ridge__sample_weight=win_tr)
        scores[a] = rmse(yin_va, m.predict(Xin_va_))
    best = min(scores, key=scores.get)
    model = make(best).fit(Xtr_, ytr, ridge__sample_weight=wtr)
    pred = model.predict(Xva_)
    coef = model.named_steps["ridge"].coef_
    names = model.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(np.abs(coef), index=names).sort_values(ascending=False)
    return pred, {"alpha": best, "inner_scores": scores}, imp, model


FITTERS = {
    "lightgbm": fit_lightgbm,
    "xgboost": fit_xgboost,
    "catboost": fit_catboost,
    "histgb": fit_histgb,
    "extratrees": fit_extratrees,
    "ridge": fit_ridge,
}


# --------------------------------------------------------------------------- #
# fallback level model for rows whose anchor is missing (Δ undefined there)
# --------------------------------------------------------------------------- #
def fallback_level_predictions(Xtr, ytr, Xva, cat) -> np.ndarray:
    import lightgbm as lgb

    p = lgbm_default_params() | {"num_leaves": 31}
    d = lgb.Dataset(Xtr, ytr, categorical_feature=cat, free_raw_data=False)
    m = lgb.train(p, d, num_boost_round=400)
    return m.predict(Xva)


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #
def run(args) -> dict:
    t0 = time.time()
    panel, groups = load_panel()
    sets = feature_sets(groups)
    feats = list(sets[args.features])
    for g in args.drop_group:
        drop = set(groups[g]) if g in groups else set()
        feats = [c for c in feats if c not in drop]
    for c in args.drop_cols:
        feats = [c_ for c_ in feats if c_ != c]
    for c in args.add_cols:
        if c not in feats:
            feats.append(c)
    cat = [c for c in CAT_COLS if c in feats]

    real = panel[(~panel["is_pad"]) & (~panel["is_test"])].reset_index(drop=True)
    ts = real[TIME_COL]
    have_anchor = real[ANCHOR].notna()

    results, oof_frames, importances = [], [], []
    for f in folds():
        if args.folds and f.idx not in args.folds:
            continue
        otr, ova, itr, iva = f.masks(ts)
        if args.target == "delta":
            y_all = (real[TARGET] - real[ANCHOR]).to_numpy(np.float32)
            fit_mask = have_anchor
        else:
            y_all = real[TARGET].to_numpy(np.float32)
            fit_mask = pd.Series(True, index=real.index)

        tr = real[otr & fit_mask]
        in_tr = real[itr & fit_mask]
        in_va = real[iva & fit_mask]
        va = real[ova]

        w_tr = make_weights(tr, args.weights, f.train_end)
        w_in = make_weights(in_tr, args.weights, f.inner_start)

        if args.model == "persistence":
            pred = va[ANCHOR].to_numpy(float)
            info, imp = {}, pd.Series(dtype=float)
        else:
            fitter = FITTERS[args.model]
            pred_t, info, imp, _ = fitter(
                tr[feats],
                y_all[tr.index],
                w_tr,
                in_tr[feats],
                y_all[in_tr.index],
                w_in,
                in_va[feats],
                y_all[in_va.index],
                va[feats],
                args.params,
                args.max_rounds,
                cat,
            )
            pred = np.asarray(pred_t, float)
            if args.target == "delta":
                pred = pred + va[ANCHOR].to_numpy(float)
                miss = va[ANCHOR].isna().to_numpy()
                if miss.any():
                    lvl_tr = real[otr]
                    fb = fallback_level_predictions(
                        lvl_tr[feats],
                        real.loc[lvl_tr.index, TARGET].to_numpy(np.float32),
                        va.loc[miss, feats],
                        cat,
                    )
                    pred[miss] = fb
            pred = np.maximum(pred, 0.0)

        # persistence itself has no answer for missing anchors: use the network LOO mean
        if args.model == "persistence":
            miss = np.isnan(pred)
            pred[miss] = (
                va.loc[miss, "net_loo_mean"].fillna(real.loc[otr, ANCHOR].mean()).to_numpy()
            )

        sc = score_fold(va, pred)
        sc |= {"fold": f.idx, "window": window_label(f), "heating": f.heating, "fit_info": info,
               "n_train": int(len(tr)), "n_inner_train": int(len(in_tr)), "n_inner_valid": int(len(in_va))}  # fmt: skip
        results.append(sc)
        if len(imp):
            importances.append(imp.rename(f"fold{f.idx}"))
        oof_frames.append(
            pd.DataFrame(
                {
                    "id": va["id"].to_numpy(),
                    TIME_COL: va[TIME_COL].to_numpy(),
                    STATION_COL: va[STATION_COL].to_numpy(),
                    "fold": f.idx,
                    "y": va[TARGET].to_numpy(),
                    "anchor": va[ANCHOR].to_numpy(),
                    "pred": pred,
                }
            )
        )
        print(
            f"fold {f.idx} {window_label(f)}: model {sc['rmse_anchor_present']:.3f} "
            f"persistence {sc['persistence_anchor_present']:.3f} "
            f"Δ {sc['delta_vs_persistence']:+.3f}  {info}",
            flush=True,
        )

    summary = summarise(results) if any(r["heating"] for r in results) else {}
    out = {
        "name": args.name,
        "model": args.model,
        "features": args.features,
        "n_features": len(feats),
        "target": args.target,
        "weights": args.weights,
        "params": args.params,
        "drop_group": args.drop_group,
        "drop_cols": args.drop_cols,
        "add_cols": args.add_cols,
        "git_sha": git_sha(),
        "seed": SEED,
        "seconds": round(time.time() - t0, 1),
        "summary": summary,
        "folds": results,
        "feature_list": feats,
    }
    if importances:
        imp_df = pd.concat(importances, axis=1).fillna(0)
        imp_df["mean"] = imp_df.mean(axis=1)
        imp_df = imp_df.sort_values("mean", ascending=False)
        out["importance_top40"] = imp_df["mean"].head(40).round(2).to_dict()
        out["importance_bottom20"] = imp_df["mean"].tail(20).round(4).to_dict()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{args.name}.json").write_text(json.dumps(out, indent=1, default=str))
    pd.concat(oof_frames).to_parquet(OOF_DIR / f"{args.name}.parquet", index=False)

    print()
    print(
        f"## {args.name}  ({args.model}, {args.features} [{len(feats)} cols], target={args.target}, "
        f"weights={args.weights}, {out['seconds']}s)"
    )
    print(fold_table(results))
    return out


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="lightgbm", choices=[*FITTERS, "persistence"])
    ap.add_argument(
        "--features", default="v2", choices=["jihad", "v2", "lean", "lean_nb", "slim_nb"]
    )
    ap.add_argument("--target", default="delta", choices=["delta", "level"])
    ap.add_argument("--weights", default="none", choices=["none", "heating", "recency", "both"])
    ap.add_argument("--params", type=json.loads, default={})
    ap.add_argument("--max-rounds", type=int, default=4000)
    ap.add_argument("--drop-group", nargs="*", default=[])
    ap.add_argument("--drop-cols", nargs="*", default=[])
    ap.add_argument("--add-cols", nargs="*", default=[])
    ap.add_argument("--folds", nargs="*", type=int, default=[])
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse())
