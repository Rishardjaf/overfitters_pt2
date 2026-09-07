"""One CLI for every v3 experiment (new data, no PM2.5 anchor).

    uv run python scripts/v3/run_experiment.py --name lgbm_f1 --features f1_lags
    uv run python scripts/v3/run_experiment.py --name lgbm_f7_proxy --features f7_neighbors --target proxy

Targets (all scored on the raw scale):
    level   y
    proxy   y - k_station * PM10_filled       (k = train-fold median ratio)   -> add back
    log1p   log1p(y)                                                          -> expm1
    ratio   y / max(PM10_filled, 1), sample weight PM10_filled^2              -> multiply back

Fitters, folds, sample weights and the fold table come from scripts/v2 so the
methodology matches the v2 findings exactly. `diag_pm25_now` is used ONLY to
score (persistence yardstick, falls/calm/rises split); a hard assertion refuses
any feature list that contains it.
"""

from __future__ import annotations

import argparse
import json
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
sys.path.insert(0, str(ROOT / "scripts" / "v2"))
sys.path.insert(0, str(HERE))

from cv import TARGET, TIME_COL, folds, rmse, tail_split, window_label  # noqa: E402
from features_v3 import DIAG, assert_no_pm25, feature_sets, load_panel  # noqa: E402

from pm25.config import SEED, STATION_COL  # noqa: E402

# The v2 runner shares this file's module name; load it by explicit path so the
# import can never resolve to this file (circular) regardless of sys.path order.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("run_experiment_v2", ROOT / "scripts" / "v2" / "run_experiment.py")
_v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v2)
FITTERS, git_sha, make_weights = _v2.FITTERS, _v2.git_sha, _v2.make_weights

RESULTS_DIR = ROOT / "artifacts" / "v3" / "results"
OOF_DIR = ROOT / "artifacts" / "v3" / "oof"
CAT_COLS = ["station_cat", "wd_cat"]


# --------------------------------------------------------------------------- #
# targets
# --------------------------------------------------------------------------- #
def proxy_from(tr: pd.DataFrame, df: pd.DataFrame, by_month: bool = False) -> np.ndarray:
    """k * PM10 (same-hour network fill where PM10 is missing), k fit on `tr` only.

    k is the median fine fraction per station, or per station x month when
    `by_month` (the fraction is seasonal: winter combustion makes finer particles).
    """
    ratio = tr[TARGET] / tr["PM10_filled"].replace(0, np.nan)
    if by_month:
        k = ratio.groupby([tr[STATION_COL], tr["month"]]).median()
        idx = pd.MultiIndex.from_arrays([df[STATION_COL], df["month"]])
        kk = k.reindex(idx).to_numpy(float)
        k_station = ratio.groupby(tr[STATION_COL]).median()
        kk = np.where(np.isnan(kk), df[STATION_COL].map(k_station).to_numpy(float), kk)
    else:
        k_station = ratio.groupby(tr[STATION_COL]).median()
        kk = df[STATION_COL].map(k_station).to_numpy(float)
    p = df["PM10_filled"].to_numpy(float) * kk
    return np.where(np.isnan(p), tr[TARGET].mean(), p)


def make_target(kind: str, tr: pd.DataFrame, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (y, extra_weight) for rows of `df`, with any fitted statistic from `tr`."""
    y = df[TARGET].to_numpy(float)
    if kind == "level":
        return y, None
    if kind == "proxy":
        return y - proxy_from(tr, df), None
    if kind == "proxy_sm":
        return y - proxy_from(tr, df, by_month=True), None
    if kind == "log1p":
        return np.log1p(np.maximum(y, 0)), None
    if kind == "ratio":
        d = np.maximum(df["PM10_filled"].to_numpy(float), 1.0)
        d = np.where(np.isnan(d), tr["PM10_filled"].mean(), d)
        return y / d, d**2
    if kind == "ratio_robust":
        # PM10_robust = 0.7*own + 0.3*network median (features_v3.add_deep2) — guards
        # the fine-fraction target against a single bad PM10 reading.
        d = np.maximum(df["PM10_robust"].to_numpy(float), 1.0)
        d = np.where(np.isnan(d), tr["PM10_robust"].mean(), d)
        return y / d, d**2
    raise ValueError(kind)


def back_to_level(kind: str, pred: np.ndarray, tr: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    if kind == "level":
        return pred
    if kind == "proxy":
        return pred + proxy_from(tr, df)
    if kind == "proxy_sm":
        return pred + proxy_from(tr, df, by_month=True)
    if kind == "log1p":
        return np.expm1(pred)
    if kind == "ratio":
        d = np.maximum(df["PM10_filled"].to_numpy(float), 1.0)
        d = np.where(np.isnan(d), tr["PM10_filled"].mean(), d)
        return pred * d
    if kind == "ratio_robust":
        d = np.maximum(df["PM10_robust"].to_numpy(float), 1.0)
        d = np.where(np.isnan(d), tr["PM10_robust"].mean(), d)
        return pred * d
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
# in-fold target encoding (climatology)
# --------------------------------------------------------------------------- #
def target_encode(fit: pd.DataFrame, apply: list[pd.DataFrame], keys: list[str], name: str,
                   m: float = 20.0, col: str | None = None):
    """Leave-one-out smoothed mean of `col` (default: the target) by `keys`, fit on `fit` only."""
    col = col or TARGET
    g = fit.groupby(keys, observed=True)[col]
    s, n = g.transform("sum"), g.transform("count")
    mu = fit[col].mean()
    fit[name] = ((s - fit[col]) + m * mu) / ((n - 1) + m)  # LOO on the fitting rows
    tab = fit.groupby(keys, observed=True)[col].agg(["sum", "count"])
    tab["enc"] = (tab["sum"] + m * mu) / (tab["count"] + m)
    for df in apply:
        df[name] = df[keys].merge(tab[["enc"]], left_on=keys, right_index=True, how="left")["enc"].to_numpy()
        df[name] = df[name].fillna(mu)


# fine-fraction climatology encoding: the ratio-target analogue of --te. --te encodes the
# LEVEL by station x month x hour (helps the level target, ~0 on ratio); this encodes the
# fine fraction itself, which is what the ratio target actually predicts.
TE_RATIO_KEYS = {
    "ter_smh": [STATION_COL, "month", "hour"],
    "ter_swd": [STATION_COL, "wd_key"],
    "ter_sh": [STATION_COL, "hour"],
}


def add_te_ratio(fit: pd.DataFrame, apply: list[pd.DataFrame]) -> list[str]:
    for d in [fit, *apply]:
        d["wd_key"] = d["wd_cat"].astype(str)
    fit["_ff"] = (fit[TARGET] / np.maximum(fit["PM10_filled"], 1.0)).clip(0, 2.0)
    for name, keys in TE_RATIO_KEYS.items():
        target_encode(fit, apply, keys, name, m=50.0, col="_ff")
    fit.drop(columns="_ff", inplace=True)
    return list(TE_RATIO_KEYS)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score(va: pd.DataFrame, pred: np.ndarray) -> dict:
    y = va[TARGET].to_numpy(float)
    a = va[DIAG].to_numpy(float)
    p = np.asarray(pred, float)
    ok = ~np.isnan(a)
    r = {
        "n": int(len(y)),
        "n_anchor_missing": int((~ok).sum()),
        "rmse_all": rmse(y, p),
        "rmse_anchor_present": rmse(y[ok], p[ok]),
        "persistence_anchor_present": rmse(y[ok], a[ok]),  # what the OLD easy baseline would score
        "rmse_anchor_missing": rmse(y[~ok], p[~ok]) if (~ok).any() else float("nan"),
        "tails": tail_split(y, a, p),
    }
    r["delta_vs_persistence"] = r["rmse_anchor_present"] - r["persistence_anchor_present"]
    return r


def table(res: list[dict]) -> str:
    lines = ["| fold | window | RMSE (all rows) | RMSE anchor-present | old persistence (diag) | fall (pers.) | calm (pers.) | rise (pers.) |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
    for r in res:
        t = r["tails"]
        tag = " **(heating)**" if r["heating"] else ""
        lines.append(
            f"| {r['fold']}{tag} | {r['window']} | **{r['rmse_all']:.3f}** | {r['rmse_anchor_present']:.3f} | "
            f"{r['persistence_anchor_present']:.3f} | {t['fall']['rmse']:.1f} ({t['fall']['persistence_rmse']:.1f}) | "
            f"{t['calm']['rmse']:.2f} ({t['calm']['persistence_rmse']:.2f}) | "
            f"{t['rise']['rmse']:.1f} ({t['rise']['persistence_rmse']:.1f}) |"
        )
    heat = [r for r in res if r["heating"]]
    if heat:
        h = float(np.mean([r["rmse_all"] for r in heat]))
        lines.append("")
        lines.append(f"**Headline (mean RMSE over heating folds, all rows): {h:.3f}**  "
                     f"· mean over all folds {np.mean([r['rmse_all'] for r in res]):.3f}")
    return "\n".join(lines)


def summarise(res: list[dict]) -> dict:
    heat = [r for r in res if r["heating"]]
    return {
        "headline": float(np.mean([r["rmse_all"] for r in heat])) if heat else float("nan"),
        "headline_anchor_present": float(np.mean([r["rmse_anchor_present"] for r in heat])) if heat else float("nan"),
        "diag_persistence_headline": float(np.mean([r["persistence_anchor_present"] for r in heat])) if heat else float("nan"),
        "mean_all_folds": float(np.mean([r["rmse_all"] for r in res])),
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def run(args) -> dict:
    t0 = time.time()
    panel, groups = load_panel()
    sets = feature_sets(groups)
    feats = list(sets[args.features])
    for g_ in args.add_groups:
        feats += [c for c in groups[g_] if c not in feats]
    for g_ in args.drop_groups:
        feats = [c for c in feats if c not in set(groups[g_])]
    for c in args.drop_cols:
        feats = [f for f in feats if f != c]
    for c in args.add_cols:
        if c not in feats:
            feats.append(c)
    assert_no_pm25(feats)
    if DIAG in feats or TARGET in feats:
        raise RuntimeError("diagnostic anchor / target in feature list")
    cat = [c for c in CAT_COLS if c in feats]

    real = panel[(~panel["is_pad"]) & (~panel["is_test"])].reset_index(drop=True)
    ts = real[TIME_COL]

    results, oof_frames, importances = [], [], []
    for f in folds():
        if args.folds and f.idx not in args.folds:
            continue
        otr, ova, itr, iva = f.masks(ts)
        tr, in_tr, in_va, va = (real[m].copy() for m in (otr, itr, iva, ova))

        feats_f = list(feats)
        if args.te:
            target_encode(tr, [va], [STATION_COL, "month", "hour"], "te_smh")
            target_encode(tr, [va], [STATION_COL, "hour"], "te_sh")
            target_encode(in_tr, [in_va], [STATION_COL, "month", "hour"], "te_smh")
            target_encode(in_tr, [in_va], [STATION_COL, "hour"], "te_sh")
            feats_f += ["te_smh", "te_sh"]
        if args.te_ratio:
            feats_f += add_te_ratio(tr, [va])
            add_te_ratio(in_tr, [in_va])

        y_tr, xw_tr = make_target(args.target, tr, tr)
        y_in_tr, xw_in_tr = make_target(args.target, in_tr, in_tr)
        y_in_va, _ = make_target(args.target, in_tr, in_va)
        w_tr = make_weights(tr, args.weights, f.train_end)
        w_in = make_weights(in_tr, args.weights, f.inner_start)
        if xw_tr is not None:
            w_tr = xw_tr if w_tr is None else w_tr * xw_tr
            w_in = xw_in_tr if w_in is None else w_in * xw_in_tr

        if args.model == "mean":
            pred = np.full(len(va), tr[TARGET].mean())
            info, imp = {}, pd.Series(dtype=float)
        elif args.model == "pm10_ratio":
            pred = proxy_from(tr, va)
            info, imp = {}, pd.Series(dtype=float)
        else:
            fitter = FITTERS[args.model]
            pred_t, info, imp, _ = fitter(
                tr[feats_f], y_tr, w_tr,
                in_tr[feats_f], y_in_tr, w_in,
                in_va[feats_f], y_in_va,
                va[feats_f], dict(args.params), args.max_rounds, cat,
            )
            pred = back_to_level(args.target, np.asarray(pred_t, float), tr, va)
        pred = np.maximum(pred, 0.0)

        sc = score(va, pred) | {"fold": f.idx, "window": window_label(f), "heating": f.heating,
                                "fit_info": info, "n_train": int(len(tr))}  # fmt: skip
        results.append(sc)
        if len(imp):
            importances.append(imp.rename(f"fold{f.idx}"))
        oof_frames.append(pd.DataFrame({
            "id": va["id"].to_numpy(), TIME_COL: va[TIME_COL].to_numpy(),
            STATION_COL: va[STATION_COL].to_numpy(), "fold": f.idx,
            "y": va[TARGET].to_numpy(), "diag_anchor": va[DIAG].to_numpy(), "pred": pred,
        }))
        print(f"fold {f.idx} {window_label(f)}: RMSE {sc['rmse_all']:.3f}  "
              f"(old persistence would be {sc['persistence_anchor_present']:.3f})  {info}", flush=True)

    out = {
        "name": args.name, "model": args.model, "features": args.features, "n_features": len(feats),
        "target": args.target, "weights": args.weights, "te": args.te, "te_ratio": args.te_ratio, "params": args.params,
        "add_groups": args.add_groups, "drop_groups": args.drop_groups, "drop_cols": args.drop_cols,
        "git_sha": git_sha(), "seed": SEED, "seconds": round(time.time() - t0, 1),
        "summary": summarise(results), "folds": results, "feature_list": feats,
    }
    if importances:
        imp_df = pd.concat(importances, axis=1).fillna(0)
        imp_df["mean"] = imp_df.mean(axis=1)
        imp_df = imp_df.sort_values("mean", ascending=False)
        out["importance"] = imp_df["mean"].round(2).to_dict()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{args.name}.json").write_text(json.dumps(out, indent=1, default=str))
    pd.concat(oof_frames).to_parquet(OOF_DIR / f"{args.name}.parquet", index=False)

    print()
    print(f"## {args.name}  ({args.model}, {args.features} [{len(feats)} cols], target={args.target}, "
          f"weights={args.weights}, te={args.te}, {out['seconds']}s)")
    print(table(results))
    return out


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="lightgbm", choices=[*FITTERS, "mean", "pm10_ratio"])
    ap.add_argument("--features", default="f8_full")
    ap.add_argument("--target", default="level", choices=["level", "proxy", "proxy_sm", "log1p", "ratio", "ratio_robust"])
    ap.add_argument("--weights", default="none", choices=["none", "heating", "recency", "both"])
    ap.add_argument("--te", action="store_true", help="in-fold climatology target encoding")
    ap.add_argument("--te-ratio", action="store_true", help="in-fold fine-fraction climatology encoding")
    ap.add_argument("--params", type=json.loads, default={})
    ap.add_argument("--max-rounds", type=int, default=4000)
    ap.add_argument("--add-groups", nargs="*", default=[])
    ap.add_argument("--drop-groups", nargs="*", default=[])
    ap.add_argument("--drop-cols", nargs="*", default=[])
    ap.add_argument("--add-cols", nargs="*", default=[])
    ap.add_argument("--folds", nargs="*", type=int, default=[])
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse())
