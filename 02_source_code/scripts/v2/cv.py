"""Time-aware folds and the scoring report used by every v2 experiment.

Folds come from `pm25.validation.EXPANDING_FOLDS`; every boundary is a
timestamp. The headline is the mean RMSE over the heating-season folds (0, 2).
Every score is quoted next to persistence on the same rows.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pm25.config import ANCHOR, TARGET, TIME_COL  # noqa: E402
from pm25.validation import EXPANDING_FOLDS, HEATING_SEASON_FOLDS  # noqa: E402

TAIL = 30.0


@dataclass(frozen=True)
class Fold:
    idx: int
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp

    @property
    def inner_start(self) -> pd.Timestamp:
        """Same window one year earlier — for choosing the round count honestly."""
        return self.valid_start - pd.DateOffset(years=1)

    @property
    def inner_end(self) -> pd.Timestamp:
        return self.valid_end - pd.DateOffset(years=1)

    @property
    def heating(self) -> bool:
        return self.idx in HEATING_SEASON_FOLDS

    def masks(self, ts: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """(outer_train, outer_valid, inner_train, inner_valid) boolean masks."""
        outer_train = ts <= self.train_end
        outer_valid = (ts >= self.valid_start) & (ts <= self.valid_end)
        inner_valid = (ts >= self.inner_start) & (ts <= self.inner_end)
        inner_train = ts < self.inner_start
        return outer_train, outer_valid, inner_train, inner_valid


def folds() -> list[Fold]:
    return [
        Fold(i, pd.Timestamp(a), pd.Timestamp(b), pd.Timestamp(c))
        for i, (a, b, c) in enumerate(EXPANDING_FOLDS)
    ]


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def tail_split(y: np.ndarray, anchor: np.ndarray, p: np.ndarray) -> dict[str, dict]:
    """RMSE / n / squared-error share for falls (Δ<=-30), calm, rises (Δ>=30).

    Only defined where the anchor is present.
    """
    y = np.asarray(y, float)
    a = np.asarray(anchor, float)
    p = np.asarray(p, float)
    ok = ~np.isnan(a)
    d = y[ok] - a[ok]
    se = (y[ok] - p[ok]) ** 2
    se_pers = d**2
    total = se.sum()
    out = {}
    for name, m in [("fall", d <= -TAIL), ("calm", np.abs(d) < TAIL), ("rise", d >= TAIL)]:
        out[name] = {
            "n": int(m.sum()),
            "rmse": float(np.sqrt(se[m].mean())) if m.any() else float("nan"),
            "persistence_rmse": float(np.sqrt(se_pers[m].mean())) if m.any() else float("nan"),
            "se_share": float(se[m].sum() / total) if total > 0 else float("nan"),
        }
    return out


def score_fold(valid: pd.DataFrame, pred: np.ndarray) -> dict:
    """Score one fold's predictions on the real validation rows.

    Reports the gate population (anchor present) and the operational population
    (every row), plus persistence on the gate population.
    """
    y = valid[TARGET].to_numpy(float)
    a = valid[ANCHOR].to_numpy(float)
    p = np.asarray(pred, float)
    ok = ~np.isnan(a)
    res = {
        "n": int(len(y)),
        "n_anchor_missing": int((~ok).sum()),
        "rmse_all": rmse(y, p),
        "rmse_anchor_present": rmse(y[ok], p[ok]),
        "persistence_anchor_present": rmse(y[ok], a[ok]),
        "rmse_anchor_missing": rmse(y[~ok], p[~ok]) if (~ok).any() else float("nan"),
        "tails": tail_split(y, a, p),
    }
    res["delta_vs_persistence"] = res["rmse_anchor_present"] - res["persistence_anchor_present"]
    return res


def summarise(fold_results: list[dict]) -> dict:
    heat = [r for r in fold_results if r["heating"]]
    return {
        "headline": float(np.mean([r["rmse_anchor_present"] for r in heat])),
        "headline_persistence": float(np.mean([r["persistence_anchor_present"] for r in heat])),
        "headline_operational": float(np.mean([r["rmse_all"] for r in heat])),
        "mean_all_folds": float(np.mean([r["rmse_anchor_present"] for r in fold_results])),
    }


def fold_table(fold_results: list[dict]) -> str:
    lines = [
        "| fold | window | n | model RMSE | persistence | Δ vs persistence | fall RMSE (pers.) | calm RMSE (pers.) | rise RMSE (pers.) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in fold_results:
        t = r["tails"]
        tag = " **(heating)**" if r["heating"] else ""
        lines.append(
            f"| {r['fold']}{tag} | {r['window']} | {r['n']:,} | {r['rmse_anchor_present']:.3f} | "
            f"{r['persistence_anchor_present']:.3f} | {r['delta_vs_persistence']:+.3f} | "
            f"{t['fall']['rmse']:.1f} ({t['fall']['persistence_rmse']:.1f}) | "
            f"{t['calm']['rmse']:.2f} ({t['calm']['persistence_rmse']:.2f}) | "
            f"{t['rise']['rmse']:.1f} ({t['rise']['persistence_rmse']:.1f}) |"
        )
    s = summarise(fold_results)
    lines.append("")
    lines.append(
        f"**Headline (mean of heating folds, anchor-present rows): {s['headline']:.3f}** "
        f"vs persistence {s['headline_persistence']:.3f} "
        f"(Δ {s['headline'] - s['headline_persistence']:+.3f}). "
        f"Operational (all rows incl. missing anchor): {s['headline_operational']:.3f}. "
        f"Mean over all four folds: {s['mean_all_folds']:.3f}."
    )
    return "\n".join(lines)


def window_label(f: Fold) -> str:
    return f"{f.valid_start:%Y-%m} → {f.valid_end:%Y-%m}"


__all__ = [
    "Fold", "folds", "rmse", "tail_split", "score_fold", "summarise", "fold_table",
    "window_label", "TIME_COL", "TARGET", "ANCHOR",
]  # fmt: skip
