# Finding v3-08 — Residual audit: where the error concentrates

**Plan:** `task/HANDOFF.md` Step 3e · **Source:** `artifacts/v3/oof/lgbm_f9_deep_ratio.parquet`
(out-of-fold predictions, heating folds 0 and 2, no retraining — this is free) · Overall
heating headline reproduced: 25.97 (matches `Findings/v3/06`'s 25.94 within rounding).

## RMSE by station

| station | RMSE | bias (pred − actual) |
|---|---:|---:|
| Dongsi | **30.75** (worst) | +2.16 |
| Wanshouxigong | 29.63 | **−4.40** |
| Gucheng | 28.67 | −1.99 |
| Shunyi | 27.76 | −3.67 |
| Wanliu | 26.84 | **+3.44** |
| Tiantan | 26.67 | −2.14 |
| Changping | 24.61 | +3.21 |
| Nongzhanguan | 24.29 | −2.93 |
| Aotizhongxin | 23.60 | +0.31 |
| Dingling | 22.48 | −0.23 |
| Guanyuan | 22.46 | −1.30 |
| Huairou | **21.85** (best) | −0.72 |

**1.41× spread between best and worst station, and the bias is systematic, not noise** —
Wanshouxigong under-predicts by 4.4 on average, Wanliu over-predicts by 3.4. A single global
fine fraction (PM2.5 ÷ PM10) is being asked to serve stations with structurally different
ratios; the model partially learns this through `station_cat` but not completely.

## RMSE by hour of day

Worst at **00:00–02:00 (35.7 / 32.7 / 29.1)**, best at 05:00 and 13:00 (~22). Overnight hours
are hardest — nighttime temperature inversion traps pollution and decouples the PM2.5/PM10
ratio from its daytime behaviour (combustion sources differ: residential heating at night vs.
traffic by day).

## RMSE by month

Sept 14.2 (calmest, best) → **Nov 30.4, Dec 33.5 (worst)**. Deep winter is both the most
polluted and the least predictable from co-pollutants alone — consistent with the EDA's
finding that winter volatility is highest.

## What this says, in plain terms

1. **Station identity carries real, systematic signal the model is not fully using.** The bias
   table is the direct evidence: this isn't random noise, it's a per-station offset. →
   motivates Step 3f (station × PM10 interaction, tested next).
2. **Overnight hours are the weak point in time**, not just winter months. An hour-of-day ×
   season interaction (already partially present via `hour_cos`/`hour_sin` × `is_heating`) may
   be underused — worth an explicit `hour_x_heating` feature if 3f doesn't close the gap.
3. **This is a diagnostic, not a new lever on its own** — it explains *why* 3f and 3d were
   proposed, it doesn't replace running them.

## Output feeding forward

Directly motivates `plan v3 changed` Step 3f (station × PM10 interactions) and confirms the
hour/season interactions in Step 3d are pointed at a real weak spot, not a guess.
