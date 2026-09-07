# Finding v2-06 — Blend, final model and the submission

**Plan:** `plans v2/06-blend-and-submission.md` · **Scripts:** `scripts/v2/blend.py`,
`scripts/v2/make_submission.py` · **Seed(s):** 42, 1, 2 · Persistence headline 21.316.

## 1. Does a fitted blend beat the best single model? Barely.

NNLS weights fitted on one heating fold, scored on the other (Δ-space, so zero weight is
persistence):

| members | fit f0 → eval f2 | fit f2 → eval f0 | best single on that fold | equal-weight |
|---|---:|---:|---:|---:|
| LightGBM + XGBoost + Ridge (`lean`) | 18.554 | 16.291 | 18.578 / 16.314 | 18.639 / 16.396 |
| LightGBM + XGBoost + Ridge (`lean_nb`) | 18.410 | 16.208 | 18.401 / 16.252 | 18.516 / 16.317 |
| LightGBM + XGBoost + CatBoost + HistGB (`lean`) | 18.455 | 16.242 | 18.475 / 16.314 | 18.455 / 16.212 |
| LightGBM ×2 seeds + XGBoost ×2 seeds (`lean_nb`) | 18.400 | 16.186 | 18.401 / 16.234 | **18.393 / 16.178** |

- Fitted weights swing wildly between folds (LightGBM 0.05–0.52, CatBoost 0.27–0.58) and beat
  equal weights by < 0.02 where they were not fitted. **Equal weights are used.**
- Ridge gets 6–9% weight and changes nothing. Dropped.
- Tree residuals correlate at 0.976–0.992 with each other: this is **variance reduction, not
  a blend**. The gain from averaging is real but small: 17.40 (single LightGBM) → 17.29
  (2 seeds × 2 models) → 17.30 (3 + 2 runs).

## 2. Round count for the final model — measured, not guessed

Validation RMSE against boosting rounds, LightGBM `lean_nb` Δ, heating+recency weights:

| lr | fold | 100 | 150 | 200 | 300 | 500 | 700 | 1000 | 1500 | best |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.05 | 0 | 16.23 | 16.17 | 16.17 | 16.19 | 16.22 | 16.25 | 16.30 | 16.36 | 16.16 @ 189 |
| 0.05 | 2 | 18.55 | 18.53 | 18.50 | 18.53 | 18.58 | 18.60 | 18.69 | 18.75 | 18.49 @ 177 |
| 0.02 | 0 | — | — | 16.25 | 16.15 | 16.12 | 16.13 | 16.14 | 16.19 | 16.12 @ 505 |
| 0.02 | 2 | — | — | 18.60 | 18.48 | 18.43 | 18.42 | 18.47 | 18.50 | 18.42 @ 523 |

- Both heating folds agree: at lr 0.05 the optimum is ~180–200 rounds and the curve is flat
  from 150 to 300; at lr 0.02 the optimum is 500–700 and 0.04–0.07 better.
- The previous-season early stopping used in every CV run chose 108–248 rounds on the heating
  folds — right on this curve. The 700–1500 it chose for the *summer* folds would be over-fit
  for winter; this is why the round count must come from a heating-season window.
- Final LightGBM: **lr 0.02, 700 rounds** (500–700 optimum × ~1.2 for 40% more training rows).
  Final XGBoost: **lr 0.05, 230 rounds** (its CV-selected 190–207 × 1.2; its lr 0.02 curve was
  not measured, so the validated setting is kept).

## 3. The submissions

Both files: `id,PM2_5_next_hour`, 51,063 rows, original `test.csv` order, leakage audit run
first, features for test rows from hours ≤ t on the concatenated timeline, Δ models trained on
the 358,488 anchor-present training rows, `max(0, current_PM2_5 + Δ̂)`, fallback F (finding 07)
for the 371 rows without an anchor, no upper clipping.

| file | recipe | CV headline of the recipe | sanity |
|---|---|---:|---|
| `submissions/v2_prelim_lgbm_xgb.csv` | LightGBM lr .05 ×300 + XGBoost lr .05 ×230, one seed, equal weight | ≈ 17.29 | mean 92.24 (anchor 92.32), Δ̂ mean +0.04, max 752 |
| **`submissions/v2_final_lgbm_xgb_seedavg.csv`** | LightGBM lr .02 ×700 + XGBoost lr .05 ×230, **3 seeds each**, equal weight | ≈ 17.25 (seed avg + lr 0.02 curve gain) | mean 92.26, Δ̂ mean +0.05, RMS(pred − anchor) 12.7, max 758, p99 443 |

The two files correlate at 0.9999 and differ by 1.3 RMS — the seed averaging and the slower
learning rate, nothing structural. **Submit `v2_final_lgbm_xgb_seedavg.csv`.**

## 4. What to expect on the leaderboard

- On the heating-season folds the recipe scores ≈ 17.25 where persistence scores 21.32:
  **−19%**.
- Persistence on the hidden test set is ≈ 19.7 (the bar quoted for this round). If the ratio
  transfers, anchor-present rows land near **16.0**. The 371 missing-anchor rows (RMSE ~50 on
  the folds) add roughly `sqrt(16² + 0.0073·50²) − 16 ≈ +0.55`.
- **Expected test RMSE: ~16.3–17.0.** Below 18 with room to spare; anything at or above 19 means
  the fold→test transfer failed and the first thing to check is the feature alignment on the
  test rows.
- The heating folds are *harder* than the test period (persistence 20.4 / 22.2 vs 19.7), so the
  absolute CV number understates the leaderboard score; the ratio is the honest transfer.

## 5. What was not done, deliberately

- No tuning against the leaderboard. One file is meant to be submitted; the preliminary exists
  only as insurance.
- No log target, no target clipping, no winsorising: PROJECT.md's reasons held up in the
  measurements (finding 01: clipping predictors alone cost a level model +4.8 on fold 2).
- No ExtraTrees / Ridge in the final (finding 04): no blend weight to earn.

## Verdict

Final recipe: **seed-averaged LightGBM (lr 0.02 × 700) + XGBoost (lr 0.05 × 230), equal
weights, `lean_nb` features (169), Δ target with anchor added back, heating+recency sample
weights, fallback F.** CV headline ≈ 17.25 vs persistence 21.32; expected leaderboard
≈ 16.3–17.0 against the 18.0 goal.

**Addendum — the 3-model file.** CatBoost on `lean_nb` scored 17.419 (finding 04), and adding
it to the equal-weight average improved both heating folds (17.299 → 17.275). A third
submission was therefore generated:

| file | recipe | CV headline |
|---|---:|---:|
| `submissions/v2_final_3model_seedavg.csv` | LightGBM lr .02 ×700 + XGBoost lr .05 ×230 + CatBoost lr .08 ×700, 3 seeds each, equal weight per model | ≈ 17.23 |

Sanity: 51,063 rows, 371 fallback rows, prediction mean 92.30 vs anchor mean 92.32, Δ̂ mean
+0.09, RMS(pred − anchor) 12.6, max 766, p99 442, no NaN, no negatives. Correlates 0.99994
with the 2-model file (1.05 RMS apart).

CatBoost's round count (700) is its CV-selected 350–846 on the heating folds × ~1.2 for the
larger training set. Expected gain over the 2-model file is ~0.02–0.03 — inside the noise band,
but in the same direction on both folds. **Submission order if attempts are limited:
`v2_final_3model_seedavg.csv` first, `v2_final_lgbm_xgb_seedavg.csv` second.**
