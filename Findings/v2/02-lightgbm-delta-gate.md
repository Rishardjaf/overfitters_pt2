# Finding v2-02 — LightGBM on the jihad features: Δ wins the gate

**Plan:** `plans v2/02-lightgbm-delta-gate.md` · **Features:** `jihad` (124) · **Model:**
LightGBM, L2 objective, `num_leaves 63`, `min_data_in_leaf 100`, `lr 0.05`, feature/bagging
fraction 0.8, `lambda_l2 1`; round count chosen on the previous-season inner window then refit
on the whole training fold · **Seed:** 42

## Result

| arm | fold 0 | fold 1 | fold 2 | fold 3 | **headline** | Δ vs persistence | operational |
|---|---:|---:|---:|---:|---:|---:|---:|
| persistence | 20.385 | 15.706 | 22.247 | 15.242 | 21.316 | — | 21.901 |
| Ridge Δ (finding 01) | 17.664 | 14.022 | 20.115 | 13.630 | 18.890 | −2.43 | 19.629 |
| LightGBM, raw level | 17.469 | 13.455 | **21.333** | 13.226 | 19.401 | −1.92 | 20.142 |
| LightGBM, Δ target | **17.181** | **13.254** | **19.401** | **13.234** | **18.291** | **−3.03** | 19.060 |

Signed-tail split on the heating folds (persistence in brackets; fall = Δ ≤ −30, rise = Δ ≥ 30):

| arm | fold | fall RMSE (n) | calm RMSE (n) | rise RMSE (n) | missing-anchor RMSE (n) |
|---|---|---:|---:|---:|---:|
| LightGBM level | 0 | 48.9 (70.0) (2,083) | **10.5** (9.8) (46,619) | 56.5 (61.3) (1,654) | 43.2 (443) |
| LightGBM level | 2 | 57.1 (77.1) (1,954) | **11.6** (9.2) (47,170) | **76.1** (72.8) (1,805) | 79.5 (420) |
| LightGBM Δ | 0 | **48.8** (70.0) | 10.1 (9.8) | **56.0** (61.3) | 40.0 (443) |
| LightGBM Δ | 2 | **53.0** (77.1) | 9.8 (9.2) | **71.2** (72.8) | 79.3 (420) |

Rounds chosen: fold 0 → 86, fold 1 → 300, fold 2 → 137, fold 3 → 532.

## What this says, in plain terms

1. **Δ beats the level target by 1.1 RMSE on the headline and wins on 3 of 4 folds**; the gap is
   entirely fold 2 (19.40 vs 21.33). Winter 2015-16 has levels above anything in the training
   window; a tree predicting the *level* cannot extrapolate past its last leaf, a tree
   predicting the *change* can. The gate is resolved: **every later plan trains on Δ.**
2. **The tree beats the best Ridge by 0.6** on the same features — the non-linear, regime-
   dependent structure the EDA predicted (F7) is real and worth 0.6 RMSE.
3. **Where the gain comes from — and where it is paid for.** Almost all of it is rapid *falls*:
   70→49 and 77→53. Rapid rises improve only modestly (61→56, 73→71) — the onset of a spike is
   still largely unpredicted. And calm hours get slightly **worse** than persistence
   (9.8→10.1, 9.2→9.8): the model pays a small price on 92% of rows to win big on the 8% in
   the tails. That is the right trade under RMSE, but it means a "smoother" model (more
   regularisation, higher `min_data_in_leaf`) may improve calm hours while giving back the
   tail gains — plan 05 must watch both. The level arm shows the failure mode: it made calm
   hours worse still (11.6) *and* made rises worse than persistence (76.1 vs 72.8) on fold 2.
4. **The small round counts are right, not a bug.** Fold 0's inner window leaves only six months
   of inner training data and early stopping chose 47 rounds → 86 after scaling; fold 2 chose
   137; fold 3 chose 532. I suspected under-training and tested a fixed 600 rounds
   (`--params '{"fixed_rounds":600}'`): fold 0 got *worse* (17.18 → 17.48) and so did fold 2
   (19.40 → 19.70), while fold 1 was unchanged (13.25 → 13.23). More trees overfit the
   summer-heavy training window and transfer worse into winter. The previous-season early
   stopping is doing exactly its job; stronger regularisation is the direction to explore, not
   more capacity.
5. Missing-anchor rows again cost ~0.77 on the operational number (18.29 → 19.06). Same lever as
   in finding 01.

## Verdict

- **Gate: Δ.** Margin 1.1 RMSE, same direction on folds 0, 1, 2; fold 3 tied (13.23 vs 13.23).
- LightGBM Δ on the existing features: **18.291**, −3.03 (−14%) vs persistence. If the ratio
  transfers to the test set (persistence ≈ 19.7) this alone lands near **17.0**.
- No fold below 8: no leakage signal.
- Unblocks plan 03 (feature set v2), 04 (model zoo), 05 (rounds / weights).
