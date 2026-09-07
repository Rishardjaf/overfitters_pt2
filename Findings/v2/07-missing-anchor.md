# Finding v2-07 — Missing-anchor rows: hedge, don't trust any one predictor

**Plan:** `plans v2/07-missing-anchor.md` · **Script:** `scripts/v2/missing_anchor.py --delta-oof lgbm_lean_delta`
· **Population:** validation rows with no `current_PM2_5` (fold 0: 443, fold 1: 388, fold 2: 420,
fold 3: 546) · **Seed:** 42

## What was measured

RMSE on the missing-anchor rows only, for each fallback, and the resulting *operational* fold
RMSE when combined with the `lgbm_lean_delta` predictions on the anchor-present rows.

| fallback | fold 0 | fold 1 | fold 2 | fold 3 | heating mean | operational f0 / f2 |
|---|---:|---:|---:|---:|---:|---:|
| A — LOO network mean + station offset | 65.1 | **28.0** | **49.1** | 36.6 | 57.1 | 17.34 / **19.07** |
| B — general LightGBM level model, all features (previous fallback) | **41.3** | 38.5 | 80.3 | 34.6 | 60.8 | **16.70** / 19.92 |
| C — level model without any anchor-derived feature | 46.1 | 72.0 | 76.5 | **33.7** | 61.3 | 16.80 / 19.79 |
| C2 — C trained on heating months only | 44.5 | 83.8 | 89.4 | 46.3 | 67.0 | 16.81 / 20.21 |
| D — station fine-fraction × PM10 (A where PM10 missing) | 68.6 | 35.0 | 51.8 | 45.0 | 60.2 | 17.46 / 19.13 |
| E — mean of A and C | 48.2 | 44.3 | 58.2 | 31.8 | 53.2 | 16.85 / 19.28 |
| **F — mean of A and B** | 45.5 | 30.1 | 59.9 | 34.3 | **52.7** | 16.79 / 19.32 |
| G — B bounded to A ± 2·network std | 53.9 | 28.7 | 61.7 | 34.9 | 57.8 | 17.01 / 19.37 |
| H — median of A, B, D | 61.7 | 27.5 | 49.1 | 36.4 | 55.4 | 17.23 / 19.07 |

For scale: the Δ model scores ~16.3 / 18.6 on the *other* 99.3% of rows in those folds.

## What this says, in plain terms

1. **These rows are ten times noisier than the rest, and a handful decide the number.** The ten
   worst rows carry 37% (fold 0) and 59% (fold 2) of the level model's squared error. One row —
   Changping, 2016-02-08 01:00, truth 882 while the rest of the city sat at 615 and the level
   model said 125 — is on its own worth 30 RMSE on that fold's missing-anchor slice.
2. **No fallback wins consistently.** The level model is best on fold 0 and worst on fold 2; the
   network mean is the reverse. With 400 rows per fold and this tail, the ranking is noise.
3. **Why the level model fails on fold 2:** it was trained almost entirely on rows *with* an
   anchor, so when the anchor and every lag are NaN it routes down branches it has barely seen.
   On 2015-12-02 04:00 the whole network was out (`net_loo_mean` NaN) and it predicted 412
   against a truth of 23. Training it without anchor features (C) did not fix this.
4. **PM10 is present on only 22–41% of these rows**, so the fine-fraction idea (D) rarely applies.
5. **Averaging A and B (F) is the sensible hedge**: best mean over the heating folds (52.7), never
   catastrophic (worst 59.9 vs 80.3 for B alone), and it reduces the swing between folds.
   Differences among E/F/H are within noise; F is chosen because both components are simple and
   already in the pipeline.

## What it is worth

Switching the fallback from B to F changes the heating-fold operational headline from
(16.70 + 19.92)/2 = 18.31 to (16.79 + 19.32)/2 = **18.05**, about −0.25. On the test set the
371 missing-anchor rows (0.73%) will add roughly `sqrt(17² + 0.0073·50²) − 17 ≈ +0.5` to the
final RMSE whatever we do — this is a floor set by the data, not by the model.

## Verdict

- Fallback for the submission: **F (mean of LOO network mean + station offset, and the general
  level model)**, `--fallback F` in `scripts/v2/make_submission.py`.
- Not worth more time: the gain ceiling here is ~0.3 and the measurement noise is the same size.
