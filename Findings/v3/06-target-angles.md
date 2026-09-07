# Finding v3-06 — What the model predicts matters more than what it sees

**Plan:** `plan v3 changed/06-target-angles.md` + `03-proxy-anchor-residual.md` · **Model:**
LightGBM, v2 default parameters, previous-season early stopping · **Seed:** 42 · all arms
scored on the raw scale · yardstick 30.80 (LightGBM, raw columns) · target 17.0.

## Result

| features | target | fold 0 | fold 1 | fold 2 | fold 3 | **headline** | vs level (same features) |
|---|---|---:|---:|---:|---:|---:|---:|
| f7 | level | 25.91 | 19.60 | 29.89 | 19.37 | 27.90 | — |
| f7 | proxy `y − k_station·PM10` | 26.25 | 21.03 | 28.92 | 19.22 | 27.58 | −0.32 |
| f7 | log1p(y) | 26.35 | 19.46 | 33.45 | 19.79 | 29.90 | **+2.00** |
| f7 | level + climatology encoding (station×month×hour) | 26.04 | 19.96 | 30.38 | 19.36 | 28.21 | +0.31 |
| f7 | level + heating/recency weights | 25.87 | 19.58 | 30.29 | 19.16 | 28.08 | +0.18 |
| f7 | **ratio** `y ÷ PM10`, weight `PM10²` | 24.89 | 19.70 | 27.63 | 19.29 | **26.26** | **−1.64** |
| deep (f9) | level | 26.61 | 19.39 | 29.46 | 19.41 | 28.03 | — |
| deep | proxy | 26.59 | 20.80 | 29.72 | 19.21 | 28.15 | +0.12 |
| deep | proxy, `k` by station × month | 29.53 | 22.76 | 29.96 | 20.23 | 29.75 | +1.72 |
| deep | **ratio** | **24.90** | 19.80 | **26.99** | 18.95 | **25.94** | **−2.09** |
| deep | ratio, seed 1 | 24.98 | 19.53 | 27.12 | 18.84 | 26.05 | (seed band ±0.1) |
| lean (f10) | ratio | 25.10 | 19.56 | 27.19 | 19.04 | 26.14 | |
| f7 | ratio + climatology encoding | _pending_ | | | | | see `scripts/v3/summarize.py` |
| f7 | ratio + heating/recency weights | _pending_ | | | | | see `scripts/v3/summarize.py` |

Tail split on the hardest fold (fold 2), old persistence in brackets: level 61 / 20.4 / 97;
**ratio 50 / 18.7 / 89** (falls / calm / rises; persistence 77 / 9.2 / 73).

## What this says, in plain terms

1. **The ratio target is the biggest single lever of the round: −1.6 on f7, −2.1 on the deep
   set, better on both heating folds and in every regime.** The model predicts the *fine
   fraction* (PM2.5 ÷ PM10, a bounded number around 0.5–1), weighted by `PM10²` so that the
   training loss is mathematically the same squared error on the level — it is a
   re-parameterisation, not a different objective. What changes is that each leaf holds a
   fraction that *scales with PM10*, so the model can predict above the highest level in its
   training window. That is the same extrapolation problem the Δ target solved in v2
   (`Findings/v2/02`), solved multiplicatively — and fold 2, the winter that breaks the
   training range, is where the gain is largest (29.9 → 27.0).
2. **The fine-fraction features only pay under the fine-fraction target.** The deep group is
   +0.13 on the level target and −0.32 on the ratio target. Features and target have to agree
   on what the model is estimating.
3. **The additive proxy** (`y − k·PM10`) is a weaker cousin (−0.3), and a season-specific `k`
   is much worse (+1.7): a noisier anchor is a worse anchor.
4. **log1p costs +2.0** — PROJECT.md's warning, now measured: it optimises relative error and
   under-predicts exactly the spikes RMSE punishes.
5. Climatology encoding and season/recency weights do not help on the level target (the tree
   already has `station`, `month`, `hour`); the ratio-target versions were still queued when
   this was written.
6. **The round count under the ratio target is fold-dependent in a way that matters for the
   submission.** Measured curve (deep set, lr 0.05): fold 0 keeps improving to ~1,500 rounds
   (24.87) while fold 2 is best at **100** (27.04) and degrades steadily after (27.74 at 3,000).
   Averaged, ~200 rounds is the optimum (26.14 vs 26.30 at 800). More trees learn 2013–15
   patterns that do not transfer to an unusual winter; this is why a regularised arm was queued.

## Verdict

- Target: **ratio**. Feature set: **deep (f9)**. Best honest headline so far: **25.94**
  (all-fold mean 22.66), −4.9 vs the raw-column baseline, −1.7 vs the 27.6 starting point.
- Distance to 17.0: **8.9**. The old anchor's persistence would score 21.3 on these folds; we
  are 4.6 behind a baseline that had a column we do not have.
