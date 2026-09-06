# Finding: the delta-target booster fails the promotion gate

- Status: validated (the gate outcome); the variant itself is `rejected for now`, not disproved
- Confidence: high
- Date: 2026-09-06
- Contributor: Claude Opus 5, revised-schema rebuild
- Dataset: [revised release and fingerprints](dataset_release.md)

## Conclusion

`04_delta_lightgbm` — LightGBM fitted on Δ = target − (â + b̂·PM10ₜ) — beats `03_lightgbm` on the
point estimate of **both** folds, and conclusively on one. It still fails the gate, because the
primary fold's interval includes zero.

Paired day-block bootstrap, 2,000 draws, 90% interval, both rungs scored on identical rows:

| fold | lightgbm | delta_lightgbm | Δ RMSE | 90% interval | verdict |
|---|---:|---:|---:|---|---|
| `2015-09_to_2016-03` (primary) | 29.6136 | 29.1845 | −0.4291 | [−1.1044, **+0.1410**] | inconclusive |
| `2014-09_to_2015-03` | 27.4999 | 26.9550 | −0.5449 | [−0.8168, −0.2757] | conclusive |

The gate requires all three of: wins on the primary fold, does not lose on the second, and the
primary fold's 90% interval lies entirely below zero. The third fails. `03_lightgbm` remains the
submission.

## Why this was not overridden

The point estimates are unanimous and the effect is in the same direction and magnitude on both
folds, which is genuinely suggestive. It is also exactly what a real-but-small effect and a
sampling fluctuation look like from the outside, which is the whole reason the interval is
consulted instead of the point estimate. A fold holds ~182 independent day blocks, not ~51,000
independent rows; the naive row-wise standard error is about 0.07 and the honest one is 0.2–0.4,
so a 0.43 win is inside the noise.

Promoting here would set the precedent that the gate is advisory, which would make every later
comparison worth less.

## What would settle it

Re-run the same comparison with more independent blocks — additional season-matched folds from
earlier years, which the training span (from 2013-03) supports — or with a longer round budget
so the two boosters are compared at convergence rather than at a fixed 800 rounds inherited from
a different feature set. If the primary interval clears zero, promote it.

## Evidence

```bash
.venv/bin/python -m scripts.modeling.gate 03_lightgbm 04_delta_lightgbm
```

The rung is kept in the registry rather than deleted. Its design argument — that a tree cannot
predict past its largest leaf mean, and the target's tail (median 55, maximum 999 µg/m³) is
where squared error is decided — is unaffected by this result.
