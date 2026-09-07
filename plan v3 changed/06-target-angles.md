# Plan v3-06 — Different angles: what the model is asked to predict

**Question.** With no anchor, the model predicts a heavily right-skewed level (median 55, max
999). Is the raw level the right thing to ask for, or does a transformed target learn better and
convert back to a lower RMSE on the raw scale?

All arms use the best feature set from plan 02/07 and the same fixed LightGBM parameters, so
only the target changes. Every arm is scored on the **raw scale** — the competition metric.

| arm | target | prediction | why it might help | why it might hurt |
|---|---|---|---|---|
| level | `y` | `ŷ` | the metric is L2 on this scale | trees cannot extrapolate above training-window maxima on the hardest winter |
| proxy (plan 03) | `y − k_station·PM10` | `ŷ + k·PM10` | keeps the extrapolation benefit of the old Δ trick; anchors on the strongest remaining signal | `k` varies by humidity/season; a fixed station ratio may be too crude |
| log1p | `log1p(y)` | `expm1(ŷ)` | compresses the skew, stabilises variance | optimises relative error → under-predicts spikes, exactly what RMSE punishes (PROJECT.md); tested, not assumed |
| ratio | `y ÷ PM10`, sample weight `PM10²` | `ŷ × PM10` | the fine fraction is bounded and less skewed; the `PM10²` weight makes the loss equal to L2 on the level, so it is a re-parameterisation, not a different objective | rows with missing PM10 need a same-hour network fill |
| climatology encoding | level, plus in-fold leave-one-out mean of `y` by (station, month, hour) and (station, hour) as features | `ŷ` | gives the tree the diurnal/seasonal baseline directly | target encoding is a leakage surface — must be fit inside each fold, LOO on training rows |
| heating weights | level, heating-season rows ×2 + recency | `ŷ` | test period is Sep–Feb | v2 found this within noise |

## Success criteria

- An arm is adopted only if it beats `level` by > 0.05 on both heating folds.
- The tail split (falls / calm / rises, using the diagnostic anchor) is reported for every arm —
  a transform that wins the mean by flattening spikes must be visible as such.

## Output

`Findings/v3/06-target-angles.md`
