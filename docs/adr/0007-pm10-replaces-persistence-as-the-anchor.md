# ADR-0007 — PM10 replaces persistence as the anchor

Accepted, 2026-09-06.

## Context

Every rung on the previous ladder was measured against persistence — the prediction that
next-hour PM2.5 equals this hour's PM2.5. The revised release deletes the column persistence
stood on, so the project lost not just a model but its unit of measurement. A replacement had to
be something available at the observation hour, cheap, and meaningful enough that "beats the
baseline" still means something.

## Decision

The baseline is a single global least-squares line through PM10 at the observation hour:
ŷ = â + b̂·PM10ₜ, with â, b̂ and the median used to fill a blank PM10 all fitted on training rows
alone. `ANCHOR_COL = "PM10"`.

## Alternatives measured

Holdout RMSE on the primary fold `2015-09_to_2016-03` (51,349 rows):

| candidate | RMSE |
|---|---:|
| train mean (constant) | 102.04 |
| station × hour-of-day climatology | 101.40 |
| global linear on PM10ₜ | **41.08** |
| per-station linear on PM10ₜ | 41.02 |

**Climatology was rejected on the evidence, not on taste.** It scores 101.40 against a constant's
102.04 — six tenths of a percent for twelve stations × twenty-four hourly means. PM2.5 in Beijing
is set by the weather regime, not by the clock, so a diurnal-seasonal average knows almost nothing
about any particular hour. It would have been a comfortable-looking baseline that measured nothing.

**Per-station coefficients were rejected as noise.** They buy 0.06 RMSE, against an honest
day-block standard error of 0.2–0.4. Twelve fitted lines to move the number by a quarter of its
own uncertainty is not an improvement, it is a more expensive way to write the same number.

## Consequences

The baseline scores 40.664 in the pipeline — slightly better than the 41.08 measured on raw rows,
because the pipeline falls back to `PM10_lag1` before the training median when PM10 is blank.

Old-schema scores are not comparable to new ones and must never be tabled together. The previous
ladder's best was 19.998 RMSE against a persistence baseline of 23.138; this release's best is
29.614 against an anchor of 40.664. The gap is the size of the leak the re-release closed, not a
regression.

The anchor also revives the design [ADR-0006 of the old
repository](https://github.com/Rishardjaf/overfitters/blob/97525ff/docs/adr/0006-delta-target-and-blending.md)
recorded: boosting on the deviation from an anchor rather than on the level, so the trees are not
asked to extrapolate past their leaf means into the tail that decides the squared error. That rung
is built (`04_delta_lightgbm`) and did not pass the promotion gate — see
`agent_knowledge/findings/delta_target_gate.md`. It is kept because it may pass on more data or a
longer round budget, and because a measured negative is a result.
