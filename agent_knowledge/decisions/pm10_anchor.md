# Decision: PM10 at the observation hour is the anchor and the baseline

- Status: validated as the project's reference; see the ADR for the measurements
- Date: 2026-09-06
- Contributor: Claude Opus 5, revised-schema rebuild
- Dataset: [revised release and fingerprints](../findings/dataset_release.md)

## Decision

`ANCHOR_COL = "PM10"`. The baseline rung is a single global least-squares line
ŷ = â + b̂·PM10ₜ, with â, b̂ and the blank-fill median fitted on training rows only.

Full reasoning and the rejected alternatives are in
[ADR-0007](../../docs/adr/0007-pm10-replaces-persistence-as-the-anchor.md). In short: a
station × hour-of-day climatology scores 101.40 against a constant's 102.04 and measures nothing,
while PM10ₜ scores 41.08; per-station coefficients buy 0.06 RMSE against a day-block standard
error of 0.2–0.4 and were rejected as noise.

## Consequences for anyone extending this

The anchor's slope, intercept and median are **fitted statistics**. Computing them over rows that
include a holdout is train/test contamination, and `src/leakage.py` is blind to it — truncating
the future does not change a statistic fitted across the present. Fit on training rows and reuse
the fitted object unchanged. `test_the_anchor_baseline_is_fitted_on_training_rows_only` pins this.

The same caution applies to every scaler, imputer, encoder, bin edge and calibration constant
added later. See [the validation policy](causal_validation.md).
