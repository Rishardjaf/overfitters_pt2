# ADR-0006 — The causal contract under the revised schema

Accepted, 2026-09-06.

## Context

[ADR-0001](0001-no-lookahead-features.md) forbids a feature from reading data after its
observation hour. On the previous release that rule had a spectacular corollary, recorded at the
time: for 99.25% of test rows the same station's row at *t+1* was present in `test.csv`, and it
carried `current_PM2_5` — which *is* the target for the row at *t*. The hidden answers were
sitting in the supplied file.

The organisers have re-released the data with `current_PM2_5` deleted and nothing else changed.
The two files are byte-equal to their predecessors after that column is removed.

The structural fact did not change with it. We measured the revised `test.csv`: **99.25% of test
rows still have their (station, *t+1*) partner row present** — the same figure, because the same
rows are there. That row no longer carries PM2.5, but it does carry PM10, CO, NO2, SO2, O3 and
the full weather panel for the target hour.

So the temptation survives in weaker form. A model could read PM10 at *t+1* — a measurement
taken during the very hour whose PM2.5 it is predicting — and no target value would ever be
touched. In a batch competition where the grader only sees predictions, this would not be
flagged as leakage by anything.

## Decision

Features for a station-hour at *t* may read only rows at times ≤ *t*, from `train.csv` and
`test.csv` alike. The partner row at *t+1* is out of bounds, and so is every other row dated
after *t*. Same-hour readings from *other* stations remain legal — those measurements genuinely
exist at *t*.

## Rationale

Two arguments, and the second is the one that settles it.

**Intent.** The organisers deleted a column for the sole purpose of closing the *t+1* read.
Reading *t+1* covariates instead is a reconstruction of the thing they just removed. It is
within the letter of a rule that no longer needs to be written down and outside its plain
purpose.

**Price.** We measured what the transgression would actually buy. Correlation with the target,
over training rows:

| predictor | at *t* | at *t+1* |
|---|---:|---:|
| PM10 | +0.848 | +0.866 |
| CO | +0.762 | +0.778 |
| NO2 | +0.643 | +0.642 |
| O3 | −0.133 | −0.128 |
| WSPM | −0.280 | −0.261 |

The strongest predictor gains **+0.018** of raw correlation; NO2 and O3 gain nothing. Air masses
move slowly enough that an hour of hindsight on PM10 is nearly redundant with PM10 itself. The
project's central principle would be traded for a rounding error.

Had the number been large, this ADR would have been harder to write and would have had to say so.
It was not.

## Consequences

`src/leakage.py` enforces the rule mechanically — it rebuilds the whole feature frame from
truncated raw data and asserts nothing changed — and it passes on the revised feature set of 140
columns. It is the reason this decision is a property of the code rather than an intention.

It cannot catch everything. A statistic *fitted* across rows that include the holdout — a median,
a scaler, a calibration constant, the anchor's own slope — is train/test contamination rather
than a lookahead within the frame, and truncating the future does not change it. Those are pinned
separately by tests; see `test_the_anchor_baseline_is_fitted_on_training_rows_only`.

Building features on the joined train-and-test timeline ([ADR-0004](0004-features-built-on-the-joined-timeline.md))
is unaffected and still correct: every test hour is later than every training hour, so a backward
lag reaching from test into train reads only the past.
