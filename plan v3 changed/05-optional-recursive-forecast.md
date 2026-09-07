# Plan v3-05 — Optional, gated: recursive PM2.5 reconstruction

**Status: not started. Requires your explicit sign-off before any of this is built or run.**
Nothing in plans 01–04 depends on this, and none of it is part of the default submission path.

## The idea

Migration.md §4 established a fact: for training rows, `pm25(t) = target(t-1)` exactly, so the
full PM2.5 history *could* be reconstructed for training. That reconstruction also gives you
one legitimate value at test time: the very first test row per station can borrow
`pm25 = (that station's last training-row target)`. Beyond that first row, the same trick would
require substituting your own model's prediction for the previous test row wherever the true
value is unknown — predict row 1, treat that prediction as if it were the real `pm25` for
building row 2's features, predict row 2, and so on, hour by hour, for the full test period.

## Why this is separated out instead of just being "plan 06"

Three concrete reasons, not a vague caution:

1. **It works against the stated purpose of the data change.** You were told the point of
   removing PM2.5 was to stop the model leaning on it. A recursive chain of self-predictions is
   a way of re-deriving an approximation of PM2.5 anyway, just built from guesses instead of a
   sensor. Whether that's in the spirit of the exercise is a judgment call for you, not
   something to decide unilaterally in a plan document.
2. **Errors compound over a long, unbroken chain.** Each station's test period is roughly six
   months of hourly data with no fresh ground truth ever injected — one bad early prediction
   propagates into every subsequent hour's "PM2.5 lag" feature for that station, for the rest of
   the test period.
3. **It's much harder to validate honestly.** A normal CV fold holds out real historical data
   and scores against it. To measure this technique properly, the CV harness itself would need
   to simulate the same recursive blindness inside each validation fold (predict step by step,
   feed predictions forward, never peek at the true validation-fold values) rather than reusing
   the existing fold-scoring code as-is. Getting this wrong would produce a falsely optimistic
   number, silently.

## If you decide to allow it

- Build it as a fully separate, clearly labelled arm — never blended into the main submission
  without being called out as using this technique.
- Implement the recursive-blindness validation described above before trusting any score from
  it; do not reuse the plan 01–04 CV harness unmodified, since that harness assumes every
  feature is honestly computable at validation time, which this technique specifically violates.
- Report how much of any improvement comes from the first few hours (where the borrowed
  train-label value is still accurate) versus deep into the test period (where it's several
  months of self-predictions removed from any real observation) — expect it to degrade, and
  measure by how much.

## Output, if run

`Findings/v3/05-optional-recursive-forecast.md`, with the compounding-error breakdown above as
a mandatory section, not an afterthought.
