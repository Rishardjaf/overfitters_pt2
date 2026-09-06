# Validation folds are season-matched blocks, not a chronological tail

The hidden test window is 2016-08-31 to 2017-02-28 — autumn and winter, which is Beijing's
heavy-pollution season and contains the municipal heating season. The training data ends where
the test data begins, so the conventional "hold out the last N weeks" split would validate on
spring and summer: clean air, low variance, and a score that says little about performance on
the actual test period.

We therefore validate on **season-matched blocks** — the primary fold is Sep 2015 to Feb 2016,
the second is Sep 2014 to Feb 2015 — each trained only on the data preceding its block, with a
final refit on all available data before submitting.

## Consequences

A future reader looking at `src/validation.py` will see holdout dates that are neither the end
of the data nor a random sample, and would otherwise assume it a mistake. Changing the scheme
invalidates every score recorded against it, so the ladder must be re-run in full if it changes.

The trade-off accepted: less training data per fold than a tail split would give, in exchange
for a validation score that actually predicts test-set behaviour.
