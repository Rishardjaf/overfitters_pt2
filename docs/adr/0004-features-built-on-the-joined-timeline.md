# Features are built once over the joined train-and-test timeline

The training data end at 2016-08-31 22:00 and the test window opens at 23:00 the same day —
contiguous, with no overlap and no shared ids. Building features from each file separately
would therefore give the first day of the test window empty lags, empty rolling windows and
empty deltas, purely because the hours immediately before it happen to live in the other CSV.
The `current_PM2_5_lag24` feature would be blank for every station's first twenty-four test
hours, and the deltas that separate a model from persistence would be blank for the first three.

So `src/cache.py` concatenates the two raw frames, sorts by station and hour, runs
`pipeline.build_frame` once over the joined timeline, and splits the result back apart on the
timestamp boundary. The engineered halves are cached as `data/train.parquet` and
`data/test.parquet`, which also means six algorithm runs share one sixteen-second build rather
than repeating it each time.

This does not weaken [ADR-0001](0001-no-lookahead-features.md). Every feature still reads only
times at or before its own observation hour: a test row at time *t* may see training hours
before *t*, and a training row can never see a test hour because every test hour is later than
every training hour. `src/cache.py` runs `leakage.assert_no_lookahead` against the joined frame
at four cutoffs, so the claim is checked rather than asserted.

## Consequences

The parquet files are derived competition data and are gitignored alongside the CSVs.

The persistence submission shifts very slightly against a test-only build — 52 rows of 51,063.
Those are the station-hours where `current_PM2_5` and both its lag fallbacks are blank, so the
rung falls back to the station mean, and the joined build makes a handful of boundary hours
forward-fillable that previously were not. The direction is towards more information, not less.

A reader comparing `data/train.parquet` against `train.csv` will find the parquet has more rows
— 368,628 against 360,954. The extra rows are the absent station-hours that
`grid.complete_hourly_grid` makes explicit, and they carry no id and no target, so
`pipeline.scored_rows` and `submission.build` drop them before anything is scored or written.
