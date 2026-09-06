# No feature may read data from after its observation hour

The competition's hidden test set is a near-complete hourly grid, so for 99.2% of test
station-hours the answer is simply the `current_PM2_5` of the same station's next row — 50,680
of 51,063 rows have that neighbour present, and every one of them carries a real value. The
rules forbid recovering hidden test targets, so we adopt a hard rule instead of relying on
care: every feature computed for a station-hour at time *t* may use only data at times ≤ *t*.

No backward fill, no interpolation across a gap, no centred rolling windows, no negative shifts,
and no statistic aggregated over the test period. Same-hour readings from *other* stations are
permitted, because those measurements genuinely exist at the observation hour.

## Consequences

The rule is enforced mechanically rather than by discipline. `src/leakage.py` rebuilds the
feature frame from a truncated view of the data at several cutoff times and asserts the features
on the surviving rows are byte-identical to those built from the full frame; any feature that
reads forward changes value and fails the check. The check runs inside the EDA notebook, so it
cannot be quietly skipped.

The cost is leaderboard position: entrants who exploit the leak will score near-perfectly. We
accept a worse rank for a legitimate result, and report the leak as a finding in the EDA rather
than using it.
