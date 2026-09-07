# Migration — old data/plans/findings → new data

## 1. What actually changed (verified, not assumed)

I diffed the files rather than guessing. The result is the cleanest possible case:

```
diff <(cut old train, drop column 8) new_train  →  IDENTICAL, every row, every value
diff <(cut old test,  drop column 8) new_test   →  IDENTICAL, every row, every value
```

**`new data/train_new.csv` and `new data/test_new.csv` are byte-for-byte the old files with
the `current_PM2_5` column deleted. Nothing else moved, changed, or reordered** — same ids,
same timestamps, same station order, same missingness in every other column, same target
column (`PM2_5_next_hour`) with the same values.

One more fact I checked directly, because it matters for everything below:

> For 358,476 of 360,954 training rows (everywhere the hourly timeline has no gap), the value
> `PM2_5_next_hour` of row `t` is **exactly equal** to what used to be `current_PM2_5` of row
> `t+1` at the same station. 100.00% match, not approximately.

That's not a coincidence — it's what "next hour's PM2.5" always meant. It has one important
and one dangerous consequence, both explained in §4.

## 2. Why they did this (their stated reason, and what it implies)

You were told the change is so **the model does not rely on PM2.5 to make its prediction** —
in the old data, `current_PM2_5` alone explained 94% of the variance in the target
(r = 0.968) purely through hour-to-hour persistence, which is a much easier and less
meaningful problem than "does this model understand air pollution." Removing it forces a
genuine estimation problem: predict PM2.5 from *correlated but imperfect* signals — the other
pollutants, the weather, the time of year, what neighbouring stations are doing.

This is why the new baselines are so much worse than the old ones, **and that is expected, not
a red flag**:

| | old bar | new bar |
|---|---:|---:|
| Ridge | 19.19 (with the anchor) | **31.99** (without it) |
| best tree model | 17.23 (with the anchor) | **27.60** (LGBM, without it) |

The old "beat 19.7" target no longer means anything. **The new bars to beat are 31.99 (Ridge)
and 27.60 (LightGBM)** — quote every new result against those, the same way CLAUDE.md's rule 5
used to say "always compare against persistence."

## 3. The single most important thing this migration must get right

**Persistence, as a concept, is dead.** `ŷ = current_PM2_5` cannot be computed anymore because
`current_PM2_5` does not exist in `test_new.csv` — not for one row, not for any row. Any
feature, script, or piece of old code that reads `current_PM2_5` from a live dataframe will now
either crash or silently return `NaN` for every row. Before reusing anything from
`src/pm25/config.py` or `scripts/v2/`, grep for `ANCHOR` and `current_PM2_5` and remove every
reference — do not leave it in "just in case," because a stray reference that happens not to
crash (e.g. a feature that's `NaN` everywhere) is worse than one that does, since it fails
silently instead of loudly.

## 4. The trap this migration must not fall into

§1's fact — `target(t) == old_current_PM2_5(t+1)` — means that **for training rows only**, you
can perfectly reconstruct the entire deleted PM2.5 history: `pm25(t) = target(t-1)`,
`pm25(t-1) = target(t-2)`, and so on, chained all the way back. This is completely legitimate —
it's just "yesterday's known outcome," no different from any other lag feature — and it means
almost every old PM2.5-based feature (lags, rolling stats, network aggregates, momentum) *could*
technically be rebuilt for training.

**The trap is that this reconstruction does not survive into the test set beyond one single
row.** Train ends the hour before test starts, so test's very first row per station can
legitimately borrow `pm25 = (that station's last training-row target)`. But test's *second*
row would need test's *first* row's target — which is exactly what you're being asked to
predict, not something you're allowed to know. Chaining further requires substituting your own
model's predictions in place of the missing ground truth (a "recursive" or "autoregressive"
forecast), and that is a fundamentally different, riskier technique — not because it's on
CLAUDE.md's forbidden list (it isn't a negative shift, external data, or a target self-join),
but because:

- it re-introduces exactly the PM2.5 dependency the organisers just removed, through the back
  door of your own guesses instead of the real sensor,
- one bad early prediction compounds forward through every later hour of that station's test
  series (roughly six months, hourly — a long way for an error to compound),
- it is much harder to validate honestly, since your CV folds would need to simulate the same
  recursive blindness, not just hold out real historical data.

**Decision needed from you, not made unilaterally here:** whether to allow this as a clearly
labelled, separate experiment (plan 05, gated) or rule it out entirely for this round. The
default path below (plans 01–04) never touches it and does not need it.

## 5. What carries over completely unchanged

Nothing about *how* these are built changes — same code pattern, same causal rule
(`shift(k>0)`, `.rolling(..., closed="left")`, same-hour cross-station only), just pointed at a
different (smaller) set of columns:

- **The complete-hourly-grid construction.** Reindexing each station to a gap-free hourly index
  never depended on PM2.5.
- **Calendar features** — hour/month/day-of-week/day-of-year sin/cos, is-weekend,
  is-heating-season. Zero dependency on PM2.5.
- **Wind features** — `wd_sin/cos`, wind u/v components.
- **The other five pollutants and five weather columns**, their own lags, rolling stats, and
  hour-over-hour changes (`PM10_change_1h`, `CO_change_1h`, etc.). These already existed in the
  v2 feature set and never referenced PM2.5 — they transfer with no code changes and are now
  much more important (see §6).
- **Cross-station same-hour aggregates**, as a *methodology* — `groupby(timestamp).transform`
  is unchanged; it just needs to run on PM10/CO/NO2/SO2/O3 instead of PM2.5 (`network_PM25_mean`
  becomes `network_PM10_mean`, etc.), since those columns are still observed for every station
  at every hour in both files.
- **Data-quality flags not built on PM2.5**: `co_floor`, `co_ceiling`, `o3_extreme`,
  `*_missing` indicators for PM10/SO2/NO2/CO/O3/weather, flat-run counters for PM10 and CO.
- **The entire validation/testing infrastructure**: `src/pm25/validation.py`'s expanding-window
  folds, `scripts/v2/cv.py`'s scoring, the leakage audit (`make audit`), and the causality-test
  *technique* (rebuild the panel on truncated data, compare) — all reusable as-is, though the
  causality test must be re-run against whatever new feature module you write, since that's new
  code even if the method is proven.
- **General modelling lessons that aren't tied to any specific feature:**
  - fit every imputer/scaler/proxy inside the training fold only (finding 01) — now doubly
    important, because the proxy-anchor idea in plan 03 is itself a fitted object;
  - don't clip predictor values for a level-target linear model near the extremes (finding 01
    — clipping cost +4.8 RMSE on the hardest fold);
  - boosted-tree implementations plateau within ~0.1 of each other once features are settled —
    don't re-run a full model zoo from scratch (finding 04);
  - hyperparameter tuning is mostly noise (±0.05); seed-averaging 3–5 runs is a small, reliable,
    cheap win (finding 05). Apply this recipe directly rather than re-discovering it.
  - the single biggest lever last time was a *category* of feature (motion/momentum), not a
    specific column or a specific algorithm — that lesson transfers directly (§6).

## 6. What must be dropped

Everything that was computed **from** `current_PM2_5` no longer exists as an input and cannot
be rebuilt for test (per §4):

- the raw value itself, all its lags (1–72h), all rolling mean/std/min/max windows on it
- `PM25_change_1h/3h/6h/24h`, `PM25_pct_change_*`, `PM25_accel` (its own momentum)
- `network_PM25_mean/median/std/min/max`, `PM25_vs_network_mean`, `net_loo_mean`,
  `net_loo_change_*`, `net_n_rising/falling`, `nb_level_<station>`, `nb_chg_<station>` (its
  network-level and network-momentum features — these included the single strongest predictor
  found in the entire old project, `net_loo_change_1h_mean`, and the top feature-importance
  column, `WSPM_x_PM25_vs_net`)
- every `PM25_x_*` interaction, `PM10_PM25_ratio`, `PM25_over_PM10`
- `pm25_floor_flag`, `pm10_equals_pm25`, `pm25_gt_pm10`, `current_PM2_5_missing`,
  `PM25_lag_*_missing`, `PM25_history_available`
- the Δ-target trick in its old literal form (`target − current_PM2_5`, add back) — there is no
  anchor left to subtract or add back
- the missing-anchor fallback logic (finding 07) — moot, because "anchor missing" is now
  *every* row, not a rare 0.7% edge case
- CLAUDE.md rule 5's literal wording ("always compare against persistence") — persistence
  cannot be computed; see §2 for the replacement bars

This is a big list, but note what's actually being lost: it's the feature *family* that used
the anchor, not the *idea* behind it. The idea — "the most predictive thing is how a value is
currently moving, city-wide" — is completely portable to the columns that remain (§6 of the old
project's own findings said this explicitly). Section 7 is that transfer.

## 7. What's new — the two ideas that replace what was lost

**A. Co-pollutant motion, including network momentum (plan 02).** The best old feature was
"how much did the network's PM2.5 change in the last hour" (r = 0.325 with the thing being
predicted). PM10, CO, NO2, SO2, and O3 are still observed everywhere, every hour, in both
files — so the exact same feature can be rebuilt around them: "how much did the network's PM10
change in the last hour," "are three or more other stations' CO readings rising right now,"
and so on. This is the highest-confidence next step because it's a direct, mechanical port of
something already proven to work, on data that's still available.

**B. A fitted proxy anchor, then residual modelling (plan 03).** The reason `current_PM2_5`
was such a good anchor wasn't magic — it was just a very strongly correlated, honestly-observed
number to build a small correction on top of. Nothing says the anchor has to be the literal
PM2.5 reading. PM10 alone already correlates with the target at **r = 0.848** (measured on this
exact data, back when PM2.5 wasn't the point of the analysis); PM10 + CO + NO2 + SO2 + O3 +
weather, combined by a small fitted model (Ridge or a shallow tree, fit **inside the training
fold only**), should recover a meaningfully strong stand-in for "roughly what PM2.5 probably is
right now." Call that `pm25_proxy`. Then model `target − pm25_proxy` the same way Δ was
modelled before — same well-conditioned-target benefit, same protection against a penalized
linear model under-trusting its strongest signal, none of the requires-the-real-anchor problem.
This is the closest thing to a like-for-like replacement for the single biggest structural
trick from the old project, and it's worth trying before assuming the raw level target is the
only option.

For reference, here is the correlation every remaining column already has with the raw target
(measured directly on this data, before any of the old PM2.5-anchor work began — so it needs no
re-measurement):

| column | r with `PM2_5_next_hour` |
|---|---:|
| PM10 | **0.848** |
| CO | 0.762 |
| NO2 | 0.643 |
| SO2 | 0.499 |
| WSPM | −0.280 |
| O3 | −0.133 |
| DEWP | 0.124 |
| TEMP | −0.122 |
| RAIN | −0.028 |
| PRES | 0.022 |

This table is the reason the new problem is structurally closer to the *old* problem than it
looks: PM10 alone at 0.848 plays a similar (weaker, but real) "quasi-anchor" role that
`current_PM2_5` used to play at 0.968. That's also why Ridge (31.99) and LightGBM (27.60) are
already reasonable, not catastrophic, with zero feature engineering — there's real signal to
build on, it's just more diffuse across several correlated columns instead of concentrated in
one.

## 8. Rules for a safe transfer — no breaking, no cheating, no leaking

1. **No feature may read `current_PM2_5` from a live dataframe, anywhere, for any row.** Grep
   the new feature module for the string before every run.
2. **No feature for a test row may depend on another test row's true or predicted target**,
   except the one explicitly-gated exception in plan 05 (§4), which stays off by default.
3. Every fitted object — the proxy-anchor model in plan 03, any imputer, any scaler — is fit on
   the training fold only, exactly as before. The proxy is a new kind of fitted object the old
   project didn't have, so it's the new highest-risk leakage surface; treat it the way finding
   01 treated Ridge's preprocessing.
4. Keep building features across the concatenated train+test timeline (still fine — test rows
   still need their own backward-looking lags of PM10/CO/NO2/etc., which *are* observed for
   every test row). Only the *direction* of a window is ever the problem, unchanged from before.
5. Re-run the causality test (truncate the timeline, rebuild, compare) against the new feature
   module before trusting any CV number from it — it's new code, even though the method that
   catches its bugs is proven.
6. Any CV fold RMSE below ~8 still means leakage, not success — unchanged, though now even less
   plausible given the higher baselines, which makes it a more reliable tripwire, not less.
7. Never quote a result against 19.67/19.7 again. Quote against 31.99 (Ridge) / 27.60 (LGBM),
   and treat CLAUDE.md's rule 5 as needing a rewrite (proposed wording below) before the next
   session relies on it.

## 9. Recommended sequence

1. **Plan 01** — rebuild the feature panel with `current_PM2_5` gone, confirm the two given
   baselines reproduce through the harness, and do a proper signal scan of what's left (the
   table in §7 is a start, not the finish — it doesn't yet cover any engineered feature).
2. **Plan 02** — port the motion/momentum feature family to the remaining pollutants. Highest
   confidence, most direct transfer of a proven idea.
3. **Plan 03** — build the fitted proxy anchor and test residual-vs-raw-level modelling on it,
   the way the Δ gate was tested before. This is the one genuinely new idea and the one most
   worth measuring carefully before committing to it.
4. **Plan 04** — apply the already-learned modelling recipe (seed-averaged boosted trees,
   fold-honest round counts, no over-tuning) directly, rather than re-running a full model zoo
   or hyperparameter sweep from zero.
5. **Plan 05** — only if you explicitly decide to, the gated recursive-reconstruction
   experiment from §4, measured and reported separately, never silently blended into the main
   submission.

## 10. Proposed rewrite of CLAUDE.md rule 5 (not applied — your call)

> **5. Always compare against the new baselines.** `current_PM2_5` is not available in this
> version of the data; persistence cannot be computed. Quote every result against Ridge 31.99
> and LightGBM 27.60 (measured on the new train/test files), never against the old 19.67/19.7.

I haven't edited `CLAUDE.md` or `PROJECT.md` — those are your team's working agreement, and a
change this structural is worth your team seeing and signing off on, not something to update
silently. Say the word and I'll make the edit.
