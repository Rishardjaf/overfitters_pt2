# Finding: first model ladder on the revised schema

- Status: validated
- Confidence: high for the fold scores; medium for what they imply about the hidden test set
- Date: 2026-09-06
- Contributor: Claude Opus 5, revised-schema rebuild
- Dataset: [revised release and fingerprints](dataset_release.md); feature frame 140 columns,
  built on the joined timeline per [ADR-0004](../../docs/adr/0004-features-built-on-the-joined-timeline.md)

## Conclusion

Causal features on the surviving predictors reach 29.614 RMSE on the primary fold, a 27.2%
improvement over the PM10 anchor. Lag and rolling history on the co-pollutants is where the gain
lives: the same booster on row features alone scores 34.833.

| rung | algorithm | fold `2015-09_to_2016-03` | fold `2014-09_to_2015-03` | vs anchor (fold 1) |
|---|---|---:|---:|---:|
| 01 | pm10_anchor (baseline) | 40.664 | 43.425 | — |
| 02 | ridge | 34.396 | 34.367 | +15.4% |
| 03 | lightgbm | **29.614** | **27.500** | +27.2% |
| 04 | delta_lightgbm | 29.184 | 26.955 | +28.2% |

Rung 03 is the submission. Rung 04 wins on both point estimates but fails the promotion gate;
see [the gate record](delta_target_gate.md).

## Evidence

Reproduce with, from the repository root:

```bash
.venv/bin/python -m src.cache
.venv/bin/python -m src.experiments all --all-folds
```

Per-rung metrics, figures and `RESULTS.md` are written to `ML_Algorithms/<slug>/`. Folds are
season-matched blocks with every preceding hour as training rows
([ADR-0002](../../docs/adr/0002-season-matched-validation.md)); the no-lookahead guard passes on
the full 140-column frame.

The pre-modelling reference points, measured on raw rows before any feature engineering, were:
constant 102.04, station × hour-of-day climatology 101.40, global linear on PM10ₜ 41.08, ridge on
row features 35.70, LightGBM on row features 34.83.

## Interpretation, kept separate from the facts above

Ridge scores almost identically on the two folds (34.40 / 34.37) while the boosters move by more
than two RMSE (29.61 / 27.50). The linear model is insensitive to which winter it is asked about
and the trees are not — consistent with the trees fitting episode structure that varies year to
year. This is an observation about two numbers, not an established mechanism, and no experiment
here tests it.

## Risks and limits

- These are holdout scores on training-period rows. The hidden test set is Sep 2016 – Feb 2017,
  later than any fold, and nothing here establishes what it will score.
- **Not comparable to any old-schema number.** The previous ladder's 19.998 was achieved with
  `current_PM2_5` present and a persistence baseline of 23.138. Tabling the two together would be
  a straightforward error; see [ADR-0007](../../docs/adr/0007-pm10-replaces-persistence-as-the-anchor.md).
- No hyper-parameter search was run. `lgbm.DEFAULT_PARAMS` and 800 rounds are inherited from the
  previous release and were tuned against a different feature set.
- Random forest, histogram boosting and SARIMAX were not run. On the previous release the forest
  lost to ridge and SARIMAX lost to the baseline; neither was re-tested here and both remain
  formally untested on this schema.
