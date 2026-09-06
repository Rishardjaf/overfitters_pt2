# Overfitters — revised data

Predict `PM2_5_next_hour` for 12 Beijing air-quality stations using the revised competition
inputs. RMSE, lower is better. [CONTEXT.md](CONTEXT.md) is the glossary;
[docs/adr/](docs/adr/) records every decision that would otherwise look like a bug. Start with
[the agent knowledge index](agent_knowledge/INDEX.md).

The new inputs **do not contain `current_PM2_5`**. Previous model scores, PM2.5-derived features,
persistence baselines, and cached datasets are not valid starting artifacts for this release. See
[the migration review](agent_knowledge/findings/previous_knowledge_review.md).

## Current standing

| rung | algorithm | fold `2015-09_to_2016-03` | fold `2014-09_to_2015-03` |
|---|---|---:|---:|
| 01 | pm10_anchor (baseline) | 40.664 | 43.425 |
| 02 | ridge | 34.396 | 34.367 |
| 03 | lightgbm — **the submission** | **29.614** | **27.500** |
| 04 | delta_lightgbm — fails the gate | 29.184 | 26.955 |

[The ladder](agent_knowledge/findings/revised_schema_ladder.md) ·
[why rung 04 was not promoted](agent_knowledge/findings/delta_target_gate.md)

## Never commit competition data

`*.csv`, `*.parquet` and `*.zip` are gitignored at every path, not just under `data/`. A frame
derived from the competition CSVs is still competition data, and redistributing it is against the
rules. Submissions are ~1 MB of derived competition data and are regenerable from the metrics
committed beside them. **Never `git add -f` a data file, a submission, or a derived frame.**

The raw CSVs were tracked in commit `bf61ca9` and are still present in the history already pushed
to `origin/main`, `origin/copilot/main` and `origin/chore/standardize_repository`. Removing them
there needs a coordinated force-push across those branches and the two open pull requests; it has
not been done.

## The one rule: no lookahead

Every feature for a station-hour at time *t* uses only data at times ≤ *t*. No backward fill, no
interpolation across a gap, no centred window, no negative shift. Same-hour readings from *other*
stations are legal. `src/leakage.py` enforces it by rebuilding the feature frame from truncated
data and asserting nothing changed.

99.25% of test rows still have their (station, *t+1*) partner row present in `test.csv`. It no
longer carries PM2.5, but it carries that hour's PM10 and weather. Reading it is prohibited —
[ADR-0006](docs/adr/0006-the-causal-contract-under-the-revised-schema.md) records that it would
buy +0.018 of correlation on the strongest predictor. Do not propose it.

A **fitted statistic** computed over rows that include the holdout — a median, a scaler, the
anchor's own slope — is train/test contamination, and the guard is blind to it. Fit on training
rows only.

## Nothing is promoted on a single number

A pollution episode is one event spanning many consecutive hours across all twelve stations, so
residuals are correlated within a day and the independent sample is nearer a fold's ~182 day
blocks than its ~51,000 rows. Naive row-wise SE is ~0.07; the honest one is 0.2–0.4.

**The promotion gate, all three required:** wins on the primary fold (`2015-09_to_2016-03`); does
not lose on the second (`2014-09_to_2015-03`); and the fold-1 bootstrap 90% interval lies entirely
below zero. A variant that fails is recorded, never quietly dropped — rung 04 is the worked
example.

## Commands

```bash
.venv/bin/python -m pytest tests -q                          # always before committing
.venv/bin/python -m src.cache                                # rebuild parquet after a feature change
.venv/bin/python -m src.experiments all --all-folds          # every rung, both folds
.venv/bin/python -m src.experiments 03_lightgbm              # one rung, by slug or algorithm name
.venv/bin/python -m scripts.modeling.gate 03_lightgbm 04_delta_lightgbm   # the promotion gate
```

A feature-frame change invalidates the cache and every rung's numbers. Rebuild, then re-run the
ladder once at the end rather than per experiment.

## Structure

```text
agent_knowledge/       Shared findings, feature notes, decisions, conflicts, templates
data/raw/              Revised train.csv and test.csv; immutable, never committed
data/processed/        Cached feature parquet, rebuilt by src.cache
src/                   Loading, grid completion, imputation, features, validation, scoring, cache
src/models/            The rungs: pm10_anchor, ridge, lgbm, delta_lgbm
src/experiments/       The harness that runs a rung and writes its folder of evidence
src/uncertainty.py     The paired day-block bootstrap that gates every decision
scripts/modeling/      Standalone comparison scripts
ML_Algorithms/         One numbered folder per test — results, metrics, figures
tests/                 Invariants — grid completeness, honest imputation, no lookahead, valid submissions
docs/adr/              Formal technical decisions
outputs/submissions/   The chosen submission; gitignored
```

## Conventions

- Immutable frames: build new columns and `pd.concat` them on; never mutate an argument in place.
- Module docstrings say *why*, not what. Reproduce the number that settled a choice.
- Constants are named at module top, never inline.
- Formulas in Markdown use plain Unicode (β̂, σ̂², Σ, √, µg/m³), never LaTeX.
- Blanks are treated per rung on purpose ([ADR-0003](docs/adr/0003-per-rung-imputation.md)):
  ridge gets a fitted median fill, LightGBM gets the frame intact because the pattern of absence
  is itself information. Do not unify this.
