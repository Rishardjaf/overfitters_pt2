# Overfitters — revised data

Predict `PM2_5_next_hour` for 12 Beijing air-quality stations using the revised competition inputs. Start with [the agent knowledge index](agent_knowledge/INDEX.md).

The new inputs **do not contain `current_PM2_5`**. Previous model scores, PM2.5-derived features, persistence baselines, and cached datasets are not valid starting artifacts for this release. See [the migration review](agent_knowledge/findings/previous_knowledge_review.md).

## Structure

```text
agent_knowledge/       Shared findings, feature notes, decisions, conflicts, templates
data/
  raw/                 Revised train.csv and test.csv; immutable source files
  processed/           Future derived data
notebooks/eda/         Future exploratory notebooks
scripts/
  cleaning/            Future cleaning scripts
  features/            Future feature engineering
  modeling/            Future training, evaluation, and prediction
  notebooks/           Future notebook utilities
src/
  models/              Future reusable model implementations
  experiments/         Future experiment orchestration
tests/                 Future automated checks
reports/
  eda/figures/          Future EDA plots
  eda/tables/           Future EDA extracts
  modeling/tables/      Future validation comparisons
ML_Algorithms/         Future per-model evidence
outputs/
  figures/             Future generated figures
  submissions/         Future competition submissions
docs/adr/              Future formal technical decisions
Resources/             Future reference material
```

Only agent knowledge and the revised raw datasets are populated. Empty folders contain `.gitkeep` because Git does not track empty directories. No model code, notebooks, trained models, results, old data, or dependency environment has been migrated.

## Data

- [train.csv](data/raw/train.csv): 360,954 rows, 19 columns including the target.
- [test.csv](data/raw/test.csv): 51,063 rows, 18 columns without the target; copied byte-for-byte from the supplied `test(1).csv`.
- [Release verification](agent_knowledge/findings/dataset_release.md): schema, time coverage, hashes, and reproducible comparison.

## Working conventions

Read [AGENTS.md](AGENTS.md) and the relevant knowledge entries before working. Keep raw files unchanged; place transformations in `data/processed/`. Use a descriptive `type/change_name` branch and small Conventional Commits. Validate each future implementation before proposing a merge.

This is a data-and-documentation scaffold. There is no executable application or test, lint, or build command yet.
