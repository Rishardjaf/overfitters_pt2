# Finding: which previous knowledge still applies

- Status: validated for schema compatibility; all prior model rankings remain unvalidated here
- Confidence: high for the compatibility conclusions below
- Date: 2026-09-06
- Contributor: Codex, repository migration task
- Dataset: fingerprints in [dataset_release.md](dataset_release.md)

## Evidence and scope

The source [agent knowledge folder at 97525ff](https://github.com/Rishardjaf/overfitters/tree/97525ff208185a706aa69128536ac18b8432ea3c/agent_knowledge) contained 11 workflow/template/index files and no populated topic entries. These were copied and adapted. Historical assumptions also appear in the old [glossary](https://github.com/Rishardjaf/overfitters/blob/97525ff208185a706aa69128536ac18b8432ea3c/CONTEXT.md), [causality decision](https://github.com/Rishardjaf/overfitters/blob/97525ff208185a706aa69128536ac18b8432ea3c/docs/adr/0001-no-lookahead-features.md), and [validation decision](https://github.com/Rishardjaf/overfitters/blob/97525ff208185a706aa69128536ac18b8432ea3c/docs/adr/0002-season-matched-validation.md).

The [full CSV comparison](dataset_release.md) establishes the removal of `current_PM2_5` and equality of the retained data. This is the basis for the compatibility assessment, not a new predictive experiment.

| Previous knowledge or artifact | Current applicability | Action |
| --- | --- | --- |
| Shared contribution workflow, templates, topic index, conflict records | Retained | Add dataset fingerprints and recheck applicability on every release change. |
| Station-hour grain, next-hour target, station/time identities | Retained and checked against supplied files | Preserve IDs and timestamp alignment. |
| No future observations; training-only preprocessing | Retained project policy | Reimplement and test when code is added. No enforcement code exists in this scaffold. |
| Autumn/winter chronological evaluation | Time coverage still supports it | Use causal splits and a compatible new baseline; re-run every score. |
| Current PM2.5 as a predictor and persistence reference | Superseded by revised schema | The supplied inference inputs cannot support that baseline. |
| PM2.5 lags, rolling statistics, cross-station PM2.5 and missing-PM2.5 fallback rules | Unsupported by revised input contract | Do not rebuild from historical files or shifted training labels. |
| Persistence-residual model design | Requires redesign | Its old PM2.5 anchor is unavailable; direct-target modeling is a proposed starting point. |
| Claim that next-row current_PM2_5 reveals test targets | Inapplicable to revised schema | The exposing column is absent. Keep the general causality rule. |
| Old trained models, preprocessed parquet caches, feature rankings, RMSE scores, submissions | Not validated for revised inputs | Keep them in the old repository; establish fresh evidence. |
| Sensor findings about removed current PM2.5 | Not an available-predictor finding here | Reassess any relevance to training labels separately; do not treat as verified label faults. |
| Remaining pollutant/weather missingness | Unchanged and verified | Design missing-value handling; do not assume the old six-pollutant outage definition still applies. |

## Model and evaluation cautions

Ridge, tree boosting, and shared-versus-seasonal comparisons are possible future experiments, not selected winners. PM10 and the remaining pollutants/weather may be useful but require fresh evaluation. Removing an input does not make previously inspected evaluation labels unseen; previously used validation windows must be described as retrospective when applicable.

Official rules and any extra inference-time data source still need verification before changing the current contract. This migration authorizes no model training or additional data retrieval.

Historical work remains in the previous repository for provenance. No old code, notebooks, analysis outputs, or cached datasets were migrated. The empty `ML_Algorithms/` folder intentionally does not assert a persistence-first model ladder.
