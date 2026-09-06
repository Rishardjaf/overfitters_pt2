# Knowledge index

Current release: supplied revised CSVs verified on 2026-09-06. `current_PM2_5` is absent. Historical model performance is not validated for this release, and old-schema scores must never be tabled beside new ones.

| Area | Topic | Status | Confidence | Last updated | File |
| --- | --- | --- | --- | --- | --- |
| Findings | Revised input contract and verification | validated | high | 2026-09-06 | [Dataset release](findings/dataset_release.md) |
| Findings | Applicability of previous knowledge | validated (compatibility only) | high | 2026-09-06 | [Migration review](findings/previous_knowledge_review.md) |
| Decisions | Causal features and chronological validation | validated (retained policy, not performance) | high | 2026-09-06 | [Validation policy](decisions/causal_validation.md) |
| Features | Remaining predictors to investigate | proposed | low (predictive benefit) | 2026-09-06 | [Candidate features](features/revised_schema_candidates.md) |
| Findings | First model ladder on the revised schema | validated | high (fold scores) | 2026-09-06 | [Ladder](findings/revised_schema_ladder.md) |
| Findings | Delta-target booster fails the promotion gate | validated (gate outcome) | high | 2026-09-06 | [Gate record](findings/delta_target_gate.md) |
| Decisions | PM10 is the anchor and the baseline | validated | high | 2026-09-06 | [PM10 anchor](decisions/pm10_anchor.md) |

Statuses: `proposed`, `validated`, `rejected`, `superseded`, `disputed`. Confidence: `low`, `medium`, `high`. Validation of a data fact or policy does not validate a model's performance.

[Open conflicts](conflicts/OPEN.md) · [Contribution workflow](README.md) · [Templates](templates/finding_template.md)
