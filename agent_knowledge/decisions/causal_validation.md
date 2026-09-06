# Decision: retain causal features and chronological evaluation

- Status: validated as retained project policy; predictive performance is untested
- Date: 2026-09-06
- Contributor: Codex, repository migration task
- Dataset: [revised release and fingerprints](../findings/dataset_release.md)

## Decision and rationale

Retain the previous project's no-lookahead and chronological validation principles. The verified schema change leaves timestamps and labels intact, so these protections still apply. See the [migration review](../findings/previous_knowledge_review.md) for source decisions.

- Features for observation time `t` may use only approved measurements available at or before `t`. Do not use backward fill, centered windows, future observations, or aggregate statistics learned across validation/test periods.
- Use actual elapsed hours for lag alignment; a previous row is not necessarily the previous hour. Track missing measurements separately from missing station-hours. Keep station boundaries explicit.
- Fit scalers, imputers, encoders that learn statistics, feature selection, and models on training data only. Same-hour context from other stations is a candidate only if those inputs are available under the approved inference contract.
- For an evaluation block starting at `S`, enforce `training observation time + 1 hour < S`. With the supplied test starting at `2016-08-31 23:00`, the final fit must exclude the training observation at `2016-08-31 22:00` because its target hour equals `S`.
- The test period is predominantly autumn/winter. The previous September–February comparisons remain relevant in time coverage; include other seasons as diagnostics and report each split explicitly. Compare models on the same rows and metric.
- New baselines must work without `current_PM2_5`. Direct-target Ridge or boosting are candidates, not established choices. Do not compare new scores to incompatible old-feature scores as if they used the same inputs.
- Previously inspected holdout labels remain retrospective evidence. Freeze future model selection and evaluation rules before using a new holdout.

## Alternatives and consequences

Random row splitting would mix adjacent station-hours and distort forecasting evaluation. Reusing the old persistence ladder would require a removed input. Fresh causal baselines cost new implementation and evaluation, but provide relevant comparisons.

This repository currently contains no implementation or tests of these rules. Future work should test temporal boundaries, training-only preprocessing, ID-preserving submissions, and truncated-history feature invariance before reporting validated performance.
