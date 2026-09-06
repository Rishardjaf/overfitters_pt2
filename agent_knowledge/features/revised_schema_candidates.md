# Features: candidates from the remaining inputs

- Status: proposed
- Confidence: low for predictive benefit; input availability verified
- Date: 2026-09-06
- Contributor: Codex, repository migration task
- Dataset: [revised release and fingerprints](../findings/dataset_release.md)

## Candidate definitions

| Candidate | Definition | Availability and missingness |
| --- | --- | --- |
| Remaining measurements | PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, wd, WSPM | Supplied at the observation hour; retain missingness indicators and use training-fitted handling. |
| Station and calendar | Station category, hour, month, weekday derived from timestamp | Available at prediction time; category policy must be explicit. |
| Dew-point depression | TEMP minus DEWP | Available when both measurements exist; missing if either input is missing. |
| Measurement history | Per-station elapsed-hour lags and trailing windows of the remaining measurements | Use only approved prior observations; absent hours must not silently become one-hour lags. |
| Measurement availability | Per-field missing flag and elapsed time since last observed measurement | Compute causally, with an explicit unknown-history state. |

These are investigation candidates. No feature improvement, selected lag length, rolling window, model family, or new-release score has been established. Define each selected feature in its own entry using the template and compare it against a compatible baseline on common chronological folds before marking it validated.

`current_PM2_5`, its history, and target-derived proxies are outside the revised predictor contract. Training labels are for fitting/scoring, not an inference measurement feed. See [validation policy](../decisions/causal_validation.md).
