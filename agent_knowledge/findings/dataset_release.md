# Finding: revised input contract

- Status: validated
- Confidence: high
- Verified: 2026-09-06
- Contributor: Codex, repository migration task
- Scope: complete supplied train.csv and test(1).csv, compared with the previous local data/raw files.

## Conclusion

Both revised CSVs equal their previous counterparts after removing `current_PM2_5`. The full comparison preserves row order and checks column names, parsed types, all values, and matching missing values. The training labels, IDs, remaining measurements, and time coverage have not changed. This verifies data compatibility; it does not establish attainable forecast accuracy.

| File | Rows | Columns | Stations | First observation | Last observation |
| --- | ---: | ---: | ---: | --- | --- |
| [train.csv](../../data/raw/train.csv) | 360,954 | 19 | 12 | 2013-03-01 00:00 | 2016-08-31 22:00 |
| [test.csv](../../data/raw/test.csv) | 51,063 | 18 | 12 | 2016-08-31 23:00 | 2017-02-28 22:00 |

Grain: one station at one observation hour. `id` and `(station, observation_timestamp)` are unique within each file; train/test IDs are disjoint. IDs, stations, timestamps, and training targets have no missing values. Year/month/day/hour fields agree with the parsed timestamp. Timestamp timezone is not encoded in the files; retain the supplied time convention until an authoritative specification says otherwise.

Common columns, in order:

```text
id, observation_timestamp, station, year, month, day, hour,
PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, wd, WSPM
```

Training additionally has `PM2_5_next_hour`, the next-hour target. Test has no target column. `id` is an identifier, not an approved predictive feature.

## File fingerprints and provenance

The supplied files came from the user's Downloads folder. `test(1).csv` was renamed to `data/raw/test.csv` without altering any bytes. No data was synthesized or re-exported.

| File | SHA-256 |
| --- | --- |
| Revised train | `0ceeb8b708f87007fecfbdf6365b8f13fa036fff0e84575a88a5a28d78cca037` |
| Previous local train | `7db322208c6163630d18ba43ef5ab1df372988615f168066fe0597df63d81d1e` |
| Revised test | `67cfeb0fef54be1eaa9ef399b0d99314829a386e88dad2dc623631835ecad998` |
| Previous local test | `04e148b0f13007b027171b3144a3145b5ec302af8331425125009eb0ada524e7` |

## Remaining missing values

Missing measurements remain and must not be confused with absent station-hours. Missingness of every retained field is unchanged from the previous files.

| Field | Train missing | Test missing |
| --- | ---: | ---: |
| PM10 | 1,922 | 295 |
| SO2 | 4,663 | 469 |
| NO2 | 7,521 | 670 |
| CO | 15,832 | 772 |
| O3 | 8,499 | 1,019 |
| TEMP | 177 | 218 |
| PRES | 178 | 212 |
| DEWP | 182 | 218 |
| RAIN | 174 | 213 |
| wd | 793 | 1,004 |
| WSPM | 175 | 140 |


## Reproduce the comparison

Use Python with pandas, from this repository root, setting `old_raw` to the preserved previous repository's `data/raw/`. No old data is copied into this repository. SHA-256 checks remain reproducible without the old repository.

```python
from pathlib import Path
import hashlib
import pandas as pd

old_raw = Path('../overfitters/data/raw')  # adjust to the old checkout
for split in ('train', 'test'):
    path = Path('data/raw') / f'{split}.csv'
    revised = pd.read_csv(path)
    previous = pd.read_csv(old_raw / f'{split}.csv')
    pd.testing.assert_frame_equal(
        previous.drop(columns=['current_PM2_5']), revised,
        check_exact=True,
    )
    assert revised.id.is_unique
    assert not revised.duplicated(['station', 'observation_timestamp']).any()
    times = pd.to_datetime(revised.observation_timestamp)
    for field in ('year', 'month', 'day', 'hour'):
        assert (revised[field] == getattr(times.dt, field)).all()
    print(split, len(revised), hashlib.sha256(path.read_bytes()).hexdigest())
```

## Risks and limits

- High impact, high confidence: removed PM2.5 inputs invalidate old persistence anchors, PM2.5 history features, and compatible-model claims. Rebuild and re-evaluate using the revised schema.
- Medium impact, high confidence: missing predictors remain. Fit imputation on the training partition and make missingness handling explicit.
- This was a schema, equality, key, missingness, and timestamp audit. No model was fitted, and official competition rules or external inference data availability were not re-verified.
- A future file-hash change requires a new release entry and another applicability review.

See [the migration review](previous_knowledge_review.md) and [validation policy](../decisions/causal_validation.md).
