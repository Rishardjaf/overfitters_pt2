# Beijing Multi-Site Air Quality Forecasting — revised release

Forecasting fine particulate matter one hour ahead across a network of twelve Beijing
monitoring stations, for the Inter-University Datathon (Stream 2). This file is the project's
glossary and nothing else — no implementation detail, no plan, no decisions.

The revised release deletes `current_PM2_5`. Two terms that the previous glossary defined —
*current PM2.5* and *persistence* — name things that no longer exist in the data and have been
removed rather than redefined. Anything still using them is describing the old release.

## The prediction problem

**Station-hour**:
The unit of prediction — one monitoring station observed at one hour. Carries a competition
`id`.
_Avoid_: row, record, observation

**Observation hour**:
The hour whose measurements a station-hour carries. Written *t*.
_Avoid_: current hour, now

**Target hour**:
The hour whose PM2.5 concentration is being predicted — exactly one hour after the observation
hour. Written *t+1*.
_Avoid_: forecast hour, next hour

**Next-hour PM2.5**:
The PM2.5 concentration at the target hour, in µg/m³. The quantity being predicted. Present in
the training data only. No PM2.5 measurement appears anywhere in the predictors.
_Avoid_: target, label, y

**Anchor**:
The measurement a prediction is built outwards from, and the reference every model is scored
against. In this release it is PM10 at the observation hour, which carries r = +0.848 with the
target — the strongest remaining predictor. The anchor is a fitted line through PM10, not the
raw reading.
_Avoid_: baseline model, persistence, naive model

**Partner row**:
For a station-hour at *t*, the same station's row at *t+1* — present in `test.csv` for 99.25% of
test rows. It carries that hour's co-pollutants and weather but no PM2.5. Reading it is
forbidden ([ADR-0006](docs/adr/0006-the-causal-contract-under-the-revised-schema.md)).
_Avoid_: next row, future row

## Data completeness

Three distinct conditions are all colloquially called "missing". They are named separately
because they demand different treatment.

**Blank predictor**:
A station-hour that exists but has an empty field for one measurement. Retained deliberately by
the organisers.
_Avoid_: NA, null, missing value

**Absent station-hour**:
A station and hour for which no station-hour exists at all — a hole in the hourly grid, not a
hole in a row. 2.08% of the training grid and 2.04% of the test grid.
_Avoid_: gap, missing row, missing data

**Excluded row**:
A station-hour the organisers removed because it had no valid next-hour PM2.5. Never visible in
the supplied data; the reason the hourly grid can never be assumed complete.
_Avoid_: dropped row, filtered row

**Staleness**:
How many hours have passed since a column last carried a real measurement for a station.
_Avoid_: age, freshness

## Domain conditions

**Pollution episode**:
A period during which PM2.5 rises far above its usual level and stays there. Episodes dominate
the error under a squared-error metric, and because one episode spans many consecutive hours
across all twelve stations, they are also why residuals are correlated within a day.
_Avoid_: spike, event, outlier

**Heating season**:
The Beijing municipal heating period, roughly 15 November to 15 March, during which emissions
step up. Falls inside the competition's test window.
_Avoid_: winter, cold season

**Dew-point depression**:
Temperature minus dew-point temperature, in °C. Small values indicate humid, stagnant,
inversion-prone air in which particulates accumulate.
_Avoid_: humidity, TEMP-DEWP

**Urban core / suburban stations**:
The distinction between the central stations and the outlying ones (Dingling, Huairou,
Changping), which experience different pollution levels and dynamics.
_Avoid_: city stations, rural stations

## Evaluation

**Fold**:
A season-matched holdout block — September to February of an earlier year — with every hour
preceding it as its training rows. Named for its span, e.g. `2015-09_to_2016-03`.
_Avoid_: split, validation set

**Day block**:
One calendar day of residuals, resampled as a unit by the bootstrap. A fold holds roughly 182 of
them, and they, not the ~51,000 rows, are the independent sample.
_Avoid_: cluster, group

**Conclusive**:
Said of a comparison whose 90% paired bootstrap interval excludes zero. A model that wins on the
point estimate but not conclusively has not been shown to be better.
_Avoid_: significant, better, wins
