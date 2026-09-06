# Each model rung receives a different treatment of blank predictors

Blank predictors run 0.3–2% per column. LightGBM handles missing values natively and can learn
from the pattern of absence itself; ridge regression and SARIMAX cannot see a missing value at
all. Imputing for LightGBM's benefit would hand it fabricated numbers it cannot distinguish from
measurements, and destroy real signal.

So one shared step runs first — forward-fill within each station, capped at roughly three hours
of staleness, emitting an `is_imputed` flag per column and an hours-since-observation feature —
and then the rungs diverge: residual missing values are left in place for LightGBM, and filled
with a station-by-hour-of-day median for ridge and SARIMAX only.

## Consequences

The asymmetry looks like a bug. Two models in the same ladder are fed frames that differ in
their missing values, and the comparison between them is therefore not perfectly like-for-like —
that is deliberate, and each rung is given the representation it can actually use.

Forward-fill is the only direction permitted; interpolation and backward fill both read across
the gap and would violate ADR-0001.
