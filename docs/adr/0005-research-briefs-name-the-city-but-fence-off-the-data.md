# Research briefs name the city but forbid the tool from returning the data

The competition rules state that we must not attempt to identify the original dataset,
use external copies of the data, or recover hidden test targets. Any brief we paste into
an external research tool describing twelve Beijing monitoring stations with hourly
PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, wind direction and WSPM will lead
that tool to the underlying public dataset in about one query.

We considered anonymising — describing it as twelve urban stations in a temperate
northern-hemisphere city with a winter heating season, with no city name and no date
ranges. That would have cost us the literature. Beijing's heating season, its inversion
climatology and its station geography are exactly what make published work relevant, and
a brief stripped of them returns generic time-series advice.

We name the city instead, because the organisers already did — the competition is titled
"Beijing Multi-Site Air Quality" — so identifying the city is not something an entrant
does. The residual risk is narrower and specific: that a research tool volunteers
download links and we end up holding an external copy of the data. So every brief we
send out carries an explicit instruction to return methods and findings only, never
dataset sources, download links, repository links, or reported benchmark scores on any
specific public version of this data.

## Consequences

A reader of [../../Resources/perplexity-brief.md](../../Resources/perplexity-brief.md)
will find an instruction telling the tool what *not* to return, which looks odd beside
questions asking it to be maximally helpful. It is there to keep the answer on the
correct side of the rule, not to limit the review.

The rule binds the whole team. Anyone pasting a different brief into a different tool
carries the same two lines: no lookahead, and no dataset sources. A brief that omits them
is the failure mode this ADR exists to prevent.

We accept that some published results on this problem are unusable to us. A paper
reporting an RMSE on a specific public version of this data tells us something we are not
permitted to calibrate against, so the calibration question in the brief asks for
improvement *relative to a persistence baseline* rather than for absolute error figures.
