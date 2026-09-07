# Plan v3-03 — A fitted proxy anchor, then residual modelling

**Question.** Can a fold-fitted estimate of "roughly what PM2.5 probably is right now," built
only from the columns that still exist, recover enough of the old anchor's benefit to make
residual modelling (predict `target − proxy`, add the proxy back) worth doing again?

## Why this is worth testing, not just plausible

The old anchor trick worked for two separate reasons (migration.md §7, and the old project's
finding 01/02): it gave the model a strong, well-conditioned starting point, and it protected a
penalized linear model from under-trusting its single strongest signal by removing that signal
from the penalized part of the equation entirely. Neither reason actually requires the number
to be the *real* PM2.5 reading — it only requires the number to be a strong, honestly-computed
correlate. PM10 alone already correlates with the target at r = 0.848 on this exact data. A
small model combining PM10 with CO, NO2, SO2, O3 and weather should do better than PM10 alone,
the same way the old feature set did better than `current_PM2_5` alone (0.968 → the actual
project score once engineered).

## Method

1. **Build the proxy, fitted inside the training fold only.** Simple version: Ridge or a
   shallow LightGBM trained to predict the *current-hour* target-equivalent — since there's no
   direct label for "PM2.5 right now," train it to predict `PM2_5_next_hour` from the
   *contemporaneous* other-pollutant/weather columns at hour `t`, which is the closest available
   stand-in and uses only information already legitimately available at `t`. Call its output
   `pm25_proxy`.
2. **Leakage check specific to this plan**: the proxy is a new kind of fitted object the old
   project never had. Confirm explicitly that it is refit inside every outer training fold (not
   fit once on the full training set and reused across folds) before trusting any number from
   this plan — this is the single highest-risk step in the whole v3 migration (migration.md §8,
   rule 3).
3. **Build the proxy's own motion family**: lags, changes, and network momentum of `pm25_proxy`
   itself, exactly mirroring what plan 02 does for the raw pollutants — since the proxy is
   meant to stand in for PM2.5, its momentum should carry similar information to PM2.5's old
   momentum features.
4. **The A/B this plan runs**: raw-level target vs `target − pm25_proxy` residual target
   (proxy added back at prediction time), same features otherwise, same fixed LightGBM
   parameters, same folds — the direct analogue of the old Δ gate (v2 finding 02).

## Success criteria

- The proxy itself, alone, beats the plain-column LightGBM baseline (27.599) by a real margin —
  otherwise it isn't a strong enough anchor to build a residual trick on.
- The residual arm beats the raw-level arm on both heating-season folds by more than noise,
  mirroring the old Δ-gate result (finding 02: Δ won by 1.1, mostly on the hardest winter fold).
- If the residual arm does **not** win: that's a real, useful finding — it means this problem's
  removal of PM2.5 broke the anchor trick's foundation more thoroughly than expected, and the
  raw-level target should be used everywhere going forward without further attempts to revive
  the old trick.

## Output

`Findings/v3/03-proxy-anchor-residual.md`
