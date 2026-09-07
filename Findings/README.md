# Findings

Measured results from executing the documents in `plans/`. One file per execution,
named `NN-<plan>-<what-was-measured>.md`.

A finding is a *measurement*, not an opinion. If it is not on the heating-season folds, it
does not belong here.

## What each finding must record

- **Plan and config** — which plan document, which config file, git SHA, seed.
- **Headline** — mean RMSE over the heating-season folds.
- **Per-fold breakdown** — every fold's RMSE alongside that fold's persistence score, and the
  delta between them. An averaged number alone is not acceptable.
- **Signed-tail split** — on anchor-present rows, report RMSE, row count and squared-error share
  for rapid falls (Δ ≤ −30), calm rows (|Δ| < 30) and rapid rises (Δ ≥ 30). Falling events
  carry more persistence error than rising events and must not be hidden inside "calm".
- **Missing-anchor split** — row count and raw-scale RMSE for the declared fallback path.
- **Verdict** — beat persistence or not, by how much, and whether it reproduced across folds.
  Moves under ~0.05 RMSE are noise.
- **What it changes** — which plans this result unblocks, revises or refutes.

## Open gate

`plans/01-lightgbm.md` measures Δ against the raw level. Plans 02–06 stay blocked on that
result — see "The Δ decision gate" in `plans/README.md`.
