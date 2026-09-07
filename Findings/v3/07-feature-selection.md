# Finding v3-07 — Feature selection: what stays, what goes, and why

**Plan:** `plan v3 changed/07-feature-selection.md` · **Notebook:**
`notebooks/features new/01_feature_exploration.ipynb` (§5–§6, executed) · **Base:** f7 for the
ablations, deep (f9) for the redundancy scan.

## Backward ablation from f7 (level target)

| removed | cols | headline | vs f7 | verdict |
|---|---:|---:|---:|---|
| — (f7) | 264 | 27.90 | — | |
| acceleration (2nd derivative) | 258 | 27.87 | −0.04 | neutral → **out** (it also hurt +0.43 when added) |
| rolling stats | 206 | 28.03 | +0.13 | **stays** — hurt when first added, helps once network groups exist |
| both | 200 | 28.08 | +0.18 | consistent with the two above |
| missing + quality flags | — | zero gain on every fold | — | **out** (12 columns with 0 gain) |
| weather lags + extra meteorology (f8 − f7) | +27 | 28.03 | +0.13 | **out**, same as v2 |

The resulting **lean** set (261 cols: deep − acceleration − flags) scores 26.14 on the ratio
target vs **25.94** for the full deep set. The difference is at the seed-noise edge (±0.1), so
the deep set is kept as the final set; lean is the speed alternative.

## Pairwise redundancy in the deep set (train rows)

- **119 pairs with |r| > 0.95 among 282 numeric columns; 54 columns are the lower-gain member
  of at least one pair.** They are mostly the expected families: lags 1–3h of the same
  pollutant vs its 3h/6h rolling means, the "including current hour" rolling means vs the
  excluding ones, CO lags vs each other, and `PM10_x_CO` vs its log/sqrt transforms.
- Dropping the 54 lower-gain members is a *speed* decision, not an accuracy one — the
  ablations above show that removing correlated columns from a tree costs ~0 to +0.1. It was
  **not** applied to the submission model. It matters for Ridge (plan 04), where the L2
  penalty has to fight every one of these pairs.
- The list is in the notebook (§6) and reproducible from `artifacts/v3/panel.parquet`.

## Correlation-with-target scan (why the tree likes what it likes)

Top of the raw correlation ranking is unchanged from the EDA — PM10 0.85, CO 0.76, NO2 0.64 —
and the engineered columns the tree actually uses (`PM10 × CO`, `net_PM10_median`,
`net_CO_loo_mean`) are all *level-scale* combinations of those three. Motion features sit at
r ≈ 0.1 individually yet were worth −1.1 as a group: their value is conditional (direction of
travel given a level), which a correlation scan cannot see and a tree can.

## What was learned about selection on this problem

1. **Group ablation beats column-by-column pruning** here: single columns rarely move the
   headline, groups do.
2. **Order of addition changes the verdict** (rolling: harmful alone, helpful with network
   features). A feature's value depends on what else the tree can see — one more reason to
   ablate *from the full set*, not just to build up.
3. **Feature set and target must agree.** The deep fine-fraction columns are noise for a
   level model and a −0.3 gain for the ratio model.

## Verdict

Final set **deep (f9, 284 cols)**; lean (261) within noise; 54 redundant columns documented for
a linear model but not removed for the trees.
