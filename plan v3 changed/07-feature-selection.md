# Plan v3-07 — Feature selection: pairwise redundancy and ablation

**Question.** Of everything plan 02 built, what actually matters? Prune to the set that keeps
the score and drops the noise.

## Method

1. **Pairwise redundancy.** On training rows, every pair with |r| > 0.95 within the best plan-02
   set. For each pair keep the member with the higher gain importance; drop the other. (For a
   tree this is mostly about speed and `feature_fraction` dilution; for Ridge it is what the L2
   penalty is fighting.)
2. **Backward group ablation.** From the best plan-02 step, remove one group at a time and
   re-run all folds. A group whose removal changes the headline by < 0.05 is dead weight.
3. **Bottom-of-importance sweep.** Drop every column with zero or near-zero gain across all
   folds; confirm the headline does not move.
4. **Correlation-with-target scan of every engineered column**, for the notebook — so the
   selection is explainable, not just "the tree liked it".

## Success criteria

- A `lean_v3` set that scores within noise of the full set with materially fewer columns.
- A written reason for every dropped group.

## Output

`Findings/v3/07-feature-selection.md` and the selection section of
`notebooks/features new/01_feature_exploration.ipynb`.
