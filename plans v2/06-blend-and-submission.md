# Plan v2-06 — Blend and final submission

**Question.** Does a weighted blend of the OOF predictions from plan 04 beat the best single
model, and what goes in the submission CSV?

## Method

1. Collect OOF predictions (raw PM2.5 scale) from every retained model on the heating-season
   folds.
2. Fit non-negative blend weights on fold 0, evaluate on fold 2, and vice versa. Report both.
   A blend that only helps on the fold its weights were fitted on is overfit.
3. Compare against the best single model. Adopt the blend only if it wins on both folds by
   > 0.05.
4. Final model(s): retrain on **all** training rows with the round count chosen on fold 3's
   inner window scaled by the data-size ratio, build test features on the concatenated timeline
   (backward-looking only), predict, add the anchor back for Δ models, fall back to the level
   model where the anchor is missing, lower-bound at 0. No upper clipping.
5. Run `make audit` before writing the CSV. Sanity checks on the CSV: 51,063 rows, every `id`
   present once, no NaN, no negative, distribution of predictions vs `current_PM2_5` roughly
   matching the train relationship (mean prediction ≈ mean anchor + ~0).

## Submission format

`id,PM2_5_next_hour` — matching the training column name. Written to
`submissions/v2_<model>.csv`.

## Output

`Findings/v2/06-blend-and-submission.md` — includes the expected test RMSE range implied by the
heating-season folds, so the leaderboard number can be judged against it.
