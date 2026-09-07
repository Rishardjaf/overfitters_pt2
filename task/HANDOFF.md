# Hand-off — what to do next, in order

For the model picking this up: everything below is written so you can act without re-reading
the whole project. Do the steps **in order**, one experiment at a time, and write results as
you go. Budget matters — the "cost" column tells you what each step spends.

---

## 0. Context in ten lines

- Task: predict `PM2_5_next_hour` for 12 stations. Data in `new data/train_new.csv` and
  `new data/test_new.csv`. **There is no `current_PM2_5` column any more** — the old
  "predict this hour's value" anchor is gone from both files. Metric: RMSE, lower is better.
- Scoring locally: `scripts/v3/run_experiment.py` trains on earlier months and scores on
  four later windows (`src/pm25/validation.py`). **Headline = mean RMSE over folds 0 and 2**
  (the two September–February windows, like the test period).
- Where we are: raw-column LightGBM **30.80** → best honest model **25.94** (LightGBM,
  `f9_deep` features, `ratio` target). The submitted file is
  `submissions/v3_lgbm_deep_ratio_800x3.csv`. **The goal is 17.0** — 8.9 away.
- Be honest about that goal: the deleted anchor alone would score 21.3 on these folds. If the
  steps below do not reach 17, say so plainly in the findings; do not chase it with anything
  that reads the future or reconstructs PM2.5 on the test set (see §1).
- Read these two files first, nothing else: `Findings/v3/README.md` (the story in one table)
  and `Findings/v3/06-target-angles.md` (the current best and why).

## 1. Rules that must not be broken

1. **Every feature must be built from `new data/train_new.csv` and `new data/test_new.csv`
   only.** Never read, join, or derive a statistic from the old `Data/train.csv` /
   `Data/test.csv` — those belong to a different, easier problem and touching them in any way
   (as a feature, a fitted statistic, a fallback, a sanity value) is leakage. Concretely: never
   import `TRAIN_CSV`/`TEST_CSV`/`DATA_DIR` from `pm25.config` into any v3 script — those
   constants point at the old files. Only import schema constants from it (`STATION_COL`,
   `TARGET`, `TIME_COL`, `SEED`, `N_TEST_ROWS`), as the existing scripts already do. Verify with
   `grep -rn "Data/train\|Data/test\|DATA_DIR\|TRAIN_CSV\|TEST_CSV" scripts/v3/` — it must
   return nothing except the two `new data/*_new.csv` paths defined in `features_v3.py`.
   A narrower consequence of this rule: no feature may use `current_PM2_5`,
   `PM2_5_next_hour`, or `diag_pm25_now` as model input, since those are either absent from the
   new files or are the target itself. `scripts/v3/features_v3.py::assert_no_pm25()` refuses
   such feature lists — do not bypass it. `diag_pm25_now` is a reconstruction made **from the
   new data's own target column** (previous row's label, shifted — see `new data/migration.md`
   §1, and it is legitimate exactly because it never touches the old files) and exists only so
   results can be scored against the old persistence baseline for comparison — it must never
   reach a feature list or a fitted statistic.
2. **Nothing looks forward in time.** Only `shift(k>0)`, trailing windows, same-hour
   cross-station aggregates. Run `uv run python src/pm25/audit.py` and
   `uv run python scripts/v3/test_causality.py` after any change to `features_v3.py`.
3. **Every fitted statistic is fitted inside the training fold** (the runner already does this
   for the proxy/ratio targets and the climatology encoding).
4. **Do not implement `plan v3 changed/05-optional-recursive-forecast.md`** (chaining the
   model's own test predictions to rebuild PM2.5) unless the user explicitly asks. It is the
   one remaining large lever and it needs their sign-off.
5. **One experiment at a time.** Never launch two queues. Each LightGBM run on all four folds
   takes 5–10 minutes on this machine; screen on the heating folds only (`--folds 0 2`) to
   halve that, and confirm winners on all four.
6. Never quote a result against 19.7 / 19.67 — those belong to the old data.

## 2. Where everything is

| what | where |
|---|---|
| feature builder (groups, sets f0…f10) | `scripts/v3/features_v3.py` — `--rebuild` regenerates `artifacts/v3/panel.parquet` |
| experiment runner | `scripts/v3/run_experiment.py --name X --features f9_deep --target ratio [--model lightgbm|xgboost|catboost|ridge|histgb] [--params '{...}'] [--folds 0 2] [--add-groups g] [--drop-groups g] [--drop-cols c...]` |
| run several in sequence, logged | `scripts/v3/queue.sh "name args…" "name args…"` → logs in `artifacts/v3/logs/<name>.log` |
| **see every result in one table** | `uv run python scripts/v3/summarize.py` (optional substring filter). Use this instead of reading logs. |
| results / OOF predictions | `artifacts/v3/results/<name>.json`, `artifacts/v3/oof/<name>.parquet` |
| blend evaluation | `uv run python scripts/v3/blend.py name1 name2 …` |
| write a submission | `uv run python scripts/v3/make_submission.py --name N --models lightgbm:ROUNDS [xgboost:R catboost:R] --features f9_deep --target ratio --seeds 42 1 2 [--weights w1 w2 …]` |
| findings (plain-language results) | `Findings/v3/` — one file per plan; `README.md` is the index and story table |
| plans | `plan v3 changed/` |
| feature exploration notebook | `notebooks/features new/01_feature_exploration.ipynb` (executed; reads results JSON, does not retrain) |
| EDA of the new data | `notebooks/eda_v2/01_eda_new_data.ipynb` |
| what changed vs the old data | `new data/migration.md` |

Targets in the runner: `level` (raw), `proxy` (`y − k_station·PM10`, add back), `proxy_sm`
(worse, ignore), `log1p` (worse, ignore), **`ratio`** (`y ÷ PM10` with sample weight `PM10²`,
multiply back — the winner). Feature sets: `f0_base` … `f8_full` are the cumulative build
steps; **`f9_deep`** = f7 + fine-fraction physics (the winner); `f10_lean` = f9 minus
acceleration and zero-gain flags (within noise, faster).

## 3. Step-by-step

### Step 0 — Harvest the runs that were still going at hand-off  (cost: ~0, just read)

These were queued one-at-a-time under a scheduler when the previous session stopped. They
finish on their own and write to `artifacts/v3/results/`:

`cat_f9_ratio`, `ridge_f9_ratio`, `histgb_f9_ratio`, `xgb_f9_ratio_mcw`,
`lgbm_f9_ratio_reg`, `lgbm_f7_ratio_te`, `lgbm_f7_ratio_wboth`, `lgbm_f10_lean_proxysm`.

1. `uv run python scripts/v3/summarize.py` — confirm they are present. If any is missing,
   check `artifacts/v3/logs/<name>.log` for a traceback; rerun only that one via `queue.sh`.
   (If nothing is running: `pgrep -fl run_experiment` is empty.)
2. Fill the `_pending_` rows in `Findings/v3/04-algorithms-and-blend.md` and
   `Findings/v3/06-target-angles.md` from the table. Keep the plain-language style.
3. Decisions to record:
   - `xgb_f9_ratio_mcw` vs `lgbm_f9_deep_ratio` (25.94): if within 0.5, XGBoost is a blend
     candidate; if still far behind, note that `min_child_weight` scaling was not enough and
     drop XGBoost.
   - `lgbm_f9_ratio_reg`: if it beats 25.94 by > 0.2 **and** its fold 2 is better than 26.99,
     the regularised parameters become the default for every later LightGBM run.
   - `cat_f9_ratio`: CatBoost was the most decorrelated tree in v2 (residual corr 0.977).
     Within 0.5 of the best → blend candidate.
   - `ridge_f9_ratio`: expected to be far behind; keep only if within ~1.5 (linear models
     extrapolate, which helped on the extreme winter with the level target).

### Step 1 — Seed average and blend  (cost: 2–4 runs + blend script)

1. If not already present, run one more LightGBM seed on the best config:
   `scripts/v3/queue.sh "lgbm_f9_ratio_s2 --features f9_deep --target ratio --max-rounds 6000 --params '{\"seed\":2}'"`
   (use the regularised params instead if Step 0 adopted them).
2. `uv run python scripts/v3/blend.py lgbm_f9_deep_ratio lgbm_f9_ratio_s1 lgbm_f9_ratio_s2 <every candidate from Step 0>`
3. Adopt a blend only if the NNLS blend beats the best single model on **both** "fit on one
   fold → score on the other" lines. Otherwise use the equal-weight seed average. Record in
   `Findings/v3/04`.

### Step 2 — Refined submission  (cost: one `make_submission.py` run, ~5 min)

The round count matters and is measured (`artifacts/v3/results/round_curve_f9_ratio.json`):
at lr 0.05, fold 0 wants ~1,500 rounds, fold 2 wants ~100, and the average is best near
**200–300**. The current submission used 800 (slightly past the optimum). Write:

`uv run python scripts/v3/make_submission.py --name v3_lgbm_deep_ratio_300x3 --models lightgbm:300 --features f9_deep --target ratio --seeds 42 1 2`

If Step 1 adopted a blend, add the other models with their own round counts (CatBoost: use
the fold-2 selected count × 1.2; see its JSON `fit_info`) and `--weights`. If Step 0 adopted
the regularised params, pass `--params '{"lightgbm": {...}}'`. Sanity-check the JSON the
script prints: prediction mean ≈ 90 (the test season is polluted), median pred ÷ PM10 ≈ 0.8,
no NaN, 51,063 rows. Tell the user which file to submit and the expected range (CV headline
→ leaderboard has landed ~2–3 below the heating headline in the past).

### Step 3 — The remaining honest levers, in priority order  (cost: 1–3 runs each; screen with `--folds 0 2`)

Try these in order and stop when two in a row give nothing. Each is one `queue.sh` line.

| # | idea | why it might work | exact run | keep if |
|---|---|---|---|---|
| 3a | **Regularised trees for the winter** (if not settled in Step 0) | fold 2 over-fits after ~100 rounds; fewer leaves / bigger min-leaf may transfer better to an unseen winter | `--features f9_deep --target ratio --params '{"num_leaves":15,"min_data_in_leaf":500,"lambda_l2":30,"feature_fraction":0.5}'` | fold 2 < 26.9 and headline < 25.9 |
| 3b | **Lower learning rate final** | free small gain for the submission only | in `make_submission.py`: `--params '{"lightgbm":{"learning_rate":0.02}}' --models lightgbm:800` | always do this for the final file; do not CV it |
| 3c | **Robust ratio denominator** | a single bad PM10 reading makes the ratio target explode; use a blend of own PM10 and the city median as denominator | add a column in `features_v3.add_network_levels`: `PM10_robust = 0.7*PM10_filled + 0.3*net_PM10_median`, then in `run_experiment.make_target/back_to_level` use it for `ratio` (add a target name `ratio_robust`) | headline < 25.8 |
| 3d | **Network-median × humidity / wind interactions** | the tree's top feature is the city-wide PM10 level; its interaction with dispersion conditions is not yet explicit | in `add_deep`: `netPM10med_x_ddd`, `netPM10med_x_WSPM`, `netPM10med_x_wind_v`; rebuild; `--features f9_deep --target ratio` | headline < 25.8 |
| 3e | **Per-station error audit** | find which stations carry the error; a station-specific fine-fraction feature may follow | small script over `artifacts/v3/oof/lgbm_f9_deep_ratio.parquet`: RMSE by station and by hour; write to `Findings/v3/08-residuals.md` | informs 3f |
| 3f | **Station × PM10 interaction** | fine fraction differs by station | `PM10_x_station_<s>` one-hot products (12 cols) in `add_deep` | headline < 25.8 |
| 3g | **Huber objective as a diagnostic** | shows how much the tails drive the fit; not a candidate for submission | `--params '{"objective":"huber","alpha":50}'` | never submitted; record only |
| 3h | Ridge on the ratio target as a blend member | linear models extrapolate; may decorrelate | already queued as `ridge_f9_ratio`; if within 1.5 of the best, include in `blend.py` | blend wins both fold directions |

What **not** to retry (measured, dead): second-derivative/acceleration features (+0.4), weather
lags (+0.1), log1p target (+2.0), climatology target encoding (+0.3 on level), season-specific
proxy `k` (+1.7), heating/recency sample weights (+0.2), more boosting rounds on the ratio
target (fold 2 degrades), XGBoost with default `min_child_weight` under `PM10²` weights.

### Step 4 — Close out  (cost: writing only)

1. Update `Findings/v3/README.md`'s story table with the final headline and the submitted
   file name.
2. If 17.0 was not reached, write one paragraph in the README saying what the best honest
   number is, how far from 17.0 it is, and that the only untested large lever is the gated
   plan 05 — the user decides that, not you.
3. Nothing is committed to git; leave that to the user unless asked.

## 4. Things that will bite you

- **`scripts/v2/run_experiment.py` and `scripts/v3/run_experiment.py` share a module name.**
  `scripts/v3/run_experiment.py` and `make_submission.py` already load the v2 one by explicit
  path; any new script must do the same (copy the `importlib.util` block from
  `scripts/v3/make_submission.py`). Importing `run_experiment` by name from a script in
  `scripts/v3/` will pick the wrong file.
- The panel cache is `artifacts/v3/panel.parquet`; after editing `features_v3.py` run
  `uv run python scripts/v3/features_v3.py --rebuild`, then the causality test, then the audit.
  Adding a group never changes existing sets, so in-flight runs are safe.
- The runner's early stopping picks the round count on the previous-year same-season window.
  On the ratio target that count is erratic (100–6,000). For **comparisons** that is fine (all
  arms suffer equally); for the **submission** use the measured curve (Step 2).
- Python is `uv run --no-sync python …` (project venv, Python 3.12). Do not `pip install`.
- Seed noise on this problem is ±0.1 headline. Treat differences under 0.2 as noise unless
  they are the same sign on both heating folds.
- The IDE shows "import could not be resolved" warnings for every script — that is the
  editor's environment, not a real error.

## 5. Budget guidance

- Reading: `summarize.py` output is ~30 lines; the two findings files in §0 are short. That is
  all the context you need to start.
- Each full-fold LightGBM run ≈ 5–10 min of machine time and ~1 tool call to launch + 1 to
  read. Screening with `--folds 0 2` halves the machine time.
- Do not write a finding for a run until you have read its numbers from `summarize.py`. Never
  put a number in a document from memory.
- If you are asked for a submission at any point, Step 2's command produces one in ~5 minutes
  from whatever the current best config is.
