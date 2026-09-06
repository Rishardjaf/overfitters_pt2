# Processed / cleaned data

No processed or intermediate dataset is included in this package as a file, because it is
fully and deterministically regenerable from the two competition-provided CSVs by the code in
`02_source_code/`.

## What the processed dataset is

`scripts/v3/features_v3.py` reads `new data/train_new.csv` and `new data/test_new.csv`, builds
a complete hourly grid per station, and computes the full feature panel described in §3 of
`01_Methodology_Report.md`. The result is cached at `artifacts/v3/panel.parquet` (≈218 MB,
420,756 rows × ~375 columns including experimental groups not used by the final model, plus a
`groups.json` recording which columns belong to which named feature group).

This file is a **cache**, not a hand-edited artefact — deleting it and re-running the
regeneration step below reproduces it byte-for-byte from the raw inputs, because every
transformation in `features_v3.py` is a deterministic function of the raw columns (no random
initialisation, no fitted-on-the-fly statistic baked into the cache itself — target-dependent
statistics such as climatology encodings are computed later, inside each cross-validation fold,
never written into this cache).

## Regeneration step

```bash
uv run python scripts/v3/features_v3.py --rebuild
```

Run from the project root with `new data/train_new.csv` and `new data/test_new.csv` present.
Takes under a minute and requires no GPU. See `05_README_Reproduction_Instructions.md` for the
full command sequence, including the leakage-audit and causality-test steps that should
immediately follow any regeneration.

## Feature set used for the final submission

The final model uses only the `f9_deep` feature set (284 columns) produced by this panel — the
cumulative build-up described in §3 of the methodology report, plus the fine-fraction physics
group (`deep`). Other feature sets and groups present in `features_v3.py` (`f10_lean`,
`f11_deep2`, and several exploratory groups added after this submission was generated) exist in
the shared code file but are **not** invoked by the commands that produced
`v3_lgbm_xgb_cat_ratio_800x3.csv` — no `--add-groups` or `--te-ratio` flag is passed in the
reproduction steps.
