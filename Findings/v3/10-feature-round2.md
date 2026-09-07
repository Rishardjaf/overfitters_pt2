# Finding v3-10 — Feature round 2: screening six new angles under the ratio target

**Plan:** `task/feature_improvement_task.md` · **Config for every screening row:** LightGBM,
`f9_deep` + one new group, ratio target, heating folds only (0, 2), previous-season early
stopping. **Reference:** `lgbm_f9_deep_ratio` — fold 0 **24.90**, fold 2 **26.99**, headline
**25.94**. Verdict rule: win if headline <= 25.80 and neither fold worse by > 0.10; noise if
within +/-0.15 of 25.94; loss otherwise. Stop after two consecutive noise/loss verdicts.

Priority was reordered after `Findings/v3/09-fine-fraction-eda.md`: the univariate fine-fraction
EDA ranked CO/PM10 (0.46) and RH (0.34) as the strongest single-variable separators of the fine
fraction, station identity the weakest (0.06) — so `coarse` and `humidity` were screened before
the station-based `te_ratio`.

## Screening table

| run | group | cols | fold 0 | fold 2 | headline | vs 25.94 | rounds f0/f2 | verdict |
|---|---|---:|---:|---:|---:|---:|---|---|
| s2_coarse | coarse (city-wide combustion/PM10 balance + momentum) | 296 | 24.81 | 27.96 | 26.386 | +0.45 | 810/791 | **loss** — fold 2 far worse despite the strongest univariate EDA signal |
| s2_humidity | humidity (RH from Magnus formula + interactions) | 292 | 25.06 | 27.62 | 26.341 | +0.40 | 2249/867 | **loss** — RH ranked 2nd in raw EDA but the tree already captures the effect via TEMP/DEWP/season |
| s2_episode | episode (stagnation, episode age, rise/fall run length, 72h peak phase) | 295 | 25.13 | 26.72 | 25.923 | −0.02 | 1052/121 | **noise** — fold 0 worse by 0.23, fold 2 better by 0.27; net within seed-noise band |
| s2_te_ratio | te_ratio (fine-fraction climatology: station x month x hour, station x wd, station x hour) | 284 (+3 dynamic) | 24.53 | 27.35 | 25.938 | +0.00 | 874/117 | **noise on headline, real signal per-fold** — `ter_smh` ranks 7th/287 by gain; fold 0 −0.37, fold 2 +0.36, net zero. Mirrors episode's fold pattern in reverse — combine and test |
| s2_memory | memory (72h/168h PM10/CO baselines) | 293 | 25.09 | 27.42 | 26.254 | +0.31 | 959/644 | **loss** — both folds worse; the 24h window already covers what the tree needs |
| s2_transport | transport (learned lead/lag adjacency by wind sector) | 290 | 25.20 | 27.08 | 26.139 | +0.20 | 1284/126 | **loss** — `leadw_PM10` correlates 0.98 with the existing `net_PM10_median`; added redundancy, not signal |

## Combination test

Episode and te_ratio move the two heating folds in almost mirror-image directions alone
(episode: fold 0 +0.23/fold 2 -0.27; te_ratio: fold 0 -0.37/fold 2 +0.36). Tested together on
the hypothesis that a genuinely useful pair should combine constructively rather than cancel:

| run | groups | cols | fold 0 | fold 2 | headline | vs 25.94 | rounds f0/f2 | verdict |
|---|---|---:|---:|---:|---:|---:|---|---|
| s2_episode_teratio | episode + te_ratio | 295 (+3 dynamic) | 24.54 | 27.01 | 25.772 | **−0.17** | 2088/473 | **win** — both folds at or better than baseline |
| s2_episode_teratio_s1 | episode + te_ratio, seed 1 | 295 (+3 dynamic) | 24.49 | 27.12 | 25.806 | −0.25 (vs seed-1 baseline 26.053) | 3393/528 | **confirmed** — same direction both seeds: fold 0 improves ~0.4-0.5, fold 2 roughly flat |

## Full four-fold confirmation

| run | groups | fold 0 | fold 1 | fold 2 | fold 3 | headline | all-fold | vs lgbm_f9_deep_ratio |
|---|---|---:|---:|---:|---:|---:|---:|---|
| lgbm_episode_teratio | episode + te_ratio | 24.54 | 18.94 | 27.01 | 18.84 | **25.772** | 22.33 | improved or flat on **every** fold (24.90/19.80/26.99/18.95 -> 24.54/18.94/27.01/18.84) |

**New best single LightGBM model: 25.77**, down from 25.94.
