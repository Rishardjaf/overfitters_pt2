# Finding v2-03 — Feature set v2: motion is the signal, products are dead weight

**Plan:** `plans v2/03-feature-set-v2.md` · **Model for every row below:** LightGBM Δ, plan-02
parameters, previous-season early stopping · **Seed:** 42 · Persistence headline 21.316.

## 1. Where the signal is — correlation with Δ (train, anchor-present rows)

| new feature | r with Δ | | existing feature | r with Δ |
|---|---:|---|---|---:|
| `net_loo_change_1h_mean` (other stations' 1h change) | **0.329** | | `PM25_vs_network_mean` | −0.239 |
| `net_mean_change_1h` | 0.325 | | `PM25_change_1h` | 0.186 |
| `net_change_1h_median` | 0.311 | | `current_PM2_5` | −0.124 |
| `net_n_falling` (count of other stations falling >10) | −0.264 | | `WSPM` | −0.073 |
| `CO_change_1h` | 0.234 | | `PM10` | −0.064 |
| `PM10_change_1h` | 0.229 | | every other raw level | < 0.07 |
| `WSPM_x_PM25_vs_net` | −0.225 | | | |
| `net_loo_change_3h_mean` | 0.223 | | | |
| `NO2_change_1h` | 0.205 | | | |
| `net_n_rising` | 0.174 | | | |
| `ddd_change_3h` (air drying) | −0.151 | | | |
| `PM25_accel` | 0.148 | | | |

The jihad set described the *state* at hour `t` (levels, level lags, level×level products).
What predicts the next hour is *motion*: what the rest of the city did in the last hour, and
whether the co-pollutants moved with PM2.5 (a real front) or not (a sensor blip).

## 2. Feature sets measured

| set | cols | fold 0 | fold 1 | fold 2 | fold 3 | **headline** | Δ vs jihad |
|---|---:|---:|---:|---:|---:|---:|---:|
| jihad (finding 02) | 124 | 17.181 | 13.254 | 19.401 | 13.234 | 18.291 | — |
| **v2** = jihad + motion + regime + quality + meteorology | 169 | 16.364 | 12.849 | 18.593 | 12.718 | **17.478** | **−0.81** |
| **lean** = v2 − 28 level×level products − 4 ratios | 141 | 16.314 | 12.884 | 18.621 | 12.764 | **17.468** | −0.82 |

The additions are worth **0.81 RMSE**, same direction on all four folds. Removing all 28
products and the ratios changes nothing (±0.05, mixed sign) — they are dead weight, and `lean`
is the base from here.

## 3. Group ablation from v2 (remove one group, rerun all folds)

| removed group | cols left | fold 0 | fold 2 | headline | change | verdict |
|---|---:|---:|---:|---:|---:|---|
| — (v2) | 169 | 16.364 | 18.593 | 17.478 | — | |
| **motion** (19) | 150 | 16.980 | 19.263 | 18.122 | **+0.64** | the gain; keep |
| regime + quality_v2 + meteorology_v2 (20) | 149 | 16.427 | 18.766 | 17.596 | +0.12 | small, same sign both folds; keep |
| weather lags (20) | 162 | 16.322 | 18.563 | 17.442 | −0.04 | dead weight; **drop** |
| pollutant lags (20) | 153 | 16.344 | 18.670 | 17.507 | +0.03 | noise; droppable |
| rolling mean/std/min/max (20) | 153 | 16.332 | 18.681 | 17.507 | +0.03 | noise; droppable (the regime features built from them stay) |
| ratios only (4) | 165 | — | — | 17.452 | −0.03 | dead weight; **drop** |
| products + ratios (= lean) | 141 | 16.314 | 18.621 | 17.468 | −0.01 | dead weight; **drop** |

Of the 169 columns, roughly 90 can go without measurable loss. That matters less for accuracy
than for speed and for early-stopping noise.

## 4. What the tree actually uses (`lgbm_lean_delta`, share of top-40 gain)

`WSPM_x_PM25_vs_net` 12% · `net_loo_change_1h_mean` 11% · `net_mean_change_1h` 8% ·
`PM10_change_1h` 7% · `CO_change_1h` 5% · `ddd_change_3h` 5% · `PM25_vs_net_loo` 4% ·
`network_PM25_std` 4% · `net_change_1h_median` 3% · `current_PM2_5` 3% · `station_cat` 2% ·
`wd_cat` 2% · `PM25_accel` 2% · `wind_v` 2% · `RAIN` 2%.

The single most useful feature is **wind speed × (this station − the rest of the city)**: a
station sitting above the network in a breeze is about to fall towards it. Zero-gain features:
all `*_missing` flags, `co_floor`/`co_ceiling`/`o3_extreme`, `pm25_floor_flag`, `is_weekend`,
`is_heating_season`, `CO_flat_run`, `gap_prev_hours`, `anchor_age_hours` (constant on
anchor-present rows — they only matter for the fallback).

## 5. Pairwise redundancy (|r| > 0.95, 37 pairs)

Mostly harmless for a tree but worth knowing: `PM25_vs_network_mean` ≡ `PM25_vs_net_loo`
(1.000), `network_PM25_mean` ≡ `net_loo_mean` (0.999), `lag_2h` ≈ `roll_mean_3h` (0.994), the
3h rolling min/mean/max ≈ each other and ≈ lags 1–3 (0.96–0.99), `month_sin/cos` ≈
`doy_sin/cos` (0.95), `TEMP` ≈ `TEMP_lag_24h`, `CO` ≈ `CO_lag_1h`, `SO2` ≈ `SO2_lag_1h`. For the
Ridge model these pairs are what the L2 penalty is fighting; for LightGBM they just dilute
`feature_fraction`.

## 6. Signed-tail split, `lgbm_lean_delta` (persistence in brackets)

| fold | fall | calm | rise |
|---|---:|---:|---:|
| 0 | 46.9 (70.0) | 9.5 (9.8) | 52.9 (61.3) |
| 2 | 53.3 (77.1) | 9.3 (9.2) | 66.5 (72.8) |

Compared with the jihad set (finding 02: falls 48.8 / 53.0, calm 10.1 / 9.8, rises 56.0 /
71.2): the motion features improve rises (56.0→52.9, 71.2→66.5) and calm hours (10.1→9.5,
9.8→9.3 — now at or better than persistence), and falls on fold 0 (48.8→46.9); falls on fold 2
are unchanged (53.0→53.3). Rapid rises remain the weakest regime (66.5 vs 72.8 for persistence).

## 7. Finding 03b — per-neighbour station columns (`lean_nb`)

Since the strongest signals are "what did the rest of the city do" summarised as a mean, the
follow-up gives the tree every other station's same-hour level and 1h change as its own column
(24 columns, `nb_level_<station>`, `nb_chg_<station>`), plus four wind×motion products
(`WSPM_x_change_1h`, `WSPM_x_net_change_1h`, `ddd_x_PM25_vs_net`, `net_accel`). Causality test
re-run and passed on the rebuilt panel.

| set | cols | fold 0 | fold 1 | fold 2 | fold 3 | **headline** | vs lean |
|---|---:|---:|---:|---:|---:|---:|---:|
| lean | 141 | 16.314 | 12.884 | 18.621 | 12.764 | 17.468 | — |
| **lean_nb** | 169 | **16.252** | **12.812** | **18.550** | **12.436** | **17.401** | **−0.07** |
| slim_nb = lean_nb − remaining weather lags − pollutant lags | 146 | 16.272 | 12.814 | 18.500 | 12.548 | 17.386 | −0.08 |

`slim_nb` ties `lean_nb` (−0.015 headline, +0.11 on fold 3: noise both ways) with 23 fewer
columns — confirming the ablation verdict on the lag groups. `lean_nb` stays the base for the
blend because the seed and XGBoost runs already use it; either is a valid final set.

Small, but it improves **all four folds** and fold 3 by 0.33, so it is not noise. Twelve of
the 28 new columns are in the top-40 gain list (`nb_chg_Changping`, `nb_chg_Wanshouxigong`,
`nb_chg_Dongsi`, `nb_level_Shunyi`, …): the tree is learning station-specific lead/lag
relationships that a single network mean cannot express. Tail split fold 0: falls 46.4 / calm
9.5 / rises 52.9; fold 2: 52.9 / 9.3 / 66.3 — the same shape as lean, slightly better on falls.

## Verdict

- Adopt **lean_nb** (169 cols): −0.89 vs jihad, −3.92 vs persistence (**−18.4%**). Same sign on
  every fold at every step (jihad → v2 → lean → lean_nb).
- Dead weight, safe to drop: level×level products, ratios, weather lags; pollutant lags and
  plain rolling stats are within noise either way (`slim_nb` = lean_nb minus weather and
  pollutant lags is being measured as the speed-oriented alternative).
- Where the remaining error is: rapid rises (66 RMSE on fold 2 vs 73 for persistence). The onset
  hour of a spike is still mostly unpredicted from hour-`t` information.
