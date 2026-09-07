# overfitters — next-hour PM2.5 forecasting

Hackathon entry forecasting PM2.5 one hour ahead across 12 urban air-quality
monitoring stations. Scored by **RMSE**; lower is better.

- **[PROJECT.md](PROJECT.md)** — problem statement, data dictionary, EDA
  findings, modelling plan, validation protocol.
- **[CLAUDE.md](CLAUDE.md)** — working agreement, rules, commands.
- **[reports/data_profile.md](reports/data_profile.md)** — generated data profile.
- **[reports/experiments.md](reports/experiments.md)** — run log.

## Quick start

```bash
make setup     # venv + dependencies (uv, Python 3.12)
make eda       # regenerate the data profile
make baseline  # score persistence and mean baselines
make train     # time-aware CV training
```

`make eda` runs on stdlib alone, so it works before `make setup`.

## The number to beat

| Baseline | Train RMSE |
|---|---:|
| Global mean | 77.78 |
| **Persistence (`ŷ = current_PM2_5`)** | **19.67** |

Persistence already removes ~75% of the error. The task is the residual: the
hour-over-hour *change* in PM2.5.

## Ground rules

Data in [Data/](Data/) is read-only. No external data, no attempt to identify the
source dataset, no recovering hidden test targets — including the tempting
next-row self-join on the test set. Validation is time-aware only; a random split
leaks near-identical adjacent hours. See [CLAUDE.md](CLAUDE.md).
