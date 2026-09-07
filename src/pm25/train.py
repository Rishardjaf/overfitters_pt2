"""Time-aware CV training entry point.

Not yet implemented — see the roadmap in PROJECT.md.

    uv run python -m pm25.train --config configs/baseline.yaml

Each run must record, to reports/experiments.md: the config name, git SHA,
per-fold RMSE, the heating-season mean RMSE (the headline), the delta against
persistence on the same folds, and the feature list. Out-of-fold predictions go
to artifacts/oof/ so that blending is possible later without retraining.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.parse_args()
    raise NotImplementedError("Training pipeline not yet implemented — see PROJECT.md")


if __name__ == "__main__":
    main()
