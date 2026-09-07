"""Submission generation.

Not yet implemented — see the roadmap in PROJECT.md.

    uv run python -m pm25.predict --config configs/baseline.yaml

A submission is `id,PM2_5_next_hour` for all 51,063 test rows. Before writing
one, assert: row count matches, no NaN predictions, no negative predictions, and
every test `id` appears exactly once.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.parse_args()
    raise NotImplementedError("Prediction pipeline not yet implemented — see PROJECT.md")


if __name__ == "__main__":
    main()
