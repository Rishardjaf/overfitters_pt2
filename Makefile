.PHONY: help setup eda baseline train submit audit fmt test clean

CONFIG ?= configs/baseline.yaml
PY     := uv run python

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the virtualenv and install dependencies (Python 3.12)
	uv sync --extra dev

eda:  ## Regenerate the data profile into reports/
	$(PY) scripts/profile_data.py

baseline:  ## Score the persistence and mean baselines on the time-aware folds
	$(PY) -m pm25.train --config configs/baseline.yaml

train:  ## Train with time-aware CV. Override with CONFIG=configs/<name>.yaml
	$(PY) -m pm25.train --config $(CONFIG)

submit:  ## Generate a submission from a trained model
	$(PY) -m pm25.predict --config $(CONFIG)

audit:  ## Leakage guard — fail on future-looking operations in the pipeline
	$(PY) -m pm25.audit

fmt:  ## Format and lint
	uv run ruff format src scripts tests
	uv run ruff check --fix src scripts tests

test:  ## Run the test suite
	uv run pytest -q

clean:  ## Remove caches and generated artifacts
	rm -rf .ruff_cache .pytest_cache **/__pycache__
	find artifacts -type f ! -name '.gitkeep' -delete
