# Everything in this project runs from here.
#   make setup   one-off: virtualenv + dependencies
#   make all     data -> model -> backtest -> reports (about a minute)
#   make check   lint + tests
PY := .venv/bin/python
PIP := .venv/bin/pip
OVERBOOK := .venv/bin/overbook

.DEFAULT_GOAL := help
.PHONY: help setup data prepare train backtest report all api dashboard tune test lint fmt check clean distclean

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## create the virtualenv and install the package
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[serve,app,explain,dev]"

data:  ## download the raw extract (17 MB) and verify its checksum
	$(OVERBOOK) download

prepare: data  ## clean into the canonical booking table
	$(OVERBOOK) prepare

train:  ## fit, calibrate, evaluate, write model reports
	$(OVERBOOK) train

backtest:  ## replay the authorisation policies over held-out nights
	$(OVERBOOK) backtest

report:  ## regenerate reports/RESULTS.md from the artifacts on disk
	$(OVERBOOK) report

all:  ## the whole pipeline, end to end
	$(OVERBOOK) all

tune:  ## re-run the hyperparameter search (slow; updates reports/tuning.csv)
	$(PY) scripts/tune_lightgbm.py

api:  ## serve the model on http://localhost:8000 (docs at /docs)
	.venv/bin/uvicorn overbook.api.main:app --reload --port 8000

dashboard:  ## open the revenue-management console
	.venv/bin/streamlit run src/overbook/dashboard/app.py

test:  ## run the test suite
	$(PY) -m pytest tests/ -q

lint:  ## check style
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

fmt:  ## apply formatting
	.venv/bin/ruff format .

check: lint test  ## what CI runs

clean:  ## remove generated artifacts, keep the raw download
	rm -rf data/interim/* data/processed/* models/* reports/*.csv reports/*.json \
		reports/*.parquet reports/RESULTS.md reports/figures/*.png
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

distclean: clean  ## also remove the raw download and the virtualenv
	rm -rf data/raw/hotel_bookings.csv .venv .pytest_cache .ruff_cache
