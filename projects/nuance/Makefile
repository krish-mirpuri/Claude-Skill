# Everything runs from here.
#   make setup   one-off: virtualenv + dependencies
#   make all     download -> train -> audit -> transfer -> active -> reports
#   make check   lint + tests
PY := .venv/bin/python
PIP := .venv/bin/pip
NUANCE := .venv/bin/nuance

.DEFAULT_GOAL := help
.PHONY: help setup data train audit transfer active report all api dashboard test lint fmt check clean distclean

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-11s\033[0m %s\n", $$1, $$2}'

setup:  ## create the virtualenv and install the package
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[serve,app,dev]"

data:  ## download the corpora (~20 MB) and verify checksums
	$(NUANCE) download

train:  ## fit the ladder, tune decoders, evaluate with intervals
	$(NUANCE) train

audit:  ## confident-learning audit of the training labels
	$(NUANCE) audit

transfer:  ## Reddit -> Twitter domain-transfer evaluation
	$(NUANCE) transfer

active:  ## annotation-budget simulation
	$(NUANCE) active

report:  ## regenerate reports/RESULTS.md from artifacts on disk
	$(NUANCE) report

all:  ## the whole pipeline, end to end
	$(NUANCE) all

api:  ## serve on http://localhost:8000 (docs at /docs)
	.venv/bin/uvicorn nuance.api.main:app --reload --port 8000

dashboard:  ## open the explorer
	.venv/bin/streamlit run src/nuance/dashboard/app.py

test:  ## run the test suite
	$(PY) -m pytest tests/ -q

lint:  ## check style
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

fmt:  ## apply formatting
	.venv/bin/ruff format .

check: lint test  ## what CI runs

clean:  ## remove generated artifacts, keep the downloads
	rm -rf data/processed/* models/* reports/*.csv reports/*.json \
		reports/RESULTS.md reports/figures/*.png
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

distclean: clean  ## also remove downloads and the virtualenv
	rm -rf data/raw/*.tsv data/raw/*.json data/raw/*.txt data/raw/*.zip \
		.venv .pytest_cache .ruff_cache
