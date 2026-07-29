.PHONY: setup bench reproduce demo test lint typecheck

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@echo "Run 'source $(VENV)/bin/activate' to enter the environment."

test:
	$(PYTHON) -m pytest -v

lint:
	$(PYTHON) -m ruff check src tests

typecheck:
	$(PYTHON) -m mypy src

bench:
	$(PYTHON) -m agentkv.bench.replay --config configs/experiments/phase0_smoke.yaml

reproduce:
	$(PYTHON) experiments/run_all.py

demo:
	$(PYTHON) -m agentkv.viz.dashboard
