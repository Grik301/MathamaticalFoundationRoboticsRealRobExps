PYTHON ?= python3.13
VENV_PYTHON := .venv/bin/python

.PHONY: setup dependencies assets equations run slam test verify

setup: dependencies assets equations

dependencies:
	$(PYTHON) -m venv .venv
	$(VENV_PYTHON) -m pip install -r requirements.txt
	npm ci --prefix .mathjax

assets:
	$(VENV_PYTHON) -m lecture6_sim.download_assets

equations:
	$(VENV_PYTHON) -m lecture6_sim.typeset

run:
	$(VENV_PYTHON) -m lecture6_sim

slam:
	$(VENV_PYTHON) -m lecture6_sim --pages 59

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(VENV_PYTHON) -m pytest tests -q

verify:
	$(VENV_PYTHON) -m lecture6_sim.verify
