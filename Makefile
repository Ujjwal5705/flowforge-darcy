PYTHON := .venv/bin/python
export MPLCONFIGDIR := .cache/matplotlib
export XDG_CACHE_HOME := $(CURDIR)/.cache
EXPERIMENT ?= artifacts/reproduction

.PHONY: data inspect smoke train evaluate build test serve docker
data:
	$(PYTHON) scripts/fetch_pdebench.py
inspect:
	$(PYTHON) -m ml.pipeline inspect
smoke:
	$(PYTHON) scripts/smoke.py
train:
	$(PYTHON) -m ml.pipeline train --output $(EXPERIMENT)
evaluate:
	$(PYTHON) -m ml.pipeline evaluate --artifact $(EXPERIMENT) --publish-report
build:
	npm run build --prefix frontend
test:
	$(PYTHON) -m pytest -q
	.venv/bin/ruff check ml backend deployment scripts tests
serve:
	$(PYTHON) -m uvicorn backend.app:app --host 127.0.0.1 --port 7860 --workers 1 --limit-concurrency 16
docker:
	docker compose up --build -d
