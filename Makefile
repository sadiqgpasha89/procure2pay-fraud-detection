.PHONY: setup data train rag bootstrap up down status test test-unit test-e2e benchmark compose lint
PY ?= .venv/bin/python

setup:                ## create venv and install (add EXTRAS=ollama,gcp,dev as needed)
	uv venv --python 3.12 .venv && uv pip install --python $(PY) -e ".[dev,ollama]"
data:                 ## synthetic warehouse + documents + labels
	$(PY) -m p2p.data.synthetic
train:                ## features → LightGBM/IsolationForest → model card
	$(PY) -m p2p.ml.train
rag:                  ## chunk + index contracts, policies, e-mails, dockets
	$(PY) -m p2p.rag.ingest
bootstrap: data train rag
up:                   ## start all 10 services locally (gateway :8100, console :8102)
	$(PY) -m p2p.stack up
down:
	$(PY) -m p2p.stack down
status:
	$(PY) -m p2p.stack status
benchmark:            ## replay held-out period through the live stack → docs/EVAL_REPORT.md
	$(PY) -m p2p.llmops.benchmark --fraud 60 --legit 80 --out docs/EVAL_REPORT.md
test: test-unit test-e2e
test-unit:
	$(PY) -m pytest tests/unit
test-e2e:             ## boots the full stack with auth enabled (needs free ports 8100-8302)
	$(PY) -m pytest tests/e2e
compose:
	docker compose -f deploy/docker/docker-compose.yml up --build
lint:
	$(PY) -m ruff check src tests
