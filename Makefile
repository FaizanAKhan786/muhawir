.PHONY: setup dev test lint eval docker-up docker-down ingest seed-golden-set k8s-deploy

setup:
	pip install -e ".[dev]"
	pre-commit install

dev:
	docker compose up -d qdrant redis postgres langfuse
	uvicorn src.api.main:app --reload --port 8000

test:
	pytest tests/ -v --tb=short

lint:
	ruff check .
	mypy src/

eval:
	python scripts/run_eval.py --golden-set data/golden_sets/golden_v1.0.json --log-to-mlflow

docker-up:
	docker compose up -d

docker-down:
	docker compose down

ingest:
	python scripts/ingest_corpus.py --source data/sample_docs/

seed-golden-set:
	python scripts/seed_golden_set.py

k8s-deploy:
	kubectl apply -k k8s/
