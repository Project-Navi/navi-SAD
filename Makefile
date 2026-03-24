.PHONY: test test-gpu lint typecheck all

test:
	pytest tests/ -m "not gpu" -v --tb=short

test-gpu:
	pytest tests/ -m "gpu" -v --tb=short

lint:
	ruff check src/

typecheck:
	mypy src/

all: lint typecheck test
