.PHONY: test test-gpu lint format typecheck all

test:
	uv run pytest tests/ -m "not gpu" -v --tb=short

test-gpu:
	uv run pytest tests/ -m "gpu" -v --tb=short

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format --check src/ tests/

typecheck:
	uv run mypy src/

all: lint format typecheck test
