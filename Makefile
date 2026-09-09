.PHONY: help install dev test lint format clean

help:  ## Show this help message
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install packages into venv (also creates if doesn't exist)
	uv sync

install-prod:  ## Install with development dependencies
	uv sync --no-dev

test:  ## Run tests
	uv run pytest tests/ -v

lint:  ## Run linters
	uv run python -m compileall -q llm_bench tests
	uv run flake8 llm_bench tests
	uv run mypy llm_bench tests

format:  ## Format code
	uv run autopep8 -a --in-place --recursive .

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete


run:  ## Run the benchmark CLI
	uv run llm-bench
