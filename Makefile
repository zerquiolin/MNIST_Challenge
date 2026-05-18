.PHONY: format lint typecheck test clean

CONDA_ENV ?= mnist-env
PYTHON = conda run -n $(CONDA_ENV) python

setup:
	conda env update -f environment.yml --prune

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m mypy app/ src/

test:
	$(PYTHON) -m pytest tests/ --cov

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +
