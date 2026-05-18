.PHONY: setup format lint typecheck test model train docker-build docker-run docker-up docker-down clean

CONDA_ENV ?= mnist-env
PYTHON = conda run -n $(CONDA_ENV) python
IMAGE_NAME ?= mnist-classifier
APP_PORT ?= 8000
MODEL_EPOCHS ?= 22

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

api:
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port $(APP_PORT) --reload

model:
	$(PYTHON) -m src.utils.model_generator --epochs $(MODEL_EPOCHS)

train:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute notebooks/challenge.ipynb --inplace

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run --rm -p $(APP_PORT):8000 $(IMAGE_NAME)

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +
