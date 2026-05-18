# MNIST Digit Classifier

Production-oriented MNIST digit classification project built with PyTorch and exposed through a FastAPI inference API.

The project includes a complete notebook-based challenge solution, reusable Python modules, model artifact generation, API validation, automated tests, and Docker support. It is also structured for bilingual documentation through English and Spanish notebooks and README files.

## Documentation Languages

- English documentation: `README.md`
- Spanish documentation: `README_es.md`
- English challenge notebook: `notebooks/challenge.ipynb`
- Spanish challenge notebook: `notebooks/challenge_es.ipynb`

## Project Context

This project solves the senior machine learning kata for classifying handwritten digit images through a Python API. The challenge asks for a model capable of receiving a handwritten digit image and returning its predicted class through an HTTP interface.

The solution is a production-oriented PyTorch neural network implementation with clear separation of concerns, clean code practices, and modular components for data loading, preprocessing, model definition, training, evaluation, inference, API validation, and deployment. The full training and evaluation pipeline is documented in `notebooks/challenge.ipynb` and its Spanish version, `notebooks/challenge_es.ipynb`. Those notebooks include the design principles, implementation choices, justifications, training process, learning behavior, strengths, limitations, and possible future improvements.

The project also includes a FastAPI application that exposes the trained model over HTTP. The `/predict` endpoint receives a base64-encoded image string, validates it, preprocesses it, and returns the predicted digit with a confidence score. Accepting base64 image input can be risky if left unchecked, so the API includes safeguards for malformed payloads, unsupported image types, oversized inputs, corrupted files, unsafe image dimensions, and invalid encodings.

To support maintainability and transparency, the repository includes unit and end-to-end tests that cover validation, preprocessing, inference behavior, and API endpoints. The Makefile provides a consistent command interface for setup, model generation, notebook execution, testing, local API execution, and Docker workflows. The same project can be run locally with Conda or packaged in Docker for deployment-oriented usage.

## Project Structure

```text
.
├── app/                         # FastAPI application, routes, schemas, validation, inference
├── artifacts/                   # Generated model artifacts, including mnist_classifier.pt
├── images/                      # Sample handwritten digit images for API/e2e validation
├── notebooks/                   # Challenge solution notebooks and exploratory work
│   ├── challenge.ipynb          # Complete English challenge solution
│   ├── challenge_es.ipynb       # Complete Spanish challenge solution
│   └── experiments.ipynb        # Supporting exploratory notebook
├── src/                         # Data loading, model, training, preprocessing, utilities
├── tests/                       # Unit and end-to-end tests
├── Dockerfile                   # Container image definition
├── docker-compose.yml           # Docker Compose service definition
├── environment.yml              # Conda environment definition
├── Makefile                     # Main local, test, model, and Docker commands
├── README.md                    # English documentation
├── README_es.md                 # Spanish documentation
└── requirements.txt             # Pip dependencies used by Docker
```

## Challenge Solution

The complete English challenge solution is available in:

```text
notebooks/challenge.ipynb
```

The complete Spanish challenge solution is available in:

```text
notebooks/challenge_es.ipynb
```

The notebook solution walks through the full ML workflow:

- MNIST dataset loading.
- Data augmentation and preprocessing.
- CNN model architecture implemented in PyTorch.
- Training loop with validation.
- Model evaluation.
- Model artifact export.
- API-ready inference behavior.

Additional exploratory work is available in:

```text
notebooks/experiments.ipynb
```

## Local Setup

### Prerequisites

- Conda.
- Python 3.10 through the Conda environment defined in `environment.yml`.

Create or update the local Conda environment:

```bash
make setup
```

Activate it:

```bash
conda activate mnist-env
```

If your shell, editor, or notebook runtime cannot resolve imports from `app/` or `src/`, run this from the project root:

```bash
export PYTHONPATH=./
```

This usually happens in terminal-first or custom editor setups, such as Vim or manually configured VS Code environments. IDEs like PyCharm often detect the project root automatically.

## Running Notebooks

Start Jupyter from the repository root with the Conda environment active:

```bash
jupyter notebook
```

Main notebooks:

```text
notebooks/challenge.ipynb
notebooks/challenge_es.ipynb
```

You can also execute the configured challenge notebook from the command line:

```bash
make train
```

`make train` currently executes the English notebook path configured in the `Makefile`.

## Generating the Model Artifact

The API and Docker image expect the trained model artifact at:

```text
artifacts/mnist_classifier.pt
```

Generate the model with:

```bash
make model
```

By default, this trains for `22` epochs. Override the number of epochs with `MODEL_EPOCHS`:

```bash
make model MODEL_EPOCHS=5
```

The Docker image does not train the model during build. Generate the artifact before building or running the container.

The fastest path is `make model`, which uses the lightweight model generation module. You can also generate or refine the model through experimentation in `notebooks/challenge.ipynb` or `notebooks/challenge_es.ipynb`, then export the resulting weights to the same artifact path.

## Running the API Locally

Start the FastAPI service:

```bash
make api
```

Check the root endpoint:

```bash
curl http://localhost:8000/
```

Check model availability:

```bash
curl http://localhost:8000/status
```

Expected response when the model artifact is present:

```json
{"Status":"Ok"}
```

Interactive API documentation is available at:

```text
http://localhost:8000/docs
```

## Running with Docker

Docker requires Docker Desktop or the Docker daemon to be running.

Before building the image, make sure the model artifact exists:

```bash
make model
```

Build the image:

```bash
make docker-build
```

Run the container:

```bash
make docker-run
```

The API will be available at:

```text
http://localhost:8000
```

You can also use Docker Compose:

```bash
make docker-up
```

Stop the Compose service:

```bash
make docker-down
```

The Docker image includes a healthcheck that calls:

```text
http://127.0.0.1:8000/status
```

The container is healthy only when the endpoint returns:

```json
{"Status":"Ok"}
```

## Testing and Quality Checks

Run the test suite:

```bash
make test
```

Run linting and formatting checks:

```bash
make lint
```

Run static type checking:

```bash
make typecheck
```

Test coverage includes:

- Unit tests for preprocessing, validation, and inference behavior.
- End-to-end tests for API endpoints.
- Ruff checks for linting and formatting.
- Mypy checks for static typing.

## API Usage

### Endpoints

- `GET /`: returns a basic API welcome message.
- `GET /status`: returns model artifact availability.
- `POST /predict`: predicts the digit from a base64-encoded image.

### Prediction Request

```json
{
  "image_base64": "..."
}
```

### Prediction Response

```json
{
  "prediction": 5,
  "confidence": 0.99
}
```

For interactive request testing, use:

```text
http://localhost:8000/docs
```

## Troubleshooting

### Model Missing

If `/status` does not return `{"Status":"Ok"}`, generate the model artifact:

```bash
make model
```

Then confirm this file exists:

```text
artifacts/mnist_classifier.pt
```

### Import Errors

If Python cannot import modules from `app/` or `src/`, run this from the project root:

```bash
export PYTHONPATH=./
```

### Docker Daemon Unavailable

If Docker commands fail with a daemon connection error, start Docker Desktop or your Docker daemon, then run the command again.

### Docker Status Error

If the container starts but `/status` returns an error, the model artifact is missing inside the image or `MODEL_PATH` points to the wrong location.

Regenerate the model and rebuild the image:

```bash
make model
make docker-build
```
