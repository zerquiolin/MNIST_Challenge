from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.utils.preprocessing import image_to_base64

client = TestClient(app)

TEST_IMAGES = [
    ("images/4-hw.JPG", 4),
    ("images/5-hw.JPG", 5),
    ("images/7-hw.JPG", 7),
    ("images/9-hw.JPG", 9),
]


@pytest.mark.parametrize("image_path, expected_digit", TEST_IMAGES)
def test_predict_endpoint_success(image_path: str, expected_digit: int):
    path = Path(image_path)
    if not path.exists():
        pytest.skip(f"Test image {image_path} not found")

    base64_str = image_to_base64(path)
    
    payload = {
        "image_base64": base64_str
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Error: {response.text}"
    
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] == expected_digit


def test_predict_endpoint_missing_field():
    response = client.post("/predict", json={"wrong_field": "some data"})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_predict_endpoint_empty_field():
    response = client.post("/predict", json={"image_base64": ""})
    assert response.status_code == 400
    assert response.json()["detail"] == "image_base64 field is empty."


def test_predict_endpoint_invalid_base64():
    response = client.post("/predict", json={"image_base64": "invalid_base64"})
    assert response.status_code == 422
    assert "detail" in response.json()


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome to the MNIST Digit Classifier API" in response.json()["message"]


def test_status_endpoint():
    response = client.get("/status")
    assert response.status_code == 200
    assert "Status" in response.json()
