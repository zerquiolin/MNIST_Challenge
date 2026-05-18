from unittest.mock import MagicMock, patch

import pytest
import torch

from app.services.inference import predict_digit


@pytest.fixture
def mock_model_loader():
    with (
        patch("app.services.inference.ModelLoader.get_model") as mock_get_model,
        patch("app.services.inference.ModelLoader.get_device") as mock_get_device,
    ):
        # Mock device
        mock_device = torch.device("cpu")
        mock_get_device.return_value = mock_device

        # Mock model
        mock_model = MagicMock()

        # Simulate an output tensor of shape (1, 10) for 10 classes
        # Set class index '3' to have the highest logit
        dummy_logits = torch.zeros((1, 10))
        dummy_logits[0, 3] = 10.0

        mock_model.return_value = dummy_logits
        mock_get_model.return_value = mock_model

        yield mock_model, mock_device


def test_predict_digit(mock_model_loader):
    # Create a dummy tensor image
    dummy_tensor = torch.rand(1, 1, 28, 28)

    # Run the prediction
    prediction, confidence = predict_digit(dummy_tensor)

    # Check that it predicted class '3' based on our mock
    assert prediction == 3
    # Check that the confidence is high (close to 1.0 because of softmax(10.0))
    assert confidence > 0.99

    mock_model, mock_device = mock_model_loader
    mock_model.assert_called_once()
