from typing import cast

import torch

from app.core.config import settings
from src.models.mnist_cnn import MNISTCNN
from src.utils.device import get_device
from src.utils.model_loader import load_model


class ModelLoader:
    """Lazy singleton for the trained MNIST model and its compute device."""

    _model: MNISTCNN | None = None
    _device: torch.device | None = None

    @classmethod
    def get_model(cls) -> MNISTCNN:
        """Return the loaded model, loading it on first use."""
        if cls._model is None:
            try:
                cls._device = get_device()
                cls._model = cast(
                    MNISTCNN,
                    load_model(
                        model=MNISTCNN,
                        model_path=settings.model_path,
                        device=cls._device,
                    ),
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load model from path: {e}") from e

        return cls._model

    @classmethod
    def get_device(cls) -> torch.device:
        """Return the model device, initializing the model when needed."""
        if cls._device is None:
            cls.get_model()
        if cls._device is None:
            raise RuntimeError("Model device was not initialized.")
        return cls._device


def predict_digit(tensor_img: torch.Tensor) -> tuple[int, float]:
    """Run inference for a preprocessed image tensor and return class/confidence."""
    model = ModelLoader.get_model()
    device = ModelLoader.get_device()

    tensor_img = tensor_img.to(device)

    with torch.inference_mode():
        logits = model(tensor_img)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)

        return int(prediction.item()), float(confidence.item())
