import base64
from os import PathLike
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps

from src.data.augmentation import test_transforms


def image_to_base64(image_path: str | Path) -> str:
    """Read an image file and return its base64 string."""
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if not image_path.is_file():
        raise ValueError(f"Path is not a file: {image_path}")

    with image_path.open("rb") as image_file:
        encoded_bytes = base64.b64encode(image_file.read())

    return encoded_bytes.decode("utf-8")


def preprocess_handwritten_image_with_transformation(
    image_path: str | PathLike[str], unsqueeze: bool = True
) -> torch.Tensor:
    """Load a handwritten digit image from disk and return an inference tensor."""
    img = Image.open(image_path).convert("L")
    transform = test_transforms()
    if unsqueeze:
        return transform(preprocess_handwritten_image(img)).unsqueeze(0)  # type: ignore[no-any-return]

    return transform(preprocess_handwritten_image(img))  # type: ignore[no-any-return]


def preprocess_handwritten_image(img: Image.Image) -> Image.Image:
    """Normalize a handwritten digit image into MNIST-like white-on-black pixels."""
    img = img.convert("L")
    img = ImageOps.autocontrast(img)

    np_img = np.array(img)
    borders = np.concatenate([np_img[0, :], np_img[-1, :], np_img[:, 0], np_img[:, -1]])
    if np.mean(borders) > 128:
        img = ImageOps.invert(img)

    np_img = np.array(img)
    np_img = np.where(np_img > 128, 255, 0).astype(np.uint8)
    img = Image.fromarray(np_img)

    img = img.filter(ImageFilter.MaxFilter(15))

    bbox = img.getbbox()
    if bbox is None:
        raise ValueError("Image does not contain a visible digit.")

    img = img.crop(bbox)

    w, h = img.size
    max_dim = max(w, h)
    pad_w = (max_dim - w) // 2 + int(max_dim * 0.1)
    pad_h = (max_dim - h) // 2 + int(max_dim * 0.1)
    return ImageOps.expand(img, border=(pad_w, pad_h, pad_w, pad_h), fill=0)
