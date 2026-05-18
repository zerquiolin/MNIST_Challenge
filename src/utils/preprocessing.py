from os import PathLike

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps

from src.data.augmentation import test_transforms


def preprocess_handwritten_image_with_transformation(
    image_path: str | PathLike[str],
) -> torch.Tensor:
    """Load a handwritten digit image from disk and return an inference tensor."""
    img = Image.open(image_path).convert("L")
    transform = test_transforms()
    return transform(preprocess_handwritten_image(img)).unsqueeze(0)


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
