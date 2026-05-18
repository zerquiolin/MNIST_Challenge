import base64
from pathlib import Path

import pytest

from src.utils.preprocessing import image_to_base64, preprocess_handwritten_image
from PIL import Image
import numpy as np


def test_image_to_base64_success(tmp_path: Path):
    test_file = tmp_path / "test_image.png"
    test_content = b"fake image content"
    test_file.write_bytes(test_content)

    base64_str = image_to_base64(test_file)
    expected = base64.b64encode(test_content).decode("utf-8")
    assert base64_str == expected


def test_image_to_base64_file_not_found():
    with pytest.raises(FileNotFoundError):
        image_to_base64("non_existent_file.png")


def test_image_to_base64_not_a_file(tmp_path: Path):
    directory = tmp_path / "test_dir"
    directory.mkdir()

    with pytest.raises(ValueError, match="Path is not a file"):
        image_to_base64(directory)


def test_preprocess_handwritten_image():
    # Create a dummy image representing a digit with white background and black drawing
    # or just noise, and check that it outputs a square image with a black background
    img = Image.new("RGB", (100, 50), color="white")

    # Add a small black square in the middle to simulate a digit
    for i in range(40, 60):
        for j in range(20, 30):
            img.putpixel((i, j), (0, 0, 0))

    processed_img = preprocess_handwritten_image(img)

    # Must be square
    w, h = processed_img.size
    assert w == h

    # Must be grayscale
    assert processed_img.mode == "L"

    # Background should be black (0) after preprocessing, digit should be white (255)
    # Check borders are mostly 0
    np_img = np.array(processed_img)
    # Borders should be mostly black
    assert np.mean(np_img[0, :]) < 10  # top
    assert np.mean(np_img[-1, :]) < 10  # bottom
    assert np.mean(np_img[:, 0]) < 10  # left
    assert np.mean(np_img[:, -1]) < 10  # right
    # There should be some white-ish pixels somewhere, presumably the digit
    assert np.max(np_img) > 200
    # There should be some white-ish pixels in the top 99th percentile, to ensure the digit is preserved
    assert np.percentile(np_img, 99) > 180
