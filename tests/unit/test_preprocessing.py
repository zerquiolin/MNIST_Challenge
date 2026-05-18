import base64
from pathlib import Path

import pytest

from src.utils.preprocessing import image_to_base64


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
