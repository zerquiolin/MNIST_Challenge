import base64
from io import BytesIO

import pytest
from PIL import Image

from app.lib.constants import MAX_BASE64_LENGTH
from app.services.validation import (
    decode_base64_image,
    load_validated_image,
    validate_base64_image,
)


def create_base64_image(format="PNG", size=(10, 10), color="white"):
    img = Image.new("RGB", size, color=color)
    buffer = BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_decode_base64_image_valid():
    b64_str = create_base64_image()
    bytes_data = decode_base64_image(b64_str)
    assert isinstance(bytes_data, bytes)


def test_decode_base64_image_empty():
    with pytest.raises(ValueError, match="image_base64 field is empty"):
        decode_base64_image("")


def test_decode_base64_image_too_long():
    long_b64 = "a" * (MAX_BASE64_LENGTH + 1)
    with pytest.raises(ValueError, match="Base64 payload exceeds"):
        decode_base64_image(long_b64)


def test_decode_base64_image_invalid_mime():
    b64_str = create_base64_image()
    data_url = f"data:image/svg+xml;base64,{b64_str}"
    with pytest.raises(ValueError, match="Unsupported image MIME type"):
        decode_base64_image(data_url)


def test_decode_base64_image_whitespace():
    b64_str = create_base64_image()
    invalid_b64 = b64_str[:5] + " " + b64_str[5:]
    with pytest.raises(ValueError, match="Base64 payload must not contain whitespace"):
        decode_base64_image(invalid_b64)


def test_decode_base64_image_invalid_encoding():
    with pytest.raises(ValueError, match="Invalid base64 image encoding"):
        decode_base64_image("invalid_base64_string!!!")


def test_load_validated_image_valid():
    b64_str = create_base64_image(size=(28, 28))
    img_bytes = base64.b64decode(b64_str)
    img = load_validated_image(img_bytes)
    assert isinstance(img, Image.Image)


def test_load_validated_image_too_small():
    b64_str = create_base64_image(size=(5, 5))
    img_bytes = base64.b64decode(b64_str)
    with pytest.raises(ValueError, match="Image dimensions too small"):
        load_validated_image(img_bytes)


def test_load_validated_image_unsupported_format():
    img = Image.new("RGB", (28, 28), color="white")
    buffer = BytesIO()
    img.save(buffer, format="TIFF")
    img_bytes = buffer.getvalue()

    with pytest.raises(ValueError, match="Unsupported image format"):
        load_validated_image(img_bytes)


def test_validate_base64_image_success():
    b64_str = create_base64_image(size=(28, 28))
    ok, msg = validate_base64_image(b64_str)
    assert ok is True
    assert msg == "Valid base64 image."


def test_validate_base64_image_failure():
    ok, msg = validate_base64_image("invalid base 64")
    assert ok is False
    assert msg == "Base64 payload must not contain whitespace."
