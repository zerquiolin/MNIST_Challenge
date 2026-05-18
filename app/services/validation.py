import base64
import binascii
from io import BytesIO

import torch
from PIL import Image, UnidentifiedImageError

from app.lib.constants import (
    ALLOWED_DATA_URL_MIME_TYPES,
    ALLOWED_IMAGE_FORMATS,
    BASE64_PATTERN,
    DATA_URL_PATTERN,
    MAX_BASE64_LENGTH,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_PIXELS,
    MIN_IMAGE_DIMENSION,
)
from src.data.augmentation import test_transforms
from src.utils.preprocessing import preprocess_handwritten_image


def _split_base64_payload(image_base64: str) -> tuple[str, str | None]:
    """Return the base64 payload and optional data URL MIME type."""
    value = image_base64.strip()
    match = DATA_URL_PATTERN.match(value)
    if match:
        return match.group("data").strip(), match.group("mime").lower()
    return value, None


def decode_base64_image(image_base64: str) -> bytes:
    """Decode a strict base64 image string or data URL into image bytes."""
    if not image_base64 or not image_base64.strip():
        raise ValueError("image_base64 field is empty.")

    if len(image_base64) > MAX_BASE64_LENGTH:
        raise ValueError(f"Base64 payload exceeds {MAX_BASE64_LENGTH} characters.")

    payload, mime_type = _split_base64_payload(image_base64)
    if mime_type and mime_type not in ALLOWED_DATA_URL_MIME_TYPES:
        raise ValueError(f"Unsupported image MIME type: {mime_type}.")

    if any(char.isspace() for char in payload):
        raise ValueError("Base64 payload must not contain whitespace.")

    if len(payload) % 4 != 0 or not BASE64_PATTERN.fullmatch(payload):
        raise ValueError("Invalid base64 image encoding.")

    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image encoding.") from exc

    if not image_bytes:
        raise ValueError("Decoded image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(f"Decoded image exceeds {MAX_IMAGE_BYTES} bytes.")

    return image_bytes


def load_validated_image(image_bytes: bytes) -> Image.Image:
    """Load image bytes after validating format, dimensions, and pixel count."""
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()

        with Image.open(BytesIO(image_bytes)) as image:
            if image.format not in ALLOWED_IMAGE_FORMATS:
                raise ValueError(
                    "Unsupported image format. Allowed formats: BMP, JPEG, PNG, WEBP."
                )

            width, height = image.size
            if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
                raise ValueError(
                    f"Image dimensions too small: {width}x{height}. "
                    f"Minimum required is {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}."
                )
            if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                raise ValueError(
                    f"Image dimensions too large: {width}x{height}. "
                    f"Maximum allowed is {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}."
                )
            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError(f"Image exceeds {MAX_IMAGE_PIXELS} pixels.")

            image.load()
            return image.convert("L")
    except Image.DecompressionBombError as exc:
        raise ValueError("Image is too large to process safely.") from exc
    except UnidentifiedImageError as exc:
        raise ValueError("Decoded bytes are not a valid image.") from exc
    except OSError as exc:
        raise ValueError("Image file is corrupted or unsupported.") from exc


def validate_image_bytes(image_bytes: bytes) -> None:
    """Validate image bytes without returning the decoded PIL image."""
    load_validated_image(image_bytes)


def validate_base64_image(image_base64: str) -> tuple[bool, str]:
    """Validate that a request value is a strict, safely sized base64 image."""
    try:
        image_bytes = decode_base64_image(image_base64)
        validate_image_bytes(image_bytes)
    except ValueError as exc:
        return False, str(exc)
    return True, "Valid base64 image."


def process_base64_image(image_base64: str) -> torch.Tensor:
    """Decode, validate, preprocess, and batch a base64 image for inference."""
    image_bytes = decode_base64_image(image_base64)
    image = load_validated_image(image_bytes)
    processed_image = preprocess_handwritten_image(image)
    tensor = test_transforms()(processed_image)
    return tensor.unsqueeze(0)
