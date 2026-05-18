import re
from typing import Final

MAX_BASE64_LENGTH: Final[int] = 5_000_000
MAX_IMAGE_BYTES: Final[int] = 4 * 1024 * 1024
MAX_IMAGE_PIXELS: Final[int] = 4_000_000
MAX_IMAGE_DIMENSION: Final[int] = 4096
MIN_IMAGE_DIMENSION: Final[int] = 10

ALLOWED_IMAGE_FORMATS: Final[frozenset[str]] = frozenset({"PNG", "JPEG", "WEBP", "BMP"})
ALLOWED_DATA_URL_MIME_TYPES: Final[frozenset[str]] = frozenset(
    {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp"}
)

DATA_URL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^data:(?P<mime>[-\w.]+/[-\w+.]+);base64,(?P<data>.+)$",
    re.IGNORECASE | re.DOTALL,
)
BASE64_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")
