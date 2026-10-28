"""Image decoding helpers shared by vision services."""

from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError


class InvalidImageError(ValueError):
    """Raised when an uploaded payload is not a supported image."""


def decode_image(payload: bytes) -> tuple[Image.Image, np.ndarray]:
    """Decode image bytes into a normalized RGB PIL image and NumPy array."""
    if not payload:
        raise InvalidImageError("The uploaded file is empty.")

    try:
        with Image.open(BytesIO(payload)) as source:
            source.verify()
        with Image.open(BytesIO(payload)) as source:
            image = source.convert("RGB")
            image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("The upload is not a valid image file.") from exc

    return image, np.asarray(image)
