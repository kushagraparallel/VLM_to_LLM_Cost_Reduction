"""Image decoding helpers shared by vision services."""

import numpy as np
from io import BytesIO
from PIL import Image, UnidentifiedImageError


MAX_IMAGE_PIXELS = 40_000_000


class InvalidImageError(ValueError):
    """Raised when an uploaded payload is not a supported image."""


def decode_image(payload: bytes) -> tuple[Image.Image, np.ndarray]:
    """Decode image bytes into a normalized RGB image with a pixel-count limit."""
    if not payload:
        raise InvalidImageError("The uploaded file is empty.")

    try:
        with Image.open(BytesIO(payload)) as source:
            width, height = source.size
            if width * height > MAX_IMAGE_PIXELS:
                raise InvalidImageError(
                    f"Image dimensions exceed the {MAX_IMAGE_PIXELS:,}-pixel limit."
                )
            source.verify()
        with Image.open(BytesIO(payload)) as source:
            image = source.convert("RGB")
            image.load()
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
        raise InvalidImageError("The upload is not a valid or supported image.") from exc

    return image, np.asarray(image)
