"""Build a stable, provider-neutral representation of visual evidence."""

import json
from typing import Any

from app.schemas import ImageStructure

STRUCTURED_IMAGE_SYSTEM_INSTRUCTION = (
    "You are answering a question about an image using a structured visual-evidence "
    "representation created by OCR and object detection. Treat the supplied JSON as "
    "the only visual evidence available to you; it is an internal representation of "
    "the image, not the original image itself. OCR text may be incomplete or incorrect, "
    "and object detections may be missing or mistaken. Do not invent details that are "
    "not supported by the representation. Clearly state uncertainty when the evidence "
    "is insufficient. Follow the user's request exactly."
 )


def serialize_image_structure(structure: ImageStructure) -> str:
    """Serialize a structure consistently for a text-only model."""
    return json.dumps(structure.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))


def build_structured_user_message(user_prompt: str, structure: ImageStructure) -> str:
    """Preserve the user prompt verbatim while appending structured evidence."""
    evidence = serialize_image_structure(structure)
    return f"User request (verbatim):\n{user_prompt}\n\nStructured image evidence (JSON):\n{evidence}"


def structure_to_dict(
    image_size: tuple[int, int],
    ocr_regions: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    processing: dict[str, float],
) -> ImageStructure:
    """Create a validated schema object from normalized service outputs."""
    width, height = image_size
    return ImageStructure(
        image={"width": width, "height": height, "color_mode": "RGB"},
        ocr_regions=ocr_regions,
        detections=detections,
        processing=processing,
    )
