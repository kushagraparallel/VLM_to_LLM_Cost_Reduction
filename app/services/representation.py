"""Build a stable, provider-neutral representation of visual evidence."""

import json
from typing import Any

from app.schemas import ImageStructure

STRUCTURED_IMAGE_SYSTEM_INSTRUCTION = (
    "You are answering a question about an image using a structured visual-evidence "
    "representation created by OCR and object detection. Treat the supplied JSON as "
    "the only visual evidence available to you; it is an internal representation of "
    "the image, not the original image itself. OCR text may be incomplete or incorrect, "
    "and object detections may be missing or mistaken. Spatial relationships are "
    "geometric estimates derived from bounding boxes, not proof of interaction or intent. "
    "Do not invent details that are not supported by the representation. Clearly state "
    "uncertainty when the evidence is insufficient. Follow the user's request exactly."
)


def serialize_image_structure(structure: ImageStructure) -> str:
    """Serialize a structure consistently for a text-only model."""
    return json.dumps(
        structure.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def build_structured_user_message(user_prompt: str, structure: ImageStructure) -> str:
    """Preserve the user prompt verbatim while appending structured evidence."""
    evidence = serialize_image_structure(structure)
    return (
        f"User request (verbatim):\n{user_prompt}\n\n"
        f"Structured image evidence (JSON):\n{evidence}"
    )


def _intersection_over_union(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def build_spatial_relationships(
    detections: list[dict[str, Any]],
    *,
    max_relationships: int = 150,
) -> list[dict[str, Any]]:
    """Derive bounded geometric relationships from detection bounding boxes."""
    relationships: list[dict[str, Any]] = []
    for first_index, first in enumerate(detections):
        first_box = first.get("bbox_xyxy")
        if not first_box or len(first_box) != 4:
            continue
        first_center_x = (float(first_box[0]) + float(first_box[2])) / 2
        first_center_y = (float(first_box[1]) + float(first_box[3])) / 2

        for second_index in range(first_index + 1, len(detections)):
            second = detections[second_index]
            second_box = second.get("bbox_xyxy")
            if not second_box or len(second_box) != 4:
                continue
            overlap = _intersection_over_union(first_box, second_box)
            second_center_x = (float(second_box[0]) + float(second_box[2])) / 2
            second_center_y = (float(second_box[1]) + float(second_box[3])) / 2

            if overlap >= 0.15:
                relation = "overlaps"
            elif abs(second_center_x - first_center_x) >= abs(second_center_y - first_center_y):
                relation = "left_of" if first_center_x < second_center_x else "right_of"
            else:
                relation = "above" if first_center_y < second_center_y else "below"

            relationships.append(
                {
                    "subject_index": first_index,
                    "subject_label": first.get("label", "unknown"),
                    "relation": relation,
                    "object_index": second_index,
                    "object_label": second.get("label", "unknown"),
                    "overlap_iou": round(overlap, 4),
                }
            )
            if len(relationships) >= max_relationships:
                return relationships
    return relationships


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
        relationships=build_spatial_relationships(detections),
        processing=processing,
    )
