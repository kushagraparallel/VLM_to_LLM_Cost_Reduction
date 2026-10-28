import json

from app.schemas import ImageStructure
from app.services.representation import (
    STRUCTURED_IMAGE_SYSTEM_INSTRUCTION,
    build_spatial_relationships,
    build_structured_user_message,
    serialize_image_structure,
    structure_to_dict,
)


def test_user_prompt_is_preserved_verbatim_before_evidence() -> None:
    prompt = "Summarize the visible receipt and list its total."
    structure = ImageStructure(
        image={"width": 800, "height": 600, "color_mode": "RGB"},
        ocr_regions=[{"text": "TOTAL $12.50", "confidence": 0.98}],
        detections=[],
    )

    message = build_structured_user_message(prompt, structure)

    assert message.startswith("User request (verbatim):\n" + prompt)
    assert "TOTAL $12.50" in message


def test_serialization_is_valid_json_and_keeps_unicode() -> None:
    structure = ImageStructure(
        image={"width": 100, "height": 80, "color_mode": "RGB"},
        ocr_regions=[{"text": "café", "confidence": 0.9}],
        detections=[],
    )

    serialized = serialize_image_structure(structure)

    assert json.loads(serialized)["ocr_regions"][0]["text"] == "café"
    assert "\\u00e9" not in serialized


def test_structure_builder_records_dimensions_and_timings() -> None:
    structure = structure_to_dict(
        image_size=(640, 480),
        ocr_regions=[{"text": "hello", "confidence": 0.9, "polygon": [], "bbox_xyxy": None}],
        detections=[],
        processing={"ocr_seconds": 0.12},
    )

    assert structure.image["width"] == 640
    assert structure.image["height"] == 480
    assert structure.processing["ocr_seconds"] == 0.12


def test_structured_system_instruction_explains_evidence_limitations() -> None:
    assert "structured visual-evidence" in STRUCTURED_IMAGE_SYSTEM_INSTRUCTION
    assert "Do not invent details" in STRUCTURED_IMAGE_SYSTEM_INSTRUCTION


def test_spatial_relationships_are_derived_from_bounding_boxes() -> None:
    detections = [
        {"label": "person", "bbox_xyxy": [0, 10, 20, 50]},
        {"label": "chair", "bbox_xyxy": [40, 10, 60, 50]},
    ]

    relationships = build_spatial_relationships(detections)

    assert relationships[0]["subject_label"] == "person"
    assert relationships[0]["relation"] == "left_of"
    assert relationships[0]["object_label"] == "chair"
