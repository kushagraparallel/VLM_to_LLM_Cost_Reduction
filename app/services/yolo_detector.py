"""YOLO26 Nano object detection through the OpenVINO inference backend."""

from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image

from app.settings import get_settings


@lru_cache(maxsize=1)
def _get_detector() -> Any:
    """Load an exported OpenVINO model on first use."""
    from ultralytics import YOLO

    settings = get_settings()
    model_path = Path(settings.yolo_model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"OpenVINO model not found at '{model_path}'. Export YOLO26 Nano first; "
            "see docs/SETUP.md."
        )
    return YOLO(str(model_path), task="detect")


def detect_objects(image: Image.Image) -> list[dict[str, Any]]:
    """Detect objects and return pixel and normalized bounding boxes."""
    settings = get_settings()
    model = _get_detector()
    result_list = model.predict(
        source=image,
        imgsz=settings.yolo_image_size,
        conf=settings.yolo_confidence,
        device=settings.yolo_device,
        verbose=False,
    )
    width, height = image.size
    detections: list[dict[str, Any]] = []

    for result in result_list:
        boxes = result.boxes
        if boxes is None:
            continue
        xyxy_values = boxes.xyxy.cpu().tolist()
        class_values = boxes.cls.cpu().tolist()
        confidence_values = boxes.conf.cpu().tolist()
        for coordinates, class_value, confidence in zip(
            xyxy_values, class_values, confidence_values, strict=False
        ):
            x1, y1, x2, y2 = [float(value) for value in coordinates]
            class_id = int(class_value)
            label = str(result.names.get(class_id, class_id))
            detections.append(
                {
                    "label": label,
                    "class_id": class_id,
                    "confidence": float(confidence),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_normalized": [
                        round(x1 / max(width, 1), 6),
                        round(y1 / max(height, 1), 6),
                        round(x2 / max(width, 1), 6),
                        round(y2 / max(height, 1), 6),
                    ],
                }
            )

    return detections
