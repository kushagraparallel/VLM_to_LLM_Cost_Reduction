"""PaddleOCR adapter with lazy model initialization."""

import logging
from functools import lru_cache
from time import perf_counter
from typing import Any

import numpy as np

from app.settings import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_ocr_engine() -> Any:
    """Create the OCR engine only when an image is processed."""
    from paddleocr import PaddleOCR

    logger.info("ocr_engine_initialization_started")
    settings = get_settings()
    engine = PaddleOCR(
        lang=settings.paddleocr_language,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )
    logger.info("ocr_engine_initialization_completed language=%s", settings.paddleocr_language)
    return engine


def _polygon_bounds(points: Any) -> list[float] | None:
    try:
        coordinates = np.asarray(points, dtype=float).reshape(-1, 2)
        if coordinates.size == 0:
            return None
        return [
            float(coordinates[:, 0].min()),
            float(coordinates[:, 1].min()),
            float(coordinates[:, 0].max()),
            float(coordinates[:, 1].max()),
        ]
    except (TypeError, ValueError):
        return None


def _extract_result_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result.get("res", result)
    json_value = getattr(result, "json", None)
    if callable(json_value):
        json_value = json_value()
    if isinstance(json_value, dict):
        return json_value.get("res", json_value)
    return {}


def extract_text(image_array: np.ndarray) -> list[dict[str, Any]]:
    """Return recognized text, confidence, polygon, and bounding box per region."""
    started = perf_counter()
    logger.debug("ocr_inference_started image_shape=%s", getattr(image_array, "shape", None))
    engine = _get_ocr_engine()
    results = engine.predict(input=image_array)
    regions: list[dict[str, Any]] = []

    for result in results or []:
        payload = _extract_result_payload(result)
        texts = payload.get("rec_texts") or []
        scores = payload.get("rec_scores") or []
        polygons = payload.get("rec_polys") or payload.get("dt_polys") or []

        for index, text in enumerate(texts):
            cleaned_text = str(text).strip()
            if not cleaned_text:
                continue
            score = None
            if index < len(scores):
                try:
                    score = float(scores[index])
                except (TypeError, ValueError):
                    score = None
            polygon: list[list[float]] = []
            bounds = None
            if index < len(polygons):
                try:
                    polygon = np.asarray(polygons[index], dtype=float).reshape(-1, 2).tolist()
                    bounds = _polygon_bounds(polygon)
                except (TypeError, ValueError):
                    polygon = []
            regions.append(
                {
                    "text": cleaned_text,
                    "confidence": score,
                    "polygon": polygon,
                    "bbox_xyxy": bounds,
                }
            )

    logger.info("ocr_inference_completed regions=%s duration_seconds=%.4f", len(regions), perf_counter() - started)
    return regions
