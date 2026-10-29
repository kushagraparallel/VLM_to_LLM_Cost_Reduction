"""HTTP routes for image structure extraction."""

import logging
from time import perf_counter

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.schemas import ImageStructure
from app.services.image_processing import InvalidImageError, decode_image
from app.services.paddle_ocr import extract_text
from app.services.representation import structure_to_dict
from app.services.yolo_detector import detect_objects
from app.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/vision", tags=["vision"])


@router.post("/structure", response_model=ImageStructure)
async def extract_image_structure(file: UploadFile = File(...)) -> ImageStructure:
    """Extract OCR text and object detections without calling a cloud model."""
    request_started = perf_counter()
    logger.info("vision_structure_started")
    settings = get_settings()
    max_upload_bytes = settings.max_upload_size_mb * 1024 * 1024
    payload = await file.read(max_upload_bytes + 1)
    logger.debug("vision_upload_read bytes=%s", len(payload))
    if len(payload) > max_upload_bytes:
        logger.warning("vision_upload_rejected reason=upload_too_large bytes=%s limit_bytes=%s", len(payload), max_upload_bytes)
        raise HTTPException(status_code=413, detail="Image exceeds the configured upload limit.")

    try:
        image, image_array = await run_in_threadpool(decode_image, payload)
    except InvalidImageError as exc:
        logger.warning("vision_image_validation_failed error=%s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    timings: dict[str, float] = {}
    try:
        started = perf_counter()
        ocr_regions = await run_in_threadpool(extract_text, image_array)
        timings["ocr_seconds"] = round(perf_counter() - started, 6)
        logger.info("vision_ocr_completed regions=%s duration_seconds=%.4f", len(ocr_regions), timings["ocr_seconds"])

        started = perf_counter()
        detections = await run_in_threadpool(detect_objects, image)
        timings["detection_seconds"] = round(perf_counter() - started, 6)
        logger.info("vision_detection_completed detections=%s duration_seconds=%.4f", len(detections), timings["detection_seconds"])
    except FileNotFoundError as exc:
        logger.exception("vision_model_or_artifact_missing")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("vision_processing_failed")
        raise HTTPException(
            status_code=503,
            detail=f"Local vision processing failed: {type(exc).__name__}: {exc}",
        ) from exc

    timings["total_seconds"] = round(sum(timings.values()), 6)
    structure = await run_in_threadpool(
        structure_to_dict,
        image_size=image.size,
        ocr_regions=ocr_regions,
        detections=detections,
        processing=timings,
    )
    logger.info("vision_structure_completed total_seconds=%.4f", perf_counter() - request_started)
    return structure
