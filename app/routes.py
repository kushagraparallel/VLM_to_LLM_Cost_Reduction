"""HTTP routes for image structure extraction."""

from time import perf_counter

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.schemas import ImageStructure
from app.services.image_processing import InvalidImageError, decode_image
from app.services.paddle_ocr import extract_text
from app.services.representation import structure_to_dict
from app.services.yolo_detector import detect_objects
from app.settings import get_settings

router = APIRouter(prefix="/api/v1/vision", tags=["vision"])


@router.post("/structure", response_model=ImageStructure)
async def extract_image_structure(file: UploadFile = File(...)) -> ImageStructure:
    """Extract OCR text and object detections without calling a cloud model."""
    settings = get_settings()
    max_upload_bytes = settings.max_upload_size_mb * 1024 * 1024
    payload = await file.read(max_upload_bytes + 1)
    if len(payload) > max_upload_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds the configured upload limit.")

    try:
        image, image_array = await run_in_threadpool(decode_image, payload)
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    timings: dict[str, float] = {}
    try:
        started = perf_counter()
        ocr_regions = await run_in_threadpool(extract_text, image_array)
        timings["ocr_seconds"] = round(perf_counter() - started, 6)

        started = perf_counter()
        detections = await run_in_threadpool(detect_objects, image)
        timings["detection_seconds"] = round(perf_counter() - started, 6)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Local vision processing failed: {type(exc).__name__}: {exc}",
        ) from exc

    timings["total_seconds"] = round(sum(timings.values()), 6)
    return structure_to_dict(
        image_size=image.size,
        ocr_regions=ocr_regions,
        detections=detections,
        processing=timings,
    )
