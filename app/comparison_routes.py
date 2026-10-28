"""HTTP endpoint for side-by-side model comparison."""

from io import BytesIO

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.services.comparison import run_comparison
from app.services.image_processing import InvalidImageError, decode_image
from app.settings import get_settings

router = APIRouter(prefix="/api/v1", tags=["comparison"])


@router.post("/compare")
async def compare_image_models(
    prompt: str = Form(..., min_length=1, max_length=8000),
    file: UploadFile = File(...),
) -> dict:
    """Compare a direct VLM response with OCR/detection plus a text LLM."""
    settings = get_settings()
    user_prompt = prompt
    if not user_prompt.strip():
        raise HTTPException(status_code=422, detail="Prompt must not be blank.")

    payload = await file.read()
    if len(payload) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds the configured upload limit.")

    try:
        image, image_array = await run_in_threadpool(decode_image, payload)
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Normalize supported image formats so each provider receives consistent PNG bytes.
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    normalized_image_bytes = buffer.getvalue()

    try:
        return await run_comparison(
            settings=settings,
            image_bytes=normalized_image_bytes,
            image=image,
            image_array=image_array,
            mime_type="image/png",
            user_prompt=user_prompt,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Comparison failed: {type(exc).__name__}: {exc}",
        ) from exc
