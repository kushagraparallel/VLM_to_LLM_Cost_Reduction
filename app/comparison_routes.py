"""HTTP endpoint for side-by-side model comparison."""

import logging
from io import BytesIO
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.services.comparison import run_comparison
from app.services.experiment_store import save_comparison
from app.services.image_processing import InvalidImageError, decode_image
from app.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["comparison"])


@router.post("/compare")
async def compare_image_models(
    prompt: str = Form(..., min_length=1, max_length=8000),
    file: UploadFile = File(...),
    vlm_provider: str | None = Form(default=None),
    llm_provider: str | None = Form(default=None),
) -> dict:
    """Compare a direct VLM response with OCR/detection plus a text LLM."""
    request_started = perf_counter()
    logger.info("comparison_started vlm_provider=%s llm_provider=%s", vlm_provider or "default", llm_provider or "default")
    settings = get_settings()
    user_prompt = prompt
    selected_vlm_provider = (vlm_provider or settings.vlm_provider).lower()
    selected_llm_provider = (llm_provider or settings.llm_provider).lower()
    allowed_providers = {"gemini", "groq", "bedrock"}
    if selected_vlm_provider not in allowed_providers or selected_llm_provider not in allowed_providers:
        logger.warning("comparison_rejected reason=unsupported_provider vlm_provider=%s llm_provider=%s", selected_vlm_provider, selected_llm_provider)
        raise HTTPException(status_code=422, detail="Provider must be gemini, groq, or bedrock.")
    if not user_prompt.strip():
        logger.warning("comparison_rejected reason=blank_prompt")
        raise HTTPException(status_code=422, detail="Prompt must not be blank.")

    max_upload_bytes = settings.max_upload_size_mb * 1024 * 1024
    payload = await file.read(max_upload_bytes + 1)
    logger.debug("comparison_upload_read bytes=%s", len(payload))
    if len(payload) > max_upload_bytes:
        logger.warning("comparison_rejected reason=upload_too_large bytes=%s limit_bytes=%s", len(payload), max_upload_bytes)
        raise HTTPException(status_code=413, detail="Image exceeds the configured upload limit.")

    try:
        image, image_array = await run_in_threadpool(decode_image, payload)
    except InvalidImageError as exc:
        logger.warning("comparison_image_validation_failed error=%s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Normalize supported image formats so each provider receives consistent PNG bytes.
    def normalize_image() -> bytes:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    normalized_image_bytes = await run_in_threadpool(normalize_image)
    logger.info("comparison_image_normalized bytes=%s", len(normalized_image_bytes))

    run_id = str(uuid4())
    try:
        result = await run_comparison(
            run_id=run_id,
            settings=settings,
            image_bytes=normalized_image_bytes,
            image=image,
            image_array=image_array,
            mime_type="image/png",
            user_prompt=user_prompt,
            vlm_provider_override=selected_vlm_provider,
            llm_provider_override=selected_llm_provider,
        )
        await run_in_threadpool(save_comparison, settings.results_db_path, result)
        logger.info("comparison_completed run_id=%s elapsed_seconds=%.4f", result["run_id"], perf_counter() - request_started)
        return result
    except FileNotFoundError as exc:
        logger.exception("comparison_artifact_missing")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        logger.warning("comparison_configuration_or_input_error error=%s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("comparison_failed")
        raise HTTPException(
            status_code=503,
            detail=f"Comparison failed: {type(exc).__name__}: {exc}",
        ) from exc
