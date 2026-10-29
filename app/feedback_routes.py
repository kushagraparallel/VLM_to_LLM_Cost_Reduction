"""Routes for experiment feedback."""

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.services.experiment_store import save_feedback
from app.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["feedback"])


class FeedbackRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=64)
    preferred_path: Literal["direct_vlm", "structured_pipeline", "tie"]
    direct_vlm_rating: int | None = Field(default=None, ge=1, le=5)
    structured_pipeline_rating: int | None = Field(default=None, ge=1, le=5)
    comment: str | None = Field(default=None, max_length=2000)


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest) -> dict[str, str]:
    """Save the preferred answer and optional ratings for a completed run."""
    logger.info("feedback_submission_started run_id=%s preferred_path=%s", request.run_id, request.preferred_path)
    settings = get_settings()
    try:
        await run_in_threadpool(
            save_feedback,
            settings.results_db_path,
            run_id=request.run_id,
            preferred_path=request.preferred_path,
            direct_vlm_rating=request.direct_vlm_rating,
            structured_pipeline_rating=request.structured_pipeline_rating,
            comment=request.comment,
        )
    except KeyError as exc:
        logger.warning("feedback_submission_rejected run_id=%s reason=run_not_found", request.run_id)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    logger.info("feedback_submission_completed run_id=%s", request.run_id)
    return {"status": "saved", "run_id": request.run_id}
