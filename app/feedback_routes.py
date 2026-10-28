"""Routes for experiment feedback."""

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.services.experiment_store import save_feedback
from app.settings import get_settings

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
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": "saved", "run_id": request.run_id}
