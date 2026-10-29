"""FastAPI application entry point."""

import logging
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.comparison_routes import router as comparison_router
from app.feedback_routes import router as feedback_router
from app.routes import router as vision_router
from app.settings import get_settings

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    description=(
        "Compare direct VLM image understanding with a local OCR/object-detection "
        "pipeline followed by a text-only LLM."
    ),
    version="0.1.0",
)

origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


app.include_router(vision_router)
app.include_router(comparison_router)
app.include_router(feedback_router)


@app.middleware("http")
async def log_request_lifecycle(request: Request, call_next):
    """Log request start, completion, duration, and unexpected failures."""
    started = perf_counter()
    request_id = request.headers.get("x-request-id", "-")
    logger.info(
        "request_started method=%s path=%s request_id=%s",
        request.method,
        request.url.path,
        request_id,
    )
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "request_failed method=%s path=%s request_id=%s",
            request.method,
            request.url.path,
            request_id,
        )
        raise
    logger.info(
        "request_completed method=%s path=%s status_code=%s duration_seconds=%.4f request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        perf_counter() - started,
        request_id,
    )
    return response


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    """Return process health without loading large vision models."""
    return {"status": "ok", "environment": settings.app_env}


@app.get("/", tags=["system"])
async def root() -> dict[str, str]:
    """Provide a short service description."""
    return {"name": settings.app_name, "docs": "/docs"}
