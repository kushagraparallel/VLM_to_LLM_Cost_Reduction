"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.settings import get_settings

settings = get_settings()

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


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    """Return process health without loading large vision models."""
    return {"status": "ok", "environment": settings.app_env}


@app.get("/", tags=["system"])
async def root() -> dict[str, str]:
    """Provide a short service description."""
    return {"name": settings.app_name, "docs": "/docs"}
