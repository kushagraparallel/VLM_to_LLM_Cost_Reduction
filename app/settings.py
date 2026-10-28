"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed runtime configuration; secrets are supplied through environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "VLM-to-LLM Cost Reduction"
    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: str = "http://localhost:5173"

    llm_provider: str = "gemini"
    vlm_provider: str = "gemini"
    gemini_api_key: str | None = None
    gemini_llm_model: str = "gemini-2.5-flash"
    gemini_vlm_model: str = "gemini-2.5-flash"
    groq_api_key: str | None = None
    groq_llm_model: str | None = None
    groq_vlm_model: str | None = None
    aws_region: str = "us-east-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    bedrock_llm_model_id: str | None = None
    bedrock_vlm_model_id: str | None = None

    yolo_model_path: str = "models/yolo26n_openvino_model"
    yolo_weights_path: str = "models/yolo26n.pt"
    yolo_confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    yolo_image_size: int = Field(default=640, gt=0)
    yolo_device: str = "intel:cpu"
    paddleocr_language: str = "en"
    max_upload_size_mb: int = Field(default=15, gt=0)
    results_db_path: str = "data/results.sqlite3"
    pricing_config_path: str = "config/model_pricing.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one cached settings instance per process."""
    return Settings()
