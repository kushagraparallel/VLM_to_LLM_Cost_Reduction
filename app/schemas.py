"""Typed API and vision representation schemas."""

from typing import Any

from pydantic import BaseModel, Field


class OCRTextRegion(BaseModel):
    text: str
    confidence: float | None = None
    polygon: list[list[float]] = Field(default_factory=list)
    bbox_xyxy: list[float] | None = None


class ObjectDetection(BaseModel):
    label: str
    class_id: int
    confidence: float
    bbox_xyxy: list[float]
    bbox_normalized: list[float]


class ImageStructure(BaseModel):
    schema_version: str = "1.0"
    image: dict[str, Any]
    ocr_regions: list[OCRTextRegion] = Field(default_factory=list)
    detections: list[ObjectDetection] = Field(default_factory=list)
    relationships: list[dict[str, Any]] = Field(default_factory=list)
    processing: dict[str, float] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    environment: str
