"""Run the direct-VLM and structured-text-LLM paths for one image."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from PIL import Image
from starlette.concurrency import run_in_threadpool

from app.providers import GenerationResult, ModelProvider
from app.services.experiment_store import save_ai_call
from app.services.paddle_ocr import extract_text
from app.services.representation import (
    STRUCTURED_IMAGE_SYSTEM_INSTRUCTION,
    build_structured_user_message,
    structure_to_dict,
)
from app.services.yolo_detector import detect_objects
from app.settings import Settings

logger = logging.getLogger(__name__)


async def _safe_provider_call(
    call: Callable[..., GenerationResult],
    *args: Any,
    run_id: str,
    provider: str,
    operation: str,
    database_path: str,
    pricing_config_path: str,
    pricing_provider_key: str,
) -> dict[str, Any]:
    """Run one provider call and persist a log/storage record for it regardless of outcome."""
    started = perf_counter()
    call_name = getattr(call, "__qualname__", getattr(call, "__name__", "provider_call"))
    logger.info("provider_call_started function=%s run_id=%s", call_name, run_id)
    try:
        generation_result = await run_in_threadpool(call, *args)
        payload = generation_result.to_dict()
        status = "success"
        error_message = None
        logger.info(
            "provider_call_completed function=%s provider=%s model=%s duration_seconds=%.4f",
            call_name, generation_result.provider, generation_result.model, generation_result.latency_seconds,
        )
        payload["estimated_cost_usd"] = _estimate_cost(payload, pricing_provider_key, pricing_config_path)
    except Exception as exc:
        logger.exception("provider_call_failed function=%s duration_seconds=%.4f", call_name, perf_counter() - started)
        error_message = f"{type(exc).__name__}: {exc}"
        status = "error"
        payload = {
            "error": error_message,
            "provider_call_failed": True,
            "latency_seconds": round(perf_counter() - started, 6),
        }

    await run_in_threadpool(
        save_ai_call,
        database_path,
        {
            "call_id": str(uuid4()),
            "run_id": run_id,
            "provider": provider,
            "model": payload.get("model"),
            "operation": operation,
            "status": status,
            "duration_seconds": payload.get("latency_seconds"),
            "input_tokens": payload.get("input_tokens"),
            "output_tokens": payload.get("output_tokens"),
            "reasoning_tokens": payload.get("reasoning_tokens"),
            "estimated_cost_usd": payload.get("estimated_cost_usd"),
            "error": error_message,
        },
    )
    return payload


def _estimate_cost(
    result: dict[str, Any],
    provider: str,
    pricing_config_path: str,
) -> float | None:
    input_tokens = result.get("input_tokens")
    output_tokens = result.get("output_tokens")
    reasoning_tokens = result.get("reasoning_tokens") or 0
    if input_tokens is None or output_tokens is None:
        return None

    try:
        pricing = json.loads(Path(pricing_config_path).read_text(encoding="utf-8"))
        entry = pricing.get("models", {}).get(f"{provider}/{result.get('model')}")
        if not entry:
            return None
        today = datetime.now(UTC).date().isoformat()
        if entry.get("effective_from") and today < entry["effective_from"]:
            return None
        if entry.get("effective_until") and today > entry["effective_until"]:
            return None
        input_rate = entry.get("input_per_1m")
        output_rate = entry.get("output_per_1m")
        if input_rate is None or output_rate is None:
            return None
        return round((input_tokens * float(input_rate) + (output_tokens + reasoning_tokens) * float(output_rate)) / 1_000_000, 8)
    except (OSError, ValueError, TypeError):
        return None


async def run_comparison(
    *,
    run_id: str,
    settings: Settings,
    image_bytes: bytes,
    image: Image.Image,
    image_array: Any,
    mime_type: str,
    user_prompt: str,
    vlm_provider_override: str | None = None,
    llm_provider_override: str | None = None,
) -> dict[str, Any]:
    """Run the two paths and report independent latency/usage metadata."""
    provider = ModelProvider(settings)
    vlm_provider = (vlm_provider_override or settings.vlm_provider).lower()
    llm_provider = (llm_provider_override or settings.llm_provider).lower()
    vlm_model = {
        "gemini": settings.gemini_vlm_model,
        "groq": settings.groq_vlm_model,
        "bedrock": settings.bedrock_vlm_model_id,
    }.get(vlm_provider)
    llm_model = {
        "gemini": settings.gemini_llm_model,
        "groq": settings.groq_llm_model,
        "bedrock": settings.bedrock_llm_model_id,
    }.get(llm_provider)

    if not vlm_model:
        raise ValueError(f"No vision model is configured for provider '{vlm_provider}'.")
    if not llm_model:
        raise ValueError(f"No text model is configured for provider '{llm_provider}'.")

    total_started = perf_counter()
    logger.info("comparison_pipeline_started run_id=%s vlm_provider=%s vlm_model=%s llm_provider=%s llm_model=%s", run_id, vlm_provider, vlm_model, llm_provider, llm_model)
    vlm_pricing_key = "vertex_gemini" if vlm_provider == "gemini" and settings.gemini_use_vertexai else vlm_provider
    llm_pricing_key = "vertex_gemini" if llm_provider == "gemini" and settings.gemini_use_vertexai else llm_provider
    vlm_task = asyncio.create_task(
        _safe_provider_call(
            provider.generate_image,
            vlm_provider,
            vlm_model,
            image_bytes,
            mime_type,
            user_prompt,
            run_id=run_id,
            provider=vlm_provider,
            operation="image",
            database_path=settings.results_db_path,
            pricing_config_path=settings.pricing_config_path,
            pricing_provider_key=vlm_pricing_key,
        )
    )

    ocr_started = perf_counter()
    ocr_regions = await run_in_threadpool(extract_text, image_array)
    ocr_seconds = round(perf_counter() - ocr_started, 6)
    logger.info("comparison_ocr_completed regions=%s duration_seconds=%.4f", len(ocr_regions), ocr_seconds)

    detection_started = perf_counter()
    detections = await run_in_threadpool(detect_objects, image)
    detection_seconds = round(perf_counter() - detection_started, 6)
    logger.info("comparison_detection_completed detections=%s duration_seconds=%.4f", len(detections), detection_seconds)

    structure = await run_in_threadpool(
        structure_to_dict,
        image_size=image.size,
        ocr_regions=ocr_regions,
        detections=detections,
        processing={
            "ocr_seconds": ocr_seconds,
            "detection_seconds": detection_seconds,
        },
    )
    structured_prompt = build_structured_user_message(user_prompt, structure)
    logger.debug("comparison_representation_built ocr_regions=%s detections=%s relationships=%s", len(ocr_regions), len(detections), len(structure.relationships))

    llm_result = await _safe_provider_call(
        provider.generate_text,
        llm_provider,
        llm_model,
        structured_prompt,
        STRUCTURED_IMAGE_SYSTEM_INSTRUCTION,
        run_id=run_id,
        provider=llm_provider,
        operation="text",
        database_path=settings.results_db_path,
        pricing_config_path=settings.pricing_config_path,
        pricing_provider_key=llm_pricing_key,
    )
    vlm_result = await vlm_task

    vlm_result.setdefault("provider", vlm_provider)
    vlm_result.setdefault("model", vlm_model)
    llm_result.setdefault("provider", llm_provider)
    llm_result.setdefault("model", llm_model)

    pipeline_seconds = round(
        ocr_seconds + detection_seconds + float(llm_result.get("latency_seconds", 0.0)),
        6,
    )
    vlm_seconds = vlm_result.get("latency_seconds")
    pipeline_cost = llm_result.get("estimated_cost_usd")
    logger.info("comparison_pipeline_completed elapsed_seconds=%.4f direct_vlm_seconds=%s structured_pipeline_seconds=%.4f", perf_counter() - total_started, vlm_seconds, pipeline_seconds)

    return {
        "run_id": run_id,
        "user_prompt": user_prompt,
        "direct_vlm": {
            **vlm_result,
            "path_seconds": vlm_seconds,
            "cost_scope": "provider API estimate; local compute excluded",
        },
        "structured_pipeline": {
            "provider": llm_provider,
            "model": llm_model,
            "answer": llm_result.get("text"),
            "provider_result": llm_result,
            "local_processing_seconds": round(ocr_seconds + detection_seconds, 6),
            "ocr_seconds": ocr_seconds,
            "detection_seconds": detection_seconds,
            "path_seconds": pipeline_seconds,
            "cost_estimate_usd": pipeline_cost,
            "cost_scope": "text-model API estimate; local compute excluded",
            "representation": structure.model_dump(mode="json"),
        },
        "comparison": {
            "preferred_path": None,
            "direct_vlm_seconds": vlm_seconds,
            "structured_pipeline_seconds": pipeline_seconds,
            "direct_vlm_cost_usd": vlm_result.get("estimated_cost_usd"),
            "structured_pipeline_cost_usd": pipeline_cost,
            "elapsed_wall_seconds": round(perf_counter() - total_started, 6),
        },
    }
