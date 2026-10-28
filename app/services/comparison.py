"""Run the direct-VLM and structured-text-LLM paths for one image."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from PIL import Image
from starlette.concurrency import run_in_threadpool

from app.providers import GenerationResult, ModelProvider
from app.services.paddle_ocr import extract_text
from app.services.representation import (
    STRUCTURED_IMAGE_SYSTEM_INSTRUCTION,
    build_structured_user_message,
    structure_to_dict,
)
from app.services.yolo_detector import detect_objects
from app.settings import Settings


async def _safe_provider_call(call: Callable[..., GenerationResult], *args: Any) -> dict[str, Any]:
    started = perf_counter()
    try:
        result = await run_in_threadpool(call, *args)
        return result.to_dict()
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "provider_call_failed": True,
            "latency_seconds": round(perf_counter() - started, 6),
        }


def _estimate_cost(
    result: dict[str, Any],
    provider: str,
    pricing_config_path: str,
) -> float | None:
    input_tokens = result.get("input_tokens")
    output_tokens = result.get("output_tokens")
    if input_tokens is None or output_tokens is None:
        return None

    try:
        pricing = json.loads(Path(pricing_config_path).read_text(encoding="utf-8"))
        entry = pricing.get("models", {}).get(f"{provider}/{result.get('model')}")
        if not entry:
            return None
        input_rate = entry.get("input_per_1m")
        output_rate = entry.get("output_per_1m")
        if input_rate is None or output_rate is None:
            return None
        return round((input_tokens * float(input_rate) + output_tokens * float(output_rate)) / 1_000_000, 8)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


async def run_comparison(
    *,
    settings: Settings,
    image_bytes: bytes,
    image: Image.Image,
    image_array: Any,
    mime_type: str,
    user_prompt: str,
) -> dict[str, Any]:
    """Run the two paths and report independent latency/usage metadata."""
    provider = ModelProvider(settings)
    vlm_provider = settings.vlm_provider.lower()
    llm_provider = settings.llm_provider.lower()
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
    vlm_task = asyncio.create_task(
        _safe_provider_call(
            provider.generate_image,
            vlm_provider,
            vlm_model,
            image_bytes,
            mime_type,
            user_prompt,
        )
    )

    ocr_started = perf_counter()
    ocr_regions = await run_in_threadpool(extract_text, image_array)
    ocr_seconds = round(perf_counter() - ocr_started, 6)

    detection_started = perf_counter()
    detections = await run_in_threadpool(detect_objects, image)
    detection_seconds = round(perf_counter() - detection_started, 6)

    structure = structure_to_dict(
        image_size=image.size,
        ocr_regions=ocr_regions,
        detections=detections,
        processing={
            "ocr_seconds": ocr_seconds,
            "detection_seconds": detection_seconds,
        },
    )
    structured_prompt = build_structured_user_message(user_prompt, structure)

    llm_result = await _safe_provider_call(
        provider.generate_text,
        llm_provider,
        llm_model,
        structured_prompt,
        STRUCTURED_IMAGE_SYSTEM_INSTRUCTION,
    )
    vlm_result = await vlm_task

    vlm_result.setdefault("provider", vlm_provider)
    vlm_result.setdefault("model", vlm_model)
    llm_result.setdefault("provider", llm_provider)
    llm_result.setdefault("model", llm_model)

    if not vlm_result.get("provider_call_failed"):
        vlm_result["estimated_cost_usd"] = _estimate_cost(
            vlm_result, vlm_provider, settings.pricing_config_path
        )
    if not llm_result.get("provider_call_failed"):
        llm_result["estimated_cost_usd"] = _estimate_cost(
            llm_result, llm_provider, settings.pricing_config_path
        )

    pipeline_seconds = round(
        ocr_seconds + detection_seconds + float(llm_result.get("latency_seconds", 0.0)),
        6,
    )
    vlm_seconds = vlm_result.get("latency_seconds")
    pipeline_cost = llm_result.get("estimated_cost_usd")

    return {
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
