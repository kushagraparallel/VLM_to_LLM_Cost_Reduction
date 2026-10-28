"""Provider adapters with a shared response contract."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any


@dataclass(frozen=True)
class GenerationResult:
    provider: str
    model: str
    text: str
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    estimated_cost_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result."""
        return asdict(self)


def _result(
    provider: str,
    model: str,
    response_text: str,
    started: float,
    input_tokens: Any = None,
    output_tokens: Any = None,
    reasoning_tokens: Any = None,
) -> GenerationResult:
    return GenerationResult(
        provider=provider,
        model=model,
        text=response_text or "",
        latency_seconds=round(perf_counter() - started, 6),
        input_tokens=int(input_tokens) if input_tokens is not None else None,
        output_tokens=int(output_tokens) if output_tokens is not None else None,
        reasoning_tokens=int(reasoning_tokens) if reasoning_tokens is not None else None,
    )


class ModelProvider:
    """Dispatch text and image generation to configured provider SDKs.

    Provider SDK details are kept here so the comparison service can remain
    independent of vendor-specific response shapes.
    """

    def __init__(self, settings: Any) -> None:
        self.settings = settings

    def generate_text(
        self, provider: str, model: str, prompt: str, system_instruction: str
    ) -> GenerationResult:
        if provider == "gemini":
            return self._gemini_text(model, prompt, system_instruction)
        if provider == "groq":
            return self._groq_text(model, prompt, system_instruction)
        if provider == "bedrock":
            return self._bedrock_text(model, prompt, system_instruction)
        raise ValueError(f"Unsupported text provider: {provider}")

    def generate_image(
        self, provider: str, model: str, image_bytes: bytes, mime_type: str, prompt: str
    ) -> GenerationResult:
        if provider == "gemini":
            return self._gemini_image(model, image_bytes, mime_type, prompt)
        if provider == "groq":
            return self._groq_image(model, image_bytes, mime_type, prompt)
        if provider == "bedrock":
            return self._bedrock_image(model, image_bytes, mime_type, prompt)
        raise ValueError(f"Unsupported vision provider: {provider}")

    def _gemini_client(self) -> Any:
        from google import genai

        if self.settings.gemini_use_vertexai:
            if not self.settings.google_cloud_project:
                raise ValueError("GOOGLE_CLOUD_PROJECT is required when GEMINI_USE_VERTEXAI=true.")
            return genai.Client(
                vertexai=True,
                project=self.settings.google_cloud_project,
                location=self.settings.google_cloud_location,
            )
        if not self.settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for the Gemini Developer API.")
        return genai.Client(api_key=self.settings.gemini_api_key)

    def _gemini_text(self, model: str, prompt: str, system_instruction: str) -> GenerationResult:
        from google.genai import types

        started = perf_counter()
        response = self._gemini_client().models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system_instruction),
        )
        usage = getattr(response, "usage_metadata", None)
        return _result(
            "gemini", model, getattr(response, "text", "") or "", started,
            getattr(usage, "prompt_token_count", None),
            getattr(usage, "candidates_token_count", None),
            getattr(usage, "thoughts_token_count", None),
        )

    def _gemini_image(self, model: str, image_bytes: bytes, mime_type: str, prompt: str) -> GenerationResult:
        from google.genai import types

        started = perf_counter()
        part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        response = self._gemini_client().models.generate_content(
            model=model,
            contents=[part, prompt],
        )
        usage = getattr(response, "usage_metadata", None)
        return _result(
            "gemini", model, getattr(response, "text", "") or "", started,
            getattr(usage, "prompt_token_count", None),
            getattr(usage, "candidates_token_count", None),
            getattr(usage, "thoughts_token_count", None),
        )

    def _groq_client(self) -> Any:
        from groq import Groq

        if not self.settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is required for the Groq provider.")
        return Groq(api_key=self.settings.groq_api_key)

    def _groq_text(self, model: str, prompt: str, system_instruction: str) -> GenerationResult:
        started = perf_counter()
        response = self._groq_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt},
            ],
        )
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return _result(
            "groq", model, choice.message.content or "", started,
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
        )

    def _groq_image(self, model: str, image_bytes: bytes, mime_type: str, prompt: str) -> GenerationResult:
        started = perf_counter()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{encoded}"
        response = self._groq_client().chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
        )
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return _result(
            "groq", model, choice.message.content or "", started,
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
        )

    def _bedrock_client(self) -> Any:
        import boto3

        credentials = {}
        if self.settings.aws_access_key_id and self.settings.aws_secret_access_key:
            credentials = {
                "aws_access_key_id": self.settings.aws_access_key_id,
                "aws_secret_access_key": self.settings.aws_secret_access_key,
            }
            if self.settings.aws_session_token:
                credentials["aws_session_token"] = self.settings.aws_session_token
        return boto3.client(
            "bedrock-runtime", region_name=self.settings.aws_region, **credentials
        )

    def _bedrock_text(self, model: str, prompt: str, system_instruction: str) -> GenerationResult:
        started = perf_counter()
        response = self._bedrock_client().converse(
            modelId=model,
            system=[{"text": system_instruction}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
        )
        message = response.get("output", {}).get("message", {})
        text = "\n".join(item.get("text", "") for item in message.get("content", []))
        usage = response.get("usage", {})
        return _result("bedrock", model, text, started, usage.get("inputTokens"), usage.get("outputTokens"))

    def _bedrock_image(self, model: str, image_bytes: bytes, mime_type: str, prompt: str) -> GenerationResult:
        started = perf_counter()
        image_format = {
            "image/jpeg": "jpeg",
            "image/png": "png",
            "image/gif": "gif",
            "image/webp": "webp",
        }.get(mime_type)
        if image_format is None:
            raise ValueError(f"Unsupported Bedrock image MIME type: {mime_type}")
        response = self._bedrock_client().converse(
            modelId=model,
            messages=[{
                "role": "user",
                "content": [
                    {"text": prompt},
                    {"image": {"format": image_format, "source": {"bytes": image_bytes}}},
                ],
            }],
        )
        message = response.get("output", {}).get("message", {})
        text = "\n".join(item.get("text", "") for item in message.get("content", []))
        usage = response.get("usage", {})
        return _result("bedrock", model, text, started, usage.get("inputTokens"), usage.get("outputTokens"))

