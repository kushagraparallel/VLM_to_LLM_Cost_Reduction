# Architecture

## Two comparable paths

### Direct VLM baseline
The image and the user's prompt are sent to a configured vision-capable model. The user's prompt is passed unchanged.

### Structured OCR/object pipeline
1. Validate and decode the uploaded image.
2. Run PaddleOCR to extract recognized text and geometry.
3. Run YOLO26 Nano through OpenVINO to detect objects and bounding boxes.
4. Normalize detections and OCR regions into a versioned JSON representation, including pixel/normalized bounding boxes and bounded geometric relationships such as left-of, above, overlap, and below.
5. Send that representation and the exact same user prompt to a text-capable LLM. This path adds a separate system instruction explaining that the supplied structured representation is the pipeline's internal visual evidence, not a literal image.

## Boundaries
- API routes handle HTTP validation and response shaping only.
- Vision services own PaddleOCR and YOLO/OpenVINO lifecycle and inference.
- A representation builder converts OCR and detections into provider-neutral JSON.
- LLM and VLM provider adapters encapsulate provider SDKs, model identifiers, token metadata, errors, and usage reporting.
- Evaluation services normalize latency, usage, cost estimates, errors, and user preference feedback.
- Persistence is isolated behind a repository interface; SQLite is suitable for local experiments, while hosted deployments should use a managed database.

## Fairness and measurement
- Keep the user prompt byte-for-byte identical between paths.
- Do not send the original image to the structured LLM call.
- Record preprocessing, provider-call, and total elapsed time separately.
- Report provider-reported usage when available; distinguish missing usage from zero usage.
- Treat cost as an estimate derived from configurable per-model rates, not as a billing statement.
- Preserve errors and partial results instead of silently comparing an answer against a failed run.

## Model loading
Heavy models should be initialized lazily and reused within the process. Avoid loading model weights at module import time, especially for health checks and tests. OpenVINO model export/download artifacts should be kept out of Git.

## Logging and concurrency

- Configure application logging through `LOG_LEVEL` (defaults to `INFO`); use `DEBUG` for local troubleshooting.
- Request middleware records request start/completion, HTTP status, duration, and an optional caller-supplied `X-Request-ID`. It does not log prompts, credentials, or image contents.
- Route, OCR, detector, comparison, provider, and SQLite operations log stage start/completion and failures with durations or item counts where useful.
- Synchronous provider SDK calls, image decoding/normalization, OCR, object detection, SQLite operations, representation construction, and pricing-file reads are dispatched through Starlette's thread pool from async routes/services. Declaring blocking work as `async` alone does not make it non-blocking.
- Provider adapters remain synchronous functions intentionally: they execute in a worker thread via `run_in_threadpool`, keeping blocking vendor SDK calls off the event loop without forcing a vendor-specific async client.
- Avoid logging prompts, OCR contents, raw provider responses, API keys, or uploaded image data because these may contain sensitive information.
