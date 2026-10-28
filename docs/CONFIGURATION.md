# Configuration

Copy `.env.example` to `.env`. Keep real credentials local and rotate any key that has been accidentally committed.

## Providers
- `LLM_PROVIDER`: `gemini`, `groq`, or `bedrock` for the text-only structured pipeline.
- `VLM_PROVIDER`: `gemini` or `bedrock` where the selected model supports image input. Groq VLM support depends on the selected model; configure only a model that supports vision.
- Gemini uses the Gemini Developer API by default (`GEMINI_API_KEY`) and separate text/vision model IDs. The default model is `gemini-3.8-flash` for both paths.
- To use Vertex AI instead, set `GEMINI_USE_VERTEXAI=true`, configure `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`, and use Application Default Credentials or an appropriate service identity. Model and regional availability must be checked for the chosen Vertex location.
- The sample pricing entry applies only to the Gemini Developer API. Vertex AI costs remain unavailable until a separate, verified Vertex pricing entry is configured.
- Groq uses `GROQ_API_KEY` and model IDs supported by the account.
- Bedrock accepts optional `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN` values from `.env`; otherwise it uses the standard AWS credential chain. Model IDs must be enabled in the selected region.

The UI lets each comparison choose a VLM provider and an LLM provider independently. The model IDs still come from this environment configuration. Provider/model availability changes. Validate selected model capabilities at startup or request time; do not assume every provider model supports vision, usage reporting, or the same token-counting semantics.

## Local image processing
- `YOLO_MODEL_PATH`: exported OpenVINO model directory.
- `YOLO_WEIGHTS_PATH`: source Ultralytics checkpoint used for export.
- `YOLO_CONFIDENCE`: minimum object-detection confidence.
- `YOLO_IMAGE_SIZE`: inference image size.
- `YOLO_DEVICE`: OpenVINO device such as `intel:cpu`.
- `PADDLEOCR_LANGUAGE`: PaddleOCR language selection.

## Measurement
Model pricing should live in a versioned configuration file rather than provider code. Each pricing entry should identify the provider/model, effective date, currency, input rate, output rate, and billing unit. Cost should be marked unavailable when usage or a pricing entry is missing; never represent an unknown cost as zero.
