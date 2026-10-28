# VLM-to-LLM Cost Reduction

A research project comparing direct Vision-Language Model (VLM) image understanding with a lower-cost structured pipeline that extracts text and objects locally, then asks a text-only LLM to produce an answer.

## Project status

This repository is being initialized. The initial implementation will provide a FastAPI backend and a React + Vite + TypeScript comparison UI. Core pipeline components will be separated into image validation, PaddleOCR, YOLO26 Nano/OpenVINO detection, structured representation, provider adapters, and evaluation/feedback services.

## Approach

1. **Direct VLM baseline:** send the image and the user's prompt to a vision-capable model.
2. **Structured pipeline:** run PaddleOCR and YOLO26 Nano locally, serialize OCR text, object labels, confidence scores, bounding boxes, and normalized coordinates into a structured representation, then send that representation and the same user prompt to a text-only LLM.
3. **Fair comparison:** the structured pipeline adds a system instruction explaining that the representation is the model's internal view of the image. The user prompt remains identical for both paths.
4. **Evaluation:** capture latency, token usage when returned by the provider, estimated cost where pricing is configured, errors, user ratings, and the user's preferred answer.

Gemini is the primary provider. Groq and AWS Bedrock are optional provider integrations. Provider availability and model capabilities depend on configured model IDs and credentials.

## Recommended Python version

Use **Python 3.11.x (64-bit)** for the backend. Python 3.11 is a conservative compatibility target for the computer-vision and OCR dependencies. Use a dedicated virtual environment and install the requirements in the documented order. PaddlePaddle wheel availability varies by operating system and architecture; consult the official PaddleOCR/PaddlePaddle installation instructions if the default CPU wheel is unavailable for your platform.

## Quick start

### 1. Backend environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Copy the environment template and configure the provider you intend to use:

```powershell
Copy-Item .env.example .env
```

On macOS/Linux, use `cp .env.example .env`. Never commit real API keys.

### 2. Run the API

```bash
uvicorn app.main:app --reload
```

API documentation will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 3. Frontend

Frontend setup instructions will be added with the frontend workspace. The intended stack is React, Vite, and TypeScript.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Setup and troubleshooting](docs/SETUP.md)
- [Configuration and providers](docs/CONFIGURATION.md)
- [Evaluation methodology](docs/EVALUATION.md)
- [Roadmap](docs/ROADMAP.md)

## Model and licensing notes

YOLO26 Nano is loaded through Ultralytics and can be exported to OpenVINO format. Model weights may be downloaded on first use or generated during export. Review Ultralytics licensing terms before distribution or commercial use. Review the licenses and terms for all selected model providers and OCR models.

## Development standards

- Keep provider-specific API code behind common interfaces.
- Keep image processing and external API calls out of route handlers.
- Read configuration from environment variables; do not hard-code secrets.
- Add tests for deterministic logic and provider-independent behavior.
- Use focused conventional commit messages, for example `feat: add PaddleOCR extraction service` and `test: cover structured representation serialization`.
