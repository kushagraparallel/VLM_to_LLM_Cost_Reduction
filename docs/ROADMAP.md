# Roadmap

## Phase 1 — Reproducible backend foundation
- [x] Declare Python version, dependency bounds, and environment template.
- [x] Add FastAPI application entry point and health endpoint.
- [x] Document architecture, setup, configuration, and evaluation.
- [x] Add image upload validation and image metadata extraction.
- [x] Implement PaddleOCR service.
- [x] Implement YOLO26 Nano/OpenVINO service and export workflow.
- [x] Build versioned structured representation.

## Phase 2 — Model comparison
- [x] Define common provider result and usage schemas.
- [x] Implement Gemini text and vision adapters.
- [x] Add optional Groq and AWS Bedrock adapters; model capability validation remains to be strengthened.
- [x] Run both paths with the same user prompt.
- [x] Track latency, usage, pricing estimates, and provider errors.

## Phase 3 — Evaluation UI and persistence
- [x] Build React + Vite + TypeScript upload/comparison interface.
- [x] Allow users to rate each output and choose the better summary.
- [x] Persist experiment runs and feedback.
- [ ] Add repeatable dataset evaluation and exportable results.

## Phase 4 — Hosting and controlled trials
- [ ] Prepare deployment configuration and production database.
- [ ] Add email OTP verification and one-time trial access.
- [ ] Add request limits, abuse protection, and retention controls.
- [ ] Expose an MCP adapter after the core HTTP workflow stabilizes.
