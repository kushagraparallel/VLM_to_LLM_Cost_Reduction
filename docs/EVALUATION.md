# Evaluation methodology

## Primary research question
How much cost and latency can be reduced by replacing direct image-to-VLM calls with local OCR/object extraction plus a text-only LLM, and how does answer quality change?

## Metrics
For each path, store:
- total wall-clock latency;
- local preprocessing latency, where applicable;
- provider-call latency;
- input/output token counts when returned by the provider;
- cost estimate and the pricing configuration version;
- answer text, provider/model IDs, success/error status;
- user preference and optional rating/comment.

Keep unavailable metrics null rather than substituting zero. Separate measured latency from estimated cost.

## Comparison controls
Use the exact same user prompt for both paths. Use equivalent task instructions; only the structured path receives the additional system instruction describing the OCR/detection representation. Keep temperature and other generation parameters aligned where supported. Record differences that cannot be aligned across providers.

## Dataset and scoring
Create a small, versioned test set across receipts/documents, screenshots/UI, and everyday scenes. Include difficult examples such as small text, clutter, occlusion, and images with little readable text. Remove sensitive personal information before storing sample images or results.

Combine:
1. human preference between blinded summaries;
2. task-specific correctness checks;
3. completeness and hallucination/error review;
4. cost and latency measurements.

Report sample size and provider/model versions. Avoid claiming a universal cost reduction from a small or category-biased dataset.
