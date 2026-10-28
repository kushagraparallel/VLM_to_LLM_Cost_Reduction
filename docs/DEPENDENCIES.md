# Dependency compatibility review

This file records the dependency decisions and the checks that must be repeated on the target machine. The repository has not yet been installed and executed on every supported operating system, so this is a static compatibility review, not a claim that a full environment build has passed.

## Runtime target

| Area | Selection | Reason |
|---|---|---|
| Python | CPython 3.11, 64-bit | Conservative common runtime for PaddlePaddle/PaddleOCR, Ultralytics, OpenVINO, and provider SDKs |
| OCR | PaddleOCR 3.x + PaddlePaddle 3.x | PaddleOCR 3.x Paddle inference requires PaddlePaddle 3.0 or later |
| Detection | Ultralytics 8.4.132 or later | Use a release line with YOLO26 model support |
| Inference | OpenVINO 2025.x–2026.x | Run exported YOLO26 Nano IR on CPU by default |
| NumPy | >=1.26,<2.3 | Conservative bound to reduce ABI surprises across the selected vision stack |
| API | FastAPI + Pydantic 2 | Async HTTP boundary and typed validation |
| Providers | google-genai, groq, boto3 | Optional provider SDKs are imported only when used |

Official references:
- [PaddleOCR installation](https://www.paddleocr.ai/main/en/version3.x/installation.html)
- [PaddlePaddle installation selector](https://www.paddlepaddle.org.cn/install/quick)
- [Ultralytics OpenVINO integration](https://docs.ultralytics.com/integrations/openvino)

## Known OpenCV package overlap

OpenCV is the main known packaging risk. Ultralytics declares opencv-python; PaddleOCR/PaddleX versions and installation paths may declare opencv-contrib-python or other OpenCV distributions. These distributions expose the same cv2 import namespace and can overwrite overlapping files if multiple variants are installed. pip check validates package metadata but may not detect every file-level collision.

To reduce surprises:
1. Start with a fresh virtual environment; do not reuse an environment that already contains OpenCV packages.
2. Do not manually add another OpenCV distribution unless the selected dependency set requires it.
3. Run python -m pip check.
4. Inspect installed variants with python -m pip show opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless.
5. Run the import smoke test from docs/SETUP.md and exercise both OCR and detection on a sample image.
6. If cv2 imports fail or one library fails at runtime, resolve the OpenCV package selection for the actual PaddleOCR/Ultralytics versions. A separate OCR environment/service may be the most reliable isolation boundary if upstream metadata continues to require conflicting distributions.

## Installation verification checklist

Run these after installation and retain the output in your local development notes:

    python --version
    python -m pip check
    python -m pip show paddlepaddle paddleocr ultralytics openvino numpy opencv-python opencv-contrib-python
    python -c "import fastapi, pydantic, paddleocr, openvino, ultralytics; print('imports ok')"
    python -m pytest

The first OCR/detection request downloads or initializes models. A successful import alone does not prove that model inference works. Verify the complete /api/v1/vision/structure endpoint before measuring comparisons.

## GPU and platform caveats

The default requirements install the PaddlePaddle CPU package and configure YOLO/OpenVINO for intel:cpu. Do not install a GPU PaddlePaddle wheel unless you intentionally select a matching CUDA/runtime configuration from the official installer. OpenVINO acceleration depends on supported Intel hardware and drivers; CPU inference remains the fallback.
