# Setup and troubleshooting

## Supported runtime
Use **64-bit CPython 3.11.x** for the initial backend. Create a new virtual environment rather than reusing an environment that already contains OpenCV, PaddlePaddle, or Torch packages. Windows, Linux, and macOS package availability differs; PaddlePaddle CPU wheels in particular are platform-dependent.

Check your interpreter:

```bash
python --version
python -c "import platform, sys; print(sys.executable); print(platform.architecture())"
```

## Install

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

### macOS/Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs and check http://127.0.0.1:8000/health.

## Vision models
YOLO26 Nano uses Ultralytics' `yolo26n.pt` checkpoint. Export once to OpenVINO IR using a compatible Ultralytics installation:

```bash
yolo export model=yolo26n.pt imgsz=640 format=openvino
```

Set `YOLO_MODEL_PATH=models/yolo26n_openvino_model` if the exported folder is placed there. Exact export output paths can vary by working directory and Ultralytics version. Keep model weights and exported artifacts out of Git.

PaddleOCR 3.x requires PaddlePaddle and downloads model assets on first use. If installation fails, use the official PaddlePaddle installation selector for your operating system, architecture, and CPU/GPU setup, install the matching PaddlePaddle wheel, then install PaddleOCR. Do not install a CUDA build on a CPU-only machine.

## Dependency compatibility checks
Run these commands after installing:

```bash
python -m pip check
python -c "import fastapi, pydantic, paddleocr, openvino, ultralytics; print('imports ok')"
```

OpenCV is a common source of conflicts: PaddleOCR/PaddleX and Ultralytics may declare different OpenCV distribution packages, even though those distributions write to the same `cv2` namespace. If `pip check` reports incompatible requirements or imports fail, inspect `python -m pip show opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless`. Do not install multiple OpenCV variants blindly. Resolve the selected PaddleOCR/Ultralytics versions in a clean environment and rerun `pip check`.

If the current package releases stop supporting Python 3.11, or the platform has no compatible PaddlePaddle wheel, adjust the Python version and dependency bounds together and document the tested matrix instead of bypassing dependency checks.

## Common problems
- **PowerShell blocks activation:** run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.
- **Model folder not found:** export YOLO26 Nano to OpenVINO format and set `YOLO_MODEL_PATH` to the resulting folder.
- **Slow first request:** the first model call may download model files and compile OpenVINO inference.
- **Provider authentication errors:** check the relevant API key or AWS credential chain and selected model ID.
- **Import or ABI errors:** recreate the environment and use `python -m pip check`; avoid mixing system Python packages into the virtual environment.
