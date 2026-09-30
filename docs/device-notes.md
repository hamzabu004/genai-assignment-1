# Device / Platform Notes

## Training side (PyTorch, in `research/`)
Always get the device via `research/src/device_utils.py::get_device()` — never hardcode `.cuda()`.

| Hardware | What to install | What `get_device()` returns |
|---|---|---|
| NVIDIA GPU (Colab, most University PCs) | Default PyPI PyTorch (`pip install torch`) — CUDA build | `cuda` |
| AMD GPU, Linux | ROCm-specific PyTorch wheel: `pip install torch --index-url https://download.pytorch.org/whl/rocm<version>` (check pytorch.org for the current ROCm version tag) — **not** the default `pip install torch` | `cuda` (ROCm reuses the CUDA API namespace in PyTorch — this is expected, not a bug) |
| AMD GPU, Windows | ROCm does not support Windows compute. Default PyTorch install will silently fall back to CPU. | `cpu` (GPU not actually used — see workaround below if this matters) |
| Apple Silicon laptop | Default PyPI PyTorch (MPS backend included) | `mps` |
| Any machine, no GPU or unsupported GPU | Default PyPI PyTorch | `cpu` |

**Windows + AMD workaround (optional, only if you actually hit this combo):** `torch-directml` is a separate, less-maintained package that gives DirectML-backed GPU acceleration on Windows for any GPU vendor. It requires code changes (`import torch_directml`, different device object) beyond what `device_utils.py` currently does — only worth adding if CPU training on that specific machine is too slow to be usable.

## Inference side (onnxruntime, in `backend/`)
Provider selection is automatic (`onnx_runtime_manager.py` queries `ort.get_available_providers()` and picks the best match) — but this only works if you installed a build of onnxruntime that actually contains that provider:

| Hardware | Install |
|---|---|
| CPU only | `onnxruntime` (default, works everywhere) |
| NVIDIA CUDA | `onnxruntime-gpu` |
| Windows, any GPU vendor (incl. AMD) | `onnxruntime-directml` |
| AMD GPU, Linux (ROCm) | No official PyPI wheel; either build onnxruntime from source with `--use_rocm`, or skip GPU inference for that model on that machine (CPU inference is generally fine at 128×128 image sizes) |

Only install **one** onnxruntime variant per environment — they conflict if installed together. Check `GET /health` after startup; it reports `onnx_providers`, confirming which one is actually active.
