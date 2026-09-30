"""
Single source of truth for picking a compute device in every notebook/script.
Import get_device() everywhere instead of writing .cuda()/.to("cuda") directly,
so the same notebook runs unmodified on a laptop (CPU), Colab (NVIDIA CUDA),
a University PC with an NVIDIA card (CUDA), or a machine with an AMD GPU running
a ROCm-built PyTorch (ROCm surfaces through the SAME torch.cuda.* API in
PyTorch's ROCm wheels, so no separate branch is needed for it).

Preference order: CUDA/ROCm > Apple Metal (MPS) > CPU.
"""
import torch


def get_device(verbose: bool = True) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        backend = "ROCm" if torch.version.hip is not None else "CUDA"
        if verbose:
            print(f"[device] Using GPU via {backend}: {name}")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("[device] Using Apple Metal (MPS) GPU")
    else:
        device = torch.device("cpu")
        if verbose:
            print("[device] No GPU detected — using CPU")
    return device


def device_report() -> dict:
    """Structured version for logging to W&B / notebook config cells."""
    if torch.cuda.is_available():
        return {
            "device_type": "rocm" if torch.version.hip is not None else "cuda",
            "device_name": torch.cuda.get_device_name(0),
        }
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return {"device_type": "mps", "device_name": "Apple GPU"}
    return {"device_type": "cpu", "device_name": "CPU"}
