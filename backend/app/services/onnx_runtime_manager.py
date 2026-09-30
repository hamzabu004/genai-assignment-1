import os
import onnxruntime as ort
from app.core.config import settings

_SESSIONS: dict[str, ort.InferenceSession] = {}

MODEL_FILES = {
    "universal_ae": "task1_universal_ae.onnx",
    "classifier": "task2_classifier.onnx",
    "specialist_salt": "task2_specialist_salt.onnx",
    "specialist_blur": "task2_specialist_blur.onnx",
    "specialist_occlusion": "task2_specialist_occlusion.onnx",
    "soft_moe": "task3_soft_moe.onnx",
    "generator": "task4_generator.onnx",
}

# Preference order: NVIDIA CUDA > AMD ROCm (MIGraphX) > Windows DirectML (any GPU
# vendor incl. AMD/Intel) > CPU. We don't hardcode a provider — we ask onnxruntime
# what's actually available in THIS install and filter our preferred order down to
# that, so the exact same code runs unmodified on a CUDA box, a ROCm box, a Windows
# laptop with an AMD card, or a plain CPU-only machine. Whichever onnxruntime
# variant you installed (see requirements.txt) determines what shows up here.
_PROVIDER_PREFERENCE = [
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
]

def _resolve_providers() -> list[str]:
    available = ort.get_available_providers()
    chosen = [p for p in _PROVIDER_PREFERENCE if p in available]
    return chosen or ["CPUExecutionProvider"]

def load_all_models() -> list[str]:
    loaded = []
    providers = _resolve_providers()
    for key, filename in MODEL_FILES.items():
        path = os.path.join(settings.model_dir, filename)
        if os.path.exists(path):
            _SESSIONS[key] = ort.InferenceSession(path, providers=providers)
            loaded.append(key)
    return loaded

def get_session(key: str) -> ort.InferenceSession:
    if key not in _SESSIONS:
        raise RuntimeError(f"Model '{key}' not loaded. Check MODEL_DIR and file presence.")
    return _SESSIONS[key]

def loaded_model_keys() -> list[str]:
    return list(_SESSIONS.keys())

def active_providers() -> list[str]:
    """What's actually in effect, for the /health endpoint and debugging."""
    return _resolve_providers()
