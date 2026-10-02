"""ONNX Runtime sessions and lightweight fallbacks used by the task routers."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import onnxruntime as ort
from scipy.ndimage import gaussian_filter, median_filter

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

_PROVIDER_PREFERENCE = [
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
]


def _resolve_providers() -> list[str]:
    available = ort.get_available_providers()
    chosen = [provider for provider in _PROVIDER_PREFERENCE if provider in available]
    return chosen or ["CPUExecutionProvider"]


def load_all_models() -> list[str]:
    """Load every present ONNX model; missing optional tasks remain unloaded."""
    loaded: list[str] = []
    providers = _resolve_providers()
    model_dir = settings.resolved_model_dir
    for key, filename in MODEL_FILES.items():
        path = os.path.join(model_dir, filename)
        if not os.path.isfile(path):
            _SESSIONS.pop(key, None)
            continue
        _SESSIONS[key] = ort.InferenceSession(path, providers=providers)
        loaded.append(key)
    return loaded


def has_model(key: str) -> bool:
    return key in _SESSIONS


def run_model(key: str, input_tensor: np.ndarray) -> list[np.ndarray]:
    session = get_session(key)
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: np.asarray(input_tensor, dtype=np.float32)})


def run_model_with_inputs(key: str, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Run a model that has more than one named input (for example Task 4 style conditioning)."""
    session = get_session(key)
    expected = {item.name for item in session.get_inputs()}
    if set(inputs) != expected:
        raise ValueError(f"Model '{key}' expects inputs {sorted(expected)}, received {sorted(inputs)}")
    feed = {name: np.asarray(value) for name, value in inputs.items()}
    return session.run(None, feed)


def get_session(key: str) -> ort.InferenceSession:
    if key not in _SESSIONS:
        raise RuntimeError(f"Model '{key}' is not loaded. Check MODEL_DIR and model availability.")
    return _SESSIONS[key]


def loaded_model_keys() -> list[str]:
    return list(_SESSIONS)


def active_providers() -> list[str]:
    return _resolve_providers()


def fallback_universal_ae(x: np.ndarray) -> np.ndarray:
    """Compatibility fallback. Universal inference itself requires the trained ONNX model."""
    return np.asarray(x, dtype=np.float32)


def fallback_classify_degradation(x: np.ndarray) -> tuple[np.ndarray, Any]:
    """Return a conservative clean-first probability estimate for demo-only routers."""
    batch = np.asarray(x, dtype=np.float32)
    probs = np.tile(np.asarray([0.25, 0.25, 0.25, 0.25], dtype=np.float32), (batch.shape[0], 1))
    return probs, None


def fallback_denoise_median(x: np.ndarray) -> np.ndarray:
    return median_filter(np.asarray(x, dtype=np.float32), size=(1, 1, 3, 3)).astype(np.float32)


def fallback_unsharp_mask(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    blurred = gaussian_filter(arr, sigma=(0, 0, 1.0, 1.0))
    return np.clip(arr + 0.7 * (arr - blurred), -1.0, 1.0).astype(np.float32)


def fallback_inpaint_occlusion(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    return gaussian_filter(arr, sigma=(0, 0, 2.0, 2.0)).astype(np.float32)


def fallback_face_to_sketch(x: np.ndarray, style: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    gray = np.mean(arr, axis=1, keepdims=True)
    edges = gray - gaussian_filter(gray, sigma=(0, 0, 1.2, 1.2))
    strength = {"style_1": 2.0, "style_2": 3.0, "style_3": 4.0}.get(style, 3.0)
    sketch = np.clip(1.0 - np.abs(edges) * strength, -1.0, 1.0)
    return np.repeat(sketch, 3, axis=1).astype(np.float32)
