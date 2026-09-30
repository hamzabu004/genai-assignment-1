import base64
import io
from typing import Dict, List
import numpy as np
from PIL import Image


def compute_error_map(clean: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
    """Computes element-wise absolute difference between clean and reconstructed tensors or arrays."""
    return np.abs(clean - reconstructed)


def generate_heatmap_palette() -> list[int]:
    """Generates a 256-color thermal heatmap palette [R, G, B, R, G, B, ...] for 8-bit PIL images."""
    palette = []
    for i in range(256):
        norm = i / 255.0
        # Dark blue -> Cyan -> Green -> Yellow -> Red -> White
        if norm < 0.2:
            r = int(norm * 5.0 * 60)
            g = int(norm * 5.0 * 20)
            b = int(norm * 5.0 * 180)
        elif norm < 0.4:
            t = (norm - 0.2) * 5.0
            r = int(60 * (1 - t))
            g = int(20 + t * 180)
            b = int(180 + t * 75)
        elif norm < 0.7:
            t = (norm - 0.4) * 3.33
            r = int(t * 255)
            g = int(200 + t * 55)
            b = int(255 * (1 - t))
        else:
            t = (norm - 0.7) * 3.33
            r = 255
            g = int(255 * (1 - t * 0.8))
            b = int(t * 180)
        palette.extend([min(255, max(0, r)), min(255, max(0, g)), min(255, max(0, b))])
    return palette


_HEATMAP_PALETTE = generate_heatmap_palette()


def error_map_to_base64_png(clean_img: Image.Image, reconstructed_img: Image.Image) -> str:
    """
    Computes visual residual error map between clean and reconstructed images,
    applies contrast scaling and false-color thermal mapping, and returns base64 PNG data URL.
    """
    c_arr = np.asarray(clean_img.convert("RGB")).astype(np.float32)
    r_arr = np.asarray(reconstructed_img.convert("RGB")).astype(np.float32)

    # Pixel difference across channels
    diff = np.abs(c_arr - r_arr)
    # Mean error per pixel (0 - 255)
    gray_diff = np.mean(diff, axis=2)

    # Amplify subtle errors (3.5x scale) so residual patterns are clearly inspectable
    amplified = np.clip(gray_diff * 3.5, 0, 255).astype(np.uint8)

    # Apply thermal heatmap palette
    err_img = Image.fromarray(amplified, mode="L")
    err_img.putpalette(_HEATMAP_PALETTE)
    err_rgb = err_img.convert("RGB")

    buf = io.BytesIO()
    err_rgb.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax."""
    x = logits / max(temperature, 1e-6)
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def format_probabilities(probs: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    """Formats 1D probability array into dictionary with rounded values summing to 1.0."""
    p_flat = probs.flatten().tolist()
    result = {}
    for name, p in zip(class_names, p_flat):
        result[name] = round(float(p), 4)
    return result

