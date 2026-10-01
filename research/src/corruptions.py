"""
Corruption pipeline for Tasks 1, 2a, 2b, and 3.
Applies pure PyTorch tensor operations for:
- Salt & Pepper noise
- Gaussian blur
- Random rectangular occlusion

Conforms to Plan 6 data specification:
- Salt-pepper: p ~ U(0.02, 0.15), equal 0/1 chance
- Blur: kernel in {3, 5, 7}, sigma ~ U(0.5, 2.5)
- Occlusion: 1-3 rectangles, 10-35% total coverage
"""

import math
import random
from typing import Optional, Tuple, Union, Dict, Any
import torch
import torch.nn.functional as F


def _get_gaussian_kernel1d(kernel_size: int, sigma: float) -> torch.Tensor:
    k = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    kernel = torch.exp(-0.5 * (k / sigma) ** 2)
    return kernel / kernel.sum()


def _get_gaussian_kernel2d(kernel_size: int, sigma: float) -> torch.Tensor:
    k1d = _get_gaussian_kernel1d(kernel_size, sigma)
    kernel2d = torch.outer(k1d, k1d)
    return kernel2d


def add_salt_pepper(
    x: torch.Tensor,
    prob: Optional[float] = None,
    rng: Optional[random.Random] = None
) -> torch.Tensor:
    """
    Adds salt and pepper noise to image tensor in [0, 1].
    prob: float in [0.0, 1.0]. If None, sampled from U(0.02, 0.15).
    """
    if prob is None:
        prob = rng.uniform(0.02, 0.15) if rng else random.uniform(0.02, 0.15)

    if prob <= 0:
        return x.clone()

    out = x.clone()
    # Random mask for corrupted pixels
    generator = None
    if rng is not None:
        generator = torch.Generator(device=out.device)
        generator.manual_seed(rng.randint(0, 2**31 - 1))
    mask = torch.rand(out.shape, device=out.device, dtype=out.dtype, generator=generator) < prob
    # Equal chance of 0 (pepper) or 1 (salt)
    salt_mask = torch.rand(out.shape, device=out.device, dtype=out.dtype, generator=generator) < 0.5
    out[mask & salt_mask] = 1.0
    out[mask & (~salt_mask)] = 0.0
    return out


def add_blur(
    x: torch.Tensor,
    kernel_size: Optional[int] = None,
    sigma: Optional[float] = None,
    rng: Optional[random.Random] = None
) -> torch.Tensor:
    """
    Applies Gaussian blur to image tensor.
    x: tensor of shape (C, H, W) or (B, C, H, W)
    kernel_size: odd int in {3, 5, 7}. If None, chosen uniformly.
    sigma: float. If None, sampled from U(0.5, 2.5).
    """
    if kernel_size is None:
        kernel_size = rng.choice([3, 5, 7]) if rng else random.choice([3, 5, 7])
    if sigma is None:
        sigma = rng.uniform(0.5, 2.5) if rng else random.uniform(0.5, 2.5)

    is_batched = (x.ndim == 4)
    if not is_batched:
        x = x.unsqueeze(0)

    b, c, h, w = x.shape
    kernel2d = _get_gaussian_kernel2d(kernel_size, sigma).to(device=x.device, dtype=x.dtype)
    kernel2d = kernel2d.view(1, 1, kernel_size, kernel_size).repeat(c, 1, 1, 1)

    padding = kernel_size // 2
    blurred = F.conv2d(x, kernel2d, padding=padding, groups=c)

    if not is_batched:
        blurred = blurred.squeeze(0)
    return blurred.clamp(0.0, 1.0)


def add_occlusion(
    x: torch.Tensor,
    num_rects: Optional[int] = None,
    coverage: Optional[float] = None,
    fill_value: float = 0.0,
    rng: Optional[random.Random] = None
) -> torch.Tensor:
    """
    Adds 1-3 random rectangular occlusion patches covering 10-35% total area.
    x: tensor of shape (C, H, W) or (B, C, H, W)
    """
    if num_rects is None:
        num_rects = rng.randint(1, 3) if rng else random.randint(1, 3)
    if coverage is None:
        coverage = rng.uniform(0.10, 0.35) if rng else random.uniform(0.10, 0.35)

    out = x.clone()
    is_batched = (out.ndim == 4)
    batch_tensors = out if is_batched else [out]

    # If explicit masks are passed (from manifest), use them directly
    explicit_masks = getattr(rng, 'explicit_masks', None) if rng else None

    for item in batch_tensors:
        c, h, w = item.shape[-3:]

        if explicit_masks:
            for m in explicit_masks:
                top, left = m["top"], m["left"]
                rh, rw = m["height"], m["width"]
                item[:, top : top + rh, left : left + rw] = fill_value
        else:
            total_pixels = h * w
            target_pixels_per_rect = int((total_pixels * coverage) / num_rects)

            for _ in range(num_rects):
                # Sample aspect ratio between 0.5 and 2.0
                aspect = rng.uniform(0.5, 2.0) if rng else random.uniform(0.5, 2.0)
                rw = int(math.sqrt(target_pixels_per_rect * aspect))
                rh = int(math.sqrt(target_pixels_per_rect / aspect))
                rw = max(4, min(w - 2, rw))
                rh = max(4, min(h - 2, rh))

                top = (rng.randint(0, h - rh) if rng else random.randint(0, h - rh)) if h > rh else 0
                left = (rng.randint(0, w - rw) if rng else random.randint(0, w - rw)) if w > rw else 0
                item[:, top : top + rh, left : left + rw] = fill_value

    return out


def apply_corruption(
    x: torch.Tensor,
    corruption_type: str,
    severity_params: Optional[Dict[str, Any]] = None,
    rng: Optional[random.Random] = None
) -> torch.Tensor:
    """
    Applies specified corruption ('clean', 'salt', 'blur', 'occlusion') to tensor.
    """
    params = severity_params or {}
    c = corruption_type.lower()
    if c == "clean":
        return x.clone()
    elif c in ("salt", "salt_pepper", "salt-pepper"):
        return add_salt_pepper(x, prob=params.get("prob"), rng=rng)
    elif c == "blur":
        return add_blur(x, kernel_size=params.get("kernel_size"), sigma=params.get("sigma"), rng=rng)
    elif c == "occlusion":
        if rng and "masks" in params:
            rng.explicit_masks = params["masks"]
        elif "masks" in params:
            class DummyRNG:
                pass
            rng = DummyRNG()
            rng.explicit_masks = params["masks"]

        return add_occlusion(
            x,
            num_rects=params.get("num_rects"),
            coverage=params.get("coverage"),
            fill_value=params.get("fill_value", 0.0),
            rng=rng
        )
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")


# Fixed benchmark test severities (Plan 6 §1.1)
BENCHMARK_SEVERITIES = {
    "salt": [
        {"severity": "low", "prob": 0.03},
        {"severity": "med", "prob": 0.08},
        {"severity": "high", "prob": 0.15},
    ],
    "blur": [
        {"severity": "low", "kernel_size": 3, "sigma": 0.7},
        {"severity": "med", "kernel_size": 5, "sigma": 1.5},
        {"severity": "high", "kernel_size": 7, "sigma": 2.5},
    ],
    "occlusion": [
        {"severity": "low", "num_rects": 1, "coverage": 0.10},
        {"severity": "med", "num_rects": 2, "coverage": 0.20},
        {"severity": "high", "num_rects": 3, "coverage": 0.35},
    ],
}
