import random
from typing import Optional, Tuple, Dict, Any
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

SEVERITY_LEVELS = ("low", "medium", "high")
VALID_CORRUPTIONS = ("clean", "salt_pepper", "blur", "occlusion")


def apply_salt_pepper(img_arr: np.ndarray, prob: float, seed: Optional[int] = None) -> np.ndarray:
    """
    Apply salt-and-pepper impulse noise to an image array in [0.0, 1.0] range.
    Shape: (H, W, C)
    """
    rng = np.random.default_rng(seed)
    noisy = img_arr.copy()
    h, w, c = noisy.shape

    # Generate 2D mask for impulse locations
    rand_mask = rng.random((h, w))
    salt_mask = rand_mask < (prob / 2.0)
    pepper_mask = (rand_mask >= (prob / 2.0)) & (rand_mask < prob)

    # Salt (white pixels = 1.0)
    noisy[salt_mask] = 1.0
    # Pepper (black pixels = 0.0)
    noisy[pepper_mask] = 0.0

    return np.clip(noisy, 0.0, 1.0)


def apply_gaussian_blur(img_arr: np.ndarray, kernel: int, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur to an image array in [0.0, 1.0] range.
    Shape: (H, W, C)
    """
    blurred = np.empty_like(img_arr)
    for ch in range(img_arr.shape[2]):
        blurred[:, :, ch] = gaussian_filter(img_arr[:, :, ch], sigma=sigma, mode="reflect")
    return np.clip(blurred, 0.0, 1.0)


def apply_occlusion(
    img_arr: np.ndarray,
    num_rects: int,
    coverage: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply random rectangular occlusions (cutouts) to simulate missing sensor data or objects.
    Shape: (H, W, C)
    """
    rng = np.random.default_rng(seed)
    occluded = img_arr.copy()
    h, w, _ = occluded.shape

    total_pixels = h * w
    target_area_per_rect = (total_pixels * coverage) / max(num_rects, 1)
    side = int(np.sqrt(target_area_per_rect))
    side = max(10, min(side, min(h, w) - 5))

    for _ in range(num_rects):
        rw = rng.integers(int(side * 0.75), int(side * 1.25) + 1)
        rh = rng.integers(int(side * 0.75), int(side * 1.25) + 1)
        rw = min(rw, w)
        rh = min(rh, h)

        x0 = rng.integers(0, max(1, w - rw + 1))
        y0 = rng.integers(0, max(1, h - rh + 1))

        occluded[y0 : y0 + rh, x0 : x0 + rw, :] = 0.0

    return np.clip(occluded, 0.0, 1.0)


def get_corruption_parameters(
    corruption_type: str, severity: Optional[str]
) -> Tuple[str, Dict[str, Any]]:
    """Determine the parameter dictionary for a given corruption type and severity level."""
    if corruption_type == "clean":
        return "none", {}

    sev = (severity or "medium").lower()
    if sev not in SEVERITY_LEVELS:
        raise ValueError(f"Invalid severity '{severity}'. Must be one of: {list(SEVERITY_LEVELS)}")

    if corruption_type == "salt_pepper":
        prob_map = {"low": 0.05, "medium": 0.15, "high": 0.30}
        prob = prob_map[sev]
        return sev, {"prob": prob, "intensity": prob}

    elif corruption_type == "blur":
        params_map = {
            "low": {"kernel": 3, "sigma": 1.0},
            "medium": {"kernel": 5, "sigma": 1.5},
            "high": {"kernel": 9, "sigma": 3.0},
        }
        return sev, params_map[sev]

    elif corruption_type == "occlusion":
        params_map = {
            "low": {"num_rects": 1, "coverage": 0.06},
            "medium": {"num_rects": 2, "coverage": 0.14},
            "high": {"num_rects": 3, "coverage": 0.28},
        }
        return sev, params_map[sev]

    else:
        raise ValueError(f"Invalid corruption_type '{corruption_type}'. Must be one of: {list(VALID_CORRUPTIONS)}")


def apply_corruption_to_image(
    image: Image.Image,
    corruption_type: Optional[str] = "clean",
    severity: Optional[str] = None,
    seed: Optional[int] = None,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Applies corruption to a PIL Image (RGB) and returns (corrupted_image, corruption_applied_dict).
    """
    c_type = (corruption_type or "clean").lower()
    if c_type not in VALID_CORRUPTIONS:
        raise ValueError(f"Invalid corruption_type '{corruption_type}'. Must be one of: {list(VALID_CORRUPTIONS)}")

    if c_type == "clean":
        return image.copy(), {
            "type": "clean",
            "severity": "none",
            "params": {},
        }

    actual_sev, params = get_corruption_parameters(c_type, severity)

    arr = np.asarray(image).astype(np.float32) / 255.0

    if c_type == "salt_pepper":
        corrupted_arr = apply_salt_pepper(arr, prob=params["prob"], seed=seed)
    elif c_type == "blur":
        corrupted_arr = apply_gaussian_blur(arr, kernel=params["kernel"], sigma=params["sigma"])
    elif c_type == "occlusion":
        corrupted_arr = apply_occlusion(
            arr, num_rects=params["num_rects"], coverage=params["coverage"], seed=seed
        )
    else:
        corrupted_arr = arr

    corrupted_uint8 = np.clip(corrupted_arr * 255.0, 0, 255).astype(np.uint8)
    corrupted_image = Image.fromarray(corrupted_uint8, mode="RGB")

    applied_dict = {
        "type": c_type,
        "severity": actual_sev,
        "params": params,
    }
    return corrupted_image, applied_dict

