"""Shared deterministic evaluation helpers for the research notebooks."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from constants import PET_IMAGES_DIR, RESEARCH_ROOT, IMAGE_SIZE, TEST_MANIFEST
from src.corruptions import BENCHMARK_SEVERITIES, apply_corruption
from src.datasets import PetDataset
from src.losses import compute_ssim


class PetBenchmarkDataset(Dataset):
    """Flatten a deterministic per-image corruption manifest for evaluation."""

    def __init__(self, manifest_path: Path = TEST_MANIFEST):
        manifest_path = Path(manifest_path)
        with manifest_path.open() as f:
            manifest = json.load(f)
        self.samples = [
            (name, params)
            for name, entries in manifest.items()
            for params in (entries if isinstance(entries, list) else [entries])
        ]
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
        self.class_to_idx = PetDataset.CLASS_TO_IDX

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        name, params = self.samples[index]
        with Image.open(PET_IMAGES_DIR / name) as image:
            clean = self.transform(image.convert("RGB"))
        corruption = params["corruption_type"]
        corrupted = apply_corruption(
            clean,
            corruption,
            severity_params=params,
            rng=random.Random(params["seed"]),
        )
        severity = params.get("severity", "none")
        return corrupted, clean, self.class_to_idx[corruption], corruption, severity, name


def image_quality_metrics(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> Dict[str, float]:
    """Return batch-mean MSE, PSNR, and SSIM for NCHW images."""
    pred = pred.float().clamp(0.0, data_range)
    target = target.float().clamp(0.0, data_range)
    mse_each = (pred - target).square().flatten(1).mean(1)
    peak = torch.tensor(data_range, device=pred.device)
    psnr_each = 10.0 * torch.log10(peak.square() / mse_each.clamp_min(1e-12))
    ssim_map = compute_ssim(pred, target, data_range=data_range, reduction="none")
    ssim_each = ssim_map.flatten(1).mean(1)
    return {
        "mse": mse_each.mean().item(),
        "psnr": psnr_each.mean().item(),
        "ssim": ssim_each.mean().item(),
    }


def benchmark_levels() -> List[Dict[str, Any]]:
    """Describe the clean condition and fixed low/medium/high test settings."""
    return ([{"corruption_type": "clean", "severity": "none"}] + [
        {"corruption_type": kind, **params}
        for kind, levels in BENCHMARK_SEVERITIES.items()
        for params in levels
    ])
