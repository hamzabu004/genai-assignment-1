"""
Datasets module for all research tasks.
Provides:
- PetDataset: Oxford-IIIT Pet dataset for Tasks 1, 2a, 2b, and 3.
- FS2KDataset: Paired photo-sketch dataset for Task 4.
"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from constants import (
    RESEARCH_ROOT,
    PET_IMAGES_DIR,
    SPLIT_MANIFEST,
    VAL_MANIFEST,
    TEST_MANIFEST,
    FS2K_ROOT,
    FS2K_PHOTO_DIR,
    FS2K_SKETCH_DIR,
    FS2K_ANNO_TRAIN,
    FS2K_ANNO_TEST,
    IMAGE_SIZE,
    SEED,
)
from src.corruptions import apply_corruption


class PetDataset(Dataset):
    """
    Oxford-IIIT Pet dataset for:
    - Task 1: Universal Autoencoder (corruption_mode='all') -> (corrupted, clean)
    - Task 2a: Corruption Classifier (corruption_mode='label') -> (corrupted, label_idx)
    - Task 2b: Specialist Restoration (corruption_mode='salt'|'blur'|'occlusion') -> (corrupted, clean)
    - Task 3: Soft Mixture-of-Experts -> (corrupted, clean, label_idx)
    """

    CLASSES = ["clean", "salt", "blur", "occlusion"]
    CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}

    def __init__(
        self,
        mode: str = "train",
        corruption_mode: str = "all",
        image_size: int = IMAGE_SIZE,
        tiny_size: int = 50,
        return_label: bool = False,
        seed: int = SEED,
    ):
        """
        mode: 'train', 'val', or 'tiny'
        corruption_mode: 'all', 'clean', 'salt', 'blur', 'occlusion', or 'label'
        """
        super().__init__()
        self.mode = mode
        self.corruption_mode = corruption_mode.lower()
        self.image_size = image_size
        self.return_label = return_label or (self.corruption_mode == "label")
        self.seed = seed

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # Scales to [0, 1]
        ])

        # Load or create split manifest
        self.paths = self._get_image_paths(mode, tiny_size)

        self.val_manifest = None
        self.test_manifest = None
        if self.mode == "val":
            val_path = VAL_MANIFEST
            if val_path.exists():
                import json
                with open(val_path, "r") as f:
                    self.val_manifest = json.load(f)
        elif self.mode == "test":
            test_path = TEST_MANIFEST
            if test_path.exists():
                import json
                with open(test_path, "r") as f:
                    self.test_manifest = json.load(f)

    def _get_image_paths(self, mode: str, tiny_size: int) -> List[Path]:
        if not PET_IMAGES_DIR.exists():
            raise FileNotFoundError(f"Pet images directory not found: {PET_IMAGES_DIR}")

        # Use the generated split based on Oxford's official lists.
        if SPLIT_MANIFEST.exists():
            with open(SPLIT_MANIFEST, "r") as f:
                manifest = json.load(f)
        else:
            raise FileNotFoundError(
                f"Official split manifest is missing: {SPLIT_MANIFEST}. "
                "Run research/generate_official_manifests.py first."
            )

        if mode == "train":
            names = manifest["train"]
        elif mode == "val":
            names = manifest["val"]
        elif mode == "tiny":
            names = manifest["train"][:tiny_size]
        elif mode == "test":
            names = manifest["test"]
        else:
            raise ValueError(f"Unknown mode: {mode}. Expected 'train', 'val', or 'tiny'.")

        return [PET_IMAGES_DIR / name for name in names]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        with Image.open(path) as img:
            clean = self.transform(img.convert("RGB"))

        img_name = path.name

        # Keep specialist validation on its requested corruption while making it reproducible.
        if self.mode == "val" and self.corruption_mode not in ("all", "label"):
            corruption_type = self.corruption_mode
            corrupted = apply_corruption(
                clean, corruption_type, rng=random.Random(self.seed + idx)
            )
        # Use the fixed validation manifest for universal and classifier evaluation.
        elif self.mode == "val" and self.val_manifest and img_name in self.val_manifest:
            params = self.val_manifest[img_name]
            corruption_type = params["corruption_type"]
            rng = random.Random(params["seed"])
            corrupted = apply_corruption(clean, corruption_type, severity_params=params, rng=rng)
        # Use manifest for testing if available
        elif self.mode == "test" and self.test_manifest and img_name in self.test_manifest:
            # For dataset iterating, we might just return the list of 10 tasks
            # But the dataset API expects a single return.
            # In a real test loop, the evaluator would iterate over tasks.
            # We'll just return the first task for generic dataloader compatibility,
            # or the user can parse the test_manifest directly in their test script.
            params = self.test_manifest[img_name][0]
            corruption_type = params["corruption_type"]
            rng = random.Random(params["seed"])
            corrupted = apply_corruption(clean, corruption_type, severity_params=params, rng=rng)
        else:
            # Training / dynamic behavior
            if self.corruption_mode in ("all", "label"):
                corruption_type = random.choice(self.CLASSES)
            elif self.corruption_mode in self.CLASSES:
                corruption_type = self.corruption_mode
            else:
                raise ValueError(f"Invalid corruption_mode: {self.corruption_mode}")
            corrupted = apply_corruption(clean, corruption_type)
        label_idx = self.CLASS_TO_IDX[corruption_type]

        if self.corruption_mode == "label":
            return corrupted, label_idx
        elif self.return_label:
            return corrupted, clean, label_idx
        else:
            return corrupted, clean


class FS2KDataset(Dataset):
    """
    FS2K Paired Photo-to-Sketch dataset for Task 4.
    Resizes images to 128x128 and normalizes to [-1, 1].
    Returns (photo, sketch, style_id).
    """

    NUM_STYLES = 3

    def __init__(
        self,
        mode: str = "train",
        image_size: int = IMAGE_SIZE,
        tiny_size: int = 20,
        val_ratio: float = 0.15,
        seed: int = SEED,
    ):
        super().__init__()
        self.mode = mode
        self.image_size = image_size
        self.seed = seed

        # Range [-1, 1] normalization
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.samples = self._load_samples(mode, tiny_size, val_ratio)

    def _find_file(self, base_dir: Path, rel_path: str) -> Optional[Path]:
        for ext in [".jpg", ".JPG", ".jpeg", ".png", ".PNG"]:
            p = base_dir / f"{rel_path}{ext}"
            if p.exists():
                return p
        return None

    def _load_samples(self, mode: str, tiny_size: int, val_ratio: float) -> List[Dict[str, Any]]:
        if mode in ("train", "val", "tiny"):
            anno_path = FS2K_ANNO_TRAIN
        elif mode == "test":
            anno_path = FS2K_ANNO_TEST
        else:
            raise ValueError(f"Unknown mode: {mode}")

        with open(anno_path, "r") as f:
            raw_items = json.load(f)

        valid_items = []
        for it in raw_items:
            photo_rel = it["image_name"]
            sketch_rel = photo_rel.replace("photo", "sketch").replace("image", "sketch")
            photo_path = self._find_file(FS2K_PHOTO_DIR, photo_rel)
            sketch_path = self._find_file(FS2K_SKETCH_DIR, sketch_rel)

            if photo_path and sketch_path:
                valid_items.append({
                    "photo_path": photo_path,
                    "sketch_path": sketch_path,
                    "style": int(it["style"]),
                })

        if mode in ("train", "val", "tiny"):
            # Stratified 85/15 train/val split by style
            rng = random.Random(self.seed)
            by_style = {s: [] for s in range(self.NUM_STYLES)}
            for item in valid_items:
                by_style[item["style"]].append(item)

            train_items = []
            val_items = []
            for s, items in by_style.items():
                rng.shuffle(items)
                n_val = int(len(items) * val_ratio)
                val_items.extend(items[:n_val])
                train_items.extend(items[n_val:])

            if mode == "train":
                return train_items
            elif mode == "val":
                return val_items
            elif mode == "tiny":
                return train_items[:tiny_size]
        else:
            return valid_items

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.samples[idx]
        with Image.open(sample["photo_path"]) as p_img:
            photo = self.transform(p_img.convert("RGB"))
        with Image.open(sample["sketch_path"]) as s_img:
            sketch = self.transform(s_img.convert("RGB"))
        return photo, sketch, sample["style"]
