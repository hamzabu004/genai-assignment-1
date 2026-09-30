"""
Constants and shared path definitions for all research tasks.
Tasks 1, 2a, 2b, 3: Oxford-IIIT Pet Dataset (Autoencoders, Classifiers, MoE)
Task 4: FS2K Dataset (Style-Conditioned Face-to-Sketch cGAN)
"""

from pathlib import Path

# Paths
RESEARCH_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = RESEARCH_ROOT / "datasets"
CONFIGS_ROOT = RESEARCH_ROOT / "configs"
NOTEBOOKS_ROOT = RESEARCH_ROOT / "notebooks"
CHECKPOINTS_MANIFEST = RESEARCH_ROOT / "checkpoints_manifest.json"

# Tasks 1, 2a, 2b, 3: Oxford-IIIT Pet
PET_ROOT = DATASETS_ROOT / "oxford-iiit-pet"
PET_IMAGES_DIR = PET_ROOT / "images"
SPLIT_MANIFEST = RESEARCH_ROOT / "split_manifest.json"

# Task 4: FS2K
FS2K_ROOT = DATASETS_ROOT / "FS2K"
FS2K_PHOTO_DIR = FS2K_ROOT / "photo"
FS2K_SKETCH_DIR = FS2K_ROOT / "sketch"
FS2K_ANNO_TRAIN = FS2K_ROOT / "anno_train.json"
FS2K_ANNO_TEST = FS2K_ROOT / "anno_test.json"

# Preprocessing & Normalization constants (Plan 7)
# Tasks 1, 2a, 2b, 3: [0, 1] range (/255 only)
NORM_01_MEAN = [0.0, 0.0, 0.0]
NORM_01_STD = [1.0, 1.0, 1.0]

# Task 4: [-1, 1] range (/127.5 - 1)
NORM_11_MEAN = [0.5, 0.5, 0.5]
NORM_11_STD = [0.5, 0.5, 0.5]

IMAGE_SIZE = 128
SEED = 42
