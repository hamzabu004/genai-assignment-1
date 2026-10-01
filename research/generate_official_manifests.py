"""Build the assignment's official Oxford Pet split and deterministic benchmarks."""

import json
import math
import random
import tarfile
from pathlib import Path

from constants import PET_ROOT, PET_IMAGES_DIR, SPLIT_MANIFEST, VAL_MANIFEST, TEST_MANIFEST

SEED = 42


def official_names(member_name):
    archive_path = PET_ROOT / "annotations.tar.gz"
    with tarfile.open(archive_path, "r:gz") as archive:
        stream = archive.extractfile(member_name)
        if stream is None:
            raise FileNotFoundError(f"{member_name} is not in {archive_path}")
        return [line.split()[0] + ".jpg" for line in stream.read().decode().splitlines() if line.strip()]


def rectangles(rng, count, coverage, size=128):
    target_area = size * size * coverage / count
    masks = []
    for _ in range(count):
        aspect = rng.uniform(0.5, 2.0)
        width = max(4, min(size - 2, int(math.sqrt(target_area * aspect))))
        height = max(4, min(size - 2, int(math.sqrt(target_area / aspect))))
        masks.append({"top": rng.randint(0, size - height), "left": rng.randint(0, size - width),
                      "height": height, "width": width})
    return masks


def generate_split():
    trainval = official_names("annotations/trainval.txt")
    test = official_names("annotations/test.txt")
    missing = [name for name in trainval + test if not (PET_IMAGES_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} official pet images; first: {missing[0]}")
    rng = random.Random(SEED)
    rng.shuffle(trainval)
    n_train = int(0.8 * len(trainval))
    split = {"seed": SEED, "train": trainval[:n_train], "val": trainval[n_train:], "test": test}
    SPLIT_MANIFEST.write_text(json.dumps(split, indent=2) + "\n")
    return split


def generate_val_manifest(names):
    rng = random.Random(SEED)
    classes = ["clean", "salt", "blur", "occlusion"]
    manifest = {}
    for name in names:
        kind = rng.choice(classes)
        params = {"corruption_type": kind, "seed": rng.randint(0, 2**31 - 1)}
        if kind == "salt":
            params["prob"] = rng.uniform(0.02, 0.15)
        elif kind == "blur":
            params.update(kernel_size=rng.choice([3, 5, 7]), sigma=rng.uniform(0.5, 2.5))
        elif kind == "occlusion":
            count, coverage = rng.randint(1, 3), rng.uniform(0.10, 0.35)
            params.update(num_rects=count, coverage=coverage, masks=rectangles(rng, count, coverage))
        manifest[name] = params
    VAL_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


def generate_test_manifest(names):
    manifest = {}
    for name in names:
        rng = random.Random(f"{SEED}:{name}")
        tasks = [{"corruption_type": "clean", "severity": "none"}]
        tasks.extend({"corruption_type": "salt", "severity": level, "prob": prob}
                     for level, prob in [("low", 0.03), ("med", 0.08), ("high", 0.15)])
        tasks.extend({"corruption_type": "blur", "severity": level,
                      "kernel_size": kernel, "sigma": sigma}
                     for level, kernel, sigma in [("low", 3, 0.7), ("med", 5, 1.5), ("high", 7, 2.5)])
        for level, count, coverage in [("low", 1, 0.10), ("med", 2, 0.20), ("high", 3, 0.35)]:
            tasks.append({"corruption_type": "occlusion", "severity": level,
                          "num_rects": count, "coverage": coverage,
                          "masks": rectangles(rng, count, coverage)})
        for task in tasks:
            task["seed"] = rng.randint(0, 2**31 - 1)
        manifest[name] = tasks
    TEST_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    split = generate_split()
    generate_val_manifest(split["val"])
    generate_test_manifest(split["test"])
    print(f"Wrote {len(split['train'])} train, {len(split['val'])} validation, "
          f"and {len(split['test'])} official test images.")
