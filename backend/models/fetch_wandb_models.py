#!/usr/bin/env python3
"""Fetch a versioned W&B ONNX artifact into this backend's model directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env", override=False)
except ImportError:
    pass

MODEL_FILES = [
    "task1_universal_ae.onnx", "task2_classifier.onnx",
    "task2_specialist_salt.onnx", "task2_specialist_blur.onnx",
    "task2_specialist_occlusion.onnx", "task3_soft_moe.onnx", "task4_generator.onnx",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        default=None,
        help="W&B artifact path entity/project/final-best-onnx:latest (or a specific version).",
    )
    parser.add_argument("--models-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if not os.getenv("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY is required. Set it in the environment or .env before fetching.")
    project = os.getenv("WANDB_PROJECT", "genai-assignment")
    entity = os.getenv("WANDB_ENTITY")
    artifact_path = args.artifact or (
        f"{entity}/{project}/final-best-onnx:latest" if entity else f"{project}/final-best-onnx:latest"
    )
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("W&B SDK missing. Install with `pip install wandb`.") from exc

    api = wandb.Api()
    try:
        artifact = api.artifact(artifact_path, type="model")
    except Exception as exc:
        raise SystemExit(f"Could not resolve W&B model Artifact {artifact_path!r}: {exc}") from exc

    target_dir = args.models_dir.expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="onnx-artifact-") as temp_dir:
        download_dir = Path(artifact.download(root=temp_dir))
        missing = [name for name in MODEL_FILES if not (download_dir / name).is_file()]
        if missing:
            raise SystemExit(f"Artifact {artifact_path!r} is missing required models: {', '.join(missing)}")
        manifest_path = download_dir / "final_best_onnx_validation.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        expected = {item["onnx_model"]: item["sha256"] for item in manifest.get("models", []) if "sha256" in item}
        for filename in MODEL_FILES:
            source = download_dir / filename
            if filename in expected:
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                if digest != expected[filename]:
                    raise SystemExit(f"SHA-256 mismatch for {filename}; refusing to install this artifact.")
        # Stage beside the destination: /tmp and the model directory may live on
        # different filesystems, where Path.replace() fails with EXDEV.
        with tempfile.TemporaryDirectory(prefix="onnx-staged-", dir=target_dir) as staging_dir:
            staging = Path(staging_dir)
            for filename in MODEL_FILES:
                shutil.copy2(download_dir / filename, staging / filename)
            if manifest_path.is_file():
                shutil.copy2(manifest_path, staging / "final_best_onnx_validation.json")

            for filename in MODEL_FILES:
                (staging / filename).replace(target_dir / filename)
            staged_manifest = staging / "final_best_onnx_validation.json"
            if staged_manifest.is_file():
                staged_manifest.replace(target_dir / staged_manifest.name)
    print(f"Fetched {artifact_path} into {target_dir}")


if __name__ == "__main__":
    main()
