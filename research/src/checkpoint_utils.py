"""
Checkpoint management and tracking module.
Conforms to Plan 4 §5 & Plan 6 specifications:
- Saves latest.pt and best.pt
- Records metadata: epoch, val_loss, seed, trial number, git commit hash
- Updates checkpoints_manifest.json automatically
- Safely loads checkpoints across CPU/GPU devices
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from constants import CHECKPOINTS_MANIFEST


def get_git_commit_hash() -> str:
    """Returns current git commit hash, or 'not_git_repo' if unavailable."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def create_checkpoint_dict(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    val_loss: float = 0.0,
    optuna_trial_number: Optional[int] = None,
    random_seed: int = 42,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Builds standard checkpoint dictionary."""
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "val_loss": float(val_loss),
        "optuna_trial_number": optuna_trial_number,
        "random_seed": random_seed,
        "git_commit_hash": get_git_commit_hash(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    if extra_metadata:
        ckpt.update(extra_metadata)
    return ckpt


def update_manifest(
    manifest_path: Path,
    filename: str,
    task: str,
    phase: str,
    val_loss: float,
    epoch: int,
    device_name: str,
    git_commit_hash: str,
):
    """Logs or updates an entry in checkpoints_manifest.json."""
    data = {"_description": "Tracks all model checkpoints.", "checkpoints": []}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                data = json.load(f)
        except Exception:
            pass

    # Filter out existing entry for this exact filename if already present
    existing = [c for c in data.get("checkpoints", []) if c.get("filename") != filename]

    new_entry = {
        "filename": filename,
        "task": task,
        "phase": phase,
        "val_loss": round(float(val_loss), 6),
        "epoch": epoch,
        "device": device_name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_commit_hash": git_commit_hash,
    }
    existing.append(new_entry)
    data["checkpoints"] = existing

    with open(manifest_path, "w") as f:
        json.dump(data, f, indent=2)


def save_checkpoint(
    checkpoint_dir: Path,
    prefix: str,
    ckpt_dict: Dict[str, Any],
    is_best: bool = False,
    task_name: str = "task1",
    phase_name: str = "simple",
    device_name: str = "cpu",
    manifest_path: Path = CHECKPOINTS_MANIFEST,
) -> Dict[str, Path]:
    """
    Saves latest.pt and (if is_best) best.pt.
    Updates manifest with the saved record.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    latest_filename = f"{prefix}_latest.pt"
    latest_path = checkpoint_dir / latest_filename
    torch.save(ckpt_dict, latest_path)

    saved_paths = {"latest": latest_path}

    update_manifest(
        manifest_path=manifest_path,
        filename=latest_filename,
        task=task_name,
        phase=phase_name,
        val_loss=ckpt_dict["val_loss"],
        epoch=ckpt_dict["epoch"],
        device_name=device_name,
        git_commit_hash=ckpt_dict["git_commit_hash"],
    )

    if is_best:
        best_filename = f"{prefix}_best.pt"
        best_path = checkpoint_dir / best_filename
        torch.save(ckpt_dict, best_path)
        saved_paths["best"] = best_path

        update_manifest(
            manifest_path=manifest_path,
            filename=best_filename,
            task=task_name,
            phase=f"{phase_name}_best",
            val_loss=ckpt_dict["val_loss"],
            epoch=ckpt_dict["epoch"],
            device_name=device_name,
            git_commit_hash=ckpt_dict["git_commit_hash"],
        )

    return saved_paths


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Loads weights from checkpoint into model and optimizer safely across devices."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    target_device = device if device is not None else torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=target_device, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    print(
        f"[checkpoint] Loaded {checkpoint_path.name} "
        f"(Epoch {ckpt.get('epoch')}, Val Loss: {ckpt.get('val_loss'):.4f})"
    )
    return ckpt
