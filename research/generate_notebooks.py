#!/usr/bin/env python3
"""
Generator script to build all 15 research Jupyter notebooks with standard nbformat 4.
Generates:
- 10 full-code experimentation notebooks (simple + Optuna for Tasks 1, 2a, 2b, 3, 4)
- 5 validation stub notebooks (imports + headers + TODOs)
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path("/run/media/hmz-gul/01DB003D88B96CA0/Semesters-Data/sem-7/Gen-AI/genai-assignment/research/notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def md_cell(text):
    lines = [l + "\n" for l in text.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }


def code_cell(code):
    lines = [l + "\n" for l in code.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }


# ============================================================================
# Notebook 1: task1_vae_simple.ipynb
# ============================================================================
def build_task1_vae_simple():
    cells = [
        md_cell("""# Task 1: Universal Multi-Corruption Restoration (VAE Simple Baseline)

This notebook implements the **Simple VAE Baseline** for Task 1 as specified in **Plan 4 §2** and **Plan 6 §1**.

### Workflow:
1. **Config & Environment**: Device detection and hyperparameter definitions.
2. **Data Pipeline**: Oxford-IIIT Pet dataset with runtime corruptions (clean, salt-pepper, blur, occlusion at 25% each).
3. **Model**: Symmetric Convolutional VAE (`ConvVAE`) with GroupNorm and strided convs (no skip connections in baseline).
4. **Loss**: Composite reconstruction loss ($\\alpha \\cdot \\text{L1} + (1-\\alpha) \\cdot (1-\\text{SSIM})$) + fixed small $\\beta \\cdot \\text{KL}$.
5. **Training & Validation**: Multi-epoch training loop with automatic checkpointing (`latest.pt` & `best.pt`).
6. **Evaluation & Visualization**: Visual inspection grid and transition gate checks before proceeding to Optuna.
"""),
        code_cell("""# 1. Configuration & Hyperparameters
import os
import sys
from pathlib import Path

# Ensure research root is in sys.path & load environment from .env
cwd = Path.cwd().resolve()
RESEARCH_ROOT = cwd if (cwd / "src").exists() else (cwd / "research" if (cwd / "research" / "src").exists() else cwd.parent)
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from constants import load_environment
load_environment()

# Set TINY_RUN=True for quick 2-epoch CPU/GPU sanity checks (Plan 4 §1)
TINY_RUN = False
TINY_SIZE = 50

# Training hyperparameters (Plan 7 reference)
BATCH_SIZE = 32
LR = 1e-3
LATENT_DIM = 128
BASE_CHANNELS = 32
DROPOUT = 0.1
ALPHA = 0.8       # Reconstruction loss: alpha*L1 + (1-alpha)*(1-SSIM)
BETA = 0.001      # KL divergence weight
EPOCHS = 2 if TINY_RUN else 15
SEED = 42

# Experiment tracking
# Automatically enabled if WANDB_API_KEY is set in .env, or toggle True/False manually
USE_WANDB = bool(os.getenv("WANDB_API_KEY"))
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "genai-assignment")
WANDB_RUN_NAME = "task1_vae_simple_baseline"
"""),
        code_cell("""# 2. Imports and Environment Setup
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from constants import PET_IMAGES_DIR, CHECKPOINTS_MANIFEST
from src.device_utils import get_device, device_report
from src.datasets import PetDataset
from src.models_vae import ConvVAE
from src.train_loop import train_one_epoch_vae, validate_vae, set_seed
from src.checkpoint_utils import create_checkpoint_dict, save_checkpoint

set_seed(SEED)
print("[Setup] Imports successful. Manifest path:", CHECKPOINTS_MANIFEST)
"""),
        code_cell("""# 3. Device Check
device = get_device(verbose=True)
report = device_report()
print(f"[Device Report] Type: {report['device_type']}, Name: {report['device_name']}")
if torch.cuda.is_available():
    print(f"[GPU Memory] Total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
"""),
        code_cell("""# 4. Data Loading
mode_train = "tiny" if TINY_RUN else "train"
mode_val = "tiny" if TINY_RUN else "val"

train_dataset = PetDataset(mode=mode_train, corruption_mode="all", tiny_size=TINY_SIZE, seed=SEED)
val_dataset = PetDataset(mode=mode_val, corruption_mode="all", tiny_size=TINY_SIZE, seed=SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=(device.type == "cuda")
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=(device.type == "cuda")
)

print(f"[Data] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
print(f"[Data] Batches per epoch: Train={len(train_loader)}, Val={len(val_loader)}")
"""),
        code_cell("""# 5. Data Inspection: Sample Pairs
sample_corrupted, sample_clean = next(iter(train_loader))

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for i in range(4):
    c_img = sample_corrupted[i].permute(1, 2, 0).cpu().numpy()
    cl_img = sample_clean[i].permute(1, 2, 0).cpu().numpy()
    
    axes[0, i].imshow(c_img.clip(0, 1))
    axes[0, i].set_title(f"Sample {i+1} Corrupted")
    axes[0, i].axis("off")
    
    axes[1, i].imshow(cl_img.clip(0, 1))
    axes[1, i].set_title(f"Sample {i+1} Clean Target")
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()
"""),
        code_cell("""# 6. Model & Optimizer Initialization
model = ConvVAE(
    in_channels=3,
    base_channels=BASE_CHANNELS,
    latent_dim=LATENT_DIM,
    dropout=DROPOUT,
    use_skip=False  # Baseline: no skip connections (Plan 6 §1.2)
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[Model] ConvVAE initialized with {total_params:,} trainable parameters.")

optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

# Optional Weights & Biases Logging
if USE_WANDB:
    import wandb
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        entity=os.getenv("WANDB_ENTITY") or None,
        config={
            "task": "task1_universal_vae",
            "device": report["device_type"],
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "latent_dim": LATENT_DIM,
            "alpha": ALPHA,
            "beta": BETA,
            "epochs": EPOCHS,
        }
    )
"""),
        code_cell("""# 7. Training & Validation Loop
history = {
    "train_loss": [], "recon_loss": [], "kl_loss": [], "ssim": [],
    "val_loss": [], "val_recon": [], "val_kl": [], "val_ssim": []
}

best_val_loss = float("inf")
checkpoint_dir = RESEARCH_ROOT / "checkpoints"

print(f"--- Starting Training ({EPOCHS} Epochs) ---")
for epoch in range(1, EPOCHS + 1):
    train_metrics = train_one_epoch_vae(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        device=device,
        alpha=ALPHA,
        beta=BETA,
        scaler=scaler
    )
    val_metrics = validate_vae(
        model=model,
        loader=val_loader,
        device=device,
        alpha=ALPHA,
        beta=BETA
    )
    
    # Record history
    for k, v in train_metrics.items():
        history[k].append(v)
    for k, v in val_metrics.items():
        history[k].append(v)
        
    val_loss = val_metrics["val_loss"]
    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss
        
    # Save checkpoint
    ckpt_dict = create_checkpoint_dict(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        val_loss=val_loss,
        random_seed=SEED,
        extra_metadata={"alpha": ALPHA, "beta": BETA, "latent_dim": LATENT_DIM}
    )
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        prefix="task1_universal_ae",
        ckpt_dict=ckpt_dict,
        is_best=is_best,
        task_name="task1",
        phase_name="simple",
        device_name=report["device_name"]
    )
    
    if USE_WANDB:
        wandb.log({**train_metrics, **val_metrics, "epoch": epoch})
        
    print(
        f"Epoch [{epoch:02d}/{EPOCHS:02d}] "
        f"Train Loss: {train_metrics['train_loss']:.4f} (Recon: {train_metrics['recon_loss']:.4f}, KL: {train_metrics['kl_loss']:.4f}) | "
        f"Val Loss: {val_loss:.4f} (SSIM: {val_metrics['val_ssim']:.4f}) {'*Best*' if is_best else ''}"
    )

print("--- Training Completed ---")
if USE_WANDB:
    wandb.finish()
"""),
        code_cell("""# 8. Learning Curves
epochs_range = range(1, len(history["train_loss"]) + 1)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(epochs_range, history["train_loss"], label="Train Loss", color="blue")
axes[0].plot(epochs_range, history["val_loss"], label="Val Loss", color="red", linestyle="--")
axes[0].set_title("Total Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(epochs_range, history["recon_loss"], label="Train Recon", color="green")
axes[1].plot(epochs_range, history["val_recon"], label="Val Recon", color="darkgreen", linestyle="--")
axes[1].set_title("Reconstruction Loss (L1 + SSIM)")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(epochs_range, history["kl_loss"], label="Train KL", color="purple")
axes[2].plot(epochs_range, history["val_kl"], label="Val KL", color="darkviolet", linestyle="--")
axes[2].set_title("KL Divergence")
axes[2].set_xlabel("Epoch")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()
"""),
        code_cell("""# 9. Qualitative Visual Inspection Grid (Plan 6 §1.3)
model.eval()
val_batch_corrupted, val_batch_clean = next(iter(val_loader))

with torch.no_grad():
    recon_batch, _, _ = model(val_batch_corrupted.to(device))
    recon_batch = recon_batch.cpu()

num_display = min(6, len(val_batch_corrupted))
fig, axes = plt.subplots(num_display, 4, figsize=(16, 3.5 * num_display))

for i in range(num_display):
    c_in = val_batch_corrupted[i].permute(1, 2, 0).numpy().clip(0, 1)
    target = val_batch_clean[i].permute(1, 2, 0).numpy().clip(0, 1)
    output = recon_batch[i].permute(1, 2, 0).numpy().clip(0, 1)
    error_map = np.abs(target - output).mean(axis=-1)  # Grayscale error

    axes[i, 0].imshow(c_in)
    axes[i, 0].set_title("Corrupted Input")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(target)
    axes[i, 1].set_title("Clean Target")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(output)
    axes[i, 2].set_title("VAE Reconstruction")
    axes[i, 2].axis("off")

    im_err = axes[i, 3].imshow(error_map, cmap="inferno", vmin=0, vmax=0.5)
    axes[i, 3].set_title("Absolute Error Map")
    axes[i, 3].axis("off")
    fig.colorbar(im_err, ax=axes[i, 3], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
"""),
        md_cell("""## 10. Transition Gate Checklist (Plan 6 §1.4)
Before proceeding to `task1_vae_optuna.ipynb`, confirm that all criteria pass:
- [ ] Loss curves decrease reliably with no NaNs.
- [ ] Reconstructions visually reasonable across all 4 conditions (clean, salt, blur, occlusion).
- [ ] KL term is not collapsed to ~0 throughout training.
- [ ] Checkpoint save/reload verified (`task1_universal_ae_best.pt` exists and loads cleanly).
""")
    ]
    return make_notebook(cells)


# ============================================================================
# Notebook 2: task1_vae_optuna.ipynb
# ============================================================================
def build_task1_vae_optuna():
    cells = [
        md_cell("""# Task 1: Universal VAE Hyperparameter Optimization (Optuna)

This notebook executes the **Optuna Hyperparameter Search** for Task 1 as detailed in **Plan 4 §3** and **Plan 6 §1.5**.

### Key Points:
- Uses SQLite persistent storage (`task1_study.db`) for multi-device portability.
- Uses **TPE sampler** and **MedianPruner** to terminate underperforming trials early.
- Runs trials with reduced epochs (15-20% of full budget) to conserve compute.
- Retrains the winning configuration on the full training schedule.
- Performs the mandatory **Skip-Connection Ablation** (Plan 6 §1.6) comparing the winning model with vs. without a single skip connection.
"""),
        code_cell("""# 1. Configuration & Optuna Settings
N_TRIALS = 25
OPTUNA_EPOCHS = 15      # Reduced epochs per trial (Plan 7 §Tasks 1 & 2b)
FULL_TRAIN_EPOCHS = 60  # Full schedule retrain for winner (Plan 6 §1.6)

STUDY_NAME = "task1_universal_vae"
STORAGE_DB = "sqlite:///task1_study.db"

USE_WANDB = bool(os.getenv("WANDB_API_KEY"))
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "genai-assignment")
SEED = 42
"""),
        code_cell("""# 2. Imports and Environment Setup
import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
import optuna
from optuna.trial import TrialState
import matplotlib.pyplot as plt

RESEARCH_ROOT = Path("..").resolve()
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from constants import PET_IMAGES_DIR, CHECKPOINTS_MANIFEST, CONFIGS_ROOT
from src.device_utils import get_device, device_report
from src.datasets import PetDataset
from src.models_vae import ConvVAE
from src.train_loop import train_one_epoch_vae, validate_vae, set_seed
from src.checkpoint_utils import create_checkpoint_dict, save_checkpoint

set_seed(SEED)
device = get_device(verbose=True)
print(f"[Optuna] Storage: {STORAGE_DB}, Study: {STUDY_NAME}")
"""),
        code_cell("""# 3. Data Loading
train_dataset = PetDataset(mode="train", corruption_mode="all", seed=SEED)
val_dataset = PetDataset(mode="val", corruption_mode="all", seed=SEED)
print(f"[Data] Train: {len(train_dataset)}, Val: {len(val_dataset)}")
"""),
        code_cell("""# 4. Optuna Objective Function
def objective(trial: optuna.Trial) -> float:
    # Hyperparameter search space (Plan 7 §Tasks 1 & 2b)
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    latent_dim = trial.suggest_categorical("latent_dim", [64, 128, 256])
    base_channels = trial.suggest_categorical("base_channels", [32, 64])
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    alpha = trial.suggest_float("alpha", 0.5, 0.95)
    beta = trial.suggest_float("beta", 1e-4, 1e-1, log=True)

    # Data loaders with sampled batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = ConvVAE(
        in_channels=3,
        base_channels=base_channels,
        latent_dim=latent_dim,
        dropout=dropout,
        use_skip=False
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    best_trial_val_loss = float("inf")

    for epoch in range(1, OPTUNA_EPOCHS + 1):
        train_metrics = train_one_epoch_vae(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            alpha=alpha,
            beta=beta,
            scaler=scaler
        )
        val_metrics = validate_vae(
            model=model,
            loader=val_loader,
            device=device,
            alpha=alpha,
            beta=beta
        )
        val_loss = val_metrics["val_loss"]

        if val_loss < best_trial_val_loss:
            best_trial_val_loss = val_loss

        # Report metric for pruning
        trial.report(val_loss, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_trial_val_loss
"""),
        code_cell("""# 5. Run Optuna Study Optimization
pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
study = optuna.create_study(
    study_name=STUDY_NAME,
    storage=STORAGE_DB,
    load_if_exists=True,
    direction="minimize",
    pruner=pruner
)

print(f"[Optuna] Starting optimization with {N_TRIALS} trials...")
study.optimize(objective, n_trials=N_TRIALS, timeout=None)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print(f"[Optuna Study Finished]")
print(f"  Total trials: {len(study.trials)}")
print(f"  Complete: {len(complete_trials)}, Pruned: {len(pruned_trials)}")
print(f"  Best trial: #{study.best_trial.number}")
print(f"  Best val loss: {study.best_value:.4f}")
print("  Best parameters:")
for k, v in study.best_params.items():
    print(f"    {k}: {v}")
"""),
        code_cell("""# 6. Save Best Config to YAML
best_config_path = CONFIGS_ROOT / "task1_best_config.yaml"
best_config_data = {
    "task": "task1_universal_vae",
    "source_notebook": "task1_vae_optuna.ipynb",
    "best_trial_number": int(study.best_trial.number),
    "best_value": float(study.best_value),
    "params": study.best_params,
}

with open(best_config_path, "w") as f:
    yaml.dump(best_config_data, f, indent=2)

print(f"[Config] Exported best hyperparameters to {best_config_path}")
"""),
        code_cell("""# 7. Retrain Winning Model on Full Schedule (Plan 6 §1.6)
p = study.best_params
batch_size = p.get("batch_size", 32)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

winner_model = ConvVAE(
    in_channels=3,
    base_channels=p["base_channels"],
    latent_dim=p["latent_dim"],
    dropout=p["dropout"],
    use_skip=False
).to(device)

optimizer = torch.optim.Adam(winner_model.parameters(), lr=p["lr"])
scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

best_val_loss = float("inf")
checkpoint_dir = RESEARCH_ROOT / "checkpoints"

print(f"[Full Retrain] Training winner for {FULL_TRAIN_EPOCHS} epochs...")
for epoch in range(1, FULL_TRAIN_EPOCHS + 1):
    train_metrics = train_one_epoch_vae(
        model=winner_model,
        loader=train_loader,
        optimizer=optimizer,
        device=device,
        alpha=p["alpha"],
        beta=p["beta"],
        scaler=scaler
    )
    val_metrics = validate_vae(
        model=winner_model,
        loader=val_loader,
        device=device,
        alpha=p["alpha"],
        beta=p["beta"]
    )
    val_loss = val_metrics["val_loss"]
    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss

    ckpt_dict = create_checkpoint_dict(
        model=winner_model,
        optimizer=optimizer,
        epoch=epoch,
        val_loss=val_loss,
        optuna_trial_number=study.best_trial.number,
        random_seed=SEED,
        extra_metadata=p
    )
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        prefix="task1_universal_ae_final",
        ckpt_dict=ckpt_dict,
        is_best=is_best,
        task_name="task1",
        phase_name="final",
        device_name=device_report()["device_name"]
    )

print(f"[Full Retrain Finished] Best Val Loss: {best_val_loss:.4f}")
"""),
        code_cell("""# 8. Skip-Connection Ablation Test (Plan 6 §1.6)
# Compares the winning architecture without skips vs. with a single high-resolution skip
print("--- Running Skip Connection Ablation ---")

skip_model = ConvVAE(
    in_channels=3,
    base_channels=p["base_channels"],
    latent_dim=p["latent_dim"],
    dropout=p["dropout"],
    use_skip=True  # Single skip connection enabled
).to(device)

opt_skip = torch.optim.Adam(skip_model.parameters(), lr=p["lr"])
ablation_epochs = 15

for epoch in range(1, ablation_epochs + 1):
    train_one_epoch_vae(skip_model, train_loader, opt_skip, device, alpha=p["alpha"], beta=p["beta"])

no_skip_eval = validate_vae(winner_model, val_loader, device, alpha=p["alpha"], beta=p["beta"])
skip_eval = validate_vae(skip_model, val_loader, device, alpha=p["alpha"], beta=p["beta"])

print("--- Ablation Results ---")
print(f"No-Skip Model -> Val Loss: {no_skip_eval['val_loss']:.4f}, SSIM: {no_skip_eval['val_ssim']:.4f}")
print(f"With-Skip Model -> Val Loss: {skip_eval['val_loss']:.4f}, SSIM: {skip_eval['val_ssim']:.4f}")
print("Use this quantitative comparison in your report's architecture ablation section!")
""")
    ]
    return make_notebook(cells)


# ============================================================================
# Notebook 3: task1_validation.ipynb (Stub)
# ============================================================================
def build_task1_validation():
    cells = [
        md_cell("""# Task 1: Universal VAE Validation & Evaluation

This notebook evaluates the final trained Task 1 Universal VAE model across all corruption types and severities.
It implements the benchmarking requirements from **Plan 6 §1.7**.
"""),
        code_cell("""# Imports
import sys, os, json
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

RESEARCH_ROOT = Path("..").resolve()
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from constants import PET_IMAGES_DIR, CHECKPOINTS_MANIFEST
from src.device_utils import get_device
from src.datasets import PetDataset
from src.models_vae import ConvVAE
from src.checkpoint_utils import load_checkpoint
from src.corruptions import BENCHMARK_SEVERITIES, apply_corruption
from src.losses import compute_ssim
"""),
        code_cell("""# Configuration
CHECKPOINT_PATH = RESEARCH_ROOT / "checkpoints" / "task1_universal_ae_final_best.pt"
DEVICE = get_device()
"""),
        md_cell("## 1. Model Loading"),
        code_cell("""# TODO: Initialize ConvVAE and load weights from CHECKPOINT_PATH
# model = ConvVAE(...).to(DEVICE)
# load_checkpoint(CHECKPOINT_PATH, model, device=DEVICE)
# model.eval()
"""),
        md_cell("## 2. Quantitative Evaluation: PSNR, SSIM, and MSE Grid (Plan 6 §1.7)"),
        code_cell("""# TODO: Compute PSNR and SSIM across all 4 corruption types (clean, salt, blur, occlusion)
# and their 3 benchmark severities (low, med, high).
# Report results in a structured table for the report.
"""),
        md_cell("## 3. Visual Restorations & Absolute Error Maps"),
        code_cell("""# TODO: Generate a grid of >=12 representative examples:
# [Clean Target | Corrupted Input | VAE Output | Absolute Error Map]
"""),
        md_cell("## 4. Posterior Collapse Verification"),
        code_cell("""# TODO: Compute distribution of mu and logvar across the validation set.
# Verify that KL divergence is non-zero and the latent space is actively utilized.
"""),
        md_cell("## 5. Failure Mode Analysis"),
        code_cell("""# TODO: Identify and display >= 4 representative failure cases (e.g. dense occlusion, high-sigma blur).
# Provide written hypotheses explaining the failure mechanisms.
""")
    ]
    return make_notebook(cells)


# ============================================================================
# Notebook 4: task2a_classifier_simple.ipynb
# ============================================================================
def build_task2a_classifier_simple():
    cells = [
        md_cell("""# Task 2a: Corruption Classifier (Simple Baseline)

This notebook implements the baseline CNN classifier for Task 2a as described in **Plan 5 §Task 2a** and **Plan 6 §2a**.

### Target:
Classify input images into one of 4 corruption categories:
- `0`: Clean
- `1`: Salt & Pepper
- `2`: Gaussian Blur
- `3`: Occlusion
"""),
        code_cell("""# 1. Configuration & Hyperparameters
TINY_RUN = False
TINY_SIZE = 50

BATCH_SIZE = 64
LR = 1e-3
BASE_CHANNELS = 32
DROPOUT = 0.2
EPOCHS = 2 if TINY_RUN else 20
SEED = 42

# Experiment tracking
USE_WANDB = bool(os.getenv("WANDB_API_KEY"))
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "genai-assignment")
WANDB_RUN_NAME = "task2a_classifier_simple"
"""),
        code_cell("""# 2. Imports and Environment Setup
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

RESEARCH_ROOT = Path("..").resolve()
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from constants import PET_IMAGES_DIR, CHECKPOINTS_MANIFEST
from src.device_utils import get_device, device_report
from src.datasets import PetDataset
from src.models_classifier import CorruptionClassifier
from src.train_loop import train_one_epoch_classifier, validate_classifier, set_seed
from src.checkpoint_utils import create_checkpoint_dict, save_checkpoint

set_seed(SEED)
device = get_device(verbose=True)
"""),
        code_cell("""# 3. Data Loading
mode_train = "tiny" if TINY_RUN else "train"
mode_val = "tiny" if TINY_RUN else "val"

train_dataset = PetDataset(mode=mode_train, corruption_mode="label", tiny_size=TINY_SIZE, seed=SEED)
val_dataset = PetDataset(mode=mode_val, corruption_mode="label", tiny_size=TINY_SIZE, seed=SEED)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"[Data] Train: {len(train_dataset)}, Val: {len(val_dataset)}")
print(f"[Classes] {PetDataset.CLASSES}")
"""),
        code_cell("""# 4. Inspect Sample Batches
sample_imgs, sample_labels = next(iter(train_loader))

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for i in range(4):
    img = sample_imgs[i].permute(1, 2, 0).cpu().numpy().clip(0, 1)
    label_name = PetDataset.CLASSES[sample_labels[i].item()]
    axes[i].imshow(img)
    axes[i].set_title(f"Class: {label_name} ({sample_labels[i].item()})")
    axes[i].axis("off")

plt.tight_layout()
plt.show()
"""),
        code_cell("""# 5. Model & Optimizer Initialization
model = CorruptionClassifier(
    in_channels=3,
    base_channels=BASE_CHANNELS,
    dropout=DROPOUT,
    num_classes=4
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[Model] CorruptionClassifier initialized with {total_params:,} parameters.")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None
"""),
        code_cell("""# 6. Training & Validation Loop
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0
checkpoint_dir = RESEARCH_ROOT / "checkpoints"

print(f"--- Training Classifier for {EPOCHS} Epochs ---")
for epoch in range(1, EPOCHS + 1):
    train_metrics = train_one_epoch_classifier(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        device=device,
        scaler=scaler
    )
    val_metrics = validate_classifier(
        model=model,
        loader=val_loader,
        device=device
    )

    for k in ["train_loss", "train_acc"]:
        history[k].append(train_metrics[k])
    for k in ["val_loss", "val_acc"]:
        history[k].append(val_metrics[k])

    val_acc = val_metrics["val_acc"]
    is_best = val_acc > best_val_acc
    if is_best:
        best_val_acc = val_acc

    ckpt_dict = create_checkpoint_dict(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        val_loss=val_metrics["val_loss"],
        random_seed=SEED,
        extra_metadata={"val_acc": val_acc}
    )
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        prefix="task2_classifier",
        ckpt_dict=ckpt_dict,
        is_best=is_best,
        task_name="task2a",
        phase_name="simple",
        device_name=device_report()["device_name"]
    )

    print(
        f"Epoch [{epoch:02d}/{EPOCHS:02d}] "
        f"Train Loss: {train_metrics['train_loss']:.4f}, Train Acc: {train_metrics['train_acc']:.2%} | "
        f"Val Loss: {val_metrics['val_loss']:.4f}, Val Acc: {val_acc:.2%} {'*Best*' if is_best else ''}"
    )

print("--- Training Completed ---")
"""),
        code_cell("""# 7. Confusion Matrix & Classification Metrics
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

val_results = validate_classifier(model, val_loader, device)
y_true = val_results["labels"]
y_pred = val_results["preds"]

print(classification_report(y_true, y_pred, target_names=PetDataset.CLASSES))

cm = confusion_matrix(y_true, y_pred, normalize="true")
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=PetDataset.CLASSES, yticklabels=PetDataset.CLASSES)
plt.title("Normalized Confusion Matrix")
plt.ylabel("True Class")
plt.xlabel("Predicted Class")
plt.show()
"""),
        md_cell("""## 8. Transition Gate Checklist (Plan 6 §2a.4)
- [ ] Validation accuracy clearly above random chance (>60% sanity threshold).
- [ ] No single class collapsing to ~0% recall.
- [ ] Checkpoint `task2_classifier_best.pt` saved and loadable for Task 3 initialization.
""")
    ]
    return make_notebook(cells)


# ============================================================================
# Notebook 5: task2a_classifier_optuna.ipynb
# ============================================================================
def build_task2a_classifier_optuna():
    cells = [
        md_cell("""# Task 2a: Corruption Classifier Optuna Search

Hyperparameter tuning for the Task 2a CNN classifier (Plan 6 §2a.5 & Plan 7).
Searches learning rate, batch size, channel depth, dropout, and weight decay.
"""),
        code_cell("""# 1. Config
N_TRIALS = 20
OPTUNA_EPOCHS = 15
FULL_TRAIN_EPOCHS = 35
STUDY_NAME = "task2a_classifier"
STORAGE_DB = "sqlite:///task2a_study.db"
SEED = 42
"""),
        code_cell("""# 2. Imports & Setup
import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
import optuna

RESEARCH_ROOT = Path("..").resolve()
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from constants import CONFIGS_ROOT
from src.device_utils import get_device, device_report
from src.datasets import PetDataset
from src.models_classifier import CorruptionClassifier
from src.train_loop import train_one_epoch_classifier, validate_classifier, set_seed
from src.checkpoint_utils import create_checkpoint_dict, save_checkpoint

set_seed(SEED)
device = get_device()
"""),
        code_cell("""# 3. Data Loading
train_dataset = PetDataset(mode="train", corruption_mode="label", seed=SEED)
val_dataset = PetDataset(mode="val", corruption_mode="label", seed=SEED)
"""),
        code_cell("""# 4. Optuna Objective
def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    base_channels = trial.suggest_categorical("base_channels", [16, 32, 64])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = CorruptionClassifier(
        base_channels=base_channels,
        dropout=dropout,
        num_classes=4
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_acc = 0.0

    for epoch in range(1, OPTUNA_EPOCHS + 1):
        train_one_epoch_classifier(model, train_loader, optimizer, device)
        val_res = validate_classifier(model, val_loader, device)
        acc = val_res["val_acc"]
        if acc > best_acc:
            best_acc = acc
        trial.report(acc, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_acc
"""),
        code_cell("""# 5. Run Optimization
study = optuna.create_study(
    study_name=STUDY_NAME,
    storage=STORAGE_DB,
    load_if_exists=True,
    direction="maximize",
    pruner=optuna.pruners.MedianPruner()
)
study.optimize(objective, n_trials=N_TRIALS)

print(f"Best Trial: #{study.best_trial.number} with Accuracy: {study.best_value:.2%}")
print(study.best_params)
"""),
        code_cell("""# 6. Save Best Config & Retrain
best_config_path = CONFIGS_ROOT / "task2a_best_config.yaml"
with open(best_config_path, "w") as f:
    yaml.dump({
        "task": "task2a_corruption_classifier",
        "best_trial_number": int(study.best_trial.number),
        "best_accuracy": float(study.best_value),
        "params": study.best_params,
    }, f, indent=2)

print(f"Saved to {best_config_path}")
""")
    ]
    return make_notebook(cells)


# ============================================================================
# Notebook 6: task2a_validation.ipynb (Stub)
# ============================================================================
def build_task2a_validation():
    cells = [
        md_cell("""# Task 2a: Corruption Classifier Validation

Validation and error analysis for the Task 2a classifier (Plan 6 §2a.6).
"""),
        code_cell("""# Imports
import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

RESEARCH_ROOT = Path("..").resolve()
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from constants import CHECKPOINTS_MANIFEST
from src.device_utils import get_device
from src.datasets import PetDataset
from src.models_classifier import CorruptionClassifier
from src.checkpoint_utils import load_checkpoint
"""),
        code_cell("""# Configuration
CHECKPOINT_PATH = RESEARCH_ROOT / "checkpoints" / "task2_classifier_best.pt"
DEVICE = get_device()
"""),
        md_cell("## 1. Load Trained Classifier"),
        code_cell("""# TODO: Load CorruptionClassifier model and weights
# model = CorruptionClassifier().to(DEVICE)
# load_checkpoint(CHECKPOINT_PATH, model, device=DEVICE)
# model.eval()
"""),
        md_cell("## 2. Quantitative Evaluation: Accuracy, Macro F1, Per-Class Metrics"),
        code_cell("""# TODO: Run model on test set and print full classification_report
"""),
        md_cell("## 3. Normalized Confusion Matrix"),
        code_cell("""# TODO: Plot 4x4 normalized confusion matrix heatmap
"""),
        md_cell("## 4. Failure Mode Analysis (Misclassified Corruptions)"),
        code_cell("""# TODO: Analyze class-pair confusion (e.g. low-severity blur vs clean)
# Connect findings to the Roy et al. citation from Plan 5.
""")
    ]
    return make_notebook(cells)


# ============================================================================
# Notebook 7: task2b_specialists_simple.ipynb
# ============================================================================
def build_task2b_specialists_simple():
    cells = [
        md_cell("""# Task 2b: Specialist Restoration Autoencoders (Simple Baseline)

This notebook trains the **three specialist restoration autoencoders** independently:
1. **Salt-and-Pepper Specialist**
2. **Gaussian Blur Specialist**
3. **Occlusion Specialist**

Each specialist is trained strictly on its designated corruption domain (Plan 6 §2b.1).
"""),
        code_cell("""# 1. Config
BATCH_SIZE = 32
LR = 1e-3
LATENT_DIM = 128
BASE_CHANNELS = 32
ALPHA = 0.8
BETA = 0.001
EPOCHS = 15
SEED = 42
"""),
        code_cell("""# 2. Imports & Setup
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

RESEARCH_ROOT = Path("..").resolve()
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from src.device_utils import get_device, device_report
from src.datasets import PetDataset
from src.models_vae import ConvVAE
from src.train_loop import train_one_epoch_vae, validate_vae, set_seed
from src.checkpoint_utils import create_checkpoint_dict, save_checkpoint

set_seed(SEED)
device = get_device(verbose=True)
checkpoint_dir = RESEARCH_ROOT / "checkpoints"
"""),
        code_cell("""# 3. Specialist Training Function
def train_specialist(corruption_type: str, epochs: int = EPOCHS):
    print(f"\\n==============================================")
    print(f"   Training Specialist: {corruption_type.upper()}")
    print(f"==============================================")
    
    train_ds = PetDataset(mode="train", corruption_mode=corruption_type, seed=SEED)
    val_ds = PetDataset(mode="val", corruption_mode=corruption_type, seed=SEED)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    model = ConvVAE(
        in_channels=3,
        base_channels=BASE_CHANNELS,
        latent_dim=LATENT_DIM,
        dropout=0.1,
        use_skip=False
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None
    best_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        tr = train_one_epoch_vae(model, train_loader, optimizer, device, ALPHA, BETA, scaler)
        vl = validate_vae(model, val_loader, device, ALPHA, BETA)
        
        is_best = vl["val_loss"] < best_loss
        if is_best:
            best_loss = vl["val_loss"]
            
        ckpt = create_checkpoint_dict(model, optimizer, epoch, vl["val_loss"], random_seed=SEED)
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            prefix=f"task2b_specialist_{corruption_type}",
            ckpt_dict=ckpt,
            is_best=is_best,
            task_name=f"task2b_{corruption_type}",
            phase_name="simple",
            device_name=device_report()["device_name"]
        )
        
        print(f"[{corruption_type.upper()} Epoch {epoch:02d}] Val Loss: {vl['val_loss']:.4f}, SSIM: {vl['val_ssim']:.4f} {'*' if is_best else ''}")
        
    return model
"""),
        code_cell("""# 4. Train All Three Specialists
salt_model = train_specialist("salt", epochs=EPOCHS)
blur_model = train_specialist("blur", epochs=EPOCHS)
occl_model = train_specialist("occlusion", epochs=EPOCHS)
print("\\nAll 3 specialists trained and checkpoints saved successfully!")
"""),
        code_cell("""# 5. Visual Spot-Check
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
corruptions = ["salt", "blur", "occlusion"]
models = [salt_model, blur_model, occl_model]

for idx, (ctype, mdl) in enumerate(zip(corruptions, models)):
    mdl.eval()
    ds = PetDataset(mode="val", corruption_mode=ctype, seed=SEED)
    corrupted, clean = ds[0]
    with torch.no_grad():
        recon, _, _ = mdl(corrupted.unsqueeze(0).to(device))
        
    axes[idx, 0].imshow(corrupted.permute(1, 2, 0).clip(0, 1))
    axes[idx, 0].set_title(f"{ctype.capitalize()} Input")
    axes[idx, 0].axis("off")
    
    axes[idx, 1].imshow(recon.squeeze(0).cpu().permute(1, 2, 0).clip(0, 1))
    axes[idx, 1].set_title("Specialist Recon")
    axes[idx, 1].axis("off")
    
    axes[idx, 2].imshow(clean.permute(1, 2, 0).clip(0, 1))
    axes[idx, 2].set_title("Clean Target")
    axes[idx, 2].axis("off")

plt.tight_layout()
plt.show()
""")
    ]
    return make_notebook(cells)


# ============================================================================
# Notebook 8: task2b_specialists_optuna.ipynb
# ============================================================================
def build_task2b_specialists_optuna():
    cells = [
        md_cell("""# Task 2b: Specialist Shared Architecture Search (Optuna)

Conducts a shared architecture search on representative corruption data (Plan 6 §2b.2).
The locked winning architecture is then trained across all three specialists.
"""),
        code_cell("""# 1. Config
N_TRIALS = 20
OPTUNA_EPOCHS = 15
FULL_EPOCHS = 50
STUDY_NAME = "task2b_shared_architecture"
STORAGE_DB = "sqlite:///task2b_study.db"
SEED = 42
"""),
        code_cell("""# 2. Imports & Setup
import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
import optuna

RESEARCH_ROOT = Path("..").resolve()
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from constants import CONFIGS_ROOT
from src.device_utils import get_device, device_report
from src.datasets import PetDataset
from src.models_vae import ConvVAE
from src.train_loop import train_one_epoch_vae, validate_vae, set_seed
from src.checkpoint_utils import create_checkpoint_dict, save_checkpoint

set_seed(SEED)
device = get_device()
"""),
        code_cell("""# 3. Shared Architecture Objective
# Conducted on Salt & Pepper representative subset (Plan 6 §2b.2)
train_dataset = PetDataset(mode="train", corruption_mode="salt", seed=SEED)
val_dataset = PetDataset(mode="val", corruption_mode="salt", seed=SEED)

def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    latent_dim = trial.suggest_categorical("latent_dim", [64, 128, 256])
    base_channels = trial.suggest_categorical("base_channels", [32, 64])
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    alpha = trial.suggest_float("alpha", 0.5, 0.95)
    beta = trial.suggest_float("beta", 1e-4, 1e-1, log=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = ConvVAE(
        in_channels=3,
        base_channels=base_channels,
        latent_dim=latent_dim,
        dropout=dropout,
        use_skip=False
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")

    for epoch in range(1, OPTUNA_EPOCHS + 1):
        train_one_epoch_vae(model, train_loader, optimizer, device, alpha, beta)
        vl = validate_vae(model, val_loader, device, alpha, beta)
        if vl["val_loss"] < best_loss:
            best_loss = vl["val_loss"]
        trial.report(vl["val_loss"], step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_loss
"""),
        code_cell("""# 4. Run Study & Export Configs
study = optuna.create_study(
    study_name=STUDY_NAME,
    storage=STORAGE_DB,
    load_if_exists=True,
    direction="minimize",
    pruner=optuna.pruners.MedianPruner()
)
study.optimize(objective, n_trials=N_TRIALS)

print(f"Winning Shared Architecture Params: {study.best_params}")

for ctype in ["salt", "blur", "occlusion"]:
    cfg_path = CONFIGS_ROOT / f"task2b_{ctype}_best_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump({
            "task": f"task2b_specialist_{ctype}",
            "params": study.best_params,
            "best_val_loss": float(study.best_value)
        }, f, indent=2)
print("Saved configs for all 3 specialists.")
""")
    ]
    return make_notebook(cells)


# ============================================================================
# Notebook 9: task2b_validation.ipynb (Stub)
# ============================================================================
def build_task2b_validation():
    cells = [
        md_cell("""# Task 2b: Specialist Autoencoders Validation & Routing Comparison

Evaluates the 3 specialists and conducts the **Oracle vs. Predicted Routing Test** (Plan 6 §2b.5).
"""),
        code_cell("""# Imports
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

RESEARCH_ROOT = Path("..").resolve()
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from src.device_utils import get_device
from src.datasets import PetDataset
from src.models_vae import ConvVAE
from src.models_classifier import CorruptionClassifier
from src.checkpoint_utils import load_checkpoint
"""),
        code_cell("""# Configuration
SALT_CKPT = RESEARCH_ROOT / "checkpoints" / "task2b_specialist_salt_best.pt"
BLUR_CKPT = RESEARCH_ROOT / "checkpoints" / "task2b_specialist_blur_best.pt"
OCCL_CKPT = RESEARCH_ROOT / "checkpoints" / "task2b_specialist_occlusion_best.pt"
CLS_CKPT = RESEARCH_ROOT / "checkpoints" / "task2_classifier_best.pt"
DEVICE = get_device()
"""),
        md_cell("## 1. Load All Specialists and Classifier"),
        code_cell("""# TODO: Load all 3 specialist ConvVAE models and the classifier
"""),
        md_cell("## 2. Per-Specialist Performance across Benchmark Severities"),
        code_cell("""# TODO: Benchmark each specialist on its respective domain across low/med/high severities
"""),
        md_cell("## 3. Oracle Routing vs. Predicted Hard-Routing Comparison (Plan 6 §2b.5)"),
        code_cell("""# TODO: Compare restoration metrics under:
# a) Oracle routing (true corruption label used to select specialist)
# b) Predicted routing (Task 2a classifier selects specialist)
# Quantify performance drop caused by classifier mispredictions.
"""),
        md_cell("## 4. Visual Restorations & Misrouting Failure Cases"),
        code_cell("""# TODO: Display cases where correct routing succeeds vs cases where classifier misprediction causes visible failure.
""")
    ]
    return make_notebook(cells)


# ============================================================================
# Notebook 10: task3_moe_simple.ipynb
# ============================================================================
def build_task3_moe_simple():
    cells = [
        md_cell("""# Task 3: Soft Mixture-of-Experts Restoration (Simple Baseline)

Implements the soft-routed Mixture-of-Experts framework (Plan 5 §Task 3 & Plan 6 §3).
Reuses Task 2a's classifier as the gating network and Task 2b's specialists as experts.

### Stages:
1. **Warm-up**: Freeze experts, train gate only (short schedule, e.g. 5 epochs).
2. **Joint Fine-Tuning**: Unfreeze experts, train with joint loss (reconstruction + classification + load balance regularizer).
"""),
        code_cell("""# 1. Config
WARMUP_EPOCHS = 5
FINETUNE_EPOCHS = 15
WARMUP_LR = 1e-3
FINETUNE_LR = 1e-4
BATCH_SIZE = 32
TEMPERATURE = 1.0

# Joint loss lambdas (Plan 6 §3.3)
LAMBDAS = {
    "lambda1": 0.8,   # L1 reconstruction
    "lambda2": 0.2,   # (1 - SSIM)
    "lambda3": 0.1,   # Classification cross-entropy
    "lambda4": 0.01,  # Gating balance regularizer
}
SEED = 42
"""),
        code_cell("""# 2. Imports & Setup
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

RESEARCH_ROOT = Path("..").resolve()
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from src.device_utils import get_device, device_report
from src.datasets import PetDataset
from src.models_vae import ConvVAE
from src.models_classifier import CorruptionClassifier, SoftMoERestorer
from src.train_loop import train_one_epoch_moe, validate_moe, set_seed
from src.checkpoint_utils import create_checkpoint_dict, save_checkpoint, load_checkpoint

set_seed(SEED)
device = get_device(verbose=True)
checkpoint_dir = RESEARCH_ROOT / "checkpoints"
"""),
        code_cell("""# 3. Load Trained Sub-networks (Task 2a & 2b Checkpoints)
gate = CorruptionClassifier().to(device)
sp_salt = ConvVAE().to(device)
sp_blur = ConvVAE().to(device)
sp_occl = ConvVAE().to(device)

# Load existing weights if checkpoints exist
for name, m, p in [
    ("gate", gate, checkpoint_dir / "task2_classifier_best.pt"),
    ("salt", sp_salt, checkpoint_dir / "task2b_specialist_salt_best.pt"),
    ("blur", sp_blur, checkpoint_dir / "task2b_specialist_blur_best.pt"),
    ("occl", sp_occl, checkpoint_dir / "task2b_specialist_occlusion_best.pt"),
]:
    if p.exists():
        load_checkpoint(p, m, device=device)
    else:
        print(f"[Warning] {p.name} not found. Initializing {name} with random weights for testing.")

moe = SoftMoERestorer(
    gate=gate,
    specialist_salt=sp_salt,
    specialist_blur=sp_blur,
    specialist_occlusion=sp_occl,
    temperature=TEMPERATURE
).to(device)
"""),
        code_cell("""# 4. Data Loading (PetDataset with ground-truth label for joint loss)
train_dataset = PetDataset(mode="train", corruption_mode="all", return_label=True, seed=SEED)
val_dataset = PetDataset(mode="val", corruption_mode="all", return_label=True, seed=SEED)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
"""),
        code_cell("""# 5. Stage 1: Warm-Up Training (Gate Only)
print("--- Stage 1: Gating Warm-Up (Experts Frozen) ---")
# Freeze experts
for p in moe.specialist_salt.parameters(): p.requires_grad = False
for p in moe.specialist_blur.parameters(): p.requires_grad = False
for p in moe.specialist_occlusion.parameters(): p.requires_grad = False
for p in moe.gate.parameters(): p.requires_grad = True

opt_warmup = torch.optim.Adam(moe.gate.parameters(), lr=WARMUP_LR)

for epoch in range(1, WARMUP_EPOCHS + 1):
    tr = train_one_epoch_moe(moe, train_loader, opt_warmup, device, LAMBDAS)
    vl = validate_moe(moe, val_loader, device, LAMBDAS)
    print(f"[Warmup {epoch:02d}] Val Loss: {vl['val_loss']:.4f}, SSIM: {vl['val_ssim']:.4f}, Gate Usage: {np.round(vl['avg_expert_weights'], 3)}")
"""),
        code_cell("""# 6. Stage 2: Joint Fine-Tuning (All Parameters Unfrozen)
print("--- Stage 2: Joint Fine-Tuning ---")
for p in moe.parameters():
    p.requires_grad = True

opt_joint = torch.optim.Adam(moe.parameters(), lr=FINETUNE_LR)
best_val_loss = float("inf")

for epoch in range(1, FINETUNE_EPOCHS + 1):
    tr = train_one_epoch_moe(moe, train_loader, opt_joint, device, LAMBDAS)
    vl = validate_moe(moe, val_loader, device, LAMBDAS)
    
    is_best = vl["val_loss"] < best_val_loss
    if is_best:
        best_val_loss = vl["val_loss"]
        
    ckpt = create_checkpoint_dict(moe, opt_joint, epoch, vl["val_loss"], random_seed=SEED)
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        prefix="task3_soft_moe",
        ckpt_dict=ckpt,
        is_best=is_best,
        task_name="task3",
        phase_name="simple",
        device_name=device_report()["device_name"]
    )
    print(f"[Joint Epoch {epoch:02d}] Val Loss: {vl['val_loss']:.4f} (SSIM: {vl['val_ssim']:.4f}) {'*Best*' if is_best else ''}")
""")
    ]
    return make_notebook(cells)


# ============================================================================
# Notebook 11: task3_moe_optuna.ipynb
# ============================================================================
def build_task3_moe_optuna():
    cells = [
        md_cell("""# Task 3: Soft MoE Optuna Search

Hyperparameter optimization for soft Mixture-of-Experts (Plan 6 §3.5).
Tunes fine-tuning learning rate, softmax temperature, classification weight $\\lambda_3$, and balance weight $\\lambda_4$.
"""),
        code_cell("""# 1. Config
N_TRIALS = 15
OPTUNA_EPOCHS = 12
STUDY_NAME = "task3_soft_moe"
STORAGE_DB = "sqlite:///task3_study.db"
SEED = 42
"""),
        code_cell("""# 2. Imports & Setup
import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
import optuna

RESEARCH_ROOT = Path("..").resolve()
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from constants import CONFIGS_ROOT
from src.device_utils import get_device, device_report
from src.datasets import PetDataset
from src.models_classifier import CorruptionClassifier, SoftMoERestorer
from src.models_vae import ConvVAE
from src.train_loop import train_one_epoch_moe, validate_moe, set_seed

set_seed(SEED)
device = get_device()
"""),
        code_cell("""# 3. Objective Function
train_dataset = PetDataset(mode="train", corruption_mode="all", return_label=True, seed=SEED)
val_dataset = PetDataset(mode="val", corruption_mode="all", return_label=True, seed=SEED)

def objective(trial: optuna.Trial) -> float:
    finetune_lr = trial.suggest_float("finetune_lr", 1e-5, 1e-4, log=True)
    temp = trial.suggest_float("temperature", 0.5, 2.0)
    lambda3 = trial.suggest_float("lambda3", 0.01, 0.5)
    lambda4 = trial.suggest_float("lambda4", 1e-3, 1e-1, log=True)

    lambdas = {"lambda1": 0.8, "lambda2": 0.2, "lambda3": lambda3, "lambda4": lambda4}

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    moe = SoftMoERestorer(
        gate=CorruptionClassifier(),
        specialist_salt=ConvVAE(),
        specialist_blur=ConvVAE(),
        specialist_occlusion=ConvVAE(),
        temperature=temp
    ).to(device)

    opt = torch.optim.Adam(moe.parameters(), lr=finetune_lr)
    best_loss = float("inf")

    for epoch in range(1, OPTUNA_EPOCHS + 1):
        train_one_epoch_moe(moe, train_loader, opt, device, lambdas)
        vl = validate_moe(moe, val_loader, device, lambdas)
        if vl["val_loss"] < best_loss:
            best_loss = vl["val_loss"]
        trial.report(vl["val_loss"], step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_loss
"""),
        code_cell("""# 4. Optimization & Export
study = optuna.create_study(
    study_name=STUDY_NAME,
    storage=STORAGE_DB,
    load_if_exists=True,
    direction="minimize",
    pruner=optuna.pruners.MedianPruner()
)
study.optimize(objective, n_trials=N_TRIALS)

cfg_path = CONFIGS_ROOT / "task3_best_config.yaml"
with open(cfg_path, "w") as f:
    yaml.dump({
        "task": "task3_soft_moe",
        "best_trial_number": int(study.best_trial.number),
        "best_value": float(study.best_value),
        "params": study.best_params,
    }, f, indent=2)

print(f"Best MoE config saved to {cfg_path}")
""")
    ]
    return make_notebook(cells)


# ============================================================================
# Notebook 12: task3_validation.ipynb (Stub)
# ============================================================================
def build_task3_validation():
    cells = [
        md_cell("""# Task 3: Soft Mixture-of-Experts Validation & Analysis

Validation and quantitative comparison of the soft MoE model vs. Task 1 and Task 2b (Plan 6 §3.6).
"""),
        code_cell("""# Imports
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RESEARCH_ROOT = Path("..").resolve()
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from src.device_utils import get_device
from src.datasets import PetDataset
from src.models_classifier import SoftMoERestorer, CorruptionClassifier
from src.models_vae import ConvVAE
from src.checkpoint_utils import load_checkpoint
"""),
        code_cell("""# Configuration
MOE_CKPT = RESEARCH_ROOT / "checkpoints" / "task3_soft_moe_best.pt"
DEVICE = get_device()
"""),
        md_cell("## 1. Load Trained Soft MoE Pipeline"),
        code_cell("""# TODO: Load SoftMoERestorer model from MOE_CKPT
"""),
        md_cell("## 2. Quantitative Performance vs Task 1 and Task 2b"),
        code_cell("""# TODO: Generate comparison table (PSNR, SSIM) between:
# - Task 1 Universal VAE
# - Task 2b Hard-Routed Specialists
# - Task 3 Soft Mixture-of-Experts
"""),
        md_cell("## 3. Expert Routing Weight Heatmap (Plan 6 §3.6)"),
        code_cell("""# TODO: Compute average routing weights allocated to each expert across each true corruption category.
# Plot 4x4 heatmap: (True Corruption vs Branch Weight).
"""),
        md_cell("## 4. Multi-Expert Blending Visualizations"),
        code_cell("""# TODO: Display sample images where >=2 experts are actively blended.
""")
    ]
    return make_notebook(cells)


# ============================================================================
# Notebook 13: task4_gan_simple.ipynb
# ============================================================================
def build_task4_gan_simple():
    cells = [
        md_cell("""# Task 4: Style-Conditioned Face-to-Sketch (cGAN Simple Baseline)

Implements the **Style-Conditioned pix2pix conditional GAN** for Task 4 (Plan 5 §Task 4 & Plan 6 §4).

### Key Elements:
- **Dataset**: FS2K paired photo-sketch dataset (3 style categories).
- **Normalization**: Range `[-1, 1]`.
- **Generator**: U-Net with skip connections and learned style embedding.
- **Discriminator**: 70x70 PatchGAN discriminator with gradient clipping.
- **Objective**: Adversarial BCE loss + $\\lambda_{L1} \\cdot \\text{L1}$.
"""),
        code_cell("""# 1. Config
TINY_RUN = False
TINY_SIZE = 20

BATCH_SIZE = 8
G_LR = 2e-4
D_LR = 2e-4
BETAS = (0.5, 0.999)  # Standard pix2pix GAN betas (Plan 7)
BASE_CHANNELS = 64
EMBED_DIM = 8
LAMBDA_L1 = 100.0
EPOCHS = 2 if TINY_RUN else 25
SEED = 42
"""),
        code_cell("""# 2. Imports & Setup
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

RESEARCH_ROOT = Path("..").resolve()
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from constants import FS2K_ROOT
from src.device_utils import get_device, device_report
from src.datasets import FS2KDataset
from src.models_gan import UNetGenerator, PatchGANDiscriminator
from src.train_loop import train_one_epoch_gan, validate_gan, set_seed
from src.checkpoint_utils import create_checkpoint_dict, save_checkpoint

set_seed(SEED)
device = get_device(verbose=True)
checkpoint_dir = RESEARCH_ROOT / "checkpoints"
"""),
        code_cell("""# 3. Data Loading
mode_train = "tiny" if TINY_RUN else "train"
mode_val = "tiny" if TINY_RUN else "val"

train_dataset = FS2KDataset(mode=mode_train, seed=SEED)
val_dataset = FS2KDataset(mode=mode_val, seed=SEED)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"[FS2K] Train: {len(train_dataset)}, Val: {len(val_dataset)}")
"""),
        code_cell("""# 4. Inspect Sample Training Pairs across Styles
sample_photos, sample_sketches, sample_styles = next(iter(train_loader))

fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for i in range(min(3, len(sample_photos))):
    p = (sample_photos[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
    s = (sample_sketches[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
    st = sample_styles[i].item()
    
    axes[0, i].imshow(p)
    axes[0, i].set_title(f"Photo (Style {st})")
    axes[0, i].axis("off")
    
    axes[1, i].imshow(s)
    axes[1, i].set_title(f"Sketch (Style {st})")
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()
"""),
        code_cell("""# 5. Initialize Generator & Discriminator
netG = UNetGenerator(base_channels=BASE_CHANNELS, embed_dim=EMBED_DIM).to(device)
netD = PatchGANDiscriminator(base_channels=BASE_CHANNELS).to(device)

optG = torch.optim.Adam(netG.parameters(), lr=G_LR, betas=BETAS)
optD = torch.optim.Adam(netD.parameters(), lr=D_LR, betas=BETAS)

print(f"[Generator] Parameters: {sum(p.numel() for p in netG.parameters()):,}")
print(f"[Discriminator] Parameters: {sum(p.numel() for p in netD.parameters()):,}")
"""),
        code_cell("""# 6. GAN Training Loop
best_l1 = float("inf")

print(f"--- Training Conditional GAN ({EPOCHS} Epochs) ---")
for epoch in range(1, EPOCHS + 1):
    metrics = train_one_epoch_gan(
        G=netG, D=netD,
        loader=train_loader,
        opt_G=optG, opt_D=optD,
        device=device,
        lambda_l1=LAMBDA_L1
    )
    val_metrics = validate_gan(netG, val_loader, device)
    
    is_best = val_metrics["val_l1"] < best_l1
    if is_best:
        best_l1 = val_metrics["val_l1"]
        
    ckptG = create_checkpoint_dict(netG, optG, epoch, val_metrics["val_l1"], random_seed=SEED)
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        prefix="task4_generator",
        ckpt_dict=ckptG,
        is_best=is_best,
        task_name="task4_gan",
        phase_name="simple",
        device_name=device_report()["device_name"]
    )
    
    print(
        f"Epoch [{epoch:02d}/{EPOCHS:02d}] "
        f"D_Loss: {metrics['d_loss']:.4f} (Real: {metrics['d_real']:.4f}, Fake: {metrics['d_fake']:.4f}) | "
        f"G_Loss: {metrics['g_loss']:.4f} (Adv: {metrics['g_adv']:.4f}, L1: {metrics['g_l1']:.4f}) | "
        f"Val L1: {val_metrics['val_l1']:.4f} {'*Best*' if is_best else ''}"
    )
"""),
        code_cell("""# 7. Style Conditioning Demonstration: One Photo -> 3 Styles
netG.eval()
test_photo, _, _ = next(iter(val_loader))
test_photo = test_photo[0:1].to(device)

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
p_img = (test_photo[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
axes[0].imshow(p_img)
axes[0].set_title("Input Photo")
axes[0].axis("off")

with torch.no_grad():
    for style_id in range(3):
        st_tensor = torch.tensor([style_id], device=device)
        gen_sketch = netG(test_photo, st_tensor)
        s_img = (gen_sketch[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        axes[style_id + 1].imshow(s_img)
        axes[style_id + 1].set_title(f"Generated Style {style_id}")
        axes[style_id + 1].axis("off")

plt.tight_layout()
plt.show()
"""),
        md_cell("""## 8. Transition Gate Checklist (Plan 6 §4.5)
- [ ] Generated sketches are visually sketch-like.
- [ ] Discriminator and Generator losses are balanced without collapse.
- [ ] Style conditioning produces visibly distinct sketches for the same input photo.
- [ ] `task4_generator_best.pt` checkpoint saved successfully.
""")
    ]
    return make_notebook(cells)


# ============================================================================
# Notebook 14: task4_gan_optuna.ipynb
# ============================================================================
def build_task4_gan_optuna():
    cells = [
        md_cell("""# Task 4: Conditional GAN Optuna Search

Hyperparameter optimization for the cGAN Generator & Discriminator (Plan 6 §4.6 & Plan 7).
Searches G learning rate, D learning rate, batch size, embedding dimension, and $\\lambda_{L1}$.
"""),
        code_cell("""# 1. Config
N_TRIALS = 12
OPTUNA_EPOCHS = 15
STUDY_NAME = "task4_conditional_gan"
STORAGE_DB = "sqlite:///task4_study.db"
SEED = 42
"""),
        code_cell("""# 2. Imports & Setup
import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
import optuna

RESEARCH_ROOT = Path("..").resolve()
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from constants import CONFIGS_ROOT
from src.device_utils import get_device, device_report
from src.datasets import FS2KDataset
from src.models_gan import UNetGenerator, PatchGANDiscriminator
from src.train_loop import train_one_epoch_gan, validate_gan, set_seed

set_seed(SEED)
device = get_device()
"""),
        code_cell("""# 3. Objective Function
train_dataset = FS2KDataset(mode="train", seed=SEED)
val_dataset = FS2KDataset(mode="val", seed=SEED)

def objective(trial: optuna.Trial) -> float:
    g_lr = trial.suggest_float("g_lr", 1e-5, 1e-3, log=True)
    d_lr = trial.suggest_float("d_lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    base_channels = trial.suggest_categorical("base_channels", [32, 64])
    embed_dim = trial.suggest_categorical("embed_dim", [4, 8, 16])
    lambda_l1 = trial.suggest_float("lambda_l1", 10.0, 200.0)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    G = UNetGenerator(base_channels=base_channels, embed_dim=embed_dim).to(device)
    D = PatchGANDiscriminator(base_channels=base_channels).to(device)

    optG = torch.optim.Adam(G.parameters(), lr=g_lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=d_lr, betas=(0.5, 0.999))

    best_val_l1 = float("inf")

    for epoch in range(1, OPTUNA_EPOCHS + 1):
        train_one_epoch_gan(G, D, train_loader, optG, optD, device, lambda_l1=lambda_l1)
        vl = validate_gan(G, val_loader, device)
        if vl["val_l1"] < best_val_l1:
            best_val_l1 = vl["val_l1"]
        trial.report(vl["val_l1"], step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_l1
"""),
        code_cell("""# 4. Run Study & Export Config
study = optuna.create_study(
    study_name=STUDY_NAME,
    storage=STORAGE_DB,
    load_if_exists=True,
    direction="minimize",
    pruner=optuna.pruners.MedianPruner()
)
study.optimize(objective, n_trials=N_TRIALS)

cfg_path = CONFIGS_ROOT / "task4_best_config.yaml"
with open(cfg_path, "w") as f:
    yaml.dump({
        "task": "task4_conditional_gan",
        "best_trial_number": int(study.best_trial.number),
        "best_value": float(study.best_value),
        "params": study.best_params,
    }, f, indent=2)

print(f"Exported to {cfg_path}")
""")
    ]
    return make_notebook(cells)


# ============================================================================
# Notebook 15: task4_validation.ipynb (Stub)
# ============================================================================
def build_task4_validation():
    cells = [
        md_cell("""# Task 4: Conditional GAN Validation & Qualitative Evaluation

Final evaluation of the style-conditioned generator on the official FS2K test set (Plan 6 §4.8).
"""),
        code_cell("""# Imports
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

RESEARCH_ROOT = Path("..").resolve()
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from src.device_utils import get_device
from src.datasets import FS2KDataset
from src.models_gan import UNetGenerator
from src.checkpoint_utils import load_checkpoint
"""),
        code_cell("""# Configuration
GENERATOR_CKPT = RESEARCH_ROOT / "checkpoints" / "task4_generator_best.pt"
DEVICE = get_device()
"""),
        md_cell("## 1. Load Trained Generator"),
        code_cell("""# TODO: Load UNetGenerator from GENERATOR_CKPT
# G = UNetGenerator().to(DEVICE)
# load_checkpoint(GENERATOR_CKPT, G, device=DEVICE)
# G.eval()
"""),
        md_cell("## 2. Test Set Evaluation & Quantitative Metrics"),
        code_cell("""# TODO: Run generator over official FS2K test set.
# Compute L1 reconstruction loss, PSNR, and SSIM per style category.
"""),
        md_cell("## 3. Style Conditioning Matrix (Photo x 3 Styles)"),
        code_cell("""# TODO: Display grid of input photos rendered into Style 0, Style 1, Style 2 side-by-side.
"""),
        md_cell("## 4. Failure Mode Analysis (Style Bleeding & Artifacts)"),
        code_cell("""# TODO: Identify and display >= 4 failure cases (e.g. style bleeding, missing facial features).
""")
    ]
    return make_notebook(cells)


def main():
    builders = {
        "task1_vae_simple.ipynb": build_task1_vae_simple,
        "task1_vae_optuna.ipynb": build_task1_vae_optuna,
        "task1_validation.ipynb": build_task1_validation,
        "task2a_classifier_simple.ipynb": build_task2a_classifier_simple,
        "task2a_classifier_optuna.ipynb": build_task2a_classifier_optuna,
        "task2a_validation.ipynb": build_task2a_validation,
        "task2b_specialists_simple.ipynb": build_task2b_specialists_simple,
        "task2b_specialists_optuna.ipynb": build_task2b_specialists_optuna,
        "task2b_validation.ipynb": build_task2b_validation,
        "task3_moe_simple.ipynb": build_task3_moe_simple,
        "task3_moe_optuna.ipynb": build_task3_moe_optuna,
        "task3_validation.ipynb": build_task3_validation,
        "task4_gan_simple.ipynb": build_task4_gan_simple,
        "task4_gan_optuna.ipynb": build_task4_gan_optuna,
        "task4_validation.ipynb": build_task4_validation,
    }

    for filename, builder in builders.items():
        nb_json = builder()
        out_path = NOTEBOOKS_DIR / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(nb_json, f, indent=1)
        print(f"Created: {filename} ({len(nb_json['cells'])} cells)")

    print(f"\nSuccessfully generated all {len(builders)} notebooks in {NOTEBOOKS_DIR}")


if __name__ == "__main__":
    main()
