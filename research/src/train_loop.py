"""
Training and validation loops for all research tasks.
Includes:
- DAE training & validation (Tasks 1 and 2b)
- Classifier training & validation (Task 2a)
- MoE joint training & validation (Task 3)
- Conditional GAN training & validation (Task 4)
Supports Automatic Mixed Precision (AMP), gradient clipping, and metric logging.
"""

import random
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses import (
    dae_loss,
    compute_ssim,
    moe_joint_loss,
    generator_loss,
    discriminator_loss,
)


def set_seed(seed: int = 42):
    """Ensures maximum determinism across python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# Task 1 & 2b: DAE Training / Validation
# ============================================================================

def train_one_epoch_dae(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float = 0.8,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Dict[str, float]:
    model.train()
    total_loss, total_recon, total_ssim = 0.0, 0.0, 0.0
    num_samples = 0

    for corrupted, clean in loader:
        corrupted = corrupted.to(device)
        clean = clean.to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                recon = model(corrupted)
                loss, r_val, _, s_val = dae_loss(recon, clean, alpha=alpha)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            recon = model(corrupted)
            loss, r_val, _, s_val = dae_loss(recon, clean, alpha=alpha)
            loss.backward()
            optimizer.step()

        batch_size = clean.shape[0]
        total_loss += loss.item() * batch_size
        total_recon += r_val * batch_size
        total_ssim += s_val * batch_size
        num_samples += batch_size

    if num_samples == 0:
        raise ValueError("Cannot train a DAE with an empty data loader.")

    return {
        "train_loss": total_loss / num_samples,
        "recon_loss": total_recon / num_samples,
        "ssim": total_ssim / num_samples,
    }


def validate_dae(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    alpha: float = 0.8,
) -> Dict[str, float]:
    model.eval()
    total_loss, total_recon, total_ssim = 0.0, 0.0, 0.0
    num_samples = 0

    with torch.no_grad():
        for corrupted, clean in loader:
            corrupted = corrupted.to(device)
            clean = clean.to(device)

            recon = model(corrupted)
            loss, r_val, _, s_val = dae_loss(recon, clean, alpha=alpha)

            batch_size = clean.shape[0]
            total_loss += loss.item() * batch_size
            total_recon += r_val * batch_size
            total_ssim += s_val * batch_size
            num_samples += batch_size

    if num_samples == 0:
        raise ValueError("Cannot validate a DAE with an empty data loader.")

    return {
        "val_loss": total_loss / num_samples,
        "val_recon": total_recon / num_samples,
        "val_ssim": total_ssim / num_samples,
    }


# ============================================================================
# Task 2a: Classifier Training / Validation
# ============================================================================

def train_one_epoch_classifier(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total_samples += len(labels)

    return {
        "train_loss": total_loss / total_samples,
        "train_acc": correct / total_samples,
    }


def validate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total_samples += len(labels)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    return {
        "val_loss": total_loss / total_samples,
        "val_acc": correct / total_samples,
        "preds": all_preds,
        "labels": all_labels,
    }


# ============================================================================
# Task 3: Soft MoE Training / Validation
# ============================================================================

def train_one_epoch_moe(
    moe_model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_weights: Dict[str, float],
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Dict[str, float]:
    moe_model.train()
    metrics_sum = {"loss": 0.0, "l1": 0.0, "ssim": 0.0, "cls_loss": 0.0, "balance_loss": 0.0}
    num_batches = len(loader)

    for corrupted, clean, true_labels in loader:
        corrupted = corrupted.to(device)
        clean = clean.to(device)
        true_labels = true_labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                pred, routing_probs, gate_logits = moe_model(corrupted)
                loss, m = moe_joint_loss(
                    pred=pred,
                    target=clean,
                    routing_probs=routing_probs,
                    classifier_logits=gate_logits,
                    true_labels=true_labels,
                    lambda1=lambda_weights.get("lambda1", 0.8),
                    lambda2=lambda_weights.get("lambda2", 0.2),
                    lambda3=lambda_weights.get("lambda3", 0.1),
                    lambda4=lambda_weights.get("lambda4", 0.01),
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred, routing_probs, gate_logits = moe_model(corrupted)
            loss, m = moe_joint_loss(
                pred=pred,
                target=clean,
                routing_probs=routing_probs,
                classifier_logits=gate_logits,
                true_labels=true_labels,
                lambda1=lambda_weights.get("lambda1", 0.8),
                lambda2=lambda_weights.get("lambda2", 0.2),
                lambda3=lambda_weights.get("lambda3", 0.1),
                lambda4=lambda_weights.get("lambda4", 0.01),
            )
            loss.backward()
            optimizer.step()

        for k in metrics_sum:
            metrics_sum[k] += m[k]

    return {k: v / num_batches for k, v in metrics_sum.items()}


def validate_moe(
    moe_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lambda_weights: Dict[str, float],
) -> Dict[str, Any]:
    moe_model.eval()
    metrics_sum = {"loss": 0.0, "l1": 0.0, "ssim": 0.0, "cls_loss": 0.0, "balance_loss": 0.0}
    num_batches = len(loader)
    all_routing_probs = []

    with torch.no_grad():
        for corrupted, clean, true_labels in loader:
            corrupted = corrupted.to(device)
            clean = clean.to(device)
            true_labels = true_labels.to(device)

            pred, routing_probs, gate_logits = moe_model(corrupted)
            loss, m = moe_joint_loss(
                pred=pred,
                target=clean,
                routing_probs=routing_probs,
                classifier_logits=gate_logits,
                true_labels=true_labels,
                lambda1=lambda_weights.get("lambda1", 0.8),
                lambda2=lambda_weights.get("lambda2", 0.2),
                lambda3=lambda_weights.get("lambda3", 0.1),
                lambda4=lambda_weights.get("lambda4", 0.01),
            )

            for k in metrics_sum:
                metrics_sum[k] += m[k]
            all_routing_probs.append(routing_probs.cpu().numpy())

    avg_metrics = {f"val_{k}": v / num_batches for k, v in metrics_sum.items()}
    avg_metrics["avg_expert_weights"] = np.concatenate(all_routing_probs, axis=0).mean(axis=0)
    return avg_metrics


# ============================================================================
# Task 4: Conditional GAN Training / Validation
# ============================================================================

def train_one_epoch_gan(
    G: nn.Module,
    D: nn.Module,
    loader: DataLoader,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
    device: torch.device,
    lambda_l1: float = 100.0,
    scaler_g: Optional[torch.amp.GradScaler] = None,
    scaler_d: Optional[torch.amp.GradScaler] = None,
) -> Dict[str, float]:
    G.train()
    D.train()

    total_g_loss = 0.0
    total_d_loss = 0.0
    total_d_real = 0.0
    total_d_fake = 0.0
    total_g_adv = 0.0
    total_g_l1 = 0.0
    num_batches = len(loader)

    for photo, sketch, style in loader:
        photo = photo.to(device)
        sketch = sketch.to(device)
        style = style.to(device)

        # ---------------------
        # Train Discriminator
        # ---------------------
        opt_D.zero_grad(set_to_none=True)

        fake_sketch = G(photo, style)
        real_logits = D(photo, sketch, style)
        fake_logits = D(photo, fake_sketch.detach(), style)

        d_loss, d_r, d_f = discriminator_loss(real_logits, fake_logits)
        d_loss.backward()
        # Gradient clipping for discriminator (Plan 7)
        torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=5.0)
        opt_D.step()

        # ---------------------
        # Train Generator
        # ---------------------
        opt_G.zero_grad(set_to_none=True)

        fake_logits_for_g = D(photo, fake_sketch, style)
        g_loss, g_adv, g_l1 = generator_loss(fake_logits_for_g, fake_sketch, sketch, lambda_l1=lambda_l1)
        g_loss.backward()
        opt_G.step()

        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()
        total_d_real += d_r
        total_d_fake += d_f
        total_g_adv += g_adv
        total_g_l1 += g_l1

    return {
        "g_loss": total_g_loss / num_batches,
        "d_loss": total_d_loss / num_batches,
        "d_real": total_d_real / num_batches,
        "d_fake": total_d_fake / num_batches,
        "g_adv": total_g_adv / num_batches,
        "g_l1": total_g_l1 / num_batches,
    }


def validate_gan(
    G: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    G.eval()
    total_l1 = 0.0
    num_batches = len(loader)

    with torch.no_grad():
        for photo, sketch, style in loader:
            photo = photo.to(device)
            sketch = sketch.to(device)
            style = style.to(device)

            fake_sketch = G(photo, style)
            l1 = nn.functional.l1_loss(fake_sketch, sketch)
            total_l1 += l1.item()

    return {"val_l1": total_l1 / num_batches}
