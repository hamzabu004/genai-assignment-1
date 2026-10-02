"""
Loss functions for all research tasks.
- SSIM and composite reconstruction loss (alpha * L1 + (1 - alpha) * (1 - SSIM))
- DAE reconstruction loss (same composite, wrapped for API compat)
- Mixture-of-Experts loss with classification and routing balance regularization
- Conditional GAN losses (generator and PatchGAN discriminator)
"""

import math
from typing import Tuple, Optional, Dict, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = torch.outer(g, g).unsqueeze(0).unsqueeze(0)
    return window_2d.repeat(channels, 1, 1, 1)


def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Differentiable Structural Similarity Index (SSIM) in PyTorch.
    Operates in float32 for numerical stability.
    img1, img2: (B, C, H, W) tensors.
    """
    # Enforce float32 even under autocast/fp16 for stability
    img1 = img1.to(torch.float32)
    img2 = img2.to(torch.float32)

    channels = img1.shape[1]
    window = _gaussian_window(window_size, sigma, channels).to(device=img1.device, dtype=torch.float32)
    padding = window_size // 2

    mu1 = F.conv2d(img1, window, padding=padding, groups=channels)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channels) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    if reduction == "mean":
        return ssim_map.mean()
    elif reduction == "none":
        return ssim_map
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.8,
    data_range: float = 1.0,
) -> Tuple[torch.Tensor, float, float]:
    """
    Standard assignment reconstruction loss:
    L_recon = alpha * L1 + (1 - alpha) * (1 - SSIM)
    Returns: (total_loss, l1_val, ssim_val)
    """
    l1 = F.l1_loss(pred, target)
    ssim_val = compute_ssim(pred, target, data_range=data_range)
    loss = alpha * l1 + (1.0 - alpha) * (1.0 - ssim_val)
    return loss, l1.item(), ssim_val.item()


def dae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.8,
) -> Tuple[torch.Tensor, float, float, float]:
    """
    DAE loss = L1 + SSIM composite.
    Returns: (total_loss, l1_val, dummy_kl_val, ssim_val) for compat.
    """
    total_loss, l1_val, ssim_val = reconstruction_loss(pred, target, alpha=alpha, data_range=1.0)
    return total_loss, l1_val, 0.0, ssim_val


def compute_balance_loss(routing_probs: torch.Tensor) -> torch.Tensor:
    """
    Load balancing regularization for MoE gating (Task 3).
    Matches assignment specification:
    L_balance = sum_{k=1}^4 (w_bar_k - 1/4)^2
    where w_bar_k is the mean routing weight assigned to branch k across a balanced batch.
    routing_probs: (B, num_experts)
    """
    expert_usage = routing_probs.mean(dim=0)
    target_usage = 1.0 / routing_probs.shape[1]
    balance_loss = torch.sum((expert_usage - target_usage) ** 2)
    return balance_loss



def moe_joint_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    routing_probs: torch.Tensor,
    classifier_logits: torch.Tensor,
    true_labels: torch.Tensor,
    lambda1: float = 0.8,
    lambda2: float = 0.2,
    lambda3: float = 0.1,
    lambda4: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Joint loss for Task 3:
    lambda1 * L1 + lambda2 * (1 - SSIM) + lambda3 * L_cls + lambda4 * L_balance
    """
    l1 = F.l1_loss(pred, target)
    ssim_val = compute_ssim(pred, target, data_range=1.0)
    cls_loss = F.cross_entropy(classifier_logits, true_labels)
    balance_loss = compute_balance_loss(routing_probs)

    total_loss = (
        lambda1 * l1
        + lambda2 * (1.0 - ssim_val)
        + lambda3 * cls_loss
        + lambda4 * balance_loss
    )

    metrics = {
        "loss": total_loss.item(),
        "l1": l1.item(),
        "ssim": ssim_val.item(),
        "cls_loss": cls_loss.item(),
        "balance_loss": balance_loss.item(),
    }
    return total_loss, metrics


def _logit_scales(logits: Union[torch.Tensor, Sequence[torch.Tensor]]) -> Tuple[torch.Tensor, ...]:
    return tuple(logits) if isinstance(logits, (tuple, list)) else (logits,)


def sobel_edge_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 distance between grayscale Sobel gradients, scaled to a stable range."""
    pred = pred.float()
    target = target.float()
    if pred.shape[1] == 3:
        weights = pred.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        pred = (pred * weights).sum(dim=1, keepdim=True)
        target = (target * weights).sum(dim=1, keepdim=True)
    kx = pred.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8.0
    ky = kx.transpose(-1, -2)
    pred_edges = torch.cat((F.conv2d(pred, kx, padding=1), F.conv2d(pred, ky, padding=1)), dim=1)
    target_edges = torch.cat((F.conv2d(target, kx, padding=1), F.conv2d(target, ky, padding=1)), dim=1)
    return F.l1_loss(pred_edges, target_edges)


def gan_quality_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Validation metrics for sketch fidelity and stroke sharpness in [-1, 1]."""
    edge_error = sobel_edge_loss(pred, target)
    if pred.shape[1] == 3:
        weights = pred.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        target_gray = (target.float() * weights).sum(dim=1, keepdim=True)
    else:
        target_gray = target.float()
    kx = target_gray.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8.0
    ky = kx.transpose(-1, -2)
    target_edge_scale = (
        F.conv2d(target_gray, kx, padding=1).abs().mean()
        + F.conv2d(target_gray, ky, padding=1).abs().mean()
    ).clamp_min(1e-6)
    return {
        "val_l1": F.l1_loss(pred, target),
        "val_ssim": compute_ssim(pred, target, data_range=2.0),
        "val_edge": edge_error,
        "val_edge_normalized": edge_error / target_edge_scale,
    }


def generator_loss(
    fake_logits: Union[torch.Tensor, Sequence[torch.Tensor]],
    pred_sketch: torch.Tensor,
    target_sketch: torch.Tensor,
    lambda_l1: float = 100.0,
    lambda_edge: float = 15.0,
    lambda_ssim: float = 5.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Conditional GAN loss with paired reconstruction, edge, and structure terms.
    """
    scales = _logit_scales(fake_logits)
    adv_loss = torch.stack([
        F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        for logits in scales
    ]).mean()
    l1_loss = F.l1_loss(pred_sketch, target_sketch)
    edge_loss = sobel_edge_loss(pred_sketch, target_sketch)
    ssim_loss = 1.0 - compute_ssim(pred_sketch, target_sketch, data_range=2.0)
    g_loss = adv_loss + lambda_l1 * l1_loss + lambda_edge * edge_loss + lambda_ssim * ssim_loss
    return g_loss, {
        "g_adv": adv_loss.item(),
        "g_l1": l1_loss.item(),
        "g_edge": edge_loss.item(),
        "g_ssim_loss": ssim_loss.item(),
    }


def discriminator_loss(
    real_logits: Union[torch.Tensor, Sequence[torch.Tensor]],
    fake_logits: Union[torch.Tensor, Sequence[torch.Tensor]],
) -> Tuple[torch.Tensor, float, float]:
    """
    Conditional GAN Discriminator loss averaged across all output scales.
    """
    real_scales, fake_scales = _logit_scales(real_logits), _logit_scales(fake_logits)
    if len(real_scales) != len(fake_scales):
        raise ValueError("Real and fake discriminator outputs must have the same number of scales.")
    d_real = torch.stack([
        F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        for logits in real_scales
    ]).mean()
    d_fake = torch.stack([
        F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
        for logits in fake_scales
    ]).mean()
    d_loss = 0.5 * (d_real + d_fake)
    return d_loss, d_real.item(), d_fake.item()
