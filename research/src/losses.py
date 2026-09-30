"""
Loss functions for all research tasks.
- SSIM and composite reconstruction loss (alpha * L1 + (1 - alpha) * (1 - SSIM))
- VAE loss with KL divergence
- Mixture-of-Experts loss with classification and routing balance regularization
- Conditional GAN losses (generator and PatchGAN discriminator)
"""

import math
from typing import Tuple, Optional
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


def vae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    alpha: float = 0.8,
    beta: float = 0.001,
) -> Tuple[torch.Tensor, float, float, float]:
    """
    Full VAE loss = L_recon + beta * KL
    KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) averaged over batch.
    Returns: (total_loss, recon_val, kl_val, ssim_val)
    """
    recon, l1_val, ssim_val = reconstruction_loss(pred, target, alpha=alpha, data_range=1.0)
    # Average KL divergence per sample
    kl = -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    total_loss = recon + beta * kl
    return total_loss, recon.item(), kl.item(), ssim_val


def compute_balance_loss(routing_probs: torch.Tensor) -> torch.Tensor:
    """
    Load balancing regularization for MoE gating (Task 3).
    Penalizes variance in expert usage to prevent routing collapse.
    routing_probs: (B, num_experts)
    """
    # Mean usage per expert across the batch
    expert_usage = routing_probs.mean(dim=0)
    # Coefficient of variation squared: std^2 / mean^2
    variance = torch.var(expert_usage, unbiased=False)
    mean_val = expert_usage.mean() + 1e-8
    balance_loss = variance / (mean_val ** 2)
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


def generator_loss(
    fake_logits: torch.Tensor,
    pred_sketch: torch.Tensor,
    target_sketch: torch.Tensor,
    lambda_l1: float = 100.0,
) -> Tuple[torch.Tensor, float, float]:
    """
    Conditional GAN Generator loss (pix2pix style):
    L_G = BCE_adv(fake_logits, 1) + lambda_l1 * L1(pred, target)
    """
    adv_loss = F.binary_cross_entropy_with_logits(
        fake_logits, torch.ones_like(fake_logits)
    )
    l1_loss = F.l1_loss(pred_sketch, target_sketch)
    g_loss = adv_loss + lambda_l1 * l1_loss
    return g_loss, adv_loss.item(), l1_loss.item()


def discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
) -> Tuple[torch.Tensor, float, float]:
    """
    Conditional GAN Discriminator loss:
    L_D = 0.5 * (BCE(real_logits, 1) + BCE(fake_logits, 0))
    """
    d_real = F.binary_cross_entropy_with_logits(
        real_logits, torch.ones_like(real_logits)
    )
    d_fake = F.binary_cross_entropy_with_logits(
        fake_logits, torch.zeros_like(fake_logits)
    )
    d_loss = 0.5 * (d_real + d_fake)
    return d_loss, d_real.item(), d_fake.item()
