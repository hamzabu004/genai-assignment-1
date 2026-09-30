"""
Models for:
- Task 2a: CorruptionClassifier (4 classes: clean, salt, blur, occlusion)
- Task 3: SoftMoERestorer (reuses classifier as gate + specialists + identity)
"""

from typing import Tuple, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models_vae import _get_norm_layer, ConvVAE


class CorruptionClassifier(nn.Module):
    """
    Lightweight CNN classifier for Task 2a.
    Downsamples 128x128 -> 8x8 using 4 strided conv blocks,
    followed by Global Average Pooling and a small MLP classification head.
    """

    NUM_CLASSES = 4  # 0: clean, 1: salt, 2: blur, 3: occlusion

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        dropout: float = 0.2,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_classes = num_classes

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        # 128x128 -> 64x64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=4, stride=2, padding=1),
            _get_norm_layer(c1),
            nn.ReLU(inplace=True),
        )
        # 64x64 -> 32x32
        self.conv2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1),
            _get_norm_layer(c2),
            nn.ReLU(inplace=True),
        )
        # 32x32 -> 16x16
        self.conv3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1),
            _get_norm_layer(c3),
            nn.ReLU(inplace=True),
        )
        # 16x16 -> 8x8
        self.conv4 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=4, stride=2, padding=1),
            _get_norm_layer(c4),
            nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(c4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(128, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.gap(h)
        return torch.flatten(h, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        logits = self.classifier(feat)
        return logits


class SoftMoERestorer(nn.Module):
    """
    Task 3: Soft Mixture-of-Experts.
    Combines:
    - 1 Identity branch (for clean inputs)
    - 3 Specialist restoration autoencoders (salt, blur, occlusion)
    - 1 Gating network initialized from Task 2a's classifier
    Outputs a weighted sum of branch predictions based on temperature-scaled softmax routing.
    """

    def __init__(
        self,
        gate: CorruptionClassifier,
        specialist_salt: ConvVAE,
        specialist_blur: ConvVAE,
        specialist_occlusion: ConvVAE,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.gate = gate
        self.specialist_salt = specialist_salt
        self.specialist_blur = specialist_blur
        self.specialist_occlusion = specialist_occlusion
        self.temperature = temperature

    def forward(
        self, x: torch.Tensor, temperature: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
        - out: composite restored image (B, 3, 128, 128)
        - routing_probs: softmax weights across 4 branches (B, 4)
        - gate_logits: raw logits from gate (B, 4)
        """
        temp = temperature if temperature is not None else self.temperature
        gate_logits = self.gate(x)
        routing_probs = F.softmax(gate_logits / temp, dim=1)  # shape (B, 4)

        # Branch 0: Identity (clean bypass)
        out_clean = x

        # Branch 1: Salt & Pepper specialist
        out_salt, _, _ = self.specialist_salt(x)

        # Branch 2: Blur specialist
        out_blur, _, _ = self.specialist_blur(x)

        # Branch 3: Occlusion specialist
        out_occl, _, _ = self.specialist_occlusion(x)

        # Stack outputs: (B, 4, C, H, W)
        stacked = torch.stack([out_clean, out_salt, out_blur, out_occl], dim=1)

        # Reshape routing weights for broadcasting: (B, 4, 1, 1, 1)
        weights = routing_probs.view(-1, 4, 1, 1, 1)

        # Soft weighted sum
        out = torch.sum(stacked * weights, dim=1)
        return out, routing_probs, gate_logits
