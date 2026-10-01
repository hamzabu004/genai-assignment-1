"""
Convolutional Denoising Autoencoder (ConvDAE) for Tasks 1 and 2b.
Architecture design follows Plan 5 & Plan 6:
- 5-stage strided conv encoder downsampling 128x128 -> 4x4
- GroupNorm instead of BatchNorm to handle variable/small batch sizes
- Decoder uses bilinear/nearest upsampling + Conv to eliminate checkerboard artifacts
- Optional single skip connection at the highest-resolution stage for ablation (Plan 6 §1.6)
- Sigmoid output activation for [0, 1] normalized pixel restoration
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_norm_layer(channels: int, num_groups: int = 8) -> nn.Module:
    # Ensure num_groups divides channels evenly
    while channels % num_groups != 0 and num_groups > 1:
        num_groups //= 2
    return nn.GroupNorm(num_groups=num_groups, num_channels=channels)


class ConvDAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        latent_dim: int = 128,
        dropout: float = 0.1,
        use_skip: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.use_skip = use_skip

        c1 = base_channels       # 32
        c2 = base_channels * 2   # 64
        c3 = base_channels * 4   # 128
        c4 = base_channels * 8   # 256
        c5 = base_channels * 8   # 256

        # --- Encoder (128x128 -> 4x4) ---
        # Stage 1: 128x128 -> 64x64
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=4, stride=2, padding=1),
            _get_norm_layer(c1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        # Stage 2: 64x64 -> 32x32
        self.enc2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1),
            _get_norm_layer(c2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        # Stage 3: 32x32 -> 16x16
        self.enc3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1),
            _get_norm_layer(c3),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Stage 4: 16x16 -> 8x8
        self.enc4 = nn.Sequential(
            nn.Conv2d(c3, c4, kernel_size=4, stride=2, padding=1),
            _get_norm_layer(c4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Stage 5: 8x8 -> 4x4
        self.enc5 = nn.Sequential(
            nn.Conv2d(c4, c5, kernel_size=4, stride=2, padding=1),
            _get_norm_layer(c5),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.flatten_dim = c5 * 4 * 4
        self.fc_z = nn.Linear(self.flatten_dim, latent_dim)
        
        # --- Decoder (4x4 -> 128x128) ---
        self.fc_dec = nn.Linear(latent_dim, self.flatten_dim)

        # Stage 5: 4x4 -> 8x8
        self.dec5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c5, c4, kernel_size=3, stride=1, padding=1),
            _get_norm_layer(c4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Stage 4: 8x8 -> 16x16
        self.dec4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c4, c3, kernel_size=3, stride=1, padding=1),
            _get_norm_layer(c3),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Stage 3: 16x16 -> 32x32
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c3, c2, kernel_size=3, stride=1, padding=1),
            _get_norm_layer(c2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Stage 2: 32x32 -> 64x64
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c2, c1, kernel_size=3, stride=1, padding=1),
            _get_norm_layer(c1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Stage 1: 64x64 -> 128x128
        # If skip connection is used, dec1 receives (c1 + c1) channels
        dec1_in = (c1 * 2) if use_skip else c1
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(dec1_in, c1, kernel_size=3, stride=1, padding=1),
            _get_norm_layer(c1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c1, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),  # Restores into [0, 1]
        )


    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        e1 = self.enc1(x)  # 64x64
        e2 = self.enc2(e1) # 32x32
        e3 = self.enc3(e2) # 16x16
        e4 = self.enc4(e3) # 8x8
        e5 = self.enc5(e4) # 4x4
        flat = torch.flatten(e5, start_dim=1)
        z = self.fc_z(flat)
        return z, (e1 if self.use_skip else None)

    def decode(self, z: torch.Tensor, skip1: Optional[torch.Tensor] = None) -> torch.Tensor:
        d = self.fc_dec(z)
        d = d.view(-1, self.base_channels * 8, 4, 4)
        d = self.dec5(d)  # 8x8
        d = self.dec4(d)  # 16x16
        d = self.dec3(d)  # 32x32
        d = self.dec2(d)  # 64x64
        if self.use_skip and skip1 is not None:
            d = torch.cat([d, skip1], dim=1)
        out = self.dec1(d)  # 128x128
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, skip1 = self.encode(x)
        recon = self.decode(z, skip1=skip1)
        return recon
