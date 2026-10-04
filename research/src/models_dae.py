"""
Convolutional Denoising Autoencoder (ConvDAE) for Tasks 1 and 2b.
Architecture design follows Plan 5 & Plan 6:
- 6-stage strided conv encoder downsampling 128x128 -> 2x2
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
        base_channels: int = 64,
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

        # --- Encoder (128x128 -> 2x2) ---
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

        # Stage 6: 4x4 -> 2x2
        self.enc6 = nn.Sequential(
            nn.Conv2d(c5, c5, kernel_size=4, stride=2, padding=1),
            _get_norm_layer(c5),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.flatten_dim = c5 * 2 * 2
        self.fc_z = nn.Linear(self.flatten_dim, latent_dim)
        
        # --- Decoder (2x2 -> 128x128) ---
        self.fc_dec = nn.Linear(latent_dim, self.flatten_dim)

        # Stage 6: 2x2 -> 4x4
        self.dec6 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c5, c5, kernel_size=3, stride=1, padding=1),
            _get_norm_layer(c5),
            nn.LeakyReLU(0.2, inplace=True),
        )

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
        e6 = self.enc6(e5) # 2x2
        flat = torch.flatten(e6, start_dim=1)
        z = self.fc_z(flat)
        return z, (e1 if self.use_skip else None)

    def decode(self, z: torch.Tensor, skip1: Optional[torch.Tensor] = None) -> torch.Tensor:
        d = self.fc_dec(z)
        d = d.view(-1, self.base_channels * 8, 2, 2)
        d = self.dec6(d)  # 4x4
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


class FlexibleConvDAE(nn.Module):
    """
    Flexible Convolutional Denoising Autoencoder configurable for:
    - 2 to 6 stages downsampling (128x128 down to 32x32, 16x16, 8x8, 4x4, or 2x2).
    - With skip connections (U-Net style intermediate skip pathways) or without skip connections (bottleneck compression).
    - Optional fully-connected latent projection or direct convolutional bottleneck.
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_stages: int = 4,
        use_skip: bool = True,
        latent_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert 2 <= num_stages <= 6, f"num_stages must be in [2, 6], got {num_stages}"
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.use_skip = use_skip
        self.latent_dim = latent_dim
        self.dropout = dropout

        max_channels = base_channels * 8
        self.stage_channels = [min(base_channels * (2**i), max_channels) for i in range(num_stages)]

        # Encoder: Stage 0 .. Stage (num_stages - 1)
        self.encoders = nn.ModuleList()
        curr_in = in_channels
        for i, curr_out in enumerate(self.stage_channels):
            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(curr_in, curr_out, kernel_size=4, stride=2, padding=1),
                    _get_norm_layer(curr_out),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(dropout) if (dropout > 0 and i < 2) else nn.Identity(),
                )
            )
            curr_in = curr_out

        # Bottleneck
        bottleneck_c = self.stage_channels[-1]
        bottleneck_res = 128 // (2 ** num_stages)
        self.bottleneck_shape = (bottleneck_c, bottleneck_res, bottleneck_res)
        self.flatten_dim = bottleneck_c * bottleneck_res * bottleneck_res

        if latent_dim is not None:
            self.fc_enc = nn.Linear(self.flatten_dim, latent_dim)
            self.fc_dec = nn.Linear(latent_dim, self.flatten_dim)
        else:
            self.fc_enc = None
            self.fc_dec = None

        # Decoder stages: maps from stage (num_stages-1) down to 1
        self.decoders = nn.ModuleList()
        for k in reversed(range(1, num_stages)):
            in_c = self.stage_channels[k]
            skip_c = self.stage_channels[k-1] if use_skip else 0
            conv_in = in_c + skip_c
            target_out = self.stage_channels[k-1]
            self.decoders.append(
                nn.Sequential(
                    nn.Conv2d(conv_in, target_out, kernel_size=3, padding=1),
                    _get_norm_layer(target_out),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )

        # Final reconstruction stage (maps 64x64 -> 128x128 -> in_channels)
        c0 = self.stage_channels[0]
        self.final_dec = nn.Sequential(
            nn.Conv2d(c0, c0, kernel_size=3, padding=1),
            _get_norm_layer(c0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c0, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        feats = []
        curr = x
        for enc in self.encoders:
            curr = enc(curr)
            feats.append(curr)
        return curr, feats

    def decode(self, bottleneck: torch.Tensor, skips: Optional[list] = None) -> torch.Tensor:
        if self.fc_enc is not None:
            flat = torch.flatten(bottleneck, 1)
            z = self.fc_enc(flat)
            curr = self.fc_dec(z).view(-1, *self.bottleneck_shape)
        else:
            curr = bottleneck

        for i, dec_block in enumerate(self.decoders):
            k = self.num_stages - 1 - i
            curr = F.interpolate(curr, scale_factor=2, mode="bilinear", align_corners=False)
            if self.use_skip and skips is not None and (k - 1) >= 0:
                curr = torch.cat([curr, skips[k-1]], dim=1)
            curr = dec_block(curr)

        curr = F.interpolate(curr, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.final_dec(curr)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, skips = self.encode(x)
        return self.decode(bottleneck, skips=skips if self.use_skip else None)

