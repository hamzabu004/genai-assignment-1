"""
Conditional GAN architecture for Task 4: Style-Conditioned Face-to-Sketch.
Follows Plan 5 & Plan 6 specification:
- UNetGenerator: U-Net encoder-decoder with full skip connections and learned style embedding injection.
  Final activation is Tanh -> [-1, 1].
- PatchGANDiscriminator: 70x70 PatchGAN discriminator taking (photo, sketch, style_map).
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models_vae import _get_norm_layer


class UNetBlock(nn.Module):
    """Basic conv block for U-Net encoder/decoder with normalization and activation."""
    def __init__(self, in_c: int, out_c: int, down: bool = True, use_dropout: bool = False):
        super().__init__()
        self.down = down
        if down:
            self.conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                _get_norm_layer(out_c),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.2) if use_dropout else nn.Identity(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
                _get_norm_layer(out_c),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2) if use_dropout else nn.Identity(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetGenerator(nn.Module):
    """
    Style-conditioned U-Net Generator for photo-to-sketch synthesis.
    Input: photo (B, 3, 128, 128) + style_idx (B,) in {0, 1, 2}.
    Output: sketch (B, 3, 128, 128) in range [-1, 1].
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        embed_dim: int = 8,
        num_styles: int = 3,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.embed_dim = embed_dim
        self.num_styles = num_styles

        # Style embedding
        self.style_embed = nn.Embedding(num_styles, embed_dim)

        # Style projection to bottleneck spatial dimensions (4x4)
        c1 = base_channels       # 64
        c2 = base_channels * 2   # 128
        c3 = base_channels * 4   # 256
        c4 = base_channels * 8   # 512
        c5 = base_channels * 8   # 512

        self.style_proj = nn.Linear(embed_dim, 4 * 4 * embed_dim)

        # Encoder stages: 128 -> 64 -> 32 -> 16 -> 8 -> 4
        self.e1 = nn.Conv2d(in_channels, c1, kernel_size=4, stride=2, padding=1)  # no norm on first
        self.e2 = UNetBlock(c1, c2, down=True)
        self.e3 = UNetBlock(c2, c3, down=True)
        self.e4 = UNetBlock(c3, c4, down=True)
        self.e5 = UNetBlock(c4, c5, down=True)

        # Decoder stages (with skip connections and style injection):
        # Bottleneck takes e5 (c5) + style feature map (embed_dim)
        self.d5 = UNetBlock(c5 + embed_dim, c4, down=False, use_dropout=True)
        # Skip connection from e4: d5(c4) + e4(c4) = 2*c4
        self.d4 = UNetBlock(c4 + c4, c3, down=False, use_dropout=True)
        # Skip from e3: d4(c3) + e3(c3) = 2*c3
        self.d3 = UNetBlock(c3 + c3, c2, down=False)
        # Skip from e2: d3(c2) + e2(c2) = 2*c2
        self.d2 = UNetBlock(c2 + c2, c1, down=False)

        # Final stage: d2(c1) + e1(c1) -> out_channels, activation Tanh
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c1 + c1, c1, kernel_size=3, stride=1, padding=1),
            _get_norm_layer(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),  # Maps to [-1, 1]
        )

    def forward(self, photo: torch.Tensor, style_idx: torch.Tensor) -> torch.Tensor:
        b = photo.shape[0]

        # Style embedding map: (B, embed_dim) -> (B, embed_dim, 4, 4)
        s_emb = self.style_embed(style_idx)
        s_feat = self.style_proj(s_emb).view(b, self.embed_dim, 4, 4)

        # Encoder
        x1 = F.leaky_relu(self.e1(photo), 0.2)  # 64x64
        x2 = self.e2(x1)                        # 32x32
        x3 = self.e3(x2)                        # 16x16
        x4 = self.e4(x3)                        # 8x8
        x5 = self.e5(x4)                        # 4x4

        # Inject style into bottleneck
        bot = torch.cat([x5, s_feat], dim=1)

        # Decoder with skip connections
        u5 = self.d5(bot)                       # 8x8
        u4 = self.d4(torch.cat([u5, x4], dim=1))# 16x16
        u3 = self.d3(torch.cat([u4, x3], dim=1))# 32x32
        u2 = self.d2(torch.cat([u3, x2], dim=1))# 64x64
        out = self.final(torch.cat([u2, x1], dim=1))  # 128x128

        return out


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator for conditional GAN.
    Input: photo (B, 3, H, W) + sketch (B, 3, H, W) + style condition.
    Outputs a matrix of patch realism logits.
    """

    def __init__(
        self,
        in_channels: int = 6,  # 3 photo + 3 sketch
        base_channels: int = 64,
        num_styles: int = 3,
    ):
        super().__init__()
        self.num_styles = num_styles
        # Total in_channels = photo (3) + sketch (3) + style one-hot (num_styles)
        total_in = in_channels + num_styles

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.net = nn.Sequential(
            # Stage 1: 128 -> 64
            nn.Conv2d(total_in, c1, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 2: 64 -> 32
            nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1, bias=False),
            _get_norm_layer(c2),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 3: 32 -> 16
            nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1, bias=False),
            _get_norm_layer(c3),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 4: 16 -> 15 (stride 1)
            nn.Conv2d(c3, c4, kernel_size=4, stride=1, padding=1, bias=False),
            _get_norm_layer(c4),
            nn.LeakyReLU(0.2, inplace=True),

            # Stage 5: Output 1-channel patch logit map
            nn.Conv2d(c4, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(
        self, photo: torch.Tensor, sketch: torch.Tensor, style_idx: torch.Tensor
    ) -> torch.Tensor:
        b, _, h, w = photo.shape
        # Create one-hot style channel maps of shape (B, num_styles, H, W)
        style_one_hot = F.one_hot(style_idx, num_classes=self.num_styles).float()
        style_map = style_one_hot.view(b, self.num_styles, 1, 1).expand(-1, -1, h, w)

        x = torch.cat([photo, sketch, style_map], dim=1)
        return self.net(x)
