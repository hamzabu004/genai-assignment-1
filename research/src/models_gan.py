"""
Conditional GAN architecture for Task 4: Style-Conditioned Face-to-Sketch.
Follows Plan 5 & Plan 6 specification:
- UNetGenerator: U-Net encoder-decoder with full skip connections and learned style embedding injection.
  Final activation is Tanh -> [-1, 1].
- PatchGANDiscriminator: 70x70 PatchGAN discriminator taking (photo, sketch, style_map).
"""

from typing import Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models_dae import _get_norm_layer


class UNetBlock(nn.Module):
    """Basic conv block for U-Net encoder/decoder with normalization and activation."""
    def __init__(self, in_c: int, out_c: int, down: bool = True, use_dropout: bool = False,
                 dropout_rate: float = 0.2):
        super().__init__()
        self.down = down
        if down:
            self.conv = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                _get_norm_layer(out_c),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(dropout_rate) if use_dropout and dropout_rate > 0 else nn.Identity(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
                _get_norm_layer(out_c),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate) if use_dropout and dropout_rate > 0 else nn.Identity(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualRefinementBlock(nn.Module):
    """High-resolution residual feature refinement for crisp local strokes."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            _get_norm_layer(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            _get_norm_layer(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.block(x), inplace=True)


class UNetGenerator(nn.Module):
    """
    Style-conditioned U-Net Generator for photo-to-sketch synthesis.
    Input: photo (B, 3, 128, 128) + style_idx (B,) in {0, 1, 2}.
    Output: sketch (B, 3, 128, 128) in range [-1, 1].

    Enhanced with:
    - Multi-scale style modulation (FiLM scale-and-shift at decoder layers)
    - Grayscale constraint (enforces R=G=B to eliminate chromatic aberration)
    - Optional high-resolution residual refinement for sketch strokes
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        embed_dim: int = 8,
        num_styles: int = 3,
        dropout: float = 0.2,
        multi_scale_style: bool = False,
        grayscale_constraint: bool = False,
        residual_refinement: bool = False,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.embed_dim = embed_dim
        self.num_styles = num_styles
        self.dropout = dropout
        self.multi_scale_style = multi_scale_style
        self.grayscale_constraint = grayscale_constraint
        self.residual_refinement = residual_refinement

        # Style embedding
        self.style_embed = nn.Embedding(num_styles, embed_dim)

        # Style projection to bottleneck spatial dimensions (4x4)
        c1 = base_channels       # 64
        c2 = base_channels * 2   # 128
        c3 = base_channels * 4   # 256
        c4 = base_channels * 8   # 512
        c5 = base_channels * 8   # 512

        self.style_proj = nn.Linear(embed_dim, 4 * 4 * embed_dim)

        # Optional multi-scale style modulation (Plan 5 §Task 4 FSGAN style-vector expansion)
        if multi_scale_style:
            self.style_mod_d4 = nn.Linear(embed_dim, c3 * 2)  # scale and shift for 16x16
            self.style_mod_d3 = nn.Linear(embed_dim, c2 * 2)  # scale and shift for 32x32
            self.style_mod_d2 = nn.Linear(embed_dim, c1 * 2)  # scale and shift for 64x64

        # Encoder stages: 128 -> 64 -> 32 -> 16 -> 8 -> 4
        self.e1 = nn.Conv2d(in_channels, c1, kernel_size=4, stride=2, padding=1)  # no norm on first
        self.e2 = UNetBlock(c1, c2, down=True)
        self.e3 = UNetBlock(c2, c3, down=True)
        self.e4 = UNetBlock(c3, c4, down=True)
        self.e5 = UNetBlock(c4, c5, down=True)

        # Decoder stages (with skip connections and style injection):
        # Bottleneck takes e5 (c5) + style feature map (embed_dim)
        self.d5 = UNetBlock(c5 + embed_dim, c4, down=False, use_dropout=True, dropout_rate=dropout)
        # Skip connection from e4: d5(c4) + e4(c4) = 2*c4
        self.d4 = UNetBlock(c4 + c4, c3, down=False, use_dropout=True, dropout_rate=dropout)
        # Skip from e3: d4(c3) + e3(c3) = 2*c3
        self.d3 = UNetBlock(c3 + c3, c2, down=False)
        # Skip from e2: d3(c2) + e2(c2) = 2*c2
        self.d2 = UNetBlock(c2 + c2, c1, down=False)

        # Final stage: d2(c1) + e1(c1) -> out_channels, activation Tanh
        self.final_features = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c1 + c1, c1, kernel_size=3, stride=1, padding=1),
            _get_norm_layer(c1),
            nn.ReLU(inplace=True),
        )
        self.refine = ResidualRefinementBlock(c1) if residual_refinement else nn.Identity()
        self.output = nn.Sequential(nn.Conv2d(c1, out_channels, kernel_size=3, padding=1), nn.Tanh())

    def forward(
        self,
        photo: torch.Tensor,
        style_idx: torch.Tensor,
        apply_grayscale: Optional[bool] = None,
    ) -> torch.Tensor:
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
        if self.multi_scale_style and hasattr(self, "style_mod_d4"):
            mod4 = self.style_mod_d4(s_emb).view(b, -1, 1, 1)
            gamma4, beta4 = mod4.chunk(2, dim=1)
            u4 = u4 * (1.0 + gamma4) + beta4

        u3 = self.d3(torch.cat([u4, x3], dim=1))# 32x32
        if self.multi_scale_style and hasattr(self, "style_mod_d3"):
            mod3 = self.style_mod_d3(s_emb).view(b, -1, 1, 1)
            gamma3, beta3 = mod3.chunk(2, dim=1)
            u3 = u3 * (1.0 + gamma3) + beta3

        u2 = self.d2(torch.cat([u3, x2], dim=1))# 64x64
        if self.multi_scale_style and hasattr(self, "style_mod_d2"):
            mod2 = self.style_mod_d2(s_emb).view(b, -1, 1, 1)
            gamma2, beta2 = mod2.chunk(2, dim=1)
            u2 = u2 * (1.0 + gamma2) + beta2

        out = self.final_features(torch.cat([u2, x1], dim=1))  # 128x128
        out = self.output(self.refine(out))

        # Grayscale constraint: guarantees R=G=B to eliminate chromatic aberration
        enforce_gray = self.grayscale_constraint if apply_grayscale is None else apply_grayscale
        if enforce_gray and out.shape[1] == 3:
            gray = 0.299 * out[:, 0:1] + 0.587 * out[:, 1:2] + 0.114 * out[:, 2:3]
            out = gray.repeat(1, 3, 1, 1)

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
        self.style_embed_dim = 8
        self.style_embed = nn.Embedding(num_styles, self.style_embed_dim)
        self.style_proj = nn.Linear(self.style_embed_dim, self.style_embed_dim)
        # Total in_channels = photo (3) + sketch (3) + style one-hot (num_styles)
        total_in = in_channels + self.style_embed_dim

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

        self.half_scale_net = nn.Sequential(
            nn.Conv2d(total_in, c1, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1, bias=False),
            _get_norm_layer(c2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1, bias=False),
            _get_norm_layer(c3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c3, c4, kernel_size=4, stride=1, padding=1, bias=False),
            _get_norm_layer(c4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c4, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(
        self, photo: torch.Tensor, sketch: torch.Tensor, style_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = photo.shape
        # Broadcast a learned style embedding as spatial conditioning channels.
        style_embedding = self.style_proj(self.style_embed(style_idx))
        style_map = style_embedding.view(b, self.style_embed_dim, 1, 1).expand(-1, -1, h, w)

        x = torch.cat([photo, sketch, style_map], dim=1)
        half_photo = F.avg_pool2d(photo, kernel_size=2, stride=2)
        half_sketch = F.avg_pool2d(sketch, kernel_size=2, stride=2)
        half_style = style_embedding.view(b, self.style_embed_dim, 1, 1).expand(-1, -1, h // 2, w // 2)
        half_x = torch.cat([half_photo, half_sketch, half_style], dim=1)
        return self.net(x), self.half_scale_net(half_x)


def postprocess_sketch(
    sketch: Any,
    contrast_factor: float = 1.30,
    clip_dark: float = 0.08,
    clip_light: float = 0.92,
    sharpen: bool = True,
) -> Any:
    """
    Post-processes a generated sketch:
    1. Eliminates chromatic aberration / rainbow color fringing by converting to luminance grayscale.
    2. Stretches contrast so paper is clean bright white and sketch strokes are deep black ink.
    3. Optionally applies unsharp Laplacian kernel to sharpen fine line details.

    Accepts:
      - PyTorch Tensor in [-1, 1] of shape (B, 3, H, W) or (3, H, W)
      - NumPy array in [0, 1] of shape (H, W, 3) or (H, W)
    Returns:
      Same type and shape as input (RGB 3-channel).
    """
    import numpy as np

    if isinstance(sketch, torch.Tensor):
        orig_shape = sketch.shape
        t = sketch.clone()
        if t.dim() == 3:
            t = t.unsqueeze(0)
        # Shift [-1, 1] -> [0, 1]
        t = (t + 1.0) * 0.5
        # Luminance grayscale: (B, 1, H, W)
        gray = 0.299 * t[:, 0:1] + 0.587 * t[:, 1:2] + 0.114 * t[:, 2:3]

        # Contrast stretch: dark lines -> black, paper -> white
        stretched = torch.clamp((gray - clip_dark) / (clip_light - clip_dark), 0.0, 1.0)
        enhanced = torch.clamp((stretched - 0.5) * contrast_factor + 0.5, 0.0, 1.0)

        if sharpen:
            # 3x3 unsharp Laplacian sharpening filter
            kernel = torch.tensor(
                [[0.0, -0.2, 0.0], [-0.2, 1.8, -0.2], [0.0, -0.2, 0.0]],
                device=t.device,
                dtype=t.dtype,
            ).view(1, 1, 3, 3)
            sharpened = torch.nn.functional.conv2d(enhanced, kernel, padding=1)
            enhanced = torch.clamp(sharpened, 0.0, 1.0)

        # Broadcast back to 3 identical RGB channels (guarantees zero chromatic aberration)
        out_01 = enhanced.repeat(1, 3, 1, 1)
        # Rescale back to [-1, 1]
        out_tanh = out_01 * 2.0 - 1.0
        return out_tanh.squeeze(0) if len(orig_shape) == 3 else out_tanh
    else:
        arr = np.asarray(sketch, dtype=np.float32)
        is_3d = (arr.ndim == 3 and arr.shape[2] == 3)
        if is_3d:
            gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        else:
            gray = arr
        stretched = np.clip((gray - clip_dark) / (clip_light - clip_dark), 0.0, 1.0)
        enhanced = np.clip((stretched - 0.5) * contrast_factor + 0.5, 0.0, 1.0)
        if is_3d:
            return np.repeat(enhanced[..., np.newaxis], 3, axis=2)
        return enhanced
