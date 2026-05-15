"""ViT-Small from scratch: conv stem -> 6 transformer blocks -> CLS head.

Stem: Conv2d(3->32, k=3, s=2) + Conv2d(32->dim, k=3, s=2) produces
(B, 64, dim) token sequence from 32x32 inputs (8x8 spatial grid).
Architecture follows CCT (Hassani et al., 2021) for the tokenizer.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


DIM = 384
N_HEADS = 6
N_BLOCKS = 6
MLP_RATIO = 4
STEM_MID = 32


class ConvStem(nn.Module):
    """Two-stage strided conv tokenizer: 32x32 -> 8x8 grid of dim-dim tokens."""

    def __init__(self, dim: int = DIM) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, STEM_MID, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(STEM_MID),
            nn.GELU(),
            nn.Conv2d(STEM_MID, dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 32, 32) -> (B, dim, 8, 8)
        x = self.layers(x)
        B, C, H, W = x.shape
        # (B, dim, 8, 8) -> (B, 64, dim)
        return x.flatten(2).transpose(1, 2)


class MLP(nn.Module):
    """Feed-forward block inside each transformer layer."""

    def __init__(self, dim: int, mlp_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN -> MHA -> residual -> LN -> MLP -> residual."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = MLP_RATIO) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTSmall(nn.Module):
    """ViT-Small from scratch for 32x32 inputs.

    Args:
        n_classes: Number of output classes.
        dim: Embedding dimension (default 384).
        n_heads: Attention heads per block (default 6).
        n_blocks: Transformer depth (default 6).
    """

    def __init__(
        self,
        n_classes: int = 100,
        dim: int = DIM,
        n_heads: int = N_HEADS,
        n_blocks: int = N_BLOCKS,
    ) -> None:
        super().__init__()
        n_patches = 64  # 8x8 grid from conv stem on 32x32 input
        self.stem = ConvStem(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.blocks = nn.Sequential(
            *[TransformerBlock(dim, n_heads) for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        tokens = self.stem(x)                               # (B, 64, dim)
        cls = self.cls_token.expand(B, -1, -1)              # (B, 1, dim)
        tokens = torch.cat([cls, tokens], dim=1)            # (B, 65, dim)
        tokens = tokens + self.pos_embed
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)
        return self.head(tokens[:, 0])                      # CLS token


def get_vit_small(n_classes: int = 100) -> ViTSmall:
    """Return ViT-Small from scratch for n_classes.

    Args:
        n_classes: Number of output classes.

    Returns:
        Initialized ViTSmall model.
    """
    return ViTSmall(n_classes=n_classes)
