"""Swin-Tiny adapted for 32x32 inputs: 2 stages, patch=4, window=4.

Stage 0: 8x8 feature map, 4 non-overlapping 4x4 windows -- windowed attention.
Stage 1: 4x4 feature map, 1 window (window == map) -- global attention.
Forward: patch_embed -> stage0 -> patch_merging -> stage1 -> norm -> avg -> head.
Cite: Liu et al., "Swin Transformer" (ICCV 2021) for the base design.
"""

from __future__ import annotations

import torch
import torch.nn as nn


EMBED_DIM      = 192   # Swin-Tiny stage 1 dim; stage 0 dim=96 (before PatchMerging doubles channels)
DEPTHS         = [2, 6]  # Adapted 2-stage depth for 32x32 (original Swin-Tiny is [2,2,6,2] over 4 stages)
NUM_HEADS      = [6, 12] # Scales with embed_dim to keep dim/head=16 constant across stages
WINDOW_SIZE    = 4       # Matches 32x32 resolution; 8x8->4x4 windows keep computational cost manageable
PATCH_SIZE     = 4       # Swin-Tiny standard; 32/4=8x8 initial token grid (vs ViT-S=64 tokens)
MLP_RATIO      = 4.0     # FFN expansion ratio; standard across transformer literature
DROP_PATH_RATE = 0.1     # Linear stochastic depth schedule from 0 to 0.1 over all blocks
INPUT_SIZE     = 32      # CIFAR-100 resolution

N_PATCHES_H = INPUT_SIZE // PATCH_SIZE   # 8
N_PATCHES_W = INPUT_SIZE // PATCH_SIZE   # 8

# Standard deviation for trunc_normal_ initialization (from "Attention is All You Need")
_INIT_STD = 0.02

# Additive mask value for blocked attention positions (~-inf for softmax)
_ATTN_MASK_PAD = -100.0


def window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """Partition (B, H, W, C) into (B*nW, ws, ws, C) non-overlapping windows.

    Args:
        x: Feature map of shape (B, H, W, C). H and W must be divisible by ws.
        ws: Window size.

    Returns:
        Windows of shape (B * (H//ws) * (W//ws), ws, ws, C).
    """
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def window_reverse(windows: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
    """Reconstruct (B, H, W, C) from (B*nW, ws, ws, C) windows.

    Args:
        windows: Shape (B * nW, ws, ws, C).
        ws: Window size used in window_partition.
        H, W: Target spatial dimensions.

    Returns:
        Reconstructed tensor of shape (B, H, W, C).
    """
    B = int(windows.shape[0] / ((H // ws) * (W // ws)))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with learnable relative position bias.

    Args:
        dim: Input channel dimension.
        num_heads: Number of attention heads.
        window_size: Spatial window size (height == width).
    """

    def __init__(self, dim: int, num_heads: int, window_size: int) -> None:
        super().__init__()
        self.num_heads   = num_heads
        self.window_size = window_size
        self.scale       = (dim // num_heads) ** -0.5

        ws = window_size
        self.rel_pos_bias_table = nn.Parameter(
            torch.zeros((2 * ws - 1) * (2 * ws - 1), num_heads)
        )

        coords_h = torch.arange(ws)
        coords_w = torch.arange(ws)
        coords   = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flat = coords.flatten(1)                                     # (2, ws*ws)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]            # (2, ws*ws, ws*ws)
        rel = rel.permute(1, 2, 0).contiguous()                            # (ws*ws, ws*ws, 2)
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel[:, :, 0] *= 2 * ws - 1
        self.register_buffer("rel_pos_index", rel.sum(-1).long())          # (ws*ws, ws*ws)

        self.qkv  = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B_windows, N, C) where N = ws*ws.
            mask: (num_windows, N, N) additive attention mask, or None.

        Returns:
            (B_windows, N, C)
        """
        Bw, N, C = x.shape
        heads    = self.num_heads
        head_dim = C // heads

        qkv     = self.qkv(x).reshape(Bw, N, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                                            # (Bw, heads, N, head_dim)

        attn = (q * self.scale) @ k.transpose(-2, -1)                      # (Bw, heads, N, N)

        bias = self.rel_pos_bias_table[self.rel_pos_index.view(-1)].view(N, N, heads)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)

        if mask is not None:
            num_wins = mask.shape[0]
            attn     = attn.view(Bw // num_wins, num_wins, heads, N, N)
            attn     = attn + mask.unsqueeze(1).unsqueeze(0)
            attn     = attn.view(-1, heads, N, N)

        attn = attn.softmax(dim=-1)
        x    = (attn @ v).transpose(1, 2).reshape(Bw, N, C)
        return self.proj(x)


class StochasticDepth(nn.Module):
    """Drop entire residual path with probability p (Bernoulli + survival scaling).

    Args:
        p: Drop probability in [0, 1).
    """

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        survival = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(survival)
        noise.div_(survival)
        return x * noise


class SwinBlock(nn.Module):
    """Swin Transformer block: pre-LN, W-MSA (shift=False) or SW-MSA (shift=True), MLP.

    Args:
        dim: Token embedding dimension.
        num_heads: Attention heads.
        window_size: Spatial window size.
        shift: Use SW-MSA (cyclic shift by window_size // 2) when True.
        mlp_ratio: MLP hidden-dim multiplier.
        drop_path_rate: Stochastic depth rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift: bool,
        mlp_ratio: float,
        drop_path_rate: float,
        input_resolution: tuple[int, int],
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size  = window_size // 2 if shift else 0

        H, W         = input_resolution
        actual_shift = self.shift_size if (H > window_size and W > window_size) else 0
        if actual_shift > 0:
            mask = self._compute_attn_mask(H, W, actual_shift, torch.device("cpu"))
            self.register_buffer("_attn_mask", mask)
        else:
            self.register_buffer("_attn_mask", None)

        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )
        self.drop_path = StochasticDepth(drop_path_rate)

    def _compute_attn_mask(
        self, H: int, W: int, shift: int, device: torch.device
    ) -> torch.Tensor:
        """Build additive SW-MSA mask in shifted-coordinate space.

        Labels each shifted position by its cyclic-shift region so that tokens
        from different regions within the same window are blocked (-100 additive).
        """
        ws = self.window_size
        img_mask = torch.zeros(1, H, W, 1, device=device)
        h_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        w_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        label    = 0
        for hs in h_slices:
            for ws_ in w_slices:
                img_mask[:, hs, ws_, :] = label
                label += 1
        mask_wins = window_partition(img_mask, ws).view(-1, ws * ws)
        attn_mask = mask_wins.unsqueeze(1) - mask_wins.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, _ATTN_MASK_PAD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C)

        Returns:
            (B, H, W, C)
        """
        B, H, W, C = x.shape
        shortcut = x
        ws = self.window_size
        x  = self.norm1(x)

        if self._attn_mask is not None:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        assert H % ws == 0 and W % ws == 0, (
            f"Feature map ({H}x{W}) not divisible by window size ({ws})"
        )
        x_wins = window_partition(x, ws).view(-1, ws * ws, C)
        x_wins = self.attn(x_wins, mask=self._attn_mask).view(-1, ws, ws, C)
        x      = window_reverse(x_wins, ws, H, W)

        if self._attn_mask is not None:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """2x2 spatial downsampling: concatenate 2x2 neighbors then project C -> 2C.

    Args:
        dim: Input channel dimension.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm      = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C) with H and W even.

        Returns:
            (B, H/2, W/2, 2*C)
        """
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        return self.reduction(self.norm(torch.cat([x0, x1, x2, x3], dim=-1)))


class PatchEmbed(nn.Module):
    """Partition 32x32 image into PATCH_SIZE x PATCH_SIZE patches via strided Conv2d."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            3, EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE, bias=False
        )
        self.norm = nn.LayerNorm(EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)

        Returns:
            (B, H//PATCH_SIZE, W//PATCH_SIZE, EMBED_DIM)
        """
        x = self.proj(x).permute(0, 2, 3, 1)   # (B, H/p, W/p, C)
        return self.norm(x)


class SwinTiny32(nn.Module):
    """Swin-Tiny from scratch for 32x32 inputs.

    Two stages: stage 0 at 8x8 (windowed attention), stage 1 at 4x4 (global after merge).
    No absolute position embeddings; relative position bias is learned per WindowAttention.

    Args:
        n_classes: Number of output classes.
    """

    def __init__(self, n_classes: int = 100) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed()

        total_blocks = sum(DEPTHS)
        dp_rates     = [
            DROP_PATH_RATE * i / max(total_blocks - 1, 1) for i in range(total_blocks)
        ]

        stage_resolutions = [
            (N_PATCHES_H, N_PATCHES_W),
            (N_PATCHES_H // 2, N_PATCHES_W // 2),
        ]
        self.stages: nn.ModuleList = nn.ModuleList()
        block_idx = 0
        for stage_idx, (depth, n_heads) in enumerate(zip(DEPTHS, NUM_HEADS)):
            dim   = EMBED_DIM * (2 ** stage_idx)   # 192 for stage 0, 384 for stage 1
            stage = nn.ModuleList([
                SwinBlock(
                    dim=dim,
                    num_heads=n_heads,
                    window_size=WINDOW_SIZE,
                    shift=(i % 2 == 1),
                    mlp_ratio=MLP_RATIO,
                    drop_path_rate=dp_rates[block_idx + i],
                    input_resolution=stage_resolutions[stage_idx],
                )
                for i in range(depth)
            ])
            self.stages.append(stage)
            block_idx += depth

        self.patch_merging = PatchMerging(EMBED_DIM)
        final_dim = EMBED_DIM * 2
        self.norm = nn.LayerNorm(final_dim)
        self.head = nn.Linear(final_dim, n_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=_INIT_STD)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, WindowAttention):
                nn.init.trunc_normal_(m.rel_pos_bias_table, std=_INIT_STD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)         # (B, 8, 8, 96)
        for block in self.stages[0]:
            x = block(x)                # (B, 8, 8, 96)
        x = self.patch_merging(x)       # (B, 4, 4, 192)
        for block in self.stages[1]:
            x = block(x)               # (B, 4, 4, 192)
        x = self.norm(x)
        x = x.mean(dim=(1, 2))         # global average pool -> (B, 192)
        return self.head(x)


def get_swin_tiny(n_classes: int = 100) -> SwinTiny32:
    """Return Swin-Tiny from scratch for 32x32 inputs.

    Args:
        n_classes: Number of output classes.

    Returns:
        Initialized SwinTiny32 model.
    """
    return SwinTiny32(n_classes=n_classes)
