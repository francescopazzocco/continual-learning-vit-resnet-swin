"""Device and reproducibility setup shared by all entry-point scripts."""

from __future__ import annotations

import torch

from configs.default import Config


def setup_device(cfg: Config) -> torch.device:
    """Resolve the training device, falling back to CPU if CUDA is unavailable.

    Also enables TF32 matmuls (high precision) for the FP32 GEMMs used by the
    transformer and conv backbones. Mutates cfg.device on fallback so the rest
    of the run sees the effective device.

    Args:
        cfg: Config whose device field selects the requested device.

    Returns:
        The resolved torch.device.
    """
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")
        cfg.device = "cpu"
    torch.set_float32_matmul_precision("high")
    return torch.device(cfg.device)


def set_seed(seed: int) -> None:
    """Seed CPU and all CUDA devices for reproducibility.

    Args:
        seed: Random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
