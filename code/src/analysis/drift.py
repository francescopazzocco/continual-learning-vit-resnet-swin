"""Layer-wise L2 weight drift between CL task checkpoints."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def snapshot(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Capture a detached CPU clone of all named parameters.

    Args:
        model: Source model (any device).

    Returns:
        Dict mapping parameter name to float32 CPU tensor clone.
    """
    return {
        name: param.detach().cpu().float().clone()
        for name, param in model.named_parameters()
    }


def compute_drift(
    ref: Dict[str, torch.Tensor],
    current: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Compute per-parameter L2 distance between two weight snapshots.

    Args:
        ref: Reference snapshot (e.g. after task 0), from snapshot().
        current: Current snapshot (e.g. after task t), from snapshot().

    Returns:
        Dict mapping parameter name to scalar L2 distance.
        Keys present in ref but absent in current are omitted.
    """
    result: Dict[str, float] = {}
    for name, ref_w in ref.items():
        if name in current:
            result[name] = (current[name] - ref_w).norm().item()
    return result
