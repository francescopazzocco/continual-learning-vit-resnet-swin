"""Checkpoint loading shared across analysis and evaluation scripts.

Checkpoints are saved as {"task": int, "model": state_dict}. When the model was
wrapped by torch.compile, every state-dict key carries an "_orig_mod." prefix;
this module strips it so checkpoints load into uncompiled models.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

COMPILED_PREFIX = "_orig_mod."


def strip_compiled_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove the torch.compile "_orig_mod." prefix from state-dict keys.

    Args:
        state_dict: Raw state dict, possibly from a compiled model.

    Returns:
        State dict with the prefix removed from every key that had it.
    """
    n = len(COMPILED_PREFIX)
    return {
        (k[n:] if k.startswith(COMPILED_PREFIX) else k): v
        for k, v in state_dict.items()
    }


def load_model(
    model: nn.Module,
    ckpt_path: str,
    map_location: str = "cpu",
) -> None:
    """Load a saved checkpoint into model in place, handling compiled prefixes.

    Args:
        model: Target module to receive the weights.
        ckpt_path: Path to a checkpoint produced by the CL/joint trainers.
        map_location: Device mapping passed to torch.load.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    model.load_state_dict(strip_compiled_prefix(ckpt["model"]), strict=True)
