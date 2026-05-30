"""Model registry: maps architecture names to builders and per-arch metadata.

Single source of truth for which architectures exist, how to build them, where
their classification head lives, and which submodules to probe for CKA. Keeping
this here stops scripts from re-deriving architecture-specific knowledge.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import torch.nn as nn

from src.models.resnet import get_resnet18
from src.models.swin import get_swin_tiny
from src.models.vit import get_vit_small

_BUILDERS: Dict[str, Callable[..., nn.Module]] = {
    "vit":    get_vit_small,
    "resnet": get_resnet18,
    "swin":   get_swin_tiny,
}

ARCHS: List[str] = list(_BUILDERS)

# Attribute holding the classification head per architecture.
_HEAD_ATTR: Dict[str, str] = {
    "vit":    "head",
    "resnet": "fc",
    "swin":   "head",
}

# Submodule names to probe for layer-wise CKA. For Swin, "stages.0.0" resolves
# via get_submodule to stages[0][0].
PROBE_LAYERS: Dict[str, List[str]] = {
    "vit": [
        "stem",
        "blocks.0",
        "blocks.2",
        "blocks.4",
        "blocks.5",
        "norm",
    ],
    "resnet": [
        "layer1",
        "layer2",
        "layer3",
        "layer4",
    ],
    "swin": [
        "patch_embed",
        "stages.0.0",
        "stages.0.1",
        "patch_merging",
        "stages.1.0",
        "stages.1.5",
        "norm",
    ],
}


def build_model(arch: str, n_classes: int = 100) -> nn.Module:
    """Construct a model from scratch for the given architecture.

    Args:
        arch: One of ARCHS.
        n_classes: Output classes for the classification head.

    Returns:
        Freshly initialized model.

    Raises:
        ValueError: If arch is not a known architecture.
    """
    if arch not in _BUILDERS:
        raise ValueError(f"Unknown arch: {arch!r} (known: {ARCHS})")
    return _BUILDERS[arch](n_classes=n_classes)


def replace_head_with_identity(model: nn.Module, arch: str) -> int:
    """Replace the classification head with Identity and return its input dim.

    Args:
        model: Model with its original head intact.
        arch: Architecture name; selects which attribute holds the head.

    Returns:
        Feature dimension of the backbone output (the head's in_features).
    """
    attr = _HEAD_ATTR[arch]
    head = getattr(model, attr)
    feat_dim = head.in_features
    setattr(model, attr, nn.Identity())
    return feat_dim
