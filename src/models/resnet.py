"""ResNet-18 wrapper -- no pretrained weights, head replaced for n_classes."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


# ResNet-18 fc layer input channels (architecture-fixed, not configurable)
_RESNET18_FC_IN = 512


def get_resnet18(n_classes: int = 100) -> nn.Module:
    """Return ResNet-18 trained from scratch with a fresh linear head.

    Args:
        n_classes: Number of output classes.

    Returns:
        ResNet-18 with fc replaced by Linear(_RESNET18_FC_IN, n_classes).
    """
    model    = resnet18(weights=None)
    model.fc = nn.Linear(_RESNET18_FC_IN, n_classes)
    return model
