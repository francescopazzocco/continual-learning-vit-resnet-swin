"""ResNet-18 wrapper — no pretrained weights, head replaced for n_classes."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


def get_resnet18(n_classes: int = 100) -> nn.Module:
    """Return ResNet-18 trained from scratch with a fresh linear head.

    Args:
        n_classes: Number of output classes.

    Returns:
        ResNet-18 with fc replaced by Linear(512, n_classes).
    """
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, n_classes)
    return model
