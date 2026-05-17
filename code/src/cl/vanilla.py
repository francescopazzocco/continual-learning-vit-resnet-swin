"""Vanilla fine-tuning: standard cross-entropy, no continual learning mechanism."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.cl.base import CLMethod


class Vanilla(CLMethod):
    """Pass-through CLMethod; useful as a forgetting baseline."""

    def before_task(
        self, task_id: int, train_loader: DataLoader, model: nn.Module
    ) -> None:
        pass

    def loss(
        self, logits: torch.Tensor, targets: torch.Tensor, model: nn.Module
    ) -> torch.Tensor:
        return F.cross_entropy(logits, targets)

    def after_task(
        self, task_id: int, train_loader: DataLoader, model: nn.Module
    ) -> None:
        pass
