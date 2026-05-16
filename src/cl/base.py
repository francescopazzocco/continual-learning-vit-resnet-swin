"""Abstract base class for continual learning methods."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class CLMethod(ABC):
    """Hook interface for continual learning methods.

    The CL trainer calls hooks in this order per task:
        before_task -> [prepare_batch -> loss -> after_step] x batches -> after_task
    """

    @abstractmethod
    def before_task(
        self, task_id: int, train_loader: DataLoader, model: nn.Module
    ) -> None:
        """Called once before training starts on task_id.

        Args:
            task_id: Zero-based task index.
            train_loader: DataLoader for the current task's training set.
            model: Model that will be trained.
        """

    def prepare_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optionally augment the batch before the forward pass.

        Default: identity. Override in replay-based methods to mix in exemplars.

        Args:
            x: Input tensor already on device.
            y: Target tensor already on device.
            device: Training device.

        Returns:
            Possibly-augmented (x, y) pair, on device.
        """
        return x, y

    @abstractmethod
    def loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module,
    ) -> torch.Tensor:
        """Compute the training loss for this step.

        Args:
            logits: Model output of shape (batch, n_classes).
            targets: Ground-truth class indices.
            model: Current model (for regularization terms).

        Returns:
            Scalar loss tensor.
        """

    def after_step(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Called after the optimizer step with the *original* (pre-replay) batch.

        Default: no-op. Override in buffer-based methods to update exemplar storage.

        Args:
            x: Original input batch on CPU.
            y: Original target batch on CPU.
        """

    @abstractmethod
    def after_task(
        self, task_id: int, train_loader: DataLoader, model: nn.Module
    ) -> None:
        """Called once after all epochs on task_id complete.

        Args:
            task_id: Zero-based task index.
            train_loader: DataLoader for the completed task's training set.
            model: Model after training on this task.
        """
