"""Experience Replay with reservoir sampling."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.cl.base import CLMethod

_EMPTY: torch.Tensor = torch.empty(0)


class ReservoirBuffer:
    """Fixed-size exemplar buffer using reservoir sampling.

    Maintains a uniform random sample over all samples seen so far.

    Args:
        max_size: Maximum number of exemplars to store.
    """

    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self._x: torch.Tensor | None = None
        self._y: torch.Tensor | None = None
        self._n_seen: int = 0

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Add a batch of CPU tensors to the buffer via reservoir sampling.

        Args:
            x: Input batch on CPU.
            y: Target batch on CPU (1-D integer labels).
        """
        for i in range(x.size(0)):
            self._n_seen += 1
            xi, yi = x[i : i + 1], y[i : i + 1]
            if self._x is None:
                self._x = xi.clone()
                self._y = yi.clone()
            elif len(self._x) < self.max_size:
                self._x = torch.cat([self._x, xi], dim=0)
                self._y = torch.cat([self._y, yi], dim=0)
            else:
                # Replace a random slot with probability max_size / n_seen.
                j = int(np.random.randint(0, self._n_seen))
                if j < self.max_size:
                    self._x[j] = xi[0]
                    self._y[j] = yi[0]

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return up to n uniformly sampled exemplars as CPU tensors.

        Args:
            n: Number of samples requested; capped at buffer size.

        Returns:
            (x, y) pair; both empty if the buffer is empty.
        """
        if self._x is None or len(self._x) == 0:
            return _EMPTY, _EMPTY
        n = min(n, len(self._x))
        idx = torch.randperm(len(self._x))[:n]
        return self._x[idx], self._y[idx]

    def __len__(self) -> int:
        return len(self._x) if self._x is not None else 0


class ER(CLMethod):
    """Experience Replay: concatenates replay samples into every training step.

    The buffer is updated with the original (pre-replay) batch after each
    optimizer step, so replayed exemplars are not re-added to the buffer.

    Args:
        buffer_size: Capacity of the ReservoirBuffer.
        device: Training device (used to move replay tensors on-device).
    """

    def __init__(self, buffer_size: int, device: torch.device) -> None:
        self.buffer = ReservoirBuffer(buffer_size)
        self.device = device

    def before_task(
        self, task_id: int, train_loader: DataLoader, model: nn.Module
    ) -> None:
        pass

    def prepare_batch(
        self, x: torch.Tensor, y: torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate current batch with replay samples from the buffer."""
        if len(self.buffer) == 0:
            return x, y
        rx, ry = self.buffer.sample(x.size(0))
        return (
            torch.cat([x, rx.to(device)], dim=0),
            torch.cat([y, ry.to(device)], dim=0),
        )

    def loss(
        self, logits: torch.Tensor, targets: torch.Tensor, model: nn.Module
    ) -> torch.Tensor:
        return F.cross_entropy(logits, targets)

    def after_step(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Update buffer with the original (pre-replay) batch on CPU."""
        self.buffer.update(x, y)

    def after_task(
        self, task_id: int, train_loader: DataLoader, model: nn.Module
    ) -> None:
        pass
