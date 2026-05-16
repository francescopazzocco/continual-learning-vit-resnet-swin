"""Experience Replay with reservoir sampling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.cl.base import CLMethod

_EMPTY: torch.Tensor = torch.empty(0)


class ReservoirBuffer:
    """Fixed-size exemplar buffer using reservoir sampling.

    Pre-allocates storage on the first update call to avoid repeated torch.cat
    reallocations. The fill phase is vectorized; the reservoir phase uses an
    O(1) in-place slot assignment instead of a copy-and-extend.

    Args:
        max_size: Maximum number of exemplars to store.
    """

    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self._x: torch.Tensor | None = None
        self._y: torch.Tensor | None = None
        self._size: int = 0    # valid entries in [0, _size)
        self._n_seen: int = 0

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Add a batch of CPU tensors to the buffer via reservoir sampling.

        Args:
            x: Input batch on CPU.
            y: Target batch on CPU (1-D integer labels).
        """
        x, y = x.cpu(), y.cpu()
        B = x.size(0)

        # Lazy init: allocate fixed-size storage from the first batch's shape.
        if self._x is None:
            self._x = torch.empty((self.max_size, *x.shape[1:]), dtype=x.dtype)
            self._y = torch.empty(self.max_size, dtype=y.dtype)

        # Phase 1: fill empty slots with a single vectorized slice write.
        n_free = self.max_size - self._size
        n_fill = min(n_free, B)
        if n_fill > 0:
            self._x[self._size : self._size + n_fill] = x[:n_fill]
            self._y[self._size : self._size + n_fill] = y[:n_fill]
            self._size += n_fill
            self._n_seen += n_fill
            x, y = x[n_fill:], y[n_fill:]
            B -= n_fill

        # Phase 2: reservoir sampling — vectorised over the remaining batch.
        if B > 0:
            n0 = self._n_seen
            upper = torch.arange(n0 + 1, n0 + B + 1, dtype=torch.float32)
            slots = (torch.rand(B) * upper).long()  # slot_i ~ Uniform[0, n0+i+1)
            keep = slots < self.max_size
            if keep.any():
                src = keep.nonzero(as_tuple=True)[0]
                self._x[slots[src]] = x[src]
                self._y[slots[src]] = y[src]
            self._n_seen = n0 + B

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return up to n uniformly sampled exemplars as CPU tensors.

        Args:
            n: Number of samples requested; capped at buffer fill level.

        Returns:
            (x, y) pair; both empty if the buffer has no valid entries.
        """
        if self._size == 0:
            return _EMPTY, _EMPTY
        n = min(n, self._size)
        idx = torch.randperm(self._size)[:n]
        return self._x[idx], self._y[idx]

    def __len__(self) -> int:
        return self._size


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
