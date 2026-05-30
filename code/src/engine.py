"""Shared train/eval primitives for the joint pilot and the CL loop.

Joint training is the single-task special case of continual learning with the
Vanilla method, so both loops share one train-epoch and one eval routine here
rather than re-implementing them.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.cl.base import CLMethod

# Sentinel for "no batch limit" used by max_batches.
MAX_BATCHES_NO_LIMIT = -1

# Batches per epoch in smoke mode.
SMOKE_MAX_BATCHES = 2

# Floor for denominators to avoid division by zero.
_MIN_DIVISOR = 1


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    method: CLMethod,
    device: torch.device,
    max_batches: int,
    use_amp: bool,
    grad_clip: float = 0.0,
) -> float:
    """Train one epoch through a CLMethod's hooks; return mean batch loss.

    Args:
        model: Model in train mode.
        loader: DataLoader for the current task (or full set for joint training).
        optimizer: Optimizer instance.
        method: CLMethod providing prepare_batch, loss, and after_step hooks.
            Use Vanilla for plain cross-entropy joint training.
        device: Training device.
        max_batches: Stop after this many batches when > 0 (smoke mode).
        use_amp: If True, wrap the forward pass in bfloat16 autocast.
        grad_clip: Max gradient norm; 0 disables clipping.

    Returns:
        Mean loss over processed batches.
    """
    model.train()
    total_loss = torch.zeros(1, device=device)
    n_batches  = 0

    for i, (x, y) in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break

        x_gpu, y_gpu = x.to(device), y.to(device)
        x_aug, y_aug = method.prepare_batch(x_gpu, y_gpu, device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x_aug)
            loss   = method.loss(logits, y_aug, model)
        loss.backward()
        if grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Pass the original (pre-replay) CPU batch so ER can update its buffer.
        method.after_step(x, y)

        total_loss += loss.detach()
        n_batches  += 1

    return (total_loss / max(n_batches, _MIN_DIVISOR)).item()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> float:
    """Return top-1 accuracy on loader.

    Args:
        model: Model to evaluate.
        loader: Validation DataLoader.
        device: Evaluation device.
        max_batches: Stop after this many batches when > 0.

    Returns:
        Top-1 accuracy in [0, 1].
    """
    model.eval()
    correct = torch.zeros(1, device=device, dtype=torch.long)
    total   = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches > 0 and i >= max_batches:
                break
            x, y     = x.to(device), y.to(device)
            correct += (model(x).argmax(dim=1) == y).sum()
            total   += y.size(0)
    return correct.item() / max(total, _MIN_DIVISOR)
