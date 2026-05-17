"""CL training loop: iterates tasks, calls CLMethod hooks, logs and saves results."""

from __future__ import annotations

import csv
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.default import Config
from src.cl.base import CLMethod
from src.metrics import compute_metrics

TRAIN_LOG_FILE = "train_log.csv"
METRICS_FILE   = "metrics.csv"
CKPT_TEMPLATE  = "ckpt_task{t}.pt"

_TRAIN_LOG_FIELDS = ["task", "epoch", "train_loss", "val_acc"]
_METRICS_FIELDS   = ["metric", "value"]

# Default sentinel for "no batch limit" (used by max_batches parameter)
_MAX_BATCHES_NO_LIMIT = -1

# Number of batches per epoch in smoke mode
_SMOKE_MAX_BATCHES    = 2

# Minimum value for denominators to avoid division by zero
_MIN_DIVISOR          = 1

# Attention mask fill values: padded positions get -100 (~ -inf for softmax), valid positions get 0
_ATTN_MASK_PAD   = -100.0
_ATTN_MASK_VALID = 0.0

# dtype for the per-task accuracy matrix R
_R_MATRIX_DTYPE  = np.float32

# Decimal precision for metrics CSV output
_METRICS_PRECISION = 6


def _train_task_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    method: CLMethod,
    device: torch.device,
    max_batches: int,
    use_amp: bool,
    grad_clip: float = 0.0,
) -> float:
    """Train one epoch for a single CL task; return mean batch loss.

    Args:
        model: Model in train mode.
        loader: DataLoader for the current task.
        optimizer: Optimizer instance.
        method: CLMethod providing prepare_batch, loss, and after_step hooks.
        device: Training device.
        max_batches: Stop after this many batches when > 0 (smoke mode).
        use_amp: If True, wrap forward in bfloat16 autocast.

    Returns:
        Mean cross-entropy loss over processed batches.
    """
    model.train()
    total_loss = torch.zeros(1, device=device)
    n_batches  = 0

    for i, (x, y) in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break

        x_orig, y_orig = x.to(device), y.to(device)
        x_aug, y_aug   = method.prepare_batch(x_orig, y_orig, device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x_aug)
            loss   = method.loss(logits, y_aug, model)
        loss.backward()
        if grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Pass original (pre-replay) CPU batch so ER can update its buffer.
        method.after_step(x, y)

        total_loss += loss.detach()
        n_batches  += 1

    return (total_loss / max(n_batches, _MIN_DIVISOR)).item()


def _eval_task(
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


def _write_metrics(R: np.ndarray, run_dir: str) -> None:
    """Write the R matrix and scalar summary metrics to metrics.csv.

    Args:
        R: Accuracy matrix of shape (n_tasks, n_tasks).
        run_dir: Destination directory.
    """
    m   = compute_metrics(R)
    T   = R.shape[0]
    fmt = f"{{:.{_METRICS_PRECISION}f}}"
    rows: list[dict[str, str]] = [
        {"metric": "AA",  "value": fmt.format(m["AA"])},
        {"metric": "BWT", "value": fmt.format(m["BWT"])},
        {"metric": "AF",  "value": fmt.format(m["AF"])},
    ]
    for i in range(T):
        for j in range(i + 1):
            rows.append({"metric": f"R_{i}_{j}", "value": fmt.format(R[i, j])})

    out_path = os.path.join(run_dir, METRICS_FILE)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_METRICS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_cl(
    model: nn.Module,
    splits: List[Tuple[DataLoader, DataLoader]],
    method: CLMethod,
    cfg: Config,
    run_dir: str,
    smoke: bool = False,
) -> np.ndarray:
    """Run the full class-IL continual learning loop.

    Args:
        model: Model with a single n_classes-wide linear head; on CPU at entry.
        splits: List of (train_loader, val_loader) pairs, one per task.
        method: CLMethod instance providing all hooks.
        cfg: Hyperparameter config.
        run_dir: Directory for train_log.csv, metrics.csv, and per-task checkpoints.
            No directory is created and no files are written in smoke mode.
        smoke: If True, run 2 tasks / 1 epoch / 2 batches; skip all disk writes.

    Returns:
        R: float32 ndarray of shape (n_tasks, n_tasks).
           R[i, j] = accuracy on task j after training task i (j <= i).
    """
    device  = torch.device(cfg.device)
    model   = model.to(device)

    use_amp = not smoke and device.type == "cuda"
    if not smoke and device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        model = torch.compile(model)

    n_tasks = min(len(splits), cfg.n_tasks)
    if smoke:
        n_tasks = min(n_tasks, 2)

    n_epochs    = 1 if smoke else cfg.epochs_per_task
    max_batches = _SMOKE_MAX_BATCHES if smoke else _MAX_BATCHES_NO_LIMIT

    R = np.zeros((n_tasks, n_tasks), dtype=_R_MATRIX_DTYPE)

    if not smoke:
        os.makedirs(run_dir, exist_ok=True)

    log_file   = None
    log_writer = None
    try:
        if not smoke:
            log_path   = os.path.join(run_dir, TRAIN_LOG_FILE)
            log_file   = open(log_path, "w", newline="")
            log_writer = csv.DictWriter(log_file, fieldnames=_TRAIN_LOG_FIELDS)
            log_writer.writeheader()

        for task_id in range(n_tasks):
            train_loader, val_loader = splits[task_id]
            method.before_task(task_id, train_loader, model)

            optimizer = SGD(
                model.parameters(),
                lr=cfg.lr,
                momentum=cfg.momentum,
                weight_decay=cfg.wd,
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

            bar = tqdm(
                range(n_epochs),
                desc=f"task {task_id}/{n_tasks - 1}",
                unit="epoch",
                leave=False,
            )
            for epoch in bar:
                train_loss = _train_task_epoch(
                    model, train_loader, optimizer, method, device, max_batches, use_amp,
                    grad_clip=cfg.grad_clip,
                )
                val_acc = _eval_task(model, val_loader, device, max_batches)
                scheduler.step()
                bar.set_postfix(loss=f"{train_loss:.4f}", val=f"{val_acc:.4f}")

                if log_writer:
                    log_writer.writerow({
                        "task": task_id,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_acc": val_acc,
                    })

            method.after_task(task_id, train_loader, model)

            for eval_id in range(task_id + 1):
                _, eval_loader = splits[eval_id]
                R[task_id, eval_id] = _eval_task(model, eval_loader, device, max_batches)

            if not smoke:
                ckpt_path = os.path.join(run_dir, CKPT_TEMPLATE.format(t=task_id))
                torch.save({"task": task_id, "model": model.state_dict()}, ckpt_path)

    finally:
        if log_file:
            log_file.close()

    if not smoke:
        _write_metrics(R, run_dir)

    return R
