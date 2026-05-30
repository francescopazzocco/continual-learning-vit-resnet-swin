"""CL training loop: iterates tasks, calls CLMethod hooks, logs and saves results."""

from __future__ import annotations

import csv
import os
from typing import List, Tuple

import numpy as np
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

from configs.default import Config
from src.artifacts import CKPT_TEMPLATE, METRICS_FILE, TRAIN_LOG_FILE
from src.cl.base import CLMethod
from src.engine import MAX_BATCHES_NO_LIMIT, SMOKE_MAX_BATCHES, evaluate, train_one_epoch
from src.metrics import compute_metrics

_TRAIN_LOG_FIELDS = ["task", "epoch", "train_loss", "val_acc"]
_METRICS_FIELDS   = ["metric", "value"]

# dtype for the per-task accuracy matrix R
_R_MATRIX_DTYPE  = np.float32

# Decimal precision for metrics CSV output
_METRICS_PRECISION = 6


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

    if len(splits) < cfg.n_tasks:
        print(f"[WARN] Only {len(splits)} splits available, expected {cfg.n_tasks}")
    n_tasks = min(len(splits), cfg.n_tasks)
    if smoke:
        n_tasks = min(n_tasks, 2)

    n_epochs    = 1 if smoke else cfg.epochs_per_task
    max_batches = SMOKE_MAX_BATCHES if smoke else MAX_BATCHES_NO_LIMIT

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
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, method, device, max_batches, use_amp,
                    grad_clip=cfg.grad_clip,
                )
                val_acc = evaluate(model, val_loader, device, max_batches)
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
                R[task_id, eval_id] = evaluate(model, eval_loader, device, max_batches)

            if not smoke:
                ckpt_path = os.path.join(run_dir, CKPT_TEMPLATE.format(t=task_id))
                torch.save({"task": task_id, "model": model.state_dict()}, ckpt_path)

    finally:
        if log_file:
            log_file.close()

    if not smoke:
        _write_metrics(R, run_dir)

    return R
