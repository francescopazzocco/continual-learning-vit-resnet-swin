"""Joint training loop used by the M1 pilot."""

from __future__ import annotations

import csv
import os
from typing import List

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.default import Config

SAVE_FILENAME = "{arch}_best.pt"
LOG_FILENAME = "{arch}_train.csv"


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int = -1,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """Run one training epoch, return mean cross-entropy loss.

    Args:
        model: The model to train.
        loader: Training DataLoader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Target device.
        max_batches: If > 0, stop after this many batches (smoke mode).
        scaler: GradScaler for mixed-precision training; None disables AMP.

    Returns:
        Mean loss over processed batches.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    use_amp = scaler is not None
    for i, (x, y) in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss = criterion(model(x), y)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = -1,
) -> float:
    """Evaluate model, return top-1 accuracy.

    Args:
        model: The model to evaluate.
        loader: Validation DataLoader.
        device: Target device.
        max_batches: If > 0, stop after this many batches (smoke mode).

    Returns:
        Top-1 accuracy in [0, 1].
    """
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches > 0 and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    arch_name: str = "model",
    out_dir: str | None = None,
    smoke: bool = False,
) -> List[float]:
    """Train model for cfg.epochs with cosine LR, save best checkpoint and training log.

    Args:
        model: Model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        cfg: Hyperparameter config.
        arch_name: Used for checkpoint and log filenames.
        out_dir: Directory for checkpoint and CSV log. Defaults to
            ``{cfg.results_root}/pilot``.
        smoke: If True, run 1 epoch with 2 batches; skip disk writes.

    Returns:
        List of per-epoch validation accuracies.
    """
    if out_dir is None:
        out_dir = os.path.join(cfg.results_root, "pilot")

    device = torch.device(cfg.device)
    model = model.to(device)
    use_amp = not smoke and device.type == "cuda"
    if use_amp:
        torch.backends.cudnn.benchmark = True
        model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.wd,
    )
    scaler = torch.amp.GradScaler(enabled=use_amp)
    n_epochs = 1 if smoke else cfg.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    max_batches = 2 if smoke else -1

    if not smoke:
        os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, SAVE_FILENAME.format(arch=arch_name))
    log_path = os.path.join(out_dir, LOG_FILENAME.format(arch=arch_name))

    best_acc = 0.0
    val_accs: List[float] = []

    log_file = None
    writer = None
    try:
        if not smoke:
            log_file = open(log_path, "w", newline="")
            writer = csv.DictWriter(log_file, fieldnames=["epoch", "train_loss", "val_acc"])
            writer.writeheader()

        bar = tqdm(range(n_epochs), desc=f"{arch_name}", unit="epoch")
        for epoch in bar:
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, device, max_batches, scaler
            )
            val_acc = eval_epoch(model, val_loader, device, max_batches)
            scheduler.step()

            val_accs.append(val_acc)
            bar.set_postfix(loss=f"{train_loss:.4f}", val=f"{val_acc:.4f}")

            if writer:
                writer.writerow(
                    {"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc}
                )

            if val_acc > best_acc and not smoke:
                best_acc = val_acc
                tmp_path = ckpt_path + ".tmp"
                torch.save(
                    {"epoch": epoch, "model": model.state_dict(), "acc": val_acc},
                    tmp_path,
                )
                os.replace(tmp_path, ckpt_path)
    finally:
        if log_file:
            log_file.close()

    if smoke:
        print(f"  [{arch_name}] smoke val acc: {val_accs[-1]:.4f}")
    else:
        print(f"  [{arch_name}] best val acc: {best_acc:.4f} -> {ckpt_path}")

    return val_accs
