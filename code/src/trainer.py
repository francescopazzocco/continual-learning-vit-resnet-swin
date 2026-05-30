"""Joint training loop used by the M1 pilot.

Joint training is continual learning with a single task and no anti-forgetting
mechanism, so it reuses src.engine via the Vanilla method.
"""

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
from src.cl.vanilla import Vanilla
from src.engine import MAX_BATCHES_NO_LIMIT, SMOKE_MAX_BATCHES, evaluate, train_one_epoch

SAVE_FILENAME = "{arch}_best.pt"
LOG_FILENAME = "{arch}_train.csv"

# Initial value for best accuracy tracker
_BEST_ACC_INIT = 0.0


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

    device  = torch.device(cfg.device)
    model   = model.to(device)
    use_amp = not smoke and device.type == "cuda"
    if not smoke and device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        model = torch.compile(model)

    method      = Vanilla()
    optimizer   = SGD(
        model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.wd,
    )
    n_epochs    = 1 if smoke else cfg.epochs
    scheduler   = CosineAnnealingLR(optimizer, T_max=n_epochs)
    max_batches = SMOKE_MAX_BATCHES if smoke else MAX_BATCHES_NO_LIMIT

    if not smoke:
        os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, SAVE_FILENAME.format(arch=arch_name))
    log_path  = os.path.join(out_dir, LOG_FILENAME.format(arch=arch_name))

    best_acc              = _BEST_ACC_INIT
    val_accs: List[float] = []

    log_file = None
    writer   = None
    try:
        if not smoke:
            log_file = open(log_path, "w", newline="")
            writer   = csv.DictWriter(log_file, fieldnames=["epoch", "train_loss", "val_acc"])
            writer.writeheader()

        bar = tqdm(range(n_epochs), desc=f"{arch_name}", unit="epoch")
        for epoch in bar:
            train_loss = train_one_epoch(
                model, train_loader, optimizer, method, device, max_batches, use_amp,
                grad_clip=cfg.grad_clip,
            )
            val_acc = evaluate(model, val_loader, device, max_batches)
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
