"""CIFAR-100 data loaders for joint training and class-IL split protocol."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from configs.default import Config

_CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
_CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def _train_transform(cfg: Config) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=cfg.randaug_n, magnitude=cfg.randaug_m),
        transforms.ToTensor(),
        transforms.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
    ])


def _val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD),
    ])


def get_joint_loaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) over the full CIFAR-100 dataset."""
    pin = cfg.device == "cuda"
    train_ds = datasets.CIFAR100(
        root=cfg.data_root, train=True, download=True,
        transform=_train_transform(cfg),
    )
    val_ds = datasets.CIFAR100(
        root=cfg.data_root, train=False, download=True,
        transform=_val_transform(),
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin,
        persistent_workers=cfg.num_workers > 0,
    )
    return train_loader, val_loader


def get_split_loaders(cfg: Config) -> List[Tuple[DataLoader, DataLoader]]:
    """Return list of (train_loader, val_loader) pairs for each CL task.

    Labels are kept in the original 0-99 space (no per-task remapping).
    Task t covers classes [t*classes_per_task, (t+1)*classes_per_task).
    """
    train_ds = datasets.CIFAR100(
        root=cfg.data_root, train=True, download=True,
        transform=_train_transform(cfg),
    )
    val_ds = datasets.CIFAR100(
        root=cfg.data_root, train=False, download=True,
        transform=_val_transform(),
    )

    pin = cfg.device == "cuda"
    train_targets = torch.tensor(train_ds.targets)
    val_targets = torch.tensor(val_ds.targets)

    splits: List[Tuple[DataLoader, DataLoader]] = []
    for t in range(cfg.n_tasks):
        lo = t * cfg.classes_per_task
        hi = lo + cfg.classes_per_task

        train_idx = (train_targets >= lo) & (train_targets < hi)
        val_idx = (val_targets >= lo) & (val_targets < hi)

        t_loader = DataLoader(
            Subset(train_ds, train_idx.nonzero(as_tuple=True)[0].tolist()),
            batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=pin,
        )
        v_loader = DataLoader(
            Subset(val_ds, val_idx.nonzero(as_tuple=True)[0].tolist()),
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=pin,
        )
        splits.append((t_loader, v_loader))

    return splits
