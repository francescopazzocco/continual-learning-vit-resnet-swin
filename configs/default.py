"""Single source of truth for all hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # Architecture and method
    arch: str = "vit"           # "vit" | "resnet"
    method: str = "vanilla"     # "vanilla" | "ewc" | "er"
    seed: int = 0

    # Dataset / split
    n_tasks: int = 10
    classes_per_task: int = 10
    n_classes: int = 100        # n_tasks * classes_per_task

    # Training (joint pilot)
    epochs: int = 200
    batch_size: int = 128
    lr: float = 0.1
    wd: float = 5e-4

    # Training (CL per-task)
    epochs_per_task: int = 50

    # EWC
    ewc_lambda: float = 1000.0
    fisher_subsample: float = 0.2

    # ER
    er_buffer_size: int = 500

    # Paths
    data_root: str = "./data"
    results_root: str = "./results"

    # Runtime
    device: str = "cuda"
    num_workers: int = 4

    # Augmentation
    randaug_n: int = 2
    randaug_m: int = 9
