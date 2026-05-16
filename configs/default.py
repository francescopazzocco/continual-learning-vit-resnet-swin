"""Single source of truth for all hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # Architecture and method
    arch:   str = "vit"         # "vit" | "resnet" -- architecture to evaluate
    method: str = "vanilla"     # "vanilla" | "ewc" | "er" -- continual learning method
    seed:   int = 0             # Random seed for reproducibility

    # Dataset / split
    n_tasks:          int = 10  # Standard Split-CIFAR-100 protocol (100 classes / 10 per task)
    classes_per_task: int = 10  # 100 / 10 = 10 tasks (standard class-incremental split)
    # n_classes is derived in __post_init__; do not set manually
    n_classes: int = field(init=False)

    # Training (joint pilot)
    epochs:     int = 200       # Standard for CIFAR training with cosine LR (He et al., ResNet)
    batch_size: int = 128       # Standard; fits on GPU memory with ViT-Small (384-dim)
    lr:       float = 0.1       # Standard SGD learning rate (Krizhevsky; Loshchilov & Hutter)
    momentum: float = 0.9       # SGD momentum; standard across literature (Ioffe & Szegedy)
    wd:       float = 5e-4      # Weight decay; standard for CIFAR (He et al., "Delving Deep")

    # Training (CL per-task)
    epochs_per_task: int = 50   # Per-task budget (total 500 epochs = 10 tasks) -- less than joint

    # EWC
    ewc_lambda: float       = 1000.0  # EWC regularization strength; standard (Kirkpatrick et al., 2017) @TODO run a grid search for optimal cifar100
    fisher_subsample: float = 0.2     # 20% of training set for Fisher estimation (memory constraint)
    fisher_batch_size: int  = 16      # Per-sample Fisher; smaller = more faithful mean(g^2) vs B*(mean(g))^2

    # ER
    er_buffer_size: int = 500   # Reservoir buffer capacity; 5 per class (standard ER protocol)

    # Paths
    data_root:    str = "./data"     # Default CIFAR download directory
    results_root: str = "./results"  # Output directory for checkpoints, CSVs, figures

    # Runtime
    device:      str = "cuda"   # Default to GPU; fallback to "cpu" if unavailable
    num_workers: int = 4        # DataLoader parallelism; change according to hardware specs

    # Augmentation
    randaug_n: int = 2          # RandAugment num_ops=2 (Cubuk et al., 2020 standard)
    randaug_m: int = 9          # RandAugment magnitude=9 (10-class scale, 0-9 maps to transforms)

    def __post_init__(self) -> None:
        self.n_classes = self.n_tasks * self.classes_per_task
