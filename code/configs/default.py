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
    lr:       float = 0.1       # Empirically tuned; standard SGD starting point for CIFAR
    momentum: float = 0.9       # SGD momentum; standard across literature (Ioffe & Szegedy)
    wd:       float = 5e-4      # Weight decay; standard for CIFAR (He et al., "Delving Deep")

    # Training (CL per-task)
    epochs_per_task: int = 50   # Per-task budget (total 500 epochs = 10 tasks) -- less than joint

    # EWC
    ewc_lambda:       float = 1000.0  # Kirkpatrick et al. (2017) baseline; validated by ablation (doc/hyperparameter_choices.md)
    fisher_subsample: float = 0.2     # 20% of train set (~1000 imgs/task); ablation confirms stable estimates vs 0.05-0.5
    fisher_batch_size:  int = 16      # Batch Fisher (16x speedup vs per-sample); Jensen underestimate, but ordering preserved

    # ER
    er_buffer_size: int = 500   # Reservoir buffer capacity; 5 per class (standard ER protocol)

    # Paths
    data_root:    str = "../data"     # CIFAR download dir; outside code/ (run from code/)
    results_root: str = "../results"  # Checkpoints, CSVs, figures; outside code/ (run from code/)

    # Runtime
    device:      str = "cuda"   # Default to GPU; fallback to "cpu" if unavailable
    num_workers: int = 4        # DataLoader parallelism; change according to hardware specs
    grad_clip: float = 1.0      # Max gradient norm; required for ViT stability, especially under EWC penalty

    # Augmentation
    randaug_n: int = 2          # RandAugment num_ops=2 (Cubuk et al., 2020 standard)
    randaug_m: int = 9          # RandAugment magnitude=9 (10-class scale, 0-9 maps to transforms)

    def __post_init__(self) -> None:
        self.n_classes = self.n_tasks * self.classes_per_task
        if self.n_classes != 100:
            raise ValueError(
                f"n_tasks * classes_per_task must equal 100 for Split-CIFAR-100 "
                f"(got {self.n_tasks} * {self.classes_per_task} = {self.n_classes})"
            )
