"""Linear probe evaluation of per-task representation quality.

Loads ckpt_task{t}.pt from vanilla_s0 runs for each architecture.  For each
task t, the backbone is frozen and a fresh Linear(feat_dim, 10) head is fitted
on task t's training features (pre-extracted, no gradient through backbone).
Val accuracy averaged across tasks gives the representation quality AA,
independent of head interference from the class-IL training objective.

Separates "classifier interference" (removed by the probe) from "representation
collapse" (what remains after the probe is fitted).

Writes:
    results/ablation/linear_probe.csv  -- arch, class_il_aa, probe_aa
    prints comparison table to stdout

Usage:
    python scripts/eval_linear_probe.py [--device DEVICE] [--smoke]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from configs.default import Config
from src.artifacts import CKPT_TEMPLATE, read_scalar_metric
from src.checkpoint import load_model
from src.data.cifar100 import get_split_loaders
from src.models import ARCHS, build_model, replace_head_with_identity
from src.runtime import setup_device

_RESULTS_ROOT = Config().results_root  # single source of truth for output location
_RUNS_ROOT    = os.path.join(_RESULTS_ROOT, "runs")
_OUT_CSV      = os.path.join(_RESULTS_ROOT, "ablation", "linear_probe.csv")
_EVAL_SEED    = 0
_METHOD       = "vanilla"
_CSV_FIELDS   = ["arch", "class_il_aa", "probe_aa"]

_PROBE_EPOCHS = 30
_PROBE_LR     = 1e-2
_PROBE_BATCH  = 256


@torch.no_grad()
def _extract_features(
    model: nn.Module,
    loader: DataLoader,
    task_offset: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward all batches through frozen backbone; remap labels to [0, 10).

    Args:
        model: Frozen backbone (head replaced with Identity).
        loader: DataLoader yielding (images, labels) in original 0-99 label space.
        task_offset: t * classes_per_task; subtracted from labels for the probe.
        device: Device to run the forward pass on.

    Returns:
        Tuple of (features, labels) tensors on CPU.
    """
    feats, labels = [], []
    for x, y in loader:
        feats.append(model(x.to(device)).cpu())
        labels.append(y - task_offset)
    return torch.cat(feats), torch.cat(labels)


def _fit_probe(
    feat_dim: int,
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    n_classes: int,
    device: torch.device,
) -> nn.Linear:
    """Train a linear head on pre-extracted features.

    Args:
        feat_dim:     Backbone output dimension.
        train_feats:  (N, feat_dim) float tensor on CPU.
        train_labels: (N,) long tensor in [0, n_classes) on CPU.
        n_classes: Number of probe output classes (classes_per_task = 10).
        device: Training device.

    Returns:
        Fitted nn.Linear with gradients disabled.
    """
    probe   = nn.Linear(feat_dim, n_classes).to(device)
    opt     = torch.optim.Adam(probe.parameters(), lr=_PROBE_LR)
    loss_fn = nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(train_feats, train_labels),
        batch_size=_PROBE_BATCH,
        shuffle=True,
    )

    probe.train()
    for _ in range(_PROBE_EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(probe(xb), yb).backward()
            opt.step()

    for p in probe.parameters():
        p.requires_grad = False
    return probe


@torch.no_grad()
def _eval_probe(
    probe: nn.Linear,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    device: torch.device,
) -> float:
    """Evaluate a fitted probe on pre-extracted val features.

    Args:
        probe: Fitted linear head.
        val_feats: (N, feat_dim) float tensor on CPU.
        val_labels: (N,) long tensor in [0, n_classes) on CPU.
        device: Device to run inference on.

    Returns:
        Accuracy in [0, 1].
    """
    probe.eval()
    preds = probe(val_feats.to(device)).argmax(dim=1).cpu()
    return (preds == val_labels).float().mean().item()


def _eval_arch(
    arch: str,
    run_dir: str,
    splits: list,
    cfg: Config,
    device: torch.device,
    smoke: bool,
) -> float:
    """Fit and evaluate one linear probe per task; return mean val accuracy.

    Args:
        arch: Architecture name.
        run_dir: Directory containing per-task checkpoints.
        splits: List of (train_loader, val_loader) from get_split_loaders.
        cfg: Config providing n_classes and classes_per_task.
        device: Eval device.
        smoke: If True, evaluate only the first task.

    Returns:
        Mean per-task probe accuracy (NaN tasks excluded from mean).
    """
    per_task_acc: list[float] = []
    tasks = splits[:1] if smoke else splits

    for t, (train_loader, val_loader) in enumerate(tasks):
        ckpt_path = os.path.join(run_dir, CKPT_TEMPLATE.format(t=t))
        if not os.path.exists(ckpt_path):
            per_task_acc.append(float("nan"))
            continue

        model = build_model(arch, cfg.n_classes).to(device)
        load_model(model, ckpt_path)

        feat_dim = replace_head_with_identity(model, arch)
        for p in model.parameters():
            p.requires_grad = False
        task_offset = t * cfg.classes_per_task

        train_feats, train_labels = _extract_features(
            model, train_loader, task_offset, device
        )
        val_feats, val_labels = _extract_features(
            model, val_loader, task_offset, device
        )

        probe = _fit_probe(
            feat_dim, train_feats, train_labels, cfg.classes_per_task, device
        )
        per_task_acc.append(_eval_probe(probe, val_feats, val_labels, device))

    valid = [a for a in per_task_acc if a == a]  # exclude NaN
    return float(sum(valid) / len(valid)) if valid else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Linear probe representation quality eval"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--smoke", action="store_true",
        help="Evaluate one arch (vit), one task only"
    )
    args = parser.parse_args()

    cfg = Config()
    if args.device is not None:
        cfg.device = args.device
    device = setup_device(cfg)

    cfg.seed = _EVAL_SEED
    splits   = get_split_loaders(cfg)

    archs = ["vit"] if args.smoke else ARCHS
    rows: list[dict] = []

    print("=== Linear probe representation eval ===")
    print(f"  method: {_METHOD}  seed: {_EVAL_SEED}  "
          f"probe: {_PROBE_EPOCHS} epochs  lr: {_PROBE_LR}")
    if args.smoke:
        print("  [SMOKE] vit only, task 0 only")
    print()
    print(f"  {'arch':<10}  {'class-IL AA':>12}  {'probe AA':>12}")
    print("  " + "-" * 38)

    for arch in archs:
        run_dir = os.path.join(_RUNS_ROOT, f"{arch}_{_METHOD}_s{_EVAL_SEED}")
        if not os.path.exists(run_dir):
            print(f"  [SKIP] {arch}: run directory not found at {run_dir}")
            continue

        class_il = read_scalar_metric(run_dir, "AA")
        probe_aa = _eval_arch(arch, run_dir, splits, cfg, device, args.smoke)

        print(f"  {arch:<10}  {class_il:>12.4f}  {probe_aa:>12.4f}")
        rows.append({
            "arch":        arch,
            "class_il_aa": f"{class_il:.6f}",
            "probe_aa":    f"{probe_aa:.6f}",
        })

    print()

    if not args.smoke:
        os.makedirs(os.path.dirname(_OUT_CSV), exist_ok=True)
        with open(_OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  -> {_OUT_CSV}")

    print("=== done ===")


if __name__ == "__main__":
    main()
