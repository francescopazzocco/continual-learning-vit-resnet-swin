"""Task-IL oracle evaluation on vanilla CL checkpoints.

Loads the final checkpoint (ckpt_task9.pt) from vanilla_s0 runs and re-evaluates
each task's val set with task identity known: argmax is restricted to the 10
logit positions belonging to that task.  This isolates the protocol effect and
proves that near-chance class-IL accuracy is not a training failure.

Writes:
    results/ablation/task_il.csv  -- arch, class_il_aa, task_il_aa
    prints comparison table to stdout

Usage:
    python scripts/eval_task_il.py [--device DEVICE]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from configs.default import Config
from src.artifacts import CKPT_TEMPLATE, read_scalar_metric
from src.checkpoint import load_model
from src.data.cifar100 import get_split_loaders
from src.models import ARCHS, build_model
from src.runtime import setup_device

_RESULTS_ROOT = Config().results_root  # single source of truth for output location
_RUNS_ROOT    = os.path.join(_RESULTS_ROOT, "runs")
_OUT_CSV      = os.path.join(_RESULTS_ROOT, "ablation", "task_il.csv")
_EVAL_SEED    = 0
_METHOD       = "vanilla"
_CSV_FIELDS   = ["arch", "class_il_aa", "task_il_aa"]


@torch.no_grad()
def _eval_task_il(
    arch: str,
    run_dir: str,
    splits: list,
    cfg: Config,
    device: torch.device,
) -> float:
    """Evaluate per-task checkpoints with known task identity.

    For each task t, loads ckpt_task{t}.pt and evaluates on task t's val set
    with argmax restricted to logit positions
    [t * classes_per_task, (t+1) * classes_per_task).

    Using the task-t checkpoint instead of the final checkpoint removes the
    confound of catastrophic forgetting; this isolates the protocol effect
    (class-IL near-chance arises from the missing task ID, not from bad
    representations).

    Args:
        arch: Architecture name for model construction.
        run_dir: Directory containing per-task checkpoints.
        splits: List of (train_loader, val_loader) from get_split_loaders.
        cfg: Config instance providing n_tasks and classes_per_task.
        device: Evaluation device.

    Returns:
        Task-IL average accuracy (mean of per-task accuracies).
    """
    per_task_acc: list[float] = []

    for t, (_, val_loader) in enumerate(splits):
        ckpt_path = os.path.join(run_dir, CKPT_TEMPLATE.format(t=t))
        if not os.path.exists(ckpt_path):
            per_task_acc.append(float("nan"))
            continue

        model = build_model(arch, cfg.n_classes).to(device)
        load_model(model, ckpt_path)
        model.eval()

        lo = t * cfg.classes_per_task
        hi = lo + cfg.classes_per_task

        correct = 0
        total   = 0
        for x, y in val_loader:
            x, y     = x.to(device), y.to(device)
            logits   = model(x)                        # (B, 100)
            masked   = logits[:, lo:hi]                # (B, 10)
            preds    = masked.argmax(dim=1) + lo       # shift back to global space
            correct += (preds == y).sum().item()
            total   += y.size(0)

        per_task_acc.append(correct / max(total, 1))

    valid = [a for a in per_task_acc if not (a != a)]  # exclude NaN
    return float(sum(valid) / len(valid)) if valid else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task-IL oracle evaluation")
    parser.add_argument("--device", default=None)
    args   = parser.parse_args()

    cfg    = Config()
    if args.device is not None:
        cfg.device = args.device
    device = setup_device(cfg)

    # Shared split loaders (val sets only used)
    cfg.seed = _EVAL_SEED
    splits   = get_split_loaders(cfg)

    archs = ARCHS
    rows  = []

    print("=== Task-IL ===")
    print(f"  checkpoint: {CKPT_TEMPLATE}  method: {_METHOD}  seed: {_EVAL_SEED}")
    print()
    print(f"  {'arch':<10}  {'class-IL AA':>12}  {'task-IL AA':>12}")
    print("  " + "-" * 38)

    for arch in archs:
        run_dir = os.path.join(_RUNS_ROOT, f"{arch}_{_METHOD}_s{_EVAL_SEED}")
        if not os.path.exists(run_dir):
            print(f"  [SKIP] {arch}: run directory not found at {run_dir}")
            continue

        class_il = read_scalar_metric(run_dir, "AA")
        task_il  = _eval_task_il(arch, run_dir, splits, cfg, device)

        print(f"  {arch:<10}  {class_il:>12.4f}  {task_il:>12.4f}")
        rows.append({
            "arch":        arch,
            "class_il_aa": f"{class_il:.6f}",
            "task_il_aa":  f"{task_il:.6f}",
        })

    print()

    os.makedirs(os.path.dirname(_OUT_CSV), exist_ok=True)
    with open(_OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {_OUT_CSV}")
    print("=== done ===")


if __name__ == "__main__":
    main()
