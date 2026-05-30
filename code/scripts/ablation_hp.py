"""HP ablation: mini-grid to validate EWC and ER choices before the full CL grid.

Protocol: ResNet-18, 3 tasks, 10 epochs/task, seed=0.
ResNet is chosen because it is faster and more stable than ViT for a
relative comparison; ViT-specific caveats are noted in hyperparameter_choices.md.

Metrics (AA, BWT, AF) are printed as a compact table per HP and saved to
results/ablation/<hp_name>.csv.

Usage:
    python scripts/ablation_hp.py [--target {ewc_lambda,fisher_subsample,
                                             fisher_batch_size,er_buffer_size,all}]
                                  [--smoke] [--device DEVICE] [--num_workers N]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from configs.default import Config
from src.cl import build_method
from src.cl_trainer import run_cl
from src.data.cifar100 import get_split_loaders
from src.metrics import compute_metrics
from src.models import build_model
from src.runtime import set_seed, setup_device

# Ablation protocol -- intentionally below full-grid values to keep wall time short.
# 3 tasks gives 2 BWT observations (R[1,0] and R[2,{0,1}]).
# 10 epochs/task is enough for convergence signal at 5x the speed of the 50-epoch grid.
_ABL_N_TASKS     = 3
_ABL_N_EPOCHS    = 10
_ABL_SEED        = 0
_SMOKE_N_TASKS   = 2
_SMOKE_N_EPOCHS  = 1
_SMOKE_MAX_BATCH = 2

# HP grids: 4 values each -> ~16 ResNet runs total, ~15-20 min on RTX 5070 Ti.
_GRIDS: dict[str, list] = {
    "ewc_lambda":        [100.0, 500.0, 1000.0, 5000.0],
    "fisher_subsample":  [0.05, 0.10, 0.20, 0.50],
    "fisher_batch_size": [1, 8, 16, 64],
    "er_buffer_size":    [100, 200, 500, 1000],
}

# CL method that exercises each HP
_HP_METHOD: dict[str, str] = {
    "ewc_lambda":        "ewc",
    "fisher_subsample":  "ewc",
    "fisher_batch_size": "ewc",
    "er_buffer_size":    "er",
}

_CSV_FIELDS = ["hp", "value", "AA", "BWT", "AF"]


def _run_cell(
    hp_name: str,
    val: float | int,
    cfg_base: Config,
    device: torch.device,
    splits: list,
    ablation_root: str,
    smoke: bool,
) -> dict:
    """Run one HP cell; return a result row dict.

    Creates a fresh Config for each cell so HP settings do not bleed across runs.
    n_classes stays 100 because Config.__post_init__ ran with n_tasks=10 at init
    time; overriding n_tasks afterwards does not re-trigger __post_init__.

    Args:
        hp_name: Name of the HP being swept (e.g. "ewc_lambda").
        val: Value to test for this cell.
        cfg_base: Template config (device, num_workers taken from here).
        device: Training device.
        splits: Pre-sliced list of (train, val) loader pairs.
        ablation_root: Root directory for run output.
        smoke: If True, run in smoke mode (2 tasks, 1 epoch, 2 batches).

    Returns:
        Dict with keys: hp, value, AA, BWT, AF.
    """
    cfg = Config()
    cfg.n_tasks         = _SMOKE_N_TASKS if smoke else _ABL_N_TASKS
    cfg.epochs_per_task = _SMOKE_N_EPOCHS if smoke else _ABL_N_EPOCHS
    cfg.seed            = _ABL_SEED
    cfg.device          = cfg_base.device
    cfg.num_workers     = cfg_base.num_workers
    setattr(cfg, hp_name, val)

    method_name = _HP_METHOD[hp_name]
    method      = build_method(method_name, cfg, device)
    model       = build_model("resnet", cfg.n_classes)

    label   = str(val).replace(".", "_")
    run_dir = os.path.join(ablation_root, hp_name, label)

    R = run_cl(model, splits, method, cfg, run_dir, smoke=smoke)
    m = compute_metrics(R)
    return {"hp": hp_name, "value": val, "AA": m["AA"], "BWT": m["BWT"], "AF": m["AF"]}


def _ablate_hp(
    hp_name: str,
    cfg_base: Config,
    device: torch.device,
    splits: list,
    ablation_root: str,
    smoke: bool,
) -> list[dict]:
    """Run the full grid for one HP; print table and return result rows.

    Args:
        hp_name:  HP to sweep.
        cfg_base: Template config.
        device:   Training device.
        splits:   Task splits for the ablation protocol.
        ablation_root: Root directory for output.
        smoke: If True, smoke mode (fast, no meaningful metrics).

    Returns:
        List of result row dicts, one per grid value.
    """
    method = _HP_METHOD[hp_name]
    grid   = _GRIDS[hp_name]

    n_tasks  = _SMOKE_N_TASKS  if smoke else _ABL_N_TASKS
    n_epochs = _SMOKE_N_EPOCHS if smoke else _ABL_N_EPOCHS
    print(f"\n=== {hp_name}  [{method}]  {n_tasks} tasks / {n_epochs} epochs ===")

    col_w = max(len(str(v)) for v in grid) + 2
    header = f"  {'value':>{col_w}}   {'AA':>8}   {'BWT':>8}   {'AF':>8}"
    print(header)
    print("  " + "-" * (col_w + 32))

    rows = []
    for val in grid:
        set_seed(_ABL_SEED)
        row = _run_cell(hp_name, val, cfg_base, device, splits, ablation_root, smoke)
        rows.append(row)
        tag = "[SMOKE]" if smoke else ""
        print(
            f"  {str(val):>{col_w}}   {row['AA']:8.4f}   "
            f"{row['BWT']:8.4f}   {row['AF']:8.4f}  {tag}"
        )

    return rows


def _save_csv(rows: list[dict], path: str) -> None:
    """Write result rows to a CSV file.

    Args:
        rows: List of result dicts.
        path: Output file path.
    """
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HP ablation mini-grid (ResNet-18)")
    p.add_argument(
        "--target", default="all",
        choices=list(_GRIDS) + ["all"],
        help="HP to ablate; 'all' runs every grid sequentially (default: all)",
    )
    p.add_argument("--smoke", action="store_true",
                   help="2 tasks / 1 epoch / 2 batches; validates script only")
    p.add_argument("--device", default=None)
    p.add_argument("--num_workers", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_base = Config()
    if args.device      is not None:
        cfg_base.device      = args.device
    if args.num_workers is not None:
        cfg_base.num_workers = args.num_workers

    device = setup_device(cfg_base)

    # Load full split loaders once; slice to ablation task count.
    # Seed before loading to keep task-class assignment identical to the full grid.
    set_seed(_ABL_SEED)
    splits_all = get_split_loaders(cfg_base)
    n_tasks    = _SMOKE_N_TASKS if args.smoke else _ABL_N_TASKS
    splits     = splits_all[:n_tasks]

    ablation_root = os.path.join(cfg_base.results_root, "ablation")
    os.makedirs(ablation_root, exist_ok=True)

    targets = list(_GRIDS) if args.target == "all" else [args.target]

    print(
        f"=== HP Ablation | arch=resnet | seed={_ABL_SEED} | "
        f"smoke={args.smoke} | {n_tasks} tasks / "
        f"{_SMOKE_N_EPOCHS if args.smoke else _ABL_N_EPOCHS} epochs/task ==="
    )

    for hp_name in targets:
        rows = _ablate_hp(hp_name, cfg_base, device, splits, ablation_root, args.smoke)
        if not args.smoke:
            csv_path = os.path.join(ablation_root, f"{hp_name}.csv")
            _save_csv(rows, csv_path)
            print(f"  -> {csv_path}")

    print(f"\n[OK] Ablation complete. Results in {ablation_root}/")


if __name__ == "__main__":
    main()
