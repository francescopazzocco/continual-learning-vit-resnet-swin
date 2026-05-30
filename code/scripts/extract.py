#!/usr/bin/env python3
"""Extract CKA matrices and weight-drift arrays from CL run checkpoints.

For each run directory under results/runs/, loads per-task checkpoints,
computes pairwise CKA between task representations and per-parameter L2
drift from the task-0 checkpoint, then writes:

    results/features/{run_name}/cka.npz
    results/features/{run_name}/drift.npz

CKA keys:  layer name with '.' replaced by '_' (e.g. "blocks_5").
Drift keys: param name with '.' replaced by '_' (e.g. "blocks_0_attn_proj_weight").
            Each value is a float32 array of shape (T,): drift[t] = L2(w_t - w_0).

Usage:
    python scripts/extract.py                     # all 27 runs
    python scripts/extract.py --smoke             # 2 runs, 2 ckpts, 1 probe batch
    python scripts/extract.py --runs vit_vanilla_s0 swin_er_s1
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torchvision.datasets as tv_datasets
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default import Config
from src.analysis.cka import between_task_cka
from src.analysis.drift import compute_drift, snapshot
from src.artifacts import CKPT_TEMPLATE
from src.checkpoint import load_model
from src.data.cifar100 import eval_transform
from src.models import PROBE_LAYERS, build_model

# Number of probe samples taken from task-0 val set for CKA computation
_PROBE_SAMPLES  = 200
_PROBE_BATCH_SZ = 50

_RESULTS_ROOT   = Config().results_root  # single source of truth for output location
_RUNS_ROOT      = os.path.join(_RESULTS_ROOT, "runs")
_FEATURES_ROOT  = os.path.join(_RESULTS_ROOT, "features")


def _get_probe_loader(data_root: str, n_samples: int, num_workers: int) -> DataLoader:
    """Return a DataLoader over the first n_samples of task-0 val images.

    Task 0 covers CIFAR-100 classes 0-9. Using val-set images (no augment)
    ensures the probe data is fixed and independent of training randomness.
    """
    ds      = tv_datasets.CIFAR100(root=data_root, train=False, download=False,
                                    transform=eval_transform())
    targets = torch.tensor(ds.targets)
    indices = ((targets >= 0) & (targets < 10)).nonzero(as_tuple=True)[0].tolist()
    indices = indices[:n_samples]
    return DataLoader(Subset(ds, indices), batch_size=_PROBE_BATCH_SZ,
                      shuffle=False, num_workers=num_workers)


def _process_run(
    run_name: str,
    device: torch.device,
    probe_loader: DataLoader,
    n_ckpts: int,
    probe_batches: int,
) -> None:
    """Extract CKA and drift for a single run directory.

    Args:
        run_name: Directory name under results/runs/ (e.g. "vit_vanilla_s0").
        device: Evaluation device.
        probe_loader: DataLoader for the fixed probe dataset.
        n_ckpts: Maximum number of checkpoints to load (smoke uses 2).
        probe_batches: Batches per model for CKA collection (-1 = all).
    """
    run_dir   = os.path.join(_RUNS_ROOT, run_name)
    feat_dir  = os.path.join(_FEATURES_ROOT, run_name)

    cka_out   = os.path.join(feat_dir, "cka.npz")
    drift_out = os.path.join(feat_dir, "drift.npz")
    if os.path.exists(cka_out) and os.path.exists(drift_out):
        print(f"  [skip] {run_name} (already extracted)")
        return

    arch = run_name.split("_")[0]
    if arch not in PROBE_LAYERS:
        print(f"  [FAIL] {run_name}: unknown arch '{arch}'")
        return

    layer_names = PROBE_LAYERS[arch]

    # Collect available checkpoints
    ckpt_paths  = [
        os.path.join(run_dir, CKPT_TEMPLATE.format(t=t))
        for t in range(n_ckpts)
    ]
    ckpt_paths  = [p for p in ckpt_paths if os.path.exists(p)]
    if not ckpt_paths:
        print(f"  [FAIL] {run_name}: no checkpoints found")
        return

    print(f"  {run_name}: loading {len(ckpt_paths)} ckpts, "
          f"probing layers {layer_names}")

    # Load models and capture drift snapshots
    models: list[torch.nn.Module] = []
    snapshots: list[dict] = []
    for path in ckpt_paths:
        m = build_model(arch)
        load_model(m, path)
        snapshots.append(snapshot(m))
        models.append(m)

    # Pairwise CKA across task checkpoints
    cka_dict = between_task_cka(
        models, probe_loader, layer_names, device, probe_batches
    )

    # L2 drift from task-0 snapshot to each subsequent snapshot
    ref_snap = snapshots[0]
    drift_data: dict[str, list[float]] = {}
    for snap in snapshots:
        dr = compute_drift(ref_snap, snap)
        for param_name, val in dr.items():
            key = param_name.replace(".", "_")
            drift_data.setdefault(key, []).append(val)

    # Save results
    os.makedirs(feat_dir, exist_ok=True)
    np.savez(
        cka_out,
        **{name.replace(".", "_"): mat for name, mat in cka_dict.items()},
    )
    np.savez(
        drift_out,
        **{k: np.array(v, dtype=np.float32) for k, v in drift_data.items()},
    )
    print(f"    -> saved to {feat_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CKA and drift from CL checkpoints")
    parser.add_argument(
        "--runs", nargs="*", default=None,
        help="Run names to process; default processes all runs in results/runs/",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke mode: process 2 runs, 2 ckpts, 1 probe batch",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.runs is not None: run_names = args.runs
    else:                     run_names = sorted(os.listdir(_RUNS_ROOT))

    if args.smoke:
        # Keep one vanilla_s0 run per arch for a minimal sanity check
        smoke_set     = [f"{arch}_vanilla_s0" for arch in ("vit", "resnet", "swin")]
        run_names     = [r for r in smoke_set if r in run_names]
        n_ckpts       = 2
        probe_batches = 1
    else:
        n_ckpts       = Config().n_tasks
        probe_batches = -1

    cfg          = Config()
    probe_loader = _get_probe_loader(cfg.data_root, _PROBE_SAMPLES, cfg.num_workers)

    print(f"=== extract.py | device={device} | runs={len(run_names)} ===")
    for run_name in run_names:
        _process_run(run_name, device, probe_loader, n_ckpts, probe_batches)

    print("=== done ===")


if __name__ == "__main__":
    main()
