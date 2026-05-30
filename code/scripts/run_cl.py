"""M2: full CL grid -- 3 architectures x 3 methods x 3 seeds.

Usage:
    python scripts/run_cl.py [--smoke] [--arch {vit,resnet,swin,all}]
               [--method {vanilla,ewc,er,all}] [--seed N]
               [--device DEVICE] [--num_workers N]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.default import Config
from src.artifacts import METRICS_FILE
from src.cl import build_method
from src.cl_trainer import run_cl
from src.data.cifar100 import get_split_loaders
from src.grid import ARCHS, METHODS, SEEDS
from src.metrics import compute_metrics
from src.models import build_model
from src.runtime import set_seed, setup_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M2 CL training grid")
    p.add_argument("--smoke", action="store_true",
                   help="2 tasks / 1 epoch / 2 batches; no disk writes")
    p.add_argument("--arch", choices=ARCHS + ["all"], default="all")
    p.add_argument("--method", choices=METHODS + ["all"], default="all")
    p.add_argument("--seed", type=int, default=None,
                   help="Run a single seed instead of all three")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    if args.device is not None:
        cfg.device = args.device
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    device = setup_device(cfg)

    archs   = ARCHS if args.arch == "all" else [args.arch]
    methods = METHODS if args.method == "all" else [args.method]
    seeds   = SEEDS if args.seed is None else [args.seed]

    runs_root = os.path.join(cfg.results_root, "runs")
    total     = len(archs) * len(methods) * len(seeds)
    done      = skipped = 0

    print(f"=== M2 CL grid | smoke={args.smoke} | device={cfg.device} | {total} runs ===")

    # Cache split loaders by seed: splits are dataset-only (method-independent).
    splits_cache: dict[int, list] = {}

    for arch in archs:
        for method_name in methods:
            for seed in seeds:
                run_name     = f"{arch}_{method_name}_s{seed}"
                run_dir      = os.path.join(runs_root, run_name)
                metrics_path = os.path.join(run_dir, METRICS_FILE)

                if not args.smoke and os.path.exists(metrics_path):
                    print(f"  [SKIP] {run_name}")
                    skipped += 1
                    continue

                print(f"  -> {run_name}")
                set_seed(seed)
                cfg.arch   = arch
                cfg.method = method_name
                cfg.seed   = seed

                if seed not in splits_cache:
                    splits_cache[seed] = get_split_loaders(cfg)
                splits = splits_cache[seed]

                model  = build_model(arch, cfg.n_classes)
                method = build_method(method_name, cfg, device)

                R = run_cl(model, splits, method, cfg, run_dir, smoke=args.smoke)

                if not args.smoke:
                    m = compute_metrics(R)
                    summary = (
                        f" AA={m['AA']:.3f} BWT={m['BWT']:.3f} AF={m['AF']:.3f}"
                    )
                else:
                    summary = ""

                tag = "[SMOKE]" if args.smoke else "[OK]"
                print(f"  {tag} {run_name}{summary}")
                done += 1

    print(f"\n=== done={done} skipped={skipped} total={total} ===")


if __name__ == "__main__":
    main()
