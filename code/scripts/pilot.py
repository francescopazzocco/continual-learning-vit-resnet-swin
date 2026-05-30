"""M1/M1b pilot: joint-train ViT-Small, ResNet-18, and Swin-Tiny on full CIFAR-100.

Exit code 1 if any gated architecture (vit, swin) top-1 accuracy < 55%.

Usage:
    python scripts/pilot.py [--smoke] [--arch {vit,resnet,swin,both,all}]
               [--epochs N] [--lr LR] [--batch_size B]
               [--device DEVICE] [--num_workers N]
"""

from __future__ import annotations

import argparse
import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from configs.default import Config
from src.data.cifar100 import get_joint_loaders
from src.models import build_model
from src.runtime import set_seed, setup_device
from src.trainer import fit

VIT_ACCURACY_GATE = 0.55
# Architectures that must reach VIT_ACCURACY_GATE or the run exits non-zero.
_GATED_ARCHS      = {"vit", "swin"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M1 joint-training pilot")
    p.add_argument("--smoke", action="store_true",
                   help="1 epoch, 2 batches only")
    p.add_argument("--arch", choices=["vit", "resnet", "swin", "both", "all"],
                   default="all",
                   help='"both" = vit+resnet (M1 compat); "all" = vit+resnet+swin')
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    return p.parse_args()


def _selected_archs(arg: str) -> list[str]:
    """Map the --arch flag to the ordered list of architectures to run."""
    archs: list[str] = []
    if arg in ("vit", "both", "all"):
        archs.append("vit")
    if arg in ("resnet", "both", "all"):
        archs.append("resnet")
    if arg in ("swin", "all"):
        archs.append("swin")
    return archs


def main() -> None:
    args = parse_args()
    cfg = Config()

    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.lr = args.lr
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.device is not None:
        cfg.device = args.device
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    device = setup_device(cfg)
    set_seed(cfg.seed)

    print(f"=== M1/M1b Pilot | device={cfg.device} | smoke={args.smoke} ===")
    train_loader, val_loader = get_joint_loaders(cfg)

    out_dir = os.path.join(cfg.results_root, "pilot")
    results: dict[str, float] = {}

    for arch_name in _selected_archs(args.arch):
        model    = build_model(arch_name, cfg.n_classes)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  [{arch_name}] params: {n_params:.2f}M")
        val_accs = fit(model, train_loader, val_loader, cfg,
                       arch_name=arch_name, out_dir=out_dir, smoke=args.smoke)
        results[arch_name] = max(val_accs)
        # Free memory before the next architecture.
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n=== Results ===")
    for arch, acc in results.items():
        if args.smoke:
            tag = "[SMOKE]"
        elif arch not in _GATED_ARCHS or acc >= VIT_ACCURACY_GATE:
            tag = "[OK]"
        else:
            tag = "[FAIL]"
        print(f"  {tag} {arch}: {acc * 100:.2f}%")

    if not args.smoke:
        failed = [
            a for a in _GATED_ARCHS
            if a in results and results[a] < VIT_ACCURACY_GATE
        ]
        for a in failed:
            print(
                f"[FAIL] {a} top-1 {results[a]*100:.2f}% < "
                f"{VIT_ACCURACY_GATE*100:.0f}% gate. "
                "Switch dataset to Tiny ImageNet per PRD fallback."
            )
        if failed:
            sys.exit(1)
        gated_ran = [a for a in _GATED_ARCHS if a in results]
        if gated_ran:
            print("[OK] All accuracy gates passed -> proceed to M2")


if __name__ == "__main__":
    main()
