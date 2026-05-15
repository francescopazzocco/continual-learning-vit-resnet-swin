"""M1 pilot: joint-train ViT-Small and ResNet-18 on full CIFAR-100.

Exit code 1 if ViT top-1 accuracy < 55%.

Usage:
    python scripts/pilot.py [--smoke] [--arch {vit,resnet,both}]
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
from src.models.vit import get_vit_small
from src.models.resnet import get_resnet18
from src.trainer import fit

VIT_ACCURACY_GATE = 0.55


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M1 joint-training pilot")
    p.add_argument("--smoke", action="store_true",
                   help="1 epoch, 2 batches only")
    p.add_argument("--arch", choices=["vit", "resnet", "both"], default="both")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    return p.parse_args()


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

    # Fall back to CPU if CUDA unavailable
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")
        cfg.device = "cpu"

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    print(f"=== M1 Pilot | device={cfg.device} | smoke={args.smoke} ===")
    train_loader, val_loader = get_joint_loaders(cfg)

    out_dir = os.path.join(cfg.results_root, "pilot")
    results: dict[str, float] = {}

    if args.arch in ("vit", "both"):
        model = get_vit_small(n_classes=cfg.n_classes)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  [vit] params: {n_params:.2f}M")
        val_accs = fit(model, train_loader, val_loader, cfg,
                       arch_name="vit", out_dir=out_dir, smoke=args.smoke)
        results["vit"] = max(val_accs)

    if args.arch in ("resnet", "both"):
        model = get_resnet18(n_classes=cfg.n_classes)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  [resnet] params: {n_params:.2f}M")
        val_accs = fit(model, train_loader, val_loader, cfg,
                       arch_name="resnet", out_dir=out_dir, smoke=args.smoke)
        results["resnet"] = max(val_accs)

    print("\n=== Results ===")
    for arch, acc in results.items():
        if args.smoke:
            tag = "[SMOKE]"
        elif arch != "vit" or acc >= VIT_ACCURACY_GATE:
            tag = "[OK]"
        else:
            tag = "[FAIL]"
        print(f"  {tag} {arch}: {acc * 100:.2f}%")

    if not args.smoke and "vit" in results:
        if results["vit"] < VIT_ACCURACY_GATE:
            print(
                f"[FAIL] ViT top-1 {results['vit']*100:.2f}% < "
                f"{VIT_ACCURACY_GATE*100:.0f}% gate. "
                "Switch dataset to Tiny ImageNet per PRD fallback."
            )
            sys.exit(1)
        else:
            print("[OK] ViT accuracy gate passed -> proceed to M2")


if __name__ == "__main__":
    main()
