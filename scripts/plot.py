#!/usr/bin/env python3
"""Generate all figures from CL run metrics, CKA matrices, and weight-drift arrays.

Reads:
    results/runs/{arch}_{method}_s{seed}/metrics.csv  -- scalar metrics
    results/features/{arch}_{method}_s{seed}/cka.npz  -- (T,T) CKA per layer
    results/features/{arch}_{method}_s{seed}/drift.npz -- (T,) drift per param

Writes (to results/figures/):
    metric_summary.pdf   -- AA / BWT / AF grouped bar chart
    cka_heatmaps.pdf     -- task x task CKA for the deepest probed layer
    weight_drift.pdf     -- total L2 drift from task 0 across tasks

Usage:
    python scripts/plot.py
    python scripts/plot.py --out_dir results/figures
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_RUNS_ROOT     = "results/runs"
_FEATURES_ROOT = "results/features"
_FIGURES_ROOT  = "results/figures"

ARCHS   = ["vit", "resnet", "swin"]
METHODS = ["vanilla", "ewc", "er"]
SEEDS   = [0, 1, 2]
METRICS = ["AA", "BWT", "AF"]

# Deepest probed layer per architecture (used for CKA heatmap figures)
_REPR_LAYER: dict[str, str] = {
    "vit":    "blocks_5",
    "resnet": "layer4",
    "swin":   "stages_1_5",
}

# Visual settings
_METHOD_COLORS = {"vanilla": "#4C72B0", "ewc": "#DD8452", "er": "#55A868"}
_METHOD_LABELS = {"vanilla": "Vanilla", "ewc": "EWC", "er": "ER"}
_ARCH_LABELS   = {"vit": "ViT-Small", "resnet": "ResNet-18", "swin": "Swin-Tiny"}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _run_name(arch: str, method: str, seed: int) -> str:
    return f"{arch}_{method}_s{seed}"


def _load_metrics() -> Dict[str, Dict[str, Dict[int, Dict[str, float]]]]:
    """Load scalar metrics for all runs that have a metrics.csv.

    Returns:
        Nested dict: data[arch][method][seed][metric] = value.
    """
    data: Dict = defaultdict(lambda: defaultdict(dict))
    for arch in ARCHS:
        for method in METHODS:
            for seed in SEEDS:
                path = os.path.join(
                    _RUNS_ROOT, _run_name(arch, method, seed), "metrics.csv"
                )
                if not os.path.exists(path):
                    continue
                with open(path, newline="") as f:
                    reader = csv.DictReader(f)
                    row_map: Dict[str, float] = {}
                    for row in reader:
                        row_map[row["metric"]] = float(row["value"])
                data[arch][method][seed] = row_map
    return data


def _load_cka(layer_key: Optional[str] = None) -> Dict[str, Dict[str, Dict[int, np.ndarray]]]:
    """Load CKA matrices for all available feature files.

    Args:
        layer_key: If given, load only this layer (npz key). Otherwise loads
                   the architecture-specific representative layer.

    Returns:
        Nested dict: cka[arch][method][seed] = (T, T) float32 array.
    """
    cka: Dict = defaultdict(lambda: defaultdict(dict))
    for arch in ARCHS:
        key = layer_key or _REPR_LAYER.get(arch)
        if key is None:
            continue
        for method in METHODS:
            for seed in SEEDS:
                path = os.path.join(
                    _FEATURES_ROOT, _run_name(arch, method, seed), "cka.npz"
                )
                if not os.path.exists(path):
                    continue
                data = np.load(path)
                if key not in data:
                    continue
                cka[arch][method][seed] = data[key].astype(np.float32)
    return cka


def _load_drift() -> Dict[str, Dict[str, Dict[int, np.ndarray]]]:
    """Load total L2 drift (sum over all parameters) for each run.

    Returns:
        Nested dict: drift[arch][method][seed] = (T,) float32 array.
    """
    drift: Dict = defaultdict(lambda: defaultdict(dict))
    for arch in ARCHS:
        for method in METHODS:
            for seed in SEEDS:
                path = os.path.join(
                    _FEATURES_ROOT, _run_name(arch, method, seed), "drift.npz"
                )
                if not os.path.exists(path):
                    continue
                npz = np.load(path)
                # Stack all parameter drift arrays and sum to get total drift per task
                arrays = [npz[k] for k in npz.files]
                if not arrays:
                    continue
                total = np.sum(np.stack(arrays, axis=0), axis=0)  # (T,)
                drift[arch][method][seed] = total.astype(np.float32)
    return drift


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _agg_metric(
    metric_data: Dict,
    arch: str,
    method: str,
    key: str,
) -> Tuple[float, float]:
    """Return (mean, std) of a scalar metric over available seeds."""
    vals = [
        metric_data[arch][method][s][key]
        for s in SEEDS
        if arch in metric_data
        and method in metric_data[arch]
        and s in metric_data[arch][method]
        and key in metric_data[arch][method][s]
    ]
    if not vals:
        return float("nan"), 0.0
    return float(np.mean(vals)), float(np.std(vals))


def _agg_matrix(
    cka_data: Dict,
    arch: str,
    method: str,
) -> Optional[np.ndarray]:
    """Return mean CKA matrix over available seeds, or None if missing."""
    mats = [
        cka_data[arch][method][s]
        for s in SEEDS
        if arch in cka_data
        and method in cka_data[arch]
        and s in cka_data[arch][method]
    ]
    return np.mean(np.stack(mats, axis=0), axis=0) if mats else None


def _agg_drift(
    drift_data: Dict,
    arch: str,
    method: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (mean, std) drift arrays over available seeds, or (None, None)."""
    arrs = [
        drift_data[arch][method][s]
        for s in SEEDS
        if arch in drift_data
        and method in drift_data[arch]
        and s in drift_data[arch][method]
    ]
    if not arrs:
        return None, None
    stacked = np.stack(arrs, axis=0)  # (n_seeds, T)
    return stacked.mean(axis=0), stacked.std(axis=0)


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------

def _plot_metric_summary(metric_data: Dict, out_dir: str) -> None:
    """Grouped bar chart of AA, BWT, AF per arch x method.

    Layout: 1 row x 3 cols (one per metric), each showing 3 arch groups
    of 3 bars (one per method).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("CL Method Comparison: AA / BWT / AF (mean +/- std, 3 seeds)")

    x       = np.arange(len(ARCHS))
    n_meth  = len(METHODS)
    bar_w   = 0.22
    offsets = np.linspace(-(n_meth - 1) / 2, (n_meth - 1) / 2, n_meth) * bar_w

    for ax, metric in zip(axes, METRICS):
        for off, method in zip(offsets, METHODS):
            means, stds = zip(*[_agg_metric(metric_data, a, method, metric)
                                 for a in ARCHS])
            ax.bar(
                x + off, means, bar_w,
                yerr=stds, capsize=3,
                color=_METHOD_COLORS[method],
                label=_METHOD_LABELS[method],
                error_kw={"elinewidth": 1},
            )

        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels([_ARCH_LABELS[a] for a in ARCHS])
        ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax.set_ylabel(metric)
        if metric == METRICS[0]:
            ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "metric_summary.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"  -> {path}")
    plt.close(fig)


def _plot_cka_heatmaps(cka_data: Dict, out_dir: str) -> None:
    """Task x task CKA heatmaps for the representative deep layer.

    Layout: 3 rows (arch) x 3 cols (method). Each cell is the mean CKA
    matrix across seeds for the architecture's deepest probed layer.
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle("Pairwise task CKA (mean over seeds) -- deepest probed layer")

    for row, arch in enumerate(ARCHS):
        for col, method in enumerate(METHODS):
            ax  = axes[row][col]
            mat = _agg_matrix(cka_data, arch, method)

            if mat is None:
                ax.set_visible(False)
                continue

            T   = mat.shape[0]
            im  = ax.imshow(mat, vmin=0, vmax=1, cmap="plasma", aspect="auto")
            ax.set_title(f"{_ARCH_LABELS[arch]} / {_METHOD_LABELS[method]}")
            ax.set_xlabel("task j")
            ax.set_ylabel("task i")
            ax.set_xticks(range(T))
            ax.set_yticks(range(T))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = os.path.join(out_dir, "cka_heatmaps.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"  -> {path}")
    plt.close(fig)


def _plot_weight_drift(drift_data: Dict, out_dir: str) -> None:
    """Total L2 weight drift from task-0 checkpoint across tasks.

    Layout: 1 row x 3 cols (one per arch). Each subplot has 3 lines (one
    per method), with +/- std shading over seeds.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Total L2 weight drift from task-0 checkpoint (mean +/- std, 3 seeds)")

    for ax, arch in zip(axes, ARCHS):
        ax.set_title(_ARCH_LABELS[arch])
        ax.set_xlabel("task index")
        ax.set_ylabel("total $\\|\\Delta W\\|_2$")

        for method in METHODS:
            mean, std = _agg_drift(drift_data, arch, method)
            if mean is None:
                continue
            T = len(mean)
            x = np.arange(T)
            ax.plot(x, mean, color=_METHOD_COLORS[method],
                    label=_METHOD_LABELS[method], marker="o", markersize=4)
            ax.fill_between(x, mean - std, mean + std,
                             alpha=0.2, color=_METHOD_COLORS[method])

        ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "weight_drift.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"  -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate M3 figures")
    parser.add_argument("--out_dir", default=_FIGURES_ROOT)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=== plot.py ===")
    print("Loading metrics ...")
    metric_data = _load_metrics()

    print("Loading CKA ...")
    cka_data    = _load_cka()

    print("Loading drift ...")
    drift_data  = _load_drift()

    _plot_metric_summary(metric_data, args.out_dir)
    _plot_cka_heatmaps(cka_data, args.out_dir)
    _plot_weight_drift(drift_data, args.out_dir)

    print("=== done ===")


if __name__ == "__main__":
    main()
