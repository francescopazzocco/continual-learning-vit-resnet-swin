#!/usr/bin/env python3
"""Generate all figures from CL run metrics, CKA matrices, and weight-drift arrays.

Reads:
    results/runs/{arch}_{method}_s{seed}/metrics.csv  -- scalar metrics
    results/features/{arch}_{method}_s{seed}/cka.npz  -- (T,T) CKA per layer
    results/features/{arch}_{method}_s{seed}/drift.npz -- (T,) drift per param

Writes (to results/figures/):
    metric_summary.pdf    -- AA / BWT / AF grouped bar chart + joint upper bound
    cka_heatmaps.pdf      -- task x task CKA for the deepest probed layer
    weight_drift.pdf      -- total L2 drift from task-0 checkpoint
    forgetting_curves.pdf -- accuracy on task 0 as tasks 1-9 are trained
    task_diagonals.pdf    -- per-task accuracy R[i,i] vs forgetting trade-off

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

# Joint-training upper bound from pilot (best val accuracy, epoch 200).
# Used as a reference line in the AA subplot and forgetting-curve plots.
JOINT_ACC: dict[str, float] = {
    "vit":    0.6228,
    "resnet": 0.6431,
    "swin":   0.5734,
}

# Deepest probed layer per architecture (used for CKA heatmap figures)
_REPR_LAYER: dict[str, str] = {
    "vit":    "blocks_5",
    "resnet": "layer4",
    "swin":   "stages_1_5",
}

_METHOD_COLORS = {"vanilla": "#4C72B0", "ewc": "#DD8452", "er": "#55A868"}
_METHOD_LABELS = {"vanilla": "Vanilla", "ewc": "EWC", "er": "ER"}
_ARCH_LABELS   = {"vit": "ViT-Small", "resnet": "ResNet-18", "swin": "Swin-Tiny"}
_JOINT_COLOR   = "#888888"

# Layer groupings for per-layer drift figure: (label, list-of-key-prefixes).
# A parameter key belongs to a group if it starts with any of the group's prefixes.
_LAYER_GROUPS: dict[str, list[tuple[str, list[str]]]] = {
    "vit": [
        ("stem",       ["stem_"]),
        ("blocks 0-1", ["blocks_0_", "blocks_1_"]),
        ("blocks 2-3", ["blocks_2_", "blocks_3_"]),
        ("blocks 4-5", ["blocks_4_", "blocks_5_"]),
        ("head",       ["norm_", "head_", "cls_token", "pos_embed"]),
    ],
    "resnet": [
        ("stem",   ["conv1_", "bn1_"]),
        ("layer1", ["layer1_"]),
        ("layer2", ["layer2_"]),
        ("layer3", ["layer3_"]),
        ("layer4", ["layer4_"]),
        ("head",   ["fc"]),
    ],
    "swin": [
        ("stem",       ["patch_embed_"]),
        ("stage0",     ["stages_0_"]),
        ("merge",      ["patch_merging_"]),
        ("stage1 0-2", ["stages_1_0_", "stages_1_1_", "stages_1_2_"]),
        ("stage1 3-5", ["stages_1_3_", "stages_1_4_", "stages_1_5_"]),
        ("head",       ["norm_", "head_"]),
    ],
}

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


def _load_metrics() -> Dict:
    """Load all per-run metric rows, including the full R matrix.

    Returns:
        Nested dict: data[arch][method][seed][metric_key] = float value.
        Keys include 'AA', 'BWT', 'AF', and 'R_{i}_{j}' for all (i,j).
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
                    row_map: Dict[str, float] = {}
                    for row in csv.DictReader(f):
                        row_map[row["metric"]] = float(row["value"])
                data[arch][method][seed] = row_map
    return data


def _load_cka(layer_key: Optional[str] = None) -> Dict:
    """Load CKA matrices.

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
                npz = np.load(path)
                if key not in npz:
                    continue
                cka[arch][method][seed] = npz[key].astype(np.float32)
    return cka


def _load_drift() -> Dict:
    """Load total L2 drift (nansum over parameters) per task for each run.

    nansum is used so that NaN entries from training instabilities (e.g.
    ViT vanilla collapse producing inf weights) do not propagate.

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
                npz    = np.load(path)
                arrays = [npz[k] for k in npz.files]
                if not arrays:
                    continue
                # nansum: NaN parameter drifts (collapsed ViT) are treated as 0
                total  = np.nansum(np.stack(arrays, axis=0), axis=0)
                drift[arch][method][seed] = total.astype(np.float32)
    return drift


def _in_group(key: str, prefixes: List[str]) -> bool:
    return any(key.startswith(p) for p in prefixes)


def _load_layer_drift() -> Dict:
    """Load per-layer-group L2 drift summed at the final task.

    Returns:
        Nested dict: layer_drift[arch][method][seed][group_label] = float.
        NaN parameter drifts (collapsed runs) are excluded from the sum.
    """
    result: Dict = defaultdict(lambda: defaultdict(dict))
    for arch in ARCHS:
        groups = _LAYER_GROUPS.get(arch, [])
        for method in METHODS:
            for seed in SEEDS:
                path = os.path.join(
                    _FEATURES_ROOT, _run_name(arch, method, seed), "drift.npz"
                )
                if not os.path.exists(path):
                    continue
                npz = np.load(path)
                group_drift: dict[str, float] = {}
                for label, prefixes in groups:
                    vals = [
                        float(npz[k][-1])
                        for k in npz.files
                        if _in_group(k, prefixes) and not np.isnan(npz[k][-1])
                    ]
                    group_drift[label] = float(np.sum(vals)) if vals else 0.0
                result[arch][method][seed] = group_drift
    return result


def _extract_r_column(metric_data: Dict, arch: str, method: str,
                      col: int, n_tasks: int = 10) -> np.ndarray:
    """Extract R[0..n_tasks-1, col] averaged over seeds.

    R[i, col] is defined only for i >= col; positions i < col are NaN.

    Returns:
        float32 array of shape (n_tasks,) with NaN where undefined.
    """
    curves = []
    for seed in SEEDS:
        if (arch not in metric_data or method not in metric_data[arch]
                or seed not in metric_data[arch][method]):
            continue
        row_map = metric_data[arch][method][seed]
        col_vals = [row_map.get(f"R_{i}_{col}", float("nan"))
                    for i in range(n_tasks)]
        curves.append(col_vals)
    if not curves:
        return np.full(n_tasks, float("nan"), dtype=np.float32)
    return np.nanmean(curves, axis=0).astype(np.float32)


def _extract_diagonal(metric_data: Dict, arch: str, method: str,
                      n_tasks: int = 10) -> np.ndarray:
    """Extract mean R[i, i] over seeds.

    Returns:
        float32 array of shape (n_tasks,).
    """
    diags = []
    for seed in SEEDS:
        if (arch not in metric_data or method not in metric_data[arch]
                or seed not in metric_data[arch][method]):
            continue
        row_map = metric_data[arch][method][seed]
        diags.append([row_map.get(f"R_{i}_{i}", float("nan"))
                      for i in range(n_tasks)])
    if not diags:
        return np.full(n_tasks, float("nan"), dtype=np.float32)
    return np.nanmean(diags, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _agg_metric(metric_data: Dict, arch: str, method: str,
                key: str) -> Tuple[float, float]:
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


def _agg_matrix(cka_data: Dict, arch: str, method: str) -> Optional[np.ndarray]:
    """Return mean CKA matrix over seeds; drops partial (smoke) matrices."""
    mats = [
        cka_data[arch][method][s]
        for s in SEEDS
        if arch in cka_data
        and method in cka_data[arch]
        and s in cka_data[arch][method]
    ]
    if not mats:
        return None
    target = max(set(m.shape for m in mats), key=lambda s: s[0])
    mats   = [m for m in mats if m.shape == target]
    return np.mean(np.stack(mats, axis=0), axis=0) if mats else None


def _agg_drift(drift_data: Dict, arch: str,
               method: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (mean, std) drift arrays over seeds, or (None, None)."""
    arrs = [
        drift_data[arch][method][s]
        for s in SEEDS
        if arch in drift_data
        and method in drift_data[arch]
        and s in drift_data[arch][method]
    ]
    if not arrs:
        return None, None
    stacked = np.stack(arrs, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)


# ---------------------------------------------------------------------------
# Figure 1: metric summary
# ---------------------------------------------------------------------------

def _plot_metric_summary(metric_data: Dict, out_dir: str) -> None:
    """AA / BWT / AF grouped bar chart with joint-training reference on AA.

    The joint upper bound is shown as a short horizontal bar above each arch
    group in the AA subplot, annotated with its value.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("CL Method Comparison: AA / BWT / AF (mean +/- std, 3 seeds)")

    x       = np.arange(len(ARCHS))
    bar_w   = 0.22
    n_meth  = len(METHODS)
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

        if metric == "AA":
            # Draw joint upper-bound marker for each architecture
            for i, arch in enumerate(ARCHS):
                jnt = JOINT_ACC[arch]
                ax.plot([i - 0.38, i + 0.38], [jnt, jnt],
                        color=_JOINT_COLOR, linewidth=2, linestyle="--",
                        zorder=5)
                ax.annotate(
                    f"Joint {jnt:.0%}",
                    xy=(i, jnt), xytext=(0, 5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8,
                    color=_JOINT_COLOR,
                )
            # Add joint to legend
            from matplotlib.lines import Line2D
            handles, labels = ax.get_legend_handles_labels()
            handles.append(Line2D([0], [0], color=_JOINT_COLOR,
                                  linewidth=2, linestyle="--"))
            labels.append("Joint")
            ax.legend(handles, labels)
        elif metric == METRICS[0]:
            ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "metric_summary.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"  -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: CKA heatmaps
# ---------------------------------------------------------------------------

def _plot_cka_heatmaps(cka_data: Dict, out_dir: str) -> None:
    """Task x task CKA heatmaps for the representative deep layer.

    Layout: 3 rows (arch) x 3 cols (method).
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


# ---------------------------------------------------------------------------
# Figure 3: weight drift
# ---------------------------------------------------------------------------

def _plot_weight_drift(drift_data: Dict, out_dir: str) -> None:
    """Total L2 weight drift from task-0 checkpoint across tasks."""
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
            # Mask NaN positions (ViT vanilla collapse)
            valid = ~np.isnan(mean)
            ax.plot(x[valid], mean[valid], color=_METHOD_COLORS[method],
                    label=_METHOD_LABELS[method], marker="o", markersize=4)
            ax.fill_between(x[valid],
                             (mean - std)[valid], (mean + std)[valid],
                             alpha=0.2, color=_METHOD_COLORS[method])

        ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "weight_drift.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"  -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: forgetting curves
# ---------------------------------------------------------------------------

def _plot_forgetting_curves(metric_data: Dict, out_dir: str) -> None:
    """Accuracy on task 0 as tasks 1-9 are sequentially trained.

    R[i, 0] for i in 0..9: shows the moment forgetting happens and whether
    it is a cliff (step function) or a gradual decay.
    A horizontal dashed line marks the joint-training upper bound.

    Layout: 1 row x 3 cols (one per arch).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Forgetting curve: accuracy on task 0 after each subsequent task\n"
        "(dashed line = joint training upper bound)"
    )

    for ax, arch in zip(axes, ARCHS):
        ax.set_title(_ARCH_LABELS[arch])
        ax.set_xlabel("tasks trained so far")
        ax.set_ylabel("accuracy on task 0")
        ax.set_ylim(-0.02, 1.02)

        # Joint upper bound reference
        ax.axhline(JOINT_ACC[arch], color=_JOINT_COLOR, linewidth=1.5,
                   linestyle="--", label=f"Joint ({JOINT_ACC[arch]:.0%})")

        for method in METHODS:
            curve = _extract_r_column(metric_data, arch, method, col=0)
            if np.all(np.isnan(curve)):
                continue
            x = np.arange(len(curve))
            ax.plot(x, curve, color=_METHOD_COLORS[method],
                    label=_METHOD_LABELS[method], marker="o", markersize=4)

        ax.set_xticks(range(10))
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, "forgetting_curves.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"  -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: per-task diagonal
# ---------------------------------------------------------------------------

def _plot_task_diagonals(metric_data: Dict, out_dir: str) -> None:
    """Per-task accuracy R[i, i] across methods: plasticity vs. stability.

    R[i, i] is the accuracy on task i's val set immediately after training
    task i. A high diagonal means the model can still learn new tasks; a
    falling or collapsing diagonal means the method is sacrificing plasticity.

    Layout: 1 row x 3 cols (one per arch).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Per-task accuracy R[i,i]: accuracy on task i right after training it\n"
        "(measures plasticity; high = model can still learn new tasks)"
    )

    for ax, arch in zip(axes, ARCHS):
        ax.set_title(_ARCH_LABELS[arch])
        ax.set_xlabel("task index i")
        ax.set_ylabel("accuracy on task i")
        ax.set_ylim(-0.02, 1.02)

        for method in METHODS:
            diag = _extract_diagonal(metric_data, arch, method)
            if np.all(np.isnan(diag)):
                continue
            x = np.arange(len(diag))
            ax.plot(x, diag, color=_METHOD_COLORS[method],
                    label=_METHOD_LABELS[method], marker="o", markersize=4)

        ax.set_xticks(range(10))
        ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "task_diagonals.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"  -> {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: per-layer drift
# ---------------------------------------------------------------------------

def _plot_layer_drift(out_dir: str) -> None:
    """Per-layer-group L2 drift at the final task, comparing CL methods.

    Drift is summed over all parameters in each layer group at task index 9
    and averaged over seeds.  Shows which layers are anchored most strongly
    by each method (mechanistic depth for Section IV-E).

    Layout: 1 row x 3 cols (one per arch).
    """
    layer_drift_data = _load_layer_drift()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Per-layer group L2 drift at final task (summed within group, mean +/- std, 3 seeds)"
    )

    bar_w   = 0.22
    n_meth  = len(METHODS)
    offsets = np.linspace(-(n_meth - 1) / 2, (n_meth - 1) / 2, n_meth) * bar_w

    for ax, arch in zip(axes, ARCHS):
        groups       = _LAYER_GROUPS.get(arch, [])
        group_labels = [g[0] for g in groups]
        n_groups     = len(group_labels)
        x            = np.arange(n_groups)

        ax.set_title(_ARCH_LABELS[arch])
        ax.set_ylabel("$\\|\\Delta W\\|_2$")
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, rotation=30, ha="right", fontsize=8)

        for off, method in zip(offsets, METHODS):
            seed_vals: List[List[float]] = []
            for seed in SEEDS:
                if (arch not in layer_drift_data
                        or method not in layer_drift_data[arch]
                        or seed not in layer_drift_data[arch][method]):
                    continue
                gd = layer_drift_data[arch][method][seed]
                seed_vals.append([gd.get(lbl, 0.0) for lbl in group_labels])
            if not seed_vals:
                continue
            arr   = np.array(seed_vals, dtype=np.float32)
            means = arr.mean(axis=0)
            stds  = arr.std(axis=0)
            ax.bar(
                x + off, means, bar_w,
                yerr=stds, capsize=3,
                color=_METHOD_COLORS[method],
                label=_METHOD_LABELS[method],
                error_kw={"elinewidth": 1},
            )

        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, "layer_drift.pdf")
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
    metric_data = _load_metrics()
    cka_data    = _load_cka()
    drift_data  = _load_drift()

    _plot_metric_summary(metric_data, args.out_dir)
    _plot_cka_heatmaps(cka_data, args.out_dir)
    _plot_weight_drift(drift_data, args.out_dir)
    _plot_forgetting_curves(metric_data, args.out_dir)
    _plot_task_diagonals(metric_data, args.out_dir)
    _plot_layer_drift(args.out_dir)

    print("=== done ===")


if __name__ == "__main__":
    main()
