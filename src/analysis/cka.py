"""Linear CKA for representational similarity analysis across task checkpoints.

Reference: Kornblith et al. (2019), "Similarity of Neural Network Representations
Revisited". CKA is computed in the linear-kernel form using the Frobenius-norm
shortcut that avoids forming n x n Gram matrices.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Minimum denominator to prevent division by zero in CKA
_MIN_DENOM = 1e-10


def _pool_output(t: torch.Tensor) -> torch.Tensor:
    """Reduce a forward-hook output tensor to (B, d) via average pooling.

    Dispatch rules:
    - 2-D (B, d): pass through.
    - 3-D (B, N, d): mean over token axis (handles ViT blocks/norm/stem).
    - 4-D (B, H, W, C) channels-last: mean over spatial axes 1,2 (Swin).
    - 4-D (B, C, H, W) channels-first: mean over spatial axes 2,3 (ResNet).
    - Other: flatten to (B, -1).

    Channels-last vs. channels-first is distinguished by comparing shape[-1]
    to shape[1]; for all probed layers, C > H and C > W.
    """
    if t.ndim == 2:
        return t
    if t.ndim == 3:
        return t.mean(dim=1)
    if t.ndim == 4:
        if t.shape[-1] > t.shape[1]:
            return t.mean(dim=(1, 2))   # (B, H, W, C) -> (B, C)
        return t.mean(dim=(2, 3))       # (B, C, H, W) -> (B, C)
    return t.flatten(1)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two activation matrices.

    CKA = ||Y_c^T X_c||_F^2 / (||X_c^T X_c||_F * ||Y_c^T Y_c||_F)

    where X_c, Y_c are column-centered versions of X and Y.
    The 1/n^2 HSIC factor cancels in the ratio.

    Args:
        X: Float array of shape (n_samples, d1).
        Y: Float array of shape (n_samples, d2). Must have same n_samples.

    Returns:
        Linear CKA value in [0, 1].
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    hsic_xy = float(np.linalg.norm(Y.T @ X, "fro") ** 2)
    hsic_xx = float(np.linalg.norm(X.T @ X, "fro") ** 2)
    hsic_yy = float(np.linalg.norm(Y.T @ Y, "fro") ** 2)

    denom = np.sqrt(hsic_xx) * np.sqrt(hsic_yy)
    return hsic_xy / denom if denom > _MIN_DENOM else 0.0


def collect_activations(
    model: nn.Module,
    loader: DataLoader,
    layer_names: List[str],
    device: torch.device,
    max_batches: int = -1,
) -> Dict[str, np.ndarray]:
    """Collect pooled activations from named submodules via forward hooks.

    Args:
        model: Model already on device, in eval mode.
        loader: DataLoader over the probe dataset.
        layer_names: Submodule names accessible via model.get_submodule().
        device: Evaluation device.
        max_batches: Stop after this many batches; -1 means process all.

    Returns:
        Dict mapping each layer name to a float32 ndarray of shape
        (n_samples, d) where d is the pooled feature dimension.
    """
    buckets: Dict[str, List[np.ndarray]] = {n: [] for n in layer_names}
    hooks: list = []

    def _make_hook(name: str):
        def _hook(_mod: nn.Module, _inp: tuple, out: torch.Tensor) -> None:
            buckets[name].append(
                _pool_output(out.detach().float()).cpu().numpy()
            )
        return _hook

    for name in layer_names:
        mod = model.get_submodule(name)
        hooks.append(mod.register_forward_hook(_make_hook(name)))

    model.eval()
    try:
        with torch.no_grad():
            for i, (x, _) in enumerate(loader):
                if max_batches > 0 and i >= max_batches:
                    break
                model(x.to(device))
    finally:
        for h in hooks:
            h.remove()

    return {n: np.concatenate(buckets[n], axis=0) for n in layer_names}


def between_task_cka(
    models: List[nn.Module],
    loader: DataLoader,
    layer_names: List[str],
    device: torch.device,
    max_batches: int = -1,
) -> Dict[str, np.ndarray]:
    """Compute pairwise linear CKA between representations at different task checkpoints.

    For each pair (i, j) of checkpoints, collects activations from the same
    probe data and computes CKA per layer.  Models are moved to device one at
    a time to keep GPU memory bounded.

    Args:
        models: List of T models on CPU, one per task checkpoint.
        loader: DataLoader for the fixed probe dataset (same data for all models).
        layer_names: Submodule names to probe.
        device: Evaluation device.
        max_batches: Batches per model; -1 means all.

    Returns:
        Dict mapping layer name to float32 ndarray of shape (T, T).
        result[layer][i, j] = CKA between task-i and task-j representations.
    """
    all_acts: List[Dict[str, np.ndarray]] = []
    for m in models:
        m.to(device)
        acts = collect_activations(m, loader, layer_names, device, max_batches)
        all_acts.append(acts)
        m.cpu()

    T = len(models)
    result: Dict[str, np.ndarray] = {}
    for name in layer_names:
        mat = np.zeros((T, T), dtype=np.float32)
        for i in range(T):
            for j in range(i, T):
                val = linear_cka(all_acts[i][name], all_acts[j][name])
                mat[i, j] = val
                mat[j, i] = val
        result[name] = mat
    return result
