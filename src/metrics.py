"""CL evaluation metrics derived from the per-task accuracy matrix."""

from __future__ import annotations

import numpy as np


def compute_metrics(R: np.ndarray) -> dict[str, float]:
    """Compute AA, BWT, and AF from the per-task accuracy matrix.

    R[i, j] is the accuracy on task j evaluated immediately after training
    task i. Only the lower-triangular portion (j <= i) is meaningful.

    Definitions (T = n_tasks, indices 0-based):
        AA  = (1/T)     * sum_{j=0}^{T-1}   R[T-1, j]
        BWT = (1/(T-1)) * sum_{j=0}^{T-2}  (R[T-1, j] - R[j, j])
        AF  = (1/(T-1)) * sum_{j=0}^{T-2}  (max_{l=j..T-1} R[l, j] - R[T-1, j])

    Args:
        R: Float array of shape (T, T).

    Returns:
        Dict with keys "AA", "BWT", "AF".
    """
    T = R.shape[0]
    if T == 0:
        return {"AA": 0.0, "BWT": 0.0, "AF": 0.0}

    aa = float(np.mean(R[T - 1, :T]))

    if T == 1:
        return {"AA": aa, "BWT": 0.0, "AF": 0.0}

    diag = np.array([R[j, j] for j in range(T - 1)])
    bwt = float(np.mean(R[T - 1, : T - 1] - diag))

    forgetting = np.array([
        float(np.max(R[j:T, j])) - R[T - 1, j] for j in range(T - 1)
    ])
    af = float(np.mean(forgetting))

    return {"AA": aa, "BWT": bwt, "AF": af}
