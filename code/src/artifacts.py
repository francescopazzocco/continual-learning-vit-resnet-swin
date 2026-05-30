"""Run-artifact filenames and readers shared by the CL trainer and eval scripts.

Centralizes the names of the files written into each run directory so the
producer (cl_trainer) and the consumers (eval/plot scripts) cannot drift.
"""

from __future__ import annotations

import csv
import os

METRICS_FILE   = "metrics.csv"
TRAIN_LOG_FILE = "train_log.csv"
CKPT_TEMPLATE  = "ckpt_task{t}.pt"


def read_scalar_metric(run_dir: str, name: str) -> float:
    """Return a scalar metric from a run's metrics.csv, or NaN if unavailable.

    Args:
        run_dir: Directory containing METRICS_FILE.
        name: Metric key in the 'metric' column (e.g. "AA").

    Returns:
        Float value, or float("nan") if the file or metric is missing.
    """
    path = os.path.join(run_dir, METRICS_FILE)
    if not os.path.exists(path):
        return float("nan")
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["metric"] == name:
                return float(row["value"])
    return float("nan")
