"""Experiment grid: the architectures, methods, and seeds of the full CL study.

Re-exports ARCHS and METHODS from their registries so the grid is described in
exactly one place and the run/plot scripts iterate the same set.
"""

from __future__ import annotations

from typing import List

from src.cl import METHODS
from src.models import ARCHS

# Seeds for the full grid (3 arch x 3 methods x 3 seeds = 27 runs).
SEEDS: List[int] = [0, 1, 2]

__all__ = ["ARCHS", "METHODS", "SEEDS"]
