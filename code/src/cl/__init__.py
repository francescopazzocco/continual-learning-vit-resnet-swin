"""CL method registry: maps method names to configured CLMethod instances."""

from __future__ import annotations

from typing import List

import torch

from configs.default import Config
from src.cl.base import CLMethod
from src.cl.er import ER
from src.cl.ewc import EWC
from src.cl.vanilla import Vanilla

METHODS: List[str] = ["vanilla", "ewc", "er"]


def build_method(name: str, cfg: Config, device: torch.device) -> CLMethod:
    """Instantiate a CL method from its name, drawing hyperparameters from cfg.

    Args:
        name: One of METHODS.
        cfg: Config supplying method hyperparameters.
        device: Training device (passed to methods that hold state on-device).

    Returns:
        Configured CLMethod instance.

    Raises:
        ValueError: If name is not a known method.
    """
    if name == "vanilla":
        return Vanilla()
    if name == "ewc":
        return EWC(
            ewc_lambda=cfg.ewc_lambda,
            fisher_subsample=cfg.fisher_subsample,
            fisher_batch_size=cfg.fisher_batch_size,
            device=device,
            num_workers=cfg.num_workers,
        )
    if name == "er":
        return ER(buffer_size=cfg.er_buffer_size, device=device)
    raise ValueError(f"Unknown method: {name!r} (known: {METHODS})")
