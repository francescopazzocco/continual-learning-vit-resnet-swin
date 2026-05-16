"""Online EWC with diagonal Fisher approximation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn

from src.cl.base import CLMethod

_FISHER_SUBSAMPLE_MIN = 1


class EWC(CLMethod):
    """Online EWC: accumulates diagonal Fisher across tasks.

    Penalty: (ewc_lambda / 2) * sum_n F_n * (theta_n - theta*_n)^2

    Fisher is accumulated by summing (not averaging) across tasks so that
    earlier tasks receive proportionally more regularization weight as the
    model drifts further from their optima.

    Args:
        ewc_lambda: Regularization strength (default 1000 from config).
        fisher_subsample: Fraction of training set used to estimate Fisher.
        device: Training device.
    """

    def __init__(
        self, ewc_lambda: float, fisher_subsample: float, device: torch.device
    ) -> None:
        self.ewc_lambda = ewc_lambda
        self.fisher_subsample = fisher_subsample
        self.device = device
        # Accumulated diagonal Fisher; populated after the first after_task call.
        self._fisher: dict[str, torch.Tensor] = {}
        # Consolidated parameter snapshot at the most recent after_task call.
        self._params: dict[str, torch.Tensor] = {}

    def before_task(
        self, task_id: int, train_loader: DataLoader, model: nn.Module
    ) -> None:
        pass

    def loss(
        self, logits: torch.Tensor, targets: torch.Tensor, model: nn.Module
    ) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets)
        if not self._fisher:
            return ce
        terms = [
            (self._fisher[n] * (p - self._params[n]).pow(2)).sum()
            for n, p in model.named_parameters()
            if n in self._fisher
        ]
        penalty = torch.stack(terms).sum() if terms else ce.new_zeros(())
        return ce + (self.ewc_lambda / 2.0) * penalty

    def after_task(
        self, task_id: int, train_loader: DataLoader, model: nn.Module
    ) -> None:
        """Estimate diagonal Fisher on a random subsample and accumulate it."""
        model.eval()
        n_total = len(train_loader.dataset)
        n_target = max(_FISHER_SUBSAMPLE_MIN, int(n_total * self.fisher_subsample))

        task_fisher: dict[str, torch.Tensor] = {
            name: torch.zeros_like(p, device=self.device)
            for name, p in model.named_parameters()
            if p.requires_grad
        }

        n_seen = 0
        for x, y in train_loader:
            if n_seen >= n_target:
                break
            x, y = x.to(self.device), y.to(self.device)
            batch = x.size(0)
            model.zero_grad()
            F.cross_entropy(model(x), y).backward()
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    task_fisher[name] += param.grad.pow(2) * batch
            n_seen += batch

        for name in task_fisher:
            task_fisher[name] /= max(n_seen, 1)

        # Online accumulation: sum Fisher contributions across tasks.
        for name, f in task_fisher.items():
            if name in self._fisher:
                self._fisher[name] += f
            else:
                self._fisher[name] = f

        for name, param in model.named_parameters():
            if param.requires_grad:
                self._params[name] = param.detach().clone()
