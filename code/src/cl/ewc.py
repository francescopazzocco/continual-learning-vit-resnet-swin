"""Online EWC with diagonal Fisher approximation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.cl.base import CLMethod

_FISHER_SUBSAMPLE_MIN = 1

# Divisor for EWC penalty: (lambda / 2) * sum F * (theta - theta*)^2
_EWC_PENALTY_DIVISOR  = 2.0

# Minimum number of samples for Fisher normalization (avoids div by zero)
_FISHER_MIN_SEEN      = 1


class EWC(CLMethod):
    """Online EWC: accumulates diagonal Fisher across tasks.

    Penalty: (ewc_lambda / 2) * sum_n F_n * (theta_n - theta*_n)^2

    Fisher is accumulated by summing (not averaging) across tasks so that
    earlier tasks receive proportionally more regularization weight as the
    model drifts further from their optima.

    Args:
        ewc_lambda: Regularization strength (default 1000 from config).
        fisher_subsample: Fraction of training set used to estimate Fisher.
        fisher_batch_size: Batch size for Fisher estimation (1=per-sample, default 1).
        device: Training device.
    """

    def __init__(
        self, ewc_lambda: float, fisher_subsample: float,
        fisher_batch_size: int, device: torch.device,
    ) -> None:
        self.ewc_lambda         = ewc_lambda
        self.fisher_subsample   = fisher_subsample
        self.fisher_batch_size  = fisher_batch_size
        self.device             = device
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
        return ce + (self.ewc_lambda / _EWC_PENALTY_DIVISOR) * penalty

    def after_task(
        self, task_id: int, train_loader: DataLoader, model: nn.Module
    ) -> None:
        """Estimate diagonal Fisher on a random subsample and accumulate it.

        Fisher batch_size controls the granularity of the estimate:
        - batch_size=1: true empirical Fisher mean(g_i^2)
        - batch_size>1: approximates B * (mean(g))^2 which differs from mean(g^2)
        Use fisher_batch_size=1 for the most faithful estimate (at higher memory cost).
        """
        model.eval()
        n_total  = len(train_loader.dataset)
        n_target = max(_FISHER_SUBSAMPLE_MIN, int(n_total * self.fisher_subsample))

        task_fisher: dict[str, torch.Tensor] = {
            name: torch.zeros_like(p, device=self.device)
            for name, p in model.named_parameters()
            if p.requires_grad
        }

        # Create a Fisher-specific DataLoader with controlled batch size.
        # Fisher is defined as E[grad^2]; with batch_size=1 each sample's
        # gradient is squared individually, giving the correct mean(g^2).
        dataset = train_loader.dataset
        indices = torch.randperm(n_total)[:n_target]
        fisher_loader = DataLoader(
            Subset(dataset, indices.tolist()),
            batch_size=self.fisher_batch_size,
            shuffle=False,
            num_workers=0,
        )

        n_seen = 0
        for x, y in fisher_loader:
            if n_seen >= n_target:
                break
            x, y  = x.to(self.device), y.to(self.device)
            batch = x.size(0)
            model.zero_grad()
            F.cross_entropy(model(x), y).backward()
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    task_fisher[name] += param.grad.pow(2) * batch
            n_seen += batch

        for name in task_fisher:
            task_fisher[name] /= max(n_seen, _FISHER_MIN_SEEN)

        # Online accumulation: sum Fisher contributions across tasks.
        for name, f in task_fisher.items():
            if name in self._fisher:
                self._fisher[name] += f
            else:
                self._fisher[name]  = f

        for name, param in model.named_parameters():
            if param.requires_grad:
                self._params[name] = param.detach().clone()
