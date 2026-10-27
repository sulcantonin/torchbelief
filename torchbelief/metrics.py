"""Evaluation of set-valued and selective predictions (not differentiable).

A set prediction is a bool mask (..., n), e.g. from :func:`torchbelief.strong_dominance`;
targets are class indices (...). For training objectives see
:func:`torchbelief.expected_utility_loss` and :func:`torchbelief.pl_loss`.

Zaffalon, Corani & Mauá 2012 (u65/u80).
"""

from typing import Tuple

import torch


def _reduce(x: torch.Tensor, reduction: str) -> torch.Tensor:
    return {"mean": x.mean, "sum": x.sum, "none": lambda: x}[reduction]()


def discounted_accuracy(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """1/|A| if the true class is in the predicted set A, else 0 (and 0 for A = ∅)."""
    hit = pred.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    x = hit / pred.sum(-1).clamp_min(1)
    return _reduce(x.to(torch.get_default_dtype()), reduction)


def utility_score(pred: torch.Tensor, target: torch.Tensor, a: float, b: float,
                  reduction: str = "mean") -> torch.Tensor:
    """a·x − b·x² with x the discounted accuracy: rewards cautious sets more than 1/|A| does."""
    x = discounted_accuracy(pred, target, "none")
    return _reduce(a * x - b * x * x, reduction)


def u65(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Utility-discounted accuracy: a correct pair scores 0.65 instead of 0.5."""
    return utility_score(pred, target, 1.6, 0.6, reduction)


def u80(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Utility-discounted accuracy: a correct pair scores 0.80 instead of 0.5."""
    return utility_score(pred, target, 2.2, 1.2, reduction)


def error_rejection_curve(confidence: torch.Tensor, correct: torch.Tensor
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Error rate of the precise predictions that remain after rejecting the k least
    confident ones, for k = 0..N. Inputs are (N,); returns (rejection rates, error rates),
    each (N+1,). A typical confidence is the top plausibility-transform probability:
    ``conf, pred = plausibility_transform(m, F).max(-1)``; ``correct = pred == y``."""
    N = confidence.numel()
    order = torch.sort(confidence, stable=True).indices
    kept = correct[order].to(torch.get_default_dtype()).flip(0).cumsum(0).flip(0)   # correct among k..N−1
    remaining = torch.arange(N, 0, -1, dtype=kept.dtype, device=kept.device)
    errors = torch.cat([1 - kept / remaining, kept.new_zeros(1)])
    return torch.arange(N + 1, dtype=kept.dtype, device=kept.device) / N, errors


def auerc(confidence: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
    """Area under :func:`error_rejection_curve` (lower is better)."""
    rates, errors = error_rejection_curve(confidence, correct)
    return torch.trapezoid(errors, rates)
