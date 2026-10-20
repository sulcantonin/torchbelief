"""Possibility distributions (fuzzy/consonant evidence) and their fusion rules.

π: (..., n) in [0,1]. Rules mirror the belief-function ones: conjunctive rules
create conflict (h<1), disjunctive ones don't, adaptive rules interpolate.
"""

from typing import Callable, Tuple

import torch

TNORMS = {
    "min": lambda a, b: torch.minimum(a, b),
    "product": lambda a, b: a * b,
    "lukasiewicz": lambda a, b: (a + b - 1).clamp_min(0),
}


def consistency(p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """h = max_ω min(π1, π2): degree of agreement, 1 − h is the conflict."""
    return torch.minimum(p1, p2).amax(-1)


def conjunctive(p1, p2, tnorm: str = "min", normalise: bool = False):
    p = TNORMS[tnorm](p1, p2)
    h = p.amax(-1)
    if normalise:  # Dempster-like renormalisation; blows up as h→0
        p = p / h.clamp_min(1e-8).unsqueeze(-1)
    return p, 1 - h


def disjunctive(p1, p2):
    return torch.maximum(p1, p2), torch.zeros_like(p1[..., 0])


def adaptive(p1, p2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dubois–Prade adaptive rule: π = max( min(π1,π2)/h , min(max(π1,π2), 1−h) ).
    Behaves conjunctively when sources agree and disjunctively when they conflict."""
    h = consistency(p1, p2).unsqueeze(-1)
    conj = torch.minimum(p1, p2) / h.clamp_min(1e-8)
    disj = torch.minimum(torch.maximum(p1, p2), 1 - h)
    return torch.maximum(conj, disj), 1 - h.squeeze(-1)


def yager_style(p1, p2):
    """Conflict goes to total ignorance: π = min(π1,π2) on elements, plus (1−h) everywhere."""
    p, k = conjunctive(p1, p2, "min")
    return (p + k.unsqueeze(-1)).clamp(max=1), k


def discount(p: torch.Tensor, alpha) -> torch.Tensor:
    """Reliability discounting: π' = max(π, 1−α)."""
    a = torch.as_tensor(alpha, dtype=p.dtype, device=p.device)
    if a.dim() > 0:
        a = a.unsqueeze(-1)
    return torch.maximum(p, 1 - a)
