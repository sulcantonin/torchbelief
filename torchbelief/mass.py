"""Constructors and validity checks for mass functions."""

from typing import Optional, Sequence

import torch

from .frame import Frame


def vacuous(frame: Frame, batch: Sequence[int] = (), **kw) -> torch.Tensor:
    m = torch.zeros(*batch, frame.size, **kw)
    m[..., frame.theta] = 1.0
    return m


def categorical(frame: Frame, subset: int, batch: Sequence[int] = (), **kw) -> torch.Tensor:
    m = torch.zeros(*batch, frame.size, **kw)
    m[..., subset] = 1.0
    return m


def simple_support(frame: Frame, subset: int, weight, **kw) -> torch.Tensor:
    """m(A)=w, m(Θ)=1-w. ``weight`` may be a tensor of any batch shape."""
    w = torch.as_tensor(weight, **kw)
    m = torch.zeros(*w.shape, frame.size, dtype=w.dtype, device=w.device)
    m[..., subset] = w
    m[..., frame.theta] = m[..., frame.theta] + 1 - w
    return m


def from_probabilities(frame: Frame, p: torch.Tensor) -> torch.Tensor:
    """Bayesian mass function: m({ω}) = p(ω)."""
    m = torch.zeros(*p.shape[:-1], frame.size, dtype=p.dtype, device=p.device)
    for i, s in enumerate(frame.singletons()):
        m[..., s] = p[..., i]
    return m


def from_dirichlet(frame: Frame, alpha: torch.Tensor) -> torch.Tensor:
    """Subjective-logic mapping of Dirichlet parameters α ≥ 1 (..., n) to a mass:
    m({ω_k}) = (α_k − 1)/S, m(Θ) = n/S with S = Σα (evidential deep learning)."""
    S = alpha.sum(-1, keepdim=True)
    m = from_probabilities(frame, (alpha - 1) / S)
    m[..., frame.theta] = m[..., frame.theta] + frame.n / S.squeeze(-1)
    return m


def negation(m: torch.Tensor) -> torch.Tensor:
    """m̄(A) = m(Ā): swaps the roles of conjunctive and disjunctive reasoning."""
    return m.flip(-1)


def random_mass(frame: Frame, batch: Sequence[int] = (), n_focal: Optional[int] = None,
                generator: Optional[torch.Generator] = None, dtype=torch.float64) -> torch.Tensor:
    """Random normalised mass (m(∅)=0), Dirichlet over a random focal set."""
    N = frame.size
    k = n_focal or (N - 1)
    w = torch.rand(*batch, N - 1, generator=generator, dtype=dtype)
    if k < N - 1:
        keep = w.topk(k, dim=-1).indices
        mask = torch.zeros_like(w).scatter_(-1, keep, 1.0)
        w = w * mask
    w = -torch.log(torch.rand_like(w) .clamp_min(1e-12)) * (w > 0)
    w = w / w.sum(-1, keepdim=True)
    return torch.cat([torch.zeros(*batch, 1, dtype=dtype), w], dim=-1)


def is_valid(m: torch.Tensor, atol: float = 1e-6, closed_world: bool = True) -> bool:
    ok = bool((m >= -atol).all()) and bool(((m.sum(-1) - 1).abs() <= atol).all())
    if closed_world:
        ok = ok and bool((m[..., 0].abs() <= atol).all())
    return ok
