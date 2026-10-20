"""Fast (n·2^n) zeta/Möbius transforms between mass, belief and commonality.

All functions act on the last dimension (size 2**n), are batched and
differentiable.
"""

import torch

from .frame import Frame


def _axes(x: torch.Tensor, n: int):
    return x.reshape(*x.shape[:-1], *([2] * n))


def _sum_over_subsets(x, n, d):
    return x.cumsum(dim=d)


def _inv_over_subsets(x, n, d):
    x0, x1 = x.narrow(d, 0, 1), x.narrow(d, 1, 1)
    return torch.cat([x0, x1 - x0], dim=d)


def _sum_over_supersets(x, n, d):
    return x.flip(d).cumsum(dim=d).flip(d)


def _inv_over_supersets(x, n, d):
    x0, x1 = x.narrow(d, 0, 1), x.narrow(d, 1, 1)
    return torch.cat([x0 - x1, x1], dim=d)


def _apply(x, frame: Frame, op):
    shape = x.shape
    y = _axes(x, frame.n)
    for ax in range(frame.n):
        y = op(y, frame.n, -frame.n + ax)
    return y.reshape(shape)


def subset_sum(m: torch.Tensor, frame: Frame) -> torch.Tensor:
    """f(A) = Σ_{B ⊆ A} m(B)  (includes m(∅))."""
    return _apply(m, frame, _sum_over_subsets)


def subset_sum_inverse(f: torch.Tensor, frame: Frame) -> torch.Tensor:
    return _apply(f, frame, _inv_over_subsets)


def superset_sum(m: torch.Tensor, frame: Frame) -> torch.Tensor:
    """q(A) = Σ_{B ⊇ A} m(B)  (commonality)."""
    return _apply(m, frame, _sum_over_supersets)


def superset_sum_inverse(q: torch.Tensor, frame: Frame) -> torch.Tensor:
    return _apply(q, frame, _inv_over_supersets)


def belief(m: torch.Tensor, frame: Frame) -> torch.Tensor:
    """Bel(A) = Σ_{∅ ≠ B ⊆ A} m(B)."""
    return subset_sum(m, frame) - m[..., :1]


def plausibility(m: torch.Tensor, frame: Frame) -> torch.Tensor:
    """Pl(A) = Σ_{B ∩ A ≠ ∅} m(B)  =  total − Σ_{B ⊆ Ā} m(B)."""
    return m.sum(-1, keepdim=True) - subset_sum(m, frame).flip(-1)


def commonality(m: torch.Tensor, frame: Frame) -> torch.Tensor:
    return superset_sum(m, frame)


def mass_from_belief(bel: torch.Tensor, frame: Frame) -> torch.Tensor:
    """Inverse of :func:`belief` for a normalised mass (m(∅)=0)."""
    b = bel.clone()
    b[..., 0] = 0
    return subset_sum_inverse(b, frame)


def mass_from_commonality(q: torch.Tensor, frame: Frame) -> torch.Tensor:
    return superset_sum_inverse(q, frame)
