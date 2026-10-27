"""Correction mechanisms: what to do with a source that may be unreliable or untruthful.

A correction combines the source's mass with a corrective mass built from degrees
β ∈ [0, 1] given per *context* (a subset of Θ, as an integer bitmask). β = 1 leaves the
source untouched. All functions are batched over leading dimensions of ``m`` and ``beta``
(``beta`` has shape (..., L) for L contexts) and differentiable in both.

- ``discount`` / ``contextual_discount``       : the source may be irrelevant (∪ rule)
- ``reinforce`` / ``contextual_reinforce``     : the source may be a positive liar (∩ rule)
- ``negate`` / ``contextual_negate``           : the source may say the contrary of what it knows
- ``contextual_dediscount`` / ``contextual_dereinforce`` : undo a previous CD / CR

Default contexts follow Pichon et al. (2016): singletons {ω_k} for discounting, their
complements Θ∖{ω_k} for reinforcement and negating. These are the parametrisations
whose contour functions are linear in β, which makes them easy to learn.

Mercier, Quost & Denœux 2008 · Pichon, Mercier, Lefèvre & Delmotte 2016 (IJAR 72).
"""

from typing import Optional, Sequence

import torch

from .frame import Frame
from . import transforms as T


def _singletons(frame: Frame, contexts):
    return frame.singletons() if contexts is None else list(contexts)


def _complements(frame: Frame, contexts):
    return [frame.theta ^ s for s in frame.singletons()] if contexts is None else list(contexts)


def _factor(m: torch.Tensor, beta: torch.Tensor, contexts: Sequence[int], frame: Frame,
            contains: bool) -> torch.Tensor:
    """Π_ℓ (β_ℓ + (1−β_ℓ)·[θ_ℓ ⊆ A]) if ``contains``, else Π_ℓ (β_ℓ + (1−β_ℓ)·[A ⊆ θ_ℓ]).

    The first is the implicability of ∪_ℓ (∅ w.p. β_ℓ, θ_ℓ w.p. 1−β_ℓ), the second the
    commonality of ∩_ℓ (Θ w.p. β_ℓ, θ_ℓ w.p. 1−β_ℓ)."""
    idx = torch.arange(frame.size, device=m.device).unsqueeze(-1)
    ctx = torch.tensor(contexts, device=m.device)
    inside = ((idx & ctx) == (ctx if contains else idx)).to(m.dtype)    # [N, L]
    b = torch.as_tensor(beta, dtype=m.dtype, device=m.device).unsqueeze(-2)
    return (b + (1 - b) * inside).prod(-1)


def discount(m: torch.Tensor, alpha, frame: Frame) -> torch.Tensor:
    """Shafer discounting: m' = α·m + (1-α)·δ_Θ (α = source reliability)."""
    a = torch.as_tensor(alpha, dtype=m.dtype, device=m.device)
    if a.dim() > 0:
        a = a.unsqueeze(-1)
    out = a * m
    return torch.cat([out[..., :-1], out[..., -1:] + (1 - a)], dim=-1)


def contextual_discount(m: torch.Tensor, beta: torch.Tensor, frame: Frame,
                        contexts: Optional[Sequence[int]] = None) -> torch.Tensor:
    """Mercier–Quost–Denœux contextual discounting. ``beta`` (..., L) is the source's
    reliability *given* the true value lies in context θ_ℓ (1 = trusted, 0 = ignored);
    contexts default to the singletons, a partition of Θ gives Θ-contextual discounting.

    m' = m ∪ (∪_ℓ m_ℓ) with m_ℓ(∅)=β_ℓ, m_ℓ(θ_ℓ)=1−β_ℓ, which for singletons gives
    pl'({ω_k}) = 1 − β_k + β_k·pl({ω_k}). (Equal β_k is *not* Shafer discounting;
    use :func:`discount` for that, or the single context Θ.)"""
    f = _factor(m, beta, _singletons(frame, contexts), frame, contains=True)
    return T.subset_sum_inverse(T.subset_sum(m, frame) * f, frame)


def contextual_dediscount(m: torch.Tensor, beta: torch.Tensor, frame: Frame,
                          contexts: Optional[Sequence[int]] = None, eps: float = 1e-12
                          ) -> torch.Tensor:
    """Inverse of :func:`contextual_discount` (disjunctive decombination), β ∈ (0, 1].
    Valid only if ``m`` was actually discounted at least that much; check with ``is_valid``."""
    f = _factor(m, beta, _singletons(frame, contexts), frame, contains=True)
    return T.subset_sum_inverse(T.subset_sum(m, frame) / f.clamp_min(eps), frame)


def reinforce(m: torch.Tensor, beta, frame: Frame) -> torch.Tensor:
    """Reinforcement, the dual of discounting: m'(A) = β·m(A) for A ≠ ∅ and the
    remaining 1−β goes to ∅ (with probability 1−β the source lies about everything)."""
    b = torch.as_tensor(beta, dtype=m.dtype, device=m.device)
    if b.dim() > 0:
        b = b.unsqueeze(-1)
    out = b * m
    return torch.cat([out[..., :1] + (1 - b), out[..., 1:]], dim=-1)


def contextual_reinforce(m: torch.Tensor, beta: torch.Tensor, frame: Frame,
                         contexts: Optional[Sequence[int]] = None) -> torch.Tensor:
    """Contextual reinforcement (Pichon et al. 2016): m' = m ∩ (∩_ℓ A_ℓ^{β_ℓ}) where A^β has
    m(Θ)=β, m(A)=1−β. With probability 1−β_ℓ the source is a positive liar in A_ℓ.
    Contexts default to Θ∖{ω_k}, for which pl'({ω_k}) = β_k·pl({ω_k})."""
    f = _factor(m, beta, _complements(frame, contexts), frame, contains=False)
    return T.superset_sum_inverse(T.superset_sum(m, frame) * f, frame)


def contextual_dereinforce(m: torch.Tensor, beta: torch.Tensor, frame: Frame,
                           contexts: Optional[Sequence[int]] = None, eps: float = 1e-12
                           ) -> torch.Tensor:
    """Inverse of :func:`contextual_reinforce` (conjunctive decombination), β ∈ (0, 1]."""
    f = _factor(m, beta, _complements(frame, contexts), frame, contains=False)
    return T.superset_sum_inverse(T.superset_sum(m, frame) / f.clamp_min(eps), frame)


def negate(m: torch.Tensor, beta, frame: Frame) -> torch.Tensor:
    """Negating: m' = β·m + (1−β)·m̄, where m̄(A) = m(Ā) (see :func:`torchbelief.negation`).
    β is the probability that the source is truthful; β = 0 is pure negation."""
    b = torch.as_tensor(beta, dtype=m.dtype, device=m.device)
    if b.dim() > 0:
        b = b.unsqueeze(-1)
    return b * m + (1 - b) * m.flip(-1)


def contextual_negate(m: torch.Tensor, beta: torch.Tensor, frame: Frame,
                      contexts: Optional[Sequence[int]] = None) -> torch.Tensor:
    """Contextual negating (Pichon et al. 2016): for each context A, with probability
    1−β_A the source tells the contrary of what it knows about the values in A, so the
    focal set B moves to (B∩A) ∪ (B̄∩Ā) = B XOR Ā. Contexts default to Θ∖{ω_k}, for
    which pl'({ω_k}) = ½ + (pl({ω_k}) − ½)(2β_k − 1); context ∅ gives :func:`negate`."""
    contexts = _complements(frame, contexts)
    b = torch.as_tensor(beta, dtype=m.dtype, device=m.device)
    idx = torch.arange(frame.size, device=m.device)
    for l, a in enumerate(contexts):
        bl = b[..., l:l + 1]
        m = bl * m + (1 - bl) * m[..., idx ^ (frame.theta ^ a)]
    return m
