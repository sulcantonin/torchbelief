"""Decision rules and uncertainty/conflict measures for mass functions."""

import torch

from .frame import Frame
from . import transforms as T


def pignistic(m: torch.Tensor, frame: Frame) -> torch.Tensor:
    """BetP(ω) = Σ_{A∋ω} m(A)/|A| / (1-m(∅))  → (..., n)."""
    card = frame.cardinality(m).clamp_min(1)
    w = (m / card)  # ∅ has card clamped to 1, but its row of membership is all-False
    p = w @ frame.membership(m)
    return p / (1 - m[..., :1]).clamp_min(1e-12)


def plausibility_transform(m: torch.Tensor, frame: Frame) -> torch.Tensor:
    """Smets' plausibility transform: normalised singleton plausibilities (Dempster-consistent)."""
    pl = T.plausibility(m, frame)[..., frame.singletons()]
    return pl / pl.sum(-1, keepdim=True)


def interval(m: torch.Tensor, frame: Frame):
    """(Bel, Pl) of every singleton → two tensors (..., n)."""
    s = frame.singletons()
    return T.belief(m, frame)[..., s], T.plausibility(m, frame)[..., s]


def vacuity(m: torch.Tensor, frame: Frame) -> torch.Tensor:
    """m(Θ): mass of total ignorance."""
    return m[..., frame.theta]


def expected_utility_bounds(m: torch.Tensor, util: torch.Tensor, frame: Frame):
    """Lower/upper (Choquet) expected utility of each act.

    util: (acts, n). Returns (lower, upper) each (..., acts):
      lower(a) = Σ_A m(A) min_{ω∈A} u(a,ω),   upper(a) = Σ_A m(A) max_{ω∈A} u(a,ω).
    """
    mem = frame.membership().to(m.device)  # [N, n] bool
    u = util.to(m.dtype)[:, None, :]       # [acts, 1, n]
    big = torch.finfo(m.dtype).max
    mn = torch.where(mem[None], u, torch.full_like(u, big).expand(-1, frame.size, -1)).amin(-1)
    mx = torch.where(mem[None], u, torch.full_like(u, -big).expand(-1, frame.size, -1)).amax(-1)
    mn, mx = mn.nan_to_num(), mx.nan_to_num()
    mn[:, 0] = 0; mx[:, 0] = 0  # ∅ carries no utility
    return m @ mn.T, m @ mx.T


def decide(m: torch.Tensor, util: torch.Tensor, frame: Frame, criterion: str = "pignistic",
           hurwicz: float = 0.5) -> torch.Tensor:
    """Index of the chosen act. criteria: pignistic | maximin | maximax | hurwicz."""
    if criterion == "pignistic":
        score = pignistic(m, frame) @ util.to(m.dtype).T
    else:
        lo, hi = expected_utility_bounds(m, util, frame)
        score = {"maximin": lo, "maximax": hi, "hurwicz": hurwicz * hi + (1 - hurwicz) * lo}[criterion]
    return score.argmax(-1)


# ---- uncertainty measures -------------------------------------------------

def nonspecificity(m: torch.Tensor, frame: Frame) -> torch.Tensor:
    """Dubois-Prade/Hartley: Σ m(A) log2|A|  (0 for Bayesian, log2 n for vacuous)."""
    card = frame.cardinality(m).clamp_min(1)
    return (m * card.log2()).sum(-1)


def deng_entropy(m: torch.Tensor, frame: Frame, eps: float = 1e-12) -> torch.Tensor:
    """-Σ m(A) log2( m(A) / (2^|A| − 1) )."""
    card = frame.cardinality(m)
    denom = (2.0 ** card - 1).clamp_min(1)
    return -(m * (m.clamp_min(eps) / denom).log2() * (m > 0)).sum(-1)


def pignistic_entropy(m: torch.Tensor, frame: Frame, eps: float = 1e-12) -> torch.Tensor:
    p = pignistic(m, frame)
    return -(p * p.clamp_min(eps).log2()).sum(-1)


def ambiguity_measure(m: torch.Tensor, frame: Frame) -> torch.Tensor:
    """Jiroušek–Shenoy AM = entropy of the plausibility transform + nonspecificity."""
    p = plausibility_transform(m, frame)
    return -(p * p.clamp_min(1e-12).log2()).sum(-1) + nonspecificity(m, frame)


def jousselme_distance(m1: torch.Tensor, m2: torch.Tensor, frame: Frame) -> torch.Tensor:
    idx = torch.arange(frame.size, device=m1.device)
    inter, union = idx[:, None] & idx[None, :], idx[:, None] | idx[None, :]
    c = frame.cardinality(m1)
    D = torch.where(union == 0, torch.ones_like(inter, dtype=m1.dtype),
                    c[inter] / c[union].clamp_min(1))
    d = (m1 - m2)
    return (0.5 * ((d @ D) * d).sum(-1)).clamp_min(0).sqrt()


def conflict_weight(kappa: torch.Tensor) -> torch.Tensor:
    """Con = log 1/(1-κ): Shafer's weight of conflict."""
    return -torch.log1p(-kappa.clamp(max=1 - 1e-12))
