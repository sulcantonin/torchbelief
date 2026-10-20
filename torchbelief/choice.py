"""Decision making under imprecise/inconsistent uncertainty (capacity level).

Acts are rows of a utility matrix ``util`` (acts, n). All rules accept batches of
Capacity (leading dims) and return act indices or sets. ``-1`` encodes abstention.
"""

import torch

from .capacity import Capacity


def scores(cap: Capacity, util: torch.Tensor):
    """(lower, upper) expected utility per act, each (..., acts)."""
    return cap.expectation_bounds(util)


def maximin(cap, util):  # Γ-maximin: cautious
    return scores(cap, util)[0].argmax(-1)


def maximax(cap, util):
    return scores(cap, util)[1].argmax(-1)


def hurwicz(cap, util, alpha: float = 0.5):
    lo, hi = scores(cap, util)
    return (alpha * hi + (1 - alpha) * lo).argmax(-1)


def pignistic_choice(cap, util):
    return (cap.pignistic() @ util.to(cap.v.dtype).T).argmax(-1)


def maximal_acts(cap: Capacity, util: torch.Tensor, tol: float = 1e-9) -> torch.Tensor:
    """Walley maximality. Bool (..., acts): act a is kept unless some b has
    lower expectation of (u_b − u_a) > 0 (b is strictly better for every distribution)."""
    u = util.to(cap.v.dtype)
    diff = u[None, :, :] - u[:, None, :]            # [a,b,n] = u_b − u_a
    lo, _ = cap.expectation_bounds(diff.reshape(-1, cap.frame.n))  # (..., a*b)
    lo = lo.reshape(*lo.shape[:-1], u.shape[0], u.shape[0])
    return ~((lo > tol).any(-1))                     # beaten if exists b with lo[a,b]>0


def interval_dominance_acts(cap: Capacity, util: torch.Tensor, tol: float = 1e-9) -> torch.Tensor:
    """Keep act a unless some b has lower(b) > upper(a) + tol. The tolerance stops a
    precise act (lower == upper up to rounding) from being dominated by itself."""
    lo, hi = scores(cap, util)
    best_lo = lo.amax(-1, keepdim=True)
    return hi >= best_lo - tol


def e_admissible_acts(cap: Capacity, util: torch.Tensor, tol: float = 1e-9) -> torch.Tensor:
    """Acts that are Bayes-optimal for at least one vertex of the credal set (convex v)."""
    verts = cap.credal_vertices()                              # (..., P, n)
    eu = verts @ util.to(verts.dtype).T                        # (..., P, acts)
    best = eu.amax(-1, keepdim=True)
    return ((eu >= best - tol).any(-2))


def decide_or_abstain(cap: Capacity, util: torch.Tensor, rule: str = "maximality") -> torch.Tensor:
    """Pick the unique undominated act, else −1 (abstain / defer): the principled
    answer when the evidence is too imprecise or conflicting to single out an act."""
    keep = {"maximality": maximal_acts, "interval": interval_dominance_acts,
            "e_admissible": e_admissible_acts}[rule](cap, util)
    unique = keep.sum(-1) == 1
    return torch.where(unique, keep.float().argmax(-1), torch.full_like(keep.sum(-1), -1))
