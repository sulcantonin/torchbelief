"""Detecting, measuring, attributing and resolving inconsistency between sources.

Every function takes Capacity objects (so any representation works) except where
noted. ``disagreement`` is representation-free; resolution strategies then pick a
combination rule from the amount of disagreement, and ``shapley_blame`` says which
sources are responsible for it.
"""

from typing import Callable, Dict, Optional, Sequence, Tuple

import torch

from .capacity import Capacity
from .combine import yager, dempster, dubois_prade, pcr5, conflict, combine as combine_rule
from . import possibility as P


def disagreement(a: Capacity, b: Capacity) -> torch.Tensor:
    """Sure-loss gap: max_A max(lower_a(A) − upper_b(A), lower_b(A) − upper_a(A), 0).

    Positive means some event is judged more probable by one source than the other
    allows, so the credal sets are disjoint. For two 2-monotone capacities (belief
    functions, possibility measures, probabilities, ...) zero is also *sufficient* for
    the credal sets to intersect (Frank's discrete sandwich theorem), so the test is exact.
    """
    g = torch.maximum(a.lower - b.upper, b.lower - a.upper)
    return g.amax(-1).clamp_min(0)


def specific_conflict(a: Capacity, b: Capacity) -> torch.Tensor:
    """Dempster's κ computed from the Möbius masses (meaningful when both are belief functions)."""
    return conflict(a.mass(), b.mass(), a.frame)


def credal_union(a: Capacity, b: Capacity) -> Capacity:
    """Lower envelope of the convex hull of the union of the credal sets (never conflicts)."""
    return Capacity(torch.minimum(a.lower, b.lower), a.frame)


def credal_intersection(a: Capacity, b: Capacity):
    """Pointwise max of lower bounds = tightest bound if the sets intersect.
    Returns (capacity, sure_loss_gap); gap>0 means the intersection is empty and the result
    is not a coherent capacity — resolve with ``resolve``."""
    out = Capacity(torch.maximum(a.lower, b.lower), a.frame)
    gap = (out.lower - out.upper).amax(-1).clamp_min(0)  # lower > upper for some A
    return out, gap


def fuse_belief(sources: Sequence[Capacity], rule: str = "yager"):
    frame = sources[0].frame
    m, k = combine_rule([s.mass() for s in sources], frame, rule)
    return Capacity.from_mass(m, frame), k


_FALLBACK = {
    "adaptive": lambda a, b, f: yager(a.mass(), b.mass(), f)[0],
    "yager": lambda a, b, f: yager(a.mass(), b.mass(), f)[0],
    "cautious": lambda a, b, f: credal_union(a, b).mass(),
    "union": lambda a, b, f: credal_union(a, b).mass(),
    "dubois_prade": lambda a, b, f: dubois_prade(a.mass(), b.mass(), f)[0],
    "pcr5": lambda a, b, f: pcr5(a.mass(), b.mass(), f)[0],
}


def resolve(a: Capacity, b: Capacity, tau: float = 0.2, strategy: str = "adaptive",
            temperature: Optional[float] = None):
    """Agreement-aware fusion of two capacities.

    If disagreement ≤ tau use Dempster's rule (sources agree, sharpen); otherwise fall
    back to ``strategy``: 'adaptive'/'yager' (conflict → ignorance), 'cautious'/'union'
    (credal union), 'dubois_prade' or 'pcr5'. Returns (Capacity, disagreement).

    ``temperature=None`` is a hard per-sample switch (gradients flow through the chosen
    branch only). A positive temperature blends the two masses with weight
    σ((tau − d)/temperature), which is still a valid mass and is differentiable in d.
    """
    d = disagreement(a, b)
    frame = a.frame
    alt = _FALLBACK[strategy](a, b, frame)
    agree, _ = dempster(a.mass(), b.mass(), frame)
    if temperature is None:
        fused = torch.where((d <= tau).unsqueeze(-1), agree, alt)
    else:
        w = torch.sigmoid((tau - d) / temperature).unsqueeze(-1)
        fused = w * agree + (1 - w) * alt
    return Capacity.from_mass(fused, frame), d


# ---- inconsistency measures over a set of sources ------------------------------

def disagreement_matrix(sources: Sequence[Capacity]) -> torch.Tensor:
    """Pairwise disagreement (..., S, S), symmetric with zero diagonal."""
    lo = torch.stack([s.lower for s in sources], -2)              # (..., S, N)
    up = torch.stack([s.upper for s in sources], -2)
    g = lo.unsqueeze(-2) - up.unsqueeze(-3)                       # (..., S, S, N): lo_i − up_j
    g = torch.maximum(g, g.transpose(-2, -3))
    return g.amax(-1).clamp_min(0)


def _coalitions(S: int, device) -> torch.Tensor:
    idx = torch.arange(1 << S, device=device)
    return ((idx[:, None] >> torch.arange(S, device=device)) & 1).bool()   # [2^S, S]


def coalition_inconsistency(D: torch.Tensor, measure: str = "max") -> torch.Tensor:
    """I(C) for every coalition C of sources (bitmask index), from a disagreement matrix.
    'max': largest pairwise disagreement inside C; 'sum': total over pairs in C;
    'count': number of conflicting pairs (D > 0) — the MI measure restricted to pairs."""
    S = D.shape[-1]
    mem = _coalitions(S, D.device).to(D.dtype)
    pair = mem[:, :, None] * mem[:, None, :]                      # [2^S, S, S]
    if measure == "count":
        D = (D > 1e-9).to(D.dtype)
    x = D.unsqueeze(-3) * pair                                    # (..., 2^S, S, S)
    if measure == "max":
        return x.amax((-1, -2))
    return x.sum((-1, -2)) / 2


def inconsistency_measures(sources: Sequence[Capacity], tol: float = 1e-9) -> Dict[str, torch.Tensor]:
    """Argumentation-style summary of a knowledge base of sources (Hunter–Konieczny).

    drastic   1 if any pair is inconsistent
    mi        number of pairwise minimal inconsistent subsets
    max/sum   largest / total pairwise disagreement
    problematic  bool (..., S): source belongs to some conflicting pair
    free         bool (..., S): source conflicts with nobody

    Inconsistency is detected pairwise (exact per pair for 2-monotone sources);
    a triple can be jointly inconsistent while every pair is consistent.
    """
    D = disagreement_matrix(sources)
    bad = D > tol
    return {
        "drastic": bad.flatten(-2).any(-1).to(D.dtype),
        "mi": bad.flatten(-2).sum(-1).to(D.dtype) / 2,
        "max": D.flatten(-2).amax(-1),
        "sum": D.flatten(-2).sum(-1) / 2,
        "problematic": bad.any(-1),
        "free": ~bad.any(-1),
    }


def shapley_blame(sources: Sequence[Capacity], measure: str = "max") -> torch.Tensor:
    """Shapley inconsistency value of each source (..., S): its average marginal
    contribution to the coalition inconsistency. Sums to I(all sources); an outlier
    that contradicts everyone gets most of the blame. Exact enumeration, S ≲ 12."""
    D = disagreement_matrix(sources)
    S = D.shape[-1]
    I = coalition_inconsistency(D, measure)                       # (..., 2^S)
    mem = _coalitions(S, D.device)
    size = mem.sum(-1)                                            # [2^S]
    fact = torch.tensor([float(torch.arange(1, k + 1).prod()) for k in range(S + 1)],
                        dtype=D.dtype, device=D.device)
    w = fact[size] * fact[(S - 1 - size).clamp_min(0)] / fact[S]  # |C|!(S−|C|−1)!/S!
    idx = torch.arange(1 << S, device=D.device)
    out = []
    for i in range(S):
        without = ~mem[:, i]
        gain = I[..., idx | (1 << i)] - I
        out.append((gain * (w * without)).sum(-1))
    return torch.stack(out, -1)


def blame_discount(sources: Sequence[Capacity], measure: str = "max", strength: float = 1.0):
    """Discount each source by its blame: α_i = 1 − strength·φ_i (clamped to [0,1]).
    Returns (list of discounted Capacities, α (..., S)). Fuse the result with any rule."""
    phi = shapley_blame(sources, measure)
    alpha = (1 - strength * phi).clamp(0, 1)
    return [s.discount(alpha[..., i]) for i, s in enumerate(sources)], alpha
