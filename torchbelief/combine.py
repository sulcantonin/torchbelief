"""Rules of combination and discounting.

Conventions: masses are tensors ``(..., 2**n)``; every rule broadcasts over
leading dimensions and is differentiable. Rules that discard or move conflict
return ``(m, conflict)`` where ``conflict`` has shape ``(...,)``.

- ``conjunctive``  : unnormalised (Smets TBM), keeps conflict in m(∅)
- ``dempster``     : conjunctive renormalised by 1/(1-κ)  (gradient unbounded as κ→1)
- ``yager``        : conflict goes to Θ                    (gradients bounded)
- ``dubois_prade`` : conflicting pair (A,B) goes to A∪B
- ``disjunctive``  : A∪B for every pair                    (no conflict, max cautious)
- ``pcr5`` / ``pcr6``: partial conflict sent back to its sources proportionally
- ``cautious`` / ``bold`` : idempotent rules for non-distinct sources (Denœux)
- ``average`` / ``murphy`` : weighted mean / mean combined with itself
- ``decombine_*`` : inverses of the conjunctive / disjunctive rules
- ``condition`` / ``decondition`` : Dempster conditioning and its least committed inverse
- ``discount`` / ``contextual_discount`` : reliability, global or per context
  (reinforcement and negating are in :mod:`torchbelief.corrections`)
"""

from typing import Callable, Dict, Optional, Sequence, Tuple

import torch

from .frame import Frame
from . import transforms as T
from .corrections import contextual_discount, discount  # noqa: F401  (re-exported)


def _tables(frame: Frame, device):
    idx = torch.arange(frame.size, device=device)
    return idx[:, None] & idx[None, :], idx[:, None] | idx[None, :]


def conjunctive(m1: torch.Tensor, m2: torch.Tensor, frame: Frame) -> torch.Tensor:
    """m(C) = Σ_{A∩B=C} m1(A) m2(B), via q = q1·q2 in O(n·2^n)."""
    q = T.superset_sum(m1, frame) * T.superset_sum(m2, frame)
    return T.superset_sum_inverse(q, frame)


def disjunctive(m1: torch.Tensor, m2: torch.Tensor, frame: Frame) -> torch.Tensor:
    """m(C) = Σ_{A∪B=C} m1(A) m2(B), via b = b1·b2."""
    b = T.subset_sum(m1, frame) * T.subset_sum(m2, frame)
    return T.subset_sum_inverse(b, frame)


def dempster(m1, m2, frame: Frame, eps: float = 1e-8, clamp: bool = True
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dempster's rule. ``clamp=False`` exposes the true 1/(1-κ) blow-up."""
    m = conjunctive(m1, m2, frame)
    k = m[..., 0]
    denom = 1.0 - k
    if clamp:
        denom = denom.clamp_min(eps)
    m = torch.cat([torch.zeros_like(m[..., :1]), m[..., 1:] / denom.unsqueeze(-1)], dim=-1)
    return m, k


def yager(m1, m2, frame: Frame) -> Tuple[torch.Tensor, torch.Tensor]:
    m = conjunctive(m1, m2, frame)
    k = m[..., 0]
    parts = [torch.zeros_like(k).unsqueeze(-1), m[..., 1:-1], (m[..., -1] + k).unsqueeze(-1)]
    return torch.cat(parts, dim=-1), k


def dubois_prade(m1, m2, frame: Frame) -> Tuple[torch.Tensor, torch.Tensor]:
    """Conflicting mass m1(A)m2(B) with A∩B=∅ is assigned to A∪B.
    O(4^n) memory; intended for n ≲ 10."""
    inter, union = _tables(frame, m1.device)
    target = torch.where(inter == 0, union, inter)
    a = m1.clone(); a[..., 0] = 0
    b = m2.clone(); b[..., 0] = 0
    joint = a.unsqueeze(-1) * b.unsqueeze(-2)  # [..., N, N]
    shape = torch.broadcast_shapes(a.shape[:-1], b.shape[:-1])
    joint = joint.expand(*shape, frame.size, frame.size)
    out = torch.zeros(*shape, frame.size, dtype=joint.dtype, device=joint.device)
    out = out.scatter_add(-1, target.reshape(-1).expand(*shape, -1), joint.reshape(*shape, -1))
    k = (joint * (inter == 0)).sum((-1, -2))
    return out, k


def conflict(m1, m2, frame: Frame) -> torch.Tensor:
    """κ = Σ_{A∩B=∅} m1(A) m2(B) (cheap: only the ∅ coefficient)."""
    q = T.superset_sum(m1, frame) * T.superset_sum(m2, frame)
    # m∩(∅) = Σ_A (-1)^{|A|} q(A)
    sign = (-1.0) ** frame.cardinality(q)
    return (sign * q).sum(-1)


def average(masses: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """masses (..., S, 2**n) → (..., 2**n)."""
    if weights is None:
        return masses.mean(-2)
    w = weights / weights.sum(-1, keepdim=True)
    return (w.unsqueeze(-1) * masses).sum(-2)


def murphy(masses: torch.Tensor, frame: Frame, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Murphy (2000): average the S sources, then Dempster-combine the average S-1 times."""
    avg = average(masses, weights)
    out = avg
    for _ in range(masses.shape[-2] - 1):
        out, _ = dempster(out, avg, frame)
    return out


# ---- proportional conflict redistribution (DSmT) ---------------------------

def pcr5(m1, m2, frame: Frame, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    """PCR5 (Smarandache–Dezert): the partial conflict m1(A)m2(B), A∩B=∅, is sent back
    to A and B proportionally to m1(A) and m2(B). O(4^n) memory; intended for n ≲ 10."""
    return pcr6([m1, m2], frame, eps)


def pcr6(masses: Sequence[torch.Tensor], frame: Frame, eps: float = 1e-12
         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """PCR6 (Martin–Osswald) for S sources; equals PCR5 when S=2.

    For every S-tuple (X_1..X_S) with empty intersection, the product Π m_i(X_i) is
    redistributed to each X_i proportionally to m_i(X_i). Not associative, so all
    sources are combined at once: O(2^{nS}) memory, intended for small n·S.
    """
    if isinstance(masses, torch.Tensor):
        masses = list(masses.unbind(-2))
    S, N = len(masses), frame.size
    lead = torch.broadcast_shapes(*(m.shape[:-1] for m in masses))
    ms = [m.expand(*lead, N).clone() for m in masses]
    for m in ms:
        m[..., 0] = 0
    idx = torch.arange(N, device=ms[0].device)
    grids = torch.meshgrid(*([idx] * S), indexing="ij")          # S × [N]*S
    inter = grids[0]
    for g in grids[1:]:
        inter = inter & g
    conf = (inter == 0).reshape(-1)
    # per-source factor broadcast onto the joint grid: (..., N^S)
    view = lambda i, m: m.reshape(*lead, *([1] * i), N, *([1] * (S - i - 1)))
    factors = [view(i, m) for i, m in enumerate(ms)]
    joint = factors[0]
    for f in factors[1:]:
        joint = joint * f
    joint = joint.expand(*lead, *([N] * S)).reshape(*lead, -1)
    denom = sum(factors).expand(*lead, *([N] * S)).reshape(*lead, -1).clamp_min(eps)
    out = torch.zeros(*lead, N, dtype=joint.dtype, device=joint.device)
    out = out.scatter_add(-1, inter.reshape(-1).expand(*lead, -1), joint * ~conf)
    for i, f in enumerate(factors):
        share = joint * f.expand(*lead, *([N] * S)).reshape(*lead, -1) / denom * conf
        out = out.scatter_add(-1, grids[i].reshape(-1).expand(*lead, -1), share)
    return out, (joint * conf).sum(-1)


# ---- cautious / bold rules (canonical decomposition) -------------------------

def conjunctive_weights(m: torch.Tensor, frame: Frame, eps: float = 1e-12) -> torch.Tensor:
    """log w(A) of the canonical conjunctive decomposition m = ∩_{A≠Θ} A^{w(A)} (Smets).
    Requires a non-dogmatic mass (m(Θ) > 0); the Θ entry is meaningless and set to 0."""
    q = T.superset_sum(m, frame).clamp_min(eps)
    lw = -T.superset_sum_inverse(q.log(), frame)
    return torch.cat([lw[..., :-1], torch.zeros_like(lw[..., -1:])], -1)


def from_conjunctive_weights(log_w: torch.Tensor, frame: Frame) -> torch.Tensor:
    """Inverse of :func:`conjunctive_weights`: q(C) = Π_{B≠Θ, C⊄B} w(B)."""
    lw = torch.cat([log_w[..., :-1], torch.zeros_like(log_w[..., -1:])], -1)
    log_q = lw.sum(-1, keepdim=True) - T.superset_sum(lw, frame)
    return T.superset_sum_inverse(log_q.exp(), frame)


def disjunctive_weights(m: torch.Tensor, frame: Frame, eps: float = 1e-12) -> torch.Tensor:
    """log v(A) of the canonical disjunctive decomposition (Denœux 2008), the dual of
    :func:`conjunctive_weights` through negation: v(A) = w̄(Ā) with m̄(A) = m(Ā).
    Requires a subnormal mass (m(∅) > 0); the ∅ entry is meaningless and set to 0."""
    return conjunctive_weights(m.flip(-1), frame, eps).flip(-1)


def from_disjunctive_weights(log_v: torch.Tensor, frame: Frame) -> torch.Tensor:
    """Inverse of :func:`disjunctive_weights`."""
    return from_conjunctive_weights(log_v.flip(-1), frame).flip(-1)


# ---- decombination and conditioning ----------------------------------------------

def decombine_conjunctive(m12: torch.Tensor, m2: torch.Tensor, frame: Frame, eps: float = 1e-12
                          ) -> torch.Tensor:
    """Remove m2 from a conjunctive combination: q = q12 / q2 (Smets' decombination).
    m2 must be non-dogmatic (m2(Θ) > 0). The result is a valid mass only if m12 really
    contains m2; check it with :func:`torchbelief.is_valid`."""
    q = T.superset_sum(m12, frame) / T.superset_sum(m2, frame).clamp_min(eps)
    return T.superset_sum_inverse(q, frame)


def decombine_disjunctive(m12: torch.Tensor, m2: torch.Tensor, frame: Frame, eps: float = 1e-12
                          ) -> torch.Tensor:
    """Remove m2 from a disjunctive combination: b = b12 / b2. m2 must be subnormal
    (m2(∅) > 0). As for :func:`decombine_conjunctive`, the result may be invalid."""
    b = T.subset_sum(m12, frame) / T.subset_sum(m2, frame).clamp_min(eps)
    return T.subset_sum_inverse(b, frame)


def condition(m: torch.Tensor, event: int, frame: Frame, normalise: bool = True
              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dempster conditioning on the event A: m[A](B) = Σ_{C∩A=B} m(C), i.e. the conjunctive
    combination with the categorical mass on A. Returns ``(m[A], κ)`` with κ = mass of the
    focal sets disjoint from A; ``normalise=False`` keeps κ on ∅ (TBM conditioning)."""
    idx = torch.arange(frame.size, device=m.device) & event
    out = torch.zeros_like(m).scatter_add(-1, idx.expand_as(m), m)
    return _normalise(out) if normalise else (out, out[..., 0])


def decondition(m: torch.Tensor, event: int, frame: Frame) -> torch.Tensor:
    """Least committed mass whose conditioning on A gives m: m*(B ∪ Ā) = m(B)
    (Smets' ballooning extension)."""
    idx = torch.arange(frame.size, device=m.device) | (frame.theta ^ event)
    return torch.zeros_like(m).scatter_add(-1, idx.expand_as(m), m)


def _normalise(m: torch.Tensor, eps: float = 1e-8):
    k = m[..., 0]
    return torch.cat([torch.zeros_like(m[..., :1]),
                      m[..., 1:] / (1 - k).clamp_min(eps).unsqueeze(-1)], -1), k


def cautious(m1, m2, frame: Frame, normalise: bool = True, eps: float = 1e-12
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Denœux's cautious rule: w = min(w1, w2). Idempotent, so it is safe for
    *non-distinct* sources (e.g. two classifiers trained on overlapping data).
    Sources must be non-dogmatic (m(Θ) > 0; discount first if needed)."""
    lw = torch.minimum(conjunctive_weights(m1, frame, eps), conjunctive_weights(m2, frame, eps))
    m = from_conjunctive_weights(lw, frame)
    return _normalise(m) if normalise else (m, m[..., 0])


def bold(m1, m2, frame: Frame, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    """Denœux's bold rule: the disjunctive, idempotent counterpart of :func:`cautious`
    (min of disjunctive weights). Sources must be subnormal (m(∅) > 0)."""
    m, _ = cautious(m1.flip(-1), m2.flip(-1), frame, normalise=False, eps=eps)
    return m.flip(-1), torch.zeros_like(m[..., 0])


RULES: Dict[str, Callable] = {
    "conjunctive": lambda a, b, f: (conjunctive(a, b, f), conflict(a, b, f)),
    "disjunctive": lambda a, b, f: (disjunctive(a, b, f), torch.zeros_like(a[..., 0])),
    "dempster": dempster,
    "yager": yager,
    "dubois_prade": dubois_prade,
    "pcr5": pcr5,
    "cautious": cautious,
    "bold": bold,
}


def combine(masses: Sequence[torch.Tensor], frame: Frame, rule: str = "yager"
            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Left-fold ``rule`` over a list of masses (or a tensor (..., S, 2**n)).

    Returns the fused mass and the *sum* of the pairwise conflicts encountered.
    Associative rules (conjunctive, dempster, disjunctive, cautious, bold) are
    order-independent; yager, dubois_prade and pcr5 are not, so pass sources in a fixed
    order. ``"pcr6"`` combines all sources at once; ``"murphy"`` averages first.
    """
    if isinstance(masses, torch.Tensor):
        masses = list(masses.unbind(-2))
    if rule == "murphy":
        return murphy(torch.stack(list(masses), -2), frame), torch.zeros_like(masses[0][..., 0])
    if rule == "pcr6":
        return pcr6(masses, frame)
    fn = RULES[rule]
    acc, total = masses[0], torch.zeros_like(masses[0][..., 0])
    for m in masses[1:]:
        acc, k = fn(acc, m, frame)
        total = total + k
    return acc, total
