"""Capacity: the central object.

A Capacity stores the *lower* set function v on 2^Θ (v(∅)=0, v(Θ)=1,
monotone), shape ``(..., 2**n)``. Every other uncertainty formalism is a
constructor or a view:

    probability   additive v                    Capacity.from_probability
    belief fn     ∞-monotone v (Möbius m ≥ 0)   Capacity.from_mass / .mass()
    possibility   necessity measure N=1-Π(Ā)    Capacity.from_possibility
    credal set    lower envelope of vertices    Capacity.from_credal / .credal_vertices()
    capacity      any monotone v                Capacity(v, frame)

The upper set function is the conjugate v*(A) = 1 - v(Ā).
"""

import itertools
from typing import Optional

import torch

from .frame import Frame
from . import transforms as T


class Capacity:
    def __init__(self, lower: torch.Tensor, frame: Frame):
        if lower.shape[-1] != frame.size:
            raise ValueError("last dim must be 2**n")
        self.v = lower
        self.frame = frame

    # ---- constructors (views in) ------------------------------------------
    @classmethod
    def from_mass(cls, m: torch.Tensor, frame: Frame) -> "Capacity":
        return cls(T.belief(m, frame), frame)

    @classmethod
    def from_probability(cls, p: torch.Tensor, frame: Frame) -> "Capacity":
        return cls(p @ frame.membership(p).T, frame)

    @classmethod
    def from_possibility(cls, pi: torch.Tensor, frame: Frame) -> "Capacity":
        """π(ω) ∈ [0,1] (max=1 for normalised). N(A)=1-max_{ω∉A} π(ω)."""
        mem = frame.membership().to(pi.device)
        masked = torch.where(mem, pi.unsqueeze(-2), torch.zeros_like(pi).unsqueeze(-2))
        upper = masked.amax(-1)  # Π(A); Π(∅)=0 since masked row is all 0
        return cls(1 - upper.flip(-1), frame)

    @classmethod
    def from_credal(cls, vertices: torch.Tensor, frame: Frame) -> "Capacity":
        """Lower envelope of a finite set of distributions (..., K, n)."""
        probs = vertices @ frame.membership(vertices).T  # (..., K, N)
        return cls(probs.amin(-2), frame)

    @classmethod
    def vacuous(cls, frame: Frame, **kw) -> "Capacity":
        v = torch.zeros(frame.size, **kw); v[-1] = 1
        return cls(v, frame)

    # ---- duality ------------------------------------------------------------
    @property
    def lower(self) -> torch.Tensor:
        return self.v

    @property
    def upper(self) -> torch.Tensor:
        return 1 - self.v.flip(-1)

    def dual(self) -> "Capacity":
        return Capacity(self.upper, self.frame)

    def discount(self, alpha) -> "Capacity":
        """Shafer discounting on the capacity: v' = α·v on A ≠ Θ (α = reliability).
        Works for any capacity; equals mass-level :func:`torchbelief.discount` for beliefs."""
        a = torch.as_tensor(alpha, dtype=self.v.dtype, device=self.v.device)
        if a.dim() > 0:
            a = a.unsqueeze(-1)
        v = a * self.v
        return Capacity(torch.cat([v[..., :-1], torch.ones_like(v[..., -1:])], -1), self.frame)

    def width(self) -> torch.Tensor:
        """Imprecision upper−lower for each event (0 everywhere ⇔ probability)."""
        return self.upper - self.v

    # ---- views out -----------------------------------------------------------
    def mass(self) -> torch.Tensor:
        """Möbius inverse; signed in general, ≥ 0 iff ∞-monotone (belief)."""
        return T.subset_sum_inverse(self.v, self.frame)

    def commonality(self) -> torch.Tensor:
        return T.superset_sum(self.mass(), self.frame)

    def contour(self) -> torch.Tensor:
        """Upper probability of each singleton, shape (..., n)."""
        return self.upper[..., self.frame.singletons()]

    def pignistic(self) -> torch.Tensor:
        """Shapley value / pignistic probability (Σ_{A∋ω} m(A)/|A|, valid for any capacity)."""
        m = self.mass()
        card = self.frame.cardinality(m).clamp_min(1)
        return (m / card) @ self.frame.membership(m)

    def credal_vertices(self) -> torch.Tensor:
        """Extreme points of the core, (..., n!, n). Exact when v is 2-monotone (convex);
        otherwise these are the marginal vectors, a superset-of-vertices guide only."""
        F, n = self.frame, self.frame.n
        perms = torch.tensor(list(itertools.permutations(range(n))), device=self.v.device)  # [P,n]
        cum = (1 << perms).cumsum(-1)                                   # [P,n] masks of prefixes
        vc = self.v[..., cum]                                            # (...,P,n)
        delta = vc - torch.cat([torch.zeros_like(vc[..., :1]), vc[..., :-1]], -1)
        out = torch.zeros_like(delta)
        return out.scatter(-1, perms.expand(delta.shape), delta)

    # ---- structure checks -----------------------------------------------------
    def monotonicity_violation(self) -> torch.Tensor:
        """max over A⊆B of v(A)-v(B), clipped at 0 (0 ⇔ monotone)."""
        N = self.frame.size
        idx = torch.arange(N, device=self.v.device)
        sub = (idx[:, None] & idx[None, :]) == idx[:, None]            # [A,B]: A⊆B
        d = self.v.unsqueeze(-1) - self.v.unsqueeze(-2)                 # v(A)-v(B)
        return (d * sub).amax((-1, -2)).clamp_min(0)

    def convexity_violation(self) -> torch.Tensor:
        """max of v(A)+v(B)-v(A∪B)-v(A∩B), clipped at 0 (0 ⇔ 2-monotone)."""
        N = self.frame.size
        idx = torch.arange(N, device=self.v.device)
        i, u = idx[:, None] & idx[None, :], idx[:, None] | idx[None, :]
        d = self.v.unsqueeze(-1) + self.v.unsqueeze(-2) - self.v[..., u] - self.v[..., i]
        return d.amax((-1, -2)).clamp_min(0)

    def is_belief(self, tol: float = 1e-6) -> torch.Tensor:
        return (self.mass() >= -tol).all(-1)

    def is_probability(self, tol: float = 1e-6) -> torch.Tensor:
        return self.width().abs().amax(-1) <= tol

    def is_possibility(self, tol: float = 1e-6) -> torch.Tensor:
        """Π(A∪B)=max(Π(A),Π(B)) for all A,B."""
        up = self.upper
        N = self.frame.size
        idx = torch.arange(N, device=up.device)
        u = idx[:, None] | idx[None, :]
        gap = (up[..., u] - torch.maximum(up.unsqueeze(-1), up.unsqueeze(-2))).abs()
        return gap.amax((-1, -2)) <= tol

    # ---- integration -----------------------------------------------------------
    def choquet(self, f: torch.Tensor) -> torch.Tensor:
        """Choquet integral of f (..., n) w.r.t. the lower capacity = lower expectation
        for a belief/convex v. Differentiable in both f and v."""
        return choquet(f, self.v, self.frame)

    def choquet_upper(self, f: torch.Tensor) -> torch.Tensor:
        return choquet(f, self.upper, self.frame)

    def expectation_bounds(self, util: torch.Tensor):
        """Lower/upper expectation of each row of ``util`` (K, n): shapes (..., K).
        Capacity batch dims are leading, K acts/gambles are trailing."""
        u = util.to(self.v.dtype)
        v, up = self.v.unsqueeze(-2), self.upper.unsqueeze(-2)
        return choquet(u, v, self.frame), choquet(u, up, self.frame)

    def __repr__(self) -> str:
        return f"Capacity(shape={tuple(self.v.shape)}, frame={self.frame})"


def choquet(f: torch.Tensor, v: torch.Tensor, frame: Frame) -> torch.Tensor:
    """∫ f dv = Σ_i f_(i) [v(S_i) − v(S_{i−1})], S_i the i largest elements."""
    lead = torch.broadcast_shapes(f.shape[:-1], v.shape[:-1])
    f = f.expand(*lead, frame.n)
    v = v.expand(*lead, frame.size)
    order = f.argsort(-1, descending=True)
    fs = f.gather(-1, order)
    cum = (1 << order).cumsum(-1)
    vc = v.gather(-1, cum)
    w = vc - torch.cat([torch.zeros_like(vc[..., :1]), vc[..., :-1]], -1)
    return (fs * w).sum(-1)
