"""Random intervals: belief functions on the real line, and their p-boxes.

A ``RandomInterval`` is a finite random set whose focal elements are closed
intervals [lo_k, hi_k] with masses w_k (shapes (..., K)). Its lower/upper CDFs form a
p-box. Everything is batched and differentiable in the masses; endpoints get
gradients through ``mean_bounds`` and through the ``temperature``-smoothed queries.
"""

from typing import Callable, Optional, Tuple

import torch

from .frame import Frame


def _step(x: torch.Tensor, temperature: Optional[float]) -> torch.Tensor:
    """1[x ≥ 0], or a sigmoid relaxation of it."""
    return (x >= 0).to(x.dtype) if temperature is None else torch.sigmoid(x / temperature)


class RandomInterval:
    def __init__(self, lo: torch.Tensor, hi: torch.Tensor, mass: Optional[torch.Tensor] = None):
        if mass is None:
            mass = torch.full_like(lo, 1.0 / lo.shape[-1])
        if (hi < lo).any():
            raise ValueError("hi must be ≥ lo")
        self.lo, self.hi, self.mass = lo, hi, mass

    # ---- constructors -----------------------------------------------------------------
    @classmethod
    def precise(cls, x: torch.Tensor, mass: Optional[torch.Tensor] = None) -> "RandomInterval":
        """Degenerate intervals = an ordinary discrete distribution."""
        return cls(x, x, mass)

    @classmethod
    def from_pbox(cls, cdf_lo: torch.Tensor, cdf_hi: torch.Tensor, grid: torch.Tensor,
                  levels: int = 64) -> "RandomInterval":
        """Discretise a p-box given by lower/upper CDFs on ``grid`` (G,) into ``levels``
        equal-mass focal intervals [F_hi⁻¹(u), F_lo⁻¹(u)] (the standard outer approximation)."""
        u = (torch.arange(levels, dtype=grid.dtype, device=grid.device) + 0.5) / levels
        lo = grid[torch.searchsorted(cdf_hi.contiguous(), u).clamp(max=grid.numel() - 1)]
        hi = grid[torch.searchsorted(cdf_lo.contiguous(), u).clamp(max=grid.numel() - 1)]
        return cls(lo, torch.maximum(lo, hi))

    # ---- queries ------------------------------------------------------------------------
    def cdf_bounds(self, x: torch.Tensor, temperature: Optional[float] = None):
        """p-box at points x (..., G): (F_lower, F_upper) = (Bel(X ≤ x), Pl(X ≤ x))."""
        xs = x.unsqueeze(-1)                                        # (..., G, 1)
        lo, hi, w = (t.unsqueeze(-2) for t in (self.lo, self.hi, self.mass))
        return ((w * _step(xs - hi, temperature)).sum(-1),
                (w * _step(xs - lo, temperature)).sum(-1))

    def prob_bounds(self, a, b, temperature: Optional[float] = None):
        """(Bel, Pl) of the event [a, b]."""
        w = self.mass
        inside = _step(self.lo - a, temperature) * _step(b - self.hi, temperature)
        touch = _step(self.hi - a, temperature) * _step(b - self.lo, temperature)
        return (w * inside).sum(-1), (w * touch).sum(-1)

    def mean_bounds(self):
        """Lower/upper expectation of X."""
        return (self.mass * self.lo).sum(-1), (self.mass * self.hi).sum(-1)

    def expectation_bounds(self, f: Callable[[torch.Tensor], torch.Tensor], resolution: int = 65):
        """Lower/upper expectation of f(X): Σ_k w_k·min/max of f on [lo_k, hi_k].
        Exact for monotone f; otherwise f is searched on ``resolution`` points per interval."""
        t = torch.linspace(0, 1, resolution, dtype=self.lo.dtype, device=self.lo.device)
        pts = self.lo.unsqueeze(-1) + (self.hi - self.lo).unsqueeze(-1) * t   # (..., K, R)
        fv = f(pts)
        return (self.mass * fv.amin(-1)).sum(-1), (self.mass * fv.amax(-1)).sum(-1)

    def width(self) -> torch.Tensor:
        """Expected interval length = mean imprecision (0 ⇔ precise)."""
        return (self.mass * (self.hi - self.lo)).sum(-1)

    # ---- combination ---------------------------------------------------------------------
    def conjunctive(self, other: "RandomInterval", normalise: bool = True
                    ) -> Tuple["RandomInterval", torch.Tensor]:
        """Dempster combination of independent random intervals: focal sets are the
        pairwise intersections; disjoint pairs are conflict κ. Returns (result, κ)."""
        lo = torch.maximum(self.lo.unsqueeze(-1), other.lo.unsqueeze(-2)).flatten(-2)
        hi = torch.minimum(self.hi.unsqueeze(-1), other.hi.unsqueeze(-2)).flatten(-2)
        w = (self.mass.unsqueeze(-1) * other.mass.unsqueeze(-2)).flatten(-2)
        empty = hi < lo
        k = (w * empty).sum(-1)
        w = w * ~empty
        if normalise:
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)   # = 1 − κ, without cancellation
        hi = torch.where(empty, lo, hi)
        return RandomInterval(lo, hi, w), k

    def mixture(self, other: "RandomInterval", weight: float = 0.5) -> "RandomInterval":
        """Weighted average of the two random sets (never conflicts)."""
        return RandomInterval(torch.cat([self.lo, other.lo], -1), torch.cat([self.hi, other.hi], -1),
                              torch.cat([weight * self.mass, (1 - weight) * other.mass], -1))

    # ---- bridge to finite frames ----------------------------------------------------------
    def discretize(self, edges: torch.Tensor) -> Tuple[torch.Tensor, Frame]:
        """Dense mass on the frame of bins [e_j, e_{j+1}) (≤ 16 bins). Each interval maps to
        the set of bins it touches (outer approximation); mass outside the range goes to ∅."""
        B = edges.numel() - 1
        frame = Frame(B)
        left, right = edges[:-1], edges[1:]
        touch = (self.lo.unsqueeze(-1) < right) & (self.hi.unsqueeze(-1) >= left)   # (..., K, B)
        touch[..., -1] |= (self.lo == edges[-1]) & (self.hi >= edges[-1])           # closed last bin
        masks = (touch.long() << torch.arange(B, device=edges.device)).sum(-1)       # (..., K)
        m = torch.zeros(*self.mass.shape[:-1], frame.size, dtype=self.mass.dtype, device=self.mass.device)
        return m.scatter_add(-1, masks, self.mass), frame

    def __repr__(self) -> str:
        return f"RandomInterval(K={self.lo.shape[-1]}, batch={tuple(self.lo.shape[:-1])})"
