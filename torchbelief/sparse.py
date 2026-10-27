"""Sparse mass functions: a list of focal sets (int64 bitmasks) with weights.

Lifts the dense 2^n limit to frames of up to 62 elements. Combination costs
O(k1·k2) instead of O(n·2^n) / O(4^n), and :meth:`SparseMass.summarize` keeps the
number of focal sets bounded with an *outer* approximation (Lowrance et al.
summarisation): Bel can only drop and Pl only rise, so credal intervals stay valid.

Unbatched (one mass function per object); differentiable in the weights.
"""

from typing import Optional, Sequence, Tuple, Union

import torch

from .frame import Frame

MAX_N = 62


class SparseMass:
    def __init__(self, focal: torch.Tensor, weight: torch.Tensor, n: int):
        if not 1 <= n <= MAX_N:
            raise ValueError(f"n must be in 1..{MAX_N}")
        if focal.shape != weight.shape or focal.dim() != 1:
            raise ValueError("focal and weight must be 1-D tensors of equal length")
        self.focal, self.weight, self.n = focal.long(), weight, n

    # ---- constructors ------------------------------------------------------------------
    @classmethod
    def vacuous(cls, n: int, dtype=torch.float64) -> "SparseMass":
        return cls(torch.tensor([(1 << n) - 1]), torch.ones(1, dtype=dtype), n)

    @classmethod
    def categorical(cls, n: int, subset: int, dtype=torch.float64) -> "SparseMass":
        return cls(torch.tensor([subset]), torch.ones(1, dtype=dtype), n)

    @classmethod
    def simple_support(cls, n: int, subset: int, weight, dtype=torch.float64) -> "SparseMass":
        w = torch.as_tensor(weight, dtype=dtype)
        return cls(torch.tensor([subset, (1 << n) - 1]), torch.stack([w, 1 - w]), n).coalesce()

    @classmethod
    def from_dense(cls, m: torch.Tensor, frame: Frame, tol: float = 0.0) -> "SparseMass":
        keep = (m.abs() > tol).nonzero().flatten()
        return cls(keep, m[keep], frame.n)

    def to_dense(self, frame: Optional[Frame] = None) -> torch.Tensor:
        frame = frame or Frame(self.n)
        out = torch.zeros(frame.size, dtype=self.weight.dtype, device=self.weight.device)
        return out.scatter_add(0, self.focal, self.weight)

    # ---- housekeeping ---------------------------------------------------------------------
    def coalesce(self) -> "SparseMass":
        """Merge duplicate focal sets (summing weights) and drop exact zeros."""
        uniq, inv = torch.unique(self.focal, return_inverse=True)
        w = torch.zeros(uniq.numel(), dtype=self.weight.dtype, device=self.weight.device)
        w = w.scatter_add(0, inv, self.weight)
        keep = w != 0
        return SparseMass(uniq[keep], w[keep], self.n)

    def __len__(self) -> int:
        return self.focal.numel()

    def bits(self) -> torch.Tensor:
        """Membership matrix [k, n] (bool)."""
        return ((self.focal.unsqueeze(-1) >> torch.arange(self.n, device=self.focal.device)) & 1).bool()

    def cardinality(self) -> torch.Tensor:
        return self.bits().sum(-1)

    def summarize(self, k: int) -> "SparseMass":
        """Keep the k−1 heaviest non-empty focal sets (∅ is kept aside) and move the rest onto the
        union of the dropped ones. Outer approximation: Bel ↓, Pl ↑ for every event."""
        empty = self.focal == 0
        f, w = self.focal[~empty], self.weight[~empty]
        if f.numel() <= k:
            return self
        order = w.argsort(descending=True)
        top, rest = order[: k - 1], order[k - 1:]
        bits = ((f[rest].unsqueeze(-1) >> torch.arange(self.n)) & 1).any(0)
        union = (bits.long() << torch.arange(self.n)).sum().reshape(1)
        out = SparseMass(torch.cat([self.focal[empty], f[top], union]),
                         torch.cat([self.weight[empty], w[top], w[rest].sum().reshape(1)]), self.n)
        return out.coalesce()

    # ---- combination -----------------------------------------------------------------------
    def _pairs(self, other: "SparseMass"):
        a, b = self.focal.unsqueeze(-1), other.focal.unsqueeze(-2)
        w = (self.weight.unsqueeze(-1) * other.weight.unsqueeze(-2)).flatten()
        return (a & b).flatten(), (a | b).flatten(), w

    def conjunctive(self, other: "SparseMass") -> "SparseMass":
        """Unnormalised (TBM) conjunctive rule; conflict stays on ∅ (focal 0)."""
        inter, _, w = self._pairs(other)
        return SparseMass(inter, w, self.n).coalesce()

    def disjunctive(self, other: "SparseMass") -> "SparseMass":
        _, union, w = self._pairs(other)
        return SparseMass(union, w, self.n).coalesce()

    def conflict(self) -> torch.Tensor:
        return self.weight[self.focal == 0].sum()

    def dempster(self, other: "SparseMass") -> Tuple["SparseMass", torch.Tensor]:
        m = self.conjunctive(other)
        k = m.conflict()
        keep = m.focal != 0
        w = m.weight[keep]
        return SparseMass(m.focal[keep], w / w.sum().clamp_min(1e-12), self.n), k

    def yager(self, other: "SparseMass") -> Tuple["SparseMass", torch.Tensor]:
        m = self.conjunctive(other)
        k = m.conflict()
        keep = m.focal != 0
        theta = torch.tensor([(1 << self.n) - 1])
        return SparseMass(torch.cat([m.focal[keep], theta]),
                          torch.cat([m.weight[keep], k.reshape(1)]), self.n).coalesce(), k

    def dubois_prade(self, other: "SparseMass") -> Tuple["SparseMass", torch.Tensor]:
        inter, union, w = self._pairs(other)
        empty = inter == 0
        return SparseMass(torch.where(empty, union, inter), w, self.n).coalesce(), (w * empty).sum()

    def discount(self, alpha) -> "SparseMass":
        a = torch.as_tensor(alpha, dtype=self.weight.dtype)
        theta = torch.tensor([(1 << self.n) - 1])
        return SparseMass(torch.cat([self.focal, theta]),
                          torch.cat([a * self.weight, (1 - a).reshape(1)]), self.n).coalesce()

    # ---- queries ------------------------------------------------------------------------------
    def belief(self, event: Union[int, torch.Tensor]) -> torch.Tensor:
        """Bel of one event or a tensor of events (bitmasks)."""
        e = torch.as_tensor(event).unsqueeze(-1)
        inside = ((self.focal & ~e) == 0) & (self.focal != 0)
        return (self.weight * inside).sum(-1)

    def plausibility(self, event: Union[int, torch.Tensor]) -> torch.Tensor:
        e = torch.as_tensor(event).unsqueeze(-1)
        return (self.weight * ((self.focal & e) != 0)).sum(-1)

    def pignistic(self) -> torch.Tensor:
        """BetP over the n elements."""
        bits = self.bits().to(self.weight.dtype)
        nz = self.focal != 0
        card = bits.sum(-1).clamp_min(1)
        p = ((self.weight * nz / card).unsqueeze(-1) * bits).sum(0)
        return p / (self.weight * nz).sum().clamp_min(1e-12)

    def contour(self) -> torch.Tensor:
        """Pl({ω}) for every element (n,)."""
        return (self.weight.unsqueeze(-1) * self.bits()).sum(0)

    def expectation_bounds(self, util: torch.Tensor):
        """Lower/upper expected utility of each act (acts, n) → two (acts,) tensors."""
        bits = self.bits() & (self.focal != 0).unsqueeze(-1)
        u = util.to(self.weight.dtype).unsqueeze(-2)                # [acts, 1, n]
        big = torch.finfo(u.dtype).max
        mn = torch.where(bits, u, torch.full_like(u, big).expand(-1, bits.shape[0], -1)).amin(-1)
        mx = torch.where(bits, u, torch.full_like(u, -big).expand(-1, bits.shape[0], -1)).amax(-1)
        nz = self.focal != 0
        mn, mx = torch.where(nz, mn, 0.0), torch.where(nz, mx, 0.0)
        return mn @ self.weight, mx @ self.weight

    def __repr__(self) -> str:
        return f"SparseMass(n={self.n}, focal_sets={len(self)})"
