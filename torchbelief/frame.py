"""Frame of discernment and subset indexing.

A subset A of a frame with n elements is the integer whose bit i is set iff
element i is in A. Masses over the power set are dense tensors of shape
``(..., 2**n)``; index 0 is the empty set and index ``2**n - 1`` is Θ.
"""

from typing import List, Optional, Sequence

import torch


class Frame:
    def __init__(self, n: int, names: Optional[Sequence[str]] = None):
        if not 1 <= n <= 16:
            raise ValueError("frame size must be in 1..16 (dense 2**n representation)")
        if names is not None and len(names) != n:
            raise ValueError("len(names) must equal n")
        self.n = n
        self.size = 1 << n
        self.names = list(names) if names is not None else [str(i) for i in range(n)]
        idx = torch.arange(self.size)
        self._member = ((idx[:, None] >> torch.arange(n)[None, :]) & 1).bool()  # [N, n]
        self._card = self._member.sum(1)  # [N]

    @property
    def empty(self) -> int:
        return 0

    @property
    def theta(self) -> int:
        return self.size - 1

    def subset(self, elements: Sequence) -> int:
        """Index of the subset given element positions or names."""
        out = 0
        for e in elements:
            out |= 1 << (self.names.index(e) if isinstance(e, str) else int(e))
        return out

    def elements(self, subset: int) -> List[str]:
        return [self.names[i] for i in range(self.n) if (subset >> i) & 1]

    def membership(self, like: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Bool/float matrix [2**n, n]: membership[A, ω] = ω ∈ A."""
        return self._member if like is None else self._member.to(like.device, like.dtype)

    def cardinality(self, like: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._card if like is None else self._card.to(like.device, like.dtype)

    def singletons(self) -> List[int]:
        return [1 << i for i in range(self.n)]

    def __repr__(self) -> str:
        return f"Frame({self.names})"
