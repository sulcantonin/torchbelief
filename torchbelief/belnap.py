"""Belnap–Dunn four-valued (paraconsistent) view of evidence.

A bilattice value is a tensor (..., 2) = (t, f): degree to which the event is
*supported* and *refuted*. Unlike probability, t + f may exceed 1 (contradiction,
"both") or fall short of it (ignorance, "neither"), so conflicting sources are
represented rather than averaged away.

From a capacity, an event A has t = v(A) and f = v(Ā). For one coherent source
t + f ≤ 1; pooling several sources with the knowledge join ⊕ (max of both) exposes
contradiction as t + f > 1.
"""

from typing import Sequence, Union

import torch

from .capacity import Capacity

LABELS = ("true", "false", "both", "neither")


def evidence(cap: Capacity, event: int) -> torch.Tensor:
    """(t, f) = (v(A), v(Ā)) for a subset bitmask ``event``: shape (..., 2)."""
    return torch.stack([cap.lower[..., event], cap.lower[..., cap.frame.theta ^ event]], -1)


def singletons(cap: Capacity) -> torch.Tensor:
    """(t, f) for every singleton: (..., n, 2)."""
    return torch.stack([evidence(cap, s) for s in cap.frame.singletons()], -2)


def pool(caps: Sequence[Capacity], event: int = None) -> torch.Tensor:
    """Knowledge join of the sources' (t, f): what *someone* supports / refutes.
    ``event=None`` pools every singleton, giving (..., n, 2)."""
    vals = [singletons(c) if event is None else evidence(c, event) for c in caps]
    return torch.stack(vals, 0).amax(0)


# ---- bilattice operations ----------------------------------------------------------

def negate(x):            return x.flip(-1)
def truth_and(x, y):      return torch.stack([torch.minimum(x[..., 0], y[..., 0]), torch.maximum(x[..., 1], y[..., 1])], -1)
def truth_or(x, y):       return torch.stack([torch.maximum(x[..., 0], y[..., 0]), torch.minimum(x[..., 1], y[..., 1])], -1)
def knowledge_join(x, y): return torch.maximum(x, y)   # ⊕: accept everything anyone says
def knowledge_meet(x, y): return torch.minimum(x, y)   # ⊗: keep only what everyone says


# ---- reading off the four values ----------------------------------------------------

def degrees(x: torch.Tensor) -> torch.Tensor:
    """Soft membership in (true, false, both, neither), (..., 4), sums to 1.
    Dunn's "told" semantics with independent t, f: T = t(1−f), F = f(1−t), B = tf, N = (1−t)(1−f)."""
    t, f = x[..., 0], x[..., 1]
    return torch.stack([t * (1 - f), f * (1 - t), t * f, (1 - t) * (1 - f)], -1)


def contradiction(x: torch.Tensor) -> torch.Tensor:
    """(t + f − 1)⁺: excess support; > 0 only when sources (or an incoherent capacity) clash."""
    return (x.sum(-1) - 1).clamp_min(0)


def ignorance(x: torch.Tensor) -> torch.Tensor:
    """(1 − t − f)⁺: missing support either way (the width of the [lower, upper] interval)."""
    return (1 - x.sum(-1)).clamp_min(0)


def label(x: torch.Tensor, threshold: float = 0.5):
    """Crisp Belnap value index into ``LABELS``: told-true iff t ≥ threshold, told-false iff f ≥ threshold."""
    t, f = x[..., 0] >= threshold, x[..., 1] >= threshold
    return torch.where(t & f, 2, torch.where(t, 0, torch.where(f, 1, 3)))
