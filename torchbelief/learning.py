"""Learning correction degrees and simulating imprecise labels.

``ContextualCorrection`` wraps the contextual corrections of :mod:`torchbelief.corrections`
as an ``nn.Module`` with β = sigmoid(logit), optionally one β vector per group (e.g. per
sensor or per operating condition). ``fit_contextual`` gives the closed-form least-squares
β of Pichon et al. (2016) for the default contexts, a good initialisation for it.
"""

from typing import Optional, Union

import torch
from torch import nn

from .frame import Frame
from . import corrections as C
from . import transforms as T

_CORRECTIONS = {"discount": C.contextual_discount, "reinforce": C.contextual_reinforce,
                "negate": C.contextual_negate}


class ContextualCorrection(nn.Module):
    """Learnable contextual discounting, reinforcement or negating of a source.

    ``kind`` picks the correction, ``contexts`` its contexts (default: one per class, see
    :mod:`torchbelief.corrections`). With ``groups > 1`` pass ``group`` (...) to forward to
    pick a β row per sample. ``init`` is a scalar or a (groups, L) tensor of degrees.
    """

    def __init__(self, frame: Frame, kind: str = "discount", contexts=None, groups: int = 1,
                 init: Union[float, torch.Tensor] = 0.9):
        super().__init__()
        if kind not in _CORRECTIONS:
            raise ValueError(f"kind must be one of {sorted(_CORRECTIONS)}")
        self.frame, self.kind = frame, kind
        self.contexts = None if contexts is None else list(contexts)
        L = frame.n if contexts is None else len(self.contexts)
        beta = torch.as_tensor(init, dtype=torch.get_default_dtype()).expand(groups, L)
        self.logit = nn.Parameter(torch.logit(beta.clamp(1e-4, 1 - 1e-4)).clone())

    @property
    def beta(self) -> torch.Tensor:
        """Correction degrees (groups, L)."""
        return torch.sigmoid(self.logit)

    def forward(self, m: torch.Tensor, group: Optional[torch.Tensor] = None) -> torch.Tensor:
        beta = self.beta[0] if group is None else self.beta[group]
        return _CORRECTIONS[self.kind](m, beta.to(m.dtype), self.frame, self.contexts)


def fit_contextual(m: torch.Tensor, target: torch.Tensor, frame: Frame, kind: str = "discount",
                   group: Optional[torch.Tensor] = None, groups: int = 1, eps: float = 1e-15
                   ) -> torch.Tensor:
    """Closed-form β minimising Denœux's E_pl (:func:`torchbelief.pl_loss`) for the default
    contexts, where the corrected contour is linear in β (Pichon et al. 2016, Props 12, 14, 16):

    discount  pl' = 1 − β(1 − pl)   reinforce  pl' = β·pl   negate  pl' = ½ + (pl − ½)(2β − 1)

    m (B, 2**n); target class indices (B,) or soft-label masses (B, 2**n). Returns β (groups, n),
    clipped to [0, 1]; a class with no information keeps β = 1. Not differentiable."""
    pl = T.plausibility(m, frame)[..., frame.singletons()].detach()
    if target.is_floating_point():
        delta = T.plausibility(target, frame)[..., frame.singletons()]
    else:
        delta = nn.functional.one_hot(target, frame.n).to(pl.dtype)
    x, y = {"discount": (1 - pl, 1 - delta),             # 1 − pl' = β(1 − pl)
            "reinforce": (pl, delta),                     # pl' = β·pl
            "negate": (2 * pl - 1, pl + delta - 1)}[kind]  # pl' + pl − 1 = β(2pl − 1)
    g = torch.zeros(len(pl), dtype=torch.long, device=pl.device) if group is None else group
    num = pl.new_zeros(groups, frame.n).index_add(0, g, x * y)
    den = pl.new_zeros(groups, frame.n).index_add(0, g, x * x)
    ok = den.abs() > eps
    return torch.where(ok, num / den.where(ok, torch.ones_like(den)), torch.ones_like(den)).clamp(0, 1)


def hard_to_soft_labels(target: torch.Tensor, frame: Frame, mu: float = 0.5, var: float = 0.04
                        ) -> torch.Tensor:
    """Simulate imprecise, partly wrong labels (Mutmainah 2021, Algorithm 2), as masses (B, 2**n).

    For each label draw p ~ Beta(mean mu, variance var) and, with probability p, replace the
    label by m({k}) = 1 − p, m(Θ) = p for a uniformly random class k (which may be wrong);
    otherwise keep it categorical. Uses torch's global random state."""
    common = mu * (1 - mu) / var - 1
    if common <= 0:
        raise ValueError(f"var must be < mu·(1−mu) = {mu * (1 - mu)}")
    B = target.shape[0]
    p = torch.distributions.Beta(mu * common, (1 - mu) * common).sample((B,))
    soft = torch.bernoulli(p).bool()
    k = torch.randint(frame.n, (B,), device=target.device)
    rows = torch.arange(B, device=target.device)
    m = torch.zeros(B, frame.size, dtype=p.dtype, device=target.device)
    m[rows, torch.where(soft, 1 << k, 1 << target)] = torch.where(soft, 1 - p, torch.ones_like(p))
    m[:, frame.theta] += torch.where(soft, p, torch.zeros_like(p))
    return m
