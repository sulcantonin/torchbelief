"""Neural layers that output valid mass functions, and basic losses."""

from typing import Optional, Sequence

import torch
from torch import nn

from .frame import Frame
from .decision import pignistic
from .mass import from_dirichlet
from . import transforms as T


class MassHead(nn.Module):
    """Linear → softmax over a fixed set of focal sets → dense mass (..., 2**n).

    ``focal=None`` uses singletons + Θ (as in MP-ENN); ``"all"`` uses every
    non-empty subset (2**n − 1 outputs); or pass an explicit list of subsets.
    """

    def __init__(self, in_features: int, frame: Frame, focal=None):
        super().__init__()
        self.frame = frame
        if focal is None:
            focal = frame.singletons() + ([frame.theta] if frame.n > 1 else [])
        elif focal == "all":
            focal = list(range(1, frame.size))
        self.register_buffer("focal", torch.tensor(sorted(set(focal)), dtype=torch.long))
        self.fc = nn.Linear(in_features, len(self.focal))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.fc(x).softmax(-1)
        out = torch.zeros(*p.shape[:-1], self.frame.size, dtype=p.dtype, device=p.device)
        return out.index_copy(-1, self.focal, p)


class EvidentialHead(nn.Module):
    """Linear → softplus evidence e ≥ 0 → Dirichlet α = e + 1 → mass on singletons + Θ.

    The subjective-logic / evidential-deep-learning head (Sensoy et al. 2018): little total
    evidence means a large m(Θ), i.e. the network can say "I don't know".
    ``forward`` returns the dense mass; ``alpha(x)`` the Dirichlet parameters.
    """

    def __init__(self, in_features: int, frame: Frame):
        super().__init__()
        self.frame = frame
        self.fc = nn.Linear(in_features, frame.n)

    def alpha(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.softplus(self.fc(x)) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return from_dirichlet(self.frame, self.alpha(x))


class PrototypeHead(nn.Module):
    """Denœux's evidential neural network (ENN, 2000): distance-based evidence.

    Prototype p_i supports class memberships u_i = softmax(β_i) with strength
    s_i = α_i·exp(−γ_i²‖x − p_i‖²), giving m_i({ω_k}) = s_i·u_ik, m_i(Θ) = 1 − s_i.
    All prototypes are fused with Dempster's rule in closed form (commonalities multiply).
    Far from every prototype s_i → 0, so the output tends to the vacuous mass: unlike a
    softmax or Dirichlet head, ignorance grows with distance from the training data.
    """

    def __init__(self, in_features: int, frame: Frame, n_prototypes: int = 10,
                 init: Optional[torch.Tensor] = None):
        super().__init__()
        self.frame = frame
        protos = init.clone() if init is not None else torch.randn(n_prototypes, in_features)
        self.prototypes = nn.Parameter(protos.float())
        k = self.prototypes.shape[0]
        self.beta = nn.Parameter(torch.randn(k, frame.n) * 0.1)
        self.alpha_logit = nn.Parameter(torch.zeros(k))
        self.gamma = nn.Parameter(torch.ones(k))
        focal = frame.singletons() + [frame.theta]
        self.register_buffer("focal", torch.tensor(focal, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d2 = torch.cdist(x, self.prototypes.to(x.dtype)).pow(2)          # (..., K)
        s = torch.sigmoid(self.alpha_logit) * torch.exp(-self.gamma.pow(2) * d2)
        u = self.beta.softmax(-1)                                          # (K, n)
        # log commonalities: q_i({ω_k}) = 1 − s_i + s_i·u_ik, q_i(Θ) = 1 − s_i
        log_q_theta = torch.log1p(-s.clamp(max=1 - 1e-6)).sum(-1, keepdim=True)
        log_q = torch.log1p(-s.unsqueeze(-1) * (1 - u)).sum(-2)           # (..., n)
        q_theta = log_q_theta.exp()
        m_single = log_q.exp() - q_theta                                    # unnormalised
        m = torch.cat([m_single, q_theta], -1)
        m = m / m.sum(-1, keepdim=True)                                     # Dempster normalisation
        out = torch.zeros(*m.shape[:-1], self.frame.size, dtype=m.dtype, device=m.device)
        return out.index_copy(-1, self.focal, m)


def _singleton(f: torch.Tensor, target: torch.Tensor, frame: Frame) -> torch.Tensor:
    return f[..., frame.singletons()].gather(-1, target.unsqueeze(-1)).squeeze(-1)


def belief_nll(m: torch.Tensor, target: torch.Tensor, frame: Frame, eps: float = 1e-8) -> torch.Tensor:
    """−log Bel({y}): cautious likelihood, punishes any mass not committed to y."""
    return -_singleton(T.belief(m, frame), target, frame).clamp_min(eps).log().mean()


def plausibility_nll(m: torch.Tensor, target: torch.Tensor, frame: Frame, eps: float = 1e-8) -> torch.Tensor:
    """−log Pl({y}): optimistic likelihood, only punishes mass that excludes y
    (a vacuous output costs nothing — combine with a regulariser)."""
    return -_singleton(T.plausibility(m, frame), target, frame).clamp_min(eps).log().mean()


def hurwicz_nll(m: torch.Tensor, target: torch.Tensor, frame: Frame, lam: float = 0.5,
                eps: float = 1e-8) -> torch.Tensor:
    """−log((1−λ)·Bel({y}) + λ·Pl({y})): interpolates belief_nll (λ=0) and plausibility_nll (λ=1)."""
    b = _singleton(T.belief(m, frame), target, frame)
    p = _singleton(T.plausibility(m, frame), target, frame)
    return -((1 - lam) * b + lam * p).clamp_min(eps).log().mean()


def nonspecificity_penalty(m: torch.Tensor, frame: Frame) -> torch.Tensor:
    """Mean Σ m(A)·log2|A|: add with a small weight to keep heads from staying vacuous."""
    card = frame.cardinality(m).clamp_min(1)
    return (m * card.log2()).sum(-1).mean()


def pignistic_nll(m: torch.Tensor, target: torch.Tensor, frame: Frame, eps: float = 1e-8) -> torch.Tensor:
    """Cross-entropy of the pignistic probability against class indices."""
    p = pignistic(m, frame)
    return -p.gather(-1, target.unsqueeze(-1)).clamp_min(eps).log().mean()
