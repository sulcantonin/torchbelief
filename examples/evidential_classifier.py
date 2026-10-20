"""Two evidential classifiers on 2-D blobs, probed in- and out-of-distribution.

EvidentialHead (Dirichlet / subjective logic on top of a ReLU MLP) is accurate but,
like softmax, extrapolates evidence linearly: it stays confident far from the data.
PrototypeHead (Denœux's ENN) bases evidence on distance to learned prototypes, so
m(Θ) → 1 away from the training data and decide_or_abstain() abstains.
"""

import torch
from torch import nn

import torchbelief as tb
from torchbelief import choice

torch.manual_seed(0)
F = tb.Frame(3, names=["a", "b", "c"])
centres = torch.tensor([[0., 3.], [-3., -2.], [3., -2.]])
y = torch.randint(0, 3, (600,))
x = centres[y] + torch.randn(600, 2)

mlp = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(),
                    tb.EvidentialHead(32, F))
enn = tb.PrototypeHead(2, F, init=x[torch.randperm(600)[:12]])


def fit(model, steps=300):
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(steps):
        m = model(x)
        loss = tb.belief_nll(m, y, F) + 0.05 * tb.nonspecificity_penalty(m, F)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (tb.pignistic(model(x), F).argmax(-1) == y).float().mean().item()


probe = torch.tensor([[0., 3.], [0., 0.], [3., -2.], [12., 12.], [-15., 5.]])
for name, model in (("EvidentialHead + MLP", mlp), ("PrototypeHead (ENN)", enn)):
    acc = fit(model)
    print(f"\n{name}: train accuracy {acc:.3f}")
    with torch.no_grad():
        m = model(probe)
        act = choice.decide_or_abstain(tb.Capacity.from_mass(m, F), torch.eye(3))
    print("  point         m(Θ)   BetP                  decision")
    for p, mt, bp, a in zip(probe.tolist(), tb.vacuity(m, F), tb.pignistic(m, F), act):
        print(f"  {str(p):13s} {mt:.3f}  {bp.numpy().round(3)!s:20s}  {F.names[a] if a >= 0 else 'ABSTAIN'}")
