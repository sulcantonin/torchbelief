"""Fixing a classifier after deployment: calibrate it, or retarget its decisions.

An EkNN classifier is trained on clean blobs. In deployment, 60% of trucks look like
cars (say, at night), so it answers "car" with confidence for them. A small labelled
calibration set is then used in two ways:

1. ContextualCorrection learns per-class reliabilities β_k by minimising Denœux's E_pl
   (closed form, then gradient descent). β_truck drops, so Pl(truck) rises wherever the
   classifier says "car": better calibrated plausibilities, but "car" is still the more
   likely answer there, so the set predictions do not change.
2. A MassHead over all subsets, fed the classifier's mass, is trained with
   expected_utility_loss for u80. Where 3 in 8 "cars" are trucks, answering {car, truck}
   is worth more than "car" under u80, and the head learns to say so.

Set predictions are read off with max_mass_set, the rule expected_utility_loss optimises.
"""

import torch

import torchbelief as tb
from torchbelief import metrics

torch.manual_seed(0)
F = tb.Frame(3, names=["car", "truck", "bike"])
centres = torch.tensor([[0., 3.], [-3., -2.], [3., -2.]])


def sample(n, shift=0.0):
    y = torch.randint(0, 3, (n,))
    c = centres[y].clone()
    night = (y == 1) & (torch.rand(n) < shift)
    c[night] = centres[0]                                       # these trucks look like cars
    return c + torch.randn(n, 2), y


def train(model, loss_fn, inputs, steps):
    opt = torch.optim.Adam(model.parameters(), lr=5e-2)
    for _ in range(steps):
        loss = loss_fn(model(inputs))
        opt.zero_grad(); loss.backward(); opt.step()


x, y = sample(600)
eknn = tb.EkNNHead(F, x, y, k=7)
opt = torch.optim.Adam(eknn.parameters(), lr=5e-2)
for _ in range(100):
    loss = tb.pl_loss(eknn(x, self_index=torch.arange(600)), y, F)
    opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():
    x_cal, y_cal = sample(300, shift=0.6)
    x_test, y_test = sample(3000, shift=0.6)
    m_cal, m_test = eknn(x_cal), eknn(x_test)

cd = tb.ContextualCorrection(F, "discount", init=tb.fit_contextual(m_cal, y_cal, F))
train(cd, lambda m: tb.pl_loss(m, y_cal, F), m_cal, 200)
print("learned reliability β:", dict(zip(F.names, cd.beta[0].detach().numpy().round(3))))

u80_head = tb.MassHead(F.size, F, focal="all")
train(u80_head, lambda m: tb.expected_utility_loss(m, y_cal, F, a=2.2, b=1.2), m_cal, 300)

trucks = y_test == 1
print(f"\n{'':14s} {'accuracy':>9s} {'E_pl':>6s} {'u65':>6s} {'u80':>6s} {'u80 trucks':>11s} {'set size':>9s}")
with torch.no_grad():
    for name, m in (("raw", m_test), ("discounted", cd(m_test)), ("u80 head", u80_head(m_test))):
        acc = (tb.pignistic(m, F).argmax(-1) == y_test).float().mean()
        sets = tb.max_mass_set(m, F)
        print(f"{name:14s} {acc:9.3f} {tb.pl_loss(m, y_test, F):6.3f} {metrics.u65(sets, y_test):6.3f} "
              f"{metrics.u80(sets, y_test):6.3f} {metrics.u80(sets[trucks], y_test[trucks]):11.3f} "
              f"{sets.sum(-1).float().mean():9.2f}")
