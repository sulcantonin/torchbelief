import pytest
import torch
from torch import nn

import torchbelief as tb

F = tb.Frame(4)
torch.manual_seed(0)


def test_dempster_float32_high_conflict():
    """Normalising by the non-empty mass, not 1 − κ, keeps float32 accurate as κ → 1."""
    B, eps = 128, 1e-6
    a = torch.randint(F.n, (B,))
    b = (a + 1 + torch.randint(F.n - 1, (B,))) % F.n

    def nearly_categorical(idx):
        m = tb.random_mass(F, (B,)) * eps
        m[torch.arange(B), 1 << idx] += 1 - eps
        return m

    m1, m2 = nearly_categorical(a), nearly_categorical(b)
    ref = tb.dempster(m1, m2, F)[0]
    x, y = m1.float().requires_grad_(), m2.float().requires_grad_()
    out = tb.dempster(x, y, F)[0]
    assert ((out.double() - ref).abs().amax(-1) / ref.amax(-1)).max() < 1e-5
    w = torch.randn(B, F.size, dtype=torch.float64)
    (out * w.float()).sum().backward()
    xd = m1.clone().requires_grad_()
    (tb.dempster(xd, m2, F)[0] * w).sum().backward()
    assert ((x.grad.double() - xd.grad).abs().amax(-1) / xd.grad.abs().amax(-1)).max() < 1e-3
    for cmp in (tb.pignistic(out, F), tb.condition(out, 0b0011, F)[0]):
        assert torch.isfinite(cmp).all()


class Pipeline(nn.Module):
    """Two evidential heads → correction → fusion → decisions and losses."""

    def __init__(self):
        super().__init__()
        x, y = torch.randn(30, 2), torch.randint(F.n, (30,))
        self.enn = tb.PrototypeHead(2, F, n_prototypes=5)
        self.eknn = tb.EkNNHead(F, x, y, k=3)
        self.cd = tb.ContextualCorrection(F, "discount")

    def forward(self, x, y):
        m1, m2 = self.cd(self.enn(x)), self.eknn(x)
        fused, _ = tb.dempster(m1, m2, F)
        cautious, _ = tb.cautious(m1, m2, F)
        loss = (tb.pignistic_nll(fused, y, F) + tb.pl_loss(cautious, y, F)
                + tb.expected_utility_loss(fused, y, F))
        return loss, tb.soft_decide(fused, torch.eye(F.n), F, "hurwicz"), tb.soft_dominance(fused, F)


def test_torch_compile_fullgraph_matches_eager():
    model = Pipeline()
    x, y = torch.randn(16, 2), torch.randint(F.n, (16,))
    compiled = torch.compile(model, fullgraph=True)
    for eager, comp in zip(model(x, y), compiled(x, y)):
        assert torch.allclose(eager, comp, atol=1e-5)
    compiled(x, y)[0].backward()
    assert all(p.grad is not None for p in model.parameters())


def test_display():
    from torchbelief import display
    G = tb.Frame(3, names=["car", "truck", "night_bus"])
    m = torch.zeros(G.size, dtype=torch.float64)
    m[0b001], m[0b011], m[G.theta] = 0.5, 0.3, 0.2
    text = display.to_string(m, G).splitlines()
    assert text[0].split() == ["set", "m", "Bel", "Pl", "Q"]
    assert [line.split()[0] for line in text[1:]] == ["{car}", "{car,", "Θ"]
    assert "0.800" in text[2] and "█" in text[1]
    assert len(display.to_string(m, G, all_sets=True, bar_width=0).splitlines()) == G.size + 1
    assert display.to_html(m, G).count("<tr>") == 4
    tex = display.to_latex(m, G, all_sets=True)
    assert r"$\emptyset$" in tex and r"night\_bus" in tex and r"$\Theta$" in tex
    with pytest.raises(ValueError):
        display.to_string(m.expand(2, -1), G)
