import importlib
import itertools

import pytest
import torch

import torchbelief as tb
from torchbelief import belnap as B, inconsistency as I
from torchbelief.continuous import RandomInterval
from torchbelief.sparse import SparseMass

C = importlib.import_module("torchbelief.combine")
F = tb.Frame(3)
N = F.size
D = torch.float64
torch.manual_seed(2)


def brute_pcr6(ms):
    out = torch.zeros(N, dtype=D)
    for t in itertools.product(range(1, N), repeat=len(ms)):
        ws = [m[a] for m, a in zip(ms, t)]
        p = torch.stack(ws).prod()
        inter = t[0]
        for a in t[1:]:
            inter &= a
        if inter:
            out[inter] += p
        elif sum(ws) > 0:
            for a, w in zip(t, ws):
                out[a] += p * w / sum(ws)
    return out


# ---- combination rules ---------------------------------------------------------------

def test_pcr6_matches_bruteforce_and_pcr5():
    ms = [tb.random_mass(F) for _ in range(3)]
    out, k = tb.pcr6(ms, F)
    assert torch.allclose(out, brute_pcr6(ms)) and tb.is_valid(out)
    p5, _ = tb.pcr5(ms[0], ms[1], F)
    assert torch.allclose(p5, brute_pcr6(ms[:2]))
    out2, _ = tb.combine(ms, F, "pcr6")
    assert torch.allclose(out, out2)


def test_pcr5_batched_and_vacuous_identity():
    m = tb.random_mass(F, (4,))
    out, k = tb.pcr5(m, tb.vacuous(F, (4,), dtype=D), F)
    assert torch.allclose(out, m) and torch.allclose(k, torch.zeros(4, dtype=D))


def test_pcr5_total_conflict_splits_back():
    a, b = tb.categorical(F, 1, dtype=D), tb.categorical(F, 2, dtype=D)
    out, k = tb.pcr5(a, b, F)
    assert k == pytest.approx(1.0) and out[1] == pytest.approx(0.5) and out[2] == pytest.approx(0.5)


def test_canonical_weights_roundtrip_and_cautious_bold_idempotent():
    m = tb.discount(tb.random_mass(F, (3,)), 0.8, F)
    assert torch.allclose(tb.from_conjunctive_weights(tb.conjunctive_weights(m, F), F), m)
    c, _ = tb.cautious(m, m, F)
    assert torch.allclose(c, m)
    sub = 0.8 * m
    sub[..., 0] += 0.2
    b, _ = tb.bold(sub, sub, F)
    assert torch.allclose(b, sub)


def test_cautious_commutative_associative_and_simple_supports():
    a, b, c = (tb.discount(tb.random_mass(F), 0.7, F) for _ in range(3))
    ab, _ = tb.cautious(a, b, F, normalise=False)
    ba, _ = tb.cautious(b, a, F, normalise=False)
    assert torch.allclose(ab, ba)
    l, _ = tb.combine([a, b, c], F, "cautious")
    r, _ = tb.combine([c, b, a], F, "cautious")
    assert torch.allclose(l, r)
    # two simple supports on the same set: cautious keeps the stronger, not the product
    s1, s2 = tb.simple_support(F, 3, 0.6, dtype=D), tb.simple_support(F, 3, 0.8, dtype=D)
    out, _ = tb.cautious(s1, s2, F)
    assert out[3] == pytest.approx(0.8)


def test_contextual_discount():
    m = tb.random_mass(F)
    beta = torch.tensor([0.3, 0.9, 0.5], dtype=D)
    out = tb.contextual_discount(m, beta, F)
    s = F.singletons()
    assert tb.is_valid(out)
    assert torch.allclose(tb.plausibility(out, F)[s], 1 - beta + beta * tb.plausibility(m, F)[s])
    assert torch.allclose(tb.contextual_discount(m, torch.ones(3, dtype=D), F), m)
    assert torch.allclose(tb.contextual_discount(m, torch.zeros(3, dtype=D), F), tb.vacuous(F, dtype=D))


def test_capacity_discount_matches_mass_discount():
    m = tb.random_mass(F, (2,))
    a = tb.Capacity.from_mass(m, F).discount(0.4)
    assert torch.allclose(a.mass(), tb.discount(m, 0.4, F))


# ---- inconsistency ---------------------------------------------------------------------

def _sources():
    good = [tb.Capacity.from_mass(tb.simple_support(F, 1, w, dtype=D), F) for w in (0.7, 0.8, 0.6)]
    bad = tb.Capacity.from_mass(tb.categorical(F, 4, dtype=D), F)
    return good + [bad]


def test_disagreement_matrix_and_measures():
    src = _sources()
    Dm = I.disagreement_matrix(src)
    assert torch.allclose(Dm, Dm.T) and (Dm.diagonal() == 0).all()
    assert Dm[0, 3] == pytest.approx(float(I.disagreement(src[0], src[3])))
    meas = I.inconsistency_measures(src)
    assert meas["drastic"] == 1 and meas["mi"] == 3
    assert meas["problematic"].all()
    meas_ok = I.inconsistency_measures(src[:3])
    assert meas_ok["drastic"] == 0 and meas_ok["free"].all()


def test_shapley_blame_efficiency_and_outlier():
    src = _sources()
    for measure in ("max", "sum", "count"):
        phi = I.shapley_blame(src, measure)
        total = I.coalition_inconsistency(I.disagreement_matrix(src), measure)[-1]
        assert phi.sum() == pytest.approx(float(total))          # efficiency axiom
        assert phi.argmax() == 3                                   # the outlier
    _, alpha = I.blame_discount(src)
    assert alpha.argmin() == 3 and (alpha >= 0).all() and (alpha <= 1).all()


def test_soft_resolve_is_valid_and_differentiable():
    m = tb.random_mass(F).requires_grad_()
    a = tb.Capacity.from_mass(m, F)
    b = tb.Capacity.from_mass(tb.categorical(F, 4, dtype=D), F)
    for strategy in ("adaptive", "cautious", "dubois_prade", "pcr5"):
        out, d = I.resolve(a, b, strategy=strategy, temperature=0.1)
        assert tb.is_valid(out.mass().detach(), atol=1e-6), strategy
    out.mass()[4].backward()
    assert m.grad.abs().sum() > 0


# ---- Belnap ----------------------------------------------------------------------------

def test_belnap_four_values():
    a = tb.Capacity.from_mass(tb.simple_support(F, 1, 0.9, dtype=D), F)
    b = tb.Capacity.from_mass(tb.simple_support(F, 2, 0.8, dtype=D), F)
    x = B.pool([a, b])
    assert [B.LABELS[i] for i in B.label(x)] == ["both", "both", "false"]
    deg = B.degrees(x)
    assert torch.allclose(deg.sum(-1), torch.ones(3, dtype=D))
    assert (B.contradiction(B.singletons(a)) == 0).all()             # one coherent source
    vac = B.singletons(tb.Capacity.vacuous(F, dtype=D))
    assert (B.label(vac) == 3).all() and torch.allclose(B.ignorance(vac), torch.ones(3, dtype=D))
    t = torch.tensor([1.0, 0.0])
    assert torch.equal(B.negate(t), torch.tensor([0.0, 1.0]))
    assert torch.equal(B.knowledge_join(t, B.negate(t)), torch.tensor([1.0, 1.0]))
    assert torch.equal(B.truth_and(t, B.negate(t)), torch.tensor([0.0, 1.0]))


# ---- random intervals -----------------------------------------------------------------

def _ri():
    return RandomInterval(torch.tensor([0., 1., 2.], dtype=D), torch.tensor([2., 3., 5.], dtype=D))


def test_random_interval_bounds():
    a = _ri()
    lo, hi = a.mean_bounds()
    assert lo == pytest.approx(1.0) and hi == pytest.approx(10 / 3)
    Fl, Fu = a.cdf_bounds(torch.tensor([1., 2.5, 5.], dtype=D))
    assert torch.allclose(Fl, torch.tensor([0., 1 / 3, 1.], dtype=D))
    assert torch.allclose(Fu, torch.tensor([2 / 3, 1., 1.], dtype=D))
    bel, pl = a.prob_bounds(1.0, 3.0)
    assert bel == pytest.approx(1 / 3) and pl == pytest.approx(1.0)
    elo, ehi = a.expectation_bounds(lambda x: x)
    assert elo == pytest.approx(lo.item()) and ehi == pytest.approx(hi.item())
    p = RandomInterval.precise(torch.tensor([1., 3.], dtype=D))
    assert p.width() == 0 and p.mean_bounds()[0] == pytest.approx(2.0)


def test_random_interval_conjunctive_and_discretize():
    a = _ri()
    b = RandomInterval(torch.tensor([1.5, 4.], dtype=D), torch.tensor([2.5, 6.], dtype=D),
                       torch.tensor([0.6, 0.4], dtype=D))
    c, k = a.conjunctive(b)
    assert k == pytest.approx((0.4 * 2 + 0.6 * 0) / 3)                 # [0,2]∩[4,6], [1,3]∩[4,6]
    assert c.mass.sum() == pytest.approx(1.0)
    m, fr = a.discretize(torch.arange(7, dtype=D))
    assert fr.n == 6 and tb.is_valid(m)
    assert m[0b111] == pytest.approx(1 / 3) and m[0b111100] == pytest.approx(1 / 3)


def test_random_interval_smooth_gradients_and_pbox():
    lo = torch.tensor([0., 1.], dtype=D, requires_grad=True)
    a = RandomInterval(lo, lo + 1)
    Fl, _ = a.cdf_bounds(torch.tensor([1.5], dtype=D), temperature=0.1)
    Fl.sum().backward()
    assert lo.grad.abs().sum() > 0
    g = torch.linspace(-4, 4, 801, dtype=D)
    N0 = torch.distributions.Normal
    pb = RandomInterval.from_pbox(N0(0.5, 1.).cdf(g), N0(-0.5, 1.).cdf(g), g)
    mlo, mhi = pb.mean_bounds()
    assert mlo == pytest.approx(-0.5, abs=0.05) and mhi == pytest.approx(0.5, abs=0.05)


# ---- sparse masses ---------------------------------------------------------------------

def test_sparse_matches_dense():
    G = tb.Frame(5)
    m1, m2 = tb.random_mass(G, n_focal=6), tb.random_mass(G, n_focal=5)
    s1, s2 = SparseMass.from_dense(m1, G), SparseMass.from_dense(m2, G)
    for rule in ("dempster", "yager", "dubois_prade"):
        ds, k = getattr(s1, rule)(s2)
        dd, kd = getattr(tb, rule)(m1, m2, G)
        assert torch.allclose(ds.to_dense(G), dd) and torch.allclose(k, kd), rule
    assert torch.allclose(s1.conjunctive(s2).to_dense(G), tb.conjunctive(m1, m2, G))
    assert torch.allclose(s1.disjunctive(s2).to_dense(G), tb.disjunctive(m1, m2, G))
    ev = torch.arange(G.size)
    assert torch.allclose(s1.belief(ev), tb.belief(m1, G))
    assert torch.allclose(s1.plausibility(ev), tb.plausibility(m1, G))
    assert torch.allclose(s1.pignistic(), tb.pignistic(m1, G))
    assert torch.allclose(s1.discount(0.3).to_dense(G), tb.discount(m1, 0.3, G))
    u = torch.randn(3, 5, dtype=D)
    for a, b in zip(s1.expectation_bounds(u), tb.expected_utility_bounds(m1, u, G)):
        assert torch.allclose(a, b)


def test_sparse_summarize_is_outer_approximation():
    G = tb.Frame(6)
    s = SparseMass.from_dense(tb.random_mass(G, n_focal=20), G)
    t = s.conjunctive(SparseMass.from_dense(tb.random_mass(G, n_focal=20), G))
    small = t.summarize(5)
    ev = torch.arange(G.size)
    assert len(small) <= 6                                              # 5 + possibly ∅
    assert small.weight.sum() == pytest.approx(t.weight.sum().item())
    assert (small.belief(ev) <= t.belief(ev) + 1e-12).all()
    assert (small.plausibility(ev) >= t.plausibility(ev) - 1e-12).all()


def test_sparse_large_frame():
    n = 60
    a = SparseMass.simple_support(n, (1 << 59) | 1, 0.7)
    b = SparseMass.simple_support(n, 1 | 2, 0.6)
    c, k = a.dempster(b)
    assert k == 0 and c.belief(1) == pytest.approx(0.42)
    assert c.pignistic().sum() == pytest.approx(1.0)


# ---- neural layers ---------------------------------------------------------------------

def test_dirichlet_mapping():
    alpha = torch.tensor([[1., 1., 1.], [11., 1., 1.]], dtype=D)
    m = tb.from_dirichlet(F, alpha)
    assert tb.is_valid(m)
    assert m[0, -1] == pytest.approx(1.0)                               # no evidence ⇒ vacuous
    assert m[1, 1] == pytest.approx(10 / 13) and m[1, -1] == pytest.approx(3 / 13)


def test_evidential_head_and_losses():
    head = tb.EvidentialHead(8, F)
    x, y = torch.randn(16, 8), torch.randint(0, 3, (16,))
    m = head(x)
    assert tb.is_valid(m.detach(), atol=1e-5)
    bn, pn = tb.belief_nll(m, y, F), tb.plausibility_nll(m, y, F)
    assert bn >= pn                                                     # Bel ≤ Pl
    assert tb.hurwicz_nll(m, y, F, 0.0).item() == pytest.approx(bn.item())
    assert tb.hurwicz_nll(m, y, F, 1.0).item() == pytest.approx(pn.item())
    (bn + 0.1 * tb.nonspecificity_penalty(m, F)).backward()
    assert head.fc.weight.grad.abs().sum() > 0


def test_prototype_head_matches_dempster_and_is_vacuous_far_away():
    head = tb.PrototypeHead(2, F, n_prototypes=3).double()
    x = torch.randn(5, 2, dtype=D)
    m = head(x)
    assert tb.is_valid(m.detach(), atol=1e-6)
    # closed form equals folding Dempster's rule over the per-prototype simple masses
    with torch.no_grad():
        d2 = torch.cdist(x, head.prototypes).pow(2)
        s = torch.sigmoid(head.alpha_logit) * torch.exp(-head.gamma.pow(2) * d2)
        u = head.beta.softmax(-1)
        parts = []
        for i in range(3):
            mi = torch.zeros(5, N, dtype=D)
            mi[:, F.singletons()] = s[:, i:i + 1] * u[i]
            mi[:, -1] = 1 - s[:, i]
            parts.append(mi)
        ref, _ = tb.combine(parts, F, "dempster")
    assert torch.allclose(m.detach(), ref, atol=1e-6)
    far = head(torch.full((1, 2), 1e3, dtype=D))
    assert far[0, -1].item() == pytest.approx(1.0)
    m.sum().backward()
    assert head.prototypes.grad is not None
