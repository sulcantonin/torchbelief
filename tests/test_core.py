import itertools
import math

import pytest
import torch

import torchbelief as tb

F = tb.Frame(4)
N = F.size
torch.manual_seed(0)


def brute(m1, m2, op):
    out = torch.zeros(N, dtype=m1.dtype)
    for a, b in itertools.product(range(N), range(N)):
        out[op(a, b)] += m1[a] * m2[b]
    return out


def brute_dp(m1, m2):
    out = torch.zeros(N, dtype=m1.dtype)
    for a, b in itertools.product(range(N), range(N)):
        out[(a | b) if a & b == 0 else (a & b)] += m1[a] * m2[b]
    return out


def test_bel_pl_bruteforce():
    m = tb.random_mass(F)
    bel, pl = tb.belief(m, F), tb.plausibility(m, F)
    for A in range(N):
        assert bel[A] == pytest.approx(float(sum(m[B] for B in range(1, N) if B & ~A == 0)))
        assert pl[A] == pytest.approx(float(sum(m[B] for B in range(N) if B & A)))
    assert (bel <= pl + 1e-12).all()


def test_roundtrips():
    m = tb.random_mass(F, (3,))
    assert torch.allclose(tb.mass_from_belief(tb.belief(m, F), F), m)
    assert torch.allclose(tb.mass_from_commonality(tb.commonality(m, F), F), m)


def test_conjunctive_disjunctive_match_bruteforce():
    m1, m2 = tb.random_mass(F), tb.random_mass(F)
    assert torch.allclose(tb.conjunctive(m1, m2, F), brute(m1, m2, lambda a, b: a & b))
    assert torch.allclose(tb.disjunctive(m1, m2, F), brute(m1, m2, lambda a, b: a | b))


def test_dempster_yager_dp():
    m1, m2 = tb.random_mass(F, n_focal=3), tb.random_mass(F, n_focal=3)
    raw = brute(m1, m2, lambda a, b: a & b)
    k = raw[0]
    d, kd = tb.dempster(m1, m2, F)
    assert kd == pytest.approx(k.item()) and tb.conflict(m1, m2, F) == pytest.approx(k.item())
    expect = raw.clone(); expect[0] = 0; expect /= 1 - k
    assert torch.allclose(d, expect)
    y, _ = tb.yager(m1, m2, F)
    ey = raw.clone(); ey[N - 1] += k; ey[0] = 0
    assert torch.allclose(y, ey)
    dp, _ = tb.dubois_prade(m1, m2, F)
    assert torch.allclose(dp, brute_dp(m1, m2))
    for m in (d, y, dp):
        assert tb.is_valid(m)


def test_vacuous_is_identity_and_batching():
    m = tb.random_mass(F, (5,))
    v = tb.vacuous(F, (5,), dtype=m.dtype)
    for rule in ("conjunctive", "dempster", "yager", "dubois_prade"):
        out, k = tb.combine([m, v], F, rule)
        assert torch.allclose(out, m, atol=1e-10), rule
        assert k.shape == (5,)


def test_dempster_associative_commutative_yager_commutative():
    a, b, c = (tb.random_mass(F) for _ in range(3))
    ab_c, _ = tb.combine([a, b, c], F, "dempster")
    a_bc, _ = tb.combine([c, b, a], F, "dempster")
    assert torch.allclose(ab_c, a_bc)
    y1, _ = tb.yager(a, b, F); y2, _ = tb.yager(b, a, F)
    assert torch.allclose(y1, y2)


def test_discount():
    m = tb.random_mass(F)
    assert torch.allclose(tb.discount(m, 1.0, F), m)
    assert torch.allclose(tb.discount(m, 0.0, F), tb.vacuous(F, dtype=m.dtype))
    assert tb.is_valid(tb.discount(m, 0.3, F))


def test_pignistic_and_decisions():
    m = tb.random_mass(F, (7,))
    p = tb.pignistic(m, F)
    assert torch.allclose(p.sum(-1), torch.ones(7, dtype=p.dtype))
    b, pl = tb.interval(m, F)
    assert ((b <= p + 1e-12) & (p <= pl + 1e-12)).all()
    util = torch.eye(4, dtype=m.dtype)
    lo, hi = tb.expected_utility_bounds(m, util, F)
    assert (lo <= p + 1e-12).all() and (p <= hi + 1e-12).all()
    assert torch.allclose(lo, b) and torch.allclose(hi, pl)  # 0/1 utilities recover Bel/Pl
    assert tb.decide(m, util, F).shape == (7,)


def test_uncertainty_measures_extremes():
    v = tb.vacuous(F, dtype=torch.float64)
    bay = tb.from_probabilities(F, torch.full((4,), 0.25, dtype=torch.float64))
    assert tb.nonspecificity(v, F) == pytest.approx(math.log2(4))
    assert tb.nonspecificity(bay, F) == pytest.approx(0.0)
    assert tb.deng_entropy(bay, F) == pytest.approx(2.0)  # reduces to Shannon
    assert tb.jousselme_distance(v, v, F) == pytest.approx(0.0)
    assert tb.jousselme_distance(tb.categorical(F, 1, dtype=torch.float64),
                                 tb.categorical(F, 2, dtype=torch.float64), F) == pytest.approx(1.0)


def test_gradients_dempster_vs_yager():
    """Yager's partial derivatives stay bounded as κ→1; Dempster's do not."""
    f = tb.Frame(2)
    def grad_norm(rule):
        a = torch.tensor([0.999, 0.0, 0.0, 0.001], dtype=torch.float64)  # m({0}) ~ 1
        b = torch.tensor([0.0, 0.999, 0.0, 0.001], dtype=torch.float64)
        a = torch.stack([a[0] * 0, a[1 - 1] * 0 + 0.999, torch.tensor(0., dtype=torch.float64), a[3]])
        a = torch.tensor([0., 0.999, 0., 0.001], dtype=torch.float64).requires_grad_()
        b = torch.tensor([0., 0., 0.999, 0.001], dtype=torch.float64).requires_grad_()
        out, _ = (tb.dempster(a, b, f, clamp=False) if rule == "dempster" else tb.yager(a, b, f))
        out[..., 1].backward()
        return max(a.grad.abs().max().item(), b.grad.abs().max().item())
    assert grad_norm("yager") <= 1.0 + 1e-9
    assert grad_norm("dempster") > 100


def test_masshead_and_loss_train():
    f = tb.Frame(3)
    head = tb.MassHead(8, f)
    x, y = torch.randn(32, 8), torch.randint(0, 3, (32,))
    m = head(x)
    assert tb.is_valid(m, atol=1e-5)
    loss = tb.pignistic_nll(m, y, f)
    loss.backward()
    assert head.fc.weight.grad.abs().sum() > 0
