import itertools

import pytest
import torch

import torchbelief as tb
from torchbelief import possibility as P, inconsistency as I, choice as H

F = tb.Frame(3)
torch.manual_seed(1)


def test_mass_roundtrip_and_views():
    m = tb.random_mass(F, (4,))
    c = tb.Capacity.from_mass(m, F)
    assert torch.allclose(c.mass(), m, atol=1e-12)
    assert c.is_belief().all() and (c.monotonicity_violation() < 1e-12).all()
    assert (c.convexity_violation() < 1e-12).all()  # belief ⇒ 2-monotone
    assert torch.allclose(c.pignistic(), tb.pignistic(m, F))
    assert torch.allclose(c.upper, tb.plausibility(m, F))


def test_probability_view():
    p = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
    c = tb.Capacity.from_probability(p, F)
    assert c.is_probability() and c.is_belief()
    assert torch.allclose(c.pignistic(), p)
    assert c.choquet(torch.tensor([1., 2., 3.], dtype=torch.float64)) == pytest.approx(float(p @ torch.tensor([1., 2., 3.], dtype=torch.float64)))


def test_possibility_view():
    pi = torch.tensor([1.0, 0.6, 0.2], dtype=torch.float64)
    c = tb.Capacity.from_possibility(pi, F)
    assert c.is_possibility() and c.is_belief()          # necessity measures are belief functions
    assert torch.allclose(c.contour(), pi)
    m = c.mass()
    nz = [A for A in range(F.size) if m[A] > 1e-12]
    assert all((a & b) in (a, b) for a in nz for b in nz)  # nested focal sets (consonant)


def test_credal_roundtrip():
    m = tb.random_mass(F)
    c = tb.Capacity.from_mass(m, F)
    c2 = tb.Capacity.from_credal(c.credal_vertices(), F)
    assert torch.allclose(c.lower, c2.lower, atol=1e-12)  # vertices of the core recover v


def test_choquet_matches_mass_formula():
    m = tb.random_mass(F, (3,))
    c = tb.Capacity.from_mass(m, F)
    util = torch.randn(4, 3, dtype=torch.float64)
    lo, hi = c.expectation_bounds(util)
    elo, ehi = tb.expected_utility_bounds(m, util, F)
    assert torch.allclose(lo, elo) and torch.allclose(hi, ehi)


def test_choquet_gradient():
    m = tb.random_mass(F).requires_grad_()
    f = torch.randn(3, dtype=torch.float64, requires_grad=True)
    tb.Capacity.from_mass(m, F).choquet(f).backward()
    assert f.grad.abs().sum() > 0 and m.grad.abs().sum() > 0


def test_non_belief_capacity_detected():
    # a convex-but-not-belief and a non-convex capacity on 3 points
    v = torch.zeros(8, dtype=torch.float64)
    v[7] = 1; v[3] = v[5] = v[6] = 0.5; v[1] = v[2] = v[4] = 0.1  # 2-monotone? check numerically
    c = tb.Capacity(v, F)
    assert c.monotonicity_violation() == 0
    w = torch.zeros(8, dtype=torch.float64); w[7] = 1; w[1] = w[2] = 0.6  # v(A)+v(B)>v(A∪B)
    assert tb.Capacity(w, F).convexity_violation() > 0.1


def test_possibility_rules():
    p1 = torch.tensor([1.0, 0.3, 0.0], dtype=torch.float64)
    p2 = torch.tensor([0.0, 0.2, 1.0], dtype=torch.float64)
    out, k = P.conjunctive(p1, p2)
    assert k == pytest.approx(0.8)                       # strong conflict
    ad, _ = P.adaptive(p1, p2)
    assert ad.max() == pytest.approx(1.0) and ad[0] > 0 and ad[2] > 0  # falls back to union-ish
    agree, k2 = P.adaptive(p1, p1)
    assert torch.allclose(agree, p1) and k2 == pytest.approx(0.0)


def test_disagreement_and_resolve():
    a = tb.Capacity.from_mass(tb.categorical(F, 1, dtype=torch.float64), F)   # "ω0"
    b = tb.Capacity.from_mass(tb.categorical(F, 2, dtype=torch.float64), F)   # "ω1"
    v = tb.Capacity.vacuous(F, dtype=torch.float64)
    assert I.disagreement(a, b) == pytest.approx(1.0)
    assert I.disagreement(a, v) == pytest.approx(0.0)
    fused, d = I.resolve(a, b)
    assert fused.mass()[-1] == pytest.approx(1.0)        # conflict → ignorance, no blow-up
    assert not torch.isnan(fused.mass()).any()
    _, gap = I.credal_intersection(a, b)
    assert gap > 0.9


def test_choice_rules_and_abstention():
    util = torch.eye(3, dtype=torch.float64)
    sharp = tb.Capacity.from_mass(tb.categorical(F, 1, dtype=torch.float64), F)
    assert H.decide_or_abstain(sharp, util) == 0
    assert H.maximin(sharp, util) == 0 and H.pignistic_choice(sharp, util) == 0
    vac = tb.Capacity.vacuous(F, dtype=torch.float64)
    assert H.decide_or_abstain(vac, util) == -1          # total ignorance ⇒ abstain
    keep = H.maximal_acts(vac, util)
    assert keep.all()
    # E-admissible ⊇ maximal for convex capacities; both contain the unique winner
    m = tb.random_mass(F)
    c = tb.Capacity.from_mass(m, F)
    assert (H.e_admissible_acts(c, util) | ~H.maximal_acts(c, util)).all()


def test_interval_dominance_on_precise_probability():
    # Dempster with a Bayesian source is Bayesian: lower == upper only up to rounding,
    # which must not make the best act dominated by itself.
    lidar = tb.Capacity.from_mass(tb.simple_support(F, 1, 0.7, dtype=torch.float64), F)
    camera = tb.Capacity.from_probability(torch.tensor([0.75, 0.2, 0.05], dtype=torch.float64), F)
    fused = tb.Capacity.from_mass(tb.dempster(lidar.mass(), camera.mass(), F)[0], F)
    util = torch.eye(3, dtype=torch.float64)
    assert H.interval_dominance_acts(fused, util).tolist() == [True, False, False]
    for rule in ("interval", "maximality", "e_admissible"):
        assert H.decide_or_abstain(fused, util, rule=rule) == 0, rule
