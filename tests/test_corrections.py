import pytest
import torch

import torchbelief as tb

F = tb.Frame(4)
N = F.size
torch.manual_seed(0)


def full_mass(batch=()):
    """Every subset focal, ∅ included: non-dogmatic *and* subnormal."""
    w = -torch.log(torch.rand(*batch, N, dtype=torch.float64))
    return w / w.sum(-1, keepdim=True)


def test_implicability_and_plausibility_roundtrips():
    m = full_mass((3,))
    b = tb.implicability(m, F)
    assert torch.allclose(b[..., 1:] - b[..., :1], tb.belief(m, F)[..., 1:])
    assert torch.allclose(tb.mass_from_implicability(b, F), m)
    assert torch.allclose(tb.mass_from_plausibility(tb.plausibility(m, F), F), m)  # m(∅) > 0 kept


def test_disjunctive_weights_roundtrip_and_bold():
    m1, m2 = full_mass(), full_mass()
    lv1, lv2 = tb.disjunctive_weights(m1, F), tb.disjunctive_weights(m2, F)
    assert torch.allclose(tb.from_disjunctive_weights(lv1, F), m1)
    assert torch.allclose(tb.from_disjunctive_weights(torch.minimum(lv1, lv2), F), tb.bold(m1, m2, F)[0])


def test_decombination_inverts_combination():
    m1, m2 = full_mass((2,)), full_mass((2,))
    assert torch.allclose(tb.decombine_conjunctive(tb.conjunctive(m1, m2, F), m2, F), m1)
    assert torch.allclose(tb.decombine_disjunctive(tb.disjunctive(m1, m2, F), m2, F), m1)


def test_condition_is_conjunction_with_categorical():
    m, A = tb.random_mass(F, (3,)), 0b0101
    cat = tb.categorical(F, A, dtype=m.dtype)
    out, k = tb.condition(m, A, F, normalise=False)
    assert torch.allclose(out, tb.conjunctive(m, cat, F)) and torch.allclose(k, out[..., 0])
    md, kd = tb.dempster(m, cat, F)
    out, k = tb.condition(m, A, F)
    assert torch.allclose(out, md) and torch.allclose(k, kd)
    # deconditioning is the least committed inverse: conditioning it again gives m[A] back
    mA = tb.condition(m, A, F)[0]
    assert torch.allclose(tb.condition(tb.decondition(mA, A, F), A, F)[0], mA)


def test_simple_corrections_are_special_contexts():
    m, a = tb.random_mass(F, (2,)), torch.tensor([0.3, 0.8], dtype=torch.float64)
    # one context Θ = Shafer discounting, one context ∅ = reinforcement / negating
    assert torch.allclose(tb.contextual_discount(m, a[:, None], F, contexts=[F.theta]), tb.discount(m, a, F))
    assert torch.allclose(tb.contextual_reinforce(m, a[:, None], F, contexts=[0]), tb.reinforce(m, a, F))
    assert torch.allclose(tb.contextual_negate(m, a[:, None], F, contexts=[0]), tb.negate(m, a, F))
    assert torch.allclose(tb.negate(m, 0.0, F), tb.negation(m))


def test_corrections_keep_masses_valid_and_contours_linear():
    m, beta = tb.random_mass(F, (5,)), torch.rand(5, F.n, dtype=torch.float64)
    pl = tb.plausibility(m, F)[..., F.singletons()]
    cd = tb.contextual_discount(m, beta, F)
    cr = tb.contextual_reinforce(m, beta, F)
    cn = tb.contextual_negate(m, beta, F)
    for out in (cd, cr, cn, tb.reinforce(m, 0.4, F), tb.negate(m, 0.4, F)):
        assert tb.is_valid(out, closed_world=False)
    # Pichon et al. 2016, Propositions 12, 14 and 16
    contour = lambda x: tb.plausibility(x, F)[..., F.singletons()]
    assert torch.allclose(contour(cd), 1 - beta * (1 - pl))
    assert torch.allclose(contour(cr), beta * pl)
    assert torch.allclose(contour(cn), 0.5 + (pl - 0.5) * (2 * beta - 1))


def test_de_corrections_invert_corrections():
    m, beta = tb.random_mass(F, (3,)), torch.rand(3, F.n, dtype=torch.float64) * 0.9 + 0.1
    assert torch.allclose(tb.contextual_dediscount(tb.contextual_discount(m, beta, F), beta, F), m)
    assert torch.allclose(tb.contextual_dereinforce(tb.contextual_reinforce(m, beta, F), beta, F), m)


def test_corrections_differentiable_in_beta():
    m = tb.random_mass(F)
    beta = (torch.rand(F.n, dtype=torch.float64) * 0.8 + 0.1).requires_grad_()
    for fn in (tb.contextual_discount, tb.contextual_reinforce, tb.contextual_negate):
        assert torch.autograd.gradcheck(lambda b: fn(m, b, F), (beta,))


def test_dominance():
    p = lambda *x: tb.from_probabilities(F, torch.tensor(x, dtype=torch.float64))
    assert tb.strong_dominance(tb.vacuous(F), F).all() and tb.weak_dominance(tb.vacuous(F), F).all()
    cat = tb.categorical(F, 0b0010, dtype=torch.float64)
    assert tb.strong_dominance(cat, F).tolist() == [False, True, False, False]
    # ties are kept, never dominated by each other
    tie = p(0.4, 0.4, 0.2, 0.0)
    assert tb.strong_dominance(tie, F).tolist() == [True, True, False, False]
    # weak dominance keeps a subset of strong dominance
    m = tb.random_mass(F, (50,), n_focal=6)
    strong, weak = tb.strong_dominance(m, F), tb.weak_dominance(m, F)
    assert (weak <= strong).all() and weak.any(-1).all()
