import pytest
import torch

import torchbelief as tb
from torchbelief import metrics

F = tb.Frame(3)
torch.manual_seed(0)


def test_set_valued_metrics():
    pred = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=torch.bool)
    y = torch.zeros(5, dtype=torch.long)
    assert metrics.discounted_accuracy(pred, y, "none").tolist() == pytest.approx([1, 0.5, 1 / 3, 0, 0])
    assert metrics.u65(pred, y, "none")[:2].tolist() == pytest.approx([1, 0.65])
    assert metrics.u80(pred, y, "none")[:2].tolist() == pytest.approx([1, 0.8])
    assert metrics.u65(pred, y) == pytest.approx(metrics.u65(pred, y, "none").mean().item())


def test_error_rejection_curve():
    conf = torch.tensor([0.9, 0.2, 0.6, 0.4])
    correct = torch.tensor([True, False, True, False])
    rates, errors = metrics.error_rejection_curve(conf, correct)
    assert rates.tolist() == pytest.approx([0, 0.25, 0.5, 0.75, 1])
    assert errors.tolist() == pytest.approx([0.5, 1 / 3, 0, 0, 0])
    # rejecting the least confident first can only help when confidence ranks errors first
    assert metrics.auerc(conf, correct) < metrics.auerc(-conf, correct)


def test_decision_scores_and_soft_decide():
    m = tb.random_mass(F, (20,)).requires_grad_()
    U = torch.randn(4, F.n, dtype=torch.float64)
    for crit in ("pignistic", "plausibility", "maximin", "maximax", "hurwicz"):
        s = tb.decision_scores(m, U, F, crit)
        assert torch.equal(tb.decide(m, U, F, crit), s.argmax(-1))
        p = tb.soft_decide(m, U, F, crit, temperature=1e-4)
        assert torch.equal(p.argmax(-1), s.argmax(-1))
    tb.soft_decide(m, U, F, "hurwicz").max(-1).values.sum().backward()
    assert m.grad.abs().sum() > 0


def test_soft_dominance_tends_to_strong_dominance():
    m = tb.random_mass(F, (200,), n_focal=4)
    soft = tb.soft_dominance(m, F, temperature=1e-6)
    assert torch.equal(soft > 0.5, tb.strong_dominance(m, F))


def test_pl_loss():
    m = tb.random_mass(F, (6,))
    y = torch.randint(F.n, (6,))
    pl = tb.plausibility(m, F)[..., F.singletons()]
    ref = (pl - torch.nn.functional.one_hot(y, F.n)).pow(2).sum(-1).mean()
    assert tb.pl_loss(m, y, F) == pytest.approx(ref.item())
    soft = torch.stack([tb.categorical(F, 1 << int(k), dtype=m.dtype) for k in y])
    assert tb.pl_loss(m, soft, F) == pytest.approx(ref.item())


def test_expected_utility_loss_is_u65_of_categorical_sets():
    sets = torch.arange(1, F.size)
    m = torch.eye(F.size, dtype=torch.float64)[sets]                       # categorical on each set
    pred = F.membership()[sets]
    for y in range(F.n):
        t = torch.full((len(sets),), y)
        for a, b, u in ((1.6, 0.6, metrics.u65), (2.2, 1.2, metrics.u80)):
            per_set = [-tb.expected_utility_loss(m[i:i + 1], t[:1], F, a, b) for i in range(len(sets))]
            assert torch.allclose(torch.stack(per_set), u(pred, t, "none").double())


def test_max_mass_set_ignores_empty_set():
    m = torch.zeros(2, F.size, dtype=torch.float64)
    m[0, 0b011], m[0, 0b100] = 0.6, 0.4
    m[1, 0], m[1, 0b010], m[1, F.theta] = 0.7, 0.2, 0.1                    # ∅ heaviest
    assert tb.max_mass_set(m, F).tolist() == [[True, True, False], [False, True, False]]


def test_eknn_matches_dempster_fold_and_excludes_self():
    X = torch.randn(40, 2, dtype=torch.float64)
    y = torch.randint(F.n, (40,))
    head = tb.EkNNHead(F, X, y, k=4)
    q = torch.randn(3, 2, dtype=torch.float64)
    out = head(q)
    d2, idx = torch.cdist(q, X).pow(2).topk(4, largest=False)
    for b in range(3):
        acc = tb.vacuous(F, dtype=torch.float64)
        for j in range(4):
            c = int(y[idx[b, j]])
            s = head.alpha * torch.exp(-head.gamma[c] ** 2 * d2[b, j])
            acc, _ = tb.dempster(acc, tb.simple_support(F, 1 << c, s), F)
        assert torch.allclose(out[b], acc)
    # with self_index a training point is not its own neighbour
    i = torch.tensor([7])
    rest = torch.cat([torch.arange(7), torch.arange(8, 40)])
    other = tb.EkNNHead(F, X[rest], y[rest], k=4)
    other.gamma.data.copy_(head.gamma.data)
    assert torch.allclose(head(X[i], self_index=i), other(X[i]))
    tb.pl_loss(head(X, self_index=torch.arange(40)), y, F).backward()
    assert head.gamma.grad.abs().sum() > 0


@pytest.mark.parametrize("kind", ["discount", "reinforce", "negate"])
def test_gradient_descent_reaches_closed_form(kind):
    # a confident source that knows nothing: every correction has an interior optimum
    torch.manual_seed(1)
    y, said = torch.randint(F.n, (400,)), torch.randint(F.n, (400,))
    w = torch.rand(400, dtype=torch.float64) * 0.39 + 0.6
    m = torch.zeros(400, F.size, dtype=torch.float64)
    m[torch.arange(400), 1 << said] = w
    m[:, F.theta] = 1 - w
    beta = tb.fit_contextual(m, y, F, kind)[0]
    assert ((beta > 0.02) & (beta < 0.98)).all()                         # interior optimum
    layer = tb.ContextualCorrection(F, kind, init=0.5).double()
    opt = torch.optim.LBFGS(layer.parameters(), max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = tb.pl_loss(layer(m), y, F)
        loss.backward()
        return loss

    opt.step(closure)
    assert torch.allclose(layer.beta[0], beta, atol=1e-4)


def test_fit_contextual_per_group_and_soft_labels():
    m = tb.random_mass(F, (60,))
    y = torch.randint(F.n, (60,))
    g = torch.arange(60) % 2
    both = tb.fit_contextual(m, y, F, group=g, groups=2)
    for k in range(2):
        assert torch.allclose(both[k], tb.fit_contextual(m[g == k], y[g == k], F)[0])
    cat = torch.eye(F.size, dtype=m.dtype)[1 << y]                          # hard labels as masses
    assert torch.allclose(tb.fit_contextual(m, cat, F), tb.fit_contextual(m, y, F))


def test_hard_to_soft_labels_statistics():
    torch.manual_seed(3)
    y = torch.randint(F.n, (20000,))
    s = tb.hard_to_soft_labels(y, F, mu=0.5, var=0.04)
    assert tb.is_valid(s)
    soft = s[:, F.theta] > 0
    assert soft.float().mean() == pytest.approx(0.5, abs=0.02)            # E[p] = mu
    assert s[soft, F.theta].mean() == pytest.approx(0.58, abs=0.02)       # E[p | soft] = (var + mu²)/mu
    assert (s[~soft][torch.arange(int((~soft).sum())), 1 << y[~soft]] == 1).all()
    with pytest.raises(ValueError):
        tb.hard_to_soft_labels(y, F, mu=0.5, var=0.3)
