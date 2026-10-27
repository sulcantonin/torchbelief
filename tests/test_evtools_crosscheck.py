"""Cross-check against evtools (``pip install evtools-dst``), a NumPy reference
implementation that uses the same subset bit ordering. Skipped if it is not installed."""

import numpy as np
import pytest
import torch

import torchbelief as tb

ev = pytest.importorskip("evtools")
from evtools import combinations as C, conversions as V, corrections as R, decision as D  # noqa: E402
from evtools import learning as L, metrics as M  # noqa: E402
from evtools.classifiers.eknn import EkNN  # noqa: E402
from evtools.dsvector import DSVector, Kind  # noqa: E402

F = tb.Frame(4)
NAMES = ["a", "b", "c", "d"]
G = torch.Generator().manual_seed(1)


def full_mass(batch=()):
    """Every subset focal, ∅ included: non-dogmatic *and* subnormal."""
    w = -torch.log(torch.rand(*batch, F.size, generator=G, dtype=torch.float64))
    return w / w.sum(-1, keepdim=True)


def dsv(m):
    return DSVector.from_dense(NAMES, m.numpy(), kind=Kind.M)


def dense(d):
    return torch.from_numpy(np.asarray(d.dense, dtype=np.float64))


def fs(mask):
    return frozenset(NAMES[i] for i in range(F.n) if (mask >> i) & 1)


def close(a, b):
    assert torch.allclose(a, b, atol=1e-10), (a - b).abs().max()


def test_conversions():
    m = full_mass()
    x = m.numpy()
    close(tb.implicability(m, F), torch.from_numpy(V.mtob(x)))
    close(tb.belief(m, F), torch.from_numpy(V.mtobel(x)))
    close(tb.plausibility(m, F), torch.from_numpy(V.mtopl(x)))
    close(tb.commonality(m, F), torch.from_numpy(V.mtoq(x)))
    close(tb.conjunctive_weights(m, F).exp(), torch.from_numpy(V.mtow(x)))
    close(tb.disjunctive_weights(m, F).exp(), torch.from_numpy(V.mtov(x)))
    close(tb.mass_from_plausibility(tb.plausibility(m, F), F), torch.from_numpy(V.pltom(V.mtopl(x))))


def test_combination_rules():
    m1, m2 = full_mass(), full_mass()
    close(tb.conjunctive(m1, m2, F), dense(C.crc(dsv(m1), dsv(m2))))
    close(tb.disjunctive(m1, m2, F), dense(C.drc(dsv(m1), dsv(m2))))
    close(tb.dempster(m1, m2, F)[0], dense(C.dempster(dsv(m1), dsv(m2))))
    close(tb.cautious(m1, m2, F, normalise=False)[0], dense(C.cautious(dsv(m1), dsv(m2))))
    close(tb.bold(m1, m2, F)[0], dense(C.bold(dsv(m1), dsv(m2))))


def test_decombination_and_conditioning():
    m1, m2 = full_mass(), full_mass()
    close(tb.decombine_conjunctive(m1, m2, F), dense(C.decombine_crc(dsv(m1), dsv(m2))))
    close(tb.decombine_disjunctive(m1, m2, F), dense(C.decombine_drc(dsv(m1), dsv(m2))))
    A = 0b0110
    close(tb.condition(m1, A, F, normalise=False)[0], dense(C.condition(dsv(m1), fs(A))))
    close(tb.decondition(m1, A, F), dense(C.decondition(dsv(m1), fs(A))))


def test_corrections():
    m = full_mass()
    beta = torch.rand(F.n, generator=G, dtype=torch.float64) * 0.8 + 0.2
    single = {fs(s): float(b) for s, b in zip(F.singletons(), beta)}
    compl = {fs(F.theta ^ s): float(b) for s, b in zip(F.singletons(), beta)}
    close(tb.discount(m, 0.7, F), dense(R.discount(dsv(m), 0.7)))
    close(tb.contextual_discount(m, beta, F), dense(R.contextual_discount(dsv(m), single)))
    close(tb.contextual_dediscount(m, beta, F), dense(R.contextual_dediscount(dsv(m), single)))
    close(tb.contextual_reinforce(m, beta, F), dense(R.contextual_reinforce(dsv(m), compl)))
    close(tb.contextual_dereinforce(m, beta, F), dense(R.contextual_dereinforce(dsv(m), compl)))
    close(tb.contextual_negate(m, beta, F), dense(R.contextual_negate(dsv(m), compl)))
    # Θ-contextual discounting on a coarsening of the frame
    part, b3 = [0b0011, 0b0100, 0b1000], beta[:3]
    ref = R.theta_contextual_discount(dsv(m), {fs(c): float(b) for c, b in zip(part, b3)})
    close(tb.contextual_discount(m, b3, F, contexts=part), dense(ref))


@pytest.mark.parametrize("seed", range(5))
def test_dominance(seed):
    torch.manual_seed(seed)
    m = tb.random_mass(F, n_focal=5)
    names = lambda keep: frozenset(NAMES[i] for i in range(F.n) if keep[i])
    assert names(tb.strong_dominance(m, F)) == D.strong_dominance(dsv(m))
    assert names(tb.weak_dominance(m, F)) == D.weak_dominance(dsv(m))


# ---- metrics, losses, learning, EkNN ---------------------------------------------------

def labelled(B=40):
    """Random masses with labels and the same data as DSVectors / atom names."""
    m = tb.random_mass(F, (B,), n_focal=5)
    y = torch.randint(F.n, (B,), generator=G)
    return m, y, [dsv(x) for x in m], [NAMES[i] for i in y]


def test_set_metrics():
    pred = torch.rand(50, F.n, generator=G) < 0.5
    y = torch.randint(F.n, (50,), generator=G)
    sets = [frozenset(NAMES[i] for i in range(F.n) if p[i]) for p in pred]
    labels = [NAMES[i] for i in y]
    for ours, ref in ((tb.metrics.discounted_accuracy, M.mean_discounted_accuracy),
                      (tb.metrics.u65, M.mean_u65), (tb.metrics.u80, M.mean_u80)):
        assert ours(pred, y).item() == pytest.approx(ref(sets, labels))


def test_pl_loss_hard_and_soft():
    m, y, ms, ys = labelled()
    assert tb.pl_loss(m, y, F).item() == pytest.approx(M.mean_pl_loss(ms, ys))
    soft = full_mass((40,))
    assert tb.pl_loss(m, soft, F).item() == pytest.approx(M.mean_pl_loss(ms, [dsv(s) for s in soft]))


def test_error_rejection_curve():
    m, y, ms, ys = labelled(60)
    conf, pred = tb.plausibility_transform(m, F).max(-1)
    rates, errors = tb.metrics.error_rejection_curve(conf, pred == y)
    ref_rates, ref_errors = M.error_rejection_curve(ms, ys, metric="R_plp")
    close(rates.double(), torch.from_numpy(ref_rates))
    close(errors.double(), torch.from_numpy(ref_errors))
    assert tb.metrics.auerc(conf, pred == y).item() == pytest.approx(M.auerc(ms, ys, metric="R_plp"))


@pytest.mark.parametrize("kind,fit,keys", [("discount", L.fit_cd, "single"),
                                           ("reinforce", L.fit_cr, "compl"),
                                           ("negate", L.fit_cn, "compl")])
def test_fit_contextual(kind, fit, keys):
    m, y, ms, ys = labelled(80)
    ctx = F.singletons() if keys == "single" else [F.theta ^ s for s in F.singletons()]
    for target, labels in ((y, ys), (full_mass((80,)), None)):
        ref = fit(ms, ys if labels is not None else [dsv(t) for t in target])
        expect = torch.tensor([ref[fs(c)] for c in ctx], dtype=torch.float64)
        close(tb.fit_contextual(m, target, F, kind)[0], expect)


def test_hard_to_soft_labels_statistics():
    torch.manual_seed(5)
    y = torch.randint(F.n, (20000,))
    ours = tb.hard_to_soft_labels(y, F)
    ref = L.hard_to_soft_labels([NAMES[i] for i in y], NAMES, rng=np.random.default_rng(5))
    ref_theta = torch.tensor([r.dense[F.theta] for r in ref])
    assert (ours[:, F.theta] > 0).float().mean() == pytest.approx((ref_theta > 0).float().mean(), abs=0.02)
    assert ours[:, F.theta].mean() == pytest.approx(ref_theta.mean().item(), abs=0.02)


def test_eknn_head():
    X = torch.randn(80, 3, generator=G, dtype=torch.float64)
    y = torch.randint(F.n, (80,), generator=G)
    q = torch.randn(10, 3, generator=G, dtype=torch.float64)
    ref = EkNN(k=6, alpha=0.95, optimize=False).fit(X.numpy(), y.numpy())
    head = tb.EkNNHead(F, X, y, k=6, alpha=0.95)
    close(head.gamma.detach(), torch.from_numpy(ref.gamma_))       # Denœux's initialisation
    ref.gamma_ = np.array([0.7, 1.3, 0.4, 2.0])                    # and any other γ
    head.gamma.data = torch.from_numpy(ref.gamma_.copy())
    close(head(q), torch.stack([dense(b) for b in ref.predict_bba(q.numpy())]))
