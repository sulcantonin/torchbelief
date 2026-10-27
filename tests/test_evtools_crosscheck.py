"""Cross-check against evtools (``pip install evtools-dst``), a NumPy reference
implementation that uses the same subset bit ordering. Skipped if it is not installed."""

import numpy as np
import pytest
import torch

import torchbelief as tb

ev = pytest.importorskip("evtools")
from evtools import combinations as C, conversions as V, corrections as R, decision as D  # noqa: E402
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
