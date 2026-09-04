"""``stage1="saddlepoint"`` — the permutation tail computed, not sampled.

The claim is narrow and checkable: for a statistic that is a monotone function
of a subset sum, the double saddlepoint returns the *exact permutation*
probability, so it must agree with brute-force relabelling and not with any
distributional idealization. Everything here is against brute force.

The cheap checks live here; ``docs/scaling.md`` §4.9 has the expensive ones
(1e8 permutations per design, four geometries, counts and sparse counts) that
are too slow for a suite.
"""

from __future__ import annotations

from math import lgamma

import numpy as np
import pytest

import wade
from wade.saddlepoint import mean_diff_saddlepoint_p, saddlepoint_subset_sum
from wade.saddlepoint import _cgf, _solve_t

G, N1, N0 = 24, 20, 20
COND = np.r_[np.ones(N1, int), np.zeros(N0, int)]


def _cohort(seed=0, n1=N1, n0=N0, g=G, shift=1.6):
    rng = np.random.default_rng(seed)
    x = rng.lognormal(3.0, 0.6, (g, n1 + n0))
    x[: g // 3, :n1] *= shift
    return x


def _brute(x, cond, stat, n_perms=200_000, seed=5, alternative="greater"):
    """The p-value by exhaustive relabelling — the thing being approximated."""
    rng = np.random.default_rng(seed)
    n1 = int((cond == 1).sum())
    n0 = cond.size - n1
    nexc = np.zeros(x.shape[0], dtype=np.int64)
    done = 0
    while done < n_perms:
        b = min(50_000, n_perms - done)
        P = rng.permuted(np.broadcast_to(cond, (b, cond.size)), axis=1).astype(float)
        null = (x @ P.T) / n1 - (x @ (1 - P).T) / n0
        if alternative == "greater":
            nexc += (null >= stat[:, None]).sum(axis=1)
        elif alternative == "less":
            nexc += (null <= stat[:, None]).sum(axis=1)
        else:
            nexc += (np.abs(null) >= np.abs(stat)[:, None]).sum(axis=1)
        done += b
    return (1.0 + nexc) / (done + 1.0), nexc


# ---------------------------------------------------------------------------
# the solver's own contract


@pytest.mark.parametrize("n1,n0", [(20, 20), (30, 10), (10, 30), (39, 1)])
def test_inner_solve_hits_its_root_at_every_scale_of_s(n1, n0):
    """``_solve_t`` must satisfy ``K_t = n1`` for any ``s``.

    Pinned because a version of it did not: the bracket was found by
    geometric search, the widened lower end inflated the step used for the
    upper end, and the resulting bracket was too wide to close within the
    iteration budget — ``|K_t - n1|`` reached 10.8 at ``s = 0.1``. Everything
    downstream trusted the answer, so one gene in 64 came back with a
    plausible, wrong p-value. The bracket is now closed-form.
    """
    v = np.ascontiguousarray(_cohort(1, n1, n0))
    for s_val in (0.0, 1e-3, 0.05, 0.1, 0.2, 0.5, 1.0, 5.0, 50.0, -0.3, -5.0):
        s = np.full(v.shape[0], s_val)
        t = _solve_t(v, s, n1)
        assert np.abs(_cgf(v, s, t)[1] - n1).max() < 1e-8, f"at s={s_val}"


def test_a_threshold_outside_the_reachable_range_is_exact():
    """``A`` cannot exceed the sum of the largest ``n1`` values, nor fall below
    the smallest. Above the top the probability is zero and is reported as the
    one-labelling floor, because a p-value of exactly zero is a claim no
    permutation set supports; below the bottom it is exactly 1."""
    x = _cohort(2)
    n = N1 + N0
    floor = np.exp(-(lgamma(n + 1) - lgamma(N1 + 1) - lgamma(N0 + 1)))
    srt = np.sort(x, axis=1)
    hi = saddlepoint_subset_sum(x, N1, srt[:, -N1:].sum(axis=1) * 1.01)
    np.testing.assert_allclose(hi, floor, rtol=1e-12)
    lo = saddlepoint_subset_sum(x, N1, srt[:, :N1].sum(axis=1) * 0.99)
    assert np.all(lo == 1.0)


def test_an_all_cases_shift_lands_on_the_one_labelling_floor():
    """When every case exceeds every control the observed labelling *is* the
    maximum-sum one, so the exact permutation p-value is ``1 / C(n, n1)`` and
    nothing smaller is meaningful. Brute force cannot check this — it would
    need 1e11 draws — which is precisely the regime the saddlepoint exists
    for, and where the shipped GPD refinement returns 7.4e-6 against a truth
    of 7.3e-12."""
    rng = np.random.default_rng(23)
    x = rng.lognormal(3.0, 0.3, (8, N1 + N0))
    x[:, :N1] *= 6.0                                     # cases strictly above
    n = N1 + N0
    floor = np.exp(-(lgamma(n + 1) - lgamma(N1 + 1) - lgamma(N0 + 1)))
    p = mean_diff_saddlepoint_p(x, COND, alternative="greater")
    np.testing.assert_allclose(p, floor, rtol=1e-9)


def test_the_floor_is_one_labelling_and_p_is_never_zero():
    """A permutation p-value cannot be smaller than ``1 / C(n, n1)``: that is
    one outcome out of the exchangeable set. Without the floor the far tail
    underflows ``norm.sf`` to exactly zero, which is a claim no finite
    permutation set supports."""
    x = _cohort(3)
    x[:5, :N1] *= 50.0                                   # absurdly strong
    p = mean_diff_saddlepoint_p(x, COND, alternative="greater")
    n = N1 + N0
    floor = np.exp(-(lgamma(n + 1) - lgamma(N1 + 1) - lgamma(N0 + 1)))
    assert p.min() > 0.0
    assert p.min() >= floor * (1 - 1e-12)


def test_it_refuses_shapes_it_cannot_answer():
    x = _cohort(4)
    with pytest.raises(ValueError, match="2-D"):
        saddlepoint_subset_sum(x[0], N1, np.zeros(1))
    with pytest.raises(ValueError, match="strictly between"):
        saddlepoint_subset_sum(x, 0, np.zeros(G))
    with pytest.raises(ValueError, match="shape"):
        saddlepoint_subset_sum(x, N1, np.zeros(G + 1))
    with pytest.raises(ValueError, match="alternative"):
        mean_diff_saddlepoint_p(x, COND, alternative="bigger")


# ---------------------------------------------------------------------------
# against brute force — the only standard that matters


@pytest.mark.parametrize("n1,n0", [(20, 20), (30, 10), (10, 30)])
def test_agrees_with_brute_force_at_every_geometry(n1, n0):
    """Including unbalanced, which is the case the quantile quadrature could
    not reach: there stage 1 is an L-statistic, not a subset sum, and no
    saddlepoint exists for it (``scaling.md`` §4.4)."""
    x = _cohort(7, n1, n0, g=32)
    cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
    stat = x[:, :n1].mean(axis=1) - x[:, n1:].mean(axis=1)
    truth, nexc = _brute(x, cond, stat)
    ok = nexc >= 25                                      # brute force is sound here
    got = mean_diff_saddlepoint_p(x, cond, stat, alternative="greater")
    ratio = got[ok] / truth[ok]
    assert ok.sum() >= 8
    assert 0.85 < np.median(ratio) < 1.2, f"median {np.median(ratio):.3f}"
    assert ratio.min() > 0.5, f"anti-conservative to {ratio.min():.3f}"


@pytest.mark.parametrize("alternative", ["greater", "less", "two-sided"])
def test_every_alternative_matches_its_own_brute_force_null(alternative):
    """``two-sided`` is the exact ``P(|T| >= |t|)``, not a doubling — the same
    orientation :func:`wade.pvalues.perm_pvalues` uses."""
    x = _cohort(11, g=32)
    if alternative == "less":
        x[:10, :N1] /= 1.6 ** 2                          # push some genes down
    stat = x[:, :N1].mean(axis=1) - x[:, N1:].mean(axis=1)
    truth, nexc = _brute(x, COND, stat, alternative=alternative)
    ok = nexc >= 25
    got = mean_diff_saddlepoint_p(x, COND, stat, alternative=alternative)
    ratio = got[ok] / truth[ok]
    assert ok.sum() >= 8
    assert 0.8 < np.median(ratio) < 1.25, f"median {np.median(ratio):.3f}"


def test_null_p_values_are_uniform():
    """The property a p-value has to have. Checked on counts through
    :func:`wade.wade`'s own normalization, which is what stage 1 actually
    sees — and at the sparsity the real cohort has (``scaling.md`` §7.5)."""
    rng = np.random.default_rng(13)
    g = 4000
    mu = 10 ** rng.uniform(-1.0, 0.7, (g, 1))            # ~50% zeros
    counts = rng.poisson(rng.gamma(10.0, mu / 10.0, (g, N1 + N0))).astype(float)
    assert np.mean(counts == 0) > 0.35, "the sparse regime is the point here"
    res = wade.wade(counts, np.ones(g), COND, lib_sizes=np.ones(N1 + N0),
                    nperms=50, seed=1, subset=False)
    p = mean_diff_saddlepoint_p(res.tpm, COND, alternative="two-sided")
    for level in (0.05, 0.25, 0.50):
        se = np.sqrt(level * (1 - level) / g)
        assert abs(np.mean(p <= level) - level) < 4 * se, f"at {level}"


# ---------------------------------------------------------------------------
# the wade() surface


def test_stage1_saddlepoint_reports_the_statistic_it_tested():
    """One statistic end to end, and the same one ``gemm`` reports."""
    rng = np.random.default_rng(17)
    counts = rng.poisson(60, size=(40, N1 + N0)).astype(float)
    counts[:6, :N1] *= 2.5
    kw = dict(lib_sizes=np.ones(N1 + N0), nperms=600, seed=1)
    sp = wade.wade(counts, np.ones(40), COND, stage1="saddlepoint", **kw)
    gm = wade.wade(counts, np.ones(40), COND, stage1="gemm", **kw)
    np.testing.assert_allclose(sp.mean_shift, gm.mean_shift, rtol=1e-12)
    lit = sp.tpm[:, :N1].mean(axis=1) - sp.tpm[:, N1:].mean(axis=1)
    np.testing.assert_allclose(sp.mean_shift, lit, rtol=1e-12)
    assert sp.params["stage1"] == "saddlepoint"


def test_it_needs_no_balance_where_gemm_does():
    """The saddlepoint's requirement is that the statistic be a subset sum,
    which it is at every geometry; ``gemm``'s guard is about the quadrature
    identity, not about the saddlepoint."""
    rng = np.random.default_rng(19)
    n1, n0 = 30, 10
    cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
    counts = rng.poisson(60, size=(30, n1 + n0)).astype(float)
    kw = dict(lib_sizes=np.ones(n1 + n0), nperms=400, seed=1)
    res = wade.wade(counts, np.ones(30), cond, stage1="saddlepoint", **kw)
    assert np.all(np.isfinite(res.p_mean_shift))
    with pytest.raises(ValueError, match="balanced"):
        wade.wade(counts, np.ones(30), cond, stage1="gemm", **kw)
    with pytest.raises(ValueError, match="'grid', 'gemm' or 'saddlepoint'"):
        wade.wade(counts, np.ones(30), cond, stage1="fast", **kw)


def test_it_resolves_past_the_permutation_floor():
    """The whole point. The empirical p-value stops at ``1/(B+1)`` and the GPD
    refinement at ``1/(B·n_tail)``; the saddlepoint has no floor above
    ``1/C(n, n1)``, which is what lets BH decide on a 20,000-gene cohort
    (``scaling.md`` §4.9)."""
    rng = np.random.default_rng(23)
    counts = rng.poisson(80, size=(30, N1 + N0)).astype(float)
    counts[:5, :N1] *= 3.0
    kw = dict(lib_sizes=np.ones(N1 + N0), nperms=1000, seed=1)
    grid = wade.wade(counts, np.ones(30), COND, **kw)
    sp = wade.wade(counts, np.ones(30), COND, stage1="saddlepoint", **kw)
    assert grid.p_mean_shift.min() >= 1.0 / (1000 * 250) - 1e-18
    assert sp.p_mean_shift.min() < grid.p_mean_shift.min() / 100
    # stage 2 is untouched by the stage-1 path
    np.testing.assert_array_equal(grid.p_subset, sp.p_subset)
    # and the diagnostic columns keep their meaning
    assert not sp.refined_mean_shift.any()
    assert np.all(np.isfinite(sp.z_mean_shift))


def test_missing_scipy_fails_fast_and_says_what_to_do(monkeypatch):
    """SciPy is the one dependency the statistic does not otherwise have, and
    this is the only path that needs it. A consumer who installed the wheel
    (NumPy only) and opted in used to get a bare ``ModuleNotFoundError`` from
    inside the tail computation — after the permutation loop had already run,
    which on a real cohort is minutes of work thrown away. The check now sits
    in ``_validate_stage1``, so it costs nothing and fires before any work."""
    import importlib.util

    real = importlib.util.find_spec

    def no_scipy(name, *a, **kw):
        return None if name == "scipy" else real(name, *a, **kw)

    monkeypatch.setattr(importlib.util, "find_spec", no_scipy)
    counts = _cohort(seed=0)
    with pytest.raises(ImportError, match="needs SciPy"):
        wade.wade(counts, np.ones(len(counts)), COND, nperms=10, seed=1,
                  lib_sizes=np.ones(len(COND)), stage1="saddlepoint")
    # and the default path is untouched by SciPy's absence
    wade.wade(counts, np.ones(len(counts)), COND, nperms=10, seed=1,
              lib_sizes=np.ones(len(COND)))
