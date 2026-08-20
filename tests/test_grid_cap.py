"""The quantile-grid cap (``max_probs``) — ``docs/method.md`` §1, ``docs/scaling.md`` §2.1.

The cap is the one performance change that is *allowed* to change an answer,
so what it changes is pinned here explicitly:

* nothing at all when the cap does not bind (the common case — every design
  at or under 2,000 per group);
* ``affected_fraction`` keeps its value to well within its own noise when the
  cap is respected (``m >= 2.5 / smallest fraction of interest``);
* ``mean_shift`` inflates one-sidedly for concentrated signals, which is a
  quadrature effect the null inherits, so inference is unaffected;
* the realized grid is recorded in the result and the manifest, never applied
  silently.

The full-scale fidelity table (n = 4,000 per group) is in ``docs/scaling.md``
§2.1; these tests assert the same behaviour at suite-friendly sizes.
"""

from __future__ import annotations

import numpy as np
import pytest

import wade
from wade.permutation import HAVE_RUST_KERNEL, draw_perms, null_statistics, subset_null_backend
from wade.quantiles import capped_nprobs, probability_grid
from wade.stats import wade_stats
from wade.subset import bridge, log_ratio_curve, subset_test
from wade.quantiles import type7_quantiles

G, N = 12, 300                      # per group; the cap binds at 100
COND = np.r_[np.ones(N, int), np.zeros(N, int)]


def _counts(seed=0, subset_frac=0.0, fold=1.0):
    rng = np.random.default_rng(seed)
    lam = rng.gamma(10.0, 5.0, size=(G, 1))
    c = rng.poisson(lam, size=(G, 2 * N)).astype(float)
    if fold != 1.0:
        k = round(subset_frac * N) if subset_frac else N
        for i in range(G):
            c[i, :k] = rng.poisson(lam[i, 0] * fold, k)
    return c


# ---------------------------------------------------------------------------
# The helper


def test_capped_nprobs():
    assert capped_nprobs(300, 200, None) == 200
    assert capped_nprobs(300, 200, 2000) == 200
    assert capped_nprobs(300, 200, 100) == 100
    with pytest.raises(ValueError, match="at least 2"):
        capped_nprobs(300, 200, 1)


# ---------------------------------------------------------------------------
# When the cap does not bind, nothing changes — bitwise


def test_uncapped_and_default_cap_are_bitwise_identical():
    counts = _counts()
    a = wade.wade(counts, np.ones(G), COND, nperms=50, seed=1, max_probs=None)
    b = wade.wade(counts, np.ones(G), COND, nperms=50, seed=1)   # default 2000
    for name in ("mean_shift", "p_mean_shift", "p_subset", "affected_fraction",
                 "direction", "log2_fc"):
        av, bv = getattr(a, name), getattr(b, name)
        np.testing.assert_array_equal(av, bv, err_msg=name)
    assert a.nprobs == b.nprobs == N


# ---------------------------------------------------------------------------
# The capped grid is the same statistic on a coarser quadrature, both backends


@pytest.mark.kernel
@pytest.mark.skipif(not HAVE_RUST_KERNEL, reason="kernel not built")
def test_mean_shift_null_backends_agree_on_a_capped_grid():
    counts = _counts(1)
    x = counts + np.random.default_rng(9).uniform(0, 0.01, counts.shape)
    perms = draw_perms(COND, 40, seed=3)
    a = null_statistics(x, perms, backend="numpy", max_probs=100)
    b = null_statistics(x, perms, backend="rust", max_probs=100)
    np.testing.assert_allclose(a, b, rtol=1e-15)


@pytest.mark.kernel
@pytest.mark.skipif(not HAVE_RUST_KERNEL, reason="kernel not built")
def test_subset_backends_agree_on_a_capped_grid():
    counts = _counts(2)
    x = counts + np.random.default_rng(10).uniform(0, 0.01, counts.shape)
    perms = draw_perms(COND, 30, seed=4)
    q = probability_grid(100)
    r = log_ratio_curve(type7_quantiles(x[:, :N], q), type7_quantiles(x[:, N:], q))
    b_obs = bridge(r)
    out_np = subset_null_backend(x, b_obs, perms, q, backend="numpy")
    out_rs = subset_null_backend(x, b_obs, perms, q, backend="rust")
    for a, b, name in zip(out_np, out_rs, ("statistic", "null", "mu", "sd", "argmax")):
        np.testing.assert_allclose(a, b, rtol=1e-15, err_msg=name)


# ---------------------------------------------------------------------------
# What the cap is allowed to change, and by how much


def test_affected_fraction_is_faithful_under_the_cap():
    """The rule (scaling.md §2.1): m ~ 2.5 / smallest fraction of interest.
    At m = 100 a global change and a 10% subset are both well inside that."""
    q_full, q_cap = probability_grid(N), probability_grid(100)
    for frac, fold, tol in ((0.0, 2.0, 0.02), (0.10, 8.0, 0.02)):
        counts = _counts(3, subset_frac=frac, fold=fold)
        x = counts + np.random.default_rng(11).uniform(0, 0.01, counts.shape) + 1.0
        aff = {}
        for tag, q in (("full", q_full), ("cap", q_cap)):
            r = log_ratio_curve(type7_quantiles(x[:, :N], q), type7_quantiles(x[:, N:], q))
            aff[tag] = wade.affected_fraction(r)
        worst = float(np.max(np.abs(aff["full"] - aff["cap"])))
        assert worst < tol, f"frac={frac}: affected_fraction moved {worst:.4f} under the cap"


def test_mean_shift_inflation_under_the_cap_is_one_sided():
    """A coarse uniform grid gives the extreme node weight 1/m while its value
    is huge, so the quadrature inflates |mean_shift| for concentrated signals
    — never deflates it (scaling.md §2.1). Inference is unaffected: the null
    inherits the same quadrature."""
    counts = _counts(4, subset_frac=0.05, fold=8.0)
    x = counts + np.random.default_rng(12).uniform(0, 0.01, counts.shape)
    full = wade_stats(x, COND).mean_shift
    cap = wade_stats(x, COND, max_probs=50).mean_shift
    assert np.all(np.abs(cap) >= np.abs(full) - 1e-9)


def test_capped_run_records_the_grid_everywhere():
    counts = _counts(5)
    res = wade.wade(counts, np.ones(G), COND, nperms=50, seed=1, max_probs=100)
    assert res.nprobs == 100
    assert res.params["max_probs"] == 100 and res.params["nprobs"] == 100
    man = wade.manifest(res)
    assert man["design"]["nprobs"] == 100 and man["design"]["max_probs"] == 100
    # every per-gene curve is on the capped grid
    assert res.stats.Q1.shape == (G, 100)
    assert res.subset.r.shape == (G, 100)
    d = res.gene_detail(0)
    assert d.nprobs == 100
    np.testing.assert_array_equal(d.r[::-1], res.subset.r[0])


def test_stage2_level_holds_on_a_genuine_global_shift_under_the_cap():
    """The §6 standing question: does the cap interact with the thinning?
    Genuine global 2x shifts, unit libraries, capped grid — the false-subset
    rate must stay at its nominal level (uncapped: 0.03-0.06, method.md §10.3)."""
    rng = np.random.default_rng(6)
    reps = 60
    case = rng.poisson(rng.gamma(10, 2 * 5.0, (reps, N))).astype(float)
    ctrl = rng.poisson(rng.gamma(10, 5.0, (reps, N))).astype(float)
    counts = np.c_[case, ctrl]
    res = wade.wade(counts, np.ones(reps), COND, lib_sizes=np.ones(2 * N),
                    nperms=200, seed=2, max_probs=100)
    rate = float((res.p_subset < 0.05).mean())
    assert rate <= 0.15, f"false-subset rate under the cap: {rate:.2f}"
    assert np.median(res.affected_fraction) >= 0.85


def test_bootstrap_intervals_run_on_the_capped_grid():
    counts = _counts(7)
    res = wade.wade(counts, np.ones(G), COND, nperms=20, seed=1,
                    max_probs=100, n_boot=15)
    assert res.ci_affected_fraction.shape == (2, G)
    assert np.all(res.ci_affected_fraction[0] <= res.ci_affected_fraction[1] + 1e-12)
