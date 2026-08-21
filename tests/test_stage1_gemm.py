"""Stage 1 as one matrix product (``stage1="gemm"``) — ``docs/scaling.md`` §3.1.

On a balanced design ``mean_shift`` computed through the quantile grid is the
difference of the two group means, exactly (measured 2.1e-13 relative). The
GEMM path computes that difference outright — the observed statistic as
``x @ w`` and the whole null as ``x @ W`` — so it is the *same test* on a
faster arithmetic path, agreeing to ~1e-9 rather than bitwise because BLAS
reassociates. It is opt-in beside the parity-pinned kernel, and refuses
unbalanced designs, where the identity does not hold.
"""

from __future__ import annotations

import numpy as np
import pytest

import wade
from wade.permutation import draw_perms, mean_diff_null, mean_diff_stat, null_statistics

G, N = 40, 150
COND = np.r_[np.ones(N, int), np.zeros(N, int)]


def _counts(seed=0):
    rng = np.random.default_rng(seed)
    lam = rng.gamma(10.0, 5.0, size=(G, 1))
    c = rng.poisson(lam, size=(G, 2 * N)).astype(float)
    c[:5] = rng.poisson(lam[:5] * 2, size=(5, 2 * N))[:, ::-1]  # a few shifted genes
    return c


def test_gemm_statistic_is_the_exact_mean_difference():
    x = _counts() + np.random.default_rng(1).uniform(0, 0.01, (G, 2 * N))
    stat = mean_diff_stat(x, COND)
    literal = x[:, :N].mean(axis=1) - x[:, N:].mean(axis=1)
    np.testing.assert_allclose(stat, literal, rtol=1e-12)
    # ... and the grid quadrature is the same number on a balanced, uncapped design
    grid = wade.wade_stats(x, COND).mean_shift
    np.testing.assert_allclose(stat, grid, rtol=1e-9)


def test_gemm_null_matches_the_kernel_null():
    x = _counts(2) + np.random.default_rng(2).uniform(0, 0.01, (G, 2 * N))
    perms = draw_perms(COND, 100, seed=3)
    gemm = mean_diff_null(x, perms)
    grid = null_statistics(x, perms)
    np.testing.assert_allclose(gemm, grid, rtol=1e-9)


def test_gemm_run_reports_the_statistic_it_tested():
    counts = _counts(3)
    a = wade.wade(counts, np.ones(G), COND, nperms=100, seed=1)
    b = wade.wade(counts, np.ones(G), COND, nperms=100, seed=1, stage1="gemm")
    # One statistic end to end: the reported column is the GEMM value...
    lit = b.tpm[:, :N].mean(axis=1) - b.tpm[:, N:].mean(axis=1)
    np.testing.assert_allclose(b.mean_shift, lit, rtol=1e-12)
    # ... the grid quadrature stays available and essentially equal here ...
    np.testing.assert_allclose(b.stats.mean_shift, b.mean_shift, rtol=1e-9)
    # ... and inference agrees with the grid path: same exceedance counts
    # (1e-9 relative cannot flip a comparison on continuous data), so the
    # p-values differ only through the GPD's smooth tail fit.
    np.testing.assert_array_equal(a.nexc_mean_shift, b.nexc_mean_shift)
    np.testing.assert_allclose(a.p_mean_shift, b.p_mean_shift, rtol=0.02)
    # stage 2 is untouched by the stage-1 path
    np.testing.assert_array_equal(a.p_subset, b.p_subset)
    assert b.params["stage1"] == "gemm"


def test_gemm_refuses_an_unbalanced_design():
    counts = _counts(4)[:, :-10]
    cond = np.r_[np.ones(N, int), np.zeros(N - 10, int)]
    with pytest.raises(ValueError, match="balanced"):
        wade.wade(counts, np.ones(G), cond, nperms=20, stage1="gemm")
    with pytest.raises(ValueError, match="'grid' or 'gemm'"):
        wade.wade(counts[:, : 2 * N - 10], np.ones(G), cond, nperms=20, stage1="fast")


def test_gemm_is_exact_under_the_grid_cap():
    """Under a cap the grid quadrature drifts from the mean difference; the
    GEMM statistic does not — that is its point at scale."""
    counts = _counts(5)
    res = wade.wade(counts, np.ones(G), COND, nperms=50, seed=1,
                    max_probs=40, stage1="gemm")
    lit = res.tpm[:, :N].mean(axis=1) - res.tpm[:, N:].mean(axis=1)
    np.testing.assert_allclose(res.mean_shift, lit, rtol=1e-12)
    assert res.nprobs == 40


def test_gemm_chunked_agrees_with_unchunked():
    """BLAS blocking depends on the matrix shape, so gemm mode is the one
    place chunking is allclose rather than bitwise — documented in §3.1."""
    counts = _counts(6)
    a = wade.wade(counts, np.ones(G), COND, nperms=50, seed=2, stage1="gemm")
    b = wade.wade(counts, np.ones(G), COND, nperms=50, seed=2, stage1="gemm",
                  gene_chunk=7)
    np.testing.assert_allclose(a.mean_shift, b.mean_shift, rtol=1e-12)
    np.testing.assert_array_equal(a.nexc_mean_shift, b.nexc_mean_shift)
    np.testing.assert_allclose(a.p_mean_shift, b.p_mean_shift, rtol=0.02)
    np.testing.assert_array_equal(a.p_subset, b.p_subset)


def test_gemm_without_the_subset_stage():
    """stage1='gemm' is independent of stage 2, so it works with it switched
    off — which is the cheap configuration for a stage-1-only screen."""
    counts = _counts(7)
    a = wade.wade(counts, np.ones(G), COND, nperms=50, seed=1,
                  stage1="gemm", subset=False)
    lit = a.tpm[:, :N].mean(axis=1) - a.tpm[:, N:].mean(axis=1)
    np.testing.assert_allclose(a.mean_shift, lit, rtol=1e-12)
    assert a.subset is None


def test_gemm_level_holds_on_null_data():
    rng = np.random.default_rng(8)
    counts = rng.poisson(rng.gamma(10, 5, (60, 1)), (60, 2 * N)).astype(float)
    res = wade.wade(counts, np.ones(60), COND, nperms=200, seed=3, stage1="gemm")
    rate = float((res.p_mean_shift < 0.05).mean())
    assert rate <= 0.15, f"null rejection rate under gemm: {rate:.2f}"
