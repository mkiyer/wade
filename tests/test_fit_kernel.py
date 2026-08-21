"""The fold-change fit's compiled bisection (``fit_backend="rust"``) — ``docs/scaling.md`` §3.3.

Unlike the two permutation kernels, this one is **not** bitwise against its
NumPy path: no two binomial samplers consume randomness alike, so the
realized fit for a given seed depends on the backend, and the backend is
therefore opt-in — never auto-dispatched. What is pinned instead:

* determinism given the seed, and invariance to chunking and thread schedule
  (each gene's stream is a function of ``(seed, global gene index)`` alone);
* the same statistical contract the NumPy fit carries: unbiased for an NB
  fold change, blind to a sub-quartile subset, direction-aware, ``f = 1``
  where there is nothing to fit;
* agreement with the NumPy path at the resolution the fit can have — the
  bisection cell plus the thinning's own discreteness.
"""

from __future__ import annotations

import numpy as np
import pytest

import wade
from wade.permutation import HAVE_RUST_KERNEL
from wade.thinning import fit_fold_change

pytestmark = [pytest.mark.kernel,
              pytest.mark.skipif(not HAVE_RUST_KERNEL, reason="kernel not built")]

N1 = N0 = 200
COND = np.r_[np.ones(N1, int), np.zeros(N0, int)]
PHI = 0.1


def _nb(rng, mu, size):
    return rng.poisson(rng.gamma(1 / PHI, PHI * mu, size=size)).astype(float)


def _unit_alpha(g, n):
    return lambda rows=slice(None): np.ones((np.empty(g)[rows].shape[0], n))


def test_kernel_fit_is_deterministic_and_chunk_invariant():
    rng = np.random.default_rng(1)
    counts = np.c_[_nb(rng, 40.0, (60, N1)), _nb(rng, 20.0, (60, N0))]
    alpha = _unit_alpha(60, N1 + N0)
    a = fit_fold_change(counts, COND, alpha=alpha, seed=5, backend="rust")
    b = fit_fold_change(counts, COND, alpha=alpha, seed=5, backend="rust")
    np.testing.assert_array_equal(a, b)
    for chunk in (2, 7, 59, 60):
        c = fit_fold_change(counts, COND, alpha=alpha, seed=5, backend="rust",
                            gene_chunk=chunk)
        np.testing.assert_array_equal(a, c, err_msg=f"chunk {chunk}")
    d = fit_fold_change(counts, COND, alpha=alpha, seed=6, backend="rust")
    assert not np.array_equal(a, d), "a different seed must give a different draw"


def test_kernel_fit_matches_the_numpy_fit_statistically():
    rng = np.random.default_rng(2)
    counts = np.c_[_nb(rng, 2 * 20.0, (100, N1)), _nb(rng, 20.0, (100, N0))]
    alpha = _unit_alpha(100, N1 + N0)
    f_np = fit_fold_change(counts, COND, alpha=alpha, seed=0)
    f_rs = fit_fold_change(counts, COND, alpha=alpha, seed=0, backend="rust")
    assert abs(np.median(f_rs) - 2.0) < 0.1
    assert abs(np.median(f_rs) - np.median(f_np)) < 0.03
    # per gene, the two differ by at most the fit's own resolution
    assert np.max(np.abs(np.log(f_rs / f_np))) < 0.1


def test_kernel_fit_is_blind_to_a_subset_and_direction_aware():
    rng = np.random.default_rng(3)
    case = _nb(rng, 20.0, (100, N1))
    ctrl = _nb(rng, 20.0, (100, N0))
    for i in range(100):
        case[i, rng.choice(N1, 10, replace=False)] = _nb(rng, 160.0, 10)
    f = fit_fold_change(np.c_[case, ctrl], COND, alpha=_unit_alpha(100, N1 + N0),
                        seed=0, backend="rust")
    assert np.median(f) < 1.12

    counts = np.c_[_nb(rng, 10.0, (50, N1)), _nb(rng, 20.0, (50, N0))]
    f = fit_fold_change(counts, COND, alpha=_unit_alpha(50, N1 + N0),
                        seed=0, backend="rust")
    assert np.median(f) == pytest.approx(0.5, abs=0.05)

    zeros = np.zeros((3, N1 + N0))
    f = fit_fold_change(zeros, COND, alpha=_unit_alpha(3, N1 + N0),
                        seed=0, backend="rust")
    np.testing.assert_array_equal(f, 1.0)


def test_kernel_fit_refuses_a_bad_backend():
    counts = np.zeros((3, N1 + N0))
    with pytest.raises(ValueError, match="'numpy' or 'rust'"):
        fit_fold_change(counts, COND, alpha=_unit_alpha(3, N1 + N0), backend="fast")


def test_kernel_fit_tolerates_a_zero_library_size_like_the_numpy_fit():
    """An all-zero (QC-failed) sample gives that column an infinite alpha, so
    its zero counts scale to NaN. The NumPy fit tolerates it (np.partition
    orders NaN last, the discarded top quarter absorbs it); the kernel must
    do the same rather than panic the rayon workers (found in review,
    2026-08-20)."""
    rng = np.random.default_rng(9)
    counts = np.c_[_nb(rng, 40.0, (30, N1)), _nb(rng, 20.0, (30, N0))]
    counts[:, 3] = 0.0                                  # a dead sample
    lens = np.ones(30)
    import wade.normalize as _n
    lib = _n.library_sizes(counts, lens)
    assert lib[3] == 0.0
    from wade.api import _fit_alpha
    alpha = _fit_alpha(lens, lib, 1e6, N1 + N0)
    f_np = fit_fold_change(counts, COND, alpha=alpha, seed=1)
    f_rs = fit_fold_change(counts, COND, alpha=alpha, seed=1, backend="rust")
    assert np.all(np.isfinite(f_np)) and np.all(np.isfinite(f_rs))
    assert abs(np.median(f_rs) - np.median(f_np)) < 0.05
    # and through the entry point, both backends complete
    res = wade.wade(counts, lens, COND, nperms=20, seed=1, fit_backend="rust",
                    allow_empty_samples=True)      # the dead sample is the point
    assert np.all(np.isfinite(res.fitted_fold_change))


def test_fit_backend_is_validated_eagerly():
    counts = np.zeros((3, N1 + N0))
    with pytest.raises(ValueError, match="fit_backend"):
        wade.wade(counts, np.ones(3), COND, nperms=0, subset=False,
                  fit_backend="fast")


def test_wade_with_the_kernel_fit_holds_stage2_level_on_a_global_shift():
    """The whole point of the fit is the subset stage's null; the kernel fit
    must keep it at nominal level on genuine global fold changes."""
    rng = np.random.default_rng(4)
    reps = 60
    counts = np.c_[_nb(rng, 40.0, (reps, N1)), _nb(rng, 20.0, (reps, N0))]
    res = wade.wade(counts, np.ones(reps), COND, lib_sizes=np.ones(N1 + N0),
                    nperms=200, seed=2, fit_backend="rust")
    rate = float((res.p_subset < 0.05).mean())
    assert rate <= 0.15, f"false-subset rate with the kernel fit: {rate:.2f}"
    assert abs(np.median(res.fitted_fold_change) - 2.0) < 0.15
    assert res.params["fit_backend"] == "rust"
    # and the chunked driver gives the identical result (the fit is
    # chunk-invariant, everything else is bit-identical)
    part = wade.wade(counts, np.ones(reps), COND, lib_sizes=np.ones(N1 + N0),
                     nperms=200, seed=2, fit_backend="rust", gene_chunk=17)
    np.testing.assert_array_equal(res.p_subset, part.p_subset)
    np.testing.assert_array_equal(res.fitted_fold_change, part.fitted_fold_change)
