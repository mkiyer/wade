"""The subset-test kernel, validated elementwise against the NumPy path.

Same discipline as ``test_kernel.py``: the NumPy loop in
``wade.permutation._subset_null_numpy`` was written first and validated on
its own terms (``test_subset.py``), so it is the baseline, and a disagreement
between it and the kernel has exactly one candidate cause.

The kernel is designed to be **bitwise** identical, not merely close: it
mirrors the NumPy path's arithmetic term for term — R's type-7 guard, two
``log2`` calls then a subtraction, a sequential cumulative sum, the ``k/m``
division before the multiplication, sequential accumulation of the moments
over permutations, and the ``±0.0`` an excluded width contributes to the
maximum. What it changes is only *how often* things are computed (one sort
per gene, ``log2`` only where type 7 interpolates), never *what*. The
tolerance below is ``1e-15`` as in ``test_kernel.py``; measured, every
comparison comes out at exactly zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import assert_close

import wade
from wade.permutation import HAVE_RUST_KERNEL, subset_null_backend
from wade.quantiles import probability_grid, type7_quantiles
from wade.stats import split_groups
from wade.subset import bridge, log_ratio_curve, subset_test

pytestmark = pytest.mark.kernel

LAYER = "subset kernel"
TOL = 1e-15


def test_the_kernel_was_actually_built():
    """Fail loudly rather than skipping into a false pass."""
    assert HAVE_RUST_KERNEL, (
        "the Rust kernel is not built. Run `pip install -e .` in an environment "
        "with cargo and rustc on PATH, or deselect with `-m 'not kernel'`."
    )
    from wade import _kernel

    assert hasattr(_kernel, "subset_null"), (
        "the kernel is built but predates the subset test; rebuild with `pip install -e .`"
    )


# ---------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------

def _matrix(rng, style: int, g: int, n: int) -> np.ndarray:
    """Three data regimes, all strictly positive.

    The tie-heavy and zero-heavy regimes go through ``tpm_like`` so the
    continuity jitter breaks the ties the way production does; what reaches
    the kernel is what reaches it from ``wade()``.
    """
    if style == 0:
        return rng.lognormal(3, 1, size=(g, n))
    if style == 1:
        counts = rng.integers(0, 4, size=(g, n)).astype(float)        # heavy ties
    else:
        counts = rng.integers(0, 2, size=(g, n)) * rng.poisson(30, size=(g, n))
        counts = counts.astype(float)                                  # zero-heavy
    return wade.tpm_like(counts, np.ones(g), seed=int(rng.integers(1 << 30)))


def _kernel_inputs(x, cond):
    """What ``subset_test`` hands to ``subset_null_backend``."""
    i1, i0 = split_groups(cond)
    m = min(i1.size, i0.size)
    q = probability_grid(m)
    r = log_ratio_curve(type7_quantiles(x[:, i1], q), type7_quantiles(x[:, i0], q))
    b_obs = bridge(r)
    shift = 2.0 ** np.median(r, axis=1)
    xs = x.copy()
    xs[:, i1] /= shift[:, None]
    return xs, b_obs, q


def _both(x, cond, perms, alternative):
    xs, b_obs, q = _kernel_inputs(x, cond)
    rs = subset_null_backend(xs, b_obs, perms, q, alternative=alternative, backend="rust")
    np_ = subset_null_backend(xs, b_obs, perms, q, alternative=alternative, backend="numpy")
    return rs, np_


def _assert_same(rs, np_, label):
    names = ("statistic", "null", "mu", "sd")
    for name, got, want in zip(names, rs[:4], np_[:4]):
        assert np.asarray(got).shape == np.asarray(want).shape, name
        assert_close(got, want, TOL, f"{label}: {name}", LAYER)
    got_k, want_k = rs[4], np_[4]
    assert got_k.dtype.kind == "i", "argmax_k must be an integer array"
    assert np.array_equal(got_k, want_k), (
        f"{label}: argmax_k differs at {np.flatnonzero(got_k != want_k).tolist()}"
    )


# ---------------------------------------------------------------------
# 1. Against the NumPy baseline
# ---------------------------------------------------------------------

@pytest.mark.parametrize("alternative", wade.ALTERNATIVES)
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_kernel_agrees_with_numpy_on_random_inputs(seed, alternative):
    """Sweep group sizes, gene counts, permutation counts and data regimes.

    Balanced and unbalanced designs both matter: on a balanced design the
    grid nodes land on the smaller group's order statistics and the kernel
    reads precomputed logs almost everywhere, while on an unbalanced one the
    larger group interpolates at most nodes and ``log2`` is called on the
    interpolated value. Both branches have to match.
    """
    rng = np.random.default_rng(seed)
    n1 = int(rng.integers(3, 26))
    n0 = n1 if seed % 2 == 0 else int(rng.integers(3, 26))
    g = int(rng.integers(1, 41))
    B = int(rng.integers(10, 61))
    x = _matrix(rng, seed % 3, g, n1 + n0)
    cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
    perms = wade.draw_perms(cond, B, seed=seed + 100)

    rs, np_ = _both(x, cond, perms, alternative)
    _assert_same(rs, np_, f"random {seed} {alternative} (g={g}, {n1}v{n0}, B={B})")


def test_kernel_agrees_with_numpy_with_many_genes_per_thread():
    """Enough genes that every rayon worker handles a run of them and reuses
    its scratch across genes — the regime where stale per-gene state would
    show, and which the small random cases never reach."""
    rng = np.random.default_rng(7)
    g, n1, n0, B = 200, 50, 50, 200
    x = rng.lognormal(3, 0.6, size=(g, n1 + n0))
    sub = rng.choice(g, 20, replace=False)
    for gi in sub:                                   # plant some subset genes
        idx = rng.choice(n1, 5, replace=False)
        x[gi, idx] *= 8.0
    cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
    perms = wade.draw_perms(cond, B, seed=8)

    rs, np_ = _both(x, cond, perms, "two-sided")
    _assert_same(rs, np_, "200 genes, 50v50, B=200")


def test_recomputing_pass_two_matches_rereading_it():
    """Pass 2 normally re-reads the bridges pass 1 stored; above a per-thread
    memory cap it recomputes them instead. The two must be the same bits, and
    the cap is only reachable at sizes no unit test should run, so the kernel
    exposes it and this forces the recompute path on a small case."""
    from wade import _kernel

    rng = np.random.default_rng(17)
    n1, n0, g, B = 12, 15, 30, 70
    x = _matrix(rng, 2, g, n1 + n0)
    cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
    perms = np.ascontiguousarray(wade.draw_perms(cond, B, seed=18), dtype=np.int64)
    xs, b_obs, q = _kernel_inputs(x, cond)
    xs = np.ascontiguousarray(xs)
    for alt in wade.ALTERNATIVES:
        stored = _kernel.subset_null(xs, b_obs, perms, q, alt)
        recomputed = _kernel.subset_null(xs, b_obs, perms, q, alt, 0)
        _assert_same(recomputed, stored, f"recompute vs store {alt}")
        _assert_same(recomputed, subset_null_backend(xs, b_obs, perms, q,
                                                     alternative=alt, backend="numpy"),
                     f"recompute vs numpy {alt}")


def test_subset_test_reaches_the_same_result_through_either_backend():
    """The public function, not just the dispatch."""
    rng = np.random.default_rng(9)
    n1, n0, g, B = 30, 20, 25, 80
    x = rng.lognormal(3, 0.8, size=(g, n1 + n0))
    cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
    perms = wade.draw_perms(cond, B, seed=10)
    for alt in wade.ALTERNATIVES:
        a = subset_test(x, cond, perms, alternative=alt, backend="rust")
        b = subset_test(x, cond, perms, alternative=alt, backend="numpy")
        assert_close(a.statistic, b.statistic, TOL, f"subset_test {alt}: statistic", LAYER)
        assert_close(a.null, b.null, TOL, f"subset_test {alt}: null", LAYER)
        assert np.array_equal(a.argmax_k, b.argmax_k)
        # The observed-curve quantities never touch the kernel.
        assert np.array_equal(a.affected_fraction, b.affected_fraction)
        assert np.array_equal(a.direction, b.direction)
        assert np.array_equal(a.shift, b.shift)


# ---------------------------------------------------------------------
# 2. End to end
# ---------------------------------------------------------------------

def test_wade_gives_identical_subset_pvalues_on_either_backend():
    """``p_subset`` is the real check; ``affected_fraction`` and ``direction``
    are computed on the observed curve in Python, so identity there is
    trivial and asserted only to pin that they stay that way."""
    rng = np.random.default_rng(21)
    g, n1, n0 = 60, 40, 40
    counts = rng.poisson(25, size=(g, n1 + n0)).astype(float)
    counts[:5, :4] *= 20                                       # 10% of cases, up
    cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
    kw = dict(nperms=150, seed=3, keep_null=True)
    a = wade.wade(counts, np.ones(g), cond, backend="rust", **kw)
    b = wade.wade(counts, np.ones(g), cond, backend="numpy", **kw)

    assert_close(a.p_subset, b.p_subset, TOL, "wade: p_subset", LAYER)
    assert_close(a.padj_subset, b.padj_subset, TOL, "wade: padj_subset", LAYER)
    assert_close(a.subset.statistic, b.subset.statistic, TOL, "wade: subset statistic", LAYER)
    assert_close(a.subset.null, b.subset.null, TOL, "wade: subset null", LAYER)
    assert np.array_equal(a.subset.argmax_k, b.subset.argmax_k)
    assert np.array_equal(a.affected_fraction, b.affected_fraction)
    assert np.array_equal(a.direction, b.direction)
    # And the mean-shift side is unchanged by any of this.
    assert_close(a.p_mean_shift, b.p_mean_shift, TOL, "wade: p_mean_shift", LAYER)


def test_auto_backend_uses_the_kernel_for_the_subset_test(monkeypatch):
    """With the kernel present, ``backend="auto"`` must not fall through to
    the NumPy loop. Asserted by making the NumPy loop impossible to run."""
    import wade.permutation as perm

    def boom(*a, **k):
        raise AssertionError("the NumPy subset loop ran under backend='auto'")

    monkeypatch.setattr(perm, "_subset_null_numpy", boom)
    rng = np.random.default_rng(5)
    g, n1, n0 = 8, 10, 12
    counts = rng.poisson(20, size=(g, n1 + n0)).astype(float)
    cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
    res = wade.wade(counts, np.ones(g), cond, nperms=30, backend="auto")
    assert res.p_subset is not None and np.all(np.isfinite(res.p_subset))

    # And the explicit NumPy request still reaches the NumPy loop.
    with pytest.raises(AssertionError, match="NumPy subset loop"):
        wade.wade(counts, np.ones(g), cond, nperms=30, backend="numpy")


# ---------------------------------------------------------------------
# 3. The kernel's own input validation
# ---------------------------------------------------------------------

def _valid_inputs():
    rng = np.random.default_rng(0)
    n1, n0, g, B = 5, 7, 4, 9
    x = rng.lognormal(3, 1, size=(g, n1 + n0))
    cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
    perms = np.ascontiguousarray(wade.draw_perms(cond, B, seed=1), dtype=np.int64)
    xs, b_obs, q = _kernel_inputs(x, cond)
    return np.ascontiguousarray(xs), b_obs, perms, q


def test_kernel_rejects_a_non_positive_matrix():
    """The curve is a difference of logarithms; the NumPy path raises on a
    non-positive quantile and the kernel must not silently emit -inf."""
    from wade import _kernel

    xs, b_obs, perms, q = _valid_inputs()
    xs[1, 3] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        _kernel.subset_null(xs, b_obs, perms, q, "two-sided")
    xs[1, 3] = -2.0
    with pytest.raises(ValueError, match="strictly positive"):
        _kernel.subset_null(xs, b_obs, perms, q, "two-sided")


def test_kernel_rejects_non_finite_input():
    from wade import _kernel

    xs, b_obs, perms, q = _valid_inputs()
    xs[2, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _kernel.subset_null(xs, b_obs, perms, q, "two-sided")
    xs[2, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        _kernel.subset_null(xs, b_obs, perms, q, "two-sided")


def test_kernel_rejects_a_mismatched_permutation_matrix():
    from wade import _kernel

    xs, b_obs, perms, q = _valid_inputs()
    with pytest.raises(ValueError, match="columns"):
        _kernel.subset_null(xs, b_obs, perms[:, :-1].copy(), q, "two-sided")
    bad = perms.copy()
    bad[3, 0] = 2
    with pytest.raises(ValueError, match="only 0 and 1"):
        _kernel.subset_null(xs, b_obs, bad, q, "two-sided")
    swapped = perms.copy()
    swapped[2] = 1 - swapped[2]                       # sizes (7, 5) instead of (5, 7)
    with pytest.raises(ValueError, match="group sizes"):
        _kernel.subset_null(xs, b_obs, swapped, q, "two-sided")
    with pytest.raises(ValueError, match="min\\(n1, n0\\)"):
        _kernel.subset_null(xs, b_obs[:, :3], perms, probability_grid(3), "two-sided")
    with pytest.raises(ValueError, match="at least one row"):
        _kernel.subset_null(xs, b_obs, perms[:0], q, "two-sided")


def test_kernel_rejects_a_mismatched_observed_bridge():
    from wade import _kernel

    xs, b_obs, perms, q = _valid_inputs()
    with pytest.raises(ValueError, match="b_obs must have shape"):
        _kernel.subset_null(xs, b_obs[:-1], perms, q, "two-sided")


def test_kernel_rejects_a_bad_alternative():
    from wade import _kernel

    xs, b_obs, perms, q = _valid_inputs()
    with pytest.raises(ValueError, match="alternative must be one of"):
        _kernel.subset_null(xs, b_obs, perms, q, "both")
    # The Python dispatch validates first, with the same message.
    with pytest.raises(ValueError, match="alternative must be one of"):
        subset_null_backend(xs, b_obs, perms, q, alternative="both", backend="rust")


def test_backend_selection_is_explicit_and_validated():
    xs, b_obs, perms, q = _valid_inputs()
    with pytest.raises(ValueError, match="backend must be"):
        subset_null_backend(xs, b_obs, perms, q, backend="fortran")
