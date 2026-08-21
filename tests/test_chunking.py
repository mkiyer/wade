"""Gene chunking (``gene_chunk``) — ``docs/scaling.md`` §2.2.

Chunking is a memory layout, never a numerical choice: **a chunked run is
bit-identical to the unchunked run**, whatever the chunk size, including one
that does not divide the gene count. That is the whole contract, so it is
asserted with ``assert_array_equal`` — exact equality, not tolerance — across
every output surface: both stages' statistics and p-values, the curves, the
fitted fold change, the thinned null's inputs, the bootstrap intervals, and
the normalized matrix itself.

Why it can be exact (and the two places it almost was not): the jitter is
drawn once and indexed; library sizes are one full-matrix pass (a column
sum's association depends on blocking); and the thinning fit and draw consume
their random streams in gene order across chunks — see
``wade.thinning.fit_fold_change`` and ``wade.thinning.thin_counts``.
"""

from __future__ import annotations

import numpy as np
import pytest

import wade
from wade.thinning import gene_chunks

G, N = 23, 40                       # deliberately not a multiple of any chunk size
COND = np.r_[np.ones(N, int), np.zeros(N, int)]


def _counts(seed=0):
    rng = np.random.default_rng(seed)
    lam = rng.gamma(10.0, 5.0, size=(G, 1))
    c = rng.poisson(lam, size=(G, 2 * N)).astype(float)
    c[2] = 0.0                       # an all-zero gene
    c[5, :N] = 0.0                   # a gene absent from the cases
    return c


EVERY_ARRAY = ("mean_shift", "w1", "fc", "log2_fc", "case_mean", "ctrl_mean",
               "p_mean_shift", "padj_mean_shift", "nexc_mean_shift",
               "p_subset", "padj_subset", "affected_fraction", "direction",
               "z_mean_shift", "z_subset", "subset_log2_fc",
               "tpm", "jitter", "pseudocount")


def _assert_identical(a, b):
    for name in EVERY_ARRAY:
        av, bv = getattr(a, name), getattr(b, name)
        if av is None:
            assert bv is None, name
            continue
        np.testing.assert_array_equal(av, bv, err_msg=name)
    for name in ("Q1", "Q0", "D"):
        np.testing.assert_array_equal(getattr(a.stats, name), getattr(b.stats, name),
                                      err_msg=f"stats.{name}")
    if a.subset is not None:
        for name in ("statistic", "r", "b", "r_test", "shift", "argmax_k"):
            np.testing.assert_array_equal(getattr(a.subset, name),
                                          getattr(b.subset, name),
                                          err_msg=f"subset.{name}")
        assert a.subset.correction == b.subset.correction
    else:
        assert b.subset is None
    for name in ("ci_affected_fraction", "ci_direction", "ci_log2_fc"):
        av, bv = getattr(a, name), getattr(b, name)
        if av is None:
            assert bv is None, name
        else:
            np.testing.assert_array_equal(av, bv, err_msg=name)
    np.testing.assert_array_equal(a.perms, b.perms)
    assert a.nprobs == b.nprobs


def test_gene_chunks_cover_exactly():
    chunks = gene_chunks(23, 7)
    covered = np.concatenate([np.arange(s.start, s.stop) for s in chunks])
    np.testing.assert_array_equal(covered, np.arange(23))
    assert gene_chunks(23, None) == [slice(0, 23)]
    # A single-row chunk changes NumPy's reduction path (last-ulp drift in
    # D.sum), so it can never be produced: a trailing 1-row remainder is
    # folded into the last chunk, and gene_chunk=1 is refused outright.
    assert gene_chunks(23, 11) == [slice(0, 11), slice(11, 23)]
    assert all(s.stop - s.start >= 2 for s in gene_chunks(101, 10))
    with pytest.raises(ValueError, match="at least 2"):
        gene_chunks(23, 1)
    with pytest.raises(ValueError, match="at least 2"):
        gene_chunks(23, 0)


@pytest.mark.parametrize("chunk", [2, 5, 7, 11, 23, 1000])
def test_chunked_run_is_bitwise_identical(chunk):
    counts = _counts()
    norm = np.random.default_rng(3).uniform(0.5, 2.0, G)
    full = wade.wade(counts, norm, COND, nperms=60, seed=1)
    part = wade.wade(counts, norm, COND, nperms=60, seed=1, gene_chunk=chunk)
    _assert_identical(full, part)
    assert part.params["gene_chunk"] == chunk


def test_chunked_run_is_bitwise_identical_with_bootstrap_and_cap():
    counts = _counts(1)
    full = wade.wade(counts, np.ones(G), COND, nperms=40, seed=2,
                     n_boot=25, max_probs=17)
    part = wade.wade(counts, np.ones(G), COND, nperms=40, seed=2,
                     n_boot=25, max_probs=17, gene_chunk=5)
    _assert_identical(full, part)
    assert full.nprobs == 17


def test_chunked_run_is_bitwise_identical_under_division_correction():
    counts = _counts(2)
    full = wade.wade(counts, np.ones(G), COND, nperms=40, seed=3, thin=False)
    part = wade.wade(counts, np.ones(G), COND, nperms=40, seed=3, thin=False,
                     gene_chunk=6)
    assert full.subset.correction == "division"
    _assert_identical(full, part)


def test_chunked_run_is_bitwise_identical_with_no_pseudocount_and_one_sided():
    counts = _counts(4)
    full = wade.wade(counts, np.ones(G), COND, nperms=40, seed=4,
                     pseudocount=0.0, alternative="greater")
    part = wade.wade(counts, np.ones(G), COND, nperms=40, seed=4,
                     pseudocount=0.0, alternative="greater", gene_chunk=9)
    _assert_identical(full, part)


def test_chunked_driver_matches_unchunked_on_the_edges_review_found():
    """Three parities the 2026-08-20 review caught the chunked driver missing:
    a zero-gene matrix, a mis-shaped supplied jitter, and near-int32-boundary
    counts (the thinned buffer's dtype guard must test the *rounded* max)."""
    cond = COND
    empty = np.empty((0, 2 * N))
    a = wade.wade(empty, np.empty(0), cond, nperms=10, seed=1)
    b = wade.wade(empty, np.empty(0), cond, nperms=10, seed=1, gene_chunk=5)
    assert a.gene.shape == b.gene.shape == (0,)

    counts = _counts(8)
    bad_jitter = np.zeros((G + 2, 2 * N))
    with pytest.raises(ValueError, match="genes x samples"):
        wade.wade(counts, np.ones(G), cond, nperms=5, jitter=bad_jitter)
    with pytest.raises(ValueError, match="genes x samples"):
        wade.wade(counts, np.ones(G), cond, nperms=5, jitter=bad_jitter, gene_chunk=5)

    big = _counts(9)
    big[0, 0] = 2.0**31 - 0.4                 # rounds past int32; guard must see it
    full = wade.wade(big, np.ones(G), cond, nperms=10, seed=2)
    part = wade.wade(big, np.ones(G), cond, nperms=10, seed=2, gene_chunk=6)
    _assert_identical(full, part)


def test_chunked_run_without_permutations():
    counts = _counts(5)
    full = wade.wade(counts, np.ones(G), COND, nperms=0)
    part = wade.wade(counts, np.ones(G), COND, nperms=0, gene_chunk=4)
    np.testing.assert_array_equal(full.mean_shift, part.mean_shift)
    assert part.subset is None and part.p_subset is None


def test_chunked_fit_and_thin_are_chunk_invariant():
    """The fit and the thinning draw are the two places chunking touches a
    random stream; every chunk size must give the same realization."""
    from wade import normalize as _normalize
    from wade.thinning import fit_fold_change, thin_counts

    counts = np.round(_counts(6))
    norm = np.ones(G)
    lib = _normalize.library_sizes(counts, norm)
    ref_f = fit_fold_change(counts, COND, alpha=lambda rows=slice(None):
                            np.ones((norm[rows].shape[0], lib.shape[0])), seed=9)
    ref_t = thin_counts(counts, COND, ref_f, np.random.default_rng(9))
    for chunk in (2, 4, 10, 23, 99):
        t = thin_counts(counts, COND, ref_f, np.random.default_rng(9), gene_chunk=chunk)
        np.testing.assert_array_equal(ref_t, t, err_msg=f"thin at chunk {chunk}")

    # ... and the same holds for the fit, which is the only fit there is.
    alpha = lambda rows=slice(None): np.broadcast_to(  # noqa: E731
        1.0 / lib[None, :], (norm[rows].shape[0], lib.shape[0])).copy()
    ref_a = fit_fold_change(counts, COND, alpha=alpha, seed=9)
    for chunk in (2, 4, 10, 23, 99):
        f = fit_fold_change(counts, COND, alpha=alpha, seed=9, gene_chunk=chunk)
        np.testing.assert_array_equal(ref_a, f, err_msg=f"fit at chunk {chunk}")


def test_midmean_is_independent_of_row_grouping():
    """The alpha fit reduces per-side row *subsets*, whose sizes depend on the
    chunk boundaries — so its interquartile mean must give a row the same
    value whatever rows share the array, singletons included. This pins the
    NumPy behaviour the fit relies on (fresh C-contiguous reductions are
    layout-independent; F-ordered ones are not, which is why _middle_mean
    forces contiguity)."""
    from wade.thinning import midmean

    rng = np.random.default_rng(0)
    x = rng.gamma(2.0, 50.0, size=(23, 81))          # awkward width, skewed values
    ref = midmean(x)
    for idx in ([3], [0, 7], list(range(5, 17)), [1, 4, 9, 16, 22]):
        np.testing.assert_array_equal(midmean(x[idx]), ref[idx],
                                      err_msg=f"rows {idx}")



