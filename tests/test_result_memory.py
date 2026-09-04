"""The result's memory contract: it carries no matrix it can rebuild.

A ``WadeResult`` used to hold three ``genes x samples`` float64 matrices --
``tpm``, ``jitter`` and ``pseudocount`` -- so a result cost 3.1x the count
matrix it came from, measured, and a caller still holding their own counts
paid 4x. Two of the three are *derived*: the jitter is a seeded draw and the
pseudocount is ``norm_factor / (norm * lib)`` scaled. Both are now rebuilt on
access from things the result already keeps, which is the difference between
19 GB and 60 GB on a 30,000 x 80,000 cohort (``docs/scaling.md`` §2.3).

The contract has two halves and both are load-bearing:

* what comes back must be **bit-identical** to what the run used, or every
  figure and every ``gene_detail`` silently draws a different curve;
* the result must not **store** it, or the rebuild bought nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

import wade

G, N = 40, 24
COND = np.r_[np.ones(N // 2, int), np.zeros(N // 2, int)]


def _counts(seed=0):
    return np.random.default_rng(seed).poisson(60, size=(G, N)).astype(float)


def _run(**kw):
    kw.setdefault("seed", 7)
    return wade.wade(_counts(), np.ones(G), COND, nperms=200, **kw)


def test_the_result_stores_neither_derived_matrix():
    """The point of the exercise. Both slots hold nothing rebuildable."""
    res = _run()
    assert res._jitter is None, "the jitter was stored despite being seeded"
    assert not isinstance(res._pc, np.ndarray), "the pseudocount was materialized"
    stored = sum(v.nbytes for v in vars(res).values()
                 if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape == (G, N))
    assert stored == _counts().nbytes, "more than one genes x samples matrix kept"


def test_the_rebuilt_jitter_is_the_one_the_run_used():
    """Bit-identical, not close. The four RNG streams are spawned from one
    ``SeedSequence``, so regenerating the first reproduces the draw exactly --
    and ``tpm`` was computed from that draw, so anything else would make
    ``gene_detail`` draw a curve the statistics did not come from."""
    res = _run()
    from wade import normalize as _n

    js = np.random.SeedSequence(7).spawn(4)[0]
    expected = _n.draw_jitter((G, N), noise=res.params["noise"],
                              rng=np.random.default_rng(js))
    np.testing.assert_array_equal(res.jitter, expected)
    # and it is stable across accesses, not a fresh draw each time
    np.testing.assert_array_equal(res.jitter, res.jitter)


def test_an_unreproducible_jitter_is_kept_instead():
    """``seed=None`` cannot be regenerated, so the array must be retained.
    Dropping it to save memory would return a *different* jitter on access --
    the one failure mode that would be silent."""
    res = _run(seed=None)
    assert res._jitter is not None
    np.testing.assert_array_equal(res.jitter, res._jitter)


def test_a_caller_supplied_jitter_is_kept_and_returned_unchanged():
    jit = np.random.default_rng(3).uniform(0, 1e-6, size=(G, N))
    res = _run(jitter=jit)
    np.testing.assert_array_equal(res.jitter, jit)


def test_the_rebuilt_pseudocount_matches_the_row_helper():
    """``pseudocount_row`` exists so ``gene_detail`` does not rebuild a whole
    matrix for one gene; it must agree with the matrix it avoids building."""
    res = _run()
    pc = res.pseudocount
    assert pc.shape == (G, N)
    for i in (0, G // 2, G - 1):
        np.testing.assert_allclose(res.pseudocount_row(i), pc[i], rtol=0, atol=0)


def test_the_pseudocount_survives_a_matrix_normalizer():
    """A matrix normalizer is held by reference, so this costs nothing extra --
    but it must still rebuild correctly, which the vector path would not."""
    norm = np.random.default_rng(5).uniform(0.5, 2.0, size=(G, N))
    res = wade.wade(_counts(), norm, COND, nperms=200, seed=7)
    from wade.thinning import one_count

    lib = wade.library_sizes(_counts(), norm)
    expected = one_count(norm, lib, (G, N), norm_factor=1e6)
    np.testing.assert_allclose(res.pseudocount, expected, rtol=1e-12)


def test_no_pseudocount_stays_none():
    res = _run(pseudocount=0)
    assert res.pseudocount is None
    assert res.pseudocount_row(0) is None
