"""Layer 5 — the null, against a supplied permutation matrix.

The full ``g x nperms`` null matrices for both axes, elementwise.

This is the layer that validates a vectorized or native permutation
kernel against the R's serial loop, and **it is the only place a kernel
bug is cleanly separable from a p-value bug**. When the Rust kernel
lands, this is the layer that tests it: the same fixtures, the same
assertions, a different implementation behind ``null_statistics``.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import PARITY_SCENARIOS, TOL_NULL, assert_close, load_fixture
from portrun import run_port

import wade

LAYER = "layer 5 null"

WITH_PERMS = [n for n in PARITY_SCENARIOS if int(load_fixture(n)["shapes"]["B"]) > 0]


@pytest.mark.parity
@pytest.mark.parametrize("name", WITH_PERMS)
def test_null_matrix_elementwise(name):
    fx, res = run_port(name)
    assert res.null_mean_shift.shape == fx["null"]["perm_dm"].shape
    assert_close(res.null_mean_shift, fx["null"]["perm_dm"], TOL_NULL,
                 f"{name}: mean-shift null ({res.null_mean_shift.size} cells)", LAYER)


@pytest.mark.parity
@pytest.mark.parametrize("name", WITH_PERMS)
def test_the_supplied_permutations_were_actually_used(name):
    fx, res = run_port(name)
    assert np.array_equal(res.perms, fx["perms"])


def test_null_and_observed_paths_agree_on_the_identity_permutation():
    """The lean null path must be numerically identical to the full one.

    In the reference this holds by construction rather than coincidence —
    ``.wade_null_stats()`` computes the same expressions in the same
    order as ``wade_stats()``. The port must preserve that, because the
    observed statistic is compared against these null draws directly: if
    the two paths differed even in the last bits, a gene whose observed
    value ties a null draw could be counted on either side of ``>=``.
    """
    rng = np.random.default_rng(3)
    x = rng.lognormal(3, 1, size=(30, 21))
    cond = np.r_[np.ones(11, int), np.zeros(10, int)]

    obs = wade.wade_stats(x, cond)
    d = wade.null_statistics(x, cond[None, :])

    assert np.array_equal(d[:, 0], obs.mean_shift)


def test_one_shuffle_serves_all_genes():
    """The null must preserve the gene-gene correlation structure.

    Within permutation ``b`` the same label vector applies to every gene.
    Drawing an independent shuffle per gene would compute a different —
    and, across correlated genes, anti-conservative — null. Asserted by
    construction: two perfectly correlated genes must produce perfectly
    correlated null rows.
    """
    rng = np.random.default_rng(5)
    base = rng.lognormal(3, 1, size=(1, 19))
    x = np.vstack([base, base * 2.0, base * 0.5])
    cond = np.r_[np.ones(10, int), np.zeros(9, int)]
    perms = wade.draw_perms(cond, 40, seed=17)

    d = wade.null_statistics(x, perms)
    assert np.allclose(d[1], 2.0 * d[0], rtol=1e-12)
    assert np.allclose(d[2], 0.5 * d[0], rtol=1e-12)


def test_permutations_preserve_group_sizes():
    cond = np.r_[np.ones(7, int), np.zeros(4, int)]
    perms = wade.draw_perms(cond, 50, seed=2)
    assert perms.shape == (50, 11)
    assert np.all(perms.sum(axis=1) == 7)


def test_a_malformed_permutation_matrix_is_rejected():
    from wade.permutation import validate_perms

    cond = np.r_[np.ones(4, int), np.zeros(3, int)]
    with pytest.raises(ValueError, match="shape"):
        validate_perms(np.zeros((5, 7), dtype=int), cond, 6)
    bad = np.tile(cond, (5, 1))
    bad[2, 0] = 0                       # no longer a permutation of cond
    with pytest.raises(ValueError, match="permutation of cond"):
        validate_perms(bad, cond, 5)


@pytest.mark.parity
def test_null_memory_footprint_is_as_documented():
    """The null matrix, allocated up front.

    About 71 MB at the cfRNA production scale (2,219 genes, 2,000
    permutations) and 3.2 GB at 20,000 genes and 10,000 permutations,
    which is why retaining them is opt-in. Note a streaming
    implementation cannot simply drop them: the GPD refinement needs each
    refined gene's **full** null vector, so it must either retain rows for
    candidate genes or make two passes.
    """
    assert 2219 * 2000 * 8 / 1e6 == pytest.approx(35.5, rel=0.02)
    assert 20000 * 10000 * 8 / 1e9 == pytest.approx(1.6, rel=0.02)

    _, res = run_port("main")
    assert res.null_mean_shift.dtype == np.float64
    assert res.null_mean_shift.shape == (int(load_fixture("main")["shapes"]["g"]),
                                         int(load_fixture("main")["shapes"]["B"]))
