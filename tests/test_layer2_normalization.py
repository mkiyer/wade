"""Layer 2 — normalization, against a supplied jitter.

The per-cell denominator is the subtlest arithmetic in the method, and
the failure mode is the dangerous kind: substituting the naive
``lib_sizes[j]`` denominator changes every value by about one part in
10^6, leaves within-row rank ordering completely untouched, and therefore
passes every plausibility check the quantile grid could apply. It fails
only an exact test, which is the argument for having one.

That is why the tolerance here is the tightest in the suite. A generous
tolerance at this layer would let the one arithmetic error that survives
every other check straight through.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import (
    PARITY_SCENARIOS,
    TOL_NORMALIZATION,
    assert_close,
    load_fixture,
    max_rel_dev,
    record,
)
from portrun import run_port, scalar

import wade

LAYER = "layer 2 normalization"


@pytest.mark.parity
@pytest.mark.parametrize("name", PARITY_SCENARIOS)
def test_library_sizes(name):
    fx, _ = run_port(name)
    got = wade.library_sizes(fx["counts"], fx["normalizer"]["data"])
    assert_close(got, fx["lib_sizes"], TOL_NORMALIZATION, f"{name}: lib_sizes", LAYER)


@pytest.mark.parity
@pytest.mark.parametrize("name", PARITY_SCENARIOS)
def test_normalized_matrix_elementwise(name):
    fx, res = run_port(name)
    assert res.tpm.shape == fx["tpm"].shape
    assert_close(res.tpm, fx["tpm"], TOL_NORMALIZATION, f"{name}: tpm", LAYER)


@pytest.mark.parity
@pytest.mark.parametrize("name", PARITY_SCENARIOS)
def test_the_supplied_jitter_was_actually_used(name):
    """Guards against the injection silently not taking effect.

    A parity suite whose fixture jitter is ignored would compare the port
    against R using two different noise draws and could only ever fail —
    but if the port *also* drew a jitter of the same tiny magnitude, the
    difference might sit near the tolerance and be misread as accumulated
    floating-point error. Assert the identity outright.
    """
    fx, res = run_port(name)
    assert np.array_equal(res.jitter, fx["jitter"])


def test_naive_denominator_is_detectably_wrong():
    """The specific error this layer exists to catch.

    Replacing the per-cell denominator with ``lib_sizes[j]`` is the
    natural misreading. It leaves within-row rank order unchanged — so
    every quantile-based sanity check passes — and moves the values by
    roughly 1e-6 relative, which is six orders of magnitude above this
    layer's tolerance.
    """
    rng = np.random.default_rng(11)
    g, n = 800, 6
    counts = rng.poisson(30, size=(g, n)).astype(float)
    normalizer = rng.integers(1, 13, size=g).astype(float)
    jitter = rng.uniform(0, 0.01, size=(g, n))

    correct = wade.tpm_like(counts, normalizer, jitter=jitter)

    lib = wade.library_sizes(counts, normalizer)
    naive = 1e6 * ((counts + jitter) / normalizer[:, None]) / lib[None, :]

    dev, _ = max_rel_dev(naive, correct)
    # Recorded outside the parity layers: this is the size of an error the
    # suite is designed to catch, not a deviation the port exhibits.
    record("naive lib_sizes-only denominator", dev, TOL_NORMALIZATION,
           "(diagnostic, not parity)")
    assert dev > 1e-8, "the naive denominator should be detectably different"
    assert dev < 1e-5, f"expected an error around 1e-6 relative, got {dev:.2e}"

    # ... and the reason it is dangerous: the ordering is identical.
    assert np.array_equal(np.argsort(correct, axis=1), np.argsort(naive, axis=1))

    # The bound stated in docs/implementation-notes.md section 2.
    bound = 0.01 / (normalizer.min() * lib.min())
    assert dev <= bound


def test_column_sums_are_not_exactly_norm_factor():
    """The output is TPM-*like*, not TPM, and a port must not 'fix' that.

    The excess is the difference between the per-gene and per-column
    jitter corrections. Nothing downstream depends on columns summing to
    one million; restoring the constant would change every number.
    """
    rng = np.random.default_rng(11)
    g, n = 800, 6
    counts = rng.poisson(30, size=(g, n)).astype(float)
    normalizer = rng.integers(1, 13, size=g).astype(float)
    jitter = rng.uniform(0, 0.01, size=(g, n))

    x = wade.tpm_like(counts, normalizer, jitter=jitter)
    sums = x.sum(axis=0)
    assert np.all(sums > 1e6)
    assert np.all(sums < 1e6 * (1 + 1e-3))

    # Without jitter the naive identity is exact, which is what makes the
    # excess attributable to the jitter correction rather than to noise.
    x0 = wade.tpm_like(counts, normalizer, jitter=np.zeros((g, n)))
    assert np.allclose(x0.sum(axis=0), 1e6, rtol=1e-12)


def test_a_gene_that_is_the_whole_library_normalizes_to_norm_factor():
    """If gene g is the only nonzero count in column j, its value is norm_factor.

    Because then ``lib[j] = C/L`` and ``denom = (C + z)/L``, which is
    exactly the numerator, so the ratio is 1.

    To floating point, not bitwise: the numerator evaluates ``(C + z)/L``
    while the denominator evaluates ``C/L + z/L``, and those are the same
    real number reached by different roundings. The exactly-bitwise case
    is the all-zero column tested below, where ``lib[j] = 0`` makes both
    sides literally the same expression.
    """
    counts = np.zeros((3, 2))
    counts[1, 0] = 57.0
    counts[0, 1] = 11.0
    normalizer = np.array([2.0, 3.0, 5.0])
    jitter = np.full((3, 2), 0.004)
    x = wade.tpm_like(counts, normalizer, jitter=jitter)
    assert x[1, 0] == pytest.approx(1e6, rel=1e-15)
    assert x[0, 1] == pytest.approx(1e6, rel=1e-15)


@pytest.mark.parity
def test_zero_library_column_returns_exactly_norm_factor_for_every_gene():
    """An artefact of the algebra, not a sensible value for an empty library.

    Where the whole column is zero, ``lib_sizes[j] = 0`` makes numerator
    and denominator equal for every gene. The R does not detect this and
    neither does the port; the behaviour is pinned here so it cannot
    change silently.
    """
    fx, res = run_port("zerolib")
    zero_col = int(np.flatnonzero(fx["counts"].sum(axis=0) == 0)[0])
    assert np.all(res.tpm[:, zero_col] == 1e6)
    assert np.all(fx["tpm"][:, zero_col] == 1e6)


def test_normalizer_shape_is_validated_rather_than_recycled():
    """R recycles a wrong-axis normalizer silently and returns plausible numbers.

    ``docs/implementation-notes.md`` section 1 measures it: on a 6 x 4 matrix
    a per-sample length-4 vector recycles cleanly, with no error and no
    warning, and every library size is wrong. This is the one input error
    in the reference that produces plausible output, so the port raises.
    """
    counts = np.ones((6, 4))
    with pytest.raises(ValueError, match="one value per gene"):
        wade.library_sizes(counts, np.array([1.0, 2.0, 4.0, 8.0]))
    good = wade.library_sizes(counts, np.array([1.0, 2, 4, 8, 16, 32]))
    assert np.allclose(good, 1.96875)


def test_jitter_must_be_two_dimensional():
    """Hazard 8: never a flat vector plus dimensions."""
    counts = np.ones((3, 2))
    with pytest.raises(ValueError, match="genes x samples"):
        wade.tpm_like(counts, np.ones(3), jitter=np.zeros(6))


def test_jitter_is_drawn_once_and_is_reproducible_per_seed():
    counts = np.ones((4, 3))
    a = wade.tpm_like(counts, np.ones(4), seed=1)
    b = wade.tpm_like(counts, np.ones(4), seed=1)
    c = wade.tpm_like(counts, np.ones(4), seed=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_seed_none_does_not_touch_global_state_and_varies():
    counts = np.ones((4, 3))
    a = wade.draw_jitter((4, 3), seed=None)
    b = wade.draw_jitter((4, 3), seed=None)
    assert not np.array_equal(a, b)
