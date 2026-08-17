"""The Rust kernel, validated against the layer-5 null matrices.

``ROADMAP.md`` puts this last for a reason: **establish parity on
readable code before optimizing.** If the NumPy implementation and the
kernel had been written together, a numerical disagreement with the R
would have two candidate causes — a port error or an optimization error
— and no way to tell them apart. Written in order, the NumPy path was
validated against the R first, so the kernel now has a baseline that is
already known to be correct and any disagreement has exactly one
explanation.

The kernel is therefore held to **two** standards, and both matter:

1. It must reproduce the R's null matrices on the golden fixtures — the
   same layer-5 assertions the NumPy path passes. This is the only place
   a kernel bug is cleanly separable from a p-value bug.
2. It must agree with the NumPy path **elementwise**, including on inputs
   there is no fixture for. Two implementations that each match R on 14
   scenarios can still diverge on a fifteenth.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import TOL_NULL, assert_close, load_fixture
from portrun import scalar

import wade
from wade.permutation import HAVE_RUST_KERNEL, null_statistics

pytestmark = pytest.mark.kernel

LAYER = "kernel"

WITH_PERMS = [
    n for n in
    ["tiny", "main", "even_larger", "vecnorm", "m2", "gate_closed", "refine",
     "weighted", "log2scaled", "zerolib", "nonames", "tailconc", "tiesheavy"]
    if int(load_fixture(n)["shapes"]["B"]) > 0
]


def test_the_kernel_was_actually_built():
    """Fail loudly rather than skipping into a false pass.

    A test suite that silently skips the kernel because it is missing
    reports green for an untested component. If the kernel is genuinely
    unavailable the right response is to say so, not to pass.
    """
    assert HAVE_RUST_KERNEL, (
        "the Rust kernel is not built. Run `pip install -e .` in an environment "
        "with cargo and rustc on PATH, or deselect with `-m 'not kernel'`."
    )


def _run_backend(name: str, backend: str):
    fx = load_fixture(name)
    p = fx["params"]
    return fx, wade.wade(
        fx["counts"], fx["normalizer"]["data"], fx["cond"],
        nperms=int(p["nperms"]), tail_q=scalar(p["tail_q"]),
        noise=scalar(p["noise"]), norm_factor=scalar(p["norm_factor"]),
        jitter=fx["jitter"], perms=fx["perms"],
        log2_scale=bool(p["log2_scale"]), weight=scalar(p["weight"]),
        gene_names=fx["gene_names"], keep_null=True, backend=backend,
    )


# ---------------------------------------------------------------------
# 1. Against the R
# ---------------------------------------------------------------------

@pytest.mark.parity
@pytest.mark.parametrize("name", WITH_PERMS)
def test_kernel_null_matrices_match_R(name):
    fx, res = _run_backend(name, "rust")
    assert_close(res.null_diff, fx["null"]["perm_dm"], TOL_NULL,
                 f"{name}: kernel null diff.mean", LAYER)
    assert_close(res.null_tail, fx["null"]["perm_tm"], TOL_NULL,
                 f"{name}: kernel null tail.mean", LAYER)


@pytest.mark.parity
@pytest.mark.parametrize("name", WITH_PERMS)
def test_kernel_reaches_the_same_pvalues_as_R(name):
    """The endpoint, so a kernel-only regression cannot hide behind the nulls."""
    fx, res = _run_backend(name, "rust")
    assert np.array_equal(res.nexc_diff, fx["pvalues"]["nexc_diff"])
    assert np.array_equal(res.nexc_tail, fx["pvalues"]["nexc_tail"])
    assert_close(res.p_diff, fx["pvalues"]["p_diff"], 1e-12,
                 f"{name}: kernel p.diff", LAYER)
    assert_close(res.p_tail, fx["pvalues"]["p_tail"], 1e-12,
                 f"{name}: kernel p.tail", LAYER)


# ---------------------------------------------------------------------
# 2. Against the NumPy baseline
# ---------------------------------------------------------------------

@pytest.mark.parametrize("name", WITH_PERMS)
def test_kernel_agrees_with_numpy_on_the_fixtures(name):
    _, rust = _run_backend(name, "rust")
    _, numpy_ = _run_backend(name, "numpy")
    assert_close(rust.null_diff, numpy_.null_diff, 1e-15,
                 f"{name}: rust vs numpy null diff.mean", LAYER)
    assert_close(rust.null_tail, numpy_.null_tail, 1e-15,
                 f"{name}: rust vs numpy null tail.mean", LAYER)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_kernel_agrees_with_numpy_on_random_inputs(seed):
    """Beyond the fixtures.

    Two implementations that each match R on the golden scenarios can
    still diverge on a shape or a data regime no fixture covers. This
    sweeps group sizes, gene counts and tail windows randomly, including
    the tie-heavy and zero-heavy regimes where the quantile guard matters.
    """
    rng = np.random.default_rng(seed)
    n1 = int(rng.integers(2, 20))
    n0 = int(rng.integers(2, 20))
    g = int(rng.integers(1, 40))
    n = n1 + n0

    style = seed % 3
    if style == 0:
        x = rng.lognormal(3, 1, size=(g, n))
    elif style == 1:
        x = rng.integers(0, 4, size=(g, n)).astype(float)     # heavy ties
    else:
        x = rng.integers(0, 2, size=(g, n)).astype(float) * rng.lognormal(2, 1, (g, n))

    cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
    perms = wade.draw_perms(cond, 37, seed=seed + 100)
    tail_q = float(rng.choice([0.05, 0.10, 0.25, 0.5]))

    d_np, t_np = null_statistics(x, perms, tail_q=tail_q, backend="numpy")
    d_rs, t_rs = null_statistics(x, perms, tail_q=tail_q, backend="rust")
    assert_close(d_rs, d_np, 1e-15, f"random {seed}: diff (g={g}, {n1}v{n0})", LAYER)
    assert_close(t_rs, t_np, 1e-15, f"random {seed}: tail (g={g}, {n1}v{n0})", LAYER)


def test_kernel_agrees_with_numpy_on_the_pretransform_paths():
    """``weight`` and ``log2_scale`` — the reference's non-lean path.

    The reference drops off its fast path entirely when either is set. The
    kernel implements both instead, so setting them does not silently move
    a user from compiled code to a Python loop. That is only defensible if
    the two paths agree, which is what this asserts.
    """
    rng = np.random.default_rng(9)
    x = rng.lognormal(3, 1, size=(23, 17))
    cond = np.r_[np.ones(9, int), np.zeros(8, int)]
    perms = wade.draw_perms(cond, 29, seed=4)

    for weight, log2_scale in ((2.0, False), (1.0, True), (0.5, True)):
        d_np, t_np = null_statistics(x, perms, tail_q=0.10, weight=weight,
                                     log2_scale=log2_scale, backend="numpy")
        d_rs, t_rs = null_statistics(x, perms, tail_q=0.10, weight=weight,
                                     log2_scale=log2_scale, backend="rust")
        label = f"weight={weight}, log2={log2_scale}"
        assert_close(d_rs, d_np, 1e-15, f"pretransform {label}: diff", LAYER)
        assert_close(t_rs, t_np, 1e-15, f"pretransform {label}: tail", LAYER)


def test_kernel_type7_matches_the_numpy_implementation():
    """One quantile definition, two implementations — held to bitwise equality.

    Having the kernel implement type 7 itself removes a dependency on
    NumPy's behaviour inside the hot loop, at the cost of two definitions
    that must agree exactly. This is the assertion that pays that cost.
    """
    from wade import _kernel

    rng = np.random.default_rng(31)
    for n in (1, 2, 3, 5, 8, 13, 21, 34):
        for style in range(3):
            if style == 0:
                x = rng.lognormal(2, 1.5, size=(11, n))
            elif style == 1:
                x = rng.integers(0, 3, size=(11, n)).astype(float)
            else:
                x = np.full((11, n), 4.0)
            for m in range(1, n + 1):
                q = wade.probability_grid(m)
                got = _kernel.type7_quantiles(np.ascontiguousarray(x), q)
                want = wade.type7_quantiles(x, q)
                assert np.array_equal(got, want), (
                    f"kernel type-7 differs at n={n}, m={m}, style={style}"
                )


# ---------------------------------------------------------------------
# 3. The kernel's own input validation
# ---------------------------------------------------------------------

def test_kernel_rejects_a_mismatched_permutation_matrix():
    from wade import _kernel

    x = np.ones((4, 7))
    q = wade.probability_grid(3)
    with pytest.raises(ValueError, match="columns"):
        _kernel.null_statistics(x, np.zeros((5, 9), dtype=np.int64), q, 1, 1.0, False)


def test_kernel_rejects_labels_outside_zero_and_one():
    from wade import _kernel

    x = np.ones((4, 7))
    q = wade.probability_grid(3)
    perms = np.array([[1, 1, 1, 0, 0, 0, 2]], dtype=np.int64)
    with pytest.raises(ValueError, match="only 0 and 1"):
        _kernel.null_statistics(x, perms, q, 1, 1.0, False)


def test_kernel_rejects_a_grid_that_does_not_match_the_group_sizes():
    from wade import _kernel

    x = np.ones((4, 7))
    perms = np.array([[1, 1, 1, 0, 0, 0, 0]], dtype=np.int64)   # min(n1, n0) = 3
    with pytest.raises(ValueError, match="min\\(n1, n0\\)"):
        _kernel.null_statistics(x, perms, wade.probability_grid(5), 1, 1.0, False)


def test_kernel_rejects_an_out_of_range_tail_window():
    from wade import _kernel

    x = np.ones((4, 7))
    perms = np.array([[1, 1, 1, 0, 0, 0, 0]], dtype=np.int64)
    q = wade.probability_grid(3)
    with pytest.raises(ValueError, match="k must lie"):
        _kernel.null_statistics(x, perms, q, 0, 1.0, False)
    with pytest.raises(ValueError, match="k must lie"):
        _kernel.null_statistics(x, perms, q, 4, 1.0, False)


def test_kernel_rejects_non_finite_input():
    from wade import _kernel

    x = np.ones((4, 7))
    x[2, 3] = np.nan
    perms = np.array([[1, 1, 1, 0, 0, 0, 0]], dtype=np.int64)
    with pytest.raises(ValueError, match="non-finite"):
        _kernel.null_statistics(x, perms, wade.probability_grid(3), 1, 1.0, False)


def test_backend_selection_is_explicit_and_validated():
    rng = np.random.default_rng(0)
    x = rng.lognormal(3, 1, size=(5, 9))
    cond = np.r_[np.ones(5, int), np.zeros(4, int)]
    perms = wade.draw_perms(cond, 7, seed=1)
    with pytest.raises(ValueError, match="backend must be"):
        null_statistics(x, perms, tail_q=0.1, backend="fortran")
