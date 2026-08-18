"""Layer 1 — the deterministic scalars and the probability grid.

``nprobs``, ``k``, ``n1``, ``n0`` and ``q`` itself. Asserted **exactly**:
these are integers or exactly-representable, so a tolerance here would be
hiding something.

This layer catches the grid-size and tail-window arithmetic before any
statistic is computed, and asserting ``q[0] == 1`` and ``q[-1] == 0``
catches grid orientation (hazard 9) at the earliest possible point — a
port that builds an ascending grid and keeps the ``[:k]`` slice computes
a lower-tail statistic under the name ``tail_mean``, which does not error
and is comparable in magnitude on a null gene.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import PARITY_SCENARIOS, assert_close, load_fixture
from portrun import run_port

import wade

LAYER = "layer 1 grid"


@pytest.mark.parity
@pytest.mark.parametrize("name", PARITY_SCENARIOS)
def test_scalars_exact(name):
    fx, res = run_port(name)
    assert res.stats.nprobs == int(fx["shapes"]["nprobs"])
    assert res.stats.k == int(fx["shapes"]["k"])
    assert res.stats.n1 == int(fx["shapes"]["n1"])
    assert res.stats.n0 == int(fx["shapes"]["n0"])


@pytest.mark.parity
@pytest.mark.parametrize("name", PARITY_SCENARIOS)
def test_probability_grid_matches_R_bitwise(name):
    """R's ``seq(1, 0, length.out = m)`` against NumPy's ``linspace``.

    Asserted bitwise rather than to a tolerance. The two languages build
    the sequence by different routes and this is where you find out
    whether they land on the same doubles; if they did not, every
    interpolated quantile would sit at a slightly different probability
    and the whole comparison would be off by a structured amount too
    small to notice and too large to ignore.
    """
    fx, res = run_port(name)
    q_r = fx["q"]
    q_port = res.stats.q
    assert q_port.shape == q_r.shape
    assert np.array_equal(q_port, q_r), (
        f"grid differs from R at index "
        f"{int(np.flatnonzero(q_port != q_r)[0])}"
    )
    assert_close(q_port, q_r, 0.0, f"{name}: q", LAYER)


@pytest.mark.parity
@pytest.mark.parametrize("name", PARITY_SCENARIOS)
def test_grid_orientation_is_descending(name):
    """Hazard 9, asserted structurally rather than numerically."""
    fx, res = run_port(name)
    q = res.stats.q
    assert q[0] == 1.0, "position 0 must be probability 1 — the group maximum"
    if q.size > 1:
        assert q[-1] == 0.0
        assert np.all(np.diff(q) < 0), "the grid must run high to low"


def test_tail_window_arithmetic_matches_the_documented_table():
    """``k = max(1, ceil(0.10 * m))`` at the values worked in method.md."""
    table = {5: 1, 10: 1, 18: 2, 20: 2, 22: 3, 33: 4, 34: 4, 60: 6, 100: 10}
    for m, k in table.items():
        assert wade.tail_window_size(m, 0.10) == k


def test_tail_window_is_never_zero():
    for m in range(1, 200):
        assert wade.tail_window_size(m, 0.001) >= 1


def test_nprobs_is_set_by_the_smaller_group():
    """The smaller group is read essentially without interpolation.

    This is why ``m = min(n0, n1)``: the grid nodes coincide with the
    order statistics of the *smaller* group, so it is read exactly and
    only the larger group is interpolated. ``m = min(n0, n1)`` is the
    largest grid on which at least one group is exact.

    **Refinement of the claim in method.md section 1.1**, which states
    the coincidence as exact. It is exact in real arithmetic and only to
    floating point in practice: ``1 + (m - 1) * q`` does not always land
    on an integer. Measured at ``m = 8``, the indices come out as
    ``8, 7, 6, 5, 4, 3.0000000000000009, 2.0000000000000004, 1``, so two
    of the eight nodes interpolate across an interval whose endpoints are
    adjacent order statistics.

    **R does exactly the same thing** — verified in the sandbox, where
    ``matrixStats::rowQuantiles`` and ``stats::quantile(type = 7)`` both
    fail an ``identical()`` against the sorted sample at m = 8, 13 and 21
    while agreeing bitwise with each other. So this is a property of the
    method's arithmetic, not a divergence, and it is asserted here to
    floating-point equality rather than bitwise for that reason.
    """
    rng = np.random.default_rng(7)
    for n in (2, 3, 5, 8, 13, 21):
        x = rng.normal(size=(1, n)) * 100
        q = wade.probability_grid(n)
        got = wade.type7_quantiles(x, q)[0]
        want = np.sort(x[0])[::-1]
        # Not a comparison against R — it measures the port against the exact
        # mathematical claim, which R also misses. Kept out of the parity total.
        assert_close(got, want, 1e-15, f"smaller-group exactness at m={n}",
                     "(diagnostic, not parity)")


def test_tail_window_report_tells_a_caller_what_the_design_buys():
    """The consumer layer needs this arithmetic *before* running the test."""
    assert wade.tail_window_report(10, 10)["k"] == 1
    assert wade.tail_window_report(10, 10)["single_point_tail"] is True
    assert wade.tail_window_report(40, 20)["k"] == 2
    assert wade.tail_window_report(40, 30)["k"] == 3
    assert wade.tail_window_report(77, 18)["nprobs"] == 18
