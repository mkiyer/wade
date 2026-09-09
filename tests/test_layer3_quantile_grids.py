"""Layer 3 — the quantile grids ``Q1``, ``Q0`` and ``D``, in full.

The highest-value layer in the suite. Every statistic downstream is a
reduction of ``D``, so if ``D`` agrees then hazards 1 (quantile type),
8 (jitter fill order), 9 (grid orientation) and 11 (the reshape) are all
excluded at once, and any later disagreement is in the reductions or the
p-values.

Asserted on the **full grids**, not on summaries of them: a summary can
agree while the grid it summarizes does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import PARITY_SCENARIOS, TOL_GRID, assert_close, load_fixture
from portrun import run_port

import wade
from wade.quantiles import type7_quantiles

LAYER = "layer 3 quantile grids"


@pytest.mark.parity
@pytest.mark.parametrize("name", PARITY_SCENARIOS)
def test_Q1_elementwise(name):
    fx, res = run_port(name)
    assert res.stats.Q1.shape == fx["Q1"].shape
    assert_close(res.stats.Q1, fx["Q1"], TOL_GRID, f"{name}: Q1", LAYER)


@pytest.mark.parity
@pytest.mark.parametrize("name", PARITY_SCENARIOS)
def test_Q0_elementwise(name):
    fx, res = run_port(name)
    assert res.stats.Q0.shape == fx["Q0"].shape
    assert_close(res.stats.Q0, fx["Q0"], TOL_GRID, f"{name}: Q0", LAYER)


@pytest.mark.parity
@pytest.mark.parametrize("name", PARITY_SCENARIOS)
def test_D_elementwise(name):
    fx, res = run_port(name)
    assert_close(res.stats.D, fx["D"], TOL_GRID, f"{name}: D", LAYER)


@pytest.mark.parity
@pytest.mark.parametrize("name", PARITY_SCENARIOS)
def test_the_first_column_is_the_group_maximum(name):
    """Orientation again, this time against the data rather than the grid.

    Position 0 of each row must be that group's largest normalized value.
    If it were the smallest, ``tail_mean`` would be a lower-tail statistic
    and the method would be looking for the wrong thing without erroring.
    """
    fx, res = run_port(name)
    cond = fx["cond"]
    x = res.tpm
    if bool(fx["params"]["log2_scale"]) or float(np.asarray(fx["params"]["weight"])[0]) != 1.0:
        pytest.skip("pre-transformed scenario; the raw tpm is not what is ranked")
    assert np.allclose(res.stats.Q1[:, 0], x[:, cond == 1].max(axis=1), rtol=1e-15)
    assert np.allclose(res.stats.Q0[:, 0], x[:, cond == 0].max(axis=1), rtol=1e-15)
    if res.stats.nprobs > 1:
        assert np.allclose(res.stats.Q1[:, -1], x[:, cond == 1].min(axis=1), rtol=1e-15)


# ---------------------------------------------------------------------
# Hazard 1, on its own terms: the quantile convention.
# ---------------------------------------------------------------------

@pytest.mark.parity
def test_type7_matches_R_on_the_convention_stress_vectors():
    """Both the odd- and the even-length vector, on WADE's own grid shape.

    The even-length vector is the one that matters. On the odd-length
    vector types 1, 2 and 7 agree at every probability, so a test built
    only on it would pass with type 1 substituted; on the even-length
    vector the seven conventions give five different answers at p = 0.75.
    Since ``tail_mean`` at a realistic ``nprobs`` is the mean of one or
    two of those columns, a convention mismatch does not perturb the
    statistic — it replaces it.
    """
    fx = load_fixture("quantile_type7")
    for case in fx["cases"]:
        x = np.asarray(case["x"])[None, :]
        for grid in case["grids"]:
            q = grid["q"]
            got = type7_quantiles(x, q)[0]
            assert_close(
                got, grid["rowQuantiles"], 0.0,
                f"type7 {case['label']} m={int(grid['nprobs'])} vs rowQuantiles", LAYER,
            )
            assert np.array_equal(got, grid["rowQuantiles"]), (
                "type-7 quantiles must match matrixStats::rowQuantiles bitwise"
            )


@pytest.mark.parity
def test_rowQuantiles_and_stats_quantile_agree_in_R():
    """The reference uses one; the docs claim the other is identical.

    Worth asserting from the fixture rather than trusting: if they ever
    diverged, the port would have to choose which to match and the choice
    would be invisible.
    """
    fx = load_fixture("quantile_type7")
    for case in fx["cases"]:
        for grid in case["grids"]:
            assert np.array_equal(grid["rowQuantiles"], grid["stats_quantile_type7"])


def test_type7_interpolation_leaves_ties_untouched():
    """R does not interpolate where ``x[hi] == x[lo]``, and neither does this.

    Without that guard ``(1 - h) * a + h * a`` can drift off ``a`` by an
    ulp on a run of equal values — and WADE's target regime is zero-heavy
    count data, which is nothing but runs of equal values.
    """
    x = np.array([[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]])
    for m in range(1, 8):
        got = type7_quantiles(x, wade.probability_grid(m))
        assert np.all(got == 3.0)

    x2 = np.array([[0.0, 0.0, 0.0, 0.0, 7.0, 7.0, 7.0]])
    got = type7_quantiles(x2, wade.probability_grid(7))
    assert set(np.unique(got)) <= {0.0, 7.0}
