"""Layer 6 — the p-values.

Ordered so each step separates a distinct question:

1. **Exceedance counts first.** They are integers, so they are asserted
   exactly, and asserting them before the p-values separates the
   broadcast-axis question (hazard 7) from the ``(1 + nexc)/(B + 1)``
   arithmetic.
2. **Then the empirical p-values.**
3. **Then which genes were selected for refinement**, including a
   ``nperms = 300`` case where the answer must be "none".
4. **Then the GPD refinement itself**, branch by branch, asserting
   ``xi``, ``sigma`` and the branch taken alongside the returned p —
   because a returned p that happens to be right via the wrong branch is
   not a passing test.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import (
    PARITY_SCENARIOS,
    TOL_PVALUE,
    assert_close,
    load_fixture,
)
from portrun import run_port

import wade
from wade.pvalues import gpd_tail_p

LAYER = "layer 6 p-values"

WITH_PERMS = [n for n in PARITY_SCENARIOS if int(load_fixture(n)["shapes"]["B"]) > 0]


@pytest.mark.parity
@pytest.mark.parametrize("name", WITH_PERMS)
def test_exceedance_counts_are_exact(name):
    """Integers, so bitwise. This is where hazard 7 shows up or does not."""
    fx, res = run_port(name)
    assert np.array_equal(res.nexc_mean_shift, fx["pvalues"]["nexc_diff"])


@pytest.mark.parity
@pytest.mark.parametrize("name", WITH_PERMS)
def test_pvalues(name):
    fx, res = run_port(name)
    assert_close(res.p_mean_shift, fx["pvalues"]["p_diff"], TOL_PVALUE,
                 f"{name}: p.diff", LAYER)


@pytest.mark.parity
@pytest.mark.parametrize("name", WITH_PERMS)
def test_which_genes_were_refined(name):
    fx, res = run_port(name)
    assert np.array_equal(np.flatnonzero(res.refined_mean_shift),
                          fx["pvalues"]["refined_diff_i0"])


@pytest.mark.parity
def test_the_refinement_gate_is_closed_below_500_permutations():
    """``B >= 2 * n_tail`` is a property of the run, not of the gene.

    The ``gate_closed`` fixture plants a gene strongly enough that its
    exceedance count is 0 — so the gene-level condition
    ``nexc < n_exc_min`` fires — and then runs at B = 300, where the
    run-level condition does not. A port omitting the second condition
    would refine here and produce p-values the reference cannot produce,
    on a path no fixture at 1000 or 2000 permutations would ever touch.
    """
    fx, res = run_port("gate_closed")
    assert int(fx["shapes"]["B"]) == 300
    assert res.nexc_mean_shift.min() == 0, "the planted gene should have zero exceedances"
    assert not res.refined_mean_shift.any()
    assert res.p_mean_shift.min() == pytest.approx(1 / 301)


@pytest.mark.parity
def test_refinement_fires_above_the_gate_and_replaces_the_empirical_value():
    """Above the gate, selected genes get an extrapolated p instead of the floor.

    Note the direction is **not** guaranteed to be downward. Every refined
    gene here has zero exceedances, so its empirical p is the floor
    1/502 = 1.99e-3, and the GPD returns values from 1.4e-2 to 1.9e-1 —
    up to a hundred times *larger*. That is the refinement being honest
    rather than failing: on a heavy-tailed permutation null, exceeding all
    501 draws is not strong evidence, and the empirical floor overstates
    it. The refinement exists to resolve below 1/(B+1) when the tail
    supports it, not to guarantee that it does.
    """
    fx, res = run_port("refine")
    assert int(fx["shapes"]["B"]) == 501
    assert res.refined_mean_shift.any()

    empirical_floor = 1 / 502
    idx = np.flatnonzero(res.refined_mean_shift)
    assert np.all(res.nexc_mean_shift[idx] < 10)
    for p in (res.p_mean_shift,):
        assert not np.any(np.isclose(p[idx], empirical_floor)), (
            "a refined gene should no longer be sitting on the empirical floor"
        )
    # And the floor case, where refinement genuinely does resolve further
    # down, is covered at the unit level by the gpd_branches fixture.
    assert any(c["detail"]["branch"] == "floor" for c in _gpd_cases())


@pytest.mark.parity
def test_empirical_p_is_bounded_below_by_one_over_B_plus_one():
    for name in WITH_PERMS:
        fx, res = run_port(name)
        B = int(fx["shapes"]["B"])
        unrefined = ~res.refined_mean_shift
        assert np.all(res.p_mean_shift[unrefined] >= 1 / (B + 1) - 1e-15)
        assert np.all(res.p_mean_shift <= 1.0)


# ---------------------------------------------------------------------
# The GPD, branch by branch, against hand-constructed nulls.
#
# Pure scalar math, so parity here is exact and it is the cheapest place
# to catch an error.
# ---------------------------------------------------------------------

def _gpd_cases():
    return load_fixture("gpd_branches")["cases"]


@pytest.mark.parity
@pytest.mark.parametrize("idx", range(12))
def test_gpd_branch(idx):
    case = _gpd_cases()[idx]
    d = case["detail"]
    got = gpd_tail_p(
        float(np.asarray(case["o"])[0]), case["null"],
        n_tail=int(case["n_tail_arg"]), detail=True,
    )

    label = case["label"]
    assert got.branch == d["branch"], f"{label}: took branch {got.branch}, R took {d['branch']}"
    assert got.n_tail_used == int(d["n_tail_used"])
    assert got.n_exceedances == int(d["n_exceedances"] if "n_exceedances" in d else d["n_exc"])

    assert_close(np.array([got.threshold]), d["thr"], 1e-15, f"gpd {label}: thr", LAYER)
    assert_close(np.array([got.empirical]), d["emp"], 1e-15, f"gpd {label}: empirical", LAYER)
    assert_close(np.array([got.mean_exc]), d["m"], 1e-14, f"gpd {label}: mean(exc)", LAYER)
    assert_close(np.array([got.var_exc]), d["v"], 1e-13, f"gpd {label}: var(exc)", LAYER)
    assert_close(np.array([got.xi]), d["xi"], 1e-12, f"gpd {label}: xi", LAYER)
    assert_close(np.array([got.sigma]), d["sigma"], 1e-12, f"gpd {label}: sigma", LAYER)
    assert_close(np.array([got.p]), d["p"], 1e-12, f"gpd {label}: p", LAYER)


def test_every_gpd_branch_is_actually_exercised():
    """A branch table that never reaches a branch is not a branch table.

    Reaching the ``gpd`` branch takes care: pushing the observed value far
    into the tail makes the floor bind, which is a *different* branch. The
    fixture therefore carries a heavy-tailed null observed at its 5th and
    2nd largest draws (both land on the GPD form) alongside the same null
    observed at 5x its maximum (which floors), so the two are
    distinguishable rather than conflated.

    ``bail_degenerate_sigma`` is deliberately absent and is **not
    reachable**: ``sigma = m(1 + m^2/v)/2`` with exceedances that are
    strictly positive by construction gives ``m > 0``, and the preceding
    guard has already established ``v > 0``, so ``sigma > 0`` always. The
    reference's check is defensive, not a live path.
    """
    branches = {c["detail"]["branch"] for c in _gpd_cases()}
    assert {
        "gpd",
        "exponential",
        "floor",
        "bail_few_exceedances",
        "bail_obs_at_or_below_thr",
        "bail_degenerate_variance",
    } <= branches, f"uncovered branches; got {sorted(branches)}"


def test_positive_shape_reaches_the_gpd_form_not_only_the_floor():
    """The two outcomes of a positive shape, kept apart."""
    cases = {c["label"]: c for c in _gpd_cases()}
    inside = cases["gpd_positive_shape"]["detail"]
    floored = cases["gpd_positive_shape_floored"]["detail"]
    assert inside["branch"] == "gpd" and floored["branch"] == "floor"
    assert float(np.asarray(inside["xi"])[0]) > 0
    assert float(np.asarray(floored["xi"])[0]) > 0
    assert float(np.asarray(inside["p"])[0]) > float(np.asarray(floored["p"])[0])


@pytest.mark.parity
def test_the_exceedance_inequality_is_strict():
    """Ties at the threshold are excluded, which can change the branch.

    ``exc = s[s > thr] - thr``. With a null of 400 tied values at the top,
    the threshold *is* that value and the exceedance count is **zero**, so
    the function bails to the empirical p rather than fitting anything. A
    port using ``>=`` would get 400 exceedances, a different mean and
    variance, and would fit a GPD where the reference does not.
    """
    case = next(c for c in _gpd_cases() if c["label"] == "few_exceedances_via_ties")
    got = gpd_tail_p(float(np.asarray(case["o"])[0]), case["null"], detail=True)
    assert got.n_exceedances == 0
    assert got.branch == "bail_few_exceedances"
    assert got.p == pytest.approx(got.empirical)


@pytest.mark.parity
def test_variance_is_the_sample_variance():
    """``ddof=1``, not NumPy's default of 0. Hazard 6.

    The estimators are functions of ``m^2 / v``, so an understated
    variance pushes ``xi`` down and ``sigma`` up — and the branch test is
    ``xi <= 0``, so a factor of ``n/(n-1)`` can flip the branch. Measured
    on ``[1, 2, 3, 4, 10]``: R's var is 12.5, the population variance
    10.0, a ratio of 1.25.
    """
    v = np.array([1.0, 2, 3, 4, 10])
    assert v.var(ddof=1) == pytest.approx(12.5)
    assert v.var(ddof=0) == pytest.approx(10.0)

    # And the branch really is near the operating point on ordinary nulls.
    xis = [float(np.asarray(c["detail"]["xi"])[0]) for c in _gpd_cases()]
    finite = [x for x in xis if np.isfinite(x)]
    assert any(-0.2 < x < 0.0 for x in finite), "the boundary should be crossed by noise"
    assert any(0.0 < x < 0.1 for x in finite)


@pytest.mark.parity
def test_the_floor_is_a_resolution_limit_not_a_numerical_guard():
    """``1 / (B * n_tail)``: 2.0e-6 at B = 2000, 4.0e-6 at B = 1000.

    A port that returns smaller p-values by "improving" the floor is a
    regression. It is a statement about what B permutations can support,
    and a better tail fit does not buy more resolution.
    """
    case = next(c for c in _gpd_cases() if c["label"] == "floor_binding")
    got = gpd_tail_p(float(np.asarray(case["o"])[0]), case["null"], detail=True)
    assert got.branch == "floor"
    assert got.p == pytest.approx(2.0e-6, rel=1e-12)
    assert got.p_floor == pytest.approx(1 / (2000 * 250))

    rng = np.random.default_rng(0)
    null1000 = rng.exponential(size=1000)
    got2 = gpd_tail_p(500 * null1000.max(), null1000, detail=True)
    assert got2.p_floor == pytest.approx(4.0e-6)


@pytest.mark.parity
def test_n_tail_is_capped_at_half_the_permutation_count():
    case = next(c for c in _gpd_cases() if c["label"] == "n_tail_capped_at_B_over_2")
    got = gpd_tail_p(float(np.asarray(case["o"])[0]), case["null"], detail=True)
    assert got.n_perms == 100
    assert got.n_tail_used == 50


def test_the_gate_boundary_is_exactly_500():
    """B = 250 and 499 do not qualify; 500, 1000 and 2000 do."""
    rng = np.random.default_rng(1)
    for B, expect in ((250, False), (499, False), (500, True), (1000, True)):
        null = rng.exponential(size=(3, B))
        obs = null.max(axis=1) * 10
        _, _, refined = wade.perm_pvalues(obs, null, alternative='greater')
        assert refined.any() == expect, f"B={B}: expected refinement={expect}"
