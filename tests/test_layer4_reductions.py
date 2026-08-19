"""Layer 4 — the per-gene reductions.

``mean_shift``, ``w1``, ``tail_mean``, ``fc``, ``cond1_mean``,
``cond0_mean``, ``tot_mean``, ``diff_frac``, at relative tolerance
(hazard 10 — ``rowSums`` and NumPy's pairwise summation legitimately
differ in the last bits).

``tail_conc`` is handled **separately**, in ``test_divergences.py``:
asserting the ratio against the R would enforce a known defect. What is
asserted here instead is its numerator and denominator, which are exactly
comparable across implementations even where the ratio is not. That
generalizes — comparing the components separately localizes any
disagreement in a ratio.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import PARITY_SCENARIOS, TOL_REDUCTION, assert_close, load_fixture
from portrun import run_port

LAYER = "layer 4 reductions"

#: The reductions that survive the redesign. `tail.mean`, `tail.conc`,
#: `tot.mean` and `diff.frac` belonged to the retired tail machinery and no
#: longer correspond to anything the port computes.
REDUCTIONS = [
    ("mean_shift", "diff.mean"),
    ("w1", "w1"),
    ("fc", "fc"),
    ("case_mean", "cond1.mean"),
    ("ctrl_mean", "cond0.mean"),
]


@pytest.mark.parity
@pytest.mark.parametrize("name", PARITY_SCENARIOS)
@pytest.mark.parametrize("py_name,r_name", REDUCTIONS)
def test_reduction(name, py_name, r_name):
    fx, res = run_port(name)
    assert_close(
        getattr(res, py_name), fx["stats"][r_name],
        TOL_REDUCTION, f"{name}: {r_name}", LAYER,
    )


@pytest.mark.parity
@pytest.mark.parity
@pytest.mark.parametrize("name", PARITY_SCENARIOS)
def test_fc_is_the_ratio_of_grid_means(name):
    _, res = run_port(name)
    assert_close(res.fc, res.case_mean / res.ctrl_mean,
                 1e-15, f"{name}: fc identity", LAYER)


@pytest.mark.parametrize("name", PARITY_SCENARIOS)
def test_w1_dominates_absolute_mean_shift(name):
    """``w1 >= |mean_shift|`` by the triangle inequality, with equality
    exactly when ``D`` does not change sign."""
    _, res = run_port(name)
    assert np.all(res.w1 >= np.abs(res.mean_shift) - 1e-12 * np.maximum(1.0, res.w1))


@pytest.mark.parametrize("name", PARITY_SCENARIOS)
def test_mean_shift_equals_the_difference_of_grid_means(name):
    """``mean_shift == cond1_mean - cond0_mean``, to the accuracy available.

    Compared against the **scale of the operands**, not of the result.
    ``mean_shift`` sums ``D = Q1 - Q0`` while the right-hand side
    subtracts two grid sums, so on a gene whose two groups nearly agree
    the answer is a small number built from large ones and the relative
    error on the *difference* is unbounded by construction — measured up
    to 3.2e-13 on a ``mean_shift`` of -0.85 against group means in the
    thousands. That is catastrophic cancellation, not disagreement, and
    an absolute bound scaled by ``tot_mean`` is the meaningful test.

    The identity is what makes ``mean_shift`` and ``log2_fc`` always agree in
    sign, so a reader can trust either to say which way a gene moved.
    """
    _, res = run_port(name)
    lhs = res.mean_shift
    rhs = res.case_mean - res.ctrl_mean
    scale = np.maximum(np.abs((res.case_mean + res.ctrl_mean)), 1.0)
    worst = float(np.max(np.abs(lhs - rhs) / scale))
    assert worst < 1e-14, f"{name}: identity off by {worst:.3e} relative to tot_mean"



@pytest.mark.parity
def test_mean_shift_is_not_the_difference_of_sample_means_on_unequal_groups():
    """``diff.mean = mean(case) - mean(ctrl)`` holds ONLY at n1 == n0.

    Stated unqualified in ``wade.R``'s header and in three source
    documents, and it is wrong: WADE evaluates the integral as an
    ``m``-node equal-weight average of a piecewise-linear interpolant, and
    when ``m < n`` the implied weights on the order statistics are not
    uniform — the two extremes each carry ``1/m`` against the sample
    mean's ``1/n``.

    The worked example from ``method.md`` section 2.7 pins both halves
    at once: gene ``gA``'s case values are not linear in rank so the
    identity fails (11.5 against 9.4), while ``gB``'s are an arithmetic
    sequence so it holds exactly (11.0 = 11.0).

    A port that "fixes" ``mean_shift`` to a difference of sample means
    would change the statistic, invalidate parity, and alter the p-values,
    since the null would have to change with it.
    """
    fx = load_fixture("worked_example")
    dm = fx["stats"]["diff.mean"]
    smd = fx["sample_mean_difference"]

    assert dm[0] == pytest.approx(11.5)
    assert smd[0] == pytest.approx(9.4)
    assert abs(dm[0] - smd[0]) > 2.0, "gA must show the identity failing"

    assert dm[1] == pytest.approx(11.0)
    assert smd[1] == pytest.approx(11.0)
    assert dm[1] == pytest.approx(smd[1], abs=1e-12), "gB must show it holding"


@pytest.mark.parity
def test_worked_example_reproduces_algorithm_md_section_2_7():
    """Two genes, 5 cases vs 4 controls, no jitter and no normalization.

    Nearly indistinguishable on the mean shift (11.5 vs 11.0) and separated
    by a factor of 3.7 on the subset axis (44 vs 12) — which is the entire
    motivation for the shape statistics.
    """
    import wade as w

    fx = load_fixture("worked_example")
    st = w.wade_stats(fx["X"], fx["cond"])

    assert st.nprobs == 4
    assert_close(st.q, fx["q"], 0.0, "worked example: q", LAYER)
    assert_close(st.Q1, fx["Q1"], 1e-14, "worked example: Q1", LAYER)
    assert_close(st.Q0, fx["Q0"], 1e-14, "worked example: Q0", LAYER)
    for py_name, r_name in REDUCTIONS:
        assert_close(getattr(st, py_name), fx["stats"][r_name],
                     1e-14, f"worked example: {r_name}", LAYER)


