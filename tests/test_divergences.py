"""The two places a correct port MUST disagree with the R.

``ROADMAP.md``: "Two places the port must disagree with the R, so the
suite must not enforce agreement there." A parity suite that asserted
equality on either of these would be enforcing a defect.

Both disagreements are asserted **positively** — the tests state what the
R does, state what the port does instead, and check both — rather than
being handled by omitting a test. An untested divergence is
indistinguishable from an oversight.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import PARITY_SCENARIOS, TOL_REDUCTION, assert_close, load_fixture
from portrun import run_port

import wade
from wade.stats import tail_concentration

LAYER = "divergences"


# =====================================================================
# 1. tail_conc — the R's guard is defective
# =====================================================================

@pytest.mark.parity
@pytest.mark.parametrize("name", PARITY_SCENARIOS)
def test_the_components_are_parity_targets_even_though_the_ratio_is_not(name):
    """Assert numerator and denominator, never the ratio.

    This is the cleanest of the three options in
    ``docs/porting-hazards.md`` hazard 5, and it generalizes: comparing a
    ratio's components separately localizes any disagreement in the ratio,
    and it lets the guard be tested against its own specification rather
    than against R's.
    """
    fx, res = run_port(name)
    assert_close(res.stats.tail_num, fx["stats"]["tail_num"],
                 TOL_REDUCTION, f"{name}: tail numerator", LAYER)
    assert_close(res.stats.tail_den, fx["stats"]["tail_den"],
                 TOL_REDUCTION, f"{name}: tail denominator", LAYER)


@pytest.mark.parity
def test_the_R_guard_fails_to_catch_the_pathology_it_exists_for():
    """The measurement that motivates the divergence, reproduced here.

    R guards at ``abs(diff.mean * nprobs) < 1e-8``, which reconstructs the
    ratio's own denominator and tests it against a fixed absolute
    threshold. On the ``tailconc`` fixture, built with deliberate
    bulk/tail cancellation, the guard NA's out essentially nothing while
    many genes return a "share of the signed area" far outside [0, 1].
    """
    fx, _ = run_port("tailconc")
    raw = fx["stats"]["tail.conc_raw"]
    guarded = fx["frame"]["tail.conc"]

    pathological = np.abs(raw) > 2
    assert pathological.sum() > 20, "the fixture should contain many pathological genes"
    assert np.nanmax(np.abs(raw)) > 10

    caught = np.isnan(guarded) & pathological
    assert caught.sum() == 0, (
        "R's 1e-8 guard is about nine orders of magnitude tighter than the "
        "cases that occur, so it catches none of them"
    )

    # And it is not that diff.mean is small: it is nowhere near the threshold.
    dm = np.abs(fx["stats"]["diff.mean"][pathological])
    assert dm.min() > 1e-4, f"smallest |diff.mean| among pathological genes: {dm.min():.3g}"


@pytest.mark.parity
def test_the_port_withholds_what_the_R_reports():
    """The disagreement, stated as an assertion rather than an omission."""
    fx, res = run_port("tailconc")
    raw = fx["stats"]["tail.conc_raw"]

    withheld = ~res.tail_conc_ok
    assert withheld.any(), "the port's guard must actually fire somewhere"

    # Every gene the port withholds is one R reported a value for.
    r_reported = ~np.isnan(fx["frame"]["tail.conc"])
    assert np.all(r_reported[withheld]), (
        "the port should be withholding values R reports, not the reverse"
    )

    # Where the port DOES report, it agrees with R's ratio.
    kept = res.tail_conc_ok
    assert kept.any()
    assert_close(res.tail_conc[kept], raw[kept], TOL_REDUCTION,
                 "tail.conc where the port reports it", LAYER)


def test_the_guard_bounds_its_own_output():
    """The property that makes the factor a promise rather than a knob.

    Because ``|sum(D over the tail)| <= sum(|D|)``, the condition
    ``sum(|D|) / |sum(D)| <= F`` implies ``|tail_conc| <= F``. The guard
    and the guarantee are the same number — which is also why raising it
    later widens what users were already told to expect.
    """
    rng = np.random.default_rng(23)
    for f in (1.5, 2.0, 3.0, 5.0, 10.0):
        d = rng.normal(size=(4000, 11)) * rng.lognormal(2, 1, size=(4000, 1))
        num = d[:, :2].sum(axis=1)
        den = d.sum(axis=1)
        value, ok = tail_concentration(num, den, f, np.abs(d).sum(axis=1))
        reported = value[ok]
        assert np.all(np.abs(reported) <= f + 1e-12), (
            f"F={f} promised |tail_conc| <= {f}, saw {np.abs(reported).max()}"
        )


def test_a_clamp_to_zero_one_would_be_the_wrong_fix():
    """Values slightly above 1 are legitimate and must survive.

    When the lower quantiles' differences are negative they cancel part of
    the tail's contribution, the denominator shrinks below the numerator,
    and the tail genuinely carries more than the net total. The worked
    construction from ``docs/algorithm.md`` section 2.5: a gene below
    controls across the bulk and above them in the top 2 nodes gives a
    tail sum of 674 against a total of 512, so ``tail_conc = 1.316``.
    """
    num = np.array([674.0])
    den = np.array([512.0])
    abs_sum = np.array([674.0 + 162.0])          # |negative bulk| = 674 - 512
    value, ok = tail_concentration(num, den, 3.0, abs_sum)
    assert ok[0]
    assert value[0] == pytest.approx(1.31640625)
    assert value[0] > 1.0, "a clamp to [0, 1] would silently rewrite this"


def test_a_constant_gene_is_withheld_rather_than_reported_as_nan():
    """The one pathology R's guard does catch: a constant gene's 0/0."""
    d = np.zeros((1, 8))
    value, ok = tail_concentration(np.zeros(1), np.zeros(1), 3.0, np.zeros(1))
    assert not ok[0]
    assert np.isnan(value[0])


def test_a_constant_count_gene_is_not_constant_after_jitter():
    """Why the exact-zero case is rarer in practice than it looks.

    The 'main' fixture plants a gene whose raw counts are all 12. It does
    **not** produce a missing ``tail_conc``, in either implementation,
    because the continuity jitter and the per-cell normalizer make every
    normalized value distinct — R reports 1.066 for it. So the one
    pathology R's guard does catch is reachable only from an exactly
    constant *normalized* row, which the raw-count entry point essentially
    never produces.

    Recorded because it is easy to assume the opposite and write a test
    that passes for the wrong reason.
    """
    fx, res = run_port("main")
    const = np.flatnonzero(np.ptp(fx["counts"], axis=1) == 0)
    assert const.size, "the 'main' fixture should contain a constant-count gene"
    for i in const:
        assert not np.isnan(fx["frame"]["tail.conc"][i])
        assert np.ptp(res.tpm[i]) > 0, "the jitter separates the values"


def test_an_exactly_constant_row_is_withheld_by_both():
    """Where the row really is constant, ``D`` is identically zero.

    ``tail_conc`` is then ``0/0``. R produces ``NaN`` and its guard
    converts it to ``NA``; the port's conditioning guard rejects it
    because ``sum(|D|) / |sum(D)|`` is itself ``0/0``. The two agree here
    even though the mechanism differs.
    """
    x = np.vstack([np.full(9, 5.0), np.arange(1.0, 10.0)])
    cond = np.r_[np.ones(5, int), np.zeros(4, int)]
    res = wade.wade_from_matrix(x, cond, nperms=0)
    assert not res.tail_conc_ok[0]
    assert np.isnan(res.tail_conc[0])
    assert res.stats.tail_den[0] == 0.0
    assert res.tail_conc_ok[1], "the non-constant gene must still be reported"


def test_the_factor_is_reported_in_the_result():
    _, res = run_port("main")
    assert res.params["tail_conc_max_factor"] == wade.DEFAULT_TAIL_CONC_MAX_FACTOR


def test_the_guard_refuses_a_factor_below_one():
    with pytest.raises(ValueError, match="at least 1"):
        tail_concentration(np.ones(1), np.ones(1), 0.9, np.ones(1))


# =====================================================================
# 2. A one-sample group — the R's reshape recycles a scalar
# =====================================================================

@pytest.mark.parity
def test_the_R_returns_one_recycled_value_for_every_gene():
    """Hazard 11, recorded from the fixture rather than described.

    ``matrixStats::rowQuantiles`` drops the dimension attribute when the
    result has a single row **or** a single column, and R's guard assumes
    the former. At ``nprobs == 1`` it means the latter, so a length-``g``
    vector is reshaped to ``1 x g`` — genes become probabilities — and
    every downstream ``rowSums`` reduces across *genes*. ``wade()`` then
    recycles the single value up the frame.

    No error, no warning, correct output shape, wrong answer for four of
    five genes.
    """
    fx = load_fixture("onesample")
    g = int(fx["shapes"]["g"])
    assert int(fx["shapes"]["nprobs"]) == 1
    assert fx["Q1"].shape == (1, g), "R transposes genes into probabilities here"

    dm = fx["stats"]["diff.mean"]
    assert dm.size == 1, f"R returns {dm.size} value(s) for a {g}-gene input"

    # And the frame carries that one value in all g rows.
    assert int(fx["frame"]["nrow"]) == g
    assert np.unique(fx["frame"]["diff.frac"]).size == 1


def test_the_port_refuses_a_one_sample_group():
    """Refusing loudly is better than the reference's silence.

    A one-sample group has no quantile function worth comparing: the grid
    collapses to the single point p = 1, so ``diff_mean``, ``tail_mean``
    and ``w1`` all reduce to the difference of group maxima.
    """
    fx = load_fixture("onesample")
    with pytest.raises(ValueError, match="min\\(n_case, n_ctrl\\) == 1"):
        wade.wade(
            fx["counts"], fx["normalizer"]["data"], fx["cond"],
            nperms=0, jitter=fx["jitter"],
        )


def test_the_port_can_compute_it_correctly_on_request():
    """And when it does, it gives one value per gene, not one value total.

    At ``nprobs == 1`` the specification is unambiguous: every statistic is
    the difference of the two group maxima, and ``tail_conc`` is
    identically 1. The port's answer is right and is still not R's.
    """
    fx = load_fixture("onesample")
    res = wade.wade(
        fx["counts"], fx["normalizer"]["data"], fx["cond"],
        nperms=0, jitter=fx["jitter"], allow_single_sample_group=True,
    )
    g = int(fx["shapes"]["g"])
    assert res.diff_mean.shape == (g,)
    assert np.unique(res.diff_mean).size > 1, "genes must not share one value"

    cond = fx["cond"]
    case_max = res.tpm[:, cond == 1].max(axis=1)
    ctrl_max = res.tpm[:, cond == 0].max(axis=1)
    assert np.allclose(res.diff_mean, case_max - ctrl_max, rtol=1e-15)
    assert np.allclose(res.tail_mean, case_max - ctrl_max, rtol=1e-15)
    assert np.allclose(res.w1, np.abs(case_max - ctrl_max), rtol=1e-15)

    # tail_conc is identically 1 where it is reported at all.
    ratio = res.stats.tail_num / res.stats.tail_den
    assert np.allclose(ratio, 1.0, rtol=1e-15)

    # ... and the port disagrees with R, which is the point.
    assert not np.allclose(res.diff_mean, np.full(g, fx["stats"]["diff.mean"][0]))
