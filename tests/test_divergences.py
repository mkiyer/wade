"""The one place a correct port must disagree with the R.

``tail.conc`` used to be the other, but the statistic it guarded has been
retired along with the whole tail window (``docs/method.md`` §7), so there is
no longer a ratio to disagree about. What remains is the reshape defect, which
is about the *shared* quantile machinery and therefore still live.

The disagreement is asserted **positively** — stating what R does and what the
port does instead — because an untested divergence is indistinguishable from an
oversight.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import load_fixture
from portrun import run_port

import wade

LAYER = "divergences"


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
    collapses to the single point p = 1, so ``mean_shift``, ``tail_mean``
    and ``w1`` all reduce to the difference of group maxima.
    """
    fx = load_fixture("onesample")
    with pytest.raises(ValueError, match="min\\(n_case, n_ctrl\\) == 1"):
        wade.wade(
            fx["counts"], fx["normalizer"]["data"], fx["cond"],
            nperms=0, jitter=fx["jitter"], subset=False,
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
        nperms=0, jitter=fx["jitter"], allow_single_sample_group=True, subset=False,
    )
    g = int(fx["shapes"]["g"])
    assert res.mean_shift.shape == (g,)
    assert np.unique(res.mean_shift).size > 1, "genes must not share one value"

    cond = fx["cond"]
    case_max = res.tpm[:, cond == 1].max(axis=1)
    ctrl_max = res.tpm[:, cond == 0].max(axis=1)
    assert np.allclose(res.mean_shift, case_max - ctrl_max, rtol=1e-15)
    assert np.allclose(res.w1, np.abs(case_max - ctrl_max), rtol=1e-15)

    # ... and the port disagrees with R, which is the point.
    assert not np.allclose(res.mean_shift, np.full(g, fx["stats"]["diff.mean"][0]))
