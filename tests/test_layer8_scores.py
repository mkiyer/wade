"""Layer 8 — the rank scores.

Asserted from the conventions upward: the empirical-CDF definition and
``dense_rank``'s tie and missing-value behaviour first, then the scores
built on them, then the ranks. Getting the ECDF's ``<=`` wrong changes
every score by a bounded amount that still looks like a score.

These are nomination heuristics and not part of the test, which is why
they are off by default in :func:`wade.wade` and live in their own
module. A rank of 1 is not a claim of significance.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import PARITY_SCENARIOS, assert_close, load_fixture
from portrun import run_port

import wade

LAYER = "layer 8 scores"


@pytest.mark.parity
def test_ecdf_convention_matches_R():
    """``F(t) = #{x <= t}/n``: right-continuous, ties shared, F(max) = 1."""
    fx = load_fixture("ecdf_denserank")
    for case in fx["ecdf_cases"]:
        got = wade.ecdf_values(case["x"])
        assert_close(got, case["ecdf_at_x"], 0.0, f"ecdf {case['label']}", LAYER)
        assert np.array_equal(got, case["ecdf_at_x"])


def test_ecdf_uses_less_than_or_equal_not_less_than():
    """The consequences the scores depend on."""
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    f = wade.ecdf_values(x)
    assert np.allclose(f, [0.6, 0.4, 0.8, 0.4, 1.0])
    assert f.max() == 1.0, "F must reach 1 at the maximum, so (1 - F) can be exactly 0"
    assert f.min() > 0.0, "F is never 0 at an observed point"


@pytest.mark.parity
def test_dense_rank_matches_R():
    fx = load_fixture("ecdf_denserank")
    for case in fx["dense_rank_cases"]:
        got = wade.dense_rank_desc(case["x"])
        want = np.asarray(case["dense_rank_desc"], dtype=np.float64)
        assert np.array_equal(np.isnan(got), np.isnan(want))
        ok = ~np.isnan(want)
        assert np.array_equal(got[ok], want[ok])


def test_dense_rank_ties_share_a_rank_with_no_gap_and_missing_propagates():
    got = wade.dense_rank_desc(np.array([0.5, 0.5, 0.2, np.nan, 0.9]))
    assert got[0] == got[1] == 2
    assert got[2] == 3, "the next rank after a tie is consecutive, not skipped"
    assert np.isnan(got[3])
    assert got[4] == 1, "rank 1 is the best score"


@pytest.mark.parity
@pytest.mark.parametrize("name", PARITY_SCENARIOS)
def test_scores_and_ranks(name):
    fx, res = run_port(name)
    s = res.scores
    assert_close(s["log2fc"], fx["score"]["log2fc"], 1e-13, f"{name}: log2fc", LAYER)
    assert_close(s["score"], fx["score"]["score"], 1e-13, f"{name}: score", LAYER)
    assert_close(s["tail_score"], fx["score"]["tail.score"], 1e-13,
                 f"{name}: tail.score", LAYER)

    for key, r_key in (("rank", "rank"), ("tail_rank", "tail.rank")):
        got = s[key]
        want = np.asarray(fx["score"][r_key], dtype=np.float64)
        assert np.array_equal(np.isnan(got), np.isnan(want))
        ok = ~np.isnan(want)
        assert np.array_equal(got[ok], want[ok]), f"{name}: {r_key} differs"


@pytest.mark.parametrize("name", PARITY_SCENARIOS)
def test_both_scores_lie_in_minus_one_to_one(name):
    _, res = run_port(name)
    for key in ("score", "tail_score"):
        v = res.scores[key]
        ok = np.isfinite(v)
        assert np.all(np.abs(v[ok]) <= 1.0 + 1e-12)


def test_the_two_scores_are_not_the_same_construction_with_one_term_swapped():
    """A documented asymmetry, asserted so it cannot be 'tidied' away.

    ``score`` takes its magnitude from the ECDF of an **absolute** value
    and its sign separately from ``diff_frac``. ``tail_score`` takes both
    from the ECDF of a **signed** value, so a gene strongly *down* in the
    case tail gets a small negative score rather than a large one. The
    reference's own header describes them as the same construction with
    one term swapped; they are not.
    """
    # cond0 must vary: (1 - Fctrl) is exactly 0 for the gene with the highest
    # control mean, so identical control means would zero every score.
    fc = np.array([4.0, 0.25, 1.0, 2.0])       # |log2fc| = 2, 2, 0, 1
    cond1 = np.array([10.0, 10.0, 10.0, 10.0])
    cond0 = np.array([1.0, 1.0, 1.0, 9.0])
    tail = np.array([50.0, -50.0, 0.0, 5.0])
    diff_frac = np.array([0.5, -0.5, 0.0, 0.5])

    s = wade.wade_score(fc, cond1, cond0, tail, diff_frac)

    # The two large-|log2fc| genes get the SAME magnitude from Ffc...
    assert abs(s["score"][0]) == pytest.approx(abs(s["score"][1]))
    # ...but the strongly-down tail gene gets a SMALL magnitude, not a large one.
    assert abs(s["tail_score"][1]) < abs(s["tail_score"][0])
    assert s["tail_score"][1] < 0


def test_scores_are_off_by_default():
    """They are heuristics; the caller has to ask for them."""
    rng = np.random.default_rng(0)
    counts = rng.poisson(20, size=(12, 15)).astype(float)
    cond = np.r_[np.ones(8, int), np.zeros(7, int)]
    res = wade.wade(counts, np.ones(12), cond, nperms=0)
    assert res.scores is None
    assert "score" not in res.columns()
