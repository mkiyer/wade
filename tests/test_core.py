"""Independent mathematical and computational checks of the production contract."""
from fractions import Fraction
from itertools import combinations
import numpy as np
import pytest
from scipy.stats import wasserstein_distance, false_discovery_control

import wade
from wade._wasserstein import HAVE_RUST_KERNEL, TIE_RTOL, AreaState, infer
from wade.pvalues import sequential_p

BACKENDS = ["numpy"] + (["rust"] if HAVE_RUST_KERNEL else [])


def exact_areas(case, control):
    """Integrate the quantile-function difference on the merged grid in rationals.

    Independent of the production CDF sweep: it walks probability, not expression.
    """
    a, b = sorted(map(Fraction, case)), sorted(map(Fraction, control))
    edges = sorted({Fraction(i, len(a)) for i in range(len(a) + 1)} | {Fraction(j, len(b)) for j in range(len(b) + 1)})
    up = down = Fraction(0)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mid = (lo + hi) / 2
        d = a[int(mid * len(a))] - b[int(mid * len(b))]
        up, down = (up + d * (hi - lo), down) if d > 0 else (up, down - d * (hi - lo))
    return up, down


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("sizes", [(2, 2), (3, 17), (40, 60)])
def test_areas_and_stopping_records_against_an_exact_oracle(backend, sizes):
    n1, n0 = sizes
    r = np.random.default_rng(71)
    x = np.r_[np.zeros((1, n1+n0)), r.integers(0, 12, (7, n1+n0))].astype(float)
    x[1, :n1] //= 3
    x[1, 0] = 40                                  # a raised case above a lowered bulk
    cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
    labels = wade.draw_perms(cond, 79, seed=88)
    state = AreaState(x, cond, stop=7)
    state.update(labels[:11], backend=backend)
    state.update(labels[11:], backend=backend)
    for j, row in enumerate(x):
        observed = exact_areas(row[:n1], row[n1:])
        null = [exact_areas(row[p == 1], row[p == 0]) for p in labels]
        for d in range(2):
            assert state.observed[j, d] == pytest.approx(float(observed[d]), abs=1e-13)
            # Integer data make every sum exact, so exact ties are inclusive hits.
            hits = np.array([area[d] >= observed[d] for area in null])
            crossing = np.flatnonzero(hits)
            used = crossing[6]+1 if len(crossing) >= 7 else len(labels)
            assert state.tried[j, d] == used
            assert state.exceed[j, d] == hits[:used].sum()
    assert (state.tried[0] == 7).all()
    assert (sequential_p(state.exceed[0], state.tried[0], stop=7, maximum=79) == 1).all()


def test_areas_are_the_positive_and_negative_parts_of_w1_and_the_mean_difference():
    x = np.random.default_rng(5).lognormal(2, 1.5, (6, 37))
    cond = np.r_[np.ones(12, int), np.zeros(25, int)]
    state = AreaState(x, cond)
    for row, (up, down) in zip(x, state.observed):
        assert up + down == pytest.approx(wasserstein_distance(row[:12], row[12:]), rel=1e-12)
        assert up - down == pytest.approx(row[:12].mean() - row[12:].mean(), rel=1e-12, abs=1e-12)


def test_a_raised_subset_keeps_its_area_above_a_lowered_bulk():
    state = AreaState(np.array([[2, 2, 2, 30, 10, 10, 10, 10]], float), [1]*4 + [0]*4)
    np.testing.assert_array_equal(state.observed, [[5, 6]])


@pytest.mark.parametrize("backend", BACKENDS)
def test_an_untested_direction_draws_nothing(backend):
    x = np.random.default_rng(3).poisson(20, (4, 12)).astype(float)
    cond = [1]*6 + [0]*6
    state = AreaState(x, cond, active=(True, False), stop=5)
    state.update(wade.draw_perms(cond, 60, seed=1), backend=backend)
    assert (state.tried[:, 1] == 0).all() and (state.exceed[:, 1] == 0).all()
    assert (state.tried[:, 0] > 0).all()
    _, p, _, _ = infer(x, cond, active=(False, True), maximum=60, seed=1, strata=None, backend=backend)
    assert np.isnan(p[:, 0]).all() and np.isfinite(p[:, 1]).all()
    with pytest.raises(ValueError, match="direction"):
        AreaState(x, cond, active=(False, False))


@pytest.mark.validation
def test_sequential_rule_is_superuniform_at_every_attainable_pvalue():
    for cap in (3, 8, 31, 499):
        for stop in (1, 2, min(20, cap)):
            t = np.arange(stop, cap+1)
            k = np.arange(stop)
            mass = np.r_[stop/(t*(t+1)), np.full(stop, 1/(cap+1))]
            p = sequential_p(np.r_[np.full(len(t), stop), k], np.r_[t, np.full(stop, cap)], stop=stop, maximum=cap)
            assert mass.sum() == pytest.approx(1)
            for alpha in np.unique(p):
                assert mass[p <= alpha].sum() <= alpha+2e-14


@pytest.mark.parametrize("hits,draws", [([1], [3]), ([21], [39]), ([-1], [39]), ([20], [10]), ([.5], [39]), ([np.nan], [39])])
def test_incomplete_records_cannot_be_reported(hits, draws):
    with pytest.raises(ValueError):
        sequential_p(hits, draws, stop=20, maximum=39)


def test_exhaustive_small_orbit_is_calibrated_in_each_direction_and_mirrors_labels():
    x = np.array([[0, 0, 0, 2, 4, 4], [0, 0, 0, 0, 0, 100], [3, 3, 3, 3, 3, 3], [1, 9, 2, 8, 3, 7]], float)
    orbit = []
    for idx in combinations(range(6), 3):
        p = np.zeros(6, np.uint8)
        p[list(idx)] = 1
        orbit.append(p)
    stats = np.array([AreaState(x, p).observed for p in orbit])        # orbit x genes x directions
    for gene in range(len(x)):
        for d in range(2):
            row = stats[:, gene, d]
            p = (row[None, :] >= row[:, None]*(1-TIE_RTOL)).mean(axis=1)
            for alpha in np.unique(p):
                assert np.mean(p <= alpha) <= alpha+1e-14
    # Swapping case and control labels exchanges the two areas.
    np.testing.assert_array_equal(AreaState(x, 1-orbit[0]).observed, stats[0][:, ::-1])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("direction", ["both", "up"])
def test_chunk_batch_strata_and_filter_invariance(backend, direction):
    rng = np.random.default_rng(85)
    x = rng.poisson(30, (30, 20))
    x[0] = 0
    x[1, :10] *= 8
    cond = np.tile([1, 0], 10)
    strata = np.repeat(np.arange(4), 5)
    kwargs = dict(direction=direction, max_permutations=97, seed=17, strata=strata, backend=backend)
    a = wade.wade(x, cond, gene_chunk=30, **kwargs)
    b = wade.wade(x, cond, gene_chunk=3, **kwargs)
    for key in ("area_up", "area_down", "pvalue_up", "pvalue_down", "padj_up", "padj_down",
                "n_permutations", "n_exceedances"):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    mask = np.arange(30) % 2 == 0
    c = wade.wade(x, cond, test_genes=mask, **kwargs)
    np.testing.assert_array_equal(a.normalization.size_factors, c.normalization.size_factors)
    np.testing.assert_array_equal(a.pvalue_up[mask], c.pvalue_up[mask])
    active = (True, direction == "both")
    for batch in (1, 13, 97):
        out = infer(a.expression[1:3], cond, active=active, maximum=97, seed=17, strata=strata,
                    backend=backend, batch=batch)
        np.testing.assert_array_equal(out[1][:, 0], a.pvalue_up[1:3])
        np.testing.assert_array_equal(out[3], a.n_permutations[1:3])


def test_one_direction_is_the_matching_half_of_a_both_run():
    x = np.random.default_rng(9).poisson(25, (20, 16))
    x[1, :8] *= 5
    x[2, :8] //= 5
    cond = [1]*8 + [0]*8
    both = wade.wade(x, cond, max_permutations=199, seed=4)
    for d, column in (("up", 0), ("down", 1)):
        one = wade.wade(x, cond, direction=d, max_permutations=199, seed=4)
        np.testing.assert_array_equal(getattr(one, f"pvalue_{d}"), getattr(both, f"pvalue_{d}"))
        np.testing.assert_array_equal(one.n_permutations[:, column], both.n_permutations[:, column])
        np.testing.assert_array_equal(one.area_up, both.area_up)
        other = "down" if d == "up" else "up"
        assert np.isnan(getattr(one, f"pvalue_{other}")).all() and np.isnan(getattr(one, f"padj_{other}")).all()


def test_qvalues_follow_the_declared_hypothesis_and_excluded_genes_have_no_test():
    x = np.random.default_rng(53).poisson(30, (15, 12))
    x[0] = 0
    genes = ["gene0", "gene2", "gene3"]
    both = wade.wade(x, [1]*6+[0]*6, max_permutations=39, test_genes=genes)
    assert list(both.status[:4]) == ["all_zero", "filtered", "tested", "tested"]
    assert np.isnan(both.pvalue_up[~both.tested]).all() and np.isnan(both.area_up[~both.tested]).all()
    assert (both.n_permutations[~both.tested] == 0).all()
    t = both.tested
    joint = false_discovery_control(np.r_[both.pvalue_up[t], both.pvalue_down[t]])
    np.testing.assert_allclose(np.r_[both.padj_up[t], both.padj_down[t]], joint)
    assert both.params["hypotheses"] == 2 * t.sum()
    up = wade.wade(x, [1]*6+[0]*6, direction="up", max_permutations=39, test_genes=genes)
    np.testing.assert_allclose(up.padj_up[t], false_discovery_control(up.pvalue_up[t]))
    assert up.params["hypotheses"] == t.sum() and np.isnan(up.padj_down).all()
    assert list(both.columns()) == list(wade.RESULT_COLUMNS)


@pytest.mark.parametrize("scale", ["linear", "log"])
def test_scale_sets_the_tested_expression_and_descriptions_stay_log2(scale):
    x = np.random.default_rng(11).poisson(40, (5, 14))
    x[1, :3] *= 20
    cond = np.r_[np.ones(7, int), np.zeros(7, int)]
    result = wade.wade(x, cond, scale=scale, max_permutations=19)
    linear = x / result.normalization.size_factors
    expected = linear if scale == "linear" else np.log2(linear + 1)
    np.testing.assert_array_equal(result.expression, expected)
    np.testing.assert_allclose(np.c_[result.area_up, result.area_down], AreaState(expected, cond).observed)
    other = wade.wade(x, cond, scale="log" if scale == "linear" else "linear", max_permutations=19)
    np.testing.assert_allclose(result.bulk_shift, other.bulk_shift, rtol=1e-12)
    np.testing.assert_allclose(result.upper_tail, other.upper_tail, rtol=1e-12, atol=1e-12)


def test_opposing_tails_do_not_cancel_and_description_has_no_significance():
    from wade._descriptions import describe
    x = np.array([[0, 0, 4, 4, 2, 2, 2, 2]], float)
    cond = np.array([1]*4+[0]*4)
    np.testing.assert_array_equal(AreaState(x, cond).observed, [[1, 1]])
    bulk, upper, lower = describe(x, cond)
    np.testing.assert_allclose([bulk[0], upper[0], lower[0]], [0, 2, -2])


def test_quantile_detail_integrates_to_the_exact_areas_with_unequal_sizes():
    from wade.diagnostics import gene_detail
    x = np.array([0, 0, 1, 10, 2, 3, 4], float)
    cond = np.array([1, 1, 1, 1, 0, 0, 0])
    detail = gene_detail(x, cond)
    np.testing.assert_allclose([detail.area_up, detail.area_down], AreaState(x[None], cond).observed[0])
    assert detail.area_up + detail.area_down == pytest.approx(wasserstein_distance(x[:4], x[4:]))
    assert detail.area_up - detail.area_down == pytest.approx(x[:4].mean()-x[4:].mean())


def test_drivers_are_the_cases_beyond_every_control():
    x = wade.as_counts(np.array([[5, 90, 40, 1, 7, 8, 9, 10], [0, 3, 1, 9, 4, 5, 6, 7]]), sample_names=list("abcdefgh"))
    unit = wade.Normalization.from_size_factors(x, dict.fromkeys("abcdefgh", 1.0), source="test")
    result = wade.wade(x, [1]*4 + [0]*4, normalization=unit, max_permutations=19)
    np.testing.assert_array_equal(result.drivers(0, "up"), ["b", "c"])
    np.testing.assert_array_equal(result.drivers(1, "down"), ["a", "c", "b"])
    np.testing.assert_array_equal(result.drivers(1, "up"), ["d"])
    with pytest.raises(ValueError):
        result.drivers(0, "both")


@pytest.mark.kernel
@pytest.mark.skipif(not HAVE_RUST_KERNEL, reason="native kernel not built")
def test_native_boundary_rejects_invalid_indices():
    from wade import _kernel
    state = AreaState(np.ones((2, 6)), [1, 1, 1, 0, 0, 0])
    state.order[0, 0] = 100
    with pytest.raises(ValueError):
        _kernel.update_areas(state.order, state.gaps, wade.draw_perms([1, 1, 1, 0, 0, 0], 2), state.thresholds,
                             state.exceed, state.tried, state.active, 3, 20)


@pytest.mark.parametrize("kwargs", [{"max_permutations": 0}, {"max_permutations": 2.5}, {"gene_chunk": 0},
                                    {"backend": "wrong"}, {"direction": "sideways"}, {"scale": "sqrt"}])
def test_invalid_settings_are_errors(kwargs):
    with pytest.raises(ValueError):
        wade.wade(np.ones((3, 4)), [1, 1, 0, 0], **kwargs)


def test_no_exchange_or_single_sample_design_is_rejected():
    x = np.ones((3, 4))
    with pytest.raises(ValueError, match="exchanges"):
        wade.wade(x, [1, 1, 0, 0], strata=[1, 1, 0, 0])
    with pytest.raises(ValueError, match="two independent"):
        wade.wade(x, [1, 0, 0, 0])


def test_restricted_shuffle_preserves_each_group_and_missing_strata_rejected():
    from wade.permutation import validate_perms
    cond = np.array([1, 1, 0, 1, 0, 0])
    strata = np.repeat(["a", "b"], 3)
    labels = wade.draw_perms(cond, 101, seed=42, strata=strata)
    validate_perms(labels, cond, 101, strata=strata)
    assert (labels[:, :3].sum(axis=1) == 2).all()
    assert (labels[:, 3:].sum(axis=1) == 1).all()
    with pytest.raises(ValueError, match="missing"):
        wade.draw_perms(cond, 3, strata=["a"]*5+[None])
