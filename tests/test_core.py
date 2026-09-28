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
    for column in ("bulk_shift", "subset_up_fraction", "subset_up_log2fc", "subset_down_fraction",
                   "subset_down_log2fc"):
        np.testing.assert_allclose(getattr(result, column), getattr(other, column), rtol=1e-12, equal_nan=True)


def test_opposing_subsets_do_not_cancel_and_are_both_learned():
    from wade._descriptions import describe
    x = np.array([[0, 0, 4, 4, 2, 2, 2, 2]], float)
    cond = np.array([1]*4+[0]*4)
    np.testing.assert_array_equal(AreaState(x, cond).observed, [[1, 1]])
    np.testing.assert_allclose(np.ravel(describe(x, cond)), [0, .5, 2, .5, -2])


def brute_walk(x, cond):
    """Every cut between distinct values, in plain Python; ties go to the stringent side."""
    n, n1 = len(x), int(sum(cond))
    n0 = n - n1
    best_up = best_down = (0.0, None)
    for cut in sorted(set(x)):
        if cut == max(x):
            continue
        above = [i for i in range(n) if x[i] > cut]
        k = len(above)
        excess = sum(cond[i] for i in above) / n1 - sum(1 - cond[i] for i in above) / n0
        z = excess / np.sqrt(k * (n - k) / (n1 * n0 * (n - 1)))
        if z > best_up[0] or (z == best_up[0] and best_up[1] is not None):      # highest cut wins ties
            best_up = (z, cut) if z > 0 else best_up
        if z < best_down[0]:                                                   # lowest cut wins ties
            best_down = (z, cut)
    up = [x[i] for i in range(n) if cond[i] and best_up[1] is not None and x[i] > best_up[1]]
    down = [x[i] for i in range(n) if cond[i] and best_down[1] is not None and x[i] <= best_down[1]]
    # A subset must sit on its own side of the control median.
    middle = np.median([x[i] for i in range(n) if not cond[i]])
    return (len(up) if up and np.median(up) > middle else 0,
            len(down) if down and np.median(down) < middle else 0)


def test_walk_matches_brute_force_on_tied_data():
    from wade._descriptions import subsets
    rng = np.random.default_rng(12)
    for n1, n0 in ((3, 4), (7, 5), (20, 31)):
        cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
        x = rng.integers(0, 6, (60, n1 + n0)).astype(float)
        x[:5, :n1] += 4                                 # some genes with a raised case subset
        _, up, _, _, down, _, _ = subsets(x, cond)
        for row, u, d in zip(x, up, down):
            assert (u, d) == brute_walk(row.tolist(), cond.tolist())


def test_learned_subsets_on_analytic_patterns():
    base = np.arange(1, 101) / 100
    patterns = {"spike": (np.r_[np.repeat(base, 9), np.full(100, 10.)], np.repeat(base, 10), 100, 0),
                "lowered bulk, raised subset": (np.r_[np.full(90, .5), np.full(10, 4.)], np.ones(100), 10, 90),
                "identical": (np.arange(1, 101.), np.arange(1, 101.), 0, 0)}
    for name, (case, control, n_up, n_down) in patterns.items():
        counts = np.r_[case, control][None] * 100
        names = [f"s{i}" for i in range(counts.shape[1])]
        unit = wade.Normalization.from_size_factors(wade.as_counts(counts, sample_names=names),
                                                    dict.fromkeys(names, 1.0), source="test")
        cond = np.r_[np.ones(len(case), int), np.zeros(len(control), int)]
        for scale in ("linear", "log"):
            r = wade.wade(counts, cond, normalization=unit, sample_names=names, max_permutations=19, scale=scale)
            up, down = r.subset(0, "up"), r.subset(0, "down")
            assert (len(up.samples), len(down.samples)) == (n_up, n_down), (name, scale)
            assert up.fraction == n_up / len(case) and down.fraction == n_down / len(case)
    counts = np.r_[np.full(90, 50), np.full(10, 400), np.full(100, 100)][None]
    names = [f"s{i}" for i in range(200)]
    unit = wade.Normalization.from_size_factors(wade.as_counts(counts, sample_names=names),
                                                dict.fromkeys(names, 1.0), source="test")
    for scale in ("linear", "log"):      # boundaries and cutoffs are log2, whatever the tested scale
        r = wade.wade(counts, np.r_[np.ones(100, int), np.zeros(100, int)], normalization=unit,
                      sample_names=names, max_permutations=19, scale=scale)
        up, down = r.subset(0, "up"), r.subset(0, "down")
        assert (len(up.samples), up.boundary, down.boundary) == (10, np.log2(401), np.log2(51))
        assert up.log2fc == pytest.approx(np.log2(401 / 101)) and down.log2fc == pytest.approx(np.log2(51 / 101))
        above = r.subset(0, "up", cutoff=np.log2(101))              # cases only: no control enters
        assert (len(above.samples), above.fraction, above.boundary) == (10, .1, np.log2(401))
        assert len(r.subset(0, "down", cutoff=np.log2(101)).samples) == 90
        assert len(r.subset(0, "up", cutoff=20).samples) == 0 and np.isnan(r.subset(0, "up", cutoff=20).log2fc)


def test_mathematically_tied_values_do_not_split_a_subset():
    counts = np.random.default_rng(4).integers(1, 50, (1, 40))
    r = wade.wade(counts, [1] * 20 + [0] * 20, normalization="library", max_permutations=19)
    # A lone gene under library normalization gives every sample the same value, up to rounding.
    assert r.subset(0, "up").fraction == 0 and len(r.subset(0, "up").samples) == 0


def test_table_and_subset_agree_for_every_gene():
    x = np.random.default_rng(31).poisson(30, (40, 30))
    x[:6, :4] *= 20
    x[6:9, 4:15] //= 4
    cond = np.r_[np.ones(15, int), np.zeros(15, int)]
    r = wade.wade(x, cond, max_permutations=19)
    for g in range(40):
        for d in ("up", "down"):
            s = r.subset(g, d)
            assert s.fraction == getattr(r, f"subset_{d}_fraction")[g]
            np.testing.assert_equal(s.log2fc, getattr(r, f"subset_{d}_log2fc")[g])
            assert set(s.samples) <= set(r.sample_names[:15])
            if len(s.samples):      # the boundary, given as a cutoff, selects the same cases
                np.testing.assert_array_equal(r.subset(g, d, cutoff=s.boundary).samples, s.samples)
        # Above every control z rises with each case, so a boundary never splits those cases
        # (values within the walk's tie tolerance of the top control are ties, not above it).
        walked = np.log2(r.expression[g] + 1)
        beyond = r.sample_names[:15][walked[:15] > walked[15:].max() * (1 + TIE_RTOL)]
        up = r.subset(g, "up").samples
        assert len(up) == 0 or set(beyond) <= set(up)
    assert all(len(r.subset(g, "up").samples) >= 4 for g in range(6))
    with pytest.raises(ValueError):
        r.subset(0, "both")


def test_a_subset_must_sit_on_its_own_side_of_the_controls():
    from wade._descriptions import describe
    # Every case below the controls, except that the single lowest sample is a control.
    x = np.array([[5, 6, 7, 8, 9, 1, 20, 21, 22, 23]], float)
    cond = np.array([1] * 5 + [0] * 5)
    bulk, up_fraction, up_fold, down_fraction, down_fold = (v[0] for v in describe(x, cond))
    assert up_fraction == 0 and np.isnan(up_fold)
    assert down_fraction == 1 and down_fold < 0


def test_quantile_detail_integrates_to_the_exact_areas_with_unequal_sizes():
    from wade.diagnostics import gene_detail
    x = np.array([0, 0, 1, 10, 2, 3, 4], float)
    cond = np.array([1, 1, 1, 1, 0, 0, 0])
    detail = gene_detail(x, cond)
    np.testing.assert_allclose([detail.area_up, detail.area_down], AreaState(x[None], cond).observed[0])
    assert detail.area_up + detail.area_down == pytest.approx(wasserstein_distance(x[:4], x[4:]))
    assert detail.area_up - detail.area_down == pytest.approx(x[:4].mean()-x[4:].mean())


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

