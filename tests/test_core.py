"""Independent mathematical and computational checks of the production contract."""
from fractions import Fraction
from itertools import combinations
import math
import warnings
import numpy as np
import pytest
from scipy.stats import false_discovery_control

import wade
from wade._means import area_values, equalized, expression, jittered, mean_saddlepoint, mean_test
from wade import _kernel
from wade._subsets import split, subset_test, thinned, walks
from wade.permutation import STREAM_BLOCK, stream_block
from wade.pvalues import bound, sequential_p, settle, settled

STOP = 20                                                  # a stopping exceedance for direct test calls


def cells(case, control):
    """The merged-grid intervals in exact rationals: (width, case quantile, control quantile)."""
    a, b = sorted(map(Fraction, case)), sorted(map(Fraction, control))
    edges = sorted({Fraction(i, len(a)) for i in range(len(a) + 1)} | {Fraction(j, len(b)) for j in range(len(b) + 1)})
    return [(lo, hi - lo, a[int((lo + hi) / 2 * len(a))], b[int((lo + hi) / 2 * len(b))])
            for lo, hi in zip(edges[:-1], edges[1:])]


@pytest.mark.parametrize("sizes", [(2, 2), (3, 17), (40, 60)])
def test_the_signed_area_is_exactly_the_difference_in_means(sizes):
    from wade.diagnostics import gene_detail
    n1, n0 = sizes
    x = np.random.default_rng(71).integers(0, 50, (6, n1 + n0))
    cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
    for row in x:
        area = sum(w * (q1 - q0) for _, w, q1, q0 in cells(row[:n1], row[n1:]))
        assert area == Fraction(int(row[:n1].sum()), n1) - Fraction(int(row[n1:].sum()), n0)
        assert gene_detail(row.astype(float), cond).area == pytest.approx(float(area), abs=1e-12)


@pytest.mark.parametrize("direction", ["both", "up", "down"])
@pytest.mark.parametrize("sizes", [(2, 2), (3, 17), (40, 60)])
def test_mean_test_and_stopping_records_against_exact_mean_differences(direction, sizes):
    n1, n0 = sizes
    x = np.random.default_rng(71).integers(0, 12, (8, n1 + n0))
    x[0] = 0                                               # no difference at all
    x[1, :n1] //= 3
    x[1, 0] = 40                                           # a raised case above a lowered mean
    cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
    p, exceed, tried = mean_test(x.astype(float), cond, direction=direction, stop=STOP, maximum=79, seed=88)
    labels = stream_block(cond, 0, 88)[:79]                  # the run's stream: its first 79 shuffles
    difference = lambda row, lab: (Fraction(int(row[lab == 1].sum()), n1) - Fraction(int(row[lab == 0].sum()), n0))
    for j, row in enumerate(x):
        seen = difference(row, cond)
        shuffled = [difference(row, lab) for lab in labels]
        # Integer data make every difference exact, so exact ties are inclusive hits.
        hits = np.array([d >= seen if direction == "up" else d <= seen if direction == "down" else abs(d) >= abs(seen)
                         for d in shuffled])
        crossing = np.flatnonzero(hits)
        used = crossing[STOP - 1] + 1 if len(crossing) >= STOP else len(labels)
        assert (tried[j], exceed[j]) == (used, hits[:used].sum())
    assert p[0] == 1 and tried[0] == STOP


@pytest.mark.validation
def test_the_computed_tail_matches_exact_enumeration_of_every_labelling():
    n1, n0 = 9, 11
    x = np.random.default_rng(5).negative_binomial(5, 5 / 55, (40, n1 + n0)).astype(float)   # mean 50
    x[:, :n1] *= np.linspace(1, 3, 40)[:, None]
    cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
    totals = x[:, np.array(list(combinations(range(n1 + n0), n1)))].sum(axis=2)   # 167,960 labellings
    observed, centre = x[:, :n1].sum(axis=1), n1 * x.mean(axis=1)
    exact = {"up": (totals >= observed[:, None] - 1e-9).mean(axis=1),
             "both": (np.abs(totals - centre[:, None]) >= np.abs(observed - centre)[:, None] - 1e-9).mean(axis=1)}
    for direction, p in exact.items():
        tail = (p >= 1e-5) & (p <= .05)
        ratio = mean_saddlepoint(x, cond, direction=direction)[0][tail] / p[tail]
        assert tail.sum() >= 10 and .95 < np.median(ratio) < 1.05 and .6 < ratio.min() and ratio.max() < 1.6
    # The most extreme labelling has the smallest p-value there is; a constant gene has none.
    apart = np.r_[np.arange(n1) + 100.0, np.arange(n0)][None]
    assert mean_saddlepoint(apart, cond, direction="up")[0][0] == pytest.approx(1 / len(totals[0]), rel=1e-9)
    assert mean_saddlepoint(apart, cond, direction="both")[0][0] == pytest.approx(1 / len(totals[0]), rel=1e-9)
    assert mean_saddlepoint(np.full((1, n1 + n0), 7.0), cond, direction="both")[0][0] == 1
    # A total a hair from its average is no evidence at all, even for skewed values.
    skewed = np.exp(2 * np.random.default_rng(1).normal(size=(1, n1 + n0))) * 100
    n = n1 + n0
    skewed[0, 0] += (n1 * skewed.sum() - n * skewed[0, :n1].sum()) / n0             # cases' total at its average
    for nudge in (0.0, 1e-9, 1e-6, 1e-3):
        near = skewed + np.r_[nudge, np.zeros(n - 1)]
        assert mean_saddlepoint(near, cond, direction="both")[0][0] > .99
        assert .3 < mean_saddlepoint(near, cond, direction="up")[0][0] < .7
    tied = apart.copy()
    tied[0, n1 - 1] = tied[0, n1] = 50.0                        # the smallest case ties the largest control
    assert mean_saddlepoint(tied, cond, direction="up")[0][0] == pytest.approx(2 / len(totals[0]), rel=1e-9)


def test_auto_computes_the_area_test_and_shuffles_where_the_saddlepoint_is_less_precise():
    rng = np.random.default_rng(22)
    x = rng.poisson(40, (30, 24))
    x[:10, :12] *= 3
    x[10:20, 0] *= 60                                        # one towering case: a lumpy total
    cond = np.r_[np.ones(12, int), np.zeros(12, int)]
    run = lambda **k: wade.wade(x, cond, max_permutations=199, seed=3, fdr=None, **k)    # every run finished
    auto, brute = run(), run(pvalues="permutation")
    z = area_values(x, range(30), auto.normalization.size_factors, auto.params["seed"])
    computed, error = mean_saddlepoint(z, cond, direction="both")
    shuffled = error > auto.params["relative_error"]
    assert shuffled.any() and (~shuffled).any()
    np.testing.assert_array_equal(auto.area_permutations > 0, shuffled)          # only there, from the same shuffles
    np.testing.assert_array_equal(auto.area_pvalue[~shuffled], computed[~shuffled])
    np.testing.assert_array_equal(auto.area_pvalue[shuffled], brute.area_pvalue[shuffled])
    assert np.all(brute.area_permutations > 0)


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


def test_one_sided_tests_mirror_and_are_nested_in_the_two_sided_test():
    x = np.random.default_rng(9).poisson(25, (20, 16)).astype(float)
    x[1, :8] *= 5
    x[2, :8] //= 5
    cond = np.r_[np.ones(8, int), np.zeros(8, int)]
    run = lambda v, d: mean_test(v, cond, direction=d, stop=STOP, maximum=199, seed=4)[0]
    up, down, both = run(x, "up"), run(x, "down"), run(x, "both")
    np.testing.assert_array_equal(run(-x, "down"), up)                       # negating the data mirrors the sides
    np.testing.assert_array_equal(run(-x, "both"), both)
    raised = x[:, :8].mean(axis=1) > x[:, 8:].mean(axis=1)
    assert np.all(np.where(raised, up, down) <= both)                        # the declared side can only be sharper


@pytest.mark.parametrize("direction", ["both", "up"])
def test_chunk_and_filter_invariance(direction):
    rng = np.random.default_rng(85)
    x = rng.poisson(30, (30, 20))
    x[0] = 0
    x[1, :10] *= 8
    cond = np.tile([1, 0], 10)
    for fdr in (0.05, None):                                 # decisions settled early, and every run finished
        for pvalues in ("permutation", "auto"):              # shuffled throughout, and computed where it can be
            kwargs = dict(direction=direction, max_permutations=97, seed=17, pvalues=pvalues, fdr=fdr)
            a = wade.wade(x, cond, gene_chunk=30, **kwargs)
            b = wade.wade(x, cond, gene_chunk=3, **kwargs)
            for key in wade.RESULT_COLUMNS[2:] + ("area_permutations", "area_exceedances", "subset_permutations"):
                np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    kwargs = dict(direction=direction, max_permutations=97, seed=17, pvalues="permutation", fdr=None)
    a = wade.wade(x, cond, gene_chunk=30, **kwargs)
    mask = np.arange(30) % 2 == 0
    c = wade.wade(x, cond, test_genes=mask, **kwargs)
    np.testing.assert_array_equal(a.normalization.size_factors, c.normalization.size_factors)
    np.testing.assert_array_equal(a.area_pvalue[mask], c.area_pvalue[mask])
    z = area_values(x[1:3], range(1, 3), a.normalization.size_factors, a.params["seed"])   # per gene, not per chunk
    for rows in ([0], [1], [0, 1]):                           # a gene's shuffles do not depend on its company
        p, _, tried = mean_test(z[rows], cond, direction=direction, stop=a.params["stop_exceedances"],
                                maximum=97, seed=17)
        np.testing.assert_array_equal(p, a.area_pvalue[1:3][rows])
        np.testing.assert_array_equal(tried, a.area_permutations[1:3][rows])


def test_the_bound_never_rises_and_ends_at_the_finished_p_value():
    rng = np.random.default_rng(12)
    for stop, cap in ((1, 9), (5, 40), (20, 199), (25, 2000)):
        for chance in (0.0, 0.01, 0.2, 0.9):
            path = rng.random(cap) < chance                   # one gene's exceedances, shuffle by shuffle
            k = np.cumsum(path)
            t = np.arange(1, cap + 1)
            end = int(np.argmax(k >= stop)) if (k >= stop).any() else cap - 1
            final = sequential_p(k[end:end + 1], t[end:end + 1], stop=stop, maximum=cap)[0]
            u = np.r_[bound(0, 0, stop=stop, maximum=cap), bound(k[:end + 1], t[:end + 1], stop=stop, maximum=cap)]
            assert np.all(np.diff(u) <= 0) and np.all(u >= final) and u[-1] == final
            assert settled(k[end], t[end], stop=stop, maximum=cap, threshold=-np.inf)
            assert not settled(k[:end], t[:end], stop=stop, maximum=cap, threshold=-np.inf).any()


@pytest.mark.parametrize("seed", range(40))
def test_settling_each_decision_rejects_what_finished_runs_reject(seed):
    """Fischer, Barry and Ramdas's lemma 4.1 on correlated exceedance streams: shuffles shared by
    every gene (one latent effect per shuffle), mixed with p-values fixed in advance (genes the
    saddlepoint computes), at every level, budget and stopping exceedance."""
    rng = np.random.default_rng(seed)
    m, stop = int(rng.integers(30, 200)), int(rng.choice([1, 5, 25]))
    cap, alpha = int(rng.choice([199, 999, 4999])), float(rng.choice([0.01, 0.05, 0.2]))
    alt = rng.random(m) < rng.choice([0.0, 0.1, 0.5, 1.0])
    truth = np.where(alt, 10 ** rng.uniform(-6, -1, m), rng.random(m))
    rho = rng.choice([0.0, 0.9])
    latent = np.sqrt(1 - rho ** 2) * rng.standard_normal((m, cap)) + rho * rng.standard_normal(cap)
    streams = norm_cdf(latent) <= truth[:, None]               # exceedances, shuffle by shuffle
    live = rng.random(m) < 0.7
    fixed = np.where(live, np.nan, truth)
    exceed, tried = np.zeros(m, np.int64), np.zeros(m, np.int64)

    def advance(genes, threshold):
        for i in genes:
            k = exceed[i] + np.cumsum(streams[i, tried[i]:])
            t = tried[i] + np.arange(1, cap - tried[i] + 1)
            j = int(np.argmax(settled(k, t, stop=stop, maximum=cap, threshold=threshold)))
            exceed[i], tried[i] = k[j], t[j]

    full_k, full_t = exceed.copy(), tried.copy()
    for i in range(m):                                         # every run finished
        k = np.cumsum(streams[i])
        j = int(np.argmax(k >= stop)) if k[-1] >= stop else cap - 1
        full_k[i], full_t[i] = k[j], j + 1
    full = np.where(live, sequential_p(full_k, full_t, stop=stop, maximum=cap), truth)
    advance(np.flatnonzero(live), alpha * (1 - 100 * np.finfo(float).eps))
    reported, _ = settle(fixed, live, exceed, tried, advance, fdr=alpha, stop=stop, maximum=cap)
    q, q_full = wade.bh_adjust(reported), wade.bh_adjust(full)
    called = q_full <= alpha
    for level in (alpha, 1.5 * alpha, 3 * alpha):              # the same decisions at fdr and above
        np.testing.assert_array_equal(q <= level, q_full <= level)
    np.testing.assert_array_equal(reported[~called], full[~called])
    np.testing.assert_allclose(q[~called], q_full[~called], rtol=0, atol=1e-12)
    assert np.all(reported >= full) and np.all(q[called] <= alpha)
    assert np.all(tried[live] <= full_t[live])                 # never more shuffles than a finished run
    unfinished = live & (exceed < stop) & (tried < cap)
    assert np.all(called[unfinished])                          # only rejected genes stop early


def norm_cdf(z):
    return 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))


@pytest.mark.parametrize("maximum", [199, 19_999])
@pytest.mark.parametrize("pvalues", ["auto", "permutation"])
@pytest.mark.parametrize("planted", [True, False])
def test_wade_with_fdr_rejects_exactly_what_finished_runs_reject(maximum, pvalues, planted):
    rng = np.random.default_rng(31)
    x = rng.poisson(rng.gamma(4, 50, (80, 1)) * rng.gamma(10, 0.1, (80, 60)))
    if planted:
        x[:20, :30] *= 3                                       # every case tripled
        x[20:35, :6] *= 30                                     # a fifth of the cases raised
    cond = np.r_[np.ones(30, int), np.zeros(30, int)]
    unit = wade.Normalization.from_size_factors(x, np.ones(60), source="unit")
    kwargs = dict(max_permutations=maximum, seed=5, pvalues=pvalues, normalization=unit)
    fast, full = wade.wade(x, cond, fdr=0.05, **kwargs), wade.wade(x, cond, fdr=None, **kwargs)
    for test in ("area", "subset"):
        p, q = getattr(fast, test + "_pvalue"), getattr(fast, test + "_padj")
        p_full, q_full = getattr(full, test + "_pvalue"), getattr(full, test + "_padj")
        called = q_full <= 0.05
        assert called.any() == planted
        for level in (0.05, 0.1, 0.2):
            np.testing.assert_array_equal(q <= level, q_full <= level)
        np.testing.assert_array_equal(p[~called], p_full[~called])
        np.testing.assert_array_equal(q[~called], q_full[~called])
        assert np.all(p[called] >= p_full[called]) and np.all(q[called] <= 0.05)
        assert np.all(getattr(fast, test + "_permutations") <= getattr(full, test + "_permutations"))
        assert not np.any(fast.bounded[test] & ~called) and not full.bounded[test].any()
    if planted and maximum == 19_999:                          # strong genes stop long before the budget
        assert fast.bounded["subset"].any() and fast.subset_permutations.sum() < full.subset_permutations.sum() / 2
    assert fast.params["fdr"] == 0.05 and full.params["fdr"] is None


def lorenz_gaps(x, cond):
    """Each row's gap L_w(k/n1) - L_z(k/n1) on count + 1 at every case boundary, from first principles."""
    z = np.sort(x[:, cond == 1], axis=1) + 1
    w = np.sort(x[:, cond == 0], axis=1) + 1
    n1, n0 = z.shape[1], w.shape[1]
    k = np.arange(1, n1)
    case = np.cumsum(z, axis=1)[:, :-1] / z.sum(axis=1, keepdims=True)
    whole, part = (k * n0) // n1, (k * n0 % n1) / n1
    below = np.c_[np.zeros(len(w)), np.cumsum(w, axis=1)]
    control = (below[:, whole] + part * w[:, np.minimum(whole, n0 - 1)]) / w.sum(axis=1, keepdims=True)
    return control - case


def test_rarity_ranks_each_subset_against_genes_with_a_subset_of_its_size():
    from wade._rarity import cuts, gap_curves
    rng = np.random.default_rng(23)
    x = rng.poisson(rng.gamma(6, 40, (120, 1)) * rng.gamma(20, .05, (120, 50)))
    x[:6, :2] *= 40                                            # two tumours far out: rare subsets
    x[6:12, :12] *= 3                                          # common subsets of a dozen tumours
    x[12:20] //= 50                                            # low counts: the group below the range
    cond = np.r_[np.ones(25, int), np.zeros(25, int)]
    r = wade.wade(x, cond, max_permutations=99)
    values, labels = r.expression, r.cond
    n1 = int(labels.sum())
    curves = gap_curves(values, labels)
    np.testing.assert_allclose(curves, lorenz_gaps(values, labels), atol=1e-12)    # the walk is the Lorenz gap
    size, side, groups = r._subsets()
    k = cuts(size, side, n1)
    expected = np.full(len(x), np.nan)
    for group in (0, 1):                                       # brute force, group by group
        members = np.flatnonzero(groups == group)
        share = {g: (np.sum(curves[members, k[g] - 1] >= curves[g, k[g] - 1]) / len(members) if size[g] else 1.0)
                 for g in members}
        for g in members:
            expected[g] = sum(share[h] <= share[g] for h in members) / len(members)
    np.testing.assert_array_equal(r.subset_rarity, expected)
    assert set(groups[12:20]) == {1} and 0 in groups
    assert np.all(r.subset_rarity[size == 0] == 1)             # no subset: not rare at all
    for group in (0, 1):                                       # ranks: multiples of 1/m, the largest 1
        m = np.sum(groups == group)
        scaled = r.subset_rarity[groups == group] * m
        np.testing.assert_allclose(scaled, np.round(scaled), atol=1e-9)
        assert scaled.max() == pytest.approx(m)
    assert np.nanargmin(np.where(groups == 0, r.subset_rarity, np.nan)) < 6         # a far-out pair is the rarest
    # Each group is ranked only against itself: changing the low-count genes moves nothing else.
    y = x.copy()
    y[12:20] = rng.poisson(2, (8, 50))
    other = wade.wade(y, cond, max_permutations=99, normalization=r.normalization)
    keep = groups == 0
    np.testing.assert_array_equal(other.subset_rarity[keep], r.subset_rarity[keep])
    assert r.params["rarity_groups"] == dict(validated=int(np.sum(groups == 0)), below=int(np.sum(groups == 1)))


def test_the_sample_load_and_the_rare_subsets_point_to_a_contaminated_library():
    rng = np.random.default_rng(29)
    x = rng.poisson(rng.gamma(6, 40, (200, 1)) * rng.gamma(20, .05, (200, 80)))
    x[:12, 7] *= 30                                            # one library carries a foreign programme
    x[50, :8] *= 10                                            # and a gene raised in eight tumours
    cond = np.r_[np.ones(40, int), np.zeros(40, int)]
    r = wade.wade(x, cond, max_permutations=99, gene_names=[f"g{i}" for i in range(200)],
                  sample_names=[f"s{j}" for j in range(80)])
    load = r.sample_load(drivers=3)
    assert load["load"].mean() == pytest.approx(1.0)           # random membership would give every case 1
    top = int(np.argmax(load["load"]))
    assert load["sample"][top] == "s7" and load["load"][top] > 5
    assert set(load["drivers"][top]) <= {f"g{i}" for i in range(12)} and len(load["drivers"][top]) == 3
    assert load["subsets"][top] >= 12
    rare = r.rare_subsets(n=15)
    programme = [i for i, g in enumerate(rare["gene"]) if int(g[1:]) < 12]
    assert len(programme) >= 10                                 # the programme's genes are among the rarest
    for i in programme:                                         # and share their one member
        assert "s7" in rare["members"][i] and any(int(g[1:]) < 12 for g, _ in rare["shared"][i])
    assert list(rare["rarity"]) == sorted(rare["rarity"])


def test_qvalues_cover_the_tested_genes_and_excluded_genes_have_no_test():
    x = np.random.default_rng(53).poisson(30, (15, 12))
    x[0] = 0
    r = wade.wade(x, [1]*6 + [0]*6, max_permutations=39,
                  test_genes=["gene0", "gene2", "gene3"])
    assert list(r.status[:4]) == ["all_zero", "filtered", "tested", "tested"]
    t = r.tested
    for column in ("area_pvalue", "area_padj", "subset_pvalue", "subset_padj"):
        assert np.isnan(getattr(r, column)[~t]).all()
    assert (r.area_permutations[~t] == 0).all() and (r.subset_permutations[~t] == 0).all()
    np.testing.assert_allclose(r.area_padj[t], false_discovery_control(r.area_pvalue[t]))
    np.testing.assert_allclose(r.subset_padj[t], false_discovery_control(r.subset_pvalue[t]))
    assert r.params["hypotheses"] == t.sum() and list(r.columns()) == list(wade.RESULT_COLUMNS)


def test_expression_the_effects_and_the_thinned_counts_of_the_area_test():
    rng = np.random.default_rng(11)
    x = rng.poisson(400, (200, 14))
    x[1, :3] *= 20
    cond = np.r_[np.ones(7, int), np.zeros(7, int)]
    names = [f"s{i}" for i in range(14)]
    data = wade.as_counts(x, sample_names=names)
    norm = wade.Normalization.from_size_factors(data, dict(zip(names, np.linspace(0.5, 2, 14))), source="test")
    r = wade.wade(data, cond, normalization=norm, max_permutations=19)
    sf = r.normalization.size_factors                              # centred to geometric mean one
    jitter = jittered(x, range(len(x)), r.params["seed"]) - x
    assert np.all((jitter >= 0) & (jitter < 1))                  # each count moves within its unit
    np.testing.assert_array_equal(r.expression, (x + jitter) / sf)
    np.testing.assert_array_equal(wade.wade(data, cond, normalization=norm, max_permutations=19, gene_chunk=2).expression,
                                  r.expression)                  # the jitter follows the gene, not the chunk
    plain = x / sf                                               # the effects carry no jitter
    m1, m0 = plain[:, :7].mean(axis=1), plain[:, 7:].mean(axis=1)
    np.testing.assert_allclose(r.case_mean, m1)
    np.testing.assert_allclose(r.control_mean, m0)
    np.testing.assert_allclose(r.area, m1 - m0)
    np.testing.assert_allclose(r.area_fraction, (m1 - m0) / (m1 + m0))
    np.testing.assert_allclose(r.log2fc, np.log2(m1 / m0))
    assert r.params["depth_equalized"] == pytest.approx(sf.min() / np.median(sf))
    # The area test's counts: every sample thinned until its size factor equals the smallest; the
    # sample with the smallest keeps all its reads.
    thinned = equalized(x, range(len(x)), sf, r.params["seed"])
    np.testing.assert_array_equal(thinned[:, np.argmin(sf)], x[:, np.argmin(sf)])
    assert np.all(thinned <= x)
    np.testing.assert_array_equal(equalized(x[2:4], range(2, 4), sf, r.params["seed"]), thinned[2:4])
    keep, reads = sf.min() / sf, x.sum(axis=0)                     # each sample keeps min(s) / s of its reads
    assert np.all(np.abs(thinned.sum(axis=0) / reads - keep) <= 4 * np.sqrt(keep * (1 - keep) / reads))


def test_internal_control_genes_set_the_size_factors_that_the_thinning_equalizes():
    """Whatever the normalizer, it becomes one size factor per sample, and the area test thins by
    those factors, never by raw library totals."""
    rng = np.random.default_rng(5)
    depth = np.r_[1, 1, 2, 2, 1, 1, 2, 2]                          # half the samples sequenced twice as deep
    x = rng.poisson(200, (40, 8)) * depth
    x[0, 0] = 10**6                                                # one gene holds most of sample 0's reads
    x[1, 4:] = 0                                                   # a gene with no reads in the controls
    x[2, :4] = 0                                                   # and one with none in the cases
    cond = np.r_[np.ones(4, int), np.zeros(4, int)]
    names = [f"g{i}" for i in range(40)]
    data = wade.as_counts(x, gene_names=names)
    controls = names[10:30]
    norm = wade.fit_normalization(data, reference_genes=controls)
    assert norm.used_reference_genes == tuple(controls)
    sf = norm.size_factors
    np.testing.assert_allclose(sf / sf[0], depth, rtol=0.1)        # the controls' depths, not the totals
    r = wade.wade(data, cond, normalization=norm, max_permutations=19)
    np.testing.assert_array_equal(r.normalization.size_factors, sf)
    thinned = equalized(x, range(len(x)), sf, r.params["seed"])
    # Sample 0 holds the largest library, yet it keeps about min(s) / s_0 of its reads, nearly all.
    by_totals = x.sum(axis=0).min() / x[:, 0].sum()
    assert by_totals < 0.01 and thinned[0, 0] > 0.8 * x[0, 0] * sf.min() / sf[0]
    # The fold of the group means, with no pseudocount: infinite where a group has no reads.
    assert r.log2fc[1] == np.inf and r.area_fraction[1] == 1
    assert r.log2fc[2] == -np.inf and r.area_fraction[2] == -1
    fc = 2.0 ** r.log2fc[3:]
    np.testing.assert_allclose(r.area_fraction[3:], (fc - 1) / (fc + 1))


def registration_oracle(case, control):
    """The Wasserstein-1 registration: the weighted median of the log quantile ratio of count + 1 over
    the exact merged grid, the middle of the median interval when the weights below a value make
    exactly half."""
    pieces = sorted((math.log1p(q1) - math.log1p(q0), w) for _, w, q1, q0 in cells(case, control))
    below = Fraction(0)
    for k, (value, w) in enumerate(pieces):
        below += w
        if below >= Fraction(1, 2):
            if below == Fraction(1, 2) and k + 1 < len(pieces):
                value = (value + pieces[k + 1][0]) / 2
            return math.exp(value)


def split_oracle(x, cond):
    """The split in exact rational arithmetic: the lowest running departure from the mean shift's
    prediction, the smaller side of its cut, that side's fold over what the registration predicts
    for its quantiles, and the gap (the depth over the case mean + 1)."""
    a = sorted(Fraction(v) for v, c in zip(x, cond) if c)
    b = sorted(Fraction(v) for v, c in zip(x, cond) if not c)
    n1 = len(a)
    grid = cells(a, b)
    area = sum(w * (q1 - q0) for _, w, q1, q0 in grid)
    share = [q0 + 1 for *_, q0 in grid]
    total = sum(s * w for s, (_, w, _, _) in zip(share, grid))
    predicted = [q0 + area * s / total for s, (*_, q0) in zip(share, grid)]
    shift = registration_oracle(a, b)
    lowest, cut = None, None
    for i in range(1, n1):                                  # never the whole cohort
        if a[i - 1] == a[i]:                               # ties never straddle a boundary
            continue
        value = sum((q1 - p) * w for p, (lo, w, q1, _) in zip(predicted, grid) if lo + w <= Fraction(i, n1))
        if lowest is None or value <= lowest:              # ascending i: the highest boundary wins ties
            lowest, cut = value, i
    if lowest is None or lowest >= 0:
        return 0, 0, np.nan, 0.0, shift
    up = n1 - cut <= cut
    lo_i, hi_i = (cut, n1) if up else (0, cut)
    seen = sum(a[lo_i:hi_i]) / (hi_i - lo_i)
    other = sum(q0 * w for lo, w, _, q0 in grid if Fraction(lo_i, n1) <= lo and lo + w <= Fraction(hi_i, n1)) \
        * n1 / (hi_i - lo_i)
    fold = math.log2(float(seen + 1) / (shift * float(other + 1)))
    gap = float(-lowest / (sum(a) / n1 + 1))
    return hi_i - lo_i, 1 if up else -1, fold, gap, shift


def test_split_matches_an_exact_oracle_on_tied_data():
    rng = np.random.default_rng(12)
    for n1, n0 in ((2, 2), (3, 4), (7, 5), (12, 9)):
        cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
        x = rng.integers(0, 8, (40, n1 + n0)).astype(float)
        x[:10, :rng.integers(1, n1)] *= 9                 # raised cases
        x[10:20, n1 - 1] = 0                              # a lowered case
        x[20:22] = 3                                      # no change at all
        size, side, fold, gap, shift = split(x, cond)
        for g, row in enumerate(x):
            k, s, f, h, r = split_oracle(row.tolist(), cond.tolist())
            assert (size[g], side[g]) == (k, s)
            assert np.isnan(fold[g]) if np.isnan(f) else fold[g] == pytest.approx(f, abs=1e-9)
            assert gap[g] == pytest.approx(h, abs=1e-12) and shift[g] == pytest.approx(r, rel=1e-12)


def test_the_gap_is_the_largest_fall_of_the_cases_lorenz_curve_below_the_controls():
    rng = np.random.default_rng(5)
    n1, n0 = 13, 29
    x = rng.gamma(2, 30, (50, n1 + n0))
    x[:15, :3] *= 12
    cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
    _, _, _, gap, _ = split(x, cond)
    t = np.arange(1, n1) / n1
    integral = lambda v, u: np.interp(u, np.arange(len(v) + 1) / len(v), np.r_[0, np.cumsum(v)] / len(v))
    for row, h in zip(x, gap):
        z, w = np.sort(row[:n1]) + 1, np.sort(row[n1:]) + 1           # Lorenz curves of count + 1
        case, control = integral(z, t) / z.mean(), integral(w, t) / w.mean()
        assert h == pytest.approx(max((control - case).max(), 0), abs=1e-12)


def test_the_registration_is_the_middle_of_the_median_interval_and_swapping_inverts_it():
    rng = np.random.default_rng(3)
    for n1, n0 in ((2, 2), (4, 4), (5, 7), (30, 20)):
        x = rng.integers(0, 40, (30, n1 + n0)).astype(float)
        cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
        *_, shift = split(x, cond)
        *_, back = split(x, 1 - cond)
        for g, row in enumerate(x):
            assert shift[g] == pytest.approx(registration_oracle(row[:n1], row[n1:]), rel=1e-12, abs=1e-12)
        np.testing.assert_allclose(shift * back, 1, rtol=1e-12)


def unit(counts):
    names = [f"s{i}" for i in range(counts.shape[1])]
    data = wade.as_counts(counts, sample_names=names)
    return data, wade.Normalization.from_size_factors(data, dict.fromkeys(names, 1.0), source="test")


def test_a_shared_fold_has_no_subset_and_departures_either_way_are_found():
    control = 100 + 10 * np.arange(40)                     # margins far beyond the unit jitter
    shared = 3 * (control + 1) - 1                          # the same fold on count + 1 for every case
    raised, lowered = shared.copy(), shared.copy()
    raised[-8:] *= 50                                       # eight cases above the shared fold
    lowered[:8] = 0                                         # eight cases below it
    cond = np.r_[np.ones(40, int), np.zeros(40, int)]
    for case, direction in ((shared, "none"), (raised, "up"), (lowered, "down")):
        values = np.r_[case, control][None].astype(float)
        size, *_ = split(values, cond)
        assert size[0] == (0 if direction == "none" else 8)      # the walk itself, on exact values
        data, norm = unit(np.c_[np.r_[case, control], np.full(80, 100)].T)   # a steady gene keeps no sample empty
        r = wade.wade(data, cond, normalization=norm, max_permutations=99)
        s = r.subset(0)
        if direction == "none":
            assert s.pvalue > .05                                  # after jitter, only a chance departure
        else:
            assert s.side == direction, direction
            extreme = np.argsort(case)[::-1][:8] if direction == "up" else np.argsort(case)[:8]
            assert set(s.samples) == {f"s{i}" for i in extreme}
            assert s.fraction == .2 and (s.log2fc > 0) == (direction == "up")
            # A random split puts all eight among the cases with chance C(40, 8) / C(80, 8) =
            # 0.003, so the test can reach significance; four could not (0.058). Eight raised
            # cases are significant; eight lowered to zero weigh less on count + 1.
            if direction == "up":
                assert s.pvalue <= .05


def test_ties_never_split_a_subset_and_no_change_has_none():
    data = np.random.default_rng(4).integers(1, 50, (1, 40))
    r = wade.wade(data, [1] * 20 + [0] * 20, normalization="library", max_permutations=19)
    # A lone gene under library normalization gives every sample the same value, up to rounding.
    assert (r.subset_fraction[0], r.subset_pvalue[0]) == (0, 1) and np.isnan(r.subset_log2fc[0])
    size, side, fold, gap, _ = split(np.zeros((1, 8)), np.r_[[1] * 4, [0] * 4])
    assert (size[0], side[0], gap[0]) == (0, 0, 0) and np.isnan(fold[0])


def test_the_subset_test_and_its_records():
    rng = np.random.default_rng(8)
    control = rng.poisson(rng.gamma(20, 2.5, 60))           # tight around 50 counts
    raised = rng.poisson(rng.gamma(20, 2.5, 60))
    raised[:12] *= 20                                       # a fifth of the cases, twentyfold
    shared = 3 * (control + 1) - 1                          # every case tripled: the mean shift itself
    data, norm = unit(np.c_[np.r_[raised, control], np.r_[shared, control]].T)
    run = lambda **k: wade.wade(data, np.r_[np.ones(60, int), np.zeros(60, int)], normalization=norm, seed=1,
                                max_permutations=199, fdr=None, **k)             # every run finished
    r = run(pvalues="permutation")
    assert r.subset_pvalue[0] == 1 / 200 and r.subset_padj[0] <= .05             # carried by twelve cases
    assert r.subset_pvalue[1] > .05                                               # shared by every case
    s = r.subset(0)
    assert (s.side, len(s.samples), s.pvalue, s.padj) == ("up", 12, r.subset_pvalue[0], r.subset_padj[0])
    # A shuffled record stops at the twentieth exceedance or at the cap; a gene without a
    # subset needs no shuffles (p = 1).
    ran = r.subset_fraction > 0
    k, t = r.subset_exceedances, r.subset_permutations
    assert ran.any() and np.all((k[ran] == r.params["stop_exceedances"]) | (t[ran] == 199)) and not t[~ran].any()
    np.testing.assert_array_equal(run(pvalues="permutation", gene_chunk=1).subset_pvalue, r.subset_pvalue)
    np.testing.assert_array_equal(run().subset_pvalue, r.subset_pvalue)          # the subset test always shuffles


def test_one_sided_subset_tests_split_the_shuffles_of_the_two_sided_test():
    rng = np.random.default_rng(14)
    x = rng.poisson(40, (30, 40))
    x[:10, :4] *= 8                                          # raised subsets
    x[10:20, :5] //= 8                                       # lowered subsets
    cond = np.r_[np.ones(20, int), np.zeros(20, int)]
    ones, rows = np.ones(40), np.arange(30)
    run = lambda d, k: subset_test(x, ones, cond, rows, direction=d, stop=STOP, maximum=19, seed=6, draws=k)
    # One draw: a deeper subset on the observed side counts for that side; every other shuffle
    # counts for the opposite one, and the two-sided test counts depth on either side.
    (_, both, tried), (_, up, _), (_, down, _) = run("both", 1), run("up", 1), run("down", 1)
    _, side, *_ = split(thinned(x, ones, cond, rows, 6)[0], cond)       # the side of the walk the test shuffles
    ran = tried > 0
    assert ran.any() and np.all(tried[ran] == 19)
    same, other = np.where(side > 0, up, down), np.where(side > 0, down, up)
    assert np.all(same[ran] <= both[ran]) and np.all(same[ran] + other[ran] >= 19)
    # Several draws: "up" scores the mean of the draws' signed gaps and "down" its negative, so every
    # shuffle still counts for one side or the other. (The draws may disagree on the observed side,
    # so a side's count can exceed the two-sided count.)
    (_, up, tried), (_, down, _) = run("up", 4), run("down", 4)
    ran = tried > 0
    assert ran.any() and np.all(up[ran] + down[ran] >= 19)


def test_several_draws_score_the_mean_of_their_walks():
    rng = np.random.default_rng(15)
    genes, draws, n = 6, 3, 30
    x = rng.poisson(30, (genes * draws, n)) + rng.random((genes * draws, n))
    x[::draws, :3] *= 6                                      # a raised subset in each gene's first draw
    cond = np.r_[np.ones(15, np.uint8), np.zeros(15, np.uint8)]
    for direction in ("both", "up", "down"):
        pooled, order, mean = walks(x, cond, np.ones(len(x)), direction=direction, draws=draws)
        _, _, each = walks(x, cond, np.ones(len(x)), direction=direction)
        assert mean.shape == (genes,)
        assert np.allclose(mean, each.reshape(genes, draws).mean(axis=1), rtol=0, atol=1e-14)
        # The tally compares each shuffle's mean with the observed mean, gene by gene.
        labels = stream_block(cond, 0, 3)[:50]
        scores = np.array([_kernel.score(pooled, order, direction, row, draws) for row in labels])
        exceed, tried = _kernel.tally(pooled, order, direction, labels, mean, np.zeros(genes, np.uint64),
                                      np.zeros(genes, np.uint64), 50, 50, 0, -np.inf, draws)
        assert np.all(tried == 50)
        assert np.array_equal(exceed, np.sum(scores >= mean - 100 * np.finfo(float).eps * np.abs(mean), axis=0))
    with pytest.raises(ValueError, match="same number of draws"):
        _kernel.score(pooled, order, "both", cond, 4)


def test_one_draw_is_the_single_thinned_test_and_more_draws_use_new_streams():
    rng = np.random.default_rng(16)
    x = rng.poisson(25, (8, 40))
    cond = np.r_[np.ones(20, np.uint8), np.zeros(20, np.uint8)]
    factors, rows = np.exp(rng.normal(0, 0.3, 40)), np.arange(8)
    one, fold = thinned(x, factors, cond, rows, 9)
    four, folds = thinned(x, factors, cond, rows, 9, draws=4)
    assert four.shape == (32, 40) and folds.shape == (8, 4)
    assert np.array_equal(four[::4], one) and np.array_equal(folds[:, 0], fold)    # draw 0 is the single draw
    for d in range(1, 4):
        assert not np.array_equal(four[d::4], one)                                 # the others are new draws


def test_table_and_subset_agree_for_every_gene():
    x = np.random.default_rng(31).poisson(30, (40, 30))
    x[:6, :4] *= 20
    x[6:9, 4:15] //= 4
    cond = np.r_[np.ones(15, int), np.zeros(15, int)]
    r = wade.wade(x, cond, max_permutations=19)
    for g in range(40):
        s = r.subset(g)
        assert (s.fraction, s.side) == (r.subset_fraction[g], r.subset_side[g])
        np.testing.assert_equal(s.log2fc, r.subset_log2fc[g])
        assert set(s.samples) <= set(r.sample_names[:15])
        inside = np.isin(r.sample_names[:15], s.samples)
        if inside.any():                                    # the cases furthest out, never split by a tie
            values = r.expression[g, :15]
            assert values[inside].min() > values[~inside].max() if s.side == "up" else \
                values[inside].max() < values[~inside].min()
    assert all(r.subset(g).side == "up" and len(r.subset(g).samples) >= 4 for g in range(6))


@pytest.mark.parametrize("kwargs", [{"max_permutations": 0}, {"max_permutations": 2.5}, {"gene_chunk": 0},
                                    {"direction": "sideways"}, {"relative_error": 0}, {"relative_error": 1.5},
                                    {"pvalues": "exact"}])
def test_invalid_settings_are_errors(kwargs):
    with pytest.raises(ValueError):
        wade.wade(np.ones((3, 4)), [1, 1, 0, 0], **kwargs)


@pytest.mark.parametrize("kwargs", [{"scale": "log2"}, {"subsets": "either"}])
def test_removed_settings_are_rejected(kwargs):
    with pytest.raises(TypeError):
        wade.wade(np.ones((3, 4)), [1, 1, 0, 0], **kwargs)


def test_designs_without_a_stratum_holding_both_groups_are_rejected():
    x = np.ones((3, 4))
    with pytest.raises(ValueError, match="no stratum holds both"):
        wade.wade(x, [1, 1, 0, 0], strata=[1, 1, 0, 0])         # every stratum holds one group
    with pytest.raises(ValueError, match="two independent"):
        wade.wade(x, [1, 0, 0, 0])
    with pytest.raises(ValueError, match="one label per sample"):
        wade.wade(np.ones((3, 6)), [1, 1, 1, 0, 0, 0], strata=[1, 2, 3])


def strata_data(rng, sizes, genes=60, scale=60.0):
    """Counts in strata with a study effect per gene and stratum, labels and codes, cases first in each."""
    cond = np.concatenate([np.r_[np.ones(a, int), np.zeros(b, int)] for a, b in sizes])
    code = np.concatenate([np.full(a + b, s) for s, (a, b) in enumerate(sizes)])
    study = np.exp(rng.normal(0, 0.7, (genes, len(sizes))))[:, code]
    return rng.poisson(rng.gamma(4, scale / 4, (genes, 1)) * study * rng.gamma(10, .1, (genes, len(cond)))), cond, code


def test_one_stratum_is_the_unstratified_run():
    x, cond, _ = strata_data(np.random.default_rng(31), [(14, 12)])
    a = wade.wade(x, cond, max_permutations=499, seed=2)
    b = wade.wade(x, cond, max_permutations=499, seed=2, strata=np.zeros(len(cond)))
    for column in ("case_mean", "control_mean", "area_pvalue", "area_padj", "alignment_log2fc", "lorenz_gap",
                   "subset_fraction", "subset_log2fc", "subset_pvalue", "subset_padj", "subset_rarity"):
        np.testing.assert_array_equal(getattr(a, column), getattr(b, column))
    assert b.params["strata"]["with_both_groups"] == 1 and a.params["strata"] is None


def test_every_shuffle_exchanges_labels_only_within_strata():
    cond = np.r_[1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0]
    code = np.r_[0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    rows = stream_block(cond, 3, 7, code)
    for s in range(3):
        np.testing.assert_array_equal(rows[:, code == s].sum(axis=1), cond[code == s].sum())
    assert len({r.tobytes() for r in rows}) == 3 * 4 * 4        # every within-stratum labelling appears
    np.testing.assert_array_equal(stream_block(cond, 3, 7, np.zeros(11, int)), stream_block(cond, 3, 7))


def test_the_stratified_area_test_against_every_within_stratum_labelling():
    # Three strata, 109,760 labellings, every one enumerated: the computed p-value is within 0.67-1.5x of
    # the exact one wherever its error estimate keeps it, and the shuffles agree with it within their error.
    rng = np.random.default_rng(32)
    sizes = [(5, 3), (4, 4), (2, 6)]
    x, cond, code = strata_data(rng, sizes, genes=40)
    x = x + rng.random(x.shape)
    x[:, cond == 1] *= np.exp(rng.normal(0, 0.4, (40, 1)))
    centred = x.copy()
    for s in range(3):
        centred[:, code == s] -= centred[:, code == s].mean(axis=1, keepdims=True)
    totals = np.zeros((40, 1))
    for s, (a, b) in enumerate(sizes):
        idx = np.flatnonzero(code == s)
        sub = np.stack([centred[:, list(pick)].sum(axis=1) for pick in combinations(idx, a)], axis=1)
        totals = (totals[:, :, None] + sub[:, None, :]).reshape(40, -1)
    observed = (centred * cond).sum(axis=1)
    exact = np.mean(np.abs(totals) >= np.abs(observed)[:, None] - 1e-9, axis=1)
    p, error = mean_saddlepoint(x, cond, direction="both", code=code)
    kept = error <= 0.2
    assert kept.mean() > 0.5 and np.all((p[kept] / exact[kept] > 2 / 3) & (p[kept] / exact[kept] < 1.5))
    shuffled = mean_test(x, cond, direction="both", stop=400, maximum=99_999, seed=5, code=code)[0]
    moderate = exact >= 0.01
    assert np.all(np.abs(shuffled[moderate] / exact[moderate] - 1) < 0.2)


def test_strata_report_mantel_haenszel_means_and_leave_single_group_strata_out():
    rng = np.random.default_rng(33)
    x, cond, code = strata_data(rng, [(12, 6), (5, 9), (6, 0)])            # the last stratum holds cases only
    names = np.array([f"s{j}" for j in range(len(cond))])
    with pytest.warns(UserWarning, match="1 of 3 strata hold only cases or only controls; their 6 samples"):
        r = wade.wade(x, cond, max_permutations=499, seed=4, strata=code, sample_names=names)
    norm = x / r.normalization.size_factors
    w = np.array([12 * 6 / 18, 5 * 9 / 14])
    case = sum(wi * norm[:, (code == s) & (cond == 1)].mean(axis=1) for s, wi in enumerate(w)) / w.sum()
    control = sum(wi * norm[:, (code == s) & (cond == 0)].mean(axis=1) for s, wi in enumerate(w)) / w.sum()
    np.testing.assert_allclose(r.case_mean, case, rtol=1e-12)
    np.testing.assert_allclose(r.control_mean, control, rtol=1e-12)
    np.testing.assert_allclose(r.area, case - control, rtol=1e-12)
    assert r.params["strata"]["samples_left_out"] == 6 and r.params["strata"]["with_both_groups"] == 2
    keep = code < 2                                                        # the same tests on the informative samples
    s = wade.wade(x[:, keep], cond[keep], max_permutations=499, seed=4, strata=code[keep], sample_names=names[keep],
                  normalization=r.normalization)
    np.testing.assert_array_equal(r.area_pvalue, s.area_pvalue)
    np.testing.assert_array_equal(r.subset_pvalue, s.subset_pvalue)
    assert set(r.subset(0).samples) <= set(r.sample_names[keep & (cond == 1)])


def test_a_conditions_strata_are_read_by_sample_name():
    rng = np.random.default_rng(34)
    x, cond, code = strata_data(rng, [(10, 8), (7, 9)], genes=30)
    names = [f"s{j}" for j in range(len(cond))]
    order = rng.permutation(len(cond))                                     # the sheet in another order
    meta = dict(sample=[names[j] for j in order], group=["t" if cond[j] else "n" for j in order],
                study=[f"study{code[j]}" for j in order])
    by_name = wade.condition(meta, key="sample", column="group", case="t", control="n", strata="study")
    a = wade.wade(x, by_name, sample_names=names, max_permutations=299, seed=1)
    b = wade.wade(x, cond, sample_names=names, max_permutations=299, seed=1, strata=code)
    np.testing.assert_array_equal(a.area_pvalue, b.area_pvalue)
    np.testing.assert_array_equal(a.subset_pvalue, b.subset_pvalue)
    with pytest.raises(ValueError, match="given twice"):
        wade.wade(x, by_name, sample_names=names, strata=code)


def test_strata_hold_the_level_where_a_study_effect_confounds_the_unstratified_tests():
    # Two studies, case shares 80% and 20%, a gene-specific study effect, nothing else: the unstratified
    # tests call the study effect, the stratified ones stay within binomial error of the level.
    rng = np.random.default_rng(35)
    x, cond, code = strata_data(rng, [(40, 10), (10, 40)], genes=300, scale=200.0)
    plain = wade.wade(x, cond, max_permutations=999, seed=3, fdr=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        strat = wade.wade(x, cond, max_permutations=999, seed=3, fdr=None, strata=code)
    bound = 0.05 + 3 * np.sqrt(0.05 * 0.95 / 300)
    assert np.mean(plain.area_pvalue <= 0.05) > 0.3
    assert np.mean(strat.area_pvalue <= 0.05) <= bound and np.mean(strat.subset_pvalue <= 0.05) <= bound


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


def test_exhaustive_orbit_calibrates_the_mean_test():
    """Over every relabelling of a small design, the exact permutation p-value is superuniform."""
    x = np.array([[0, 0, 0, 2, 4, 4], [0, 0, 0, 0, 0, 100], [3, 3, 3, 3, 3, 3], [1, 9, 2, 8, 3, 7]], float)
    orbit = [np.isin(np.arange(6), idx).astype(int) for idx in combinations(range(6), 3)]
    difference = np.array([x[:, p == 1].mean(axis=1) - x[:, p == 0].mean(axis=1) for p in orbit])
    for row in difference.T:
        p = (np.abs(row)[None, :] >= np.abs(row)[:, None] - 1e-12).mean(axis=1)
        for alpha in np.unique(p):
            assert np.mean(p <= alpha) <= alpha + 1e-14


def subset_genes(rng, n1, n0, genes=24):
    """Gamma genes, a third with a raised subset of cases, a third with a lowered one."""
    x = rng.gamma(3, 40, (genes, n1 + n0))
    x[: genes // 3, : max(2, n1 // 5)] *= 12
    x[genes // 3: 2 * genes // 3, : max(2, n1 // 5)] /= 12
    return x, np.r_[np.ones(n1, int), np.zeros(n0, int)]


def test_the_subset_tests_pool_is_its_counts_thinned_exactly_as_the_count_model_asks():
    # Every sample is thinned to the smallest size factor (the area test's own draw); the fold is the W1
    # fold of those jittered counts; only the higher group is thinned further, so the lower group's
    # values are exactly its jittered equalized counts; and a thinned Poisson count is Poisson at the
    # lower rate, so with the cases at 4x the pool's cases come back to the controls' level.
    rng = np.random.default_rng(21)
    cond = np.r_[np.ones(15, int), np.zeros(15, int)]
    x = rng.poisson(np.r_[np.full(15, 800.0), np.full(15, 200.0)] * np.ones((40, 1)))
    factors, rows, ones = np.exp(rng.normal(0, .2, 30)), np.arange(40), np.ones(30)
    pool, fold = thinned(x, factors, cond, rows, 4)
    level = equalized(x, rows, factors, 4)
    plain = expression(jittered(level, rows, 4), ones)
    np.testing.assert_array_equal(fold, split(plain, cond)[4])
    assert np.all(fold > 1)
    np.testing.assert_array_equal(pool[:, cond == 0], plain[:, cond == 0])
    assert np.median(fold) == pytest.approx(4, rel=0.1)
    assert np.mean(pool[:, cond == 1]) == pytest.approx(np.mean(pool[:, cond == 0]), rel=0.05)
    np.testing.assert_array_equal(equalized(x, rows, ones, 4), x)          # equal depths: nothing thinned


def test_a_fold_every_case_carries_is_not_called_a_subset():
    # Every case 3x every control, no subset, 400 genes at the subset test's validated depth and above:
    # the rejection rate stays within binomial error of the level. Rescaling rather than thinning the
    # cases left their Poisson noise unscaled, which reads as concentration at lower counts.
    rng = np.random.default_rng(27)
    level = rng.gamma(20, 3, (400, 1))
    lam = level * rng.gamma(8, 1 / 8, (400, 60))
    x = rng.poisson(lam * np.where(np.r_[np.ones(30, int), np.zeros(30, int)] == 1, 3.0, 1.0))
    cond = np.r_[np.ones(30, int), np.zeros(30, int)]
    p = subset_test(x, np.ones(60), cond, np.arange(400), direction="both", stop=STOP, maximum=999, seed=5)[0]
    assert np.mean(p <= 0.05) <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 400)


def test_cases_more_even_than_the_controls_hold_no_subset():
    # Cases more even than controls (a mean-preserving contraction): the cases' Lorenz curve never
    # falls below the controls', so the cases hold no subset.
    rng = np.random.default_rng(23)
    case = 1000 + rng.normal(0, 20, (6, 40))
    control = np.c_[500 + rng.normal(0, 20, (6, 32)), 3000 + rng.normal(0, 60, (6, 8))]
    x, cond = np.c_[case, control], np.r_[np.ones(40, int), np.zeros(40, int)]
    size, _, _, gap, _ = split(x, cond)
    assert (size == 0).all() and (gap == 0).all()
    p = subset_test(np.rint(x).astype(int), np.ones(80), cond, np.arange(6), direction="both", stop=STOP,
                    maximum=999, seed=1)[0]
    assert (p == 1).all()


def test_the_subset_fold_is_measured_against_the_registered_controls():
    # Controls at b; nine tenths of the cases at b and a tenth at M. Against the mean shift's
    # prediction the subset's fold saturates at 1/(its share); against the registered controls
    # (the registration here is 1) it is the ordinary fold (M + 1) / (b + 1).
    b = 100.0
    for m in (1e3, 1e5, 1e7):
        x = np.r_[np.full(36, b), np.full(4, m), np.full(40, b)][None]
        cond = np.r_[np.ones(40, int), np.zeros(40, int)]
        size, side, fold, gap, shift = split(x, cond)
        assert (size[0], side[0], shift[0]) == (4, 1, 1.0)
        assert fold[0] == pytest.approx(np.log2((m + 1) / (b + 1)), rel=1e-12)


def test_a_gene_with_one_extreme_sample_is_calibrated_under_random_labels():
    # One sample fifty times the rest, under 400 random 40 v 40 labellings of the same values:
    # the extreme lands among the cases in half of them, and the registered pool keeps it
    # exchangeable, so the rejection rate stays within binomial error of the level. The released
    # construction, registering by the mean fold, called such genes at p <= 0.001.
    rng = np.random.default_rng(24)
    gene = np.r_[rng.gamma(3, 30, 79), 4500.0]
    rows = np.rint(np.stack([gene[rng.permutation(80)] for _ in range(400)])).astype(int)
    cond = np.r_[np.ones(40, int), np.zeros(40, int)]
    p = subset_test(rows, np.ones(80), cond, np.arange(400), direction="both", stop=STOP, maximum=999, seed=2)[0]
    assert np.mean(p <= 0.05) <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 400)


def test_a_bimodal_gene_is_calibrated_under_random_labels():
    # One gene with two separated modes (as in a common germline deletion), under 400 random
    # 50 v 50 labellings of the same 100 values: the rejection rate stays within binomial error
    # of the level.
    rng = np.random.default_rng(25)
    gene = np.exp(rng.normal(3, 0.3, 100) + 2 * (rng.random(100) < .5))
    rows = np.rint(np.stack([gene[rng.permutation(100)] for _ in range(400)])).astype(int)
    cond = np.r_[np.ones(50, int), np.zeros(50, int)]
    p = subset_test(rows, np.ones(100), cond, np.arange(400), direction="both", stop=STOP, maximum=999, seed=3)[0]
    assert np.mean(p <= 0.05) <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 400)


def test_a_depth_difference_between_the_groups_is_not_called():
    # Nothing differs between the groups but depth: the cases have twice the controls' reads. At low
    # counts the normalized values then differ in their Poisson noise and in their jitter, which a
    # test of the normalized values calls; thinned to one depth, the area test holds its level.
    rng = np.random.default_rng(30)
    n1 = n0 = 30
    depth = np.r_[np.full(n1, 2.0), np.full(n0, 1.0)]
    counts = rng.poisson(rng.gamma(2, 1, (400, 1)) * rng.gamma(10, 0.1, (400, n1 + n0)) * depth)
    names = [f"s{i}" for i in range(n1 + n0)]
    data = wade.as_counts(counts, sample_names=names)
    norm = wade.Normalization.from_size_factors(data, dict(zip(names, depth)), source="test")
    cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
    r = wade.wade(data, cond, normalization=norm, max_permutations=999, seed=5)
    t = r.tested
    bound = 0.05 + 3 * np.sqrt(0.05 * 0.95 / t.sum())                # the level, within binomial error
    assert np.mean(r.area_pvalue[t] <= 0.05) <= bound
    unthinned = mean_test(r.expression[t], cond, direction="both", stop=STOP, maximum=999, seed=5)[0]
    assert np.mean(unthinned <= 0.05) > bound                         # the same test without thinning fails


def test_genes_below_the_subset_tests_validated_range_are_counted_in_a_warning():
    rng = np.random.default_rng(26)
    counts = rng.poisson(np.r_[np.full(10, 20.0), np.full(10, 400.0)][:, None] * np.ones((1, 30)))
    cond = np.r_[np.ones(15, int), np.zeros(15, int)]
    with pytest.warns(UserWarning, match="10 of 20 tested genes average fewer than 32"):
        r = wade.wade(counts, cond, max_permutations=99)
    assert r.params["subset_validated_genes"] == 10
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wade.wade(counts[10:], cond, max_permutations=99)        # every gene well expressed: no warning
    assert set(wade.manifest(r)["results"]["n_subsets"]) == {"up", "down"}
