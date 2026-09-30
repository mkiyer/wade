"""Independent mathematical and computational checks of the production contract."""
from fractions import Fraction
from itertools import combinations
import math
import warnings
import numpy as np
import pytest
from scipy.stats import false_discovery_control

import wade
from wade._means import area_values, equalized, jittered, mean_saddlepoint, mean_test
from wade._subsets import split, subset_test
from wade.pvalues import sequential_p

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
    labels = wade.draw_perms(cond, 79, rng=np.random.default_rng(88))
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
    run = lambda **k: wade.wade(x, cond, max_permutations=199, seed=3, **k)
    auto, brute = run(), run(pvalues="permutation")
    z = area_values(x, range(30), auto.normalization.size_factors, auto.params["seed"])
    computed, error = mean_saddlepoint(z, cond, direction="both")
    shuffled = error > auto.params["relative_error"]
    assert shuffled.any() and (~shuffled).any()
    np.testing.assert_array_equal(auto.area_permutations > 0, shuffled)          # only there, from the same shuffles
    np.testing.assert_array_equal(auto.area_pvalue[~shuffled], computed[~shuffled])
    np.testing.assert_array_equal(auto.area_pvalue[shuffled], brute.area_pvalue[shuffled])
    assert np.all(brute.area_permutations > 0)
    with pytest.raises(ValueError, match="strata"):                               # independent samples only
        run(strata=np.arange(24) % 2)


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
def test_chunk_batch_and_filter_invariance(direction):
    rng = np.random.default_rng(85)
    x = rng.poisson(30, (30, 20))
    x[0] = 0
    x[1, :10] *= 8
    cond = np.tile([1, 0], 10)
    for pvalues in ("permutation", "auto"):                  # shuffled throughout, and computed where it can be
        kwargs = dict(direction=direction, max_permutations=97, seed=17, pvalues=pvalues)
        a = wade.wade(x, cond, gene_chunk=30, **kwargs)
        b = wade.wade(x, cond, gene_chunk=3, **kwargs)
        for key in wade.RESULT_COLUMNS[2:] + ("area_permutations", "area_exceedances", "subset_permutations"):
            np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    kwargs = dict(direction=direction, max_permutations=97, seed=17, pvalues="permutation")
    a = wade.wade(x, cond, gene_chunk=30, **kwargs)
    mask = np.arange(30) % 2 == 0
    c = wade.wade(x, cond, test_genes=mask, **kwargs)
    np.testing.assert_array_equal(a.normalization.size_factors, c.normalization.size_factors)
    np.testing.assert_array_equal(a.area_pvalue[mask], c.area_pvalue[mask])
    z = area_values(x[1:3], range(1, 3), a.normalization.size_factors, a.params["seed"])   # per gene, not per chunk
    for batch in (1, 13, 97):
        p, _, tried = mean_test(z, cond, direction=direction, stop=a.params["stop_exceedances"],
                                maximum=97, seed=17, batch=batch)
        np.testing.assert_array_equal(p, a.area_pvalue[1:3])
        np.testing.assert_array_equal(tried, a.area_permutations[1:3])


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
    assert r.params["depth_equalized"] == pytest.approx(sf.min() / np.median(sf))
    # The area test's counts: every sample thinned to the shallowest depth, which keeps all its reads.
    thinned = equalized(x, range(len(x)), sf, r.params["seed"])
    np.testing.assert_array_equal(thinned[:, np.argmin(sf)], x[:, np.argmin(sf)])
    assert np.all(thinned <= x)
    np.testing.assert_array_equal(equalized(x[2:4], range(2, 4), sf, r.params["seed"]), thinned[2:4])
    keep, reads = sf.min() / sf, x.sum(axis=0)                     # each sample keeps min(s) / s of its reads
    assert np.all(np.abs(thinned.sum(axis=0) / reads - keep) <= 4 * np.sqrt(keep * (1 - keep) / reads))


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
                                max_permutations=199, **k)
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
    x = rng.poisson(40, (30, 40)).astype(float)
    x[:10, :4] *= 8                                          # raised subsets
    x[10:20, :5] //= 8                                       # lowered subsets
    cond = np.r_[np.ones(20, int), np.zeros(20, int)]
    run = lambda d: subset_test(x, cond, direction=d, stop=STOP, maximum=19, seed=6)    # 19 shuffles for every gene
    (_, both, tried), (_, up, _), (_, down, _) = run("both"), run("up"), run("down")
    _, side, *_ = split(x, cond)
    ran = tried > 0
    assert ran.any() and np.all(tried[ran] == 19)
    # A deeper subset on the observed side counts for that side; every other shuffle counts
    # for the opposite one, and the two-sided test counts depth on either side.
    same, other = np.where(side > 0, up, down), np.where(side > 0, down, up)
    assert np.all(same[ran] <= both[ran]) and np.all(same[ran] + other[ran] >= 19)


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


def test_strata_and_single_sample_designs_are_rejected():
    x = np.ones((3, 4))
    with pytest.raises(ValueError, match="strata"):
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


def test_a_fold_on_the_cases_leaves_every_subset_pvalue_unchanged():
    # The gap ignores a fold on the cases and the registration carries it, so the registered
    # pool, the observed gap and every shuffled gap are the same: the whole p-value is unchanged.
    x, cond = subset_genes(np.random.default_rng(21), 18, 30)
    moved = x.copy()
    moved[:, cond == 1] = (x[:, cond == 1] + 1) * 7.3 - 1
    for direction in ("both", "up", "down"):
        run = lambda v: subset_test(v, cond, direction=direction, stop=STOP, maximum=199, seed=4)
        (p, hits, tried), (q, hits2, tried2) = run(x), run(moved)
        np.testing.assert_array_equal(p, q)
        np.testing.assert_array_equal(hits, hits2)
        np.testing.assert_array_equal(tried, tried2)


def test_cases_more_even_than_the_controls_hold_no_subset():
    # Cases more even than controls (a mean-preserving contraction): the cases' Lorenz curve never
    # falls below the controls', so the cases hold no subset.
    rng = np.random.default_rng(23)
    case = 1000 + rng.normal(0, 20, (6, 40))
    control = np.c_[500 + rng.normal(0, 20, (6, 32)), 3000 + rng.normal(0, 60, (6, 8))]
    x, cond = np.c_[case, control], np.r_[np.ones(40, int), np.zeros(40, int)]
    size, _, _, gap, _ = split(x, cond)
    assert (size == 0).all() and (gap == 0).all()
    p = subset_test(x, cond, direction="both", stop=STOP, maximum=999, seed=1)[0]
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
    rows = np.stack([gene[rng.permutation(80)] for _ in range(400)])
    cond = np.r_[np.ones(40, int), np.zeros(40, int)]
    p = subset_test(rows, cond, direction="both", stop=STOP, maximum=999, seed=2)[0]
    assert np.mean(p <= 0.05) <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 400)


def test_a_bimodal_gene_is_calibrated_under_random_labels():
    # One gene with two separated modes (as in a common germline deletion), under 400 random
    # 50 v 50 labellings of the same 100 values: the rejection rate stays within binomial error
    # of the level.
    rng = np.random.default_rng(25)
    gene = np.exp(rng.normal(3, 0.3, 100) + 2 * (rng.random(100) < .5))
    rows = np.stack([gene[rng.permutation(100)] for _ in range(400)])
    cond = np.r_[np.ones(50, int), np.zeros(50, int)]
    p = subset_test(rows, cond, direction="both", stop=STOP, maximum=999, seed=3)[0]
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
    with pytest.warns(UserWarning, match="10 of 20 tested genes average fewer than 128"):
        r = wade.wade(counts, cond, max_permutations=99)
    assert r.params["subset_validated_genes"] == 10
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wade.wade(counts[10:], cond, max_permutations=99)        # every gene well expressed: no warning
    assert set(wade.manifest(r)["results"]["n_subsets"]) == {"up", "down"}
