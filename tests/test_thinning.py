"""The count-native shift correction — ``docs/method.md`` §10.3–10.5, :mod:`wade.thinning`.

Unit properties of the three pieces (the fitted fold change, the thinning,
the one-count pseudocount), then the wiring through the public entry points:
which correction a result says it used, reproducibility from the seed, the
division fallback, and the bootstrap intervals.
"""

from __future__ import annotations

import numpy as np
import pytest

import wade
from wade import normalize as N
from wade.thinning import fit_fold_change, midmean, one_count, thin_counts

pytestmark = pytest.mark.validation

N1 = N0 = 200
COND = np.r_[np.ones(N1, int), np.zeros(N0, int)]
PHI = 0.1


def _nb(rng, mu, size):
    return rng.poisson(rng.gamma(1 / PHI, PHI * mu, size=size)).astype(float)


def _unit_normalizer(counts):
    """Unit library sizes, no jitter: one count is one unit, nothing couples
    genes, and the fit sees counts (which is what wade() hands it too)."""
    g, n = counts.shape
    return lambda c: N.tpm_like(c, np.ones(g), np.ones(n), jitter=np.zeros(counts.shape), norm_factor=1.0)


# ---------------------------------------------------------------------------
# midmean


def test_midmean_is_the_mean_of_the_middle_half():
    x = np.array([[10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 100]], dtype=float)
    # sorted: 0 1 2 3 | 4 5 6 7 8 9 | 10 100  ->  n=12, lo=3, hi=9
    assert midmean(x)[0] == pytest.approx(np.mean([3, 4, 5, 6, 7, 8]))


def test_midmean_is_defined_where_a_median_is_zero():
    x = np.zeros((1, 100)); x[0, 60:] = 1.0          # 60% zeros: median 0, middle half not all zero
    assert midmean(x)[0] > 0


# ---------------------------------------------------------------------------
# fit_fold_change


def test_fitted_fold_change_is_unbiased_for_an_nb_global_shift_at_low_and_moderate_counts():
    """The interquartile-mean *ratio* reads 2.08 for a true 2 on skewed counts;
    matching after thinning reads 2.00 (method.md 10.3)."""
    rng = np.random.default_rng(1)
    for mu in (2.0, 20.0):
        counts = np.c_[_nb(rng, 2 * mu, (100, N1)), _nb(rng, mu, (100, N0))]
        f = fit_fold_change(counts, COND, _unit_normalizer(counts), seed=0)
        assert abs(np.median(f) - 2.0) < 0.1, f"mu={mu}: median f {np.median(f):.3f}"
        # and the plain interquartile-mean ratio is visibly biased at low counts
        if mu == 2.0:
            x = _unit_normalizer(counts)(counts)
            ratio = midmean(x[:, :N1]) / midmean(x[:, N1:])
            assert np.median(ratio) > 2.04


def test_fitted_fold_change_is_blind_to_a_subset_below_a_quarter():
    """A 5% subset at 8x moves the mean ratio to ~1.35; the fit stays near 1."""
    rng = np.random.default_rng(2)
    case = _nb(rng, 20.0, (100, N1)); ctrl = _nb(rng, 20.0, (100, N0))
    for i in range(100):
        case[i, rng.choice(N1, 10, replace=False)] = _nb(rng, 160.0, 10)
    counts = np.c_[case, ctrl]
    f = fit_fold_change(counts, COND, _unit_normalizer(counts), seed=0)
    assert np.median(f) < 1.12
    assert np.median(case.mean(1) / ctrl.mean(1)) > 1.25


def test_fitted_fold_change_thins_whichever_group_is_higher():
    rng = np.random.default_rng(3)
    counts = np.c_[_nb(rng, 10.0, (50, N1)), _nb(rng, 20.0, (50, N0))]     # controls higher
    f = fit_fold_change(counts, COND, _unit_normalizer(counts), seed=0)
    assert np.median(f) == pytest.approx(0.5, abs=0.05)


def test_fitted_fold_change_is_one_where_there_is_nothing_to_fit():
    counts = np.zeros((3, N1 + N0))
    f = fit_fold_change(counts, COND, _unit_normalizer(counts), seed=0)
    np.testing.assert_array_equal(f, 1.0)


def test_fit_and_thin_refuse_non_integer_counts():
    x = np.random.default_rng(0).lognormal(3, 0.5, (4, N1 + N0))
    with pytest.raises(ValueError, match="integer counts"):
        fit_fold_change(x, COND, _unit_normalizer(x))
    with pytest.raises(ValueError, match="integer counts"):
        thin_counts(x, COND, np.ones(4), np.random.default_rng(0))


# ---------------------------------------------------------------------------
# thin_counts


def test_thinning_keeps_integers_and_zeros_and_touches_only_the_higher_group():
    rng = np.random.default_rng(4)
    counts = np.c_[_nb(rng, 20.0, (20, N1)), _nb(rng, 20.0, (20, N0))]
    counts[0, :] = 0.0
    fold = np.full(20, 2.0); fold[5:10] = 0.5; fold[10] = 1.0
    out = thin_counts(counts, COND, fold, np.random.default_rng(0))
    assert out.shape == counts.shape
    np.testing.assert_array_equal(out, np.round(out))
    assert (out >= 0).all() and (out <= counts).all()
    np.testing.assert_array_equal(out[0], 0.0)                              # zeros stay zeros
    np.testing.assert_array_equal(out[:5, N1:], counts[:5, N1:])            # f>=1: controls untouched
    np.testing.assert_array_equal(out[5:10, :N1], counts[5:10, :N1])        # f<1: cases untouched
    np.testing.assert_array_equal(out[10], counts[10])                      # f=1: nothing
    # thinned by 1/2 on average
    assert out[:5, :N1].mean() == pytest.approx(0.5 * counts[:5, :N1].mean(), rel=0.05)


def test_thinning_an_nb_by_its_fold_change_reproduces_the_control_distribution():
    """NB is closed under thinning with the same dispersion: mean AND variance
    of the thinned cases match the controls (a division would match the mean
    and overshoot the variance at low counts)."""
    rng = np.random.default_rng(5)
    case = _nb(rng, 4.0, (1, 20000)); ctrl = _nb(rng, 2.0, (1, 20000))
    thinned = thin_counts(np.c_[case, ctrl], np.r_[np.ones(20000, int), np.zeros(20000, int)],
                          np.array([2.0]), np.random.default_rng(0))[0, :20000]
    assert thinned.mean() == pytest.approx(ctrl.mean(), rel=0.03)
    assert thinned.var() == pytest.approx(ctrl.var(), rel=0.06)
    # Dividing matches the mean and misses the variance: the Poisson part does
    # not scale (NB(4)/2 has variance ~1.4 against the controls' ~2.4).
    assert abs((case / 2.0).var() - ctrl.var()) > 0.3 * ctrl.var()


# ---------------------------------------------------------------------------
# one_count


def test_one_count_is_the_normalized_value_of_one_count_per_cell():
    rng = np.random.default_rng(6)
    counts = rng.poisson(30, (5, 8)).astype(float)
    L = rng.uniform(0.5, 2.0, 5)
    lib = N.library_sizes(counts, L)
    pc = one_count(L, lib, counts.shape, norm_factor=1e6)
    x1 = N.tpm_like(counts + 1, L, lib, jitter=np.zeros(counts.shape), norm_factor=1e6)
    x0 = N.tpm_like(counts, L, lib, jitter=np.zeros(counts.shape), norm_factor=1e6)
    np.testing.assert_allclose(pc, x1 - x0, rtol=1e-12)
    # an all-zero sample has no count scale
    counts[:, 3] = 0.0
    lib = N.library_sizes(counts, L)
    pc = one_count(L, lib, counts.shape, norm_factor=1e6)
    np.testing.assert_array_equal(pc[:, 3], 0.0)
    assert np.all(pc[:, [0, 1, 2, 4, 5, 6, 7]] > 0)


# ---------------------------------------------------------------------------
# Through the entry points


@pytest.fixture(scope="module")
def counts():
    rng = np.random.default_rng(7)
    c = np.c_[_nb(rng, 20.0, (60, N1)), _nb(rng, 20.0, (60, N0))]
    c[:5, :N1] = _nb(rng, 40.0, (5, N1))                          # global 2x
    for i in range(5, 10):                                         # 5% subset at 8x
        c[i, rng.choice(N1, 10, replace=False)] = _nb(rng, 160.0, 10)
    return c


def test_wade_uses_thinning_by_default_and_division_on_request(counts):
    res = wade.wade(counts, np.ones(60), COND, nperms=100, seed=1)
    assert res.subset.correction == "thinning" and res.params["correction"] == "thinning"
    assert res.subset.r_test is not None and not np.array_equal(res.subset.r_test, res.subset.r)
    assert res.fitted_fold_change.shape == (60,)
    assert abs(np.median(res.fitted_fold_change[:5]) - 2.0) < 0.25
    res_div = wade.wade(counts, np.ones(60), COND, nperms=100, seed=1, thin=False)
    assert res_div.subset.correction == "division"
    # Stage 1 is untouched by either choice.
    np.testing.assert_array_equal(res.p_mean_shift, res_div.p_mean_shift)
    np.testing.assert_array_equal(res.tpm, res_div.tpm)


def test_wade_from_matrix_has_no_counts_to_thin_and_says_so(counts):
    x = counts + 0.5
    res = wade.wade_from_matrix(x, COND, nperms=100, seed=1)
    assert res.subset.correction == "division"
    assert res.pseudocount is None
    res2 = wade.wade_from_matrix(x, COND, nperms=100, seed=1, pseudocount=1.0)
    np.testing.assert_array_equal(res2.pseudocount, 1.0)


def test_the_pseudocount_is_one_count_in_each_samples_units_and_can_be_switched_off(counts):
    res = wade.wade(counts, np.ones(60), COND, nperms=50, seed=1)
    lib = N.library_sizes(counts, np.ones(60))
    np.testing.assert_allclose(res.pseudocount, np.broadcast_to(1e6 / lib[None, :], res.pseudocount.shape), rtol=1e-12)
    res0 = wade.wade(counts, np.ones(60), COND, nperms=50, seed=1, pseudocount=0.0)
    assert res0.pseudocount is None
    res2 = wade.wade(counts, np.ones(60), COND, nperms=50, seed=1, pseudocount=2.0)
    np.testing.assert_allclose(res2.pseudocount, np.broadcast_to(2e6 / lib[None, :], res2.pseudocount.shape), rtol=1e-12)
    with pytest.raises(ValueError, match="non-negative"):
        wade.wade(counts, np.ones(60), COND, nperms=50, pseudocount=-1.0)


def test_results_are_reproducible_from_the_seed_and_stage1_is_unchanged_by_the_new_streams(counts):
    a = wade.wade(counts, np.ones(60), COND, nperms=100, seed=3, n_boot=30)
    b = wade.wade(counts, np.ones(60), COND, nperms=100, seed=3, n_boot=30)
    np.testing.assert_array_equal(a.p_subset, b.p_subset)
    np.testing.assert_array_equal(a.fitted_fold_change, b.fitted_fold_change)
    np.testing.assert_array_equal(a.ci_affected_fraction, b.ci_affected_fraction)
    c = wade.wade(counts, np.ones(60), COND, nperms=100, seed=4)
    assert not np.array_equal(a.fitted_fold_change, c.fitted_fold_change)
    # The jitter and permutation streams are the first two children of the
    # seed, as before, so nothing in stage 1 moved when thinning arrived.
    np.testing.assert_array_equal(a.perms, b.perms)


def test_non_integer_counts_are_accepted_and_rounded_for_the_thinning(counts):
    est = counts + np.random.default_rng(0).uniform(-0.3, 0.3, counts.shape).clip(-counts, None)
    res = wade.wade(est, np.ones(60), COND, nperms=50, seed=1)
    assert res.subset.correction == "thinning"
    np.testing.assert_array_equal(res.tpm.shape, est.shape)


def test_bootstrap_intervals_are_reported_and_bracket_the_point_estimates(counts):
    res = wade.wade(counts, np.ones(60), COND, nperms=50, seed=1, n_boot=100)
    for ci, point in ((res.ci_affected_fraction, res.affected_fraction),
                      (res.ci_direction, res.direction),
                      (res.ci_log2_fc, res.log2_fc)):
        assert ci.shape == (2, 60)
        assert np.all(ci[0] <= ci[1])
        inside = np.mean((ci[0] <= point) & (point <= ci[1]))
        assert inside >= 0.8, f"own-estimate coverage {inside:.2f}"
    cols = res.columns()
    assert {"affected_fraction_lo", "affected_fraction_hi", "log2_fc_lo", "direction_hi"} <= set(cols)
    # A 5% subset's interval is narrow and near 0.05; a global gene's is near 1.
    assert np.median(res.ci_affected_fraction[1, 5:10]) < 0.2
    assert np.median(res.ci_affected_fraction[0, :5]) > 0.7
    assert wade.wade(counts, np.ones(60), COND, nperms=50, seed=1).ci_affected_fraction is None


def test_gene_detail_and_plot_gene_use_the_pseudocounted_curve(counts):
    res = wade.wade(counts, np.ones(60), COND, nperms=50, seed=1, n_boot=30)
    d = res.gene_detail(0)
    np.testing.assert_allclose(d.r, res.subset.r[0][::-1], rtol=1e-12)
    from wade.plotting import gene_panels
    [p] = gene_panels(res, gene=0)
    assert "[" in p.subtitle_lines[-1]          # the CI is shown
