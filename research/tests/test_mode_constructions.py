"""Independent structural checks and counterexamples; not performance claims."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import exp1, ndtr
from scipy.stats import beta, binom

from wade_research.mode_constructions import (
    contracted_beta_cdf, contracted_dirichlet_parameters, contraction_moments,
    draw_contracted_dirichlet, multinomial_parameters, smoothed_binomial,
)


@pytest.mark.parametrize('values', [[0, 2, 8], [1, 2, 7], [2, 8], [0, 0, 0, 1]])
def test_actual_discrete_modes_total_and_unchanged_input(values):
    counts = np.array(values, dtype=np.int64)
    before = counts.copy()
    counts.flags.writeable = False
    weights = np.arange(1, len(counts)+1, dtype=float)
    weights /= weights.sum()
    model = multinomial_parameters(counts, weights)
    support = np.arange(counts.sum()+1)
    pmf = binom.pmf(support[:, None], counts.sum(), model['q'])
    np.testing.assert_array_equal(np.argmax(pmf, axis=0), counts)
    np.testing.assert_array_equal((pmf == pmf.max(axis=0)).sum(axis=0), 1)
    np.testing.assert_allclose(model['mean'].sum(), counts.sum(), rtol=1e-15)
    assert np.all(model['variance'] > 0)
    draws = np.random.default_rng(23).multinomial(counts.sum(), model['q'], size=1000)
    np.testing.assert_array_equal(draws.sum(axis=1), counts.sum())
    np.testing.assert_array_equal(counts, before)
    assert counts.dtype == before.dtype


def test_zero_budget_is_a_family_bound_and_uniform_zeros_practically_collapse():
    counts = np.array([0, 0, 1, 9])
    weights = np.array([.499, .499, .001, .001])
    model = multinomial_parameters(counts, weights)
    assert model['mean'][counts == 0].sum() < counts.sum()/(counts.sum()+1) < 1
    zero = smoothed_binomial(0, 1000, 1/16384)
    assert zero['variance'] > 0
    np.testing.assert_array_equal(binom.ppf([.025, .975], 1000, zero['q']), [0, 0])
    assert 0 < binom.sf(0, 1000, zero['q']) < .000062


def test_discrete_target_is_not_the_integrated_posterior_predictive():
    # Dir(C+w) has A=N+1; integrating q adds extra binomial variability.
    counts = np.array([0, 2, 8])
    model = multinomial_parameters(counts, np.full(3, 1/3))
    a = counts+1/3
    total = a.sum()
    predictive_variance = 10*(a/total)*(1-a/total)*(total+10)/(total+1)
    np.testing.assert_allclose(predictive_variance/model['variance'], 21/12)


@pytest.mark.parametrize('values', [[0, 2, 8], [1, 2, 7], [0, 0, 0, 1]])
def test_continuous_joint_draws_conserve_sum_and_match_moments(values):
    counts = np.array(values, dtype=np.int64)
    original = counts.copy()
    counts.flags.writeable = False
    prior = np.full(len(counts), 1/len(counts))
    model = contracted_dirichlet_parameters(counts, prior)
    draws = draw_contracted_dirichlet(counts, prior, 50000, np.random.default_rng(351))
    np.testing.assert_allclose(draws.sum(axis=1), 1, atol=1e-15, rtol=0)
    np.testing.assert_allclose(model['mean'].sum(), 1, atol=1e-15, rtol=0)
    assert np.all(draws > 0) and np.all(draws < 1)
    se = np.sqrt(model['variance']/len(draws))
    assert np.all(abs(draws.mean(axis=0)-model['mean']) < 5*se)
    np.testing.assert_allclose(draws.var(axis=0), model['variance'], rtol=.07)
    np.testing.assert_array_equal(counts, original)


def test_contraction_cdf_against_closed_form_independent_uniform_product():
    x = np.array([1e-7, .001, .01, .2, .9, .999])
    actual, error = contracted_beta_cdf(x, 0, 1, 1)
    np.testing.assert_allclose(actual, x*(1-np.log(x)), atol=2e-9, rtol=0)
    assert error < 1e-9
    moments = contraction_moments(0, 1, 1)
    assert moments['mean'] == .25
    assert moments['variance'] == pytest.approx(7/144)


def test_continuous_actual_density_peaks_at_observed_fraction():
    counts = np.array([0, 2, 8])
    params = contracted_dirichlet_parameters(counts, np.full(3, 1/3))
    # Independent integration over the base abundance Q. The implementation
    # instead integrates a bounded CDF over U. Never override a reported mode.
    for c, a, b in zip(counts, params['a'], params['b']):
        m = c/counts.sum()
        for side, extent in [(-1, m), (1, 1-m)]:
            if not extent:
                continue
            densities = []
            for distance in [.5, .1, .02, .004]:
                x = m+side*extent*distance
                bounds = (0, x) if side < 0 else (x, 1)
                value, error = quad(lambda q: beta.pdf(q, a, b)/abs(q-m),
                                    *bounds, epsabs=1e-8)
                assert error < 1e-5
                densities.append(value)
            assert np.all(np.diff(densities) > 0)


def test_cdf_boundary_and_tolerance_checks():
    m, a, b = np.array([0., .2, .8]), np.array([1/3, 7/3, 25/3]), np.array([32/3, 26/3, 8/3])
    at_mode, _ = contracted_beta_cdf(m, m, a, b)
    np.testing.assert_allclose(at_mode, beta.cdf(m, a, b), atol=1e-15)
    outside, _ = contracted_beta_cdf([-np.inf, np.inf], .2, 2, 8)
    np.testing.assert_array_equal(outside, [0, 1])
    x = np.array([.03, .24, .7])
    loose, _ = contracted_beta_cdf(x, m, a, b, tolerance=1e-8)
    tight, _ = contracted_beta_cdf(x, m, a, b, tolerance=1e-11)
    np.testing.assert_allclose(loose, tight, atol=2e-8, rtol=0)


def test_extreme_depth_cdf_tolerance_is_audited():
    n, g = 100000000, 16384
    counts = np.arange(20)
    a, b = counts+1/g, n-counts+1-1/g
    loose, _ = contracted_beta_cdf(.1/n, counts/n, a, b, tolerance=1e-7)
    tight, error = contracted_beta_cdf(.1/n, counts/n, a, b, tolerance=5e-9)
    np.testing.assert_allclose(loose, tight, atol=2e-7, rtol=0)
    assert error < 5e-9


def test_repeated_sampling_moments_against_full_binomial_enumeration():
    from derive_mode_constructions import moment_score

    n, g, truth = 10, 3, 2.
    counts = np.arange(n+1)
    weights = binom.pmf(counts, n, truth/n)
    models = [(None, False, smoothed_binomial(counts, n, 1/g))]
    for budget in (1, g):
        a, b = counts+budget/g, n-counts+budget-budget/g
        model = contraction_moments(counts/n, a, b)
        for contract in (False, True):
            prefix = '' if contract else 'base_'
            models.append((budget, contract, dict(mean=n*model[prefix+'mean'],
                                                  variance=n*n*model[prefix+'variance'])))
    for budget, contract, model in models:
        score = moment_score(n, g, truth, budget=budget, contract=contract)
        mean = weights @ model['mean']
        variance_mean = weights @ (model['mean']-mean)**2
        assert score['mean_bias_count'] == pytest.approx(mean-truth, abs=1e-14)
        assert score['mean_rmse_count']**2 == pytest.approx(weights @ (model['mean']-truth)**2)
        assert score['average_kernel_variance_count'] == pytest.approx(weights @ model['variance'])
        assert score['mixture_variance_count'] == pytest.approx(variance_mean+weights @ model['variance'])


def test_high_information_contraction_is_not_calibrated_by_mode_preservation():
    def product_cdf(x):
        return ndtr(x)+x*exp1(x*x/2)/(2*np.sqrt(2*np.pi))
    half_width = brentq(lambda x: product_cdf(x)-.975, .01, 5)
    coverage = 2*ndtr(half_width)-1
    assert half_width == pytest.approx(1.25824709736)
    assert coverage == pytest.approx(.79169759307)


@pytest.mark.parametrize('counts,weights', [([0, 0], [.5, .5]), ([.2, 1.8], [.5, .5]),
    ([0, 2], [0, 1]), ([0, 2], [.3, .3]), ([0, 2], [.5])])
def test_invalid_whole_library_inputs_fail(counts, weights):
    with pytest.raises(ValueError):
        multinomial_parameters(counts, weights)


def test_continuous_does_not_acquire_an_atom_from_its_infinite_peak():
    # Product of two uniforms has infinite density at zero, but mass in [0,e]
    # tends to zero. An infinite PDF value is not a point mass.
    eps = np.array([1e-2, 1e-4, 1e-6])
    mass, _ = contracted_beta_cdf(eps, 0, 1, 1)
    assert np.all(np.diff(mass) < 0) and mass[-1] < 2e-5
