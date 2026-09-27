"""Independent count-model and diagnostic checks; no scientific pass thresholds."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import beta, binom, gamma

from wade_research import BetaKernelArea
from wade_research.measurement import (
    BinomialBeta, beta_marginal_mode, binomial_support, uniform_dirichlet_summary,
)


@pytest.mark.parametrize('h', [1., .5])
def test_measurement_matches_existing_normalized_kernel(h):
    c, n, sf = np.array([0, 1, 100, 1000]), np.array([1000, 10000, 1000000, 100000000]), np.array([.002, .03, 2., 100.])
    model = BinomialBeta(c, n, sf, h)
    engine = BetaKernelArea(c, n, size_factors=sf, prior_shape=h)
    np.testing.assert_allclose(model.mean, engine.mean[engine.patient_state])
    q, _, residual = model.quantiles([.025, .5, .975])
    np.testing.assert_allclose(model.cdf(q[:, 1]), .5, atol=1e-7)
    assert residual <= 1e-7
    np.testing.assert_allclose(model.cdf(1), engine.evaluate(np.array([1.]))[0][engine.patient_state, 0])
    assert model.variance[0] > 0 and model.mean[0] > 0


@pytest.mark.parametrize('h', [1., .5])
def test_predictive_mass_matches_independent_likelihood_integration(h):
    model = BinomialBeta(np.array([0, 1, 7]), 10, 2., h)
    for i, c in enumerate(model.counts):
        for y in (0, 1, 5):
            expected = quad(lambda p: binom.pmf(y, 5, p)*beta.pdf(p, c+h, 10-c+h),
                            0, 1, epsabs=1e-11)[0]
            actual = np.exp(model.predictive_logpmf(y, 5)[i])
            assert actual == pytest.approx(expected, rel=1e-8)


def test_zero_uniform_prediction_has_closed_form_even_at_large_depth():
    model = BinomialBeta(0, 100000000, 100., 1.)
    for m in (1000, 100000000, 1000000000):
        expected = 100000001/(100000001+m)
        assert model.predictive_zero(m) == pytest.approx(expected, rel=1e-7)
        assert np.exp(model.predictive_logpmf(0, m)) == pytest.approx(expected, rel=1e-6)


def test_discrete_coverage_agrees_with_exhaustive_small_support():
    k, weights, omitted = binomial_support(20, .2)
    model = BinomialBeta(k, 20, 1., .5)
    q, _, _ = model.quantiles([.025, .975])
    by_interval = weights @ ((q[:, 0] <= 4) & (4 <= q[:, 1]))
    by_cdf = weights @ ((model.cdf(4) >= .025) & (model.cdf(4) <= .975))
    assert by_interval == pytest.approx(by_cdf, abs=1e-12)
    assert weights.sum()+omitted == pytest.approx(1)


def test_poisson_limit_does_not_remove_zero_location_shift():
    model = BinomialBeta(np.array([0, 1, 10]), 100000000, 1., .5)
    q, _, _ = model.quantiles([.025, .5, .975])
    expected = gamma.ppf([.025, .5, .975], model.counts[:, None]+.5)
    np.testing.assert_allclose(q, expected, rtol=1e-6)
    assert model.mean[0] == pytest.approx(.5, rel=1e-7)


def test_gene_fraction_means_expose_incompatible_marginal_prior_budget():
    counts = np.array([0, 1, 5, 10, 84])
    for h in (1., .5):
        model = BinomialBeta(counts, counts.sum(), counts.sum(), h)
        assert model.mean.sum() > 1
        assert model.mean.sum() == pytest.approx((100+5*h)/(100+2*h))
        # Joint Dirichlet bookkeeping is an identity check, not a selected prior.
        alpha = np.array([.01, .1, .2, .5, 1.])
        assert ((counts+alpha)/(100+alpha.sum())).sum() == pytest.approx(1)


def test_zero_data_are_immutable_while_mode_and_mean_are_distinct():
    counts = np.array([0, 2, 8], dtype=np.int64)
    original = counts.copy()
    counts.flags.writeable = False
    model = BinomialBeta(counts, 10, 1., 1.)
    model.quantiles([.025, .975])
    assert model.mean[0] > 0 and model.variance[0] > 0
    np.testing.assert_allclose(10*beta_marginal_mode(model.a, model.b), counts)
    np.testing.assert_array_equal(counts, original)
    assert counts.dtype == np.int64 and counts.sum() == 10


def test_uniform_dirichlet_joint_mode_is_not_its_marginal_modes():
    counts = np.array([0, 2, 8])
    model = uniform_dirichlet_summary(counts)
    np.testing.assert_allclose(10*model['joint_mode'], counts)
    np.testing.assert_allclose(10*model['marginal_mode'], [0, 20/11, 80/11])
    assert model['mean'].sum() == pytest.approx(1)
    assert model['variance'][0] > 0 and model['marginal_mode'][0] == 0
    # Check marginal density peaks independently of the analytic mode helper.
    for a, b, m in zip(model['a'], model['b'], model['marginal_mode']):
        neighbours = np.clip([m-.001, m+.001], 0, 1)
        assert np.all(beta.logpdf(m, a, b) >= beta.logpdf(neighbours, a, b))


def test_two_categories_are_the_dirichlet_exception():
    model = uniform_dirichlet_summary([2, 8])
    np.testing.assert_allclose(model['marginal_mode'], [.2, .8])
    np.testing.assert_allclose(model['joint_mode'], model['marginal_mode'])


def test_preserving_sum_of_modes_does_not_preserve_each_gene():
    counts = np.array([0, 2, 8])
    mode = 10*beta_marginal_mode(counts+.5, 10-counts+.5)
    assert mode.sum() == pytest.approx(10)
    assert not np.allclose(mode, counts)


@pytest.mark.parametrize('counts, alpha', [
    ([1, 2, 7], [1, 1, 1]),
    ([1, 2, 7], [.01, .2, 20]),
    ([0, 2, 8], [.01, 1, 3]),
    ([0, 1, 2, 7], [.5, .5, .5, .5]),
])
def test_general_dirichlet_marginal_mode_equations_have_nonzero_residual(counts, alpha):
    c, a = np.asarray(counts), np.asarray(alpha)
    positive = c > 0
    # These equations would all be zero if the interior marginal modes were C/N.
    residual = a[positive]-1-c[positive]/c.sum()*(a.sum()-2)
    expected_sum = 2-positive.sum()-a[~positive].sum()
    assert residual.sum() == pytest.approx(expected_sum)
    assert expected_sum < 0  # no proper positive alpha can solve these cases


def test_log_density_mode_is_not_a_transformed_physical_mode():
    n, scale = 1000, 1e6
    # Physical Beta(1,N+1) kernel peaks at zero. The log1p-coordinate density
    # includes a Jacobian; changing coordinates never changes the observations.
    assert beta_marginal_mode(1, n+1) == 0
    transformed_peak = (scale-n)/(n+1)
    def log_coordinate_density(x):
        return beta.logpdf(x/scale, 1, n+1)-np.log(scale)+np.log1p(x)
    assert log_coordinate_density(transformed_peak) > log_coordinate_density(0)


def test_nonunique_beta_modes_are_not_silently_assigned_a_peak():
    modes = beta_marginal_mode([1, .5, .5, 1], [1, .5, 1, .5])
    assert np.isnan(modes[:2]).all()
    np.testing.assert_array_equal(modes[2:], [0, 1])
