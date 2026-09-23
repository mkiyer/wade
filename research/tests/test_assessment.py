"""Analytic biological examples and independent checks of benchmark instruments."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import beta, wasserstein_distance

from wade_research import BetaKernelArea, label_coefficients, point_scores
from wade_research.assessment import (
    points, beta_distribution, describe, membership_metrics, average_precision, group_w1,
)


@pytest.mark.parametrize('pattern,fraction,area,mean_ratio,tail_ratio', [
    ('null', 0, 0, 1, None),
    ('shared_tail', 0, 0, 1, None),
    ('bulk', 1, 5, 1.5, 1.5),
    ('subset', .1, 3, 1.3, 4),
    ('lower_bulk', .1, 3, .85, 4),
])
def test_known_patterns(pattern, fraction, area, mean_ratio, tail_ratio):
    control = np.full(100, 10.)
    case = control.copy()
    if pattern == 'bulk':
        case *= 1.5
    if pattern == 'lower_bulk':
        case *= .5
    if pattern in ('shared_tail', 'subset', 'lower_bulk'):
        case[-10:] = 40
    if pattern == 'shared_tail':
        control[-10:] = 40
    labels = np.r_[np.ones(100), np.zeros(100)]
    x = np.r_[case, control]
    result, selected = describe(points(x), labels)
    assert result['tail_fraction'] == pytest.approx(fraction)
    assert result['area'] == pytest.approx(area)
    assert result['mean_log2'] == pytest.approx(np.log2(mean_ratio))
    assert selected.mean() == pytest.approx(fraction)
    if tail_ratio is None:
        assert np.isnan(result['tail_vs_control_log2'])
    else:
        assert result['tail_vs_control_log2'] == pytest.approx(np.log2(tail_ratio))
    assert result['area'] == pytest.approx(point_scores(x, labels[None])[0])


def test_area_is_not_fold_change_and_is_scale_equivariant():
    x, labels = np.array([1, 2, 12, 1, 2, 3.]), np.array([1, 1, 1, 0, 0, 0])
    a, am = describe(points(x), labels)
    b, bm = describe(points(100*x), labels)
    assert b['area'] == pytest.approx(100*a['area'])
    assert a['mean_log2'] == pytest.approx(b['mean_log2'])
    np.testing.assert_allclose(am, bm)


def test_unequal_groups_and_boundary_ties_keep_equal_patient_mass():
    x = np.array([1, 1, 3, 1, 1.])
    labels = np.array([1, 1, 1, 0, 0])
    r, selected = describe(points(x), labels)
    assert r['tail_fraction'] == pytest.approx(1/3)
    np.testing.assert_allclose(selected, [0, 0, 1])
    assert r['area'] == pytest.approx(2/3)
    assert group_w1(points(x), x, labels) == 0


def test_ties_do_not_invent_patient_ranking():
    y = np.array([1, 0, 1, 0], bool)
    assert average_precision(y, np.ones(4)) == .5
    assert average_precision(y[::-1], np.ones(4)) == .5
    assert average_precision(y, y) == 1
    assert np.isnan(average_precision(np.zeros(4, bool), np.arange(4)))
    result = membership_metrics(y, np.full(4, .5), np.ones(4))
    assert result['soft_precision'] == .5
    assert result['soft_recall'] == .5
    assert result['membership_brier'] == .25


def test_quantile_approximation_cost_against_independent_cdf_integral():
    engine = BetaKernelArea([0], [50], size_factors=[2.], prior_shape=.5)
    d = beta_distribution(engine, 16)
    nodes = d.nodes[0]
    scale = 25
    def distance(x):
        return abs(beta.cdf(x/scale, .5, 50.5)-np.mean(nodes <= x))
    integral = quad(distance, 0, scale, points=nodes, epsabs=1e-10, limit=200)[0]
    assert d.w1_error[0]-d.roundoff_bound[0] == pytest.approx(integral, rel=1e-7)
    assert d.w1_error[0] >= integral
    assert d.intervals[0, 1] > d.intervals[0, 0] > 0


@pytest.mark.parametrize('shape', [.5, 1.])
def test_kernel_area_bounds_and_positive_zero_count_uncertainty(shape):
    labels = np.array([1, 1, 1, 0, 0, 0])
    e = BetaKernelArea([0, 1, 100, 0, 1, 3], [1000]*6,
                       size_factors=[.1, 1, 2, 2, 1, .1], prior_shape=shape)
    d = beta_distribution(e, 64)
    r, selected = describe(d, labels)
    exact = e.scores(label_coefficients(labels), atol=1e-8)
    assert r['area'] + r['area_numerical_bound'] >= exact.lower[0]
    assert r['area'] - r['area_numerical_bound'] <= exact.upper[0]
    np.testing.assert_allclose(d.means, e.mean[e.patient_state])
    assert selected.mean() == pytest.approx(r['tail_fraction'])
    assert np.all(d.intervals[:, 1] > d.intervals[:, 0])
    q = beta_distribution(e, 128)
    assert np.all(q.w1_error <= d.w1_error)


def test_finite_latent_null_area_need_not_be_zero():
    x = [1, 2, 4, 1, 2, 3]
    r, _ = describe(points(x), [1, 1, 1, 0, 0, 0])
    assert r['area'] > 0  # finite cohorts differ even under one common population
    assert r['area'] == pytest.approx((wasserstein_distance(x[:3], x[3:])
                                       +np.mean(x[:3])-np.mean(x[3:]))/2)


def test_walk_is_not_a_mixture_prevalence_estimator():
    # A minority spike shifts quantiles throughout a heterogeneous distribution.
    # Unaffected cases exactly match controls. Even with NO counting noise the
    # cumulative walk includes almost all cases, not just the 10% altered cases.
    baseline = np.arange(1, 101)/100
    case = np.r_[np.repeat(baseline, 9), np.full(100, 10.)]
    control = np.repeat(baseline, 10)
    r, _ = describe(points(np.r_[case, control]), np.r_[np.ones(1000), np.zeros(1000)])
    assert r['tail_fraction'] == pytest.approx(.991)
    assert r['down_area'] == 0


def test_large_asymmetric_beta_quantiles_obey_the_defining_cdf_identity():
    # Regression from a saved high-depth transcriptome. A finite library ppf
    # value alone is not an accuracy certificate at these asymmetric shapes.
    from scipy.special import betainc
    from wade_research.assessment import _beta_quantiles
    q = np.arange(129)/128
    values, _, residual = _beta_quantiles(1000., 70893603., q)
    assert residual < 1e-7
    np.testing.assert_allclose(betainc(1000., 70893603., values), q, atol=1e-7, rtol=0)
    assert np.all(np.diff(values) > 0)
