"""Analytic invariants and limitations of the descriptive localization candidate."""
import numpy as np
import pytest
from scipy.stats import ks_2samp

from wade_research.assessment import PatientDistribution, points, describe
from wade_research.localization import localize


def evaluate(case, control):
    return localize(points(np.r_[case, control]), np.r_[np.ones(len(case)), np.zeros(len(control))])


def test_heterogeneous_minority_spike_is_localized_without_planted_fraction_input():
    baseline = np.arange(1, 101)/100
    case, control = np.r_[np.repeat(baseline, 9), np.full(100, 10.)], np.repeat(baseline, 10)
    r = evaluate(case, control)
    assert r.summary['cutoff'] == 1
    assert r.summary['excess_tail_mass'] == pytest.approx(.1)
    np.testing.assert_array_equal(r.membership, np.r_[np.zeros(900), np.ones(100)])
    assert r.summary['tail_area_share'] > .95
    assert .1 < r.summary['effective_contributor_fraction'] < .12
    # Preserve the walk: its cumulative area has a different descriptive purpose.
    walk, _ = describe(points(np.r_[case, control]), np.r_[np.ones(1000), np.zeros(1000)])
    assert walk['tail_fraction'] == pytest.approx(.991)


def test_shared_biological_tail_has_no_differential_localization():
    values = np.r_[np.ones(90), np.full(10, 10)]
    r = evaluate(values, values)
    assert r.summary['area'] == r.summary['excess_tail_mass'] == 0
    assert np.isnan(r.summary['cutoff'])
    assert not r.membership.any()


def test_additive_bulk_area_is_distributed_even_when_control_tail_does_not_include_all_cases():
    control = np.arange(101)
    r = evaluate(control+50, control)
    np.testing.assert_allclose(r.contributions, 50)
    assert r.summary['effective_contributor_fraction'] == pytest.approx(1)
    assert .45 < r.summary['case_tail_mass'] < .55
    # Mass above a control-relative cutoff is not biological affected prevalence.


def test_near_ties_can_move_cdf_cutoff_despite_stable_area_and_concentration():
    tied = evaluate([1, 2, 3, 4], [0, 1, 2, 3])
    separated = evaluate([1, 2+1e-10, 3, 4], [0, 1, 2, 3])
    assert tied.summary['case_tail_mass'] == .25
    assert separated.summary['case_tail_mass'] == .75
    assert separated.summary['area'] == pytest.approx(tied.summary['area'])
    assert separated.summary['effective_contributor_fraction'] == pytest.approx(1)
    # Preserve this limitation; do not tune a tolerance to recover planted truth.


def test_proportional_bulk_does_not_promise_equal_absolute_contributions():
    control = np.linspace(1, 100, 100)
    r = evaluate(2*control, control)
    np.testing.assert_allclose(r.contributions, control)
    assert r.summary['effective_contributor_fraction'] < .8  # all cases ARE altered


def test_lower_bulk_and_upward_spike():
    r = evaluate(np.r_[np.full(90, .5), np.full(10, 4.)], np.ones(100))
    np.testing.assert_allclose(r.contributions, np.r_[np.zeros(90), np.full(10, 3.)])
    assert r.summary['effective_contributor_fraction'] == pytest.approx(.1)
    assert r.summary['case_tail_mass'] == pytest.approx(.1)


def test_ties_share_transport_without_sample_order_dependence():
    r = evaluate([3, 3], [1, 2])
    np.testing.assert_allclose(r.contributions, [1.5, 1.5])
    assert r.summary['area'] == pytest.approx(1.5)
    other = evaluate([3, 3], [2, 1])
    np.testing.assert_array_equal(r.contributions, other.contributions)


def test_identical_patient_kernels_have_identical_contributions():
    nodes = np.array([[0, 4], [0, 4], [0, 1], [0, 1.]])
    d = PatientDistribution(nodes, nodes.mean(axis=1), np.zeros(4))
    r = localize(d, [1, 1, 0, 0])
    np.testing.assert_allclose(r.contributions, [1.5, 1.5])
    np.testing.assert_allclose(r.membership, [.5, .5])
    assert r.summary['excess_tail_mass'] == .5


def test_excess_matches_independent_scipy_ks_statistic_with_unequal_groups():
    # Only the statistic is used; continuous-data KS p-values are not used for counts.
    case, control = [0, 1, 4, 8, 8], [0, 1, 1]
    r = evaluate(case, control)
    assert r.summary['excess_tail_mass'] == pytest.approx(ks_2samp(control, case, alternative='greater').statistic)
    assert r.summary['excess_tail_mass'] == pytest.approx(r.summary['case_tail_mass']-r.summary['control_tail_mass'])


def test_attribution_and_selection_are_scale_equivariant():
    x, y = np.array([0., 2, 10]), np.array([0., 1, 2, 3])
    r, s = evaluate(x, y), evaluate(x*100, y*100)
    np.testing.assert_allclose(s.contributions, r.contributions*100)
    np.testing.assert_allclose(s.membership, r.membership)
    assert s.summary['effective_contributor_fraction'] == pytest.approx(r.summary['effective_contributor_fraction'])


def test_unequal_case_control_noise_is_not_cured_by_a_descriptive_cutoff():
    r = evaluate([0, 0, 3], [1, 1, 1])  # equal means, different precision/shape
    assert r.summary['area'] > 0
    assert r.summary['excess_tail_mass'] > 0
