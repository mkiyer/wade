"""Endpoint denominators and effect estimates are checked separately from fits."""
import numpy as np
import pytest

from compare_points import effect_summary, mean_ratio, operating_point


def test_detection_units_and_no_discovery_convention():
    result = operating_point([1, 1, 0, 0], [1, 0, 1, 0])
    assert [result[k] for k in ('tp', 'fp', 'tn', 'fn')] == [1, 1, 1, 1]
    assert result['fdp'] == result['specificity'] == result['sensitivity'] == .5
    null = operating_point([0, 0], [0, 0])
    assert null['fdp'] == 0
    assert np.isnan(null['sensitivity'])
    assert np.isnan(null['precision'])
    pair = operating_point([[1, 1], [0, 0]], [[0, 0], [1, 1]])
    assert pair['fp'] == 2


def test_fold_change_errors_preserve_undefined_values_and_masks():
    ratio = mean_ratio(np.array([[8, 8, 2, 2], [0, 0, 0, 0], [1, 0, 0, 0.]]), np.array([1, 1, 0, 0]))
    np.testing.assert_array_equal(ratio, [2, np.nan, np.nan])
    result = effect_summary(np.array([1, 3, np.nan, 99]), np.array([0, 1, 2, 3]), np.array([1, 1, 1, 0], bool))
    assert (result['n'], result['n_finite'], result['n_missing']) == (3, 2, 1)
    assert result['rmse'] == pytest.approx(np.sqrt(2.5))
    assert result['bias'] == 1.5
