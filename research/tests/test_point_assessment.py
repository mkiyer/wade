"""Check known-truth bookkeeping and metric denominators before cohort runs."""
import json
from pathlib import Path

import numpy as np
import pytest

from assess_points import generate, mean_ratio, operating_point


def test_truth_tracks_depressed_bulk_without_relabeling_it_as_upward_mean_de():
    settings = json.loads((Path(__file__).parents[1]/'protocols/point-release.json').read_text())['engineering']
    settings = dict(settings, genes=128)
    data = generate(settings, 'mixed')
    assert data['counts'].dtype.kind in 'iu'
    assert data['altered'].sum() == 16
    assert not data['altered'][:64].any()
    assert (data['counts'][-1] == 0).all()
    assert (data['counts'].sum(axis=0) > 0).all()
    assert np.exp(np.log(data['exposure']).mean()) == pytest.approx(1)
    selected = data['patterns'] == 'bulk_half_subset_up_10pct_4x'
    np.testing.assert_allclose(data['population_mean_ratio'][selected], .85)
    # This bookkeeping describes perturbations, not estimates of enriched membership.
    assert data['affected'][selected].all()
    null = generate(settings, 'complete_null')
    assert not null['altered'].any()


def test_metrics_distinguish_false_discovery_fraction_from_specificity():
    result = operating_point([1, 1, 0, 0, 0], [1, 0, 1, 0, 0])
    assert (result['tp'], result['fp'], result['tn'], result['fn']) == (1, 1, 2, 1)
    assert result['sensitivity'] == .5
    assert result['specificity'] == 2/3
    assert result['fdp'] == .5
    assert operating_point([0, 0], [0, 0])['sensitivity'] is None
    ratio = mean_ratio(np.array([[0, 0, 0, 0], [2, 2, 1, 1.]]), np.array([1, 1, 0, 0]))
    assert np.isnan(ratio[0])  # No invented fold change for 0/0.
    assert ratio[1] == 1
