"""Generating truth is independent of observed calls and respects the design."""
import json
from pathlib import Path

import numpy as np
import pytest

from wade_research.point_simulation import datasets, generate


@pytest.mark.parametrize("biology", ["gamma", "correlated_lognormal"])
def test_population_truth_membership_and_null_sentinels(biology):
    cfg = json.loads((Path(__file__).parents[1]/"protocols/confirmation.json").read_text())
    cfg = dict(cfg, genes=256)
    settings = dict(next(datasets(cfg)), panel="mixed", biology=biology)
    data = generate(cfg, settings)
    assert data['counts'].dtype.kind in 'iu'
    assert data['altered'].sum() == 60
    assert data['mean_altered'].sum() == 52
    for profile in ('mean_preserving_subset_10pct_4x', 'same_mean_wider'):
        np.testing.assert_allclose(data['population_log2_ratio'][data['profile'] == profile], 0, atol=1e-15)
    selected = data['profile'] == 'bulk_half_subset_up_10pct_32x'
    np.testing.assert_allclose(data['population_log2_ratio'][selected], np.log2(.9*.5 + .1*32))
    assert data['perturbed'][selected].all()
    np.testing.assert_array_equal(data['enriched'][selected].sum(axis=1), 25)
    assert not data['altered'][:64].any()
    assert not data['altered'][data['profile'] == 'shared_tail_null'].any()
    np.testing.assert_array_equal(data['counts'][-1], 0)
    assert (data['counts'].sum(axis=0) > 0).all()
    null = generate(cfg, dict(settings, panel='complete_null'))
    assert not null['altered'].any()
    assert not null['mean_altered'].any()


def test_registry_has_independent_seeds_and_fixed_balanced_biology():
    cfg = json.loads((Path(__file__).parents[1]/"protocols/confirmation.json").read_text())
    rows = list(datasets(cfg))
    assert len(rows) == 120
    assert len({tuple(r['seed']) for r in rows}) == 120
    assert len({tuple(r['permutation_seed']) for r in rows}) == 120
    assert {tuple(r['seed']) for r in rows}.isdisjoint({tuple(r['permutation_seed']) for r in rows})
    for allocation in cfg['allocations']:
        for panel in cfg['panels']:
            selected = [r for r in rows if [r['n_case'], r['n_control']] == allocation and r['panel'] == panel]
            assert sum(r['biology'] == 'gamma' for r in selected) == 10
