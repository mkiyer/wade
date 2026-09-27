"""Independent quantile/orbit checks before directional production migration."""
from itertools import combinations

import numpy as np
import pytest
from scipy.stats import false_discovery_control, wasserstein_distance

from wade import fit_normalization
from wade._wasserstein import TIE_RTOL
from wade.diagnostics import gene_detail
from wade.permutation import draw_perms
from wade.pvalues import sequential_p
from wade_research.point_workflow import DirectionalState, point_workflow


def test_crossing_curves_keep_both_directions_without_cancellation():
    x = np.array([[2, 2, 2, 30, 10, 10, 10, 10.]])
    cond = np.r_[np.ones(4), np.zeros(4)]
    np.testing.assert_allclose(DirectionalState(x, cond).observed, [[5, 6]])
    assert wasserstein_distance(x[0, :4], x[0, 4:]) == 11
    assert x[0, :4].mean() - x[0, 4:].mean() == -1


@pytest.mark.parametrize("ncase,ncontrol", [(3, 4), (4, 4), (2, 5)])
def test_exact_orbits_against_independent_quantile_integration(ncase, ncontrol):
    n = ncase + ncontrol
    x = np.random.default_rng(728).integers(0, 7, (4, n)).astype(float)
    x[0] = 0
    labels = []
    for selected in combinations(range(n), ncase):
        cond = np.zeros(n, int)
        cond[list(selected)] = 1
        labels.append(cond)
    values = []
    for cond in labels:
        result = DirectionalState(x, cond).observed
        inverse = DirectionalState(x, 1-cond).observed
        np.testing.assert_allclose(result, inverse[:, ::-1])
        for gene, row in enumerate(x):
            detail = gene_detail(row, cond)
            expected = [np.sum(np.maximum(sign*detail.effect, 0)*np.diff(detail.probability))
                        for sign in (1, -1)]
            np.testing.assert_allclose(result[gene], expected, atol=1e-14)
            assert result[gene].sum() == pytest.approx(wasserstein_distance(row[cond == 1], row[cond == 0]))
            assert result[gene, 0]-result[gene, 1] == pytest.approx(row[cond == 1].mean()-row[cond == 0].mean())
        values.append(result)
    values = np.array(values)
    for gene in range(len(x)):
        for direction in range(2):
            score = values[:, gene, direction]
            p = (score[None, :] >= score[:, None]*(1-TIE_RTOL)).mean(axis=1)
            for alpha in np.unique(p):
                assert np.mean(p <= alpha) <= alpha + 1e-14


def test_sequential_records_match_fixed_bank_and_batch_boundaries():
    x = np.random.default_rng(445).poisson(10, (9, 11)).astype(float)
    cond = np.r_[np.ones(5), np.zeros(6)]
    bank = draw_perms(cond, 79, seed=234)
    original = DirectionalState(x, cond, stop=7)
    scores = np.array([DirectionalState(x, row).observed for row in bank])
    hits = scores >= original.observed[None]*(1-TIE_RTOL)
    expected_k, expected_t = np.zeros((9, 2), int), np.zeros((9, 2), int)
    for gene in range(9):
        for direction in range(2):
            crossings = np.flatnonzero(hits[:, gene, direction])
            used = crossings[6]+1 if len(crossings) >= 7 else len(bank)
            expected_k[gene, direction] = hits[:used, gene, direction].sum()
            expected_t[gene, direction] = used
    for batch in (1, 11, 79):
        state = DirectionalState(x, cond, stop=7)
        for start in range(0, len(bank), batch):
            state.update(bank[start:start+batch])
        np.testing.assert_array_equal(state.exceed, expected_k)
        np.testing.assert_array_equal(state.tried, expected_t)
        sequential_p(state.exceed, state.tried, stop=7, maximum=79)


@pytest.mark.parametrize("scale", ["linear", "log"])
def test_full_workflow_reuses_normalization_and_corrects_one_pair_family(scale):
    x = np.random.default_rng(803).poisson(10, (80, 20))
    x[0] = 0
    x[1, :10] *= 10
    cond = np.tile([0, 1], 10)
    strata = np.repeat(np.arange(4), 5)
    norm = fit_normalization(x)
    settings = dict(normalization=norm, scale=scale, max_permutations=127, seed=789, strata=strata)
    result = point_workflow(x, cond, batch=17, **settings)
    mask = np.arange(len(x)) % 2 == 0
    filtered = point_workflow(x, cond, test_genes=mask, batch=31, **settings)
    np.testing.assert_array_equal(result.normalization.size_factors, filtered.normalization.size_factors)
    np.testing.assert_array_equal(result.pvalue[mask], filtered.pvalue[mask])
    np.testing.assert_array_equal(result.n_permutations[mask], filtered.n_permutations[mask])
    finite = np.isfinite(filtered.pvalue)
    np.testing.assert_allclose(filtered.padj[finite], false_discovery_control(filtered.pvalue[finite]))
    assert np.isnan(filtered.padj[~filtered.tested]).all()
    assert (filtered.n_permutations[~filtered.tested] == 0).all()
    assert filtered.params['hypotheses'] == 2*filtered.tested.sum()


def test_no_eligible_genes_and_fixed_ties_complete_cleanly():
    x = np.ones((4, 6))
    cond = [1, 1, 1, 0, 0, 0]
    empty = point_workflow(x, cond, test_genes=np.zeros(4, bool), max_permutations=23)
    assert np.isnan(empty.pvalue).all()
    tied = point_workflow(x, cond, max_permutations=23)
    np.testing.assert_array_equal(tied.pvalue, 1)
    np.testing.assert_array_equal(tied.statistic, 0)
    with pytest.raises(ValueError, match="exchanges"):
        point_workflow(x, cond, strata=cond)
