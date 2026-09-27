"""Matrix resampling and boundary intervals must retain finite-study uncertainty."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta
from itertools import combinations

from review_points import interval, summarize, two_level_tail
from wade.diagnostics import gene_detail


def test_exact_null_intervals_and_no_degenerate_zero_fdr_claim():
    values = np.r_[1, np.zeros(19)]
    result = interval(values, ['gamma']*20, binary=True)
    assert result['mean'] == .05
    assert result['low'] == pytest.approx(beta.ppf(.025, 1, 20))
    assert result['high'] == pytest.approx(beta.ppf(.975, 2, 19))
    zeros = interval(np.zeros(20), ['gamma']*10+['lognormal']*10)
    assert zeros['mean'] == zeros['low'] == 0
    assert zeros['high'] > .05
    assert zeros['high'] == pytest.approx(1-.0125**.1)


def test_stratified_summary_does_not_treat_genes_as_replicates():
    frame = pd.DataFrame(dict(method=['A']*4, biology=['gamma']*2+['lognormal']*2,
                              value=[.1, .3, .5, .7]))
    summary = summarize(frame, ['method'], 'value')
    pooled = summary[summary.biology == 'pooled'].iloc[0]
    assert pooled.n == 4
    assert pooled['mean'] == pytest.approx(.4)
    assert pooled.low < .4 < pooled.high
    assert summary[summary.biology == 'gamma'].iloc[0].n == 2
    missing = interval([np.nan], ['gamma'])
    assert missing['n'] == 0 and np.isnan(missing['mean'])


@pytest.mark.parametrize('nc,n0,k', [(3, 4, 1), (4, 3, 1), (4, 4, 2)])
def test_tail_limit_matches_exhaustive_quantile_permutations(nc, n0, k):
    x = np.r_[np.full(k, 17.), np.ones(nc+n0-k)]
    upward, unsigned = [], []
    for cases in combinations(range(nc+n0), nc):
        labels = np.zeros(nc+n0, int)
        labels[list(cases)] = 1
        detail = gene_detail(x, labels)
        upward.append(np.sum(np.maximum(detail.effect, 0)*np.diff(detail.probability)))
        unsigned.append(detail.cumulative_area[-1])
    observed = 16*k/nc
    exact = two_level_tail(nc, n0, k)
    assert exact['p_up'] == pytest.approx(np.mean(np.array(upward) >= observed-1e-12))
    assert exact['p_unsigned'] == pytest.approx(np.mean(np.array(unsigned) >= observed-1e-12))
