"""Ranking past the p-value floor: permutation z-scores and the subset's
magnitude (``subset_log2_fc``).

On large cohorts the empirical/GPD p-values saturate — measured on the
rna100k plasma contrast, most of the transcriptome ties at the floor — so
the p-value stops ordering genes. Two additions fix the two halves of that:

* ``z_mean_shift`` / ``z_subset`` — the observed statistic standardized
  against the gene's own permutation null (the GSEA-NES analogue), free
  because the null matrix is already in hand. A *ranking*, explicitly not a
  calibrated tail probability.
* ``subset_log2_fc`` — the magnitude the subset stage lacked: the log2 fold
  change within the affected fraction, read off the log-ratio curve over
  the region ``affected_fraction`` names. A 5% subset at 8x and one at 100x
  have the same saturated p-value and very different values here.
"""

from __future__ import annotations

import numpy as np
import pytest

import wade

N1 = N0 = 150
COND = np.r_[np.ones(N1, int), np.zeros(N0, int)]
MU = 50.0


def _cohort(seed=0):
    """Null genes plus planted subsets of known fraction and magnitude."""
    rng = np.random.default_rng(seed)
    nb = lambda mu, size: rng.poisson(rng.gamma(10.0, mu / 10.0, size))  # noqa: E731
    blocks = {}
    genes = []
    for name, frac, fold, reps in (
        ("null", 0.0, 1.0, 30),
        ("sub8", 0.10, 8.0, 8),          # 10% subset at 8x
        ("sub100", 0.10, 100.0, 8),      # 10% subset at 100x
        ("down8", 0.10, 1 / 8.0, 8),     # 10% subset at 1/8
        ("global2", 1.0, 2.0, 8),        # global 2x
    ):
        idx = []
        for _ in range(reps):
            case = nb(MU, N1)
            k = round(frac * N1)
            if k:
                case[:k] = nb(MU * fold, k)
            idx.append(len(genes))
            genes.append(np.r_[case, nb(MU, N0)])
        blocks[name] = np.array(idx)
    return np.array(genes, dtype=float), blocks


@pytest.fixture(scope="module")
def res_and_blocks():
    counts, blocks = _cohort()
    res = wade.wade(counts, np.ones(counts.shape[0]), COND,
                    lib_sizes=np.ones(N1 + N0), nperms=300, seed=1)
    return res, blocks


def test_subset_log2_fc_reads_the_planted_magnitude(res_and_blocks):
    """The comparison is quantile-matched — affected cases against the
    controls' own upper tail, not the control mean — so a subset at 8x the
    NB(50) mean reads ~2, not log2(8) = 3 (the controls' top decile sits at
    ~70-90). Measured medians: sub8 +2.04, sub100 +5.53, down8 -1.91,
    global2 +0.99, null -0.01."""
    res, b = res_and_blocks
    m = res.subset_log2_fc
    assert 1.5 < np.median(m[b["sub8"]]) < 2.8, np.median(m[b["sub8"]])
    assert 4.5 < np.median(m[b["sub100"]]) < 6.5, np.median(m[b["sub100"]])
    # magnitude separates what the saturated p-value cannot
    assert np.median(m[b["sub100"]]) - np.median(m[b["sub8"]]) > 2.5
    # a downward subset reads negative, mirrored
    assert np.median(m[b["down8"]]) < -1.3
    # a global 2x has affected_fraction ~ 1, so this is the gene's overall fold change
    assert abs(np.median(m[b["global2"]]) - 1.0) < 0.3
    # null genes sit near zero
    assert abs(np.median(m[b["null"]])) < 0.3


def test_z_scores_rank_where_p_values_tie(res_and_blocks):
    res, b = res_and_blocks
    # At B=300 every planted subset saturates to the same floor p-value...
    floor = res.p_subset.min()
    assert (res.p_subset[b["sub8"]] == floor).all()
    assert (res.p_subset[b["sub100"]] == floor).all()
    # ...but the z-scores still separate signal from null and keep ordering.
    z = res.z_subset
    assert np.median(z[b["sub8"]]) > np.percentile(z[b["null"]], 95)
    assert np.median(z[b["sub100"]]) > np.median(z[b["null"]])
    zm = res.z_mean_shift
    assert np.all(np.isfinite(zm))
    assert abs(np.median(zm[b["null"]])) < 1.0        # centred on the null
    assert np.median(np.abs(zm[b["global2"]])) > 3.0  # far from it


def test_new_columns_are_reported_and_written(res_and_blocks):
    res, _ = res_and_blocks
    for name in ("z_mean_shift", "z_subset", "subset_log2_fc"):
        assert name in res.columns(), name
        assert name in res.report(), name
        assert name in wade.RESULT_COLUMNS, name
    rep = res.report()
    np.testing.assert_array_equal(rep["subset_log2_fc"], res.subset_log2_fc)


def test_without_permutations_the_z_scores_say_so():
    counts, _ = _cohort(1)
    res = wade.wade(counts, np.ones(counts.shape[0]), COND,
                    lib_sizes=np.ones(N1 + N0), nperms=0)
    assert np.all(np.isnan(res.z_mean_shift))
    assert res.z_subset is None and res.subset_log2_fc is None
    assert "z_subset" not in res.columns()
