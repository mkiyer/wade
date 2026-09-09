"""Edge cases of the public surface found in the pre-publication review, each
of which produced a crash or a plausible wrong answer rather than an error."""

from __future__ import annotations

import numpy as np
import pytest

import wade
from wade.subset import _resolve_pseudocount

G, N1, N0 = 30, 12, 12
COND = np.r_[np.ones(N1, int), np.zeros(N0, int)]


@pytest.fixture(scope="module")
def counts():
    return np.random.default_rng(0).poisson(20, (G, N1 + N0)).astype(float)


def test_gene_detail_works_with_a_matrix_normalizer(counts):
    norm = np.random.default_rng(1).uniform(500, 2000, size=(G, N1 + N0))
    res = wade.wade(counts, norm, COND, nperms=20, seed=1)
    d = res.gene_detail(3)
    np.testing.assert_array_equal(res.pseudocount_row(3), res.pseudocount[3])
    assert d.r.shape == (res.nprobs,) and np.all(np.isfinite(d.r))


def test_an_all_zero_matrix_is_refused_like_any_empty_library():
    with pytest.raises(ValueError, match="zero library size"):
        wade.wade(np.zeros((G, N1 + N0)), np.ones(G), COND, nperms=10)
    # ... while a matrix with no genes has no libraries to judge.
    res = wade.wade(np.zeros((0, N1 + N0)), np.ones(0), COND, nperms=10)
    assert res.gene.shape == (0,)


def test_stage1_intervals_do_not_need_the_subset_stage(counts):
    res = wade.wade(counts, np.ones(G), COND, nperms=0, n_boot=20, seed=1)
    assert res.ci_log2_fc.shape == (2, G) and res.ci_mean_shift.shape == (2, G)
    assert res.ci_affected_fraction is None
    assert {"log2_fc_lo", "mean_shift_hi"} <= set(res.columns())
    assert "affected_fraction_lo" not in res.columns()


def test_supplied_permutations_set_nperms(counts):
    perms = wade.draw_perms(COND, 37, seed=2)
    res = wade.wade(counts, np.ones(G), COND, perms=perms)
    assert res.params["nperms"] == 37 and res.perms.shape == (37, N1 + N0)
    with pytest.raises(ValueError, match="non-negative"):
        wade.wade(counts, np.ones(G), COND, nperms=-1)


def test_subset_drivers_are_sized_by_the_case_group_not_the_grid():
    """``affected_fraction`` is a fraction of the cases, so on an unbalanced
    design the driver count follows ``n_case``, not ``min(n_case, n_ctrl)``."""
    rng = np.random.default_rng(3)
    n1, n0 = 60, 15
    c = rng.poisson(30, (20, n1 + n0)).astype(float)
    c[0, :30] = rng.poisson(300, 30)                 # half the cases, sharply up
    cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
    res = wade.wade(c, np.ones(20), cond, lib_sizes=np.ones(n1 + n0), nperms=50, seed=1)
    d = wade.subset_drivers(res, 0)
    assert len(d["columns"]) == int(np.ceil(res.affected_fraction[0] * n1))
    assert len(d["columns"]) > res.nprobs


def test_pseudocount_validation_reaches_every_form():
    shape = (4, 6)
    for bad in (-5.0, np.nan, [1, -1, 1, 1, 1, 1], np.full(shape, np.inf)):
        with pytest.raises(ValueError, match="finite and non-negative"):
            _resolve_pseudocount(bad, shape)
    assert _resolve_pseudocount(0, shape) is None
    assert _resolve_pseudocount(2.0, shape).shape == shape


def test_a_single_gene_name_labels_that_gene(counts):
    from wade.plotting import volcano_data

    res = wade.wade(counts, np.ones(G), COND, nperms=20, seed=1)
    assert volcano_data(res, "mean_shift", label="gene4").labelled.sum() == 1
    assert volcano_data(res, "mean_shift", label=["gene4", "gene5"]).labelled.sum() == 2


def test_condition_names_the_type_of_a_mismatched_level():
    sheet = {"sample_id": [f"s{i}" for i in range(4)], "group": [1, 1, 0, 0]}
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        wade.condition(sheet, key="sample_id", column="group", case="1", control="0")
