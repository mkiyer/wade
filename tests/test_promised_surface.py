"""Things the README and docstrings promise that nothing exercised.

Found by the package audit, 2026-08-20: pandas frame input is advertised and
was never run (pandas was not even installed); ``cpm`` and ``rle`` are
README-listed normalizers with no coverage; ``alternative="less"`` is
documented on both stages and tested nowhere. A promise with no test is a
promise about code nobody runs.
"""

from __future__ import annotations

import numpy as np
import pytest

import wade

pd = pytest.importorskip("pandas")

G, N = 30, 24
SAMPLES = [f"S{j}" for j in range(N)]
GENES = [f"g{i}" for i in range(G)]
COND = np.r_[np.ones(N // 2, int), np.zeros(N // 2, int)]


@pytest.fixture(scope="module")
def matrix():
    rng = np.random.default_rng(5)
    return rng.poisson(60, size=(G, N)).astype(float)


# ---------------------------------------------------------------------------
# pandas: the duck-typed frame path, never previously run against pandas


def test_as_counts_reads_a_pandas_frame(matrix):
    frame = pd.DataFrame({"gene_id": GENES,
                          **{s: matrix[:, j] for j, s in enumerate(SAMPLES)}})
    c = wade.as_counts(frame)
    assert c.shape == (G, N)
    assert list(c.gene_names) == GENES and list(c.sample_names) == SAMPLES
    np.testing.assert_allclose(c.values, matrix)


def test_as_counts_pandas_index_carries_the_gene_labels(matrix):
    """The pandas-idiomatic layout puts gene ids in the **index**, which is
    not a column — so the labels would be lost silently. Reading the index
    explicitly is the documented remedy, and it works."""
    frame = pd.DataFrame(matrix, index=GENES, columns=SAMPLES)
    plain = wade.as_counts(frame)
    assert plain.gene_names[0] == "gene0"            # positional: index ignored
    named = wade.as_counts(frame, gene_names=list(frame.index))
    assert list(named.gene_names) == GENES
    np.testing.assert_allclose(named.values, matrix)


def test_pandas_metadata_and_normalizer_column(matrix):
    frame = pd.DataFrame({"gene_id": GENES, "Length": np.arange(1.0, G + 1.0),
                          **{s: matrix[:, j] for j, s in enumerate(SAMPLES)}})
    res = wade.wade(frame, "Length", COND, nperms=50, seed=1)
    assert res.gene.shape == (G,) and list(res.gene) == GENES
    assert np.all(np.isfinite(res.mean_shift))


def test_condition_from_a_pandas_sheet(matrix):
    sheet = pd.DataFrame({"sample_id": SAMPLES,
                          "grp": ["t"] * (N // 2) + ["n"] * (N // 2)})
    cond = wade.condition(sheet, key="sample_id", column="grp", case="t", control="n")
    np.testing.assert_array_equal(cond.vector(SAMPLES), COND)


# ---------------------------------------------------------------------------
# cpm and rle: README-listed normalizers, previously untested


def test_cpm_columns_sum_to_the_norm_factor(matrix):
    x = wade.cpm(matrix, seed=3)
    np.testing.assert_allclose(x.sum(axis=0), 1e6, rtol=1e-9)
    assert np.all(x > 0)                             # the jitter makes it strict
    with pytest.raises(ValueError, match="zero total count"):
        wade.cpm(np.zeros((4, 3)), jitter=np.zeros((4, 3)))


def test_cpm_is_scale_invariant_per_column(matrix):
    """Doubling a library must not change its CPM profile."""
    jit = np.zeros_like(matrix)
    a = wade.cpm(matrix, jitter=jit)
    doubled = matrix.copy(); doubled[:, 0] *= 2
    b = wade.cpm(doubled, jitter=jit)
    np.testing.assert_allclose(a[:, 0], b[:, 0], rtol=1e-9)


def test_rle_recovers_planted_size_factors():
    rng = np.random.default_rng(7)
    base = rng.gamma(20.0, 5.0, (200, 1))
    factors = np.array([0.5, 1.0, 2.0, 4.0])
    counts = rng.poisson(base * factors[None, :]).astype(float)
    x = wade.rle(counts, jitter=np.zeros(counts.shape))
    # after RLE the columns are on one scale, so their medians agree
    med = np.median(x, axis=0)
    assert med.max() / med.min() < 1.25, med
    # the documented refusal: nothing detected in every sample
    sparse = np.zeros((5, 4)); sparse[0, 0] = 5.0
    with pytest.raises(ValueError, match="strictly positive raw count"):
        wade.rle(sparse, jitter=np.zeros(sparse.shape))


def test_rle_accepts_an_explicit_reference():
    rng = np.random.default_rng(8)
    counts = rng.poisson(50, (40, 4)).astype(float)
    ref = np.full(40, 50.0)
    x = wade.rle(counts, reference=ref, jitter=np.zeros(counts.shape))
    assert x.shape == counts.shape and np.all(np.isfinite(x))
    with pytest.raises(ValueError, match="one value per gene"):
        wade.rle(counts, reference=ref[:-1])


# ---------------------------------------------------------------------------
# alternative="less": documented on both stages, previously untested


def _planted(rng, direction):
    """A gene shifted down (or up) in a subset of cases."""
    counts = rng.poisson(60, (40, N)).astype(float)
    k = N // 2 // 4
    fold = 0.1 if direction == "down" else 10.0
    counts[0, :k] = rng.poisson(60 * fold, k)
    return counts


def test_alternative_less_finds_a_downward_subset_and_greater_does_not():
    rng = np.random.default_rng(9)
    counts = _planted(rng, "down")
    less = wade.wade(counts, np.ones(40), COND, nperms=400, seed=1,
                     alternative="less", lib_sizes=np.ones(N))
    greater = wade.wade(counts, np.ones(40), COND, nperms=400, seed=1,
                        alternative="greater", lib_sizes=np.ones(N))
    assert less.p_mean_shift[0] < greater.p_mean_shift[0]
    assert less.p_mean_shift[0] < 0.05
    assert less.params["alternative"] == "less"
    # ... and the mirror image holds for an upward subset
    up = _planted(np.random.default_rng(9), "up")
    a = wade.wade(up, np.ones(40), COND, nperms=400, seed=1,
                  alternative="greater", lib_sizes=np.ones(N))
    b = wade.wade(up, np.ones(40), COND, nperms=400, seed=1,
                  alternative="less", lib_sizes=np.ones(N))
    assert a.p_mean_shift[0] < b.p_mean_shift[0]


def test_alternative_less_is_accepted_by_both_stages_and_the_plots():
    rng = np.random.default_rng(10)
    counts = _planted(rng, "down")
    res = wade.wade(counts, np.ones(40), COND, nperms=200, seed=1, alternative="less")
    assert res.p_subset is not None and np.all(np.isfinite(res.p_subset))
    v = wade.plotting.volcano_data(res, "subset")
    assert v.alternative == "less"
