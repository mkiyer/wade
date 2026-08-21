"""Things the README and docstrings promise that nothing exercised.

Found by the package audit, 2026-08-20: pandas frame input is advertised and
was never run (pandas was not even installed), and ``alternative="less"`` is
documented on both stages and tested nowhere. A promise with no test is a
promise about code nobody runs. (``cpm`` and ``rle`` were the third case; the
simplification audit then showed they were unreachable from ``wade()`` and
they were deleted instead — see the git log.)
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
