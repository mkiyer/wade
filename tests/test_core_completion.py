"""The last core gaps, closed — `docs/plan.md` Phase D.

Five of the six are things the documentation already promised or already made
a reading rule out of, and one is a framing decision: WADE's core needs a
single gene id and nothing more, so any other per-gene column is **carried and
optionally displayed, never read**.
"""

from __future__ import annotations

import numpy as np
import pytest

import wade

G, N = 30, 50
COND = np.r_[np.ones(N // 2, int), np.zeros(N // 2, int)]


@pytest.fixture(scope="module")
def result():
    rng = np.random.default_rng(0)
    counts = rng.poisson(50, (G, N)).astype(float)
    counts[0, :5] = rng.poisson(50 * 30, 5)          # a strong subset gene
    return wade.wade(counts, np.ones(G), COND, nperms=600, seed=1)


# --- D1/D2: stage 2 keeps and exports what stage 1 always did ---------------


def test_both_stages_report_exceedances_and_refinement(result):
    """`docs/method.md` §10.3 makes a reading rule out of these — a gene *at* the floor
    was censored there, not measured there — and it was appliable to stage 1
    only, because stage 2 computed the pair and discarded it."""
    for nexc, refined in ((result.nexc_mean_shift, result.refined_mean_shift),
                          (result.nexc_subset, result.refined_subset)):
        assert nexc.shape == (G,) and refined.shape == (G,)
        assert nexc.dtype.kind == "i" and refined.dtype == bool
        # refinement fires only where the exceedance count is short
        assert np.all(nexc[refined] < wade.pvalues.DEFAULT_N_EXC_MIN)

    rep = result.columns()
    assert "refined_mean_shift" in rep and "refined_subset" in rep
    # the branch is exercised on this design, for both stages
    assert result.refined_subset.any() and result.refined_mean_shift.any()

    # Worth pinning because it is counter-intuitive: refinement replaces a
    # coarse exceedance count with a *model* estimate, so it can move a
    # p-value in either direction. Here a refined stage-2 p-value (0.047) sits
    # above an unrefined one (0.018) — which is exactly why the flag has to be
    # exported rather than inferred from the p-value's size.
    refined_max = result.p_subset[result.refined_subset].max()
    plain_min = result.p_subset[~result.refined_subset].min()
    assert refined_max > plain_min


def test_no_subset_stage_means_no_subset_diagnostics():
    rng = np.random.default_rng(1)
    counts = rng.poisson(50, (G, N)).astype(float)
    res = wade.wade(counts, np.ones(G), COND, nperms=0)
    assert res.nexc_subset is None and res.refined_subset is None
    assert "refined_subset" not in res.columns()


# --- D3: the check method.md §10 says to run before anything -------------------


def test_detectability_floor_is_the_documented_formula():
    from math import comb

    for n1, n0, k in ((77, 18, 15), (48, 47, 15), (100, 100, 5), (10, 10, 1)):
        assert wade.detectability_floor(n1, n0, k) == pytest.approx(
            comb(n1, k) / comb(n1 + n0, k))

    # method.md §10.1's own example: 77 v 18 cannot reach 0.05 with 15 affected
    assert wade.detectability_floor(77, 18, 15) > 0.03
    # ... and balancing the same 95 samples buys three orders of magnitude
    assert wade.detectability_floor(48, 47, 15) < 1e-4
    # more affected samples make a signal easier to detect
    floors = [wade.detectability_floor(50, 50, k) for k in (2, 5, 10, 20)]
    assert floors == sorted(floors, reverse=True)
    # k that cannot fit in the case group imposes no floor
    assert wade.detectability_floor(5, 50, 10) == 0.0
    with pytest.raises(ValueError, match="k must be"):
        wade.detectability_floor(10, 10, 0)
    with pytest.raises(ValueError, match="non-empty"):
        wade.detectability_floor(0, 10, 1)


# --- D4: the interval level ------------------------------------------------


def test_boot_level_widens_the_intervals():
    rng = np.random.default_rng(2)
    counts = rng.poisson(50, (12, N)).astype(float)
    kw = dict(nperms=40, seed=1, n_boot=80)
    narrow = wade.wade(counts, np.ones(12), COND, boot_level=0.50, **kw)
    wide = wade.wade(counts, np.ones(12), COND, boot_level=0.99, **kw)
    for name in ("affected_fraction", "direction", "subset_log2_fc",
                 "log2_fc", "mean_shift"):
        n = getattr(narrow, f"ci_{name}")
        w = getattr(wide, f"ci_{name}")
        assert np.all((w[1] - w[0]) >= (n[1] - n[0]) - 1e-12), name


# --- D5: a bad count is located --------------------------------------------


def test_a_bad_count_is_located_and_not_substituted():
    counts = np.full((3, 4), 5.0)
    counts[2, 1] = np.nan
    with pytest.raises(ValueError, match=r"gene2/sample1"):
        wade.as_counts(counts)
    with pytest.raises(ValueError, match="not a zero"):
        wade.as_counts(counts)


# --- D6: metadata is carried, never read -----------------------------------


def test_gene_metadata_is_carried_and_changes_no_number():
    pl = pytest.importorskip("polars")
    rng = np.random.default_rng(3)
    m = rng.poisson(40, (G, N)).astype(float)
    samples = [f"S{j}" for j in range(N)]
    ids = [f"ENSG{i}" for i in range(G)]
    bare = pl.DataFrame({"gene_id": ids, **{s: m[:, j] for j, s in enumerate(samples)}})
    rich = bare.with_columns(
        pl.Series("gene_name", [f"SYM{i}" for i in range(G)]),
        pl.Series("gene_type", ["protein_coding"] * G),
    ).select(["gene_id", "gene_name", "gene_type", *samples])

    a = wade.wade(bare, 1.0, COND, nperms=40, seed=1)
    b = wade.wade(wade.as_counts(rich, sample_columns=samples), 1.0, COND,
                  nperms=40, seed=1)

    # carried, and available for display
    assert sorted(b.gene_meta) == ["gene_name", "gene_type"]
    assert list(b.gene_meta["gene_name"][:2]) == ["SYM0", "SYM1"]
    assert a.gene_meta == {}
    # the gene id remains the label the core uses
    assert list(b.gene[:2]) == ["ENSG0", "ENSG1"]
    # and not one number moved
    for name in ("mean_shift", "p_mean_shift", "p_subset", "affected_fraction",
                 "subset_log2_fc", "direction"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)
