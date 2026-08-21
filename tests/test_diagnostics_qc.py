"""Per-library QC and subset attribution — ``wade.library_qc``, ``wade.subset_drivers``.

Both exist because of a real false positive on real data: a handful of
low-complexity plasma libraries topped the subset ranking of thousands of
genes at once (``notebooks/rna100k.qmd``). These tests plant that exact
failure mode and check the diagnostics separate it from a genuine subset.
"""

from __future__ import annotations

import numpy as np
import pytest

import wade

N1 = N0 = 60
COND = np.r_[np.ones(N1, int), np.zeros(N0, int)]
G = 600        # enough genes that one gene's share of a library is small,
               # which is what makes `share` discriminating at all


@pytest.fixture(scope="module")
def planted():
    """A cohort with one genuine subset gene and one artefact library set.

    ``real`` is elevated in 10% of cases across ordinary libraries.
    ``hitch`` is elevated in the *same* handful of samples that are
    low-complexity overall — the artefact: those libraries are zero in most
    genes, so whatever they do detect dominates them.
    """
    rng = np.random.default_rng(0)
    counts = rng.poisson(50, (G, N1 + N0)).astype(float)
    real, hitch = 0, 1
    genuine = np.arange(6)                       # 10% of cases, ordinary libraries
    counts[real, genuine] = rng.poisson(50 * 20, genuine.size)

    bad = np.arange(N1 - 5, N1)                  # 5 low-complexity case libraries
    detects = np.r_[hitch, np.arange(2, 32)]     # ... the only genes they see
    blind = np.setdiff1d(np.arange(G), detects)
    counts[np.ix_(blind, bad)] = 0.0
    counts[hitch, bad] = rng.poisson(50 * 20, bad.size)
    return counts, real, hitch, genuine, bad


def test_library_qc_flags_the_low_complexity_libraries(planted):
    counts, _, _, _, bad = planted
    qc = wade.library_qc(counts)
    assert set(qc) == {"depth", "n_detected", "detected_fraction", "top_share"}
    for v in qc.values():
        assert v.shape == (N1 + N0,)
    ok = np.ones(N1 + N0, bool); ok[bad] = False
    assert qc["n_detected"][bad].max() < qc["n_detected"][ok].min()
    assert qc["top_share"][bad].min() > qc["top_share"][ok].max()
    assert np.all(qc["detected_fraction"] <= 1.0)
    # a normalizer moves it to the rate scale the statistic sees
    qc_rate = wade.library_qc(counts, normalizer=np.arange(1.0, G + 1.0))
    assert not np.allclose(qc_rate["top_share"], qc["top_share"])
    with pytest.raises(ValueError, match="2-D"):
        wade.library_qc(np.arange(5.0))


def test_subset_drivers_names_the_samples_behind_a_call(planted):
    counts, real, hitch, genuine, bad = planted
    names = np.array([f"lib{j}" for j in range(N1 + N0)], dtype=object)
    res = wade.wade(counts, np.ones(G), COND, nperms=200, seed=1, sample_names=names)

    d_real = wade.subset_drivers(res, real)
    d_hitch = wade.subset_drivers(res, hitch)
    # the genuine gene's drivers are the samples it was planted in
    assert set(d_real["columns"]) >= set(genuine[:4]), d_real["columns"]
    # the artefact gene's drivers are the low-complexity libraries
    assert set(d_hitch["columns"]) >= set(bad[:3]), d_hitch["columns"]
    assert d_real["samples"][0].startswith("lib")
    assert d_real["direction"] > 0

    # `share` is the discriminator: the artefact gene owns its drivers'
    # libraries, the genuine one is a small part of ordinary libraries.
    assert np.median(d_hitch["share"]) > 10 * np.median(d_real["share"])

    # QC of the drivers tells the same story, which is the intended pairing
    qc = wade.library_qc(counts)
    assert (np.median(qc["n_detected"][d_hitch["columns"]])
            < 0.5 * np.median(qc["n_detected"][d_real["columns"]]))


def test_subset_drivers_edges(planted):
    counts, real, _, _, _ = planted
    res = wade.wade(counts, np.ones(G), COND, nperms=100, seed=1)
    d = wade.subset_drivers(res, real, k=3)
    assert d["columns"].size == 3
    assert wade.subset_drivers(res, res.gene[real])["gene"] == res.gene[real]
    assert d["samples"] is not None            # positional names are synthesized
    no_sub = wade.wade(counts, np.ones(G), COND, nperms=0)
    with pytest.raises(ValueError, match="no subset stage"):
        wade.subset_drivers(no_sub, real)
