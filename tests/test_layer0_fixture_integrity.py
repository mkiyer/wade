"""Layer 0 — the fixtures themselves.

Before anything is compared, establish that the fixtures are what they
claim to be. A parity suite built on fixtures that are silently
transposed, square where they should not be, or lossy in their last bits
is a suite that passes on broken code.

Note that merely *loading* a fixture is already a test: the loader
asserts, for every value in every file, that the 17-digit decimal channel
round-trips to the same double as the exact hex channel.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import (
    DIVERGENT_SCENARIOS,
    FIXTURE_DIR,
    PARITY_SCENARIOS,
    load_fixture,
)

ALL_SCENARIOS = PARITY_SCENARIOS + DIVERGENT_SCENARIOS


def test_fixtures_exist():
    assert FIXTURE_DIR.is_dir(), f"no fixture directory at {FIXTURE_DIR}"
    emitted = {p.stem for p in FIXTURE_DIR.glob("*.json")}
    expected = set(ALL_SCENARIOS) | {
        "gpd_branches", "bh_padjust", "ecdf_denserank",
        "worked_example", "quantile_type7",
    }
    assert expected <= emitted, f"missing fixtures: {sorted(expected - emitted)}"


@pytest.mark.parametrize("name", ALL_SCENARIOS)
def test_shapes_are_self_consistent(name):
    """Declared shapes must match the arrays, and the grid arithmetic must hold."""
    fx = load_fixture(name)
    g = int(fx["shapes"]["g"])
    n = int(fx["shapes"]["n"])
    m = int(fx["shapes"]["nprobs"])
    n1 = int(fx["shapes"]["n1"])
    n0 = int(fx["shapes"]["n0"])
    B = int(fx["shapes"]["B"])

    assert fx["counts"].shape == (g, n)
    assert fx["jitter"].shape == (g, n)
    assert fx["tpm"].shape == (g, n)
    assert fx["cond"].shape == (n,)
    assert fx["lib_sizes"].shape == (n,)
    assert fx["q"].shape == (m,)
    assert fx["perms"].shape == (B, n)

    assert n1 + n0 == n
    assert m == min(n1, n0)

    norm = fx["normalizer"]
    if norm["form"] == "matrix":
        assert norm["data"].shape == (g, n)
    else:
        assert norm["data"].shape == (g,), "a vector normalizer is per-gene, not per-sample"


@pytest.mark.parametrize("name", PARITY_SCENARIOS)
def test_quantile_grids_are_genes_by_probabilities(name):
    """``Q1`` and ``Q0`` must be ``(genes, nprobs)``.

    Split out from the general shape test because ``onesample`` violates
    it *in the reference*: at ``nprobs == 1`` R's reshape guard turns a
    length-``g`` vector into a ``1 x g`` matrix, transposing genes into
    probabilities. That is hazard 11 and it is asserted deliberately in
    ``test_divergences.py`` rather than tolerated here.
    """
    fx = load_fixture(name)
    g, m = int(fx["shapes"]["g"]), int(fx["shapes"]["nprobs"])
    assert fx["Q1"].shape == (g, m)
    assert fx["Q0"].shape == (g, m)
    assert fx["D"].shape == (g, m)


@pytest.mark.parametrize("name", ALL_SCENARIOS)
def test_permutation_rows_preserve_group_sizes(name):
    """Every permutation must be a relabelling, not a resampling."""
    fx = load_fixture(name)
    perms = fx["perms"]
    if perms.size == 0:
        return
    want = np.sort(fx["cond"])
    got = np.sort(perms, axis=1)
    assert np.array_equal(got, np.broadcast_to(want, got.shape))


def test_fixtures_are_non_square_where_it_matters():
    """Hazard 7 is invisible in a square fixture.

    ``perm >= obs`` with a bare broadcast aligns the observed vector
    against the permutation axis instead of the gene axis, and NumPy only
    *raises* when the two differ. A fixture at 6 genes x 6 permutations is
    worse than useless: it is a test that passes on broken code. The same
    argument applies to genes versus samples.
    """
    offenders = []
    for name in PARITY_SCENARIOS:
        fx = load_fixture(name)
        g, n, B = int(fx["shapes"]["g"]), int(fx["shapes"]["n"]), int(fx["shapes"]["B"])
        if g == n or (B > 0 and g == B) or (B > 0 and n == B):
            offenders.append((name, g, n, B))
    assert not offenders, f"square fixtures hide the broadcast hazard: {offenders}"


def test_suite_covers_both_group_size_parities():
    """Hazard 1 needs an even *and* an odd larger group.

    The odd-length example in ``docs/implementation-notes.md`` cannot separate
    quantile types 1, 2 and 7 at all — it agrees across all three. A suite
    built only on odd lengths would pass with type 1 substituted.
    """
    parities = set()
    for name in PARITY_SCENARIOS:
        fx = load_fixture(name)
        n1, n0 = int(fx["shapes"]["n1"]), int(fx["shapes"]["n0"])
        parities.add(max(n1, n0) % 2)
        parities.add(min(n1, n0) % 2)
    assert parities == {0, 1}, "fixtures must include both even and odd group sizes"


def test_suite_covers_both_normalizer_forms():
    forms = {load_fixture(n)["normalizer"]["form"] for n in PARITY_SCENARIOS}
    assert forms == {"vector", "matrix"}


def test_suite_covers_ties_and_zeros():
    """Quantile conventions and the GPD's strict exceedance test diverge on ties."""
    has_zeros = has_ties = False
    for name in PARITY_SCENARIOS:
        counts = load_fixture(name)["counts"]
        if np.any(counts == 0):
            has_zeros = True
        for row in counts:
            if np.unique(row).size < row.size:
                has_ties = True
                break
    assert has_zeros and has_ties


def test_suite_covers_the_refinement_gate_on_both_sides():
    """Below B = 500 no gene is ever refined; above it, some must be."""
    below = [n for n in PARITY_SCENARIOS
             if 0 < int(load_fixture(n)["shapes"]["B"]) < 500]
    above = [n for n in PARITY_SCENARIOS
             if int(load_fixture(n)["shapes"]["B"]) >= 500]
    assert below and above

    for name in below:
        fx = load_fixture(name)
        assert fx["pvalues"]["refined_diff_i0"].size == 0
        assert fx["pvalues"]["refined_tail_i0"].size == 0

    refined_somewhere = any(
        load_fixture(n)["pvalues"]["refined_diff_i0"].size
        + load_fixture(n)["pvalues"]["refined_tail_i0"].size
        for n in above
    )
    assert refined_somewhere, "no fixture actually exercises the GPD refinement"


@pytest.mark.parametrize("name", ALL_SCENARIOS)
def test_matrix_orientation_is_carried_by_the_file(name):
    """Hazard 8: a 2-D array must carry its own shape, never a flat vector.

    Read the raw JSON rather than the parsed arrays — the point is that
    the *file* is unambiguous about orientation, not that the loader
    guessed right.
    """
    with (FIXTURE_DIR / f"{name}.json").open(encoding="utf-8") as fh:
        raw = json.load(fh)
    for key in ("counts", "jitter", "tpm", "Q1", "Q0", "D"):
        node = raw[key]
        assert "shape" in node, f"{key} has no declared shape"
        rows, cols = node["shape"]
        assert len(node["dec"]) == rows
        assert all(len(r) == cols for r in node["dec"])
