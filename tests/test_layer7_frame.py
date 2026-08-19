"""Layer 7 — BH-FDR and the result frame.

Adjusted p-values at a looser tolerance than the rest of the suite
(hazard 3: the available implementations differ from R by at most one
unit in the last place, which cannot change a thresholding decision at
any realistic q), plus the missing-value handling, the frame's shape, and
the ``gene_names = None`` behaviour — where the port deliberately differs.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import (
    PARITY_SCENARIOS,
    TOL_BH,
    assert_close,
    load_fixture,
)
from portrun import run_port

import wade

LAYER = "layer 7 BH and frame"


@pytest.mark.parity
@pytest.mark.parametrize("name", PARITY_SCENARIOS)
def test_adjusted_pvalues(name):
    fx, res = run_port(name)
    assert_close(res.padj_mean_shift, fx["frame"]["padj.diff"], TOL_BH,
                 f"{name}: padj.diff", LAYER)


@pytest.mark.parity
@pytest.mark.parametrize("idx", range(5))
def test_bh_against_R_directly(idx):
    """Including the two behaviours that are not about the arithmetic."""
    fx = load_fixture("bh_padjust")
    case = fx["cases"][idx]
    got = wade.bh_adjust(case["p"])
    assert_close(got, case["padj"], TOL_BH, f"BH {case['label']}", LAYER)


@pytest.mark.parity
def test_bh_drops_missing_values_rather_than_propagating_them():
    """R adjusts using the reduced count and puts the NA back in position.

    This matters because ``wade()`` produces all-missing p-value columns
    when ``nperms == 0``. An implementation that propagated the missing
    value across the whole vector, or counted it in ``n``, would disagree
    on any frame containing one — and it would disagree by a factor of
    ``n / n_nonmissing``, which looks like a plausible number.
    """
    p = np.array([0.9, 0.01, np.nan, 0.5, 0.01, 1.0])
    got = wade.bh_adjust(p)
    want = np.array([1.0, 0.025, np.nan, 0.8333333333333334, 0.025, 1.0])
    assert np.isnan(got[2])
    ok = ~np.isnan(want)
    assert np.allclose(got[ok], want[ok], rtol=1e-12)

    # n = 5, not 6. With n = 6 the tied 0.01s would adjust to 0.03.
    assert got[1] == pytest.approx(0.025)


@pytest.mark.parity
def test_bh_of_an_all_missing_vector_is_all_missing():
    got = wade.bh_adjust(np.full(5, np.nan))
    assert np.all(np.isnan(got))


def test_bh_enforces_monotonicity_and_shares_values_across_ties():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042])
    got = wade.bh_adjust(p)
    assert np.all(np.diff(got) >= -1e-15), "adjusted values must be non-decreasing in p"
    tied = wade.bh_adjust(np.array([0.02, 0.02, 0.5]))
    assert tied[0] == tied[1]


@pytest.mark.parity
def test_each_stage_gets_its_own_FDR_family():
    """The mean-shift and subset stages are adjusted separately, not pooled.

    Asserted by construction: pooling would use n = 2G and produce
    systematically larger adjusted values.
    """
    _, res = run_port("main")
    # Each stage's adjustment uses only its own p-values.
    assert np.allclose(res.padj_mean_shift, wade.bh_adjust(res.p_mean_shift),
                       rtol=0, atol=0, equal_nan=True)

    # Pooling genuinely differs: two stages with different p-values adjusted
    # together use n = 2G, which is systematically more conservative.
    rng = np.random.default_rng(0)
    a = rng.uniform(size=40) ** 3
    b = rng.uniform(size=40)
    separate = wade.bh_adjust(a)
    pooled = wade.bh_adjust(np.concatenate([a, b]))[:40]
    assert np.all(pooled >= separate - 1e-12)
    assert not np.allclose(pooled, separate)


# ---------------------------------------------------------------------
# The frame
# ---------------------------------------------------------------------

@pytest.mark.parametrize("name", PARITY_SCENARIOS)
def test_frame_shape_and_columns(name):
    fx, res = run_port(name)
    g = int(fx["shapes"]["g"])
    cols = res.columns()
    for key, arr in cols.items():
        assert len(arr) == g, f"{key} has length {len(arr)}, expected {g}"


@pytest.mark.parity
def test_nperms_zero_returns_effect_sizes_with_missing_pvalues():
    """A supported mode, not an error."""
    fx, res = run_port("nperms0")
    assert np.all(np.isnan(res.p_mean_shift))
    assert np.all(np.isnan(res.padj_mean_shift))
    assert np.all(np.isfinite(res.mean_shift))
    assert np.all(np.isfinite(res.w1))


@pytest.mark.parity
def test_gene_names_none_synthesizes_identifiers_instead_of_dropping_the_column():
    """A DELIBERATE divergence from the reference.

    R defaults ``gene_names`` to ``rownames(counts)``, which is ``NULL``
    for an unnamed matrix, and ``tibble()`` drops a ``NULL`` column rather
    than erroring. Measured in the fixture: the R frame has 13 columns and
    **no gene column at all** — the identifier silently disappears while
    every other column is present and correct, so anything downstream that
    joins on it breaks somewhere else entirely.

    The port synthesizes positional identifiers. This is asserted rather
    than discovered.
    """
    fx, res = run_port("nonames")
    assert fx["gene_names"] is None
    assert fx["frame"]["has_gene_col"] is False
    assert len(fx["frame"]["columns"]) == 13

    assert res.gene is not None
    assert list(res.gene[:3]) == ["gene0", "gene1", "gene2"]
    assert "gene" in res.columns()


@pytest.mark.parity
@pytest.mark.parametrize("name", ["tiny", "main"])
def test_named_genes_are_carried_through(name):
    fx, res = run_port(name)
    assert list(res.gene) == list(fx["gene_names"])


def test_result_reports_the_design_without_running_anything():
    _, res = run_port("main")
    assert res.nprobs == 12
    assert res.params["n1"] == 15 and res.params["n0"] == 12
