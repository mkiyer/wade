"""The plotting layer — the data behind each figure, and the two renderers.

Most of what a figure *says* lives in a small pure-NumPy dataclass
(``gene_panels``, ``volcano_data``, ``stages_data``), so most of what is
asserted here needs no plotting library at all: the BH cutoff line is where BH
rejects, the quadrants are the README's table, the x axis of the volcano is
the fold change, the panel's characterization is the result's. The renderers
are then smoke-tested for each backend that is installed, and one test pins
the contract that neither library is imported by ``import wade``.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import wade
from wade import plotting as P
from wade.plotting import data as PD, theme as PT

# ---------------------------------------------------------------------------
# A small, fast result with every kind of gene in it. Signal fraction is kept
# low (10 of 160) so library-size normalization does not couple the genes.

G, N1, N0, B = 160, 40, 40, 300
COND = np.r_[np.ones(N1, int), np.zeros(N0, int)]


@pytest.fixture(scope="module")
def res():
    rng = np.random.default_rng(11)
    counts = rng.poisson(60, size=(G, N1 + N0)).astype(float)
    counts[:4, :N1] *= 3.0                              # global up
    counts[4:6, :N1] *= 0.3                             # global down
    for i in range(6, 10):                              # 10% subset up, x6
        counts[i, rng.choice(N1, 4, replace=False)] *= 6.0
    names = [f"g{i}" for i in range(G)]
    return wade.wade(counts, np.ones(G), COND, nperms=B, gene_names=names, seed=3)


@pytest.fixture(scope="module")
def res_nosubset():
    rng = np.random.default_rng(12)
    counts = rng.poisson(30, size=(40, N1 + N0)).astype(float)
    return wade.wade(counts, np.ones(40), COND, nperms=50, subset=False)


@pytest.fixture(scope="module")
def res_greater():
    rng = np.random.default_rng(13)
    counts = rng.poisson(30, size=(60, N1 + N0)).astype(float)
    counts[:3, :N1] *= 3.0
    counts[3:6, :N1] *= 0.3
    return wade.wade(counts, np.ones(60), COND, nperms=200, alternative="greater")


def _backends():
    return P.available_backends()


# ---------------------------------------------------------------------------
# The optional-dependency contract


def test_importing_wade_imports_neither_plotting_library():
    """The statistic keeps its one-dependency surface.

    Both plotting libraries are imported inside the plotting functions, so
    ``import wade`` — and therefore every caller that only wants the test —
    must not pull either of them in. Checked in a fresh interpreter because
    this process has long since imported them.
    """
    code = (
        "import sys, wade; "
        "bad = [m for m in ('plotly', 'matplotlib') if m in sys.modules]; "
        "sys.exit(1 if bad else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, (
        "import wade pulled in a plotting library: " + proc.stdout + proc.stderr
    )


def test_backend_resolution_is_explicit_and_validated(monkeypatch):
    with pytest.raises(ValueError, match="backend must be one of"):
        P._resolve_backend("bokeh")
    monkeypatch.setattr(P, "available_backends", lambda: ())
    with pytest.raises(ImportError, match="no plotting backend is installed"):
        P._resolve_backend("auto")
    monkeypatch.setattr(P, "available_backends", lambda: ("matplotlib",))
    monkeypatch.setattr(P, "DEFAULT_BACKEND", "auto")
    if "matplotlib" in _backends():
        assert P._resolve_backend(None) == "matplotlib"


# ---------------------------------------------------------------------------
# gene_panels: the single-gene data


def test_gene_panels_from_a_result_carries_the_reported_statistics(res):
    [p] = P.gene_panels(res, gene="g6")
    i = res.gene_index("g6")
    assert p.name == "g6"
    assert p.stats["p_mean_shift"] == res.p_mean_shift[i]
    assert p.stats["p_subset"] == res.p_subset[i]
    # The panel's characterization is computed from the same curve the test
    # used (reversed for display), so it agrees with the result to rounding.
    assert np.isclose(p.affected_fraction, res.affected_fraction[i], rtol=1e-12)
    assert np.isclose(p.direction, res.direction[i], rtol=1e-12)
    assert p.median_r == np.median(p.detail.r)
    assert len(p.subtitle_lines) == 2 and "p subset" in p.subtitle_lines[0]


def test_gene_panels_accepts_every_documented_input_form(res):
    pc = res.pseudocount
    d = wade.wade_gene(res.tpm[6], COND, pseudocount=pc[6])
    by_detail = P.gene_panels(d, names="x")
    by_row = P.gene_panels(res.tpm[6], COND, names=["x"], pseudocount=pc[6])
    by_block = P.gene_panels(res.tpm[[6, 7]], COND, names=["x", "y"], pseudocount=pc[6])
    by_list = P.gene_panels([d, d])
    by_index = P.gene_panels(res, gene=[6, "g7"])
    assert [p.name for p in by_detail] == ["x"]
    assert [p.name for p in by_row] == ["x"]
    assert [p.name for p in by_block] == ["x", "y"]
    assert [p.name for p in by_list] == ["gene 0", "gene 1"]
    assert [p.name for p in by_index] == ["g6", "g7"]
    np.testing.assert_array_equal(by_detail[0].detail.r, by_row[0].detail.r)
    np.testing.assert_array_equal(by_block[0].detail.r, by_index[0].detail.r)
    # Rows without a condition vector cannot be turned into curves.
    with pytest.raises(ValueError, match="cond is required"):
        P.gene_panels(res.tpm[6])
    with pytest.raises(ValueError, match="pass gene="):
        P.gene_panels(res)
    with pytest.raises(KeyError, match="no gene named"):
        P.gene_panels(res, gene="nope")


def test_a_global_gene_reads_flat_and_a_subset_gene_reads_zero_then_climbing(res):
    """The property the figure exists to show, asserted on the data it draws."""
    glob = P.gene_panels(res, gene="g0")[0]
    sub = P.gene_panels(res, gene="g6")[0]
    # Global: the curve hugs its median everywhere; the median is the fold change.
    assert np.median(np.abs(glob.detail.r - glob.median_r)) < 0.25
    assert abs(glob.median_r - np.log2(3.0)) < 0.3
    assert glob.affected_fraction > 0.8
    # Subset: the median sits near zero; the top decile is far above it.
    assert abs(sub.median_r) < 0.2
    top = sub.detail.r[sub.detail.p > 0.92]
    assert top.mean() > 1.0
    assert sub.affected_fraction < 0.3


# ---------------------------------------------------------------------------
# The BH cutoff line


def test_bh_cutoff_is_where_bh_rejects():
    rng = np.random.default_rng(0)
    p = np.r_[rng.uniform(0, 1, 300), rng.uniform(0, 1e-4, 12)]
    padj = wade.bh_adjust(p)
    cut = PD._bh_cutoff(p, padj, 0.05)
    # Every gene at or below the cutoff is rejected, and every rejected gene is at or below it.
    np.testing.assert_array_equal(p <= cut, padj <= 0.05)


def test_bh_cutoff_when_nothing_passes_is_alpha_over_G():
    p = np.linspace(0.2, 0.9, 50)
    padj = wade.bh_adjust(p)
    assert not (padj <= 0.05).any()
    assert PD._bh_cutoff(p, padj, 0.05) == pytest.approx(0.05 / 50)


def test_bh_cutoff_ignores_nan_and_is_nan_when_everything_is():
    p = np.array([np.nan, 0.001, 0.5, np.nan])
    padj = wade.bh_adjust(p)
    assert np.isfinite(PD._bh_cutoff(p, padj, 0.05))
    assert np.isnan(PD._bh_cutoff(np.full(3, np.nan), np.full(3, np.nan), 0.05))


# ---------------------------------------------------------------------------
# volcano_data


def test_volcano_x_axis_is_the_log2_fold_change_and_y_is_neglog10_p(res):
    v = P.volcano_data(res, "mean_shift")
    np.testing.assert_array_equal(v.x, res.log2_fc)
    np.testing.assert_allclose(v.y, -np.log10(res.p_mean_shift))
    np.testing.assert_array_equal(v.significant, res.padj_mean_shift <= 0.05)
    assert v.n_significant == int((res.padj_mean_shift <= 0.05).sum())
    assert v.alternative == "two-sided"
    assert "log" in v.xlabel and "fold change" in v.xlabel
    # The line and the significant set agree, by construction of the cutoff.
    np.testing.assert_array_equal(v.p <= v.cutoff, v.significant)


def test_volcano_subset_stage_uses_the_subset_pvalues_and_refuses_without_them(res, res_nosubset):
    v = P.volcano_data(res, "subset")
    np.testing.assert_allclose(v.y, -np.log10(res.p_subset))
    with pytest.raises(ValueError, match="no subset stage"):
        P.volcano_data(res_nosubset, "subset")
    with pytest.raises(ValueError, match="stage must be"):
        P.volcano_data(res, "both")


def test_volcano_colour_options(res, res_nosubset):
    v = P.volcano_data(res, color="affected_fraction")
    np.testing.assert_array_equal(v.color, res.affected_fraction)
    assert v.color_spec["range"] == (0.0, 1.0)
    v = P.volcano_data(res, color="direction")
    np.testing.assert_array_equal(v.color, res.direction)
    assert v.color_spec["range"] == (-1.0, 1.0)
    assert P.volcano_data(res, color=None).color is None
    # Without the characterization there is nothing to colour by: single hue, no error.
    assert P.volcano_data(res_nosubset, color="affected_fraction").color is None
    # Any other column works too, autoscaled, and picks its scale by whether
    # it spans zero — so a new statistic is colourable the day it exists.
    v = P.volcano_data(res, color="w1")
    np.testing.assert_array_equal(v.color, res.w1)
    assert v.color_spec["range"] is None and v.color_spec["scale"] == PT._SEQUENTIAL
    assert P.volcano_data(res, color="log2_fc").color_spec["scale"] == PT._DIVERGING
    # ... and a supplied per-gene array, for anything WADE cannot know about
    own = np.arange(float(G))
    np.testing.assert_array_equal(P.volcano_data(res, color=own).color, own)
    with pytest.raises(ValueError, match="one value per gene"):
        P.volcano_data(res, color=own[:-1])
    with pytest.raises(ValueError, match="not a column"):
        P.volcano_data(res, color="nope")


def test_volcano_labels_are_selective(res):
    """Top-n by y, **broken by |x|** — because on a saturated cohort thousands
    of genes share the floor and ranking by y alone names an arbitrary few."""
    v = P.volcano_data(res, label=5)
    assert v.labelled.sum() == 5
    lab, unl = v.labelled, ~v.labelled
    assert v.y[lab].min() >= v.y[unl].max() - 1e-12          # y-optimal up to ties
    cut = v.y[lab].min()
    tied_lab = np.abs(v.x[lab & (v.y == cut)])
    tied_unl = np.abs(v.x[unl & (v.y == cut)])
    if tied_lab.size and tied_unl.size:                       # ties broken by effect
        assert tied_lab.min() >= tied_unl.max()
    v = P.volcano_data(res, label=["g0", "g6", "absent"])
    assert set(res.gene[v.labelled]) == {"g0", "g6"}
    assert P.volcano_data(res).labelled.sum() == 0
    assert P.volcano_data(res, label=0).labelled.sum() == 0


def test_volcano_table_is_the_figures_twin(res):
    v = P.volcano_data(res, "mean_shift")
    t = v.table()
    assert set(t) >= {"gene", "log2_fc", "p_mean_shift", "padj_mean_shift", "significant", "affected_fraction"}
    assert all(len(col) == G for col in t.values())


def test_volcano_reads_the_alternative_from_the_result(res_greater):
    v = P.volcano_data(res_greater)
    assert v.alternative == "greater"
    # Under "greater" a strongly reduced gene is untested: its p is ~1.
    down = np.flatnonzero(res_greater.log2_fc < -1)
    assert down.size > 0 and np.all(v.p[down] > 0.5)


# ---------------------------------------------------------------------------
# stages_data


def test_stages_quadrants_are_the_readme_table(res):
    s = P.stages_data(res, alpha=0.05)
    np.testing.assert_allclose(s.x, -np.log10(res.p_mean_shift))
    np.testing.assert_allclose(s.y, -np.log10(res.p_subset))
    np.testing.assert_array_equal(s.sig_mean, res.padj_mean_shift <= 0.05)
    np.testing.assert_array_equal(s.sig_subset, res.padj_subset <= 0.05)
    q = s.quadrant()
    assert set(q) <= {v.replace("\n", " ") for v in P.QUADRANTS.values()}
    np.testing.assert_array_equal(q == "global shift", s.sig_mean & ~s.sig_subset)
    np.testing.assert_array_equal(q == "not differential", ~s.sig_mean & ~s.sig_subset)
    assert sum(s.counts().values()) == G
    # The quadrant lines are the BH cutoffs, so the halves agree with the masks.
    np.testing.assert_array_equal(10.0 ** -s.x <= s.cutoff_mean, s.sig_mean)
    np.testing.assert_array_equal(10.0 ** -s.y <= s.cutoff_subset, s.sig_subset)
    # The planted global genes land in the global-shift quadrant.
    assert (q[:4] == "global shift").all()


def test_stages_refuses_a_result_without_the_subset_stage(res_nosubset):
    with pytest.raises(ValueError, match="no subset stage"):
        P.stages_data(res_nosubset)


def test_stages_table_and_labels(res):
    s = P.stages_data(res, label=3)
    assert s.labelled.sum() == 3
    t = s.table()
    assert set(t) >= {"gene", "p_mean_shift", "p_subset", "quadrant", "affected_fraction"}
    np.testing.assert_allclose(t["p_mean_shift"], res.p_mean_shift)


# ---------------------------------------------------------------------------
# .table(): the one contract all three dataclasses share


def test_every_dataclass_has_a_table_of_the_results_own_numbers(res):
    """The chart's table-view twin, on all three.

    This is the seam ``docs/plotting.md`` promises: the exact numbers a figure
    draws, reachable without reading them off the pixels, and what a third
    renderer or an export to ggplot is written against. Columns must be equal
    length and must be the result's own values, not a re-derivation.
    """
    i = res.gene_index("g6")
    [panel] = P.gene_panels(res, gene="g6")
    panel_t, volcano_t, stages_t = (panel.table(), P.volcano_data(res).table(),
                                    P.stages_data(res).table())
    for t in (panel_t, volcano_t, stages_t):
        assert "gene" in t
        assert len({len(col) for col in t.values()}) == 1, "ragged table"

    # The panel's rows are grid nodes, and its curves are the result's own.
    d = res.gene_detail(i)
    assert len(panel_t["p"]) == res.nprobs
    assert set(panel_t["gene"]) == {"g6"}
    for name, arr in (("p", d.p), ("r", d.r), ("y1", d.y1), ("y0", d.y0),
                      ("cumulative_area", d.cumulative_area)):
        np.testing.assert_array_equal(panel_t[name], arr)
    # ... including stage 1's statistic, which is the last node of the area.
    assert panel_t["cumulative_area"][-1] == pytest.approx(res.mean_shift[i], rel=1e-12)
    # The scalars stay attributes, as the volcano's cutoff does.
    assert panel.reference == pytest.approx(np.log2(res.fitted_fold_change[i]))

    # The clouds' rows are genes, in the result's order.
    for t in (volcano_t, stages_t):
        np.testing.assert_array_equal(t["gene"], res.gene)
    np.testing.assert_array_equal(volcano_t["log2_fc"], res.log2_fc)
    np.testing.assert_array_equal(volcano_t["p_mean_shift"], res.p_mean_shift)
    np.testing.assert_allclose(stages_t["p_subset"], res.p_subset)


# ---------------------------------------------------------------------------
# The renderers, per installed backend


@pytest.fixture(autouse=True)
def _agg():
    """Matplotlib must never try to open a window from the test suite."""
    try:
        import matplotlib
    except ImportError:
        yield
        return
    matplotlib.use("Agg")
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


def _n_axes(fig, backend):
    """Data panels only: a matplotlib colour bar is an Axes too."""
    if backend == "matplotlib":
        return sum(1 for ax in fig.axes if ax.get_label() != "<colorbar>")
    return sum(1 for k in fig.layout if k.startswith("xaxis"))


@pytest.mark.parametrize("backend", _backends())
def test_plot_gene_renders_one_column_per_gene(res, backend):
    fig = wade.plot_gene(res, gene=["g0", "g6", "g100"], backend=backend)
    if backend == "plotly":
        import plotly.graph_objects as go
        assert isinstance(fig, go.Figure)
        assert _n_axes(fig, backend) == 6
        lines = [t for t in fig.data if t.mode == "lines" and t.x is not None and len(t.x) > 1]
        assert len(lines) == 9                       # three traces per gene
        # The subtitle carries the gene's statistics.
        assert any("p subset" in a.text for a in fig.layout.annotations)
    else:
        import matplotlib.figure
        assert isinstance(fig, matplotlib.figure.Figure)
        assert _n_axes(fig, backend) == 6
        assert fig.axes[3].get_yscale() == "log"
    # A single gene, from a bare row, also renders.
    fig1 = wade.plot_gene(res.tpm[6], COND, names="g6", backend=backend)
    assert _n_axes(fig1, backend) == 2


@pytest.mark.parametrize("backend", _backends())
def test_plot_volcano_renders_both_stages_with_a_cutoff_and_labels(res, backend):
    fig = wade.plot_volcano(res, stage="both", label=4, backend=backend)
    assert _n_axes(fig, backend) == 2
    if backend == "plotly":
        shapes = [s for s in fig.layout.shapes if s.type == "line"]
        assert len(shapes) == 4                      # x = 0 and the FDR line, per panel
        texts = [t for t in fig.data if t.mode == "text"]
        assert sum(len(t.text) for t in texts) == 8  # four labels per panel
        assert any("FDR" in a.text for a in fig.layout.annotations)
    else:
        ax = fig.axes[0]
        assert sum(1 for l in ax.get_lines() if l.get_linestyle() == "--") == 1
        assert len(ax.texts) >= 4
    # One stage, with the diverging colour and a single-hue fallback.
    wade.plot_volcano(res, "subset", color="direction", backend=backend)
    wade.plot_volcano(res, color=None, backend=backend)


@pytest.mark.parametrize("backend", _backends())
def test_plot_volcano_shades_the_untested_half_under_a_one_sided_alternative(res_greater, backend):
    fig = wade.plot_volcano(res_greater, backend=backend)
    if backend == "plotly":
        rects = [s for s in fig.layout.shapes if s.type == "rect"]
        assert len(rects) == 1 and rects[0].x1 == 0.0 and rects[0].x0 < 0
        assert any("untested" in a.text for a in fig.layout.annotations)
    else:
        spans = [p for p in fig.axes[0].patches if p.__class__.__name__ == "Rectangle"]
        assert spans, "expected an axvspan over the untested half"
        assert any("untested" in t.get_text() for t in fig.axes[0].texts)


@pytest.mark.parametrize("backend", _backends())
def test_plot_stages_draws_the_four_quadrants(res, backend):
    fig = wade.plot_stages(res, label=2, backend=backend)
    if backend == "plotly":
        lines = [s for s in fig.layout.shapes if s.type == "line"]
        assert len(lines) == 2
        corner = [a for a in fig.layout.annotations if "genes" in a.text]
        assert len(corner) == 4
        assert any("global shift" in a.text for a in corner)
    else:
        ax = fig.axes[0]
        assert sum(1 for l in ax.get_lines() if l.get_linestyle() == "--") == 2
        corner = [t for t in ax.texts if "genes" in t.get_text()]
        assert len(corner) == 4


@pytest.mark.parametrize("backend", _backends())
def test_plot_functions_refuse_what_the_data_layer_refuses(res_nosubset, backend):
    with pytest.raises(ValueError, match="no subset stage"):
        wade.plot_stages(res_nosubset, backend=backend)
    with pytest.raises(ValueError, match="no subset stage"):
        wade.plot_volcano(res_nosubset, stage="subset", backend=backend)
    # But the mean-shift volcano of such a result is fine, uncoloured.
    fig = wade.plot_volcano(res_nosubset, backend=backend)
    assert _n_axes(fig, backend) == 1


def test_a_requested_backend_that_is_not_installed_says_so(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name.split(".")[0] == "plotly":
            raise ImportError("no plotly here")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ImportError, match="is not installed"):
        P._resolve_backend("plotly")


def test_wade_result_carries_cond_and_resolves_genes(res):
    np.testing.assert_array_equal(res.cond, COND)
    assert res.gene_index("g5") == 5 and res.gene_index(-1) == G - 1
    with pytest.raises(IndexError):
        res.gene_index(G)
    d = res.gene_detail("g5")
    assert d.nprobs == res.nprobs
    np.testing.assert_array_equal(d.r, wade.wade_gene(res.tpm[5], COND, pseudocount=res.pseudocount[5]).r)
    # The result's curve is the pseudocounted one the statistics were read from.
    assert not np.array_equal(d.r, wade.wade_gene(res.tpm[5], COND).r)


def test_volcano_axes_can_be_any_column(res):
    """The README tells users to rank by magnitude and by z once p-values
    saturate; those columns must therefore be plottable."""
    v = P.volcano_data(res, "subset", x="subset_log2_fc", y="z_subset")
    np.testing.assert_array_equal(v.x, res.subset_log2_fc)
    np.testing.assert_array_equal(v.y, res.z_subset)
    assert v.x_name == "subset_log2_fc" and v.y_name == "z_subset"
    assert "subset log₂ fold change" in v.xlabel
    assert v.ylabel == PT._AXIS_LABELS["z_subset"]
    # off the p axis the BH line has no meaning and is not drawn
    assert np.isnan(v.cutoff_y)
    # the table names what it actually holds
    t = v.table()
    assert "subset_log2_fc" in t and "z_subset" in t
    # the default is unchanged
    d = P.volcano_data(res)
    assert d.x_name == "log2_fc" and d.y_name is None
    assert np.isfinite(d.cutoff_y) or np.isnan(d.cutoff)
    assert "log₂ fold change" in d.xlabel and "−log₁₀ p" in d.ylabel
    with pytest.raises(ValueError, match="not a column"):
        P.volcano_data(res, x="nope")


def test_volcano_hover_carries_every_column(res):
    v = P.volcano_data(res)
    assert set(v.hover) == {k for k, val in res.columns().items() if val is not None}
    for name in ("subset_log2_fc", "z_subset", "z_mean_shift", "w1"):
        assert name in v.hover, name


@pytest.mark.parametrize("backend", P.available_backends())
def test_both_backends_render_a_custom_axis_volcano(res, backend):
    fig = P.plot_volcano(res, stage="subset", x="subset_log2_fc", y="z_subset",
                         backend=backend, label=3)
    assert fig is not None
