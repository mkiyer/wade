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


def _counts():
    rng = np.random.default_rng(11)
    counts = rng.poisson(60, size=(G, N1 + N0)).astype(float)
    counts[:4, :N1] *= 3.0                              # global up
    counts[4:6, :N1] *= 0.3                             # global down
    for i in range(6, 10):                              # 10% subset up, x6
        counts[i, rng.choice(N1, 4, replace=False)] *= 6.0
    return counts


@pytest.fixture(scope="module")
def res():
    return wade.wade(_counts(), np.ones(G), COND, nperms=B, seed=3,
                     gene_names=[f"g{i}" for i in range(G)])


@pytest.fixture(scope="module")
def res_boot():
    """The same result with bootstrap intervals, for the figures that draw them.

    Kept separate from ``res`` so every other test still runs the ``n_boot=0``
    path, where a figure must draw no interval at all.
    """
    return wade.wade(_counts(), np.ones(G), COND, nperms=B, seed=3, n_boot=200,
                     gene_names=[f"g{i}" for i in range(G)])


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


GD, GN = 300, 30
COND_D = np.r_[np.ones(GN, int), np.zeros(GN, int)]
BAD = np.array([1, 4, 9])              # low-complexity libraries
ARTEFACT, GENUINE = 3, 200


@pytest.fixture(scope="module")
def res_meta():
    """A result carrying gene symbols and biotypes, as an input frame would."""
    pl = pytest.importorskip("polars")
    counts = _counts()
    samples = [f"S{j}" for j in range(N1 + N0)]
    frame = pl.DataFrame({
        "gene_id": [f"ENSG{i:05d}" for i in range(G)],
        "gene_name": [f"SYM{i}" for i in range(G)],
        "gene_type": ["protein_coding" if i % 3 else "lncRNA" for i in range(G)],
        **{s: counts[:, j] for j, s in enumerate(samples)},
    })
    return wade.wade(wade.as_counts(frame, sample_columns=samples), 1.0, COND,
                     nperms=B, seed=3)


@pytest.fixture(scope="module")
def res_drivers():
    """``docs/scaling.md`` §7.2 in miniature, and the two genes are built to be
    the two real cases rather than merely different.

    ``BAD`` are three libraries that detect half as many genes as their peers
    and hold most of their mass in a handful — blood transcripts dominating a
    plasma prep. ``ARTEFACT`` is one of that handful and is abundant, as FLI1
    was at ~297,000 junction counts against ~1,600 elsewhere. ``GENUINE`` is
    ``ETV4``: near zero across the cohort, elevated in nine ordinary libraries
    chosen disjoint from ``BAD``. Both of §7.2's discriminators — the gene's
    share of its drivers' libraries, and those libraries' complexity — should
    therefore separate them, which is what the figure has to show.
    """
    rng = np.random.default_rng(7)
    mu = 10 ** rng.uniform(0, 2.6, GD)
    mu[ARTEFACT] = 300.0
    mu[GENUINE] = 8.0
    counts = rng.poisson(mu[:, None] * np.ones(2 * GN)).astype(float)
    for j in BAD:
        # The dominant handful is exempt from the dropout: a blood-dominated
        # prep still detects its own dominant transcripts. Zeroing them instead
        # makes ARTEFACT a *downward* subset, which is a different gene.
        counts[rng.choice(np.arange(6, GD), int(0.55 * GD), replace=False), j] = 0.0
        counts[:6, j] *= 120.0
    ordinary = np.setdiff1d(np.arange(GN), BAD)
    counts[GENUINE, rng.choice(ordinary, 9, replace=False)] *= 9.0
    names = [f"g{i}" for i in range(GD)]
    names[ARTEFACT], names[GENUINE] = "ARTEFACT", "GENUINE"
    res = wade.wade(counts, np.ones(GD), COND_D, nperms=200, seed=3, gene_names=names,
                    sample_names=[f"lib{j:02d}" for j in range(2 * GN)])
    return res, counts


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
    assert v.color_spec["range"] == (0.0, 1.0) and v.color_spec["role"] == "sequential"
    v = P.volcano_data(res, color="direction")
    np.testing.assert_array_equal(v.color, res.direction)
    assert v.color_spec["range"] == (-1.0, 1.0) and v.color_spec["role"] == "diverging"
    assert P.volcano_data(res, color=None).color is None
    # Without the characterization there is nothing to colour by: single hue, no error.
    assert P.volcano_data(res_nosubset, color="affected_fraction").color is None
    # Any other column works too, autoscaled, and picks its role by whether it
    # spans zero — so a new statistic is colourable the day it exists. The role
    # is as far as the data layer goes; the theme resolves it to a colormap.
    v = P.volcano_data(res, color="w1")
    np.testing.assert_array_equal(v.color, res.w1)
    assert v.color_spec["range"] is None and v.color_spec["role"] == "sequential"
    assert P.volcano_data(res, color="log2_fc").color_spec["role"] == "diverging"
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


# ---------------------------------------------------------------------------
# Themes


@pytest.mark.parametrize("backend", P.available_backends())
@pytest.mark.parametrize("name", sorted(P.THEMES))
def test_every_built_in_theme_renders_every_figure_on_every_backend(res, backend, name):
    """The whole point of a token set: three figures x two backends x three
    themes all draw, and none of them reaches for a colour of its own."""
    assert wade.plot_gene(res, gene=["g0", "g6"], backend=backend, theme=name) is not None
    assert wade.plot_volcano(res, stage="both", label=3, backend=backend, theme=name) is not None
    assert wade.plot_stages(res, label=2, backend=backend, theme=name) is not None
    # ... including uncoloured, which takes the single-hue path.
    assert wade.plot_volcano(res, color=None, backend=backend, theme=name) is not None


@pytest.mark.parametrize("backend", P.available_backends())
@pytest.mark.parametrize("name", sorted(P.THEMES))
def test_every_theme_renders_the_driver_figure(res_drivers, backend, name):
    res, counts = res_drivers
    assert wade.plot_drivers(res, "ARTEFACT", counts, backend=backend, theme=name) is not None


def test_a_theme_can_be_a_name_an_instance_or_the_session_default(res, monkeypatch):
    from dataclasses import replace

    assert P._resolve_theme(None) is P.THEMES["light"]
    assert P._resolve_theme("dark") is P.THEMES["dark"]
    monkeypatch.setattr(P, "DEFAULT_THEME", "high-contrast")
    assert P._resolve_theme(None) is P.THEMES["high-contrast"]
    # An instance passes through, so a house style is one replace() away.
    house = replace(P.THEMES["light"], case="#7b3fa0")
    assert P._resolve_theme(house) is house
    assert wade.plot_stages(res, theme=house, backend=P.available_backends()[0]) is not None
    with pytest.raises(ValueError, match="theme must be a Theme or one of"):
        P._resolve_theme("solarized")


def test_the_theme_resolves_colour_roles_and_the_label_box(res):
    light, dark = P.THEMES["light"], P.THEMES["dark"]
    assert light.scale("sequential") == "viridis"
    assert light.scale("diverging") == "RdBu_r"
    assert P.THEMES["high-contrast"].scale("sequential") == "cividis"
    with pytest.raises(ValueError, match="colour role must be"):
        light.scale("categorical")
    # The box behind in-plot text is the surface, so it follows the theme
    # instead of being a white slab on a dark figure.
    assert light.box() == "rgba(255,255,255,0.8)"
    assert dark.box(0.5) == "rgba(20,20,16,0.5)"
    # plotly's template has to follow too, or the ground stays light.
    assert light.plotly_template == "plotly_white" and dark.plotly_template == "plotly_dark"


# ---------------------------------------------------------------------------
# Bootstrap intervals, drawn


def test_an_axis_gets_an_interval_exactly_when_the_result_has_one(res, res_boot):
    """The rule is the column's name: an axis holding ``foo`` gets error bars
    iff the result carries ``ci_foo``. Nothing else is wired."""
    assert P.volcano_data(res).x_ci is None                       # n_boot = 0
    v = P.volcano_data(res_boot, label=4)
    np.testing.assert_array_equal(v.x_ci, res_boot.ci_log2_fc)
    assert v.y_ci is None                                         # y is −log₁₀ p
    v2 = P.volcano_data(res_boot, "subset", x="subset_log2_fc", y="affected_fraction")
    np.testing.assert_array_equal(v2.x_ci, res_boot.ci_subset_log2_fc)
    np.testing.assert_array_equal(v2.y_ci, res_boot.ci_affected_fraction)
    # A permutation z has no interval, so that axis gets none rather than erroring.
    assert P.volcano_data(res_boot, y="z_subset").y_ci is None
    # The table names them as the result does, so it stays a subset of it.
    t = v.table()
    np.testing.assert_array_equal(t["log2_fc_lo"], res_boot.ci_log2_fc[0])
    np.testing.assert_array_equal(t["log2_fc_hi"], res_boot.ci_log2_fc[1])
    assert "log2_fc_lo" not in P.volcano_data(res).table()


def test_the_gene_panel_band_is_where_the_affected_region_ends(res, res_boot):
    """``affected_fraction`` is a participation ratio, so it is an extent along
    the quantile axis — on the side ``direction`` points to."""
    assert P.gene_panels(res, gene="g6")[0].affected_span is None  # n_boot = 0

    [up] = P.gene_panels(res_boot, gene="g6")                      # a 10% subset, up
    lo, hi = up.stats["ci_affected_fraction"]
    assert up.direction > 0
    assert up.affected_span == pytest.approx((1.0 - hi, 1.0 - lo))
    assert up.affected_span[1] > 0.5, "an upward subset moved the top of the distribution"

    [down] = P.gene_panels(res_boot, gene="g4")                    # a global down gene
    assert down.direction < 0
    assert down.affected_span == pytest.approx(tuple(down.stats["ci_affected_fraction"]))
    # A wider interval is a wider band: that is the whole point of drawing it.
    assert up.affected_span[1] - up.affected_span[0] == pytest.approx(hi - lo)


@pytest.mark.parametrize("backend", P.available_backends())
def test_both_backends_draw_the_intervals_and_omit_them_at_n_boot_zero(res, res_boot, backend):
    """Not merely "it rendered": the bars and the band have to be in the figure
    when there are intervals, and **absent** when there are none — a figure must
    not imply a precision the run did not measure."""
    def drawn(r):
        v = wade.plot_volcano(r, label=4, backend=backend)
        g = wade.plot_gene(r, gene=["g6"], backend=backend)
        if backend == "plotly":
            bars = any(t.error_x is not None and t.error_x.array is not None for t in v.data)
            band = any(sh.type == "rect" for sh in g.layout.shapes)
        else:
            bars = any(type(c).__name__ == "LineCollection" for c in v.axes[0].collections)
            band = bool(g.axes[0].patches)
        return bars, band

    assert drawn(res_boot) == (True, True)
    assert drawn(res) == (False, False)


@pytest.mark.parametrize("backend", P.available_backends())
def test_a_drawn_error_bar_spans_exactly_the_reported_interval(res_boot, backend):
    """The endpoints, not merely the presence. A doubled arm or a swapped
    lo/hi renders perfectly plausibly and says the wrong thing."""
    i = res_boot.gene_index("g0")
    lo, hi = res_boot.ci_log2_fc[0, i], res_boot.ci_log2_fc[1, i]
    fig = wade.plot_volcano(res_boot, label=["g0"], backend=backend)
    if backend == "matplotlib":
        [lc] = [c for c in fig.axes[0].collections if type(c).__name__ == "LineCollection"]
        seg = lc.get_segments()[0]
        assert (seg[0][0], seg[-1][0]) == pytest.approx((lo, hi))
    else:
        t = next(t for t in fig.data if t.error_x is not None and t.error_x.array is not None)
        assert t.x[0] - t.error_x.arrayminus[0] == pytest.approx(lo)
        assert t.x[0] + t.error_x.array[0] == pytest.approx(hi)


# ---------------------------------------------------------------------------
# Per-gene metadata: displayed, never read


def test_metadata_names_the_points_and_joins_the_hover_and_the_table(res_meta):
    plain = P.volcano_data(res_meta, label=3)
    shown = P.volcano_data(res_meta, label=3, meta=["gene_name", "gene_type"])

    # Without meta= nothing changes: the id is still what a point is called.
    assert plain.meta == {}
    np.testing.assert_array_equal(plain.display, res_meta.gene)
    assert str(plain.display[0]).startswith("ENSG")

    # With it, the first column names the points and both join hover and table.
    np.testing.assert_array_equal(shown.display, res_meta.gene_meta["gene_name"])
    assert str(shown.display[0]) == "SYM0"
    np.testing.assert_array_equal(shown.gene, res_meta.gene)     # the id is kept
    assert set(shown.table()) >= {"gene", "gene_name", "gene_type"}
    np.testing.assert_array_equal(shown.table()["gene_type"], res_meta.gene_meta["gene_type"])
    # ... and every number is untouched.
    for name in ("x", "y", "p", "padj"):
        np.testing.assert_array_equal(getattr(shown, name), getattr(plain, name))

    # stages_data too, and plot_gene's panel titles.
    st = P.stages_data(res_meta, meta="gene_name")
    np.testing.assert_array_equal(st.display, res_meta.gene_meta["gene_name"])
    assert "gene_name" in st.table()
    [panel] = P.gene_panels(res_meta, gene="ENSG00006", meta="gene_name")
    assert panel.name == "SYM6"
    assert P.gene_panels(res_meta, gene="ENSG00006")[0].name == "ENSG00006"


def test_labels_match_what_the_points_are_called(res_meta):
    """Name what you see: with ``meta=`` the label list is symbols, not ids."""
    v = P.volcano_data(res_meta, label=["SYM0", "SYM6"], meta="gene_name")
    assert set(v.display[v.labelled]) == {"SYM0", "SYM6"}
    np.testing.assert_array_equal(v.gene[v.labelled],
                                  np.array(["ENSG00000", "ENSG00006"], dtype=object))
    # Without meta= the same list matches nothing, because that is not the name.
    assert P.volcano_data(res_meta, label=["SYM0"]).labelled.sum() == 0


def test_asking_for_metadata_a_result_does_not_carry_says_what_it_has(res, res_meta):
    with pytest.raises(ValueError, match="not in this result's gene metadata"):
        P.volcano_data(res_meta, meta="gene_symbol")
    # A result built from arrays carries none at all, and says so.
    with pytest.raises(ValueError, match="available: none"):
        P.volcano_data(res, meta="gene_name")


def test_metadata_is_display_only_and_moves_no_number(res_meta):
    """The Phase D invariant, restated at the display layer: metadata reaches
    the figures and reaches no statistic. ``tests/test_core_completion.py``
    pins the same thing on the way in."""
    bare = wade.wade(_counts(), np.ones(G), COND, nperms=B, seed=3,
                     gene_names=[f"ENSG{i:05d}" for i in range(G)])
    assert res_meta.gene_meta and not bare.gene_meta
    for name in ("mean_shift", "log2_fc", "p_mean_shift", "p_subset",
                 "affected_fraction", "direction", "subset_log2_fc"):
        np.testing.assert_array_equal(getattr(res_meta, name), getattr(bare, name),
                                      err_msg=f"metadata moved {name}")


@pytest.mark.parametrize("backend", P.available_backends())
def test_both_backends_label_by_the_metadata_column(res_meta, backend):
    fig = wade.plot_volcano(res_meta, label=2, meta=["gene_name", "gene_type"],
                            backend=backend)
    if backend == "matplotlib":
        texts = [a.get_text() for a in fig.axes[0].texts]
        assert any(t.startswith("SYM") for t in texts)
        assert not any(t.startswith("ENSG") for t in texts)
    else:
        labels = [str(t) for tr in fig.data if tr.mode == "text" for t in np.asarray(tr.text)]
        assert labels and all(str(t).startswith("SYM") for t in labels)
        # The hover carries the requested columns as text, ahead of the numbers.
        cloud = fig.data[0]
        assert "gene_name = %{customdata[0]}" in cloud.hovertemplate
        assert "gene_type = %{customdata[1]}" in cloud.hovertemplate
        assert str(cloud.text[0]) == "SYM0"
    assert wade.plot_gene(res_meta, gene=["ENSG00006"], meta="gene_name",
                          backend=backend) is not None
    assert wade.plot_stages(res_meta, label=2, meta="gene_name", backend=backend) is not None


# ---------------------------------------------------------------------------
# Label de-collision


def _overlapping_label_pairs(data):
    """How many pairs of drawn labels overlap, in the normalized units the
    placement reasons in. The figure's actual defect, counted."""
    import itertools

    from wade.plotting.data import _LABEL_H, _LABEL_W, _axis_span, _label_positions

    idx, dx, dy = _label_positions(data)
    x = np.asarray(data.x, dtype=np.float64)
    y = np.asarray(data.y, dtype=np.float64)
    x0, xw = _axis_span(x)
    y0, yw = _axis_span(y)
    u = (x[idx] + dx - x0) / xw
    v = (y[idx] + dy - y0) / yw
    return sum(1 for a, b in itertools.combinations(range(idx.size), 2)
               if abs(u[a] - u[b]) < _LABEL_W and abs(v[a] - v[b]) < _LABEL_H)


def test_labels_are_placed_without_colliding(res):
    """The alternate-above-and-below scheme this replaced left 24 overlapping
    pairs out of 8 labels on this data; the slot assignment leaves 0–1. Numbers
    and method are in ``docs/plotting.md``."""
    for d in (P.volcano_data(res, "mean_shift", label=8),
              P.volcano_data(res, "subset", label=8),
              P.stages_data(res, label=8)):
        assert _overlapping_label_pairs(d) <= 1

    # Displacement is bounded, because there are no leader lines to reconnect a
    # label that wandered: at most the furthest slot, on each axis.
    from wade.plotting.data import _LABEL_H, _LABEL_SLOTS, _LABEL_W, _axis_span, _label_positions
    d = P.volcano_data(res, "subset", label=12)
    idx, dx, dy = _label_positions(d)
    _, xw = _axis_span(np.asarray(d.x, dtype=np.float64))
    _, yw = _axis_span(np.asarray(d.y, dtype=np.float64))
    assert np.abs(dx).max() <= max(abs(sx) for sx, _ in _LABEL_SLOTS) * _LABEL_W * xw + 1e-12
    assert np.abs(dy).max() <= max(abs(sy) for _, sy in _LABEL_SLOTS) * _LABEL_H * yw + 1e-12


def test_label_placement_is_deterministic_and_prioritises_the_top_gene(res):
    from wade.plotting.data import _LABEL_H, _axis_span, _label_positions

    d = P.volcano_data(res, "subset", label=6)
    a, b = _label_positions(d), _label_positions(d)
    for first, second in zip(a, b):
        np.testing.assert_array_equal(first, second)
    # The most significant labelled gene keeps the best slot: straight above.
    idx, dx, dy = a
    k = int(np.argmax(np.nan_to_num(d.y[idx], nan=-np.inf)))
    _, yw = _axis_span(np.asarray(d.y, dtype=np.float64))
    assert dx[k] == pytest.approx(0.0)
    assert dy[k] == pytest.approx(_LABEL_H * yw)


@pytest.mark.parametrize("backend", P.available_backends())
def test_both_backends_place_labels_at_the_same_resolved_positions(res, backend):
    """The layer's promise is that a figure says the same thing in either
    backend; the placement is computed once, in the data layer, for that reason."""
    from wade.plotting.data import _label_positions

    d = P.volcano_data(res, "subset", label=5)
    idx, dx, dy = _label_positions(d)
    want = sorted(zip(np.round(d.x[idx] + dx, 9), np.round(d.y[idx] + dy, 9)))
    fig = wade.plot_volcano(res, "subset", label=5, backend=backend)
    if backend == "matplotlib":
        # The FDR annotation is an ax.text too; only the gene labels count.
        wanted = set(map(str, d.display[idx]))
        got = sorted((round(a.get_position()[0], 9), round(a.get_position()[1], 9))
                     for a in fig.axes[0].texts if a.get_text() in wanted)
    else:
        [tr] = [t for t in fig.data if t.mode == "text"]
        got = sorted(zip(np.round(np.asarray(tr.x, float), 9),
                         np.round(np.asarray(tr.y, float), 9)))
    assert got == want


# ---------------------------------------------------------------------------
# Stage 1's statistic, drawn


def test_the_cumulative_area_row_is_stage_1s_statistic_accumulating(res):
    """Its last point *is* ``mean_shift``, and where it gets there from is the
    reading: a global gene accumulates evenly, a subset gene almost entirely in
    its last few percent. That is the two stages agreeing on one gene."""
    glob = res.gene_detail(res.gene_index("g0"))          # planted global up
    sub = res.gene_detail(res.gene_index("g6"))           # planted 10% subset up
    for name, d in (("g0", glob), ("g6", sub)):
        i = res.gene_index(name)
        # The endpoint identity is a tolerance assertion, not an exact one: the
        # statistic sums D forwards and this accumulates it backwards
        # (wade.wade_gene's docstring).
        assert d.cumulative_area[-1] == pytest.approx(res.mean_shift[i], rel=1e-10)
    half = len(glob.p) // 2
    assert glob.cumulative_area[half] / glob.cumulative_area[-1] > 0.3   # evenly
    assert sub.cumulative_area[half] / sub.cumulative_area[-1] < 0.15    # at the end
    # It is already in the panel's table, from B2 — nothing new was carried.
    [panel] = P.gene_panels(res, gene="g6")
    np.testing.assert_array_equal(panel.table()["cumulative_area"], sub.cumulative_area)


@pytest.mark.parametrize("backend", P.available_backends())
def test_the_cumulative_area_row_is_off_by_default_and_adds_one_row(res, backend):
    genes = ["g0", "g6"]
    plain = wade.plot_gene(res, gene=genes, backend=backend)
    with_area = wade.plot_gene(res, gene=genes, backend=backend, cumulative_area=True)
    k = len(genes)
    assert _n_axes(plain, backend) == 2 * k
    assert _n_axes(with_area, backend) == 3 * k
    if backend == "matplotlib":
        # The middle row holds it, and the quantile functions stay last.
        assert with_area.axes[k].get_ylabel() == "cumulative area"
        assert with_area.axes[2 * k].get_ylabel() == "expression"
        np.testing.assert_array_equal(with_area.axes[k].lines[-1].get_ydata(),
                                      res.gene_detail(res.gene_index("g0")).cumulative_area)
        # One curve in the row, so no legend entry: the y axis names it.
        assert "cumulative area" not in [t.get_text() for t in with_area.legends[0].get_texts()]
    else:
        area = [t for t in with_area.data if t.name == "cumulative area"]
        assert len(area) == k and not any(t.showlegend for t in area)
        np.testing.assert_array_equal(area[0].y,
                                      res.gene_detail(res.gene_index("g0")).cumulative_area)
        assert not [t for t in plain.data if t.name == "cumulative area"]


# ---------------------------------------------------------------------------
# Linked views (plotly only, live kernel only)


@pytest.fixture
def linked(res):
    pytest.importorskip("plotly")
    pytest.importorskip("anywidget", reason="plot_linked needs plotly's FigureWidget")
    return wade.plot_linked(res, label=3)


def _click(widget, i):
    """Dispatch a click without a browser — which is also the only way to test
    a live-kernel callback."""
    from plotly.callbacks import InputDeviceState, Points

    widget.data[0]._dispatch_on_click(Points(point_inds=[] if i is None else [i]),
                                      InputDeviceState())


def test_clicking_a_volcano_point_redraws_that_genes_panel(res, linked):
    import plotly.graph_objects as go

    assert isinstance(linked, go.FigureWidget)
    assert linked.layout.annotations[1].text == "click a point"

    i = res.gene_index("g6")
    _click(linked, i)
    ratio, reference, case, ctrl = linked.data[-4:]
    d = res.gene_detail(i)
    # The panel is the result's own curve, not an approximation of it.
    np.testing.assert_array_equal(ratio.y, d.r)
    np.testing.assert_array_equal(case.y, d.y1)
    np.testing.assert_array_equal(ctrl.y, d.y0)
    np.testing.assert_array_equal(ratio.customdata, np.c_[d.y1, d.y0])
    # The dashed reference is the fitted global shift, as in plot_gene.
    assert reference.y == pytest.approx((np.log2(res.fitted_fold_change[i]),) * 2)
    assert "<b>g6</b>" in linked.layout.annotations[1].text

    # A second click replaces rather than accumulates: a redraw, not a rebuild.
    n_traces = len(linked.data)
    _click(linked, res.gene_index("g0"))
    assert len(linked.data) == n_traces
    np.testing.assert_array_equal(linked.data[-4].y, res.gene_detail(res.gene_index("g0")).r)

    # Clicking empty space is a no-op, not a traceback.
    _click(linked, None)
    np.testing.assert_array_equal(linked.data[-4].y, res.gene_detail(res.gene_index("g0")).r)


def test_a_linked_view_carries_the_metadata_and_the_theme(res_meta):
    pytest.importorskip("anywidget", reason="plot_linked needs plotly's FigureWidget")
    w = wade.plot_linked(res_meta, meta="gene_name", theme="dark", label=2)
    assert w.layout.paper_bgcolor == P.THEMES["dark"].surface
    assert w.layout.template.layout.paper_bgcolor is not None      # plotly_dark
    _click(w, res_meta.gene_index("ENSG00006"))
    assert "<b>SYM6</b>" in w.layout.annotations[1].text     # named as displayed


def test_a_linked_view_says_what_it_needs_when_the_widget_will_not_build(res, monkeypatch):
    """anywidget is a fourth optional dependency and the message has to name it;
    every static figure works without it."""
    pytest.importorskip("plotly")
    import plotly.graph_objects as go

    def refuse(*a, **k):
        raise ImportError("Please install anywidget to use the FigureWidget class")

    monkeypatch.setattr(go, "FigureWidget", refuse)
    with pytest.raises(ImportError, match="needs anywidget"):
        wade.plot_linked(res)


# ---------------------------------------------------------------------------
# The driver figure


def test_the_driver_panel_is_subset_drivers_and_library_qc_and_not_a_re_derivation(res_drivers):
    res, counts = res_drivers
    p = P.driver_panel(res, "GENUINE", counts)
    d = wade.subset_drivers(res, "GENUINE")
    qc = wade.library_qc(counts)
    np.testing.assert_array_equal(p.columns, d["columns"])
    np.testing.assert_array_equal(p.value, d["values"])
    np.testing.assert_array_equal(p.share, d["share"])
    for name in ("detected_fraction", "depth", "top_share"):
        np.testing.assert_array_equal(getattr(p, name), qc[name][d["columns"]])
    # The cohort context is over ALL samples, not the case arm: a library's
    # complexity has nothing to do with which arm it is in.
    for name in ("detected_fraction", "depth", "top_share"):
        assert p.cohort[name] == pytest.approx(tuple(np.nanpercentile(qc[name], [25, 50, 75])))
    # One row per driver, and the table carries the concentration the figure
    # does not draw.
    t = p.table()
    assert len({len(c) for c in t.values()}) == 1 and len(t["sample"]) == p.columns.size
    assert set(t["gene"]) == {"GENUINE"} and "top_share" in t


def test_the_driver_panel_separates_an_artefact_from_a_genuine_hit(res_drivers):
    """The figure exists for exactly this call, and it is the check that
    reversed a tempting Ewing-sarcoma reading of a real cohort
    (``docs/scaling.md`` §7.2)."""
    res, counts = res_drivers
    art = P.driver_panel(res, "ARTEFACT", counts)
    gen = P.driver_panel(res, "GENUINE", counts)
    cohort_complexity = gen.cohort["detected_fraction"][1]

    # The artefact's drivers ARE the planted low-complexity libraries, and they
    # sit far below the cohort; the genuine hit's are ordinary libraries.
    assert len(set(art.columns) & set(BAD)) >= 2      # most of its drivers are BAD
    assert np.median(art.detected_fraction) < 0.6 * cohort_complexity
    assert np.median(gen.detected_fraction) > 0.95 * cohort_complexity
    # On the median, not the min: k is ceil(affected_fraction * nprobs), so the
    # last driver can sit one past the real edge of the subset — here that is a
    # single BAD library among nine ordinary ones. The figure shows it honestly,
    # as one dot away from the rest, which is the behaviour to keep.
    assert len(set(gen.columns) & set(BAD)) <= 1
    # ... and the share is the other discriminating number: the artefact takes a
    # large slice of its drivers' libraries — that is what it is telling you
    # about — where the genuine gene takes a fraction of a percent.
    assert art.share.max() > 20 * gen.share.max()
    assert gen.share.max() < 0.01


def test_the_driver_axes_go_logarithmic_only_when_the_data_spans_a_decade(res_drivers):
    res, counts = res_drivers
    p = P.driver_panel(res, "GENUINE", counts)
    # The gene's value spans decades across a cohort; depth within one does not.
    assert p.log_axis("value") is True
    assert p.log_axis("depth") is False
    assert p.log_axis("share") is False and p.log_axis("detected_fraction") is False


def test_the_driver_panel_needs_the_counts_it_was_run_on(res_drivers):
    """A WadeResult cannot supply them: res.tpm is normalized *and* jittered,
    so it has no exact zeros left and cannot be asked what a library detected.
    """
    res, counts = res_drivers
    assert (np.asarray(res.tpm) > 0).all(), "the jitter removed every zero"
    with pytest.raises(ValueError, match="counts must be the matrix"):
        P.driver_panel(res, "GENUINE", counts[:, :-1])
    with pytest.raises(ValueError, match="no subset stage"):
        P.driver_panel(wade.wade(counts, np.ones(GD), COND_D, subset=False, nperms=20),
                       0, counts)


@pytest.mark.parametrize("backend", P.available_backends())
def test_both_backends_draw_one_row_per_driver_and_the_cohort_context(res_drivers, backend):
    res, counts = res_drivers
    p = P.driver_panel(res, "ARTEFACT", counts)
    fig = wade.plot_drivers(res, "ARTEFACT", counts, backend=backend)
    n = len(P.data.DRIVER_PANELS)
    if backend == "matplotlib":
        assert _n_axes(fig, backend) == n
        # A dashed cohort median and a shaded IQR band in every panel.
        assert all(ax.patches for ax in fig.axes[:n])
        np.testing.assert_array_equal(fig.axes[0].get_yticks(), np.arange(p.sample.size))
        assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == list(p.sample)
    else:
        assert _n_axes(fig, backend) == n
        assert sum(1 for sh in fig.layout.shapes if sh.type == "rect") == n
        assert sum(1 for sh in fig.layout.shapes if sh.type == "line") == n
    # Strongest driver at the top, in subset_drivers' order.
    assert p.sample[0] == f"lib{p.columns[0]:02d}"


def test_no_colour_literal_lives_outside_theme_py():
    """Every colour in the package is a theme token.

    A hex or ``rgb()`` string written into a renderer is one the other themes
    cannot follow — which is exactly how a dark figure ends up a light slab
    with dark points. ``theme.py`` is the only file allowed to spell one.
    """
    import re
    from pathlib import Path

    root = Path(wade.__file__).parent
    pattern = re.compile(r"#[0-9a-fA-F]{6}\b|\brgba?\(")
    offenders = {}
    for path in sorted(root.rglob("*.py")):
        if path.name == "theme.py":
            continue
        hits = pattern.findall(path.read_text())
        if hits:
            offenders[str(path.relative_to(root))] = hits
    assert not offenders, f"colour literals outside theme.py: {offenders}"
