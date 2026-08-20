"""Figures: the single-gene panel, the volcano, and the two-stage plot.

Three functions, each answering one question about a result:

``plot_gene``
    *What does this gene's difference look like?* The log-ratio curve
    ``R(p)`` against quantile, with the two quantile functions beneath it.
    This is the figure that makes the method legible: a **flat** curve is a
    global fold change, a curve that sits at zero and then **climbs** is a
    subset. ``wade_gene()`` already computes everything it draws.
``plot_volcano``
    *Which genes?* Effect size against significance for one stage (or both
    side by side), coloured by ``affected_fraction`` so the shape of each
    difference is visible in the overview, not only per gene.
``plot_stages``
    *What kind of difference?* ``p_mean_shift`` against ``p_subset`` — the
    four quadrants of the README's reading table, drawn.

Two backends, one data layer
----------------------------
Every figure is built from a small pure-NumPy dataclass (:func:`gene_panels`,
:func:`volcano_data`, :func:`stages_data`) that holds exactly the arrays the
figure draws. The renderers are thin and there are two of them:

* ``"plotly"`` — interactive. Hover shows the gene name and every statistic,
  zoom and pan work, and a 20,000-gene volcano renders through WebGL. This is
  the backend for exploring a result in a notebook, and it is the default when
  both are installed.
* ``"matplotlib"`` — static. Vector PDF/SVG with no browser involved, which is
  what a manuscript needs.

Neither library is a dependency of the statistic. Both are imported *inside*
the functions below, so ``import wade`` and the whole test stay at their one
NumPy dependency, and the data layer can be used with any other tool — it is
also the table-view twin of every chart.

Conventions shared by both backends: case in blue, control in orange (a pair
that survives colour-vision deficiency); ``affected_fraction`` on the viridis
ramp, dark at 0 (a subset) and light at 1 (everything); ``direction`` on a
blue–grey–red diverging ramp, blue down and red up; thin marks, hairline grid,
no chart junk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .api import WadeResult
from .diagnostics import GeneDetail, wade_gene
from .subset import affected_fraction as _affected_fraction, direction as _direction

__all__ = [
    "BACKENDS",
    "DEFAULT_BACKEND",
    "GenePanel",
    "VolcanoData",
    "StagesData",
    "available_backends",
    "gene_panels",
    "volcano_data",
    "stages_data",
    "plot_gene",
    "plot_volcano",
    "plot_stages",
]

BACKENDS = ("plotly", "matplotlib")

#: Which backend ``backend=None`` resolves to. ``"auto"`` takes the first of
#: :data:`BACKENDS` that imports. Set it to a name to pin one for a session.
DEFAULT_BACKEND = "auto"

# ---------------------------------------------------------------------------
# Theme. One light palette, shared by both renderers so a figure reads the
# same whichever one drew it.

_INK = "#0b0b0b"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"
_SURFACE = "#ffffff"
_CASE = "#2a78d6"
_CTRL = "#eb6834"
_FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
_SEQUENTIAL = "viridis"
_DIVERGING = "RdBu_r"

#: ``affected_fraction`` is a fraction, ``direction`` a signed coherence.
_COLOR_SPECS = {
    "affected_fraction": dict(label="affected fraction", scale=_SEQUENTIAL, range=(0.0, 1.0)),
    "direction": dict(label="direction", scale=_DIVERGING, range=(-1.0, 1.0)),
}

_STAGE_LABELS = {"mean_shift": "mean shift", "subset": "subset"}


# ---------------------------------------------------------------------------
# Backend resolution


def available_backends() -> tuple[str, ...]:
    """The plotting backends that import in this environment, in preference order."""
    out = []
    for name in BACKENDS:
        try:
            __import__(name)
        except ImportError:
            continue
        out.append(name)
    return tuple(out)


def _resolve_backend(backend: str | None) -> str:
    if backend is None:
        backend = DEFAULT_BACKEND
    if backend == "auto":
        avail = available_backends()
        if not avail:
            raise ImportError(
                "no plotting backend is installed. WADE's figures need plotly "
                "(interactive) or matplotlib (static); neither is a dependency of the "
                "statistic. Install one: `conda install plotly` or `conda install "
                "matplotlib-base`, or `pip install 'wade[plot]'`."
            )
        return avail[0]
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS} or 'auto'; got {backend!r}")
    try:
        __import__(backend)
    except ImportError as e:
        raise ImportError(
            f"backend {backend!r} was requested but is not installed: {e}. "
            f"It is optional — install it with conda or pip, or choose another "
            f"backend (available: {available_backends() or 'none'})."
        ) from e
    return backend


# ---------------------------------------------------------------------------
# Data layer: the single-gene panel


@dataclass(frozen=True)
class GenePanel:
    """One gene's curves plus the numbers the panel is annotated with.

    The dashed reference is the global fold change stage 2 tests the curve
    *against*: ``log2_fitted_shift`` when the panel came from a result (the
    fold change the thinning or the division was performed at), and
    ``median_r`` otherwise. The curve lying on it is a global fold change;
    the curve leaving it is what ``p_subset`` prices.
    """

    detail: GeneDetail
    name: str
    median_r: float
    affected_fraction: float
    direction: float
    #: The global fold change the subset stage's null was built under, as
    #: log2, when the panel came from a :class:`WadeResult`. ``None``
    #: otherwise, where ``median_r`` stands in for it.
    log2_fitted_shift: float | None = None
    #: Stage p-values and effect sizes when the panel came from a
    #: :class:`WadeResult`; empty otherwise.
    stats: dict = field(default_factory=dict)

    @property
    def reference(self) -> float:
        """The dashed line: the fitted global shift, in log2."""
        return self.median_r if self.log2_fitted_shift is None else self.log2_fitted_shift

    @property
    def subtitle_lines(self) -> list[str]:
        """The two p-values when known, then the characterization — one line each."""
        lines = []
        if "p_mean_shift" in self.stats:
            ps = [f"p mean-shift {_fmt_p(self.stats['p_mean_shift'])}"]
            if self.stats.get("p_subset") is not None:
                ps.append(f"p subset {_fmt_p(self.stats['p_subset'])}")
            lines.append(" · ".join(ps))
        aff = f"affected {self.affected_fraction:.2f}"
        if "ci_affected_fraction" in self.stats:
            lo, hi = self.stats["ci_affected_fraction"]
            aff += f" [{lo:.2f}, {hi:.2f}]"
        dirn = f"direction {self.direction:+.2f}"
        if "ci_direction" in self.stats:
            lo, hi = self.stats["ci_direction"]
            dirn += f" [{lo:+.2f}, {hi:+.2f}]"
        lines.append(f"{aff} · {dirn}")
        return lines

    @property
    def subtitle(self) -> str:
        return " · ".join(self.subtitle_lines)


def _fmt_p(p) -> str:
    p = float(p)
    if not np.isfinite(p):
        return "n/a"
    return f"{p:.2g}" if p < 0.01 else f"{p:.3f}"


def _panel_from_detail(d: GeneDetail, name: str, stats: dict | None = None,
                       log2_fitted_shift: float | None = None) -> GenePanel:
    r = d.r[None, :]
    return GenePanel(
        detail=d, name=name, median_r=float(np.median(d.r)),
        affected_fraction=float(_affected_fraction(r)[0]),
        direction=float(_direction(r)[0]),
        log2_fitted_shift=log2_fitted_shift,
        stats=dict(stats or {}),
    )


def _stats_for(res: WadeResult, i: int) -> dict:
    out = {
        "mean_shift": float(res.mean_shift[i]),
        "log2_fc": float(res.log2_fc[i]),
        "p_mean_shift": float(res.p_mean_shift[i]),
        "padj_mean_shift": float(res.padj_mean_shift[i]),
    }
    if res.p_subset is not None:
        out["p_subset"] = float(res.p_subset[i])
        out["padj_subset"] = float(res.padj_subset[i])
    if res.ci_affected_fraction is not None:
        out["ci_affected_fraction"] = (float(res.ci_affected_fraction[0, i]),
                                      float(res.ci_affected_fraction[1, i]))
    if res.ci_direction is not None:
        out["ci_direction"] = (float(res.ci_direction[0, i]), float(res.ci_direction[1, i]))
    return out


def gene_panels(source, cond=None, *, gene=None, names=None, pseudocount=None) -> list[GenePanel]:
    """Normalize every accepted input of :func:`plot_gene` to a list of panels.

    ``source`` may be

    * a :class:`~wade.diagnostics.GeneDetail`, or a sequence of them;
    * a 1-D **normalized** row (``cond`` required), or a 2-D genes x samples
      block of rows (``cond`` required; ``names`` optional);
    * a :class:`~wade.api.WadeResult` with ``gene`` naming one gene or a list
      of genes, by name or by index. Curves are re-derived from the matrix the
      test ran on, so they are the curves behind the reported statistics, and
      the panel is annotated with them.

    Rows are taken as already normalized because that is what
    :func:`wade.wade_gene` takes; hand it raw counts and the log-ratio curve
    will be on the wrong scale with no error raised. ``pseudocount`` (scalar or
    per-sample) applies to the row and block forms; a :class:`WadeResult`
    carries its own.
    """
    if isinstance(source, WadeResult):
        if gene is None:
            raise ValueError("pass gene= (a name, an index, or a list of them) with a WadeResult")
        genes = list(gene) if isinstance(gene, (list, tuple, np.ndarray)) else [gene]
        out = []
        fitted = source.fitted_fold_change
        for gsel in genes:
            i = source.gene_index(gsel)
            shift = None if fitted is None else float(np.log2(fitted[i]))
            out.append(_panel_from_detail(source.gene_detail(i), str(source.gene[i]),
                                          _stats_for(source, i), shift))
        return out

    if isinstance(source, GeneDetail):
        return [_panel_from_detail(source, _name_or(names, 0, "gene"))]

    if isinstance(source, (list, tuple)) and source and all(isinstance(s, GeneDetail) for s in source):
        return [_panel_from_detail(d, _name_or(names, k, f"gene {k}")) for k, d in enumerate(source)]

    arr = np.asarray(source, dtype=np.float64)
    if cond is None:
        raise ValueError("cond is required when plotting from a normalized row or rows")
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"expected a 1-D row or a 2-D genes x samples block, got shape {arr.shape}")
    return [_panel_from_detail(wade_gene(arr[k], cond, pseudocount=pseudocount),
                               _name_or(names, k, f"gene {k}"))
            for k in range(arr.shape[0])]


def _name_or(names, k: int, default: str) -> str:
    if names is None:
        return default
    if isinstance(names, str):
        return names if k == 0 else f"{names} {k}"
    return str(names[k])


# ---------------------------------------------------------------------------
# Data layer: the point clouds (volcano and stages share the machinery)


def _bh_cutoff(p: np.ndarray, padj: np.ndarray, alpha: float) -> float:
    """The raw p-value at which BH rejects at level ``alpha``.

    BH rejects every gene whose raw p is at or below the largest raw p among
    the genes with ``padj <= alpha``, so that value is the line a volcano
    should draw. When nothing passes, the line is drawn where the *first*
    rejection would have to fall — ``alpha / G`` — so the figure still says
    how far the best gene was from significance instead of omitting the
    threshold altogether.
    """
    ok = np.isfinite(p) & np.isfinite(padj)
    if not ok.any():
        return np.nan
    passed = ok & (padj <= alpha)
    if passed.any():
        return float(p[passed].max())
    return float(alpha / ok.sum())


def _neglog10(p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log10(np.asarray(p, dtype=np.float64))


def _stage_arrays(res: WadeResult, stage: str):
    if stage == "mean_shift":
        return res.p_mean_shift, res.padj_mean_shift
    if stage == "subset":
        if res.p_subset is None:
            raise ValueError(
                "this result has no subset stage (nperms == 0, subset=False, or "
                "min(n_case, n_ctrl) < 3); plot stage='mean_shift'"
            )
        return res.p_subset, res.padj_subset
    raise ValueError(f"stage must be 'mean_shift' or 'subset'; got {stage!r}")


def _color_arrays(res: WadeResult, color):
    """Resolve ``color`` to ``(values, spec)`` or ``(None, None)``."""
    if color is None:
        return None, None
    if color not in _COLOR_SPECS:
        raise ValueError(f"color must be 'affected_fraction', 'direction' or None; got {color!r}")
    vals = getattr(res, color)
    if vals is None:
        # The characterization was not computed; fall back to a single hue
        # rather than fail, and say so through the returned spec.
        return None, None
    return np.asarray(vals, dtype=np.float64), dict(_COLOR_SPECS[color], key=color)


def _hover_columns(res: WadeResult) -> dict[str, np.ndarray]:
    cols = {
        "gene": res.gene,
        "log2_fc": res.log2_fc,
        "mean_shift": res.mean_shift,
        "p_mean_shift": res.p_mean_shift,
        "padj_mean_shift": res.padj_mean_shift,
    }
    if res.p_subset is not None:
        cols.update(
            p_subset=res.p_subset, padj_subset=res.padj_subset,
            affected_fraction=res.affected_fraction, direction=res.direction,
        )
    return cols


def _label_mask(y: np.ndarray, gene: np.ndarray, label) -> np.ndarray:
    """Which points get a direct text label: none, the top-``n`` by ``y``, or named genes."""
    mask = np.zeros(gene.shape[0], dtype=bool)
    if label is None or label is False:
        return mask
    if isinstance(label, (int, np.integer)) and not isinstance(label, bool):
        n = int(label)
        if n > 0:
            order = np.argsort(-np.nan_to_num(y, nan=-np.inf))
            mask[order[:n]] = True
        return mask
    wanted = set(np.asarray(label, dtype=object).tolist())
    return np.isin(gene, list(wanted))


@dataclass(frozen=True)
class VolcanoData:
    """Everything one volcano panel draws. ``table()`` is its table-view twin."""

    stage: str
    gene: np.ndarray
    x: np.ndarray                 # log2 fold change
    y: np.ndarray                 # -log10 p of the stage
    p: np.ndarray
    padj: np.ndarray
    significant: np.ndarray       # padj <= alpha
    alpha: float
    cutoff: float                 # raw p at which BH rejects; nan when undefined
    color: np.ndarray | None
    color_spec: dict | None
    labelled: np.ndarray          # bool, which points get a text label
    alternative: str
    hover: dict = field(default_factory=dict)

    @property
    def xlabel(self) -> str:
        return "log₂ fold change (case / control)"

    @property
    def ylabel(self) -> str:
        return f"−log₁₀ p ({_STAGE_LABELS[self.stage]})"

    @property
    def cutoff_y(self) -> float:
        return float(_neglog10(np.array([self.cutoff]))[0]) if np.isfinite(self.cutoff) else np.nan

    @property
    def n_significant(self) -> int:
        return int(self.significant.sum())

    def table(self) -> dict[str, np.ndarray]:
        cols = {"gene": self.gene, "log2_fc": self.x, f"p_{self.stage}": self.p,
                f"padj_{self.stage}": self.padj, "significant": self.significant}
        if self.color is not None:
            cols[self.color_spec["key"]] = self.color
        return cols


def volcano_data(res: WadeResult, stage: str = "mean_shift", *, alpha: float = 0.05,
                 color: str | None = "affected_fraction", label=None) -> VolcanoData:
    """The arrays behind one volcano panel.

    The x axis is ``log2_fc``, and only that. ``mean_shift`` is in TPM-like
    units and spans thousands across a transcriptome, so plotted as an effect
    size it collapses the cloud onto a vertical line; the fold change is the
    comparable scale.

    ``alternative`` is read from the result: under a one-sided alternative the
    untested half of the fold-change axis cannot produce a small p-value, and
    the renderers shade it so the emptiness is read as *untested* rather than
    as *nothing there*.
    """
    p, padj = _stage_arrays(res, stage)
    p = np.asarray(p, dtype=np.float64)
    padj = np.asarray(padj, dtype=np.float64)
    x = np.asarray(res.log2_fc, dtype=np.float64)
    y = _neglog10(p)
    colors, spec = _color_arrays(res, color)
    return VolcanoData(
        stage=stage, gene=res.gene, x=x, y=y, p=p, padj=padj,
        significant=np.nan_to_num(padj, nan=np.inf) <= alpha, alpha=alpha,
        cutoff=_bh_cutoff(p, padj, alpha), color=colors, color_spec=spec,
        labelled=_label_mask(y, res.gene, label),
        alternative=str(res.params.get("alternative", "two-sided")),
        hover=_hover_columns(res),
    )


#: The README's reading table, keyed by (mean-shift significant, subset significant).
QUADRANTS = {
    (True, False): "global shift",
    (True, True): "subset, strong enough\nto move the mean",
    (False, True): "distributional change,\nno net mean shift",
    (False, False): "not differential",
}


@dataclass(frozen=True)
class StagesData:
    """Everything the two-stage plot draws. ``table()`` is its table-view twin."""

    gene: np.ndarray
    x: np.ndarray                 # -log10 p_mean_shift
    y: np.ndarray                 # -log10 p_subset
    sig_mean: np.ndarray
    sig_subset: np.ndarray
    alpha: float
    cutoff_mean: float            # raw-p BH cutoffs
    cutoff_subset: float
    color: np.ndarray | None
    color_spec: dict | None
    labelled: np.ndarray
    hover: dict = field(default_factory=dict)

    xlabel: str = "−log₁₀ p (mean shift)"
    ylabel: str = "−log₁₀ p (subset)"

    @property
    def cutoff_x(self) -> float:
        return float(_neglog10(np.array([self.cutoff_mean]))[0])

    @property
    def cutoff_y(self) -> float:
        return float(_neglog10(np.array([self.cutoff_subset]))[0])

    def quadrant(self) -> np.ndarray:
        """The reading-table label of every gene, as an object array."""
        return np.array([QUADRANTS[(bool(a), bool(b))].replace("\n", " ")
                         for a, b in zip(self.sig_mean, self.sig_subset)], dtype=object)

    def counts(self) -> dict[str, int]:
        q = self.quadrant()
        return {k.replace("\n", " "): int((q == k.replace("\n", " ")).sum()) for k in QUADRANTS.values()}

    def table(self) -> dict[str, np.ndarray]:
        cols = {"gene": self.gene, "p_mean_shift": 10.0 ** -self.x, "p_subset": 10.0 ** -self.y,
                "significant_mean_shift": self.sig_mean, "significant_subset": self.sig_subset,
                "quadrant": self.quadrant()}
        if self.color is not None:
            cols[self.color_spec["key"]] = self.color
        return cols


def stages_data(res: WadeResult, *, alpha: float = 0.05,
                color: str | None = "affected_fraction", label=None) -> StagesData:
    """The arrays behind :func:`plot_stages`.

    Both axes are ``-log10`` of the *raw* p-value and the quadrant lines sit at
    each stage's BH cutoff (:func:`_bh_cutoff`), so a gene is in the
    "significant" half of an axis exactly when its ``padj`` for that stage is
    at or below ``alpha`` — the same criterion the README's table means.
    """
    p_m, padj_m = _stage_arrays(res, "mean_shift")
    p_s, padj_s = _stage_arrays(res, "subset")
    x = _neglog10(p_m)
    y = _neglog10(p_s)
    colors, spec = _color_arrays(res, color)
    # Label by the joint surprise, so the corner genes are the ones named.
    score = np.nan_to_num(x, nan=0.0) + np.nan_to_num(y, nan=0.0)
    return StagesData(
        gene=res.gene, x=x, y=y,
        sig_mean=np.nan_to_num(padj_m, nan=np.inf) <= alpha,
        sig_subset=np.nan_to_num(padj_s, nan=np.inf) <= alpha,
        alpha=alpha,
        cutoff_mean=_bh_cutoff(np.asarray(p_m, float), np.asarray(padj_m, float), alpha),
        cutoff_subset=_bh_cutoff(np.asarray(p_s, float), np.asarray(padj_s, float), alpha),
        color=colors, color_spec=spec, labelled=_label_mask(score, res.gene, label),
        hover=_hover_columns(res),
    )


# ---------------------------------------------------------------------------
# Public API


def plot_gene(source, cond=None, *, gene=None, names=None, pseudocount=None,
              backend: str | None = None, share_y: bool = True, title: str | None = None,
              width: float | None = None, height: float | None = None):
    """The single-gene panel: ``R(p)`` over quantile, quantile functions beneath.

    Parameters
    ----------
    source, cond, gene, names
        See :func:`gene_panels`. The common calls are
        ``plot_gene(result, gene="MYC")``, ``plot_gene(result, gene=[...])``
        for several genes side by side, ``plot_gene(row, cond)`` for one
        normalized row, and ``plot_gene(wade_gene(row, cond))``.
    backend
        ``"plotly"``, ``"matplotlib"``, or ``None`` for :data:`DEFAULT_BACKEND`.
    share_y
        Put every gene's log-ratio panel on the same y axis, so magnitudes are
        comparable across columns. Quantile panels are always per-gene.
    width, height
        Figure size in **pixels** for plotly and in **inches** for
        matplotlib — each library's native unit. Defaults scale with the
        number of genes.

    Returns the backend's figure object: a ``plotly.graph_objects.Figure`` or a
    ``matplotlib.figure.Figure``.

    How to read it
    --------------
    Top: ``R(p) = log2 Q_case(p) - log2 Q_ctrl(p)``. A flat line at ``c`` is a
    global fold change of ``2**c``; a line at zero that climbs near ``p = 1``
    is a subset of cases with elevated expression; one that dips near
    ``p = 0`` is a subset with reduced expression. The dashed line is the
    fitted global shift — the null the subset test argues against — so the
    curve's departure from it is what ``p_subset`` prices. The solid line is
    zero. Bottom: the two quantile functions on a log axis, which is
    the raw evidence the curve above is the ratio of.
    """
    panels = gene_panels(source, cond, gene=gene, names=names, pseudocount=pseudocount)
    be = _resolve_backend(backend)
    if be == "plotly":
        return _gene_plotly(panels, share_y=share_y, title=title, width=width, height=height)
    return _gene_mpl(panels, share_y=share_y, title=title, width=width, height=height)


def plot_volcano(res: WadeResult, stage: str = "mean_shift", *, alpha: float = 0.05,
                 color: str | None = "affected_fraction", label=None,
                 backend: str | None = None, title: str | None = None,
                 width: float | None = None, height: float | None = None):
    """Effect size against significance, coloured by the shape of the difference.

    Parameters
    ----------
    stage
        ``"mean_shift"``, ``"subset"``, or ``"both"`` for the two side by side
        on a shared x axis — which is the view that shows a subset gene sitting
        low on the first and high on the second.
    alpha
        BH FDR level. The dashed line is the raw p at which BH rejects at this
        level (see :func:`_bh_cutoff`); ``significant`` in the hover and the
        table means ``padj <= alpha``.
    color
        ``"affected_fraction"`` (default, viridis: dark is a subset, light is a
        global change), ``"direction"`` (blue down, red up), or ``None``.
    label
        Genes to annotate directly: an integer ``n`` labels the ``n`` most
        significant, a list of names labels those. Labels are selective by
        design; the hover carries the rest.

    The x axis is ``log2_fc``. ``mean_shift`` is deliberately not offered as
    an x axis: it is in TPM-like units and spans thousands, so it collapses a
    transcriptome onto a vertical line. Under a one-sided ``alternative`` the
    untested half of the axis is shaded.
    """
    stages = ("mean_shift", "subset") if stage == "both" else (stage,)
    data = [volcano_data(res, s, alpha=alpha, color=color, label=label) for s in stages]
    be = _resolve_backend(backend)
    if be == "plotly":
        return _volcano_plotly(data, title=title, width=width, height=height)
    return _volcano_mpl(data, title=title, width=width, height=height)


def plot_stages(res: WadeResult, *, alpha: float = 0.05,
                color: str | None = "affected_fraction", label=None,
                backend: str | None = None, title: str | None = None,
                width: float | None = None, height: float | None = None):
    """``p_mean_shift`` against ``p_subset``: the README's four quadrants, drawn.

    The dashed lines are each stage's BH cutoff at ``alpha``, so the quadrant a
    gene falls in is exactly its row in the reading table: significant on the
    mean shift only is a **global shift**; on both, a **subset strong enough
    to move the mean**; on the subset stage only, a **distributional change
    with no net mean shift**; on neither, **not differential**. Colour by
    ``affected_fraction`` to see that the upper-right corner is dark (small
    fractions) and the lower-right light (everything moved).

    Parameters are as for :func:`plot_volcano`; ``label=n`` names the ``n``
    genes with the largest joint ``-log10 p``.
    """
    data = stages_data(res, alpha=alpha, color=color, label=label)
    be = _resolve_backend(backend)
    if be == "plotly":
        return _stages_plotly(data, title=title, width=width, height=height)
    return _stages_mpl(data, title=title, width=width, height=height)


# ---------------------------------------------------------------------------
# plotly renderers


def _plotly_layout(fig, *, title, width, height, legend_below=False):
    fig.update_layout(
        template="plotly_white", title=title, width=width, height=height,
        font=dict(family=_FONT, size=12, color=_INK),
        paper_bgcolor=_SURFACE, plot_bgcolor=_SURFACE,
        margin=dict(l=70, r=30, t=70 if title else 50, b=60),
        hoverlabel=dict(font=dict(family=_FONT)),
    )
    fig.update_xaxes(showgrid=True, gridcolor=_GRID, gridwidth=1, zeroline=False,
                     linecolor=_AXIS, ticks="outside", tickcolor=_AXIS, title_font=dict(color=_INK),
                     tickfont=dict(color=_MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=_GRID, gridwidth=1, zeroline=False,
                     linecolor=_AXIS, ticks="outside", tickcolor=_AXIS, title_font=dict(color=_INK),
                     tickfont=dict(color=_MUTED))
    if legend_below:
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.12, x=0,
                                      font=dict(color=_INK)))
    return fig


def _gene_plotly(panels, *, share_y, title, width, height):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    k = len(panels)
    titles = []
    for p in panels:
        sub = "<br>".join(p.subtitle_lines)
        titles.append(f"<b>{p.name}</b><br><span style='font-size:10px;color:{_MUTED}'>{sub}</span>")
    fig = make_subplots(rows=2, cols=k, shared_xaxes=True, shared_yaxes=False,
                        subplot_titles=titles, row_heights=[0.55, 0.45],
                        vertical_spacing=0.12, horizontal_spacing=0.06 if k > 1 else 0.05)
    if share_y:
        # Only the log-ratio row shares a scale: magnitudes of R are comparable
        # across genes, raw expression levels are not.
        for j in range(2, k + 1):
            fig.update_yaxes(matches="y", row=1, col=j)
    for j, p in enumerate(panels, start=1):
        d = p.detail
        fig.add_trace(go.Scatter(
            x=d.p, y=d.r, mode="lines", line=dict(color=_INK, width=1.8),
            name="log₂ ratio", legendgroup="r", showlegend=(j == 1),
            customdata=np.c_[d.y1, d.y0],
            hovertemplate=("p = %{x:.3f}<br>log₂ ratio = %{y:.2f}<br>"
                           "Q<sub>case</sub> = %{customdata[0]:.3g}<br>Q<sub>ctrl</sub> = %{customdata[1]:.3g}"
                           "<extra></extra>"),
        ), row=1, col=j)
        fig.add_hline(y=0.0, line=dict(color=_AXIS, width=1), row=1, col=j)
        fig.add_hline(y=p.reference, line=dict(color=_MUTED, width=1, dash="dash"), row=1, col=j)
        fig.add_trace(go.Scatter(
            x=d.p, y=d.y1, mode="lines", line=dict(color=_CASE, width=1.8),
            name="case", legendgroup="case", showlegend=(j == 1),
            hovertemplate="p = %{x:.3f}<br>case Q = %{y:.3g}<extra></extra>",
        ), row=2, col=j)
        fig.add_trace(go.Scatter(
            x=d.p, y=d.y0, mode="lines", line=dict(color=_CTRL, width=1.8),
            name="control", legendgroup="ctrl", showlegend=(j == 1),
            hovertemplate="p = %{x:.3f}<br>control Q = %{y:.3g}<extra></extra>",
        ), row=2, col=j)
        fig.update_yaxes(type="log", dtick=1, minor=dict(showgrid=False), row=2, col=j)
        # No explicit x range: p already spans [0, 1], and a fixed range here
        # breaks plotly.js's autorange on the matched y axes above.
        fig.update_xaxes(title_text="quantile p", row=2, col=j)
    # The dashed reference is a legend entry drawn with an empty trace, so the
    # figure explains itself without a caption.
    label = ("fitted global shift" if any(p.log2_fitted_shift is not None for p in panels)
             else "median(R): fitted global shift")
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                             line=dict(color=_MUTED, width=1, dash="dash"),
                             name=label), row=1, col=1)
    fig.update_yaxes(title_text="log₂(Q<sub>case</sub> / Q<sub>ctrl</sub>)", row=1, col=1)
    fig.update_yaxes(title_text="expression", row=2, col=1)
    for ann in fig.layout.annotations:
        ann.font.size = 13
        ann.yshift = 8
    _plotly_layout(fig, title=title, width=width or (min(1700, 200 + 330 * k)),
                   height=height or 600, legend_below=True)
    fig.update_layout(margin=dict(t=110 if title else 90))
    fig.update_layout(hovermode="x unified")
    return fig


def _marker_plotly(data, go):
    """Marker spec for a point cloud: colour ramp when a colour column exists,
    a single hue otherwise; a thin white ring so overlapping points stay
    separable."""
    base = dict(size=6, line=dict(width=0.5, color=_SURFACE), opacity=0.85)
    if data.color is None:
        return dict(base, color=_CASE)
    spec = data.color_spec
    return dict(base, color=data.color, colorscale=spec["scale"],
                cmin=spec["range"][0], cmax=spec["range"][1],
                colorbar=dict(title=dict(text=spec["label"]), thickness=12, len=0.6,
                              outlinewidth=0, tickfont=dict(color=_MUTED)))


def _hover_plotly(data):
    """A customdata block and a template listing every statistic WADE reports."""
    cols = data.hover
    names = [k for k in cols if k != "gene"]
    cd = np.column_stack([np.asarray(cols[k], dtype=np.float64) for k in names]) if names else None
    lines = ["<b>%{text}</b>"]
    for i, k in enumerate(names):
        fmt = ".3g" if k.startswith("p") else ".3f"
        lines.append(f"{k} = %{{customdata[{i}]:{fmt}}}")
    return cd, "<br>".join(lines) + "<extra></extra>"


def _cloud_trace(data, go, *, name="genes"):
    """One scatter trace; WebGL past a few thousand points so a transcriptome
    stays responsive."""
    n = data.gene.shape[0]
    cls = go.Scattergl if n > 2000 else go.Scatter
    cd, template = _hover_plotly(data)
    return cls(x=data.x, y=data.y, mode="markers", text=data.gene, customdata=cd,
               hovertemplate=template, marker=_marker_plotly(data, go), name=name,
               showlegend=False)


def _label_positions(data) -> tuple[np.ndarray, np.ndarray]:
    """Labelled indices and, for each, whether the text goes above or below.

    Genes at a p-value floor share a y, so consecutive labels along x
    alternate sides. Cheap, deterministic, and enough for the *selective*
    labelling this layer does; dense labelling belongs to the hover.
    """
    idx = np.flatnonzero(data.labelled)
    idx = idx[np.argsort(data.x[idx], kind="stable")]
    above = (np.arange(idx.size) % 2) == 0
    return idx, above


def _labels_plotly(data, go, fig, row=None, col=None):
    idx, above = _label_positions(data)
    if idx.size == 0:
        return
    for sel, pos in ((above, "top center"), (~above, "bottom center")):
        if not sel.any():
            continue
        fig.add_trace(go.Scatter(
            x=data.x[idx[sel]], y=data.y[idx[sel]], mode="text", text=data.gene[idx[sel]],
            textposition=pos, textfont=dict(size=11, color=_INK),
            hoverinfo="skip", showlegend=False,
        ), row=row, col=col)


def _volcano_plotly(data, *, title, width, height):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    k = len(data)
    fig = make_subplots(rows=1, cols=k, shared_xaxes=True, shared_yaxes=False,
                        subplot_titles=[f"{_STAGE_LABELS[d.stage]} stage" for d in data] if k > 1 else None,
                        horizontal_spacing=0.08)
    cloud_idx = []
    for j, d in enumerate(data, start=1):
        fig.add_trace(_cloud_trace(d, go), row=1, col=j)
        cloud_idx.append(len(fig.data) - 1)
        # One-sided alternative: shade the half of the fold-change axis the test
        # cannot find anything on, so emptiness there reads as untested. Added
        # after the trace: plotly skips shapes on subplots that are still empty.
        if d.alternative != "two-sided":
            x_lim = float(np.nanmax(np.abs(d.x))) * 1.08 if np.isfinite(d.x).any() else 1.0
            x0, x1 = (0.0, x_lim) if d.alternative == "less" else (-x_lim, 0.0)
            fig.add_vrect(x0=x0, x1=x1, fillcolor=_GRID, opacity=0.45, line_width=0,
                          layer="below", row=1, col=j,
                          annotation_text=f"untested (alternative = {d.alternative})",
                          annotation_position="bottom left" if d.alternative == "greater" else "bottom right",
                          annotation_font=dict(size=10, color=_MUTED),
                          annotation_bgcolor="rgba(255,255,255,0.8)")
        fig.add_vline(x=0.0, line=dict(color=_AXIS, width=1), row=1, col=j)
        if np.isfinite(d.cutoff_y):
            fig.add_hline(y=d.cutoff_y, line=dict(color=_MUTED, width=1, dash="dash"), row=1, col=j,
                          annotation_text=f"FDR {d.alpha:g}: {d.n_significant} genes",
                          annotation_position="bottom right" if d.alternative == "less" else "bottom left",
                          annotation_font=dict(size=10, color=_MUTED))
        _labels_plotly(d, go, fig, row=1, col=j)
        fig.update_xaxes(title_text=d.xlabel, row=1, col=j)
        fig.update_yaxes(title_text=d.ylabel, row=1, col=j)
    if k > 1 and data[0].color is not None:
        # One colour bar for both panels: the same column, the same scale.
        for idx in cloud_idx[:-1]:
            fig.data[idx].marker.showscale = False
    _plotly_layout(fig, title=title, width=width or (520 * k + 120), height=height or 520)
    return fig


def _stages_plotly(data, *, title, width, height):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(_cloud_trace(data, go))
    fig.add_vline(x=data.cutoff_x, line=dict(color=_MUTED, width=1, dash="dash"))
    fig.add_hline(y=data.cutoff_y, line=dict(color=_MUTED, width=1, dash="dash"))
    # Headroom, so the upper quadrant labels do not sit on the genes they count.
    finite = data.y[np.isfinite(data.y)]
    if finite.size:
        fig.update_yaxes(range=[-0.04 * finite.max(), finite.max() * 1.22])
    counts = data.counts()
    # Corner labels: the reading table, in place. Positions in paper fractions.
    corners = {
        (True, False): dict(x=0.99, y=0.02, xanchor="right", yanchor="bottom"),
        (True, True): dict(x=0.99, y=0.98, xanchor="right", yanchor="top"),
        (False, True): dict(x=0.01, y=0.98, xanchor="left", yanchor="top"),
        (False, False): dict(x=0.01, y=0.02, xanchor="left", yanchor="bottom"),
    }
    for key, text in QUADRANTS.items():
        pos = corners[key]
        n = counts[text.replace("\n", " ")]
        fig.add_annotation(xref="x domain", yref="y domain", showarrow=False,
                           text=f"<b>{text.replace(chr(10), '<br>')}</b><br>{n} genes",
                           font=dict(size=11, color=_MUTED), align="left" if pos["xanchor"] == "left" else "right",
                           bgcolor="rgba(255,255,255,0.8)", borderpad=3, **pos)
    _labels_plotly(data, go, fig)
    fig.update_xaxes(title_text=data.xlabel)
    fig.update_yaxes(title_text=data.ylabel)
    _plotly_layout(fig, title=title or f"the two stages (FDR {data.alpha:g})",
                   width=width or 700, height=height or 600)
    return fig


# ---------------------------------------------------------------------------
# matplotlib renderers


def _mpl_style(ax):
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
    ax.tick_params(colors=_MUTED, labelcolor=_MUTED, width=0.8)
    ax.grid(True, color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(_INK)
    ax.yaxis.label.set_color(_INK)


def _gene_mpl(panels, *, share_y, title, width, height):
    import matplotlib.pyplot as plt

    k = len(panels)
    fig, axes = plt.subplots(2, k, figsize=(width or min(18, 1.2 + 3.2 * k), height or 6.4),
                             sharex=True, sharey=False, squeeze=False, constrained_layout=True,
                             gridspec_kw=dict(height_ratios=[0.55, 0.45]))
    fig.patch.set_facecolor(_SURFACE)
    if share_y:
        # Only the log-ratio row shares a scale: magnitudes of R are comparable
        # across genes, raw expression levels are not.
        for j in range(1, k):
            axes[0, j].sharey(axes[0, 0])
    for j, p in enumerate(panels):
        d = p.detail
        top, bot = axes[0, j], axes[1, j]
        top.axhline(0.0, color=_AXIS, lw=1)
        top.axhline(p.reference, color=_MUTED, lw=1, ls="--",
                    label=("fitted global shift" if p.log2_fitted_shift is not None
                           else "median(R): fitted global shift") if j == 0 else None)
        top.plot(d.p, d.r, color=_INK, lw=1.6, label="log$_2$ ratio" if j == 0 else None)
        top.set_title(p.name, fontsize=11.5, color=_INK, pad=30)
        top.text(0.5, 1.015, "\n".join(p.subtitle_lines), transform=top.transAxes,
                 ha="center", va="bottom", fontsize=8, color=_MUTED, linespacing=1.35)
        bot.plot(d.p, d.y1, color=_CASE, lw=1.6, label="case" if j == 0 else None)
        bot.plot(d.p, d.y0, color=_CTRL, lw=1.6, label="control" if j == 0 else None)
        bot.set_yscale("log")
        bot.set_xlabel("quantile p")
        bot.set_xlim(-0.02, 1.02)
        _mpl_style(top)
        _mpl_style(bot)
    axes[0, 0].set_ylabel("log$_2$(Q$_{case}$ / Q$_{ctrl}$)")
    axes[1, 0].set_ylabel("expression")
    handles, labels = [], []
    for ax in (axes[0, 0], axes[1, 0]):
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    fig.legend(handles, labels, loc="outside lower center", frameon=False, fontsize=9,
               ncol=len(labels) if k > 1 else 2)
    if title:
        fig.suptitle(title, color=_INK)
    return fig


def _scatter_mpl(ax, data, *, colorbar_ax=None, fig=None):
    n = data.gene.shape[0]
    size = 14 if n <= 2000 else (8 if n <= 10000 else 5)
    kw = dict(s=size, linewidths=0.3, edgecolors=_SURFACE, alpha=0.85, rasterized=n > 5000)
    if data.color is None:
        sc = ax.scatter(data.x, data.y, color=_CASE, **kw)
    else:
        spec = data.color_spec
        sc = ax.scatter(data.x, data.y, c=data.color, cmap=spec["scale"],
                        vmin=spec["range"][0], vmax=spec["range"][1], **kw)
        if fig is not None:
            cb = fig.colorbar(sc, ax=colorbar_ax or ax, fraction=0.04, pad=0.02)
            cb.set_label(spec["label"], color=_INK)
            cb.outline.set_visible(False)
            cb.ax.tick_params(colors=_MUTED, labelcolor=_MUTED)
    idx, above = _label_positions(data)
    for i, up in zip(idx, above):
        ax.annotate(str(data.gene[i]), (data.x[i], data.y[i]), xytext=(0, 4 if up else -4),
                    textcoords="offset points", ha="center", va="bottom" if up else "top",
                    fontsize=8, color=_INK)
    return sc


def _volcano_mpl(data, *, title, width, height):
    import matplotlib.pyplot as plt

    k = len(data)
    fig, axes = plt.subplots(1, k, figsize=(width or (5.2 * k + 1.2), height or 5.0),
                             sharex=True, squeeze=False, constrained_layout=True)
    fig.patch.set_facecolor(_SURFACE)
    axes = axes[0]
    for j, d in enumerate(data):
        ax = axes[j]
        if d.alternative != "two-sided":
            x_lim = float(np.nanmax(np.abs(d.x))) * 1.08 if np.isfinite(d.x).any() else 1.0
            x0, x1 = (0.0, x_lim) if d.alternative == "less" else (-x_lim, 0.0)
            ax.axvspan(x0, x1, color=_GRID, alpha=0.45, lw=0, zorder=0)
            ax.text(x0 if d.alternative == "less" else x1, 0.03, f" untested (alternative = {d.alternative}) ",
                    transform=ax.get_xaxis_transform(), ha="left" if d.alternative == "less" else "right",
                    va="bottom", fontsize=8, color=_MUTED,
                    bbox=dict(facecolor=_SURFACE, alpha=0.8, edgecolor="none", pad=2))
        _scatter_mpl(ax, d, fig=fig if j == k - 1 else None)
        ax.axvline(0.0, color=_AXIS, lw=1)
        if np.isfinite(d.cutoff_y):
            ax.axhline(d.cutoff_y, color=_MUTED, lw=1, ls="--")
            right = d.alternative == "less"
            ax.text(1.0 if right else 0.0, d.cutoff_y, f" FDR {d.alpha:g}: {d.n_significant} genes ",
                    transform=ax.get_yaxis_transform(), ha="right" if right else "left", va="top",
                    fontsize=8, color=_MUTED)
        ax.set_xlabel(d.xlabel)
        ax.set_ylabel(d.ylabel)
        if k > 1:
            ax.set_title(f"{_STAGE_LABELS[d.stage]} stage", color=_INK, fontsize=11)
        _mpl_style(ax)
    if title:
        fig.suptitle(title, color=_INK)
    return fig


def _stages_mpl(data, *, title, width, height):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width or 6.8, height or 6.0), constrained_layout=True)
    fig.patch.set_facecolor(_SURFACE)
    _scatter_mpl(ax, data, fig=fig)
    ax.axvline(data.cutoff_x, color=_MUTED, lw=1, ls="--")
    ax.axhline(data.cutoff_y, color=_MUTED, lw=1, ls="--")
    # Headroom for the two upper quadrant labels. The upper-right corner is
    # exactly where the most interesting genes are — a label there would sit
    # on top of the data it is counting.
    top = np.nanmax(data.y[np.isfinite(data.y)]) if np.isfinite(data.y).any() else 1.0
    ax.set_ylim(top=top * 1.22 if top > 0 else 1.0)
    counts = data.counts()
    corners = {
        (True, False): dict(x=0.99, y=0.02, ha="right", va="bottom"),
        (True, True): dict(x=0.99, y=0.99, ha="right", va="top"),
        (False, True): dict(x=0.01, y=0.99, ha="left", va="top"),
        (False, False): dict(x=0.01, y=0.02, ha="left", va="bottom"),
    }
    for key, text in QUADRANTS.items():
        pos = corners[key]
        n = counts[text.replace("\n", " ")]
        ax.text(pos["x"], pos["y"], f"{text}\n{n} genes", transform=ax.transAxes,
                ha=pos["ha"], va=pos["va"], fontsize=8.5, color=_MUTED,
                bbox=dict(facecolor=_SURFACE, alpha=0.8, edgecolor="none", pad=3))
    ax.set_xlabel(data.xlabel)
    ax.set_ylabel(data.ylabel)
    ax.set_title(title or f"the two stages (FDR {data.alpha:g})", color=_INK)
    _mpl_style(ax)
    return fig
