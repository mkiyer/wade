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

Conventions shared by both backends and every theme: case in blue, control in
orange (a pair that survives colour-vision deficiency); ``affected_fraction``
on a sequential ramp, dark at 0 (a subset) and light at 1 (everything);
``direction`` on a diverging ramp, blue down and red up; thin marks, hairline
grid, no chart junk.

``theme=`` restyles any of the three figures — :data:`THEMES` holds
``"light"`` (the default, and the one the README's figures were drawn with),
``"dark"`` and ``"high-contrast"``, and :data:`DEFAULT_THEME` sets one for a
whole session. The data layer names a colour *role*; the theme says what that
role looks like, so a figure carries no colour of its own.

Where the code lives: ``theme`` is the tokens and every label and imports
nothing at all; ``data`` is the dataclasses and imports no backend; ``_plotly``
and ``_matplotlib`` are the two renderers, each importing its library inside
its functions; this module resolves the backend and the theme and holds the
three ``plot_*``.
"""

from __future__ import annotations

from ..api import WadeResult
from .data import (
    GenePanel,
    StagesData,
    VolcanoData,
    gene_panels,
    stages_data,
    volcano_data,
)
from .theme import QUADRANTS, THEMES, Theme

__all__ = [
    "BACKENDS",
    "DEFAULT_BACKEND",
    "DEFAULT_THEME",
    "THEMES",
    "Theme",
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

#: Which theme ``theme=None`` resolves to — a key of :data:`THEMES` or a
#: :class:`Theme`. Set it once to restyle a whole session::
#:
#:     import wade.plotting
#:     wade.plotting.DEFAULT_THEME = "dark"
DEFAULT_THEME = "light"

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


def _resolve_theme(theme) -> Theme:
    if theme is None:
        theme = DEFAULT_THEME
    if isinstance(theme, Theme):
        return theme
    if theme in THEMES:
        return THEMES[theme]
    raise ValueError(
        f"theme must be a Theme or one of {tuple(THEMES)}; got {theme!r}. "
        f"Build a variant with dataclasses.replace(THEMES['light'], case=...)."
    )


# ---------------------------------------------------------------------------
# Public API


def plot_gene(source, cond=None, *, gene=None, names=None, pseudocount=None,
              backend: str | None = None, theme=None, share_y: bool = True,
              title: str | None = None,
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
    theme
        A key of :data:`THEMES` — ``"light"``, ``"dark"``,
        ``"high-contrast"`` — or a :class:`~wade.plotting.theme.Theme`, or
        ``None`` for :data:`DEFAULT_THEME`. Every figure takes it.
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

    At ``n_boot > 0`` a shaded band marks where the **edge of the affected
    region** lies (:attr:`GenePanel.affected_span`) — the bootstrap interval on
    ``affected_fraction``, on the side ``direction`` points to. A narrow band is
    a well-determined extent; a wide one is a subset resting on a handful of
    samples, which the point estimate alone does not distinguish.
    """
    panels = gene_panels(source, cond, gene=gene, names=names, pseudocount=pseudocount)
    be = _resolve_backend(backend)
    th = _resolve_theme(theme)
    if be == "plotly":
        from ._plotly import _gene_plotly
        return _gene_plotly(panels, th, share_y=share_y, title=title, width=width, height=height)
    from ._matplotlib import _gene_mpl
    return _gene_mpl(panels, th, share_y=share_y, title=title, width=width, height=height)


def plot_volcano(res: WadeResult, stage: str = "mean_shift", *, alpha: float = 0.05,
                 color="affected_fraction", label=None,
                 x: str = "log2_fc", y: str | None = None,
                 backend: str | None = None, theme=None, title: str | None = None,
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
        global change), ``"direction"`` (blue down, red up), ``None``, any
        other result column by name, or your own per-gene array.
    x, y
        Which result columns the axes hold (see :func:`volcano_data`).
        Defaults are the classic volcano, ``log2_fc`` against −log₁₀ p. On a
        saturated cohort the two worth reaching for are
        ``x="subset_log2_fc"`` (the subset's own magnitude) and
        ``y="z_subset"`` (which does not tie at the p-value floor); off the p
        axis the BH line is suppressed because it means nothing there.
    label
        Genes to annotate directly: an integer ``n`` labels the ``n`` most
        significant, a list of names labels those. Labels are selective by
        design; the hover carries the rest. **The labelled points also carry
        their bootstrap error bars**, on whichever axis holds a descriptor the
        result has an interval for — so the genes you named are the ones whose
        firmness you can see.

    ``mean_shift`` is available as an axis but is a poor choice: it is in
    TPM-like units and spans thousands, so it collapses a transcriptome onto a
    vertical line. Under a one-sided ``alternative`` the untested half of the
    fold-change axis is shaded.
    """
    stages = ("mean_shift", "subset") if stage == "both" else (stage,)
    data = [volcano_data(res, s, alpha=alpha, color=color, label=label, x=x, y=y)
            for s in stages]
    be = _resolve_backend(backend)
    th = _resolve_theme(theme)
    if be == "plotly":
        from ._plotly import _volcano_plotly
        return _volcano_plotly(data, th, title=title, width=width, height=height)
    from ._matplotlib import _volcano_mpl
    return _volcano_mpl(data, th, title=title, width=width, height=height)


def plot_stages(res: WadeResult, *, alpha: float = 0.05,
                color="affected_fraction", label=None,
                backend: str | None = None, theme=None, title: str | None = None,
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
    th = _resolve_theme(theme)
    if be == "plotly":
        from ._plotly import _stages_plotly
        return _stages_plotly(data, th, title=title, width=width, height=height)
    from ._matplotlib import _stages_mpl
    return _stages_mpl(data, th, title=title, width=width, height=height)
