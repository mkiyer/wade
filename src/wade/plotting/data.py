"""Small shared plot specification; no inferential logic lives in a renderer."""
import numpy as np
from .theme import THEMES, Theme

EXPRESSION_LABEL = "log₂(count / size factor + 1)"


def _theme(value):
    if isinstance(value, Theme):
        return value
    if value not in THEMES:
        raise ValueError(f"theme must be one of {tuple(THEMES)} or a Theme")
    return THEMES[value]


def _renderer(backend):
    if backend == "matplotlib":
        from . import _matplotlib
        return _matplotlib
    if backend == "plotly":
        from . import _plotly
        return _plotly
    raise ValueError("plot backend must be 'matplotlib' or 'plotly'")


def _display(result, i):
    """Figures show log2 expression; a 1000x subset flattens a linear axis."""
    return np.log2(result.expression[i] + 1)


def plot_gene(result, gene, *, backend="matplotlib", theme="light", title=None):
    """ECDF and exact empirical quantile effects for one gene, on the log2 scale.

    The annotation gives the fold of the group means and the area and subset q-values. The dotted
    line on the quantile panel is the common fold (alignment_log2fc), the fold the subset test
    holds every case to. Vertical lines mark the subset's boundary on both panels.
    """
    from ..diagnostics import gene_detail
    i = result.gene_index(gene)
    annotation = (f"log₂ fold {result.log2fc[i]:+.2f}, area q = {result.area_padj[i]:.3g}, "
                  f"subset q = {result.subset_padj[i]:.3g}" if result.tested[i] else str(result.status[i]))
    x = _display(result, i)
    boundaries = []
    s = result.subset(gene)
    if len(s.samples):
        up, inside = s.side == "up", x[np.isin(result.sample_names, s.samples)]
        boundaries.append(dict(expression=inside.min() if up else inside.max(),
                               quantile=1 - s.fraction if up else s.fraction, style=":",
                               label=f"{s.side.capitalize()} subset: {len(s.samples)} cases, "
                                     f"{s.log2fc:+.2f} log₂ vs the alignment"))
    spec = dict(title=f"{result.gene[i] if title is None else title} · {annotation}",
                expression=x, cond=result.cond, detail=gene_detail(x, result.cond),
                alignment=float(result.alignment_log2fc[i]),
                boundaries=boundaries, theme=_theme(theme))
    return _renderer(backend).gene(spec)


def _scatter(result, kind, alpha, label):
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if kind == "volcano":
        padj, x, y = result.area_padj, result.area_fraction, -np.log10(result.area_padj)
    else:
        padj, x, y = result.subset_padj, result.subset_fraction, result.subset_log2fc
    keep = result.tested & np.isfinite(x) & np.isfinite(y)
    labelled = []
    if isinstance(label, (int, np.integer)):
        if label < 0:
            raise ValueError("label count must be nonnegative")
        labelled = np.flatnonzero(keep)[np.argsort(padj[keep], kind="stable")[:label]].tolist()
    elif label is not None:
        labelled = [result.gene_index(g) for g in label]
    return dict(x=x, y=y, gene=result.gene, keep=keep, significant=keep & (padj <= alpha),
                labelled=labelled, alpha=alpha, kind=kind,
                called=f"{'Area' if kind == 'volcano' else 'Subset'} q ≤ {alpha:g}",
                xlabel="Area fraction" if kind == "volcano" else "The subset's share of the cases",
                ylabel="−log₁₀(area q)" if kind == "volcano" else "Subset vs the alignment (log₂)")


def plot_volcano(result, *, alpha=.05, label=None, backend="matplotlib", theme="light", title=None):
    """Each gene's area fraction against its q-value."""
    spec = _scatter(result, "volcano", alpha, label)
    spec.update(theme=_theme(theme), title=title or "Areas")
    return _renderer(backend).scatter(spec)


def plot_patterns(result, *, alpha=.05, label=None, backend="matplotlib", theme="light", title=None):
    """Each gene's subset: its share of the cases against its fold over the alignment.

    Subsets above what the alignment predicts lie above zero and subsets below it underneath; a
    few samples far from it sit at the left, far from zero. Colored by the subset calls.
    """
    spec = _scatter(result, "patterns", alpha, label)
    spec.update(theme=_theme(theme), title=title or "Subsets")
    return _renderer(backend).scatter(spec)
