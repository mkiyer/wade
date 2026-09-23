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


def plot_gene(result, gene, *, backend="matplotlib", theme="light", title=None):
    """ECDF and exact empirical quantile effects for one gene.

    The annotation is the gene's distribution-test q-value. A tail shape
    or a sample extreme is descriptive, not a separately significant event.
    """
    i = result.gene_index(gene)
    q = result.padj[i]
    annotation = f"distribution q = {q:.3g}" if np.isfinite(q) else str(result.status[i])
    spec = dict(title=f"{result.gene[i] if title is None else title} · {annotation}",
                expression=result.expression[i], cond=result.cond, detail=result.gene_detail(gene),
                bulk=float(result.bulk_shift[i]), theme=_theme(theme))
    return _renderer(backend).gene(spec)


def _scatter(result, kind, alpha, label, side):
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if side not in ("upper", "lower"):
        raise ValueError("side must be 'upper' or 'lower'")
    x = result.bulk_shift
    y = -np.log10(result.padj) if kind == "volcano" else getattr(result, side + "_tail")
    keep = result.tested & np.isfinite(x) & np.isfinite(y)
    labelled = []
    if isinstance(label, (int, np.integer)):
        if label < 0:
            raise ValueError("label count must be nonnegative")
        labelled = np.flatnonzero(keep)[np.argsort(result.padj[keep], kind="stable")[:label]].tolist()
    elif label is not None:
        labelled = [result.gene_index(g) for g in label]
    return dict(x=x, y=y, gene=result.gene, keep=keep, significant=keep & (result.padj <= alpha),
                labelled=labelled, alpha=alpha, kind=kind,
                xlabel="Bulk: difference in medians (log₂ units)",
                ylabel="−log₁₀(distribution q)" if kind == "volcano" else
                       f"{side.capitalize()} decile difference above bulk (log₂ units)")


def plot_volcano(result, *, alpha=.05, label=None, backend="matplotlib", theme="light", title=None):
    """Bulk direction versus distribution significance; the q-value is not a sign test."""
    spec = _scatter(result, "volcano", alpha, label, "upper")
    spec.update(theme=_theme(theme), title=title or "Distribution differences and bulk direction")
    return _renderer(backend).scatter(spec)


def plot_patterns(result, *, side="upper", alpha=.05, label=None, backend="matplotlib", theme="light", title=None):
    """Bulk and tail descriptions, colored by the single distribution-test call."""
    spec = _scatter(result, "patterns", alpha, label, side)
    spec.update(theme=_theme(theme), title=title or "Bulk shift and tail shape")
    return _renderer(backend).scatter(spec)
