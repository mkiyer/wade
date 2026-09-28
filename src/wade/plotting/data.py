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
    """Figures show log2 expression whatever the test scale; a 1000x subset flattens a linear axis."""
    x = result.expression[i]
    return x if result.params["scale"] == "log" else np.log2(x + 1)


def plot_gene(result, gene, *, backend="matplotlib", theme="light", title=None):
    """ECDF and exact empirical quantile effects for one gene, on the log2 scale.

    The annotation gives each tested direction's q-value; vertical lines mark
    the boundaries of the subsets the walk learned. Subsets are descriptions,
    not separately significant events.
    """
    from ..diagnostics import gene_detail
    i = result.gene_index(gene)
    tested = [f"{d} q = {q:.3g}" for d, q in (("up", result.padj_up[i]), ("down", result.padj_down[i]))
              if np.isfinite(q)]
    annotation = ", ".join(tested) if tested else str(result.status[i])
    x = _display(result, i)
    scale = result.params["scale"]
    boundaries = []
    for direction in ("up", "down"):
        s = result.subset(gene, direction)
        if len(s.samples):
            edge = s.boundary if scale == "log" else np.log2(s.boundary + 1)
            boundaries.append((float(edge), f"{direction.capitalize()} subset: {len(s.samples)} cases",
                               "--" if direction == "up" else ":"))
    spec = dict(title=f"{result.gene[i] if title is None else title} · {annotation}",
                expression=x, cond=result.cond, detail=gene_detail(x, result.cond),
                bulk=float(result.bulk_shift[i]), boundaries=boundaries, theme=_theme(theme))
    return _renderer(backend).gene(spec)


def _scatter(result, kind, direction, alpha, label):
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if direction not in ("up", "down"):
        raise ValueError("direction must be 'up' or 'down'")
    padj = getattr(result, f"padj_{direction}")
    if not np.isfinite(padj).any() and result.tested.any():
        raise ValueError(f"direction {direction!r} was not tested in this result")
    if kind == "volcano":
        x, y = result.bulk_shift, -np.log10(padj)
    else:
        x, y = getattr(result, f"subset_{direction}_fraction"), getattr(result, f"subset_{direction}_log2fc")
    keep = result.tested & np.isfinite(x) & np.isfinite(y)
    labelled = []
    if isinstance(label, (int, np.integer)):
        if label < 0:
            raise ValueError("label count must be nonnegative")
        labelled = np.flatnonzero(keep)[np.argsort(padj[keep], kind="stable")[:label]].tolist()
    elif label is not None:
        labelled = [result.gene_index(g) for g in label]
    return dict(x=x, y=y, gene=result.gene, keep=keep, significant=keep & (padj <= alpha),
                labelled=labelled, alpha=alpha, kind=kind, called=f"{direction.capitalize()} q ≤ {alpha:g}",
                xlabel="Bulk: difference in medians (log₂ units)" if kind == "volcano" else
                       f"Share of cases in the {direction} subset",
                ylabel=f"−log₁₀({direction} q)" if kind == "volcano" else
                       f"{direction.capitalize()} subset median − control median (log₂ units)")


def plot_volcano(result, *, direction="up", alpha=.05, label=None, backend="matplotlib", theme="light", title=None):
    """Bulk shift against one direction's q-value.

    An upward call left of zero is a raised subset above a lowered bulk.
    """
    spec = _scatter(result, "volcano", direction, alpha, label)
    spec.update(theme=_theme(theme), title=title or f"{direction.capitalize()} areas and bulk shift")
    return _renderer(backend).scatter(spec)


def plot_patterns(result, *, direction="up", alpha=.05, label=None, backend="matplotlib", theme="light", title=None):
    """Each gene's learned subset: the share of cases in it against its fold over controls.

    Subset biomarkers sit at a small share and a large fold; whole-cohort
    shifts at a large share and a modest fold. Colored by that direction's calls.
    """
    spec = _scatter(result, "patterns", direction, alpha, label)
    spec.update(theme=_theme(theme), title=title or f"{direction.capitalize()} subsets")
    return _renderer(backend).scatter(spec)
