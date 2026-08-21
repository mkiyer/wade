"""The data layer: one pure-NumPy dataclass per figure, and nothing else.

Each of :func:`gene_panels`, :func:`volcano_data` and :func:`stages_data`
turns a :class:`~wade.api.WadeResult` into exactly the arrays a figure draws,
and each dataclass has a ``table()`` — the chart's table-view twin, and the
seam a third renderer would be written against. **This module imports no
plotting backend**, which is what makes that promise checkable.

``res.columns()`` is the single namespace for the axes, the colour and the
hover, so a new statistic is plottable the day it exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..api import WadeResult
from ..diagnostics import GeneDetail, wade_gene
from ..subset import affected_fraction as _affected_fraction, direction as _direction
from .theme import QUADRANTS, _COLOR_SPECS, _STAGE_LABELS, _axis_label


def _column_array(res: WadeResult, name: str) -> np.ndarray:
    """One of the result's own columns, by name, as float — the namespace for
    every plottable quantity is exactly ``res.columns()``."""
    cols = res.columns()
    if name not in cols:
        raise ValueError(
            f"{name!r} is not a column of this result; available: "
            f"{sorted(k for k in cols if k != 'gene')}"
        )
    vals = cols[name]
    if vals is None:
        raise ValueError(f"column {name!r} was not computed for this result")
    return np.asarray(vals, dtype=np.float64)


# ---------------------------------------------------------------------------
# The single-gene panel


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

    @property
    def affected_span(self) -> tuple[float, float] | None:
        """The quantile interval the affected fraction's bootstrap CI puts the
        edge of the affected region in — ``None`` when there is no interval.

        ``affected_fraction`` is a participation ratio: for a subset of size
        ``pi`` it reads ``pi`` (``wade.subset.affected_fraction``), so the part
        of the distribution that moved is the **top** ``pi`` of quantiles when
        the change is upward and the **bottom** ``pi`` when it is downward.
        This is therefore where the edge of that region lies, and a wide band
        is a poorly determined extent — which is exactly what a point estimate
        hides: a subset resting on four affected samples reads the same as one
        resting on four hundred.
        """
        ci = self.stats.get("ci_affected_fraction")
        if ci is None:
            return None
        lo, hi = float(ci[0]), float(ci[1])
        return (lo, hi) if self.direction < 0 else (1.0 - hi, 1.0 - lo)

    def table(self) -> dict[str, np.ndarray]:
        """The panel's curves, one row per grid node — its table-view twin.

        These are the arrays :func:`wade.plot_gene` draws, so several panels
        concatenate into one long-format table. As on the point clouds, the
        scalars a panel is annotated with stay attributes rather than becoming
        constant columns: :attr:`reference` is the dashed line, and
        :attr:`affected_fraction`, :attr:`direction` and :attr:`stats` are the
        characterization beneath the title.
        """
        d = self.detail
        return {"gene": np.full(d.p.shape, self.name, dtype=object),
                "p": d.p, "r": d.r, "y1": d.y1, "y0": d.y0,
                "cumulative_area": d.cumulative_area}


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
# The point clouds (volcano and stages share the machinery)


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
    """Resolve ``color`` to ``(values, spec)`` or ``(None, None)``.

    Accepts any column of :meth:`WadeResult.columns` by name, or an array of
    per-gene values (a gene-set membership, a cluster id, a QC score — WADE
    has no way to know what you want to colour by, so it does not guess).
    ``affected_fraction`` and ``direction`` keep their pinned roles and
    ranges; anything else gets the diverging role if it spans zero and the
    sequential one otherwise, autoscaled. The *role* is as far as the data
    layer goes — which colormap draws it is the theme's business.
    """
    if color is None:
        return None, None
    if isinstance(color, str):
        if color in _COLOR_SPECS:
            vals = getattr(res, color, None)
            if vals is None:
                # The characterization was not computed; fall back to a single
                # hue rather than fail.
                return None, None
            return np.asarray(vals, dtype=np.float64), dict(_COLOR_SPECS[color], key=color)
        vals = _column_array(res, color)
        key = color
    else:
        vals = np.asarray(color, dtype=np.float64)
        if vals.shape != (res.gene.shape[0],):
            raise ValueError(
                f"a colour array needs one value per gene: expected "
                f"{res.gene.shape[0]}, got {vals.shape}")
        key = "colour"
    finite = vals[np.isfinite(vals)]
    spans_zero = finite.size > 0 and finite.min() < 0.0 < finite.max()
    return vals, dict(label=_axis_label(key),
                      role="diverging" if spans_zero else "sequential",
                      range=None, key=key)


def _column_ci(res: WadeResult, name: str | None) -> np.ndarray | None:
    """The bootstrap interval for an axis's column, ``(2, genes)`` or ``None``.

    The rule is exactly the name: an axis holding ``foo`` gets error bars iff
    the result carries ``ci_foo``. So a descriptor that gains an interval needs
    nothing wired here, and an axis that is a p-value or a permutation z — for
    which there is no interval — silently gets none.
    """
    if name is None:
        return None
    ci = getattr(res, f"ci_{name}", None)
    return None if ci is None else np.asarray(ci, dtype=np.float64)


def _result_columns(res: WadeResult) -> dict[str, np.ndarray]:
    """Every column the result carries, for the hover. ``res.columns()`` is the
    single namespace for the axes, the colour and the hover, which is what
    keeps a new statistic from having to be wired into three places."""
    return {k: v for k, v in res.columns().items() if v is not None}


def _label_mask(y: np.ndarray, gene: np.ndarray, label, tiebreak=None) -> np.ndarray:
    """Which points get a direct text label: none, the top-``n``, or named genes.

    Ranked by ``y``, **broken by ``|tiebreak|``**. The tie-break is not a
    nicety: on a large cohort thousands of genes share the p-value floor, so
    ranking by ``y`` alone names an arbitrary handful of them. The volcano
    passes its x axis, so the labels fall on the largest effects among the
    equally-significant.
    """
    mask = np.zeros(gene.shape[0], dtype=bool)
    if label is None or label is False:
        return mask
    if isinstance(label, (int, np.integer)) and not isinstance(label, bool):
        n = int(label)
        if n > 0:
            primary = -np.nan_to_num(y, nan=-np.inf)
            if tiebreak is None:
                order = np.argsort(primary)
            else:
                secondary = -np.abs(np.nan_to_num(tiebreak, nan=0.0))
                order = np.lexsort((secondary, primary))
            mask[order[:n]] = True
        return mask
    wanted = set(np.asarray(label, dtype=object).tolist())
    return np.isin(gene, list(wanted))


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
    #: Which result columns the axes hold. ``y_name = None`` means the y axis
    #: is −log₁₀ p of ``stage``, which is the default and the only case where
    #: the BH cutoff line is meaningful.
    x_name: str = "log2_fc"
    y_name: str | None = None
    #: Bootstrap 95% intervals for whichever axes hold a descriptor that has
    #: one, ``(2, genes)`` each. ``None`` at ``n_boot=0``, and on the p-value
    #: and permutation-z axes, which have no interval.
    x_ci: np.ndarray | None = None
    y_ci: np.ndarray | None = None

    @property
    def xlabel(self) -> str:
        return _axis_label(self.x_name)

    @property
    def ylabel(self) -> str:
        if self.y_name is None:
            return f"−log₁₀ p ({_STAGE_LABELS[self.stage]})"
        return _axis_label(self.y_name)

    @property
    def cutoff_y(self) -> float:
        """Where BH rejects, on the y axis — ``nan`` (so the renderers draw no
        line) whenever y is not the p-value axis."""
        if self.y_name is not None or not np.isfinite(self.cutoff):
            return np.nan
        return float(_neglog10(np.array([self.cutoff]))[0])

    @property
    def n_significant(self) -> int:
        return int(self.significant.sum())

    def table(self) -> dict[str, np.ndarray]:
        cols = {"gene": self.gene, self.x_name: self.x, f"p_{self.stage}": self.p,
                f"padj_{self.stage}": self.padj, "significant": self.significant}
        if self.y_name is not None:
            cols[self.y_name] = self.y
        if self.color is not None:
            cols[self.color_spec["key"]] = self.color
        # Named as the result names them, so the table stays a subset of it.
        for name, ci in ((self.x_name, self.x_ci), (self.y_name, self.y_ci)):
            if ci is not None:
                cols[f"{name}_lo"], cols[f"{name}_hi"] = ci[0], ci[1]
        return cols


def volcano_data(res: WadeResult, stage: str = "mean_shift", *, alpha: float = 0.05,
                 color="affected_fraction", label=None,
                 x: str = "log2_fc", y: str | None = None) -> VolcanoData:
    """The arrays behind one volcano panel.

    ``x`` and ``y`` name any columns of :meth:`WadeResult.columns`. The
    defaults are the classic volcano — ``log2_fc`` against −log₁₀ p of
    ``stage`` — and the two most useful alternatives are:

    * ``x="subset_log2_fc"`` on a subset volcano: the subset's magnitude
      rather than the whole gene's fold change, which is what a concentrated
      finding should be read on.
    * ``y="z_subset"`` (or ``"z_mean_shift"``): on a large cohort the
      p-values saturate at the resolution floor and the y axis becomes a flat
      line of ties; the permutation z keeps separating genes. The BH cutoff
      line is then suppressed, because it has no meaning off the p axis.

    ``mean_shift`` is available as an axis but is a poor default: it is in
    TPM-like units and spans thousands across a transcriptome, so plotted as
    an effect size it collapses the cloud onto a vertical line.

    ``color`` takes a column name or a per-gene array; ``label=n`` names the
    top ``n`` by y, **broken by |x|**, so ties at the p-value floor do not
    produce an arbitrary selection.

    Either axis gets **bootstrap error bars** on the labelled points when it
    holds a descriptor the result has an interval for — ``log2_fc``,
    ``mean_shift``, ``subset_log2_fc``, ``affected_fraction`` or ``direction``
    at ``n_boot > 0``. Only the labelled ones: a transcriptome of error bars is
    mush, and every gene's interval is in ``res.columns()`` and so in the
    hover. A p-value or a permutation-z axis has no interval and gets none.

    ``alternative`` is read from the result: under a one-sided alternative the
    untested half of the fold-change axis cannot produce a small p-value, and
    the renderers shade it so the emptiness is read as *untested* rather than
    as *nothing there*.
    """
    p, padj = _stage_arrays(res, stage)
    p = np.asarray(p, dtype=np.float64)
    padj = np.asarray(padj, dtype=np.float64)
    xv = _column_array(res, x)
    yv = _neglog10(p) if y is None else _column_array(res, y)
    colors, spec = _color_arrays(res, color)
    return VolcanoData(
        stage=stage, gene=res.gene, x=xv, y=yv, p=p, padj=padj,
        significant=np.nan_to_num(padj, nan=np.inf) <= alpha, alpha=alpha,
        cutoff=_bh_cutoff(p, padj, alpha), color=colors, color_spec=spec,
        labelled=_label_mask(yv, res.gene, label, tiebreak=xv),
        alternative=str(res.params.get("alternative", "two-sided")),
        hover=_result_columns(res), x_name=x, y_name=y,
        x_ci=_column_ci(res, x), y_ci=_column_ci(res, y),
    )


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
                color="affected_fraction", label=None) -> StagesData:
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
        hover=_result_columns(res),
    )
