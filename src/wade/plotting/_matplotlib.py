"""The matplotlib renderer: static figures, one function per public figure.

The twin of :mod:`wade.plotting._plotly`: the same dataclasses in, so a
figure says the same thing in either backend. ``import matplotlib`` happens
inside each renderer, never here.
"""

from __future__ import annotations

import numpy as np

from .data import DRIVER_PANELS, _label_positions
from .theme import QUADRANTS, _STAGE_LABELS


def _mpl_style(ax, theme):
    ax.set_facecolor(theme.surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.axis)
    ax.tick_params(colors=theme.muted, labelcolor=theme.muted, width=0.8)
    ax.grid(True, color=theme.grid, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(theme.ink)
    ax.yaxis.label.set_color(theme.ink)


def _gene_mpl(panels, theme, *, share_y, title, width, height):
    import matplotlib.pyplot as plt

    k = len(panels)
    fig, axes = plt.subplots(2, k, figsize=(width or min(18, 1.2 + 3.2 * k), height or 6.4),
                             sharex=True, sharey=False, squeeze=False, constrained_layout=True,
                             gridspec_kw=dict(height_ratios=[0.55, 0.45]))
    fig.patch.set_facecolor(theme.surface)
    if share_y:
        # Only the log-ratio row shares a scale: magnitudes of R are comparable
        # across genes, raw expression levels are not.
        for j in range(1, k):
            axes[0, j].sharey(axes[0, 0])
    for j, p in enumerate(panels):
        d = p.detail
        top, bot = axes[0, j], axes[1, j]
        top.axhline(0.0, color=theme.axis, lw=1)
        top.axhline(p.reference, color=theme.muted, lw=1, ls="--",
                    label=("fitted global shift" if p.log2_fitted_shift is not None
                           else "median(R): fitted global shift") if j == 0 else None)
        span = p.affected_span
        if span is not None:
            # How firm the extent of the affected region is.
            top.axvspan(span[0], span[1], color=theme.case, alpha=0.16, lw=0, zorder=0,
                        label="affected fraction, 95% CI" if j == 0 else None)
        top.plot(d.p, d.r, color=theme.ink, lw=1.6, label="log$_2$ ratio" if j == 0 else None)
        top.set_title(p.name, fontsize=11.5, color=theme.ink, pad=30)
        top.text(0.5, 1.015, "\n".join(p.subtitle_lines), transform=top.transAxes,
                 ha="center", va="bottom", fontsize=8, color=theme.muted, linespacing=1.35)
        bot.plot(d.p, d.y1, color=theme.case, lw=1.6, label="case" if j == 0 else None)
        bot.plot(d.p, d.y0, color=theme.ctrl, lw=1.6, label="control" if j == 0 else None)
        bot.set_yscale("log")
        bot.set_xlabel("quantile p")
        bot.set_xlim(-0.02, 1.02)
        _mpl_style(top, theme)
        _mpl_style(bot, theme)
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
        fig.suptitle(title, color=theme.ink)
    return fig


def _scatter_mpl(ax, data, theme, *, colorbar_ax=None, fig=None):
    n = data.gene.shape[0]
    size = 14 if n <= 2000 else (8 if n <= 10000 else 5)
    kw = dict(s=size, linewidths=0.3, edgecolors=theme.surface, alpha=0.85, rasterized=n > 5000)
    if data.color is None:
        sc = ax.scatter(data.x, data.y, color=theme.case, **kw)
    else:
        spec = data.color_spec
        sc = ax.scatter(data.x, data.y, c=data.color, cmap=theme.scale(spec["role"]),
                        vmin=spec["range"][0], vmax=spec["range"][1], **kw)
        if fig is not None:
            cb = fig.colorbar(sc, ax=colorbar_ax or ax, fraction=0.04, pad=0.02)
            cb.set_label(spec["label"], color=theme.ink)
            cb.outline.set_visible(False)
            cb.ax.tick_params(colors=theme.muted, labelcolor=theme.muted)
    idx, above = _label_positions(data)
    for i, up in zip(idx, above):
        ax.annotate(str(data.display[i]), (data.x[i], data.y[i]), xytext=(0, 4 if up else -4),
                    textcoords="offset points", ha="center", va="bottom" if up else "top",
                    fontsize=8, color=theme.ink)
    return sc


def _intervals_mpl(ax, data, theme):
    """Bootstrap intervals on the **labelled** points, as error bars.

    Only the labelled ones: a transcriptome of error bars is mush, every gene's
    interval is already in the result's own columns, and ``label=`` is exactly
    the "these are the genes I care about" the caller has already given.
    """
    idx = np.flatnonzero(data.labelled)
    if idx.size == 0 or (data.x_ci is None and data.y_ci is None):
        return

    def arms(ci, v):
        return None if ci is None else np.vstack([v[idx] - ci[0][idx], ci[1][idx] - v[idx]])

    ax.errorbar(data.x[idx], data.y[idx], xerr=arms(data.x_ci, data.x),
                yerr=arms(data.y_ci, data.y), fmt="none", ecolor=theme.muted,
                elinewidth=0.9, capsize=2, capthick=0.9, zorder=2)


def _volcano_mpl(data, theme, *, title, width, height):
    import matplotlib.pyplot as plt

    k = len(data)
    fig, axes = plt.subplots(1, k, figsize=(width or (5.2 * k + 1.2), height or 5.0),
                             sharex=True, squeeze=False, constrained_layout=True)
    fig.patch.set_facecolor(theme.surface)
    axes = axes[0]
    for j, d in enumerate(data):
        ax = axes[j]
        if d.alternative != "two-sided":
            x_lim = float(np.nanmax(np.abs(d.x))) * 1.08 if np.isfinite(d.x).any() else 1.0
            x0, x1 = (0.0, x_lim) if d.alternative == "less" else (-x_lim, 0.0)
            ax.axvspan(x0, x1, color=theme.grid, alpha=0.45, lw=0, zorder=0)
            ax.text(x0 if d.alternative == "less" else x1, 0.03, f" untested (alternative = {d.alternative}) ",
                    transform=ax.get_xaxis_transform(), ha="left" if d.alternative == "less" else "right",
                    va="bottom", fontsize=8, color=theme.muted,
                    bbox=dict(facecolor=theme.surface, alpha=0.8, edgecolor="none", pad=2))
        _scatter_mpl(ax, d, theme, fig=fig if j == k - 1 else None)
        _intervals_mpl(ax, d, theme)
        ax.axvline(0.0, color=theme.axis, lw=1)
        if np.isfinite(d.cutoff_y):
            ax.axhline(d.cutoff_y, color=theme.muted, lw=1, ls="--")
            right = d.alternative == "less"
            ax.text(1.0 if right else 0.0, d.cutoff_y, f" FDR {d.alpha:g}: {d.n_significant} genes ",
                    transform=ax.get_yaxis_transform(), ha="right" if right else "left", va="top",
                    fontsize=8, color=theme.muted)
        ax.set_xlabel(d.xlabel)
        ax.set_ylabel(d.ylabel)
        if k > 1:
            ax.set_title(f"{_STAGE_LABELS[d.stage]} stage", color=theme.ink, fontsize=11)
        _mpl_style(ax, theme)
    if title:
        fig.suptitle(title, color=theme.ink)
    return fig


def _stages_mpl(data, theme, *, title, width, height):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width or 6.8, height or 6.0), constrained_layout=True)
    fig.patch.set_facecolor(theme.surface)
    _scatter_mpl(ax, data, theme, fig=fig)
    ax.axvline(data.cutoff_x, color=theme.muted, lw=1, ls="--")
    ax.axhline(data.cutoff_y, color=theme.muted, lw=1, ls="--")
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
                ha=pos["ha"], va=pos["va"], fontsize=8.5, color=theme.muted,
                bbox=dict(facecolor=theme.surface, alpha=0.8, edgecolor="none", pad=3))
    ax.set_xlabel(data.xlabel)
    ax.set_ylabel(data.ylabel)
    ax.set_title(title or f"the two stages (FDR {data.alpha:g})", color=theme.ink)
    _mpl_style(ax, theme)
    return fig


def _drivers_mpl(panel, theme, *, title, width, height):
    import matplotlib.pyplot as plt

    k = panel.sample.size
    n = len(DRIVER_PANELS)
    fig, axes = plt.subplots(1, n, figsize=(width or (2.4 * n + 1.8),
                                            height or max(3.0, 1.4 + 0.30 * k)),
                             sharey=True, squeeze=False, constrained_layout=True)
    fig.patch.set_facecolor(theme.surface)
    axes = axes[0]
    y = np.arange(k, dtype=float)
    for ax, (field_name, label) in zip(axes, DRIVER_PANELS):
        v = np.asarray(getattr(panel, field_name), dtype=np.float64)
        q25, med, q75 = panel.cohort[field_name]
        ax.axvspan(q25, q75, color=theme.grid, alpha=0.55, lw=0, zorder=0,
                   label="cohort interquartile range" if ax is axes[0] else None)
        ax.axvline(med, color=theme.muted, lw=1, ls="--", zorder=1,
                   label="cohort median" if ax is axes[0] else None)
        ax.scatter(v, y, s=34, color=theme.case, linewidths=0.5,
                   edgecolors=theme.surface, zorder=3)
        if panel.log_axis(field_name):
            # One label per decade: minor labels mash together in a panel this
            # narrow, which is the same reason _gene_mpl's y axis has dtick=1.
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel(label, fontsize=9)
        _mpl_style(ax, theme)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(panel.sample, fontsize=8)
    axes[0].set_ylim(k - 0.5, -0.5)          # strongest driver at the top
    fig.legend(loc="outside lower center", frameon=False, fontsize=9, ncol=2)
    fig.suptitle(title or f"{panel.gene}: {panel.subtitle}", color=theme.ink, fontsize=11)
    return fig
