"""The matplotlib renderer: static figures, one function per public figure.

The twin of :mod:`wade.plotting._plotly`: the same dataclasses in, so a
figure says the same thing in either backend. ``import matplotlib`` happens
inside each renderer, never here.
"""

from __future__ import annotations

import numpy as np

from .data import _label_positions
from .theme import QUADRANTS, _AXIS, _CASE, _CTRL, _GRID, _INK, _MUTED, _STAGE_LABELS, _SURFACE


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
