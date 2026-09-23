"""Static scientific figures. Each renderer returns an ordinary Figure."""
import numpy as np
from .data import EXPRESSION_LABEL


def _style(fig, axes, theme):
    fig.set_facecolor(theme.surface)
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor(theme.surface)
        ax.tick_params(colors=theme.ink)
        ax.xaxis.label.set_color(theme.ink)
        ax.yaxis.label.set_color(theme.ink)
        ax.title.set_color(theme.ink)
        for spine in ax.spines.values():
            spine.set_color(theme.axis)
        ax.grid(color=theme.grid, linewidth=.5, alpha=.6)
        ax.set_axisbelow(True)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(theme.surface)
            legend.get_frame().set_edgecolor(theme.axis)
            for text in legend.get_texts():
                text.set_color(theme.ink)


def gene(spec):
    import matplotlib.pyplot as plt
    t = spec["theme"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for value, name, color in [(1, "Case", t.case), (0, "Control", t.ctrl)]:
        x = np.sort(spec["expression"][spec["cond"] == value])
        axes[0].step(np.r_[x[0], x], np.r_[0, np.arange(1, len(x)+1)/len(x)],
                     where="post", color=color, label=f"{name} (n={len(x)})", linewidth=1.8)
    axes[0].set(xlabel=EXPRESSION_LABEL, ylabel="Cumulative fraction", ylim=(-.02, 1.02))
    axes[0].legend(loc="best", fontsize=9)
    d = spec["detail"]
    axes[1].stairs(d.effect, d.probability, color=t.ink, linewidth=1.5, baseline=None)
    axes[1].axhline(0, color=t.axis, linewidth=.8)
    axes[1].axhline(spec["bulk"], color=t.case, linestyle="--", label="Median shift")
    axes[1].set(xlabel="Expression quantile", ylabel="Case − control (log₂ units)", xlim=(0, 1))
    axes[1].legend(loc="best", fontsize=9)
    fig.suptitle(spec["title"], color=t.ink, fontsize=12)
    _style(fig, axes, t)
    return fig


def scatter(spec):
    import matplotlib.pyplot as plt
    t = spec["theme"]
    fig, ax = plt.subplots(figsize=(7, 4.8), layout="constrained")
    for mask, color, name in [(spec["keep"] & ~spec["significant"], t.muted, "Not called"),
                              (spec["significant"], t.case, f"Distribution q ≤ {spec['alpha']:g}")]:
        ax.scatter(spec["x"][mask], spec["y"][mask], s=15, c=color, alpha=.65, linewidths=0, label=name, rasterized=True)
    ax.axvline(0, color=t.axis, linewidth=.8)
    if spec["kind"] == "volcano":
        ax.axhline(-np.log10(spec["alpha"]), color=t.axis, linestyle="--", linewidth=.8)
    else:
        ax.axhline(0, color=t.axis, linewidth=.8)
    for i in spec["labelled"]:
        if spec["keep"][i]:
            lo, hi = ax.get_xlim()
            right = spec["x"][i] > lo + .75 * (hi - lo)
            ax.annotate(str(spec["gene"][i]), (spec["x"][i], spec["y"][i]),
                        xytext=(-4 if right else 4, 4), ha="right" if right else "left",
                        textcoords="offset points", fontsize=9, color=t.ink)
    ax.set(xlabel=spec["xlabel"], ylabel=spec["ylabel"], title=spec["title"])
    ax.legend(loc="best", fontsize=9)
    _style(fig, ax, t)
    return fig
