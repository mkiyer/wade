"""Palette and labels: everything a figure says that is not a number.

A :class:`Theme` is the tokens a figure is drawn with, and **every colour
literal in the package is in one of the three built-ins below** — a test pins
that, because a colour spelled in a renderer is a colour the other renderer
and the other themes cannot follow.

Both renderers read the same tokens, so a figure reads the same whichever one
drew it, and the labels live here once rather than in each backend. This module
imports nothing but the standard library, which is what keeps it the leaf of
the subpackage.

Colour scales are named by **role**, not by colormap: the data layer says a
quantity is sequential or diverging, and the theme says what those look like.
Only names both backends accept may be used (``viridis`` yes, ``coolwarm`` no —
plotly does not have it).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    """The tokens both renderers draw with.

    Build a variant with :func:`dataclasses.replace`::

        from dataclasses import replace
        from wade.plotting import THEMES, plot_volcano

        house = replace(THEMES["light"], case="#7b3fa0", ctrl="#3fa07b")
        plot_volcano(res, theme=house)

    ``font`` and ``plotly_template`` are read by the plotly renderer only.
    matplotlib's fonts are left to matplotlib: typesetting is not this layer's
    job (``docs/plotting.md``), and the template is plotly's own concept — it
    has to follow the theme or a dark figure is a light slab with dark points.
    """

    #: Titles, axis labels, the log-ratio curve, direct point labels.
    ink: str
    #: Subtitles, tick labels, the dashed reference lines, annotations.
    muted: str
    #: The hairline grid.
    grid: str
    #: Axis lines, ticks, and the zero rule.
    axis: str
    #: The figure's ground, and the ring that keeps overlapping points apart.
    surface: str
    #: The case group, and the single hue a point cloud falls back to.
    case: str
    #: The control group.
    ctrl: str
    #: A quantity with a floor and a ceiling, ``affected_fraction`` above all.
    sequential: str
    #: A signed quantity read against zero, ``direction`` above all.
    diverging: str
    font: str
    plotly_template: str

    def scale(self, role: str) -> str:
        """The colormap for a colour spec's ``role`` (see :data:`_COLOR_SPECS`)."""
        if role not in ("sequential", "diverging"):
            raise ValueError(f"colour role must be 'sequential' or 'diverging'; got {role!r}")
        return self.sequential if role == "sequential" else self.diverging

    def box(self, alpha: float = 0.8) -> str:
        """The surface as a translucent ``rgba()``, for boxes behind in-plot text.

        plotly wants one string where matplotlib takes a colour and an alpha,
        so the conversion lives here rather than in the renderer — which is
        what keeps ``rgba(255,255,255,0.8)`` from being written into it.
        """
        r, g, b = (int(self.surface[i:i + 2], 16) for i in (1, 3, 5))
        return f"rgba({r},{g},{b},{alpha})"


_SANS = "system-ui, -apple-system, 'Segoe UI', sans-serif"

#: The three built-ins. ``"light"`` is the palette the README's figures were
#: drawn with and is the default; the warm greys are deliberate, and the dark
#: theme inverts them rather than going to neutral. ``case``/``ctrl`` stay a
#: blue–orange pair in all three because it survives colour-vision deficiency.
THEMES = {
    "light": Theme(
        ink="#0b0b0b", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
        surface="#ffffff", case="#2a78d6", ctrl="#eb6834",
        sequential="viridis", diverging="RdBu_r",
        font=_SANS, plotly_template="plotly_white",
    ),
    "dark": Theme(
        ink="#f2f1ea", muted="#8f8d85", grid="#2b2c26", axis="#4a4b42",
        surface="#141410", case="#5aa2f2", ctrl="#f5904f",
        sequential="viridis", diverging="RdBu_r",
        font=_SANS, plotly_template="plotly_dark",
    ),
    # Maximum legibility rather than a house style: pure black on pure white,
    # a grid dark enough to survive a projector, and cividis, which is built
    # for colour-vision deficiency where viridis is only safe for it.
    "high-contrast": Theme(
        ink="#000000", muted="#3a3a3a", grid="#a8a8a8", axis="#000000",
        surface="#ffffff", case="#0033aa", ctrl="#cc4400",
        sequential="cividis", diverging="RdBu_r",
        font=_SANS, plotly_template="plotly_white",
    ),
}

#: ``affected_fraction`` is a fraction, ``direction`` a signed coherence.
_COLOR_SPECS = {
    "affected_fraction": dict(label="affected fraction", role="sequential", range=(0.0, 1.0)),
    "direction": dict(label="direction", role="diverging", range=(-1.0, 1.0)),
}

_STAGE_LABELS = {"mean_shift": "mean shift", "subset": "subset"}

#: Axis labels for the columns worth plotting. Anything else falls back to its
#: own name, so a new statistic is plottable the day it exists.
_AXIS_LABELS = {
    "log2_fc": "log₂ fold change (case / control)",
    "subset_log2_fc": "subset log₂ fold change (within the affected fraction)",
    "mean_shift": "mean shift (TPM-like units)",
    "affected_fraction": "affected fraction",
    "direction": "direction",
    "z_mean_shift": "z, mean shift vs its permutation null",
    "z_subset": "z, subset vs its permutation null",
    "subset_stat": "subset statistic",
    "w1": "1-Wasserstein distance",
    "case_mean": "case mean", "ctrl_mean": "control mean",
}


def _axis_label(name: str) -> str:
    return _AXIS_LABELS.get(name, name)


#: The README's reading table, keyed by (mean-shift significant, subset significant).
QUADRANTS = {
    (True, False): "global shift",
    (True, True): "subset, strong enough\nto move the mean",
    (False, True): "distributional change,\nno net mean shift",
    (False, False): "not differential",
}
