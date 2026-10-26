"""Palette and labels: everything a figure says that is not a number.

A :class:`Theme` is the tokens a figure is drawn with, and every colour literal in the package is
in one of the three built-ins below, so that both renderers and every theme follow it. Both
renderers read the same tokens, so a figure reads the same whichever one drew it. This module
imports nothing but the standard library, which keeps it the leaf of the subpackage.
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

    Every colour is a ``#rrggbb`` hex string. ``font`` and ``plotly_template``
    are read by the plotly renderer only; matplotlib's fonts are left to
    matplotlib.
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
    font: str
    plotly_template: str


_SANS = "system-ui, -apple-system, 'Segoe UI', sans-serif"

#: The three built-ins. ``"light"`` is the default; the warm greys are
#: deliberate, and the dark theme inverts them rather than going to neutral. ``case``/``ctrl`` stay a
#: blue–orange pair in all three because it survives colour-vision deficiency.
THEMES = {
    "light": Theme(
        ink="#0b0b0b", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
        surface="#ffffff", case="#2a78d6", ctrl="#eb6834",
        font=_SANS, plotly_template="plotly_white",
    ),
    "dark": Theme(
        ink="#f2f1ea", muted="#8f8d85", grid="#2b2c26", axis="#4a4b42",
        surface="#141410", case="#5aa2f2", ctrl="#f5904f",
        font=_SANS, plotly_template="plotly_dark",
    ),
    # Maximum legibility rather than a house style: pure black on pure white,
    # and a grid dark enough to survive a projector.
    "high-contrast": Theme(
        ink="#000000", muted="#3a3a3a", grid="#a8a8a8", axis="#000000",
        surface="#ffffff", case="#0033aa", ctrl="#cc4400",
        font=_SANS, plotly_template="plotly_white",
    ),
}

