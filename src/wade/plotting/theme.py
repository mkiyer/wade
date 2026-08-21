"""Palette and labels: everything a figure says that is not a number.

Two renderers draw from these, so a figure reads the same whichever one drew
it, and the strings live once rather than in each backend. This module imports
nothing — not even NumPy — which is what keeps it the leaf of the subpackage.
"""

from __future__ import annotations

_INK = "#0b0b0b"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"
_SURFACE = "#ffffff"
_CASE = "#2a78d6"
_CTRL = "#eb6834"
_FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
_SEQUENTIAL = "viridis"
_DIVERGING = "RdBu_r"

#: ``affected_fraction`` is a fraction, ``direction`` a signed coherence.
_COLOR_SPECS = {
    "affected_fraction": dict(label="affected fraction", scale=_SEQUENTIAL, range=(0.0, 1.0)),
    "direction": dict(label="direction", scale=_DIVERGING, range=(-1.0, 1.0)),
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
