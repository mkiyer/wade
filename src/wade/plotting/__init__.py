"""Distribution and descriptive-effect plots, with lazy optional dependencies."""
from .theme import THEMES, Theme
from .data import plot_gene, plot_patterns, plot_volcano
__all__ = ["THEMES", "Theme", "plot_gene", "plot_patterns", "plot_volcano"]
