"""WADE: one exact Wasserstein distribution test, with descriptive bulk and tails."""
from importlib.metadata import version as _version, PackageNotFoundError
try:
    __version__ = _version("wade-rnaseq")
except PackageNotFoundError:
    __version__ = "0.2.0"

from .api import DEFAULT_MAX_PERMUTATIONS, WadeResult, wade, wade_contrast
from .io import Counts, Condition, RESULT_COLUMNS, as_counts, condition, manifest, write_results
from .normalize import Normalization, cpm, fit_normalization, tpm
from .filtering import filter_counts
from .pvalues import bh_adjust
from .permutation import draw_perms, permutation_space
from .diagnostics import GeneDetail, library_qc
from .plotting import plot_gene, plot_patterns, plot_volcano

__all__ = sorted([
    "Condition", "Counts", "DEFAULT_MAX_PERMUTATIONS", "GeneDetail", "Normalization", "RESULT_COLUMNS",
    "WadeResult", "as_counts", "bh_adjust", "condition", "cpm", "draw_perms", "filter_counts",
    "fit_normalization", "library_qc", "manifest", "permutation_space", "plot_gene", "plot_patterns",
    "plot_volcano", "tpm", "wade", "wade_contrast", "write_results",
])
