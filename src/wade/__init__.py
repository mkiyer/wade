"""WADE: each gene's area between cases and controls, and the subset of cases that departs from a common fold."""
from importlib.metadata import version as _version, PackageNotFoundError
try:
    __version__ = _version("wade-rnaseq")
except PackageNotFoundError:
    __version__ = "0+unknown"                  # not installed: no package metadata to read

from .api import DEFAULT_MAX_PERMUTATIONS, Subset, WadeResult, wade, wade_contrast
from .io import Counts, Condition, RESULT_COLUMNS, as_counts, condition, manifest, write_results
from .normalize import Normalization, cpm, fit_normalization, tpm
from .filtering import filter_counts
from .pvalues import bh_adjust
from .permutation import permutation_space
from .diagnostics import GeneDetail, library_qc
from .plotting import plot_gene, plot_patterns, plot_volcano

__all__ = sorted([
    "Condition", "Counts", "DEFAULT_MAX_PERMUTATIONS", "GeneDetail", "Normalization", "RESULT_COLUMNS",
    "Subset", "WadeResult", "as_counts", "bh_adjust", "condition", "cpm", "filter_counts",
    "fit_normalization", "library_qc", "manifest", "permutation_space", "plot_gene", "plot_patterns",
    "plot_volcano", "tpm", "wade", "wade_contrast", "write_results",
])
