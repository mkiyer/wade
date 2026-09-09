"""WADE — Wasserstein Area Differential Expression.

A two-group differential-*distribution* test that answers two questions
ordinary differential expression collapses into one:

1. **Is there a difference?** — ``p_mean_shift``, the ordinary
   mean-difference test.
2. **What kind of difference is it?** — ``p_subset`` asks whether a global
   shift is an inadequate explanation, and ``affected_fraction`` and
   ``direction`` say what the difference looks like.

A gene altered in 5% of cases and a gene shifted 2x in all of them can produce
the same mean difference, and a first-moment test cannot tell them apart.
**Nothing here asks the user to declare in advance which they are looking
for** — there is no percentile cutoff, no window width and no guard factor.

Quick start
-----------
>>> import numpy as np
>>> from wade import wade
>>> rng = np.random.default_rng(0)
>>> counts = rng.poisson(20, size=(50, 200)).astype(float)
>>> counts[0, :5] *= 30                       # 5% of cases, sharply elevated
>>> cond = np.r_[np.ones(100, int), np.zeros(100, int)]
>>> res = wade(counts, np.ones(50), cond, nperms=500)
>>> res.nprobs                                # what the design resolves
100

The entry point takes **raw counts**, and nothing else: the continuity jitter
that breaks ties in sparse data is applied at count precision before
division, and the subset stage's null is built by thinning reads. A
pre-normalized matrix can reproduce neither, so there is no entry point for
one.
"""

from importlib.metadata import PackageNotFoundError, version as _version

from .api import DEFAULT_MAX_PROBS, DEFAULT_NPERMS, WadeResult, wade, wade_contrast
from .diagnostics import GeneDetail, library_qc, subset_drivers, wade_gene
from .io import RESULT_COLUMNS, Condition, Counts, as_counts, condition, manifest, write_results
from .normalize import draw_jitter, library_sizes, tpm_like
from .permutation import (detectability_floor, draw_perms, null_statistics,
                          permutation_space)
from .plotting import plot_drivers, plot_gene, plot_linked, plot_stages, plot_volcano
from .pvalues import ALTERNATIVES, bh_adjust, perm_pvalues
from .quantiles import probability_grid
from .stats import wade_stats
from .subset import affected_fraction, direction, subset_log2_fc

try:
    __version__ = _version("wade-rnaseq")
except PackageNotFoundError:                      # pragma: no cover - source tree, not installed
    __version__ = "0+unknown"

__all__ = [
    "ALTERNATIVES",
    "Condition",
    "Counts",
    "DEFAULT_MAX_PROBS",
    "DEFAULT_NPERMS",
    "GeneDetail",
    "RESULT_COLUMNS",
    "WadeResult",
    "affected_fraction",
    "as_counts",
    "bh_adjust",
    "condition",
    "detectability_floor",
    "direction",
    "draw_jitter",
    "draw_perms",
    "library_qc",
    "library_sizes",
    "manifest",
    "null_statistics",
    "perm_pvalues",
    "permutation_space",
    "plot_drivers",
    "plot_gene",
    "plot_linked",
    "plot_stages",
    "plot_volcano",
    "probability_grid",
    "subset_drivers",
    "subset_log2_fc",
    "tpm_like",
    "wade",
    "wade_contrast",
    "wade_gene",
    "wade_stats",
    "write_results",
]
