"""WADE — Wasserstein Area Differential Expression.

A two-group differential-*distribution* test for per-gene abundance data,
built to detect genes altered in only a fraction of the case group.

For each gene WADE compares the two groups' empirical quantile functions
on a shared grid of ``min(n_case, n_ctrl)`` probabilities and reports two
axes:

* ``diff_mean`` — the **bulk** axis, the signed area between the quantile
  functions. A whole-group location shift moves it.
* ``tail_mean`` — the **subset** axis, the mean quantile difference over
  the upper-tail window. A gene elevated in a minority of cases has a
  small ``diff_mean`` and a large ``tail_mean``.

Inference is by label permutation on both axes independently, with a
Generalized Pareto refinement where the empirical resolution runs out,
then BH-FDR per axis.

**WADE is one-sided upward throughout.** Every statistic is signed
case-minus-control and every p-value is an upper-tail probability. Genes
*depleted* in cases carry negative statistics and p-values near 1. That
is a design property, not an oversight.

Quick start
-----------
>>> import numpy as np
>>> from wade import wade
>>> rng = np.random.default_rng(0)
>>> counts = rng.poisson(20, size=(50, 24)).astype(float)
>>> counts[0, :6] *= 30                      # a rare high-expressing subset
>>> normalizer = rng.uniform(1, 5, size=(50, 24))
>>> cond = np.r_[np.ones(12, int), np.zeros(12, int)]
>>> res = wade(counts, normalizer, cond, nperms=200)
>>> res.nprobs, res.k                        # grid resolution and tail window
(12, 2)

The entry point takes **raw counts**, not a normalized matrix: the
continuity jitter that breaks ties in sparse data is applied at count
precision before division, so a pre-normalized matrix cannot reproduce
it. :func:`wade_from_matrix` accepts one anyway and documents the cost.
"""

from .api import (
    DEFAULT_NPERMS,
    DEFAULT_TAIL_CONC_MAX_FACTOR,
    WadeResult,
    tail_window_report,
    wade,
    wade_contrast,
    wade_from_matrix,
)
from .diagnostics import GeneDetail, wade_gene
from .normalize import cpm, draw_jitter, library_sizes, rle, tpm_like
from .permutation import draw_perms, null_statistics
from .pvalues import GPDFit, bh_adjust, gpd_tail_p, perm_pvalues
from .quantiles import probability_grid, tail_window_size, type7_quantiles
from .scores import dense_rank_desc, ecdf_values, wade_score
from .shape import (ShapeResult, affected_fraction, bridge, log_ratio_curve,
                    shape_test, up_share)
from .stats import DEFAULT_TAIL_Q, WadeStats, wade_stats

__version__ = "0.1.0.dev0"

__all__ = [
    "DEFAULT_NPERMS",
    "DEFAULT_TAIL_CONC_MAX_FACTOR",
    "DEFAULT_TAIL_Q",
    "GPDFit",
    "GeneDetail",
    "WadeResult",
    "WadeStats",
    "up_share",
    "shape_test",
    "log_ratio_curve",
    "bridge",
    "affected_fraction",
    "ShapeResult",
    "bh_adjust",
    "cpm",
    "dense_rank_desc",
    "draw_jitter",
    "draw_perms",
    "ecdf_values",
    "gpd_tail_p",
    "library_sizes",
    "null_statistics",
    "perm_pvalues",
    "probability_grid",
    "rle",
    "tail_window_report",
    "tail_window_size",
    "tpm_like",
    "type7_quantiles",
    "wade",
    "wade_contrast",
    "wade_from_matrix",
    "wade_gene",
    "wade_score",
    "wade_stats",
]
