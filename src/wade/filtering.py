"""Pooled, label-independent gene selection, separate from normalization."""
import numpy as np
from .io import as_counts


def filter_counts(counts, *, min_count=1, min_samples=1, min_total=0):
    """Return a Boolean mask; combine it with any prespecified custom mask.

    Keep genes with at least min_count counts in min_samples samples and
    pooled total >= min_total. No single threshold is suitable for all assays.
    Pass this mask as wade(..., test_genes=mask) to preserve the normalization
    universe. Do not choose thresholds from group differences or q-values.
    """
    x = as_counts(counts).values
    if not np.isfinite(min_count) or min_count <= 0 or not np.isfinite(min_total) or min_total < 0:
        raise ValueError("min_count must be positive and min_total nonnegative")
    if isinstance(min_samples, bool) or int(min_samples) != min_samples or not 0 <= min_samples <= x.shape[1]:
        raise ValueError("min_samples must be an integer between zero and the cohort size")
    return ((x >= min_count).sum(axis=1) >= min_samples) & (x.sum(axis=1) >= min_total)
