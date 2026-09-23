"""Empirical gene curves and library QC; no sample-level outlier calls."""
from dataclasses import dataclass
import numpy as np
from .io import as_counts
from .permutation import validate_labels


@dataclass(frozen=True)
class GeneDetail:
    """Exact empirical quantile intervals, ordered from low to high expression."""
    probability: np.ndarray
    case_quantiles: np.ndarray
    control_quantiles: np.ndarray
    effect: np.ndarray
    cumulative_area: np.ndarray


def gene_detail(expression, cond):
    x = np.asarray(expression, float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("expression must be a finite vector")
    cond = validate_labels(cond, len(x))
    a, b = np.sort(x[cond == 1]), np.sort(x[cond == 0])
    edges = np.unique(np.r_[np.arange(len(a)+1)/len(a), np.arange(len(b)+1)/len(b)])
    mid = (edges[:-1] + edges[1:]) / 2
    qa, qb = a[np.minimum((mid*len(a)).astype(int),len(a)-1)], b[np.minimum((mid*len(b)).astype(int),len(b)-1)]
    effect = qa-qb
    area = np.r_[0, np.cumsum(np.abs(effect)*np.diff(edges))]
    return GeneDetail(edges, qa, qb, effect, area)


def library_qc(counts, *, top_n=10):
    """Raw depth, detected genes, and fraction held by the top_n genes.

    Fix technical sample exclusions before examining gene significance.
    These diagnostics do not distinguish technical artifacts from biology.
    """
    x = as_counts(counts).values
    if not x.shape[0]:
        raise ValueError("library_qc requires at least one measured gene")
    if isinstance(top_n, bool) or int(top_n) != top_n or top_n < 1:
        raise ValueError("top_n must be a positive integer")
    k = min(int(top_n), x.shape[0])
    total = x.sum(axis=0)
    top = np.partition(x, x.shape[0]-k, axis=0)[-k:].sum(axis=0)
    share = np.divide(top, total, out=np.full(total.shape, np.nan), where=total > 0)
    detected = (x > 0).sum(axis=0)
    return dict(depth=total, n_detected=detected, detected_fraction=detected/x.shape[0], top_share=share)
