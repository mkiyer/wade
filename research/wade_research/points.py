"""Matched empirical area control; no normalization fitting or null model."""
import numpy as np

from .resampling import label_coefficients


def point_scores(x, labels, *, upward=True):
    """Exact empirical CDF integral; ties have zero-width intervals."""
    x = np.asarray(x, float)
    labels = np.asarray(labels)
    label_coefficients(labels)
    if (x.ndim != 1 or labels.ndim != 2 or labels.shape[1] != len(x)
            or not np.isfinite(x).all() or np.any(x < 0)
            or not np.all(labels.sum(axis=1) == labels[0].sum())):
        raise ValueError("Require nonnegative patient abundances and fixed-size label groups")
    order = np.argsort(x, kind='stable')
    n, nc = len(x), int(labels[0].sum())
    mass = nc*np.arange(1, n)-n*np.cumsum(labels[:, order], axis=1, dtype=np.int64)[:, :-1]
    if upward:
        mass = np.maximum(mass, 0)
    else:
        mass = np.abs(mass)
    return np.sum(mass*np.diff(x[order]), axis=1)/(nc*(n-nc))
