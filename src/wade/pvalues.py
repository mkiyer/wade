"""Finite Monte Carlo p-values and one family-wide BH adjustment."""
import numpy as np


def sequential_p(exceed, tried, *, stop, maximum):
    """Besag–Clifford p-values, only at the prespecified stopping time.

    Stop at the stop-th inclusive exceedance: p=stop/tried. Otherwise
    finish all maximum shuffles: p=(exceed+1)/(maximum+1). An arbitrary
    interrupted run cannot produce a completed p-value.
    """
    k, t = np.asarray(exceed), np.asarray(tried)
    if (not 0 < stop <= maximum or k.shape != t.shape
            or not np.isfinite(k).all() or not np.isfinite(t).all()
            or np.any(k < 0) or np.any(k > stop) or np.any(t < k)
            or np.any(t > maximum) or np.any((k < stop) & (t != maximum))
            or np.any(k != np.floor(k)) or np.any(t != np.floor(t))):
        raise ValueError("invalid or incomplete sequential stopping records")
    return np.where(k == stop, stop / np.maximum(t, 1), (k + 1) / (maximum + 1))


def bh_adjust(p):
    """Benjamini–Hochberg across all finite p-values; retain excluded NaNs."""
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or np.isinf(p).any() or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be a vector in [0, 1], with NaN for exclusions")
    out = np.full(p.shape, np.nan)
    idx = np.flatnonzero(np.isfinite(p))
    order = idx[np.argsort(p[idx], kind="stable")]
    if len(order):
        q = p[order] * len(order) / np.arange(1, len(order) + 1)
        out[order] = np.minimum.accumulate(q[::-1])[::-1].clip(max=1)
    return out
