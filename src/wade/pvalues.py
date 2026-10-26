"""Finite Monte Carlo p-values, the BH decision they serve, and one family-wide BH adjustment."""
import numpy as np

TIE_RTOL = 100 * np.finfo(float).eps                       # values this close are equal


def certified(threshold):
    """Just below threshold, so that BH's own arithmetic, m p / rank <= fdr, holds for a bound
    certified at it."""
    return threshold * (1 - TIE_RTOL)


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


def bound(exceed, tried, *, stop, maximum):
    """The Besag–Clifford p-value under the most conservative future: an upper bound, on every
    path, on the p-value a gene with `exceed` exceedances in `tried` shuffles finishes with.

    Were every remaining shuffle an exceedance, the run would stop at the stop-th one, giving
    stop / (tried + stop - exceed), or, if that comes after maximum shuffles, end at maximum with
    (exceed + maximum - tried + 1) / (maximum + 1). The bound never rises as shuffles are added,
    and equals sequential_p once the run stops. The native kernel computes it in the same
    arithmetic.
    """
    k, t = np.asarray(exceed, dtype=np.int64), np.asarray(tried, dtype=np.int64)
    reach = t + stop - k                                     # where the stop-th exceedance could fall
    return np.where(k >= stop, stop / np.maximum(t, 1),
                    np.where(reach <= maximum, stop / reach, (k + maximum - t + 1) / (maximum + 1)))


def settled(exceed, tried, *, stop, maximum, threshold):
    """Whether each gene needs no more shuffles: its run has stopped, or its bound has reached
    threshold (-inf runs every gene to the end)."""
    k, t = np.asarray(exceed), np.asarray(tried)
    return (k >= stop) | (t >= maximum) | (bound(k, t, stop=stop, maximum=maximum) <= threshold)


def settle(fixed, live, exceed, tried, advance, *, fdr, stop, maximum):
    """Shuffle each gene only as far as its BH decision at level fdr needs: one family of tests.

    fixed holds the p-values of the family's genes that are not shuffled (NaN outside the family
    and for live genes); live marks the shuffled genes, whose records (exceed, tried) the caller
    has already advanced to the first threshold, fdr itself (BH with every gene rejected). Then,
    top-down: count the genes whose bound is at most t = r fdr / m; if the count is r, t is BH's
    threshold; otherwise lower r to the count and advance(genes, threshold) every live gene not
    yet settled at the new t. Each gene ends either settled below t or with its run finished, so
    every count is exactly what the finished p-values would give, and BH on the reported values
    rejects exactly the genes BH on the finished p-values rejects, on every run (Fischer, Barry and
    Ramdas, Electron. J. Statist. 2026, lemma 4.1). The genes it rejects report their bound;
    every other gene reports its finished p-value. fdr=None finishes every run. Returns the
    reported p-values and the number of passes.
    """
    p = np.array(fixed, dtype=float)
    p[live] = bound(exceed[live], tried[live], stop=stop, maximum=maximum)
    m = int(np.sum(np.isfinite(p)))
    if fdr is None or m == 0:
        return p, 1
    r, t, passes = m, fdr, 1
    while (count := int(np.sum(p <= t))) != r:
        r, t = count, count * fdr / m
        below = certified(t)
        need = np.flatnonzero(live & ~settled(exceed, tried, stop=stop, maximum=maximum, threshold=below))
        if len(need):
            advance(need, below)
        p[live] = bound(exceed[live], tried[live], stop=stop, maximum=maximum)
        passes += 1
    return p, passes


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
