"""Descriptive effects: the bulk shift and the enriched subsets learned from each gene.

Neither is a separate test, an outlier call or a prevalence estimate.
"""
import numpy as np

from ._wasserstein import TIE_RTOL


def walk(x, cond):
    """The standardized running sum of cases over controls down the pooled ranking.

    Rows are genes. Cut c places the c lowest samples below it (c = 1..n-1).
    At each cut, the excess is the fraction of cases above it minus the fraction
    of controls above it, divided by that excess's standard deviation under
    random labelling of a set that size. Cuts inside a run of tied values (equal
    within 100 ulps) are NaN, so ties never straddle a boundary. Returns the
    sorted values, the cases below each cut, and z; the maximum of z bounds the
    upward subset, the minimum the downward one.
    """
    x = np.asarray(x, dtype=float)
    cond = np.asarray(cond)
    n, n1 = x.shape[1], int(cond.sum())
    n0 = n - n1
    order = np.argsort(x, axis=1, kind="stable")
    values = np.take_along_axis(x, order, axis=1)
    below = np.cumsum(cond[order], axis=1)[:, :-1]
    c = np.arange(1, n)
    excess = (n1 - below) / n1 - ((n - c) - (n1 - below)) / n0
    z = excess / np.sqrt(c * (n - c) / (n1 * n0 * (n - 1)))
    distinct = np.diff(values, axis=1) > TIE_RTOL * np.abs(values[:, 1:])
    return values, below, np.where(distinct, z, np.nan)


def _median_of_sorted_slice(sorted_rows, start, count):
    """Median of sorted_rows[g, start[g]:start[g]+count[g]] per row; NaN where count is 0."""
    rows = np.arange(len(sorted_rows))
    safe = np.maximum(count, 1)
    lo = sorted_rows[rows, np.minimum(start + (safe - 1) // 2, sorted_rows.shape[1] - 1)]
    hi = sorted_rows[rows, np.minimum(start + safe // 2, sorted_rows.shape[1] - 1)]
    return np.where(count > 0, (lo + hi) / 2, np.nan)


def subsets(x, cond):
    """Each gene's learned subsets, up and down: boundary cut, number of cases, fold.

    The upward subset is every case above the cut where z is largest, choosing
    the highest such cut on ties; the downward subset is every case below the
    cut where z is smallest, choosing the lowest. A subset must sit on its own
    side of the controls: the upward one's median above the control median, the
    downward one's below; otherwise, as when z's extreme is a noise peak at the
    end of the ranking, it is empty. The fold is the subset median minus the
    control median. Returns (up_cut, up_cases, up_fold, down_cut, down_cases,
    down_fold, values); a cut c splits values[:, :c] from values[:, c:].
    """
    cond = np.asarray(cond)
    values, below, z = walk(x, cond)
    n1 = int(cond.sum())
    rows = np.arange(len(z))
    high, low = np.where(np.isnan(z), -np.inf, z), np.where(np.isnan(z), np.inf, z)
    up = z.shape[1] - 1 - np.argmax(high[:, ::-1], axis=1)      # the highest cut among maxima
    down = np.argmin(low, axis=1)                                # the lowest cut among minima
    up_cases = np.where(high[rows, up] > 0, n1 - below[rows, up], 0)
    down_cases = np.where(low[rows, down] < 0, below[rows, down], 0)
    case = np.sort(np.asarray(x, float)[:, cond == 1], axis=1)
    control = np.median(np.asarray(x, float)[:, cond == 0], axis=1)
    up_fold = _median_of_sorted_slice(case, n1 - up_cases, up_cases) - control
    down_fold = _median_of_sorted_slice(case, np.zeros_like(down_cases), down_cases) - control
    up_cases = np.where(up_fold > 0, up_cases, 0)
    down_cases = np.where(down_fold < 0, down_cases, 0)
    return (up + 1, up_cases, np.where(up_cases > 0, up_fold, np.nan),
            down + 1, down_cases, np.where(down_cases > 0, down_fold, np.nan), values)


def describe(expression, cond):
    """Bulk shift and learned subsets on log2 expression, one row per gene.

    Returns bulk_shift (difference in medians), then the fraction of cases in
    the upward subset and its fold over the control median, and the same two
    for the downward subset.
    """
    cond = np.asarray(cond)
    bulk = np.median(expression[:, cond == 1], axis=1) - np.median(expression[:, cond == 0], axis=1)
    _, up_cases, up_fold, _, down_cases, down_fold, _ = subsets(expression, cond)
    n1 = int(cond.sum())
    return bulk, up_cases / n1, up_fold, down_cases / n1, down_fold
