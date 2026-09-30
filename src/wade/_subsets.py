"""The subset: the samples that depart from the one fold every case would share, and its test.

Under the null every case carries one common fold of the controls' distribution, on count + 1
(on the log2 scale, one common difference). The two groups' Lorenz curves of count + 1 then
coincide (on log2, their absolute Lorenz curves). The walk (rust/src/lib.rs) finds the largest
gap by which the cases' curve falls below the controls': how much more concentrated expression
is among the cases. Its cut splits the cases in two, and the smaller side, which departs
further per case, is the subset: above the rest of the cases or below them. Walking the
controls against the cases as well finds the reverse, a subset among the controls, so that a
test of both groups asks whether the two shapes differ at all.

The test pools the controls with the cases registered by the fold that best aligns the two
groups' quantile functions of log(count + 1): the Wasserstein-1 fold, the weighted median of
their log ratio. It shuffles labels over that pool and walks each shuffle. The gap is unchanged
by a fold on the cases, so the observed statistic is the same whatever the fold; the
registration only shapes the pool. Each walk is a prefix sum, O(n) per shuffle.
"""
import numpy as np

from . import _kernel
from .permutation import draw_perms
from .pvalues import sequential_p


def split(x, cond, scale, *, both_groups=False, direction="both"):
    """Each gene's subset, and its registration.

    Rows are genes on the tested scale. Returns, per gene: the subset's size (0 for none); its
    side of its group (1 above the rest, -1 below, 0 none); its group (1 cases, 0 controls);
    its log2 fold over what the common fold predicts for its quantiles (NaN for none); the
    walk's gap (0 for none); and the registration, a fold on count + 1 (a difference on log2)
    carrying the controls onto the cases. A change every case carries alike has no subset.
    Boundaries fall only between distinct values (equal within 100 ulps). With both_groups the
    controls are walked against the cases too, and the subset scoring higher for direction is
    kept (see subset_test).
    """
    return _kernel.split(*_sorted(x, cond), scale == "linear", both_groups, direction)


def subset_test(x, cond, scale, *, direction, stop, maximum, seed, both_groups=False, strata=None, batch=4096):
    """Permutation p-values for each gene's subset, under the null that every case carries one common fold.

    Each shuffle draws new labels over the registered pool and walks them. direction "both"
    counts the shuffles whose gap is at least the observed one. "up", which asks whether the
    cases are higher, counts those with a subset of cases above the rest of the cases at least
    as far out; with both_groups, a subset of controls below the rest of the controls also
    counts. "down" is the reverse. A subset on the other side scores as a negative gap. Shuffles
    stop at the stop-th exceedance or at maximum (Besag-Clifford), and a gene without a subset
    has p = 1. Returns the p-values and the exceedance and shuffle counts behind them.
    """
    cond = np.ascontiguousarray(cond, dtype=np.uint8)
    linear = scale == "linear"
    _, _, _, _, gap, shift = split(x, cond, scale, both_groups=both_groups, direction=direction)
    ran = np.flatnonzero(gap > 0)
    pooled = np.where(cond == 1, _carry(x[ran], shift[ran, None], scale, undo=True), x[ran])
    order = np.ascontiguousarray(np.argsort(pooled, axis=1, kind="stable"), dtype=np.int64)
    pooled = np.ascontiguousarray(np.take_along_axis(pooled, order, axis=1))
    observed = _kernel.score(pooled, order, linear, direction, both_groups, cond)
    stop = min(stop, maximum)
    exceed, tried = np.zeros(len(ran), np.uint64), np.zeros(len(ran), np.uint64)
    rng, size = np.random.default_rng(seed), 32
    while (live := (exceed < stop) & (tried < maximum)).any():
        labels = draw_perms(cond, int(min(size, maximum - tried[live].max())), rng=rng, strata=strata)
        exceed, tried = _kernel.tally(pooled, order, linear, direction, both_groups, labels, observed,
                                      exceed, tried, stop, maximum)
        size = min(2 * size, batch)
    pvalue, hits, draws = np.ones(len(x)), np.zeros(len(x), np.uint64), np.zeros(len(x), np.uint64)
    hits[ran], draws[ran] = exceed, tried
    pvalue[ran] = sequential_p(exceed, tried, stop=stop, maximum=maximum)
    return pvalue, hits, draws


def _sorted(x, cond):
    """Each gene's cases and controls, sorted, as C-ordered rows for the native walk."""
    cond = np.asarray(cond)
    return (np.ascontiguousarray(np.sort(x[:, cond == 1], axis=1)),
            np.ascontiguousarray(np.sort(x[:, cond == 0], axis=1)))


def _carry(values, shift, scale, undo=False):
    """Apply a registration to values, or remove it: a fold on count + 1, or a difference on the log2 scale."""
    if scale == "log2":
        return values - shift if undo else values + shift
    return (values + 1) / shift - 1 if undo else (values + 1) * shift - 1
