"""The subset: the cases that depart from the one fold every case would share, and its test.

Under the null every case carries one common fold of the controls' distribution, on count + 1.
The two groups' Lorenz curves of count + 1 then coincide. The walk (rust/src/lib.rs) finds the
largest gap by which the cases' curve falls below the controls': how much more concentrated
expression is among the cases. Its cut splits the cases in two, and the smaller side, which
departs further per case, is the subset: above the rest of the cases or below them.

The test pools the controls with the cases registered by the fold that best aligns the two
groups' quantile functions of log(count + 1): the Wasserstein-1 fold, the weighted median of
their log ratio. It shuffles labels over that pool and walks each shuffle. The gap is unchanged
by a fold on the cases, so the observed statistic is the same whatever the fold; the
registration only shapes the pool. Each walk is a prefix sum, O(n) per shuffle.
"""
import numpy as np

from . import _kernel
from .permutation import STREAM_BLOCK, stream_block
from .pvalues import sequential_p, settled


def split(x, cond):
    """Each gene's subset, and its registration.

    Rows are genes of normalized expression. Returns, per gene: the subset's size (0 for none);
    its side of the cases (1 above the rest, -1 below, 0 none); its log2 fold over what the
    common fold predicts for its quantiles (NaN for none); the walk's gap (0 for none); and the
    registration, a fold on count + 1 carrying the controls onto the cases. A change every case
    carries alike has no subset. Boundaries fall only between distinct values (equal within 100
    ulps).
    """
    return _kernel.split(*_sorted(x, cond))


def walks(x, cond, shift, *, direction):
    """Each gene's registered pool, ready to be walked under any labelling, and the observed
    labelling's score.

    The cases are carried back by the gene's registration shift (a fold on count + 1) and pooled
    with the controls; the pool is sorted, with each value's sample beside it, so the walk reads
    both groups in order whatever the labels. Returns the pool, its sample order and the scores.
    """
    cond = np.ascontiguousarray(cond, dtype=np.uint8)
    pooled = np.where(cond == 1, _carry(x, shift[:, None], undo=True), x)
    order = np.ascontiguousarray(np.argsort(pooled, axis=1, kind="stable"), dtype=np.int64)
    pooled = np.ascontiguousarray(np.take_along_axis(pooled, order, axis=1))
    return pooled, order, _kernel.score(pooled, order, direction, cond)


def advance(pooled, order, observed, cond, exceed, tried, *, direction, stop, maximum, threshold, seed):
    """Walk the run's stream of shuffles for every gene not yet settled at threshold.

    Each gene resumes at its own position, the number of shuffles it has tried, and stops at its
    stop-th exceedance, at maximum shuffles, or once its Besag–Clifford bound reaches threshold
    (pvalues.settled). direction "both" counts the shuffles whose gap is at least the observed one;
    "up" counts those with a subset of cases above the rest at least as far out; "down" the
    reverse, a subset on the other side scoring as a negative gap. The records exceed and tried are
    updated in place.
    """
    cond = np.ascontiguousarray(cond, dtype=np.uint8)
    while (live := ~settled(exceed, tried, stop=stop, maximum=maximum, threshold=threshold)).any():
        block = int(tried[live].min()) // STREAM_BLOCK
        exceed[:], tried[:] = _kernel.tally(pooled, order, direction, stream_block(cond, block, seed), observed,
                                            exceed, tried, stop, maximum, block * STREAM_BLOCK, float(threshold))


def subset_test(x, cond, *, direction, stop, maximum, seed):
    """Permutation p-values for each gene's subset, under the null that every case carries one common
    fold, every gene run to the end of its Besag–Clifford rule: its stop-th exceedance, or maximum
    shuffles (see advance). Each shuffle draws new labels over the registered pool and walks them; a
    gene without a subset has p = 1. Returns the p-values and the exceedance and shuffle counts
    behind them.
    """
    cond = np.ascontiguousarray(cond, dtype=np.uint8)
    _, _, _, gap, shift = split(x, cond)
    ran = np.flatnonzero(gap > 0)
    pooled, order, observed = walks(x[ran], cond, shift[ran], direction=direction)
    stop = min(stop, maximum)
    exceed, tried = np.zeros(len(ran), np.uint64), np.zeros(len(ran), np.uint64)
    advance(pooled, order, observed, cond, exceed, tried, direction=direction, stop=stop, maximum=maximum,
            threshold=-np.inf, seed=seed)
    pvalue, hits, draws = np.ones(len(x)), np.zeros(len(x), np.uint64), np.zeros(len(x), np.uint64)
    hits[ran], draws[ran] = exceed, tried
    pvalue[ran] = sequential_p(exceed, tried, stop=stop, maximum=maximum)
    return pvalue, hits, draws


def _sorted(x, cond):
    """Each gene's cases and controls, sorted, as C-ordered rows for the native walk."""
    cond = np.asarray(cond)
    return (np.ascontiguousarray(np.sort(x[:, cond == 1], axis=1)),
            np.ascontiguousarray(np.sort(x[:, cond == 0], axis=1)))


def _carry(values, shift, undo=False):
    """Apply a registration to values, or remove it: a fold on count + 1."""
    return (values + 1) / shift - 1 if undo else (values + 1) * shift - 1
