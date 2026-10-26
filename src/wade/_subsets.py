"""The subset: the cases that depart from the one fold every case would share, and its test.

Under the null every case carries one common fold of the controls' distribution. The two groups'
Lorenz curves of count + 1 then coincide. The walk (rust/src/lib.rs) finds the largest gap by
which the cases' curve falls below the controls': how much more concentrated expression is among
the cases. Its cut splits the cases in two, and the smaller side, which departs further per case,
is the subset: above the rest of the cases or below them. The subset is read on normalized
counts (split).

The test runs on counts made exchangeable by thinning, which keeps the count model exact: a
thinned mixed-Poisson count is mixed Poisson at the lower rate, whatever the latent level
(thinned). Every sample is first thinned to the smallest size factor, as the area test thins
them, so that no difference in depth reads as a subset. The fold that best aligns the two groups'
quantile functions of log(count + 1), the Wasserstein-1 fold of those equalized counts, is then
taken out of the higher group by thinning it further, rather than by rescaling it: at low counts
a rescaling changes levels but not Poisson noise or zeros, and the extra noise reads as
concentration. The test shuffles labels over the thinned pool and walks each shuffle. Each walk
is a prefix sum, O(n) per shuffle.
"""
import numpy as np

from . import _kernel
from ._means import equalized, expression, jittered, stream
from .permutation import STREAM_BLOCK, stratum_columns, stream_block
from .pvalues import sequential_p, settled

# The thinning draws the subset test averages its walk over. On TCGA prostate two seeds agreed on 90%
# of its calls at 128 reads and above with 4, against 81% with one; each draw costs one walk per shuffle
# (research/subset_pooled.py).
DRAWS = 4


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


def thinned(counts, factors, cond, rows, seed, code=None, draws=1):
    """The values the subset test shuffles, and the fold it takes out, gene by gene.

    counts are raw (one row per gene, rows its row indices for the seeded draws), factors the
    normalization's size factors (one per sample, or one row per gene). Every sample is thinned to the
    smallest size factor (with strata, its stratum's smallest), the area test's own draw. The fold is
    the Wasserstein-1 fold on count + 1 of those jittered counts; with strata, the weighted median of
    the strata's own folds, weights n1s n0s / ns. The higher group is then thinned by that fold, the
    cases by 1/fold where it is above 1, the controls by the fold where it is below, each gene from its
    own stream (seed, row index, 3), and the counts are jittered. With strata, each stratum is finally
    moved, by a fold on count + 1, to the cohort's median log(count + 1): within a stratum every sample
    moves alike, so no label is favoured. Every stratum must hold both groups.

    With draws > 1, each draw repeats all of this from its own streams (_means.stream): draw 0 is the
    one above, and each gene's draws are consecutive rows (genes x draws rows), with a fold for each
    (genes x draws).
    """
    cond = np.ascontiguousarray(cond, dtype=np.uint8)
    pools, folds = [], []
    for draw in range(draws):
        level = equalized(counts, rows, factors, seed, code, draw)
        ones = np.ones(level.shape[1])
        x = expression(jittered(level, rows, seed, draw), ones)
        if code is None:
            fold = _kernel.split(*_sorted(x, cond))[4]
        else:
            fold = np.exp(_weighted_median(*_stratum_folds(x, cond, code)))
        share = np.where(cond == 1, np.where(fold > 1, 1 / fold, 1.0)[:, None], np.where(fold < 1, fold, 1.0)[:, None])
        out = np.empty(level.shape, np.int64)
        for i, (row, index) in enumerate(zip(level, rows)):
            out[i] = stream(seed, index, 3, draw).binomial(row, share[i])
        pool = expression(jittered(out, rows, seed, draw), ones)
        if code is not None:
            logs = np.log1p(pool)
            centre = np.median(logs, axis=1, keepdims=True)
            for idx in stratum_columns(code):
                logs[:, idx] += centre - np.median(logs[:, idx], axis=1, keepdims=True)
            pool = np.expm1(logs)
        pools.append(pool)
        folds.append(fold)
    if draws == 1:
        return pools[0], folds[0]
    return np.stack(pools, axis=1).reshape(-1, len(cond)), np.stack(folds, axis=1)


def _stratum_folds(x, cond, code):
    """Each stratum's own log W1 fold (genes x strata) and its weight n1s n0s / ns."""
    columns = stratum_columns(code)
    folds = np.empty((len(x), len(columns)))
    weights = np.empty(len(columns))
    for i, idx in enumerate(columns):
        n1 = int(cond[idx].sum())
        weights[i] = n1 * (len(idx) - n1) / len(idx)
        folds[:, i] = np.log(_kernel.split(*_sorted(x[:, idx], cond[idx]))[4])
    return folds, weights


def _weighted_median(values, weights):
    """Each row's weighted median of values (genes x strata); where the weights below a value make
    exactly half, the midpoint to the next, as the registration's own median."""
    order = np.argsort(values, axis=1, kind="stable")
    ranked = np.take_along_axis(values, order, axis=1)
    below = np.cumsum(weights[order], axis=1)
    total = weights.sum()
    rows = np.arange(len(values))
    first = np.argmax(2 * below >= total * (1 - 1e-12), axis=1)       # the first value at or past half the weight
    half = np.abs(2 * below[rows, first] - total) <= 1e-12 * total
    after = ranked[rows, np.minimum(first + 1, values.shape[1] - 1)]
    return np.where(half, 0.5 * (ranked[rows, first] + after), ranked[rows, first])


def aligned(x, cond, code):
    """The subset test's pool with strata: one fold for every case, and a level for each stratum.

    Every stratum must hold both groups. The fold is the weighted median of the strata's own
    Wasserstein-1 folds on count + 1, with weights n1s n0s / ns; where the weights below a value make
    exactly half, it is the midpoint, as the registration's own median. Each stratum's level is the
    median of its log(count + 1) once that fold is taken out of its cases, so it rests on all the
    stratum's samples and no group is centred on a few of its members. Each stratum is moved, by a
    fold on count + 1, to the cohort's level, and the cases are carried back by the fold. The null is
    that within each stratum every case carries one fold of the stratum's controls, the same in every
    stratum; strata may differ in level. Returns the pool and the fold, gene by gene.
    """
    cond = np.ascontiguousarray(cond, dtype=np.uint8)
    columns = stratum_columns(code)
    beta = _weighted_median(*_stratum_folds(x, cond, code))
    logs = np.log1p(x)
    adjusted = logs - beta[:, None] * cond
    centre = np.median(adjusted, axis=1, keepdims=True)         # the cohort's level
    shift = np.empty_like(logs)
    for idx in columns:
        shift[:, idx] = centre - np.median(adjusted[:, idx], axis=1, keepdims=True)
    level = np.expm1(logs + shift)
    fold = np.exp(beta)
    return np.where(cond == 1, _carry(level, fold[:, None], undo=True), level), fold


def walks(x, cond, shift, *, direction, draws=1):
    """Each gene's registered pool, ready to be walked under any labelling, and the observed
    labelling's score.

    The cases are carried back by the gene's registration shift (a fold on count + 1) and pooled
    with the controls; the pool is sorted, with each value's sample beside it, so the walk reads
    both groups in order whatever the labels. Returns the pool, its sample order and the scores.
    With draws > 1, x holds each gene's draws in consecutive rows (a shift for each), and a gene's
    score is the mean of its draws' scores.
    """
    cond = np.ascontiguousarray(cond, dtype=np.uint8)
    pooled = np.where(cond == 1, _carry(x, shift[:, None], undo=True), x)
    order = np.ascontiguousarray(np.argsort(pooled, axis=1, kind="stable"), dtype=np.int64)
    pooled = np.ascontiguousarray(np.take_along_axis(pooled, order, axis=1))
    return pooled, order, _kernel.score(pooled, order, direction, cond, draws)


def advance(pooled, order, observed, cond, exceed, tried, *, direction, stop, maximum, threshold, seed, code=None,
            draws=1):
    """Walk the run's stream of shuffles for every gene not yet settled at threshold.

    Each gene resumes at its own position, the number of shuffles it has tried, and stops at its
    stop-th exceedance, at maximum shuffles, or once its Besag–Clifford bound reaches threshold
    (pvalues.settled). direction "both" counts the shuffles whose gap is at least the observed one;
    "up" counts those with a subset of cases above the rest at least as far out; "down" the
    reverse, a subset on the other side scoring as a negative gap. The records exceed and tried are
    updated in place. With strata (code), labels are shuffled within strata. With draws > 1 (walks),
    each shuffle's score is the mean of the gene's draws'.
    """
    cond = np.ascontiguousarray(cond, dtype=np.uint8)
    while (live := ~settled(exceed, tried, stop=stop, maximum=maximum, threshold=threshold)).any():
        block = int(tried[live].min()) // STREAM_BLOCK
        exceed[:], tried[:] = _kernel.tally(pooled, order, direction, stream_block(cond, block, seed, code), observed,
                                            exceed, tried, stop, maximum, block * STREAM_BLOCK, float(threshold),
                                            draws)


def subset_test(counts, factors, cond, rows, *, direction, stop, maximum, seed, code=None, draws=DRAWS):
    """Permutation p-values for each gene's subset, under the null that every case carries one common
    fold, as wade() computes them, every gene run to the end of its Besag–Clifford rule: its stop-th
    exceedance, or maximum shuffles (see advance). counts are raw, rows their gene row indices,
    factors the size factors. A gene without a subset on its normalized counts (the subset wade()
    reports) has p = 1; every other gene's shuffles draw new labels over its thinned pools (thinned,
    `draws` of them) and walk each, scoring the mean of the walks. With strata (code; every stratum
    holding both groups), labels are shuffled within strata. Returns the p-values and the exceedance
    and shuffle counts behind them.
    """
    cond = np.ascontiguousarray(cond, dtype=np.uint8)
    rows = np.asarray(rows)
    described = expression(jittered(counts, rows, seed), factors)       # the subset as wade() reads it
    if code is not None:
        described, _ = aligned(described, cond, code)
    ran = np.flatnonzero(split(described, cond)[3] > 0)
    own = factors[ran] if np.ndim(factors) == 2 else factors
    x, _ = thinned(counts[ran], own, cond, rows[ran], seed, code, draws)
    pooled, order, observed = walks(x, cond, np.ones(len(x)), direction=direction, draws=draws)
    stop = min(stop, maximum)
    exceed, tried = np.zeros(len(ran), np.uint64), np.zeros(len(ran), np.uint64)
    advance(pooled, order, observed, cond, exceed, tried, direction=direction, stop=stop, maximum=maximum,
            threshold=-np.inf, seed=seed, code=code, draws=draws)
    genes = len(counts)
    pvalue, hits, draws = np.ones(genes), np.zeros(genes, np.uint64), np.zeros(genes, np.uint64)
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
