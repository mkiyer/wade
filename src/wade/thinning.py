"""The count-native shift correction for the subset stage — ``docs/method.md`` §10.3–10.4.

Stage 2 tests "is a global fold change an adequate explanation?" by first
making the two groups look identical *under that hypothesis*, then shuffling
labels. For continuous data that was a division: divide the case values by the
fitted fold change and the groups are exchangeable if the shift is the whole
story. For counts at low expression a division does not do that — a zero does
not divide, and a fold change in a negative-binomial mean is not a
multiplicative shift of the negative-binomial distribution — and the result,
measured, was a test that called genuine 2× shifts "not a global shift" most of
the time below a few counts and most of the time at 20 counts with large
cohorts.

The count-native operation is **binomial thinning**: keep each read of the
higher group with probability ``1/f``. That is what a sample would have looked
like at ``1/f`` of its sequencing depth, it maps NB(``f·μ``, ``φ``) to
NB(``μ``, ``φ``) exactly (Poisson–Gamma is closed under thinning), zeros stay
zeros and integers stay integers. This module owns three pieces:

* :func:`fit_fold_change` — ``f`` as *the thinning factor that makes the two
  groups' interquartile means agree*. It is unbiased under an NB fold change
  by construction (it asks "after thinning, do the middles agree?", assuming
  nothing about how the middle scales) and out of reach of any subset below
  25%. Two simpler estimators were measured and rejected: the interquartile
  mean ratio is biased for skewed counts (2.08 for a true 2), and the mean
  ratio is moved to 1.35 by a 5% subset at 8×. Accuracy matters because an
  inaccurate ``f`` on a true global shift is *anti*-conservative (30% off →
  0.26–0.28 false subsets at 200 v 200).
* :func:`thin_counts` — the thinning itself, drawn **once**, before any
  permutation, like the jitter; inference is conditional on the draw.
* :func:`one_count` — the pseudocount for the log-ratio curve: one count,
  expressed in each sample's normalized units. A count of zero means "less
  than one"; without this the tie-breaking jitter turns it into a log-ratio
  of 7–10 against any nonzero count, and one such node can halve
  ``affected_fraction`` at any sample size.

Everything here works on the **raw count matrix**, which is why WADE takes
one. :func:`wade.wade_from_matrix` has no counts to thin and keeps the
division.
"""

from __future__ import annotations

import numpy as np

from .stats import split_groups

__all__ = ["midmean", "fit_fold_change", "thin_counts", "one_count"]

#: Bisection range for the fitted fold change, in natural log. 1024× covers
#: anything a global shift plausibly is; a gene absent from one group pins at
#: the bound and is a stage-1 finding, not a stage-2 one.
MAX_LOG_FOLD = float(np.log(1024.0))


def midmean(x: np.ndarray) -> np.ndarray:
    """Row-wise mean of the middle half of the sorted values.

    The interquartile mean: robust to anything that lives in the top or bottom
    quarter, which is where a subset of fewer than 25% of the samples lives,
    and defined where a median is not — a gene that is zero in most samples
    still has a middle half with a mean.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[1]
    lo, hi = n // 4, n - n // 4
    return np.sort(x, axis=1)[:, lo:hi].mean(axis=1)


def thin_counts(
    counts: np.ndarray,
    cond: np.ndarray,
    fold: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Binomial-thin the higher group of every gene by its fitted fold change.

    ``fold[g] >= 1`` thins the **case** columns of gene ``g`` with keep
    probability ``1/fold[g]``; ``fold[g] < 1`` thins the **control** columns
    with keep probability ``fold[g]``. Nothing is ever scaled up, and a
    ``fold`` of exactly 1 leaves the gene untouched. Returns a new
    integer-valued float matrix of the same shape.
    """
    counts = np.asarray(counts, dtype=np.float64)
    cond = np.asarray(cond)
    fold = np.asarray(fold, dtype=np.float64)
    if fold.shape != (counts.shape[0],):
        raise ValueError(f"fold must have one value per gene; got {fold.shape} for {counts.shape[0]} genes")
    if not np.all(np.isfinite(fold)) or np.any(fold <= 0):
        raise ValueError("fold must be finite and positive")
    if np.any(counts != np.round(counts)):
        raise ValueError("thinning needs integer counts; pass raw counts, not a normalized matrix")
    i1, i0 = split_groups(cond)
    keep_case = np.where(fold >= 1.0, 1.0 / fold, 1.0)
    keep_ctrl = np.where(fold < 1.0, fold, 1.0)
    out = counts.copy()
    c = counts.astype(np.int64)
    out[:, i1] = rng.binomial(c[:, i1], keep_case[:, None])
    out[:, i0] = rng.binomial(c[:, i0], keep_ctrl[:, None])
    return out


def fit_fold_change(
    counts: np.ndarray,
    cond: np.ndarray,
    normalize,
    *,
    seed: int = 0,
    iters: int = 16,
    max_log_fold: float = MAX_LOG_FOLD,
) -> np.ndarray:
    """Per gene, the thinning factor at which the two groups' interquartile
    means agree on the normalized scale.

    Bisection on ``log f`` over ``[0, max_log_fold]``, thinning the higher
    group inside the loop and re-normalizing with ``normalize`` (a callable
    from a count matrix to the normalized matrix, closed over the *original*
    library sizes so that only the thinned gene moves — and **without the
    jitter**: a hundredth of a count has no business in a fold-change
    estimate, and on an all-zero gene it would decide which group is higher
    and keep deciding it at every step). The same ``seed`` is used at every
    evaluation — common random numbers — so the objective is monotone in
    ``f`` up to the thinning algorithm's own discreteness, which sixteen
    halvings of a 10-bit range absorb.

    Returns ``f`` with the convention of :func:`thin_counts`: ``f >= 1`` means
    cases are higher and get thinned by ``1/f``; ``f < 1`` means controls are
    higher and get thinned by ``f``. A gene whose two middles are both zero
    returns ``1``: nothing to correct, and nothing to say.
    """
    counts = np.asarray(counts, dtype=np.float64)
    cond = np.asarray(cond)
    if np.any(counts != np.round(counts)):
        raise ValueError("fit_fold_change needs integer counts; pass raw counts, not a normalized matrix")
    i1, i0 = split_groups(cond)
    g = counts.shape[0]
    c = counts.astype(np.int64)
    c1, c0 = c[:, i1], c[:, i0]

    x = normalize(counts)
    m1, m0 = midmean(x[:, i1]), midmean(x[:, i0])
    up = m1 >= m0                                        # which group gets thinned
    nothing = (m1 == 0) & (m0 == 0)                      # both middles empty: f = 1

    lo = np.zeros(g)
    hi = np.full(g, float(max_log_fold))
    thinned = counts.copy()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        keep = np.exp(-mid)[:, None]
        r = np.random.default_rng(seed)
        thinned[:, i1] = np.where(up[:, None], r.binomial(c1, keep), c1)
        r = np.random.default_rng(seed)
        thinned[:, i0] = np.where(up[:, None], c0, r.binomial(c0, keep))
        xt = normalize(thinned)
        d = midmean(xt[:, i1]) - midmean(xt[:, i0])
        too_little = np.where(up, d > 0, d < 0)          # still higher: thin more
        lo = np.where(too_little, mid, lo)
        hi = np.where(too_little, hi, mid)
    f = np.exp(0.5 * (lo + hi))
    # Equal middles collapse the bracket onto 0 and f is 1; empty middles are
    # forced there, because nothing can be learned from them.
    f = np.where(nothing, 1.0, f)
    return np.where(up, f, 1.0 / f)


def one_count(
    normalizer,
    lib_sizes: np.ndarray,
    shape: tuple[int, int],
    *,
    norm_factor: float,
) -> np.ndarray:
    """The normalized value of one count, per cell: ``norm_factor / (L_gj * l_j)``.

    This is the pseudocount added before the log-ratio curve is taken, so
    that "+1" means one count in *that sample's* units rather than 1 TPM. A
    sample with a zero library size has no meaningful count scale (every
    value in it is the ``norm_factor`` artefact of ``tpm_like``) and gets 0.
    """
    from .normalize import _broadcast_normalizer

    norm = _broadcast_normalizer(normalizer, shape)
    lib = np.asarray(lib_sizes, dtype=np.float64)
    if lib.shape != (shape[1],):
        raise ValueError(f"lib_sizes must have one value per sample; got {lib.shape}")
    with np.errstate(divide="ignore", invalid="ignore"):
        pc = norm_factor / (norm * lib[None, :])
    pc = np.broadcast_to(pc, shape).copy()
    pc[:, lib == 0] = 0.0
    return pc
