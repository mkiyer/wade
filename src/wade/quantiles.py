"""Type-7 empirical quantiles, on WADE's descending probability grid.

Every WADE statistic is a function of two quantile grids, so the quantile
convention is part of the specification and not an implementation detail:
a different type changes every number the method returns and raises
nothing (``docs/implementation-notes.md`` hazard 1).

Type 7 is pinned here explicitly rather than inherited from a library
default. It is also implemented directly rather than delegated to
:func:`numpy.quantile`, for two reasons:

1. **R's arithmetic is not NumPy's.** R's ``quantile.default`` evaluates
   the interpolation as ``(1 - h) * x[lo] + h * x[hi]``, while NumPy's
   ``method="linear"`` uses a two-sided lerp that switches formula at
   ``t >= 0.5``. Both are correct type 7; they differ in the last bits.
   Reproducing R's form removes that difference at the source instead of
   absorbing it into a tolerance.

2. **R skips the interpolation when it cannot matter, and that is
   load-bearing on tied data.** ``quantile.default`` only interpolates
   where ``index > lo`` *and* ``x[hi] != x[lo]``, returning ``x[lo]``
   untouched otherwise. On a run of equal values ``(1 - h) * a + h * a``
   is not guaranteed to be exactly ``a`` in floating point, so without
   the guard a tie could shift by an ulp. WADE's target regime is
   zero-heavy count data, which is nothing but ties.

There is one definition here and both the observed path and the
permutation null call it, so the two cannot drift apart.
"""

from __future__ import annotations

import numpy as np

__all__ = ["probability_grid", "tail_window_size", "type7_quantiles"]


def probability_grid(nprobs: int) -> np.ndarray:
    """The ``nprobs`` probabilities WADE compares the two groups on.

    Descending, from 1 down to 0 — R's ``seq(1, 0, length.out = nprobs)``.
    **The order is load-bearing.** Position 0 is the group maximum, so the
    first ``k`` positions are the upper tail where a rare high-expressing
    subset lives. An ascending grid with the same ``[:k]`` slice computes a
    lower-tail statistic and calls it ``tail_mean``, with no error
    (``docs/implementation-notes.md`` hazard 9).

    ``nprobs = 1`` gives the single probability 1.0, matching R, which
    ``seq(1, 0, length.out = 1)`` returns.
    """
    if nprobs < 1:
        raise ValueError(f"nprobs must be >= 1, got {nprobs}")
    if nprobs == 1:
        return np.array([1.0])
    return np.linspace(1.0, 0.0, nprobs)


def tail_window_size(nprobs: int, tail_q: float) -> int:
    """``k = max(1, ceil(tail_q * nprobs))`` — the upper-tail window.

    The ceiling and the floor of 1 together mean the statistic is always
    computable, including in regimes where it should not be believed: at
    ``nprobs <= 10`` and the default ``tail_q = 0.10`` the "tail mean" is a
    single order statistic. See ``docs/limits.md`` section 3.
    """
    if not (0.0 < tail_q <= 1.0):
        raise ValueError(f"tail_q must lie in (0, 1], got {tail_q}")
    return max(1, int(np.ceil(tail_q * nprobs)))


def type7_quantiles(x: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Row-wise type-7 quantiles of ``x`` (genes x samples) at ``probs``.

    Returns a ``(genes, len(probs))`` array. Mirrors R's
    ``quantile.default(type = 7)`` term for term, including its
    no-interpolation guard.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D genes x samples array, got shape {x.shape}")
    probs = np.asarray(probs, dtype=np.float64)
    n = x.shape[1]
    if n == 0:
        raise ValueError("cannot take quantiles of an empty group")

    xs = np.sort(x, axis=1)

    # R: index <- 1 + (n - 1) * probs, with lo/hi as 1-based positions.
    # Kept 1-based through the floor/ceil so the arithmetic matches R's
    # exactly, then shifted to 0-based only for the actual indexing.
    index = 1.0 + (n - 1) * probs
    lo = np.floor(index)
    hi = np.ceil(index)
    lo_i = lo.astype(np.intp) - 1
    hi_i = hi.astype(np.intp) - 1

    lo_val = xs[:, lo_i]
    hi_val = xs[:, hi_i]
    h = index - lo

    # R interpolates only where it can change the answer.
    interp = (h > 0.0) & (hi_val != lo_val)
    return np.where(interp, (1.0 - h) * lo_val + h * hi_val, lo_val)
