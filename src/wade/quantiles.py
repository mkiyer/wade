"""Type-7 empirical quantiles, on WADE's descending probability grid.

Every WADE statistic is a function of two quantile grids, so the quantile
convention is part of the specification: a different type changes every
number the method returns and raises nothing. Type 7 is pinned here
explicitly, and implemented directly rather than delegated to
:func:`numpy.quantile`, so that two details are fixed:

1. The interpolation is the one-sided ``(1 - h) * x[lo] + h * x[hi]``.
   NumPy's ``method="linear"`` uses a two-sided lerp that switches formula
   at ``t >= 0.5``; both are correct type 7 and they differ in the last bits.
2. **Interpolation is skipped where it cannot matter**: only where
   ``h > 0`` *and* ``x[hi] != x[lo]``. On a run of equal values
   ``(1 - h) * a + h * a`` is not guaranteed to be exactly ``a`` in floating
   point, and WADE's target regime is zero-heavy count data, which is
   nothing but ties.

The compiled kernel reimplements the same definition term for term and is
held to it bitwise. There is one Python definition here, and both the
observed path and the permutation null call it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["probability_grid", "type7_quantiles", "capped_nprobs"]


def capped_nprobs(n1: int, n0: int, max_probs: int | None) -> int:
    """The realized grid size: ``min(n1, n0)``, capped at ``max_probs``.

    ``max_probs`` (``docs/method.md`` §1) bounds the grid so that per-gene
    storage stops growing with the cohort: nothing needs an affected fraction
    resolved to 1/40,000, and every ``(genes x m)`` array scales with the
    answer here. ``None`` means the design's full grid. The realized value is
    recorded in the result and the manifest, never applied silently.
    """
    m = min(int(n1), int(n0))
    if max_probs is None:
        return m
    max_probs = int(max_probs)
    if max_probs < 2:
        raise ValueError(f"max_probs must be at least 2, got {max_probs}")
    return min(m, max_probs)


def probability_grid(nprobs: int) -> np.ndarray:
    """The ``nprobs`` probabilities WADE compares the two groups on.

    Descending, from 1 down to 0. **The order is load-bearing**: position 0
    is the group maximum, so the first ``k`` positions are the upper tail
    where a rare high-expressing subset lives, and an ascending grid would
    silently compute a lower-tail statistic. ``nprobs = 1`` gives the single
    probability 1.0.
    """
    if nprobs < 1:
        raise ValueError(f"nprobs must be >= 1, got {nprobs}")
    if nprobs == 1:
        return np.array([1.0])
    return np.linspace(1.0, 0.0, nprobs)


def type7_quantiles(x: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Row-wise type-7 quantiles of ``x`` (genes x samples) at ``probs``.

    Returns a ``(genes, len(probs))`` array.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D genes x samples array, got shape {x.shape}")
    probs = np.asarray(probs, dtype=np.float64)
    n = x.shape[1]
    if n == 0:
        raise ValueError("cannot take quantiles of an empty group")

    xs = np.sort(x, axis=1)

    # 1-based positions through the floor/ceil (the type-7 definition),
    # shifted to 0-based only for the indexing.
    index = 1.0 + (n - 1) * probs
    lo = np.floor(index)
    hi = np.ceil(index)
    lo_i = lo.astype(np.intp) - 1
    hi_i = hi.astype(np.intp) - 1

    lo_val = xs[:, lo_i]
    hi_val = xs[:, hi_i]
    h = index - lo

    # Interpolate only where it can change the answer.
    interp = (h > 0.0) & (hi_val != lo_val)

    return np.where(interp, (1.0 - h) * lo_val + h * hi_val, lo_val)
