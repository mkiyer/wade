"""Single-gene diagnostic: the quantile pair and the cumulative signed area.

R's ``wade_gene()``. The downstream reading of this panel is what makes
the statistic legible — a broad location shift accumulates steadily
across all quantiles, a rare high-expressing subset stays flat and then
climbs inside the tail window, and a single-outlier gene stays flat and
then spikes at the last node. The second and third are not
distinguishable from this curve alone; separating them is the permutation
p-value's job.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stats import DEFAULT_TAIL_Q, wade_stats

__all__ = ["GeneDetail", "wade_gene"]


@dataclass(frozen=True)
class GeneDetail:
    """One gene's grid, in **ascending** probability order."""

    p: np.ndarray      # (m,) ascending, 0 -> 1
    y1: np.ndarray     # (m,) case quantiles, reordered to match p
    y0: np.ndarray     # (m,) control quantiles
    cum: np.ndarray    # (m,) cumulative signed area; cum[-1] == diff_mean
    nprobs: int
    k: int


def wade_gene(
    row: np.ndarray,
    cond: np.ndarray,
    tail_q: float = DEFAULT_TAIL_Q,
    log2_scale: bool = False,
    weight: float = 1.0,
    *,
    allow_single_sample_group: bool = False,
) -> GeneDetail:
    """Per-gene detail from one **already-normalized** row.

    Takes normalized values, not counts, and not the matrix.

    Why the reversal happens *after* the statistic, not before
    ----------------------------------------------------------
    Two reasons, and only the second is about floating point.

    The statistics must be computed on the **descending** grid, because
    the tail is defined as the first ``k`` positions and that is the upper
    tail only while the grid runs 1 to 0. Reversing first would make
    ``[:k]`` select the *lowest* ``k`` probabilities and measure the
    bottom of the distribution while still calling it ``tail_mean``. The
    reversal is for display only, and doing it in this order is what keeps
    the plotted curve consistent with the number the test used.

    Second, reversing changes the summation order. The curve is built so
    that ``cum[-1] == diff_mean``, and in exact arithmetic that is
    trivially true — but ``diff_mean`` sums ``D`` forwards while the
    endpoint accumulates it backwards. Measured in the R over 2,000 random
    rows, the two differ bitwise in about two thirds of genes, at a worst
    relative difference of 3.5e-14. **So the endpoint identity is a
    tolerance assertion, not an exact one.** It is still the cheapest
    correctness test the implementation has and it is asserted in the test
    suite.

    Note the division is by ``nprobs`` — the full grid size — at every
    point, not by the number of terms accumulated so far.
    """
    row = np.asarray(row, dtype=np.float64)
    if row.ndim != 1:
        raise ValueError(f"expected a single gene's row (1-D), got shape {row.shape}")

    st = wade_stats(
        row[None, :], cond, tail_q=tail_q, log2_scale=log2_scale, weight=weight,
        allow_single_sample_group=allow_single_sample_group,
    )
    y1 = st.Q1[0][::-1]
    y0 = st.Q0[0][::-1]
    return GeneDetail(
        p=st.q[::-1].copy(),
        y1=y1.copy(),
        y0=y0.copy(),
        cum=np.cumsum(y1 - y0) / st.nprobs,
        nprobs=st.nprobs,
        k=st.k,
    )
