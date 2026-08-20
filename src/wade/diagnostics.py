"""Single-gene diagnostic: the quantile pair and the cumulative signed area.

R's ``wade_gene()``. The downstream reading of this panel is what makes
the statistic legible — on the log-ratio curve ``r`` a global fold change
is flat, a rare high-expressing subset sits at zero and then climbs near
``p = 1``, and a single-outlier gene stays flat and then spikes at the
last node. The second and third are not distinguishable from this curve
alone; separating them is the permutation p-value's job.
:func:`wade.plot_gene` draws it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stats import wade_stats

__all__ = ["GeneDetail", "wade_gene"]


@dataclass(frozen=True)
class GeneDetail:
    """One gene's grid, in **ascending** probability order."""

    p: np.ndarray      # (m,) ascending, 0 -> 1
    y1: np.ndarray     # (m,) case quantiles, reordered to match p
    y0: np.ndarray     # (m,) control quantiles
    cum: np.ndarray    # (m,) cumulative signed area; cum[-1] == mean_shift
    r: np.ndarray      # (m,) the log-ratio curve, ascending
    nprobs: int


def wade_gene(
    row: np.ndarray,
    cond: np.ndarray,
    *,
    allow_single_sample_group: bool = False,
    pseudocount=None,
    max_probs: int | None = None,
) -> GeneDetail:
    """Per-gene detail from one **already-normalized** row.

    Takes normalized values, not counts, and not the matrix. ``pseudocount``
    (a scalar, or one value per sample) is added before the log-ratio curve
    ``r`` is taken, as the subset stage does (``docs/method.md`` §10.4); the
    quantile functions ``y1``/``y0`` and the cumulative area stay raw.
    ``max_probs`` caps the grid as the test does (``§1``); pass the run's
    value — :meth:`wade.WadeResult.gene_detail` does — so the curves here are
    the ones the statistics were read from.

    Why the reversal happens *after* the statistic, not before
    ----------------------------------------------------------
    Two reasons, and only the second is about floating point.

    The statistics are computed on the **descending** grid the test uses;
    the reversal here is for display only, so the curve reads left to right
    in increasing quantile. Doing it in this order is what keeps the plotted
    curve consistent with the numbers the test produced.

    Second, reversing changes the summation order. The curve is built so
    that ``cum[-1] == mean_shift``, and in exact arithmetic that is
    trivially true — but ``mean_shift`` sums ``D`` forwards while the
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
        row[None, :], cond, allow_single_sample_group=allow_single_sample_group,
        max_probs=max_probs,
    )
    y1 = st.Q1[0][::-1]
    y0 = st.Q0[0][::-1]
    if pseudocount is None or np.all(np.asarray(pseudocount) == 0):
        ry1, ry0 = y1, y0
    else:
        pc = np.broadcast_to(np.asarray(pseudocount, dtype=np.float64), row.shape)
        stp = wade_stats((row + pc)[None, :], cond,
                         allow_single_sample_group=allow_single_sample_group,
                         max_probs=max_probs)
        ry1, ry0 = stp.Q1[0][::-1], stp.Q0[0][::-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log2(ry1) - np.log2(ry0)
    return GeneDetail(
        p=st.q[::-1].copy(),
        y1=y1.copy(),
        y0=y0.copy(),
        cum=np.cumsum(y1 - y0) / st.nprobs,
        r=r,
        nprobs=st.nprobs,
    )
