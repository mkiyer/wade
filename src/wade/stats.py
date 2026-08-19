"""The quantile grid and the mean-shift statistic — ``docs/method.md`` §1–2.

WADE compares two groups on a shared grid of ``min(n_case, n_ctrl)``
probabilities and reads two curves off it: the absolute difference ``D``,
which is what detection is powerful on, and the log-ratio ``R``, which is
where the difference's *shape* is legible. This module owns the grid and
the first; :mod:`wade.subset` owns the second.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .quantiles import probability_grid, type7_quantiles

__all__ = ["WadeStats", "split_groups", "wade_stats"]


def split_groups(cond: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Case and control column indices from a binary condition vector.

    ``1`` marks a case, ``0`` a control, and both groups must be non-empty.

    R assigns membership by exact equality to 1 and 0, so a sample labelled
    anything else is silently dropped from *both* groups — which changes the
    grid size, which changes every number, with no diagnostic. This raises.
    """
    cond = np.asarray(cond)
    if cond.ndim != 1:
        raise ValueError(f"cond must be a 1-D vector, got shape {cond.shape}")
    if not np.all(np.isin(cond, (0, 1))):
        bad = np.unique(cond[~np.isin(cond, (0, 1))])
        raise ValueError(
            f"cond must contain only 0 (control) and 1 (case); found {bad.tolist()}."
        )
    i1 = np.flatnonzero(cond == 1)
    i0 = np.flatnonzero(cond == 0)
    if i1.size == 0 or i0.size == 0:
        raise ValueError(
            f"both groups must be non-empty; got {i1.size} cases and {i0.size} controls"
        )
    return i1, i0


@dataclass(frozen=True)
class WadeStats:
    """The quantile grids and the per-gene quantities read off them."""

    q: np.ndarray            # (m,)   descending probability grid, 1 -> 0
    nprobs: int              #        m = min(n0, n1); a property of the design
    n1: int
    n0: int
    Q1: np.ndarray           # (g, m) case quantile grid
    Q0: np.ndarray           # (g, m) control quantile grid
    D: np.ndarray            # (g, m) Q1 - Q0
    mean_shift: np.ndarray   # (g,)   signed grid area between the two curves
    w1: np.ndarray           # (g,)   1-Wasserstein distance
    fc: np.ndarray           # (g,)   ratio of the two grid means
    case_mean: np.ndarray    # (g,)
    ctrl_mean: np.ndarray    # (g,)

    @property
    def log2_fc(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log2(self.fc)


def wade_stats(
    x: np.ndarray,
    cond: np.ndarray,
    *,
    allow_single_sample_group: bool = False,
) -> WadeStats:
    """Quantile grids and the mean-shift statistic, from a normalized matrix.

    ``mean_shift`` is the signed area between the two quantile functions. When
    the groups are equal-sized it is *exactly* the difference of the two sample
    means; on unequal groups it is a grid quadrature that over-weights the
    larger group's extremes. It is named plainly because that is what it is —
    the test any conventional DE method already performs. WADE's claim is to
    add sensitivity it lacks, not to improve on it.

    On ``min(n0, n1) == 1`` the grid collapses to the single probability 1, so
    every statistic reduces to the difference of group maxima. Refused by
    default; a one-sample group has no quantile function worth comparing.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D genes x samples array, got shape {x.shape}")
    cond = np.asarray(cond)
    if cond.shape[0] != x.shape[1]:
        raise ValueError(
            f"cond has {cond.shape[0]} entries but the matrix has {x.shape[1]} samples"
        )

    i1, i0 = split_groups(cond)
    n1, n0 = int(i1.size), int(i0.size)
    nprobs = min(n0, n1)
    if nprobs == 1 and not allow_single_sample_group:
        raise ValueError(
            "min(n_case, n_ctrl) == 1: the quantile grid collapses to the single "
            "point p = 1, so every statistic reduces to the difference of group "
            "maxima. Pass allow_single_sample_group=True to compute it anyway."
        )

    q = probability_grid(nprobs)
    Q1 = type7_quantiles(x[:, i1], q)
    Q0 = type7_quantiles(x[:, i0], q)
    D = Q1 - Q0
    s1 = Q1.sum(axis=1)
    s0 = Q0.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        fc = s1 / s0

    return WadeStats(
        q=q, nprobs=nprobs, n1=n1, n0=n0, Q1=Q1, Q0=Q0, D=D,
        mean_shift=D.sum(axis=1) / nprobs,
        w1=np.abs(D).mean(axis=1),
        fc=fc, case_mean=s1 / nprobs, ctrl_mean=s0 / nprobs,
    )
