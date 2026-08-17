"""The WADE statistic: quantile grids and the per-gene reductions.

Port of R's ``wade_stats()``. Written for clarity — this is the
correctness baseline the Rust kernel will later be validated against, so
it follows ``docs/algorithm.md`` term for term rather than being clever.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .quantiles import probability_grid, tail_window_size, type7_quantiles

__all__ = ["WadeStats", "split_groups", "wade_stats", "DEFAULT_TAIL_Q"]

DEFAULT_TAIL_Q = 0.10


def split_groups(cond: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Case and control column indices from a binary condition vector.

    ``1`` marks a case, ``0`` a control. **Both groups must be non-empty.**

    R assigns membership by exact equality to 1 and to 0, so a sample
    labelled anything else — 2, ``NA``, -1 — is silently dropped from
    *both* groups, which changes ``nprobs``, which changes the quantile
    grid, which changes every number, with no diagnostic
    (``docs/r-implementation.md`` section 3). This raises instead: the
    labels must partition the columns.
    """
    cond = np.asarray(cond)
    if cond.ndim != 1:
        raise ValueError(f"cond must be a 1-D vector, got shape {cond.shape}")
    if not np.all(np.isin(cond, (0, 1))):
        bad = np.unique(cond[~np.isin(cond, (0, 1))])
        raise ValueError(
            f"cond must contain only 0 (control) and 1 (case); found {bad.tolist()}. "
            f"R would silently drop those samples from both groups and shrink the "
            f"quantile grid without warning."
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
    """Everything ``wade_stats()`` computes for one condition vector.

    Field names are the R's with dots replaced by underscores. The grids
    are kept reachable rather than discarded: they are what localizes a
    parity failure, and what the single-gene diagnostic is built from.
    """

    q: np.ndarray            # (m,)   descending probability grid, 1 -> 0
    nprobs: int              #        m = min(n0, n1)
    k: int                   #        upper-tail window size
    n1: int
    n0: int
    Q1: np.ndarray           # (g, m) case quantile grid
    Q0: np.ndarray           # (g, m) control quantile grid
    D: np.ndarray            # (g, m) Q1 - Q0
    diff_mean: np.ndarray    # (g,)   bulk axis: signed grid area
    w1: np.ndarray           # (g,)   1-Wasserstein distance
    tail_mean: np.ndarray    # (g,)   subset axis: mean of the tail window
    tail_num: np.ndarray     # (g,)   sum(D over the tail window)
    tail_den: np.ndarray     # (g,)   sum(D over the whole grid)
    fc: np.ndarray           # (g,)
    cond1_mean: np.ndarray   # (g,)
    cond0_mean: np.ndarray   # (g,)
    tot_mean: np.ndarray     # (g,)   NOTE: a SUM of the two group means

    @property
    def diff_frac(self) -> np.ndarray:
        """``diff_mean / tot_mean``, a normalized contrast in [-1, 1].

        Not a fraction of overall abundance: ``tot_mean`` is the *sum* of
        the two group grid-means, so this is
        ``(mu1 - mu0) / (mu1 + mu0)``.
        """
        return self.diff_mean / self.tot_mean


def wade_stats(
    x: np.ndarray,
    cond: np.ndarray,
    tail_q: float = DEFAULT_TAIL_Q,
    log2_scale: bool = False,
    weight: float = 1.0,
    *,
    allow_single_sample_group: bool = False,
) -> WadeStats:
    """The statistic, from an already-normalized genes x samples matrix.

    Parameters
    ----------
    x
        Genes are rows, samples are columns. Nothing here transposes.
    cond
        Binary condition vector, one entry per column.
    tail_q
        Upper-tail fraction defining the subset window. The window is
        ``k = max(1, ceil(tail_q * nprobs))`` grid nodes.
    log2_scale, weight
        The two optional pre-transformations. ``weight != 1`` multiplies
        the **control** columns only (values > 1 inflate controls and so
        raise the bar a case group must clear); ``log2_scale`` applies
        ``log2(x + 1)`` *after* weighting. Neither is used at any call site
        in the reference material, and both are open questions
        (``docs/design-decisions.md`` O5). Note that ``weight`` is applied
        by condition, so under label permutation the weighted group changes
        membership each iteration.
    allow_single_sample_group
        See below.

    On ``min(n0, n1) == 1``
    ----------------------
    Refused by default. The grid collapses to the single probability
    ``p = 1``, so ``diff_mean``, ``tail_mean`` and ``w1`` are all just the
    difference of the two group maxima and ``tail_conc`` is identically 1 —
    a one-sample group has no quantile function worth comparing.

    This is also the one place besides ``tail_conc`` where a correct port
    **must** disagree with the R. R's reshape guard assumes a dropped
    dimension means a single gene; at ``nprobs == 1`` it means a single
    probability instead, so a length-``g`` vector gets reshaped to
    ``1 x g``, genes become probabilities, and ``wade()`` returns a
    ``g``-row frame in which every gene reports the same number. No error,
    no warning, right shape, wrong answer (``docs/porting-hazards.md``
    hazard 11). Refusing loudly is better than the R's silence; passing
    ``allow_single_sample_group=True`` computes the mathematically correct
    answer, which is still not the R's.
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

    if weight != 1.0:
        scale = np.where(cond == 0, weight, 1.0)
        x = x * scale[None, :]
    if log2_scale:
        x = np.log2(x + 1.0)

    nprobs = min(n0, n1)
    if nprobs == 1 and not allow_single_sample_group:
        raise ValueError(
            "min(n_case, n_ctrl) == 1: the quantile grid collapses to the single "
            "point p = 1, so diff_mean, tail_mean and w1 all reduce to the "
            "difference of group maxima and tail_conc is identically 1. A "
            "one-sample group has no quantile function worth comparing. Pass "
            "allow_single_sample_group=True to compute it anyway. Note the R "
            "reference is WRONG here and returns one recycled value for every "
            "gene (docs/porting-hazards.md hazard 11), so agreement with R is "
            "not available at nprobs == 1 either way."
        )

    q = probability_grid(nprobs)
    k = tail_window_size(nprobs, tail_q)

    Q1 = type7_quantiles(x[:, i1], q)
    Q0 = type7_quantiles(x[:, i0], q)
    D = Q1 - Q0

    s1 = Q1.sum(axis=1)
    s0 = Q0.sum(axis=1)
    tail_den = D.sum(axis=1)
    tail_num = D[:, :k].sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        fc = s1 / s0

    return WadeStats(
        q=q,
        nprobs=nprobs,
        k=k,
        n1=n1,
        n0=n0,
        Q1=Q1,
        Q0=Q0,
        D=D,
        diff_mean=tail_den / nprobs,
        w1=np.abs(D).mean(axis=1),
        tail_mean=D[:, :k].mean(axis=1),
        tail_num=tail_num,
        tail_den=tail_den,
        fc=fc,
        cond1_mean=s1 / nprobs,
        cond0_mean=s0 / nprobs,
        tot_mean=(s1 + s0) / nprobs,
    )


def tail_concentration(
    tail_num: np.ndarray,
    tail_den: np.ndarray,
    max_factor: float,
    abs_d_sum: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``tail_conc`` with a conditioning guard. Returns ``(value, ok)``.

    ``tail_conc = sum(D over the tail) / sum(D)`` is the share of the total
    signed area that falls in the upper-tail window. **This is the one
    statistic where the port deliberately disagrees with the R**, and the
    disagreement is the fix working (``docs/porting-hazards.md`` hazard 5).

    The problem is structural. The denominator is the total signed area,
    which vanishes whenever the lower quantiles' differences cancel the
    upper ones — a gene up in part of the case distribution and down in
    another part. R guards with ``abs(diff_mean * nprobs) < 1e-8``, which
    reconstructs the denominator and tests it against a fixed absolute
    threshold roughly nine orders of magnitude tighter than the cases that
    actually occur. Measured on the cfRNA primary contrast, 120 of 2,219
    genes (5.4%) return ``|tail_conc| > 2``, the largest 149. A "share of
    the signed area" of 149 is not a share of anything.

    **A clamp to [0, 1] would be the wrong fix.** Values slightly above 1
    are legitimate: when the lower quantiles' differences are negative they
    cancel part of the tail's contribution, the denominator shrinks below
    the numerator, and the tail genuinely carries more than the net total.

    The guard here is on the ratio's *conditioning* instead: report the
    value only when

    ``sum(|D|) / |sum(D)| <= max_factor``

    Because ``|sum(D over the tail)| <= sum(|D|)``, that condition bounds
    the reported statistic at ``|tail_conc| <= max_factor``. The guard and
    the guarantee are the same number, which makes ``max_factor`` a
    documentable promise rather than a tuning knob — and it is why raising
    it later widens what users were already told to expect.

    Genes failing the guard get ``nan`` and ``ok = False``; the components
    remain available as ``tail_num`` and ``tail_den`` on
    :class:`WadeStats`, which are exactly comparable across
    implementations even where the ratio is not.
    """
    if max_factor < 1.0:
        raise ValueError(
            f"max_factor must be at least 1 (values just above 1 are legitimate), "
            f"got {max_factor}"
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        conditioning = abs_d_sum / np.abs(tail_den)
        value = tail_num / tail_den
    ok = np.isfinite(conditioning) & (conditioning <= max_factor) & np.isfinite(value)
    return np.where(ok, value, np.nan), ok
