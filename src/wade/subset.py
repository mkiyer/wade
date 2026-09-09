"""The subset test and the characterization statistics (``docs/method.md`` §3–4).

WADE asks two questions, and this module answers the second one: **is the
difference confined to a subset of samples?** Everything is read off the
log-ratio curve ``R(p) = log2 Q_case(p) - log2 Q_ctrl(p)``, the scale on which
a global fold change is flat regardless of its magnitude.

Nothing here takes a threshold: no tail fraction, no window width, no rounding
rule. The user is never asked what shape of difference to look for.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .quantiles import capped_nprobs, probability_grid, type7_quantiles
from .stats import split_groups
from .thinning import gene_chunks


__all__ = [
    "log_ratio_curve",
    "bridge",
    "affected_fraction",
    "direction",
    "subset_log2_fc",
    "SubsetResult",
    "subset_test",
    "characterization_ci",
]


def log_ratio_curve(q1: np.ndarray, q0: np.ndarray) -> np.ndarray:
    """``R(p) = log2 Q_case(p) - log2 Q_ctrl(p)``, genes x probabilities.

    Requires strictly positive quantiles. The continuity jitter guarantees
    that for the raw-count entry point; a caller supplying an already
    normalized matrix containing exact zeros has to deal with it, and gets
    told so rather than receiving silent infinities.
    """
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = np.asarray(q0, dtype=np.float64)
    if np.any(q1 <= 0) or np.any(q0 <= 0):
        bad = int(np.flatnonzero((q1 <= 0).any(axis=1) | (q0 <= 0).any(axis=1))[0])
        raise ValueError(
            f"the log-ratio curve needs strictly positive quantiles; gene {bad} has "
            f"a non-positive one. The continuity jitter normally guarantees this — if "
            f"you passed an already-normalized matrix, it has exact zeros in it."
        )
    return np.log2(q1) - np.log2(q0)


def bridge(r: np.ndarray) -> np.ndarray:
    """``B_k = S_k - (k/m) S_m`` where ``S_k`` is the running total of ``R``.

    The departure from *proportional* growth. If the difference were a pure
    global shift the top 20% of quantiles would carry 20% of the total, so
    ``B`` would be identically zero; it is positive when the difference is
    front-loaded.

    Two properties make it the right object:

    * **Exactly invariant to a global fold change.** ``R -> R - c`` sends
      ``S_k -> S_k - kc`` and leaves ``B_k`` unchanged.
    * **Orthogonal to the total by construction**, so scanning it is not
      re-testing the mean shift. Measured correlation with the log fold change
      under the null: +0.02.

    ``B_m`` is identically zero and carries no information; :func:`subset_test`
    scans ``k = 1 .. m-1``.
    """
    r = np.asarray(r, dtype=np.float64)
    m = r.shape[1]
    s = np.cumsum(r, axis=1)
    k = np.arange(1, m + 1, dtype=np.float64)
    return s - (k / m) * s[:, -1][:, None]


def affected_fraction(r: np.ndarray) -> np.ndarray:
    """The effective fraction of the distribution that differs.

    .. math::
        \\hat{\\pi} = \\frac{\\left(\\sum_p R(p)^2\\right)^2}{m \\sum_p R(p)^4}

    A participation ratio. For a **step** — fraction ``pi`` of cases shifted,
    the rest untouched — this is exactly ``pi``. For a curve **flat** at
    ``log2(FC)`` it is exactly 1. So it answers the question a biologist asks:
    *what fraction of cases is this gene altered in?*

    **Why the fourth moment and not the second.** A broad low-level noise floor
    of height ``e`` against signal ``h`` contributes ``e/h`` to the
    second-moment form and ``(e/h)^2`` to this one. The log scale is what
    anchors a global change at 1.0, but its unaffected quantiles carry enough
    sampling noise to swamp a small signal; the quartic form suppresses that
    floor and keeps the anchor. Measured, the second-moment version reads a 2%
    subset as 0.23 and this one as 0.025.

    **Resolution limit.** The grid has ``m`` points, so no fraction finer than
    ``1/m`` is resolvable. Quantitative above ``m ~ 100``, degrading through
    ``m ~ 50``, and below that only qualitative — global still separates from
    concentrated, but 2% and 5% do not separate from each other.
    """
    r = np.asarray(r, dtype=np.float64)
    m = r.shape[1]
    s2 = (r * r).sum(axis=1)
    s4 = (r ** 4).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(s4 > 0, s2 * s2 / (m * s4), np.nan)


def direction(r: np.ndarray) -> np.ndarray:
    """Which way the difference goes, as a signed score in ``[-1, +1]``.

    .. math::
        \\texttt{direction} = \\frac{\\sum_p R(p)}{\\sum_p |R(p)|}

    ``+1`` means every part of the distribution moved up, ``-1`` every part
    moved down, and ``0`` means upward and downward movement exactly balance —
    a two-sided change such as a variance increase, or a subset raised and an
    equal subset lowered.

    It is the *coherence* of the direction, not its size: a gene shifted 1.01x
    everywhere and one shifted 8x everywhere both score ``+1``. Magnitude lives
    in ``mean_shift`` and ``log2_fc``.

    Together with :func:`affected_fraction` it is the whole characterization.
    Measured at 500 v 500:

    ==========================  ====================  ============
    gene                        affected_fraction     direction
    ==========================  ====================  ============
    global fold change 2x       0.978                 +1.000
    global fold change 0.5x     0.979                 -1.000
    variance x1.6               0.316                 -0.010
    subset 5% up                0.053                 +0.916
    subset 5% down              0.053                 -0.928
    subset 5% up + 5% down      0.095                 -0.006
    ==========================  ====================  ============

    A symmetric variance change and a one-sided 30% subset both give
    ``affected_fraction ~ 0.3``; only this tells them apart.
    """
    r = np.asarray(r, dtype=np.float64)
    total = np.abs(r).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, r.sum(axis=1) / total, np.nan)


def subset_log2_fc(r: np.ndarray) -> np.ndarray:
    """The magnitude of the subset: log2 fold change **within the
    affected fraction**.

    The log-ratio curve already *is* per-quantile magnitude —
    ``R(p) = log2 Q1(p) - log2 Q0(p)`` says how many folds above the
    corresponding control quantile the cases sit at each ``p`` — so the
    subset's magnitude is the mean of ``R`` over the affected region:
    the top ``ceil(affected_fraction * m)`` grid nodes for an upward
    subset (``direction >= 0``), the bottom nodes for a downward one.
    No threshold is introduced: the region width is the data's own
    :func:`affected_fraction`.

    Reads exactly as a fold change, with one deliberate property: the
    comparison is **quantile-matched** — the affected cases against what
    the *controls themselves* do at those same extreme quantiles, not
    against the control mean. A subset planted at 8× the control mean
    of an NB(50) reads ~2 rather than ``log2 8 = 3``, because the
    controls' own top decile sits well above their mean; a subset that
    barely clears the controls' natural tail reads near 0 however far
    above the control *mean* it is. That discounting of the controls'
    spread is what makes the number a measure of how *distinctive* the
    subset is, and it is the distinction ``p_subset`` cannot make once
    it saturates: an 8× and a 100× subset tie on p and differ by ~3.5
    here. Two boundary behaviours make it coherent with the
    rest of the table: a global change has ``affected_fraction ~ 1``, so
    this becomes the mean of the whole curve — the gene's overall log2
    fold change — and the reported number *includes* any global shift
    the gene also carries (``shift`` is reported separately, so the
    subset's excess over it is one subtraction away). For a balanced
    up-and-down mixture (``direction ~ 0``) a single signed number is
    the wrong shape, as it is for ``direction`` itself.
    """
    r = np.asarray(r, dtype=np.float64)
    g, m = r.shape
    frac = np.nan_to_num(affected_fraction(r), nan=1.0)
    k = np.clip(np.ceil(frac * m).astype(np.intp), 1, m)
    cs = np.cumsum(r, axis=1)
    rows = np.arange(g)
    top = cs[rows, k - 1] / k
    # sum of the last k nodes = total - sum of the first m - k
    below = np.where(k < m, cs[rows, np.maximum(m - k - 1, 0)], 0.0)
    bottom = (cs[:, -1] - below) / k
    down = np.nan_to_num(direction(r), nan=0.0) < 0
    return np.where(down, bottom, top)


@dataclass(frozen=True)
class SubsetResult:
    """Everything the subset test produces.

    ``r`` is the characterization curve — the log-ratio of the observed
    quantiles, with the pseudocount — which ``affected_fraction``,
    ``direction`` and ``subset_log2_fc`` are read from and which
    :func:`wade.plot_gene` draws. The statistic itself is computed on the
    thinned matrix's curve, which is not kept.
    """

    statistic: np.ndarray     # (g,)   max_k standardized bridge
    null: np.ndarray | None   # (g, B) the permutation null
    r: np.ndarray             # (g, m) the log-ratio curve (characterization)
    affected_fraction: np.ndarray  # (g,) effective fraction of samples that differ
    direction: np.ndarray          # (g,) -1 all down .. +1 all up
    argmax_k: np.ndarray      # (g,)   width at which the scan peaked. A diagnostic,
                              #        not a fraction estimate (method.md §3)
    shift: np.ndarray         # (g,)   the fitted global fold change the null was built under

    @property
    def subset_log2_fc(self) -> np.ndarray:
        """The subset's magnitude — see :func:`subset_log2_fc`."""
        return subset_log2_fc(self.r)


def _resolve_pseudocount(pseudocount, shape) -> np.ndarray | None:
    """``None``/``0`` → no pseudocount; a scalar → the same everywhere (the
    matrix's units); a per-sample vector or a genes x samples array → per cell."""
    if pseudocount is None:
        return None
    pc = np.asarray(pseudocount, dtype=np.float64)
    if pc.ndim == 0:
        if float(pc) == 0.0:
            return None
        pc = np.full(shape, float(pc))
    elif pc.ndim == 1:
        if pc.shape[0] != shape[1]:
            raise ValueError(f"a per-sample pseudocount needs {shape[1]} values, got {pc.shape[0]}")
        pc = np.broadcast_to(pc[None, :], shape).copy()
    elif pc.shape != shape:
        raise ValueError(f"pseudocount must be a scalar, a per-sample vector or a {shape} array; got {pc.shape}")
    if np.any(pc < 0) or not np.all(np.isfinite(pc)):
        raise ValueError("pseudocount must be finite and non-negative")
    return pc


def subset_test(
    x: np.ndarray,
    cond: np.ndarray,
    perms: np.ndarray,
    corrected: np.ndarray,
    shift: np.ndarray,
    *,
    alternative: str = "two-sided",
    backend: str = "auto",
    pseudocount=None,
    max_probs: int | None = None,
) -> SubsetResult:
    """Scan the bridge over every window width, against a global-shift null.

    Answers **"is the difference confined to a subset of samples?"** — stated
    precisely, "a global shift does not explain this".

    .. math::
        T = \\max_{k<m} \\frac{B_k - \\mu_k}{\\sigma_k}

    **Threshold-free by maximization rather than by choosing.** Every window
    width is tried and the permutation null prices in the multiplicity of
    having looked — which is the direct answer to COPA's defect of requiring a
    percentile cutoff and therefore having to be run at several.

    **The null is a global shift, not "no difference".** Permuting data that
    contains a real shift produces groups that are mixtures of shifted and
    unshifted samples, and the test then fires on 14–19% of genuine global
    fold changes. So both the observed statistic and the null are computed on
    ``corrected``: the matrix made exchangeable under the fitted global shift
    by binomial thinning (:mod:`wade.thinning`, ``docs/method.md`` §9.3).
    The observed statistic is read off it too, because the bridge is not
    invariant to thinning.

    Two passes over the permutations are unavoidable: the first estimates
    ``mu_k`` and ``sigma_k``, which the second needs in order to standardize
    before maximizing.

    ``alternative`` selects which end of the distribution the concentration
    must sit at. ``"two-sided"`` (the default) scans ``|Z_k|`` and detects
    concentration anywhere; ``"greater"`` requires the difference to be
    top-heavy, which is the classic cancer-outlier case of a subset with
    *elevated* expression; ``"less"`` requires it to be bottom-heavy.

    What the test claims is **"a global shift does not explain this"**, which
    is broader than "a subset is higher". A pure variance increase fires it,
    and so does a downward subset — the first is a real finding about
    heterogeneity, the second is signal the mean test cannot see at all.
    :func:`direction` distinguishes them.

    Parameters
    ----------
    x
        The normalized matrix, genes x samples; the characterization curve
        is read off it.
    corrected, shift
        The thinned, normalized matrix (same shape as ``x``) and the per-gene
        fold change it was thinned under (:func:`wade.thinning.fit_fold_change`,
        :func:`wade.thinning.thin_counts`).
    pseudocount
        Added to every value before the log-ratio curve is taken — a scalar
        in the matrix's units, a per-sample vector, or a genes x samples
        array. :func:`wade.wade` passes one count in each cell's normalized
        units (``docs/method.md`` §9.4); ``None`` means none.
    max_probs
        Caps the grid at large cohorts, exactly as :func:`wade.wade_stats`
        does (``docs/method.md`` §1). ``affected_fraction``'s resolution
        becomes ``1/min(n0, n1, max_probs)``.
    """
    x = np.asarray(x, dtype=np.float64)
    cond = np.asarray(cond)
    i1, i0 = split_groups(cond)
    m = capped_nprobs(i1.size, i0.size, max_probs)
    if m < 3:
        raise ValueError(
            f"the subset test needs at least 3 grid points to have a bridge with "
            f"any interior; the grid has {m}. Use the mean-shift test alone."
        )
    q = probability_grid(m)
    pc = _resolve_pseudocount(pseudocount, x.shape)
    xp = x if pc is None else x + pc

    # The characterization curve: observed quantiles, pseudocounted.
    r_obs = log_ratio_curve(type7_quantiles(xp[:, i1], q), type7_quantiles(xp[:, i0], q))

    corrected = np.asarray(corrected, dtype=np.float64)
    if corrected.shape != x.shape:
        raise ValueError(f"corrected must have the matrix's shape {x.shape}, got {corrected.shape}")
    shift = np.asarray(shift, dtype=np.float64)
    if shift.shape != (x.shape[0],):
        raise ValueError(f"shift must have one value per gene; got {shift.shape}")
    xs = corrected if pc is None else corrected + pc
    b_test = bridge(log_ratio_curve(type7_quantiles(xs[:, i1], q),
                                    type7_quantiles(xs[:, i0], q)))

    from .permutation import subset_null_backend

    stat, null, _mu, _sd, argmax = subset_null_backend(
        xs, b_test, perms, q, alternative=alternative, backend=backend
    )
    return SubsetResult(
        statistic=stat, null=null, r=r_obs,
        affected_fraction=affected_fraction(r_obs), direction=direction(r_obs),
        argmax_k=argmax, shift=shift,
    )


def characterization_ci(
    x: np.ndarray,
    cond: np.ndarray,
    *,
    pseudocount=None,
    n_boot: int = 300,
    rng: np.random.Generator | None = None,
    level: float = 0.95,
    max_probs: int | None = None,
    gene_chunk: int | None = None,
    threads: int | None = None,
) -> dict[str, np.ndarray]:
    """Bootstrap percentile intervals for the five per-gene descriptors —
    ``affected_fraction``, ``direction``, ``subset_log2_fc``, ``log2_fc`` and
    ``mean_shift`` — ``docs/method.md`` §9.5.

    Samples are resampled **within each group** with replacement, the
    log-ratio curve is recomputed on the same grid with the same pseudocount,
    and the descriptors are read off it. ``log2_fc`` is bootstrapped as the
    log ratio of group means and ``mean_shift`` as their difference. Returns a
    dict of ``(2, genes)`` arrays, lower row first.

    The interval is about sampling uncertainty in the *estimator*; it does
    not remove the estimator's known biases (a 2× step against 30% noise is a
    ramp and reads larger than the planted fraction; a two-level departure
    reads as one effective fraction).

    ``gene_chunk`` bounds the working set and ``threads`` (default: all
    cores) runs the replicates concurrently. Both are bit-identical to the
    plain call: the replicate index sets are drawn once, up front, and every
    replicate's arithmetic is self-contained.
    """
    x = np.asarray(x, dtype=np.float64)
    cond = np.asarray(cond)
    i1, i0 = split_groups(cond)
    n1, n0 = i1.size, i0.size
    q = probability_grid(capped_nprobs(n1, n0, max_probs))
    pc = _resolve_pseudocount(pseudocount, x.shape)
    rng = np.random.default_rng(0) if rng is None else rng
    g = x.shape[0]

    chunks = gene_chunks(g, gene_chunk)
    J1 = np.empty((n_boot, n1), dtype=np.intp)
    J0 = np.empty((n_boot, n0), dtype=np.intp)
    for b in range(n_boot):
        J1[b] = i1[rng.integers(0, n1, n1)]
        J0[b] = i0[rng.integers(0, n0, n0)]

    aff = np.empty((n_boot, g)); dirn = np.empty((n_boot, g))
    lfc = np.empty((n_boot, g)); slfc = np.empty((n_boot, g)); ms = np.empty((n_boot, g))
    if threads is None:
        threads = os.cpu_count() or 1
    for ch in chunks:
        x_ch = x[ch]
        xp_ch = x_ch if pc is None else x_ch + pc[ch]

        def one_replicate(b, x_ch=x_ch, xp_ch=xp_ch, ch=ch):
            with np.errstate(divide="ignore", invalid="ignore"):
                r = log_ratio_curve(type7_quantiles(xp_ch[:, J1[b]], q),
                                    type7_quantiles(xp_ch[:, J0[b]], q))
                aff[b, ch] = affected_fraction(r)
                dirn[b, ch] = direction(r)
                slfc[b, ch] = subset_log2_fc(r)
                m1 = x_ch[:, J1[b]].mean(axis=1)
                m0 = x_ch[:, J0[b]].mean(axis=1)
                lfc[b, ch] = np.log2(m1 / m0)
                ms[b, ch] = m1 - m0

        with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
            list(pool.map(one_replicate, range(n_boot)))
    lo_q, hi_q = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    out = {}
    for name, arr in (("affected_fraction", aff), ("direction", dirn),
                      ("subset_log2_fc", slfc), ("log2_fc", lfc), ("mean_shift", ms)):
        out[name] = np.nanpercentile(arr, [lo_q, hi_q], axis=0)
    return out
