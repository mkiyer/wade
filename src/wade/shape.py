"""The shape test and the characterization statistics.

Implements ``docs/method.md`` sections 3 and 4.

WADE asks two questions, and this module answers the second one:

1. Is there a difference?  ->  :mod:`wade.stats`, the mean-shift test.
2. **Is a global shift an inadequate explanation?**  -> here.

Everything is read off the **log-ratio curve**

    R(p) = log2 Q_case(p) - log2 Q_ctrl(p)

which is the scale on which a global fold change is *flat* regardless of its
magnitude. On the absolute scale a 2x shift produces a difference curve that is
itself concentrated at high quantiles and is not distinguishable from a subset
effect -- measured, the same characterization statistic reads a pure 2x shift
as 0.70 on the absolute scale and 0.98 on this one.

Nothing here takes a threshold. There is no tail fraction, no window width, no
rounding rule and no guard factor: the user is never asked what shape of
difference to look for, which is the defect that made COPA require re-running at
every percentile cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .quantiles import probability_grid, type7_quantiles
from .stats import split_groups

__all__ = [
    "log_ratio_curve",
    "bridge",
    "affected_fraction",
    "up_share",
    "shift_correct",
    "ShapeResult",
    "shape_test",
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
      ``S_k -> S_k - kc`` and leaves ``B_k`` unchanged. This is what lets the
      observed statistic stay untouched while only its null is rebuilt
      (see :func:`shift_correct`).
    * **Orthogonal to the total by construction**, so scanning it is not
      re-testing the mean shift. Measured correlation with the log fold change
      under the null: +0.02.

    ``B_m`` is identically zero and carries no information; :func:`shape_test`
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


def up_share(r: np.ndarray) -> np.ndarray:
    """The share of total distributional movement that is **upward**, in [0, 1].

    The direction indicator, and what separates a one-sided subset from a
    symmetric change. Together with :func:`affected_fraction` it is a complete
    description — measured at 500 v 500:

    ==========================  ========  ==========
    gene                        pi_hat    up_share
    ==========================  ========  ==========
    global fold change 2x       0.978     1.000
    global fold change 0.5x     0.979     0.000
    variance x1.6               0.316     0.495
    subset 5% up                0.053     0.958
    subset 5% down              0.053     0.031
    subset 5% up + 5% down      0.095     0.495
    ==========================  ========  ==========

    A symmetric variance change and a one-sided 30% subset both give
    ``pi_hat ~ 0.3`` and are told apart by this.
    """
    r = np.asarray(r, dtype=np.float64)
    total = np.abs(r).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, np.maximum(r, 0.0).sum(axis=1) / total, np.nan)


def shift_correct(x: np.ndarray, cond: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Divide out the estimated global fold change, per gene.

    **This is the whole reason the shape test works**, and getting it wrong is
    not subtle.

    Permutation generates the null of *no difference at all*. The shape test's
    null is *a pure global shift* — the mean shift is the hypothesis being
    argued against. Permuting data that still contains a real shift produces
    groups that are mixtures of shifted and unshifted samples, whose spread
    does not match the shift model, so the standardization's denominator comes
    out too small. Measured, that error fires the test on **14-19%** of genuine
    global fold changes; correcting first restores exact nominal level (0.040 /
    0.045 / 0.050 at fold changes of 1.5 / 2 / 8) at no cost in power.

    The shift is estimated by the **median** of ``R``, not the mean: a subset
    signal moves the top quantiles and leaves the median alone, so estimating
    the shift this way does not quietly remove the signal being tested for.

    The observed statistic needs no adjustment — :func:`bridge` is exactly
    shift-invariant — so this matrix is used *only* to generate the null.
    """
    x = np.asarray(x, dtype=np.float64)
    cond = np.asarray(cond)
    shift = 2.0 ** np.median(r, axis=1)
    out = x.copy()
    out[:, cond == 1] /= shift[:, None]
    return out


@dataclass(frozen=True)
class ShapeResult:
    """Everything the shape test produces."""

    statistic: np.ndarray     # (g,)   max_k standardized bridge
    null: np.ndarray | None   # (g, B) the permutation null
    r: np.ndarray             # (g, m) the log-ratio curve
    b: np.ndarray             # (g, m) the bridge
    pi_hat: np.ndarray        # (g,)   effective affected fraction
    up_share: np.ndarray      # (g,)   direction, in [0, 1]
    argmax_k: np.ndarray      # (g,)   width at which the scan peaked
    shift: np.ndarray         # (g,)   the fold change divided out to build the null

    @property
    def scan_fraction(self) -> np.ndarray:
        """``argmax_k / m`` — the width the scan chose.

        **Not a fraction estimate, because it is not robust to the shape of the
        signal.** Where the affected samples are shifted multiplicatively it
        happens to be accurate; where their values are replaced outright, the
        log-ratio curve declines steeply across the affected region, the
        cumulative departure peaks well before that region ends, and the width
        comes out several-fold too small. :func:`affected_fraction` is accurate
        under both. Measured at 400 v 400, absolute error against the planted
        fraction:

        ===========  ======  ============  ==========
        signal       true    argmax / m    pi_hat
        ===========  ======  ============  ==========
        multiply     0.05    0.050         0.054
        multiply     0.25    0.299         0.283
        replace      0.05    0.010         0.057
        replace      0.25    0.100         0.273
        ===========  ======  ============  ==========

        Total absolute error over that grid: 0.315 for the argmax against 0.084
        for ``pi_hat``. Exposed for diagnostics only.
        """
        return self.argmax_k / self.r.shape[1]


def _bridge_from(x: np.ndarray, cond: np.ndarray, q: np.ndarray) -> np.ndarray:
    i1, i0 = split_groups(cond)
    return bridge(log_ratio_curve(type7_quantiles(x[:, i1], q),
                                  type7_quantiles(x[:, i0], q)))


def shape_test(
    x: np.ndarray,
    cond: np.ndarray,
    perms: np.ndarray,
    *,
    backend: str = "auto",
) -> ShapeResult:
    """Scan the bridge over every window width, against a global-shift null.

    .. math::
        T = \\max_{k<m} \\frac{B_k - \\mu_k}{\\sigma_k}

    **Threshold-free by maximization rather than by choosing.** Every window
    width is tried and the permutation null prices in the multiplicity of
    having looked — which is the direct answer to COPA's defect of requiring a
    percentile cutoff and therefore having to be run at several.

    Two passes over the permutations are unavoidable: the first estimates
    ``mu_k`` and ``sigma_k``, which the second needs in order to standardize
    before maximizing. Both run on the shift-corrected matrix.

    What the test claims is **"a global shift does not explain this"**, which is
    broader than "a subset is higher". A pure variance increase fires it, and so
    does a downward subset — the first is a real finding about heterogeneity,
    the second is signal the mean test cannot see at all. ``up_share``
    distinguishes them.
    """
    x = np.asarray(x, dtype=np.float64)
    cond = np.asarray(cond)
    i1, i0 = split_groups(cond)
    m = min(i1.size, i0.size)
    if m < 3:
        raise ValueError(
            f"the shape test needs at least 3 grid points to have a bridge with "
            f"any interior; min(n_case, n_ctrl) = {m}. Use the mean-shift test alone."
        )
    q = probability_grid(m)

    r_obs = log_ratio_curve(type7_quantiles(x[:, i1], q), type7_quantiles(x[:, i0], q))
    b_obs = bridge(r_obs)

    # The null is generated under the fitted global shift, not under
    # no-difference. b_obs is unchanged by this because bridge() is exactly
    # shift-invariant; only the null moves.
    shift = 2.0 ** np.median(r_obs, axis=1)
    xs = x.copy()
    xs[:, i1] /= shift[:, None]

    from .permutation import shape_null_backend

    stat, null, mu, sd, argmax = shape_null_backend(xs, b_obs, perms, q, backend=backend)
    return ShapeResult(
        statistic=stat, null=null, r=r_obs, b=b_obs,
        pi_hat=affected_fraction(r_obs), up_share=up_share(r_obs),
        argmax_k=argmax, shift=shift,
    )
