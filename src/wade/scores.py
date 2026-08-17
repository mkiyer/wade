"""Rank scores — nomination, not inference.

R's ``wade_score()``. These are **heuristics** and the framing has to
travel with them: a rank of 1 is not a claim of significance. They are
here because they are what the source project's nomination actually ran
on — on the cfRNA cohort no gene survived BH on either axis, so genes
were nominated by rank score with the p-values read as effect-strength
annotation — and a package that omits them omits the thing the historical
analysis used.

They live in their own module for the same reason: offering ``score``
next to ``padj_diff`` without comment invites a reader to treat a
heuristic rank as inference (``docs/design-decisions.md`` O3).

A documented asymmetry, since the R's own header describes the two scores
as the same construction with one term swapped and they are not:
``score`` takes its magnitude from the ECDF of an **absolute** value
(``|log2 fc|``) and its sign separately from ``diff_frac``, while
``tail_score`` takes both from the ECDF of a **signed** value
(``tail_mean``). So a gene strongly *down* in the case tail gets a small
negative ``tail_score`` rather than a large one.

Directionality: both scores reward up-in-case genes. Genes *lost* in
cases are visible in ``diff_mean`` but are not what this ranking
surfaces.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ecdf_values", "dense_rank_desc", "wade_score"]


def ecdf_values(x: np.ndarray, at: np.ndarray | None = None) -> np.ndarray:
    """R's ``stats::ecdf(x)`` evaluated at ``at`` (default: at ``x`` itself).

    ``F(t) = #{x <= t} / n``: right-continuous, ties counted together.
    **The comparison is ``<=``, not ``<``.** Consequences the scores
    depend on: ``F`` reaches exactly 1 at the maximum, so ``1 - F`` can be
    exactly 0; and ``F`` is never 0 at an observed point.

    Missing values are dropped from the sample (R's ``ecdf`` sorts, and
    ``sort`` drops ``NA``), so ``n`` is the non-missing count; evaluating
    *at* a missing value gives ``nan``.
    """
    x = np.asarray(x, dtype=np.float64)
    xs = np.sort(x[~np.isnan(x)])
    n = xs.size
    if n == 0:
        raise ValueError("ecdf needs at least one non-missing value")
    t = x if at is None else np.asarray(at, dtype=np.float64)
    out = np.searchsorted(xs, t, side="right") / n
    return np.where(np.isnan(t), np.nan, out)


def dense_rank_desc(x: np.ndarray) -> np.ndarray:
    """``dplyr::dense_rank(dplyr::desc(x))``: rank 1 is the largest value.

    Ties share a rank and the next rank is consecutive (no gaps). Missing
    values propagate. Returned as float so ``nan`` is representable.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.full(x.shape, np.nan)
    ok = ~np.isnan(x)
    if not np.any(ok):
        return out
    uniq = np.unique(x[ok])                       # ascending
    out[ok] = uniq.size - np.searchsorted(uniq, x[ok], side="left")
    return out


def wade_score(
    fc: np.ndarray,
    cond1_mean: np.ndarray,
    cond0_mean: np.ndarray,
    tail_mean: np.ndarray,
    diff_frac: np.ndarray,
    *,
    log2fc_exp: float = 1.0,
    case_exp: float = 1.0,
    ctrl_exp: float = 1.0,
    tail_exp: float = 1.0,
) -> dict[str, np.ndarray]:
    """The two rank scores and their dense ranks, computed across genes.

    Both are products of three quantities in [0, 1] times a sign, so both
    lie in [-1, 1]. Each rewards a gene that is high in cases and low in
    controls: ``score`` on the bulk axis via fold change, ``tail_score``
    on the subset axis via the tail statistic.

    The four empirical CDFs are built **across the genes in this frame**,
    so every value here depends on which genes were analysed together.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        log2fc = np.log2(np.asarray(fc, dtype=np.float64))

    f_fc = ecdf_values(np.abs(log2fc))
    f_case = ecdf_values(np.asarray(cond1_mean, dtype=np.float64))
    f_ctrl = ecdf_values(np.asarray(cond0_mean, dtype=np.float64))
    f_tail = ecdf_values(np.asarray(tail_mean, dtype=np.float64))

    # Written out left to right, matching the reference's association exactly.
    #
    # Factoring the two shared terms into a subexpression looks harmless and
    # is not: it changes ((s*A)*B)*C into (s*A)*(B*C), which differs in the
    # last bit. The scores themselves would still agree to ~1e-16, but
    # dense_rank breaks ties on EXACT equality, so a one-ulp difference
    # merges or splits a tie and shifts every rank below it by one. Measured
    # on the tiesheavy fixture: two genes tied at rank 9 in the factored form
    # were ranks 9 and 10 in R, and all six lower ranks shifted.
    score = (np.sign(diff_frac) * f_fc ** log2fc_exp
             * f_case ** case_exp * (1.0 - f_ctrl) ** ctrl_exp)
    tail_score = (np.sign(tail_mean) * f_tail ** tail_exp
                  * f_case ** case_exp * (1.0 - f_ctrl) ** ctrl_exp)

    return {
        "log2fc": log2fc,
        "score": score,
        "tail_score": tail_score,
        "rank": dense_rank_desc(score),
        "tail_rank": dense_rank_desc(tail_score),
    }
