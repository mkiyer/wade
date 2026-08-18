"""Permutation p-values, GPD tail refinement, and BH-FDR.

Ports R's ``.gpd_tail_p()``, ``wade_perm_pvalues()`` and the two
``stats::p.adjust(., "BH")`` calls. Every step is arithmetically explicit
so a port can be tested branch by branch, which is how it is tested.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "GPDFit",
    "DEFAULT_N_EXC_MIN",
    "DEFAULT_N_TAIL",
    "exceedance_counts",
    "empirical_p",
    "gpd_tail_p",
    "perm_pvalues",
    "bh_adjust",
]

DEFAULT_N_EXC_MIN = 10
DEFAULT_N_TAIL = 250


def exceedance_counts(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Per-gene ``#{b : null[g, b] >= obs[g]}``.

    The broadcast axis is the trap. R's ``rowSums(perm >= obs)`` recycles
    ``obs`` *down* each column, so gene ``i``'s observed value meets gene
    ``i``'s null draws. NumPy broadcasts trailing dimensions first, so a
    bare ``null >= obs`` aligns ``obs`` against the **permutation** axis —
    and it only raises when the number of genes differs from the number of
    permutations. On a square fixture it silently compares gene ``i``'s
    null draw ``j`` against gene ``j``'s observed value
    (``docs/implementation-notes.md`` hazard 7). Hence the explicit
    ``obs[:, None]``, and hence every parity fixture being non-square.
    """
    obs = np.asarray(obs, dtype=np.float64)
    null = np.asarray(null, dtype=np.float64)
    if null.ndim != 2:
        raise ValueError(f"null must be a 2-D genes x permutations array, got {null.shape}")
    if obs.shape[0] != null.shape[0]:
        raise ValueError(
            f"obs has {obs.shape[0]} genes but the null matrix has {null.shape[0]} rows"
        )
    return (null >= obs[:, None]).sum(axis=1)


def empirical_p(nexc: np.ndarray, n_perms: int) -> np.ndarray:
    """``(1 + nexc) / (B + 1)``.

    The add-one form counts the observed labelling among the exchangeable
    outcomes. It makes the p-value valid (never zero) and bounds it below
    by ``1 / (B + 1)`` — 5.00e-4 at B = 2000, 9.99e-4 at B = 1000.
    """
    return (1.0 + np.asarray(nexc, dtype=np.float64)) / (n_perms + 1.0)


@dataclass(frozen=True)
class GPDFit:
    """The internals of one :func:`gpd_tail_p` call, for testing and audit."""

    n_perms: int
    n_tail_used: int
    threshold: float
    n_exceedances: int
    empirical: float
    mean_exc: float
    var_exc: float
    xi: float
    sigma: float
    y: float
    tail_prob: float
    p_floor: float
    branch: str
    p: float


def gpd_tail_p(
    obs: float,
    null: np.ndarray,
    n_tail: int = DEFAULT_N_TAIL,
    *,
    detail: bool = False,
) -> float | GPDFit:
    """Generalized-Pareto refinement of one gene's upper-tail p-value.

    The empirical p-value cannot resolve below ``1 / (B + 1)``, and a gene
    whose observed statistic exceeds every null draw is known only to be
    "below that". For such genes the upper tail of the permutation null is
    modelled parametrically (Knijnenburg et al. 2009).

    Six independently reachable outcomes, all of which the test suite
    hits: the ``xi > 0`` GPD form; the ``xi <= 0`` exponential limit; the
    floor binding; fewer than ten exceedances; an observation at or below
    the threshold; and a degenerate (non-finite or non-positive) variance
    or scale.

    Three details that are easy to get wrong and that change the answer:

    * **The exceedance inequality is strict.** ``exc = s[s > thr] - thr``,
      so ties at the threshold are excluded and ``len(exc)`` is normally
      exactly ``n_tail`` but is smaller whenever the null ties there — and
      permutation nulls of discrete-ish statistics do tie. Using ``>=``
      gives a different mean, a different variance, and possibly a
      different branch.
    * **The variance is the sample variance,** denominator ``n - 1``.
      NumPy's default is ``ddof=0``. Both moment estimators are functions
      of ``m^2 / v``, so an understated ``v`` pushes ``xi`` down and
      ``sigma`` up, and since the branch test is ``xi <= 0`` the wrong
      divisor can flip the branch. The measured ``xi`` values from
      ordinary nulls include +0.031 and -0.133, so the boundary sits at
      the operating point and is crossed by noise
      (``docs/implementation-notes.md`` hazard 6).
    * **The rescaling uses the nominal ``n_tail / B``, not
      ``len(exc) / B``.** When ties reduce the exceedance count below
      ``n_tail`` those differ; the reference uses the nominal one.

    The ``xi <= 0`` branch takes the ``xi -> 0`` exponential limit rather
    than the GPD form, because a GPD with negative shape has a hard upper
    bound at ``-sigma / xi`` beyond which the survival function is zero —
    an observation past it would collapse a strong statistic to a
    machine-epsilon p-value.

    The floor ``1 / (B * n_tail)`` is a **deliberate honesty constraint**,
    not a numerical convenience: it is what B permutations and ``n_tail``
    tail points can support. 2.0e-6 at B = 2000, 4.0e-6 at B = 1000. A
    port that "improves" it by returning smaller p-values is a regression.
    """
    null = np.asarray(null, dtype=np.float64)
    if null.ndim != 1:
        raise ValueError(f"null must be a 1-D vector for one gene, got shape {null.shape}")
    obs = float(obs)
    n_perms = int(null.size)
    n_tail_used = int(min(n_tail, n_perms // 2))

    s = np.sort(null)[::-1]                       # descending
    thr = float(s[n_tail_used]) if n_tail_used < n_perms else float(s[-1])
    exc = s[s > thr] - thr
    emp = float((1 + int(np.sum(null >= obs))) / (n_perms + 1))

    def _bail(branch: str, **kw) -> float | GPDFit:
        if not detail:
            return emp
        base = dict(
            n_perms=n_perms, n_tail_used=n_tail_used, threshold=thr,
            n_exceedances=int(exc.size), empirical=emp,
            mean_exc=np.nan, var_exc=np.nan, xi=np.nan, sigma=np.nan,
            y=np.nan, tail_prob=np.nan,
            p_floor=(1.0 / (n_perms * n_tail_used)) if n_tail_used > 0 else np.inf,
            branch=branch, p=emp,
        )
        base.update(kw)
        return GPDFit(**base)

    if exc.size < 10:
        return _bail("bail_few_exceedances")
    if obs <= thr:
        return _bail("bail_obs_at_or_below_thr")

    mean_exc = float(exc.mean())
    var_exc = float(exc.var(ddof=1))            # R's stats::var, denominator n-1
    if not np.isfinite(var_exc) or var_exc <= 0:
        return _bail("bail_degenerate_variance", mean_exc=mean_exc, var_exc=var_exc)

    y = obs - thr
    p_floor = 1.0 / (n_perms * n_tail_used)
    ratio = mean_exc * mean_exc / var_exc
    xi = 0.5 * (1.0 - ratio)
    sigma = 0.5 * mean_exc * (1.0 + ratio)
    if not np.isfinite(sigma) or sigma <= 0:
        return _bail("bail_degenerate_sigma", mean_exc=mean_exc, var_exc=var_exc,
                     xi=xi, sigma=sigma, y=y)

    if xi <= 0:
        tail_prob = float(np.exp(-y / sigma))
    else:
        tail_prob = float((1.0 + xi * y / sigma) ** (-1.0 / xi))

    scaled = (n_tail_used / n_perms) * tail_prob
    p = max(scaled, p_floor)
    if not detail:
        return p
    branch = "floor" if (p == p_floor and scaled < p_floor) else ("exponential" if xi <= 0 else "gpd")
    return GPDFit(
        n_perms=n_perms, n_tail_used=n_tail_used, threshold=thr,
        n_exceedances=int(exc.size), empirical=emp,
        mean_exc=mean_exc, var_exc=var_exc, xi=xi, sigma=sigma,
        y=y, tail_prob=tail_prob, p_floor=p_floor, branch=branch, p=p,
    )


def perm_pvalues(
    obs: np.ndarray,
    null: np.ndarray,
    n_exc_min: int = DEFAULT_N_EXC_MIN,
    n_tail: int = DEFAULT_N_TAIL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Empirical p-values across genes, then selective GPD refinement.

    Returns ``(p, nexc, refined_mask)``.

    **The refinement gate has two conditions and the second is a property
    of the run, not of the gene.** A gene is refined when it has fewer than
    ``n_exc_min`` exceedances *and* ``B >= 2 * n_tail``. With the defaults
    that second condition is ``B >= 500``, so at ``nperms < 500`` **no
    refinement ever happens** and the minimum p-value is ``1 / (B + 1)``. A
    port that omits it would refine at ``nperms = 200`` and produce
    p-values the reference cannot produce, on a path no fixture at
    ``nperms`` of 1000 or 2000 would ever exercise.

    Genes with enough exceedances keep their empirical p-value: a
    well-resolved p-value is never replaced by an extrapolated one.
    """
    obs = np.asarray(obs, dtype=np.float64)
    null = np.asarray(null, dtype=np.float64)
    n_perms = null.shape[1]
    nexc = exceedance_counts(obs, null)
    p = empirical_p(nexc, n_perms)

    refined = np.zeros(obs.shape[0], dtype=bool)
    if n_perms >= 2 * n_tail:
        refined = nexc < n_exc_min
        for i in np.flatnonzero(refined):
            p[i] = gpd_tail_p(obs[i], null[i], n_tail=n_tail)
    return p, nexc, refined


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up adjustment, matching ``p.adjust(p, "BH")``.

    Applied **separately to each axis** across all genes: ``padj_diff``
    from the G values of ``p_diff``, ``padj_tail`` from the G values of
    ``p_tail``. The two axes are not pooled into one family of 2G tests.

    Two behaviours pinned here that are not about the arithmetic, because
    ``wade()`` produces all-NaN p-value columns when ``nperms == 0``:

    * **NaN is dropped, not propagated**, the adjustment uses the reduced
      count, and the NaN is put back in position. R gets this via lazy
      evaluation of its ``n = length(p)`` default, which is forced only
      after ``p`` has been subset to the non-missing entries.
    * Ties receive equal adjusted values, and the monotonicity enforcement
      — the cumulative minimum from the largest p-value downward — is
      applied.
    """
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"p must be a 1-D vector, got shape {p.shape}")
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    lp = q.size
    if lp <= 1:
        out[ok] = q
        return out

    order = np.argsort(-q, kind="stable")        # R's order(p, decreasing = TRUE)
    ranks = np.arange(lp, 0, -1, dtype=np.float64)
    adjusted = np.minimum(1.0, np.minimum.accumulate(lp / ranks * q[order]))
    inverse = np.empty(lp, dtype=np.intp)
    inverse[order] = np.arange(lp)
    out[ok] = adjusted[inverse]
    return out
