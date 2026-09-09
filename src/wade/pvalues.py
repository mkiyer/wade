"""Permutation p-values, GPD tail refinement, and BH-FDR (``docs/method.md`` §6).

Every step is arithmetically explicit so it can be tested branch by branch,
which is how it is tested.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ALTERNATIVES",
    "GPDFit",
    "DEFAULT_N_EXC_MIN",
    "DEFAULT_N_TAIL",
    "exceedance_counts",
    "empirical_p",
    "gpd_tail_p",
    "perm_pvalues",
    "bh_adjust",
]

#: Exceedances below which the GPD refinement fires, and the number of null
#: draws it fits. Together with ``nperms`` they set the smallest reportable
#: p-value, ``1 / (nperms * n_tail)``.
DEFAULT_N_EXC_MIN = 10
DEFAULT_N_TAIL = 250

#: The GPD fit needs at least this many exceedances above its threshold.
MIN_GPD_EXCEEDANCES = 10

ALTERNATIVES = ("two-sided", "greater", "less")


def _orient(obs: np.ndarray, null: np.ndarray, alternative: str):
    """Map an alternative onto an upper-tail comparison: the statistic as it
    stands for ``"greater"``, negated for ``"less"``, and its magnitude for
    ``"two-sided"`` (the null of ``|T|`` already contains both directions, so
    no doubling is needed)."""
    if alternative not in ALTERNATIVES:
        raise ValueError(
            f"alternative must be one of {ALTERNATIVES}; got {alternative!r}"
        )
    if alternative == "greater":
        return obs, null
    if alternative == "less":
        return -obs, -null
    return np.abs(obs), np.abs(null)


def exceedance_counts(obs: np.ndarray, null: np.ndarray,
                      alternative: str = "greater") -> np.ndarray:
    """Per-gene ``#{b : null[g, b] >= obs[g]}``, after orienting for ``alternative``.

    ``obs[:, None]`` is explicit because a bare ``null >= obs`` would
    broadcast ``obs`` along the permutation axis, and on a square matrix
    that mistake raises nothing.
    """
    obs = np.asarray(obs, dtype=np.float64)
    null = np.asarray(null, dtype=np.float64)
    if null.ndim != 2:
        raise ValueError(f"null must be a 2-D genes x permutations array, got {null.shape}")
    if obs.shape[0] != null.shape[0]:
        raise ValueError(
            f"obs has {obs.shape[0]} genes but the null matrix has {null.shape[0]} rows"
        )
    obs, null = _orient(obs, null, alternative)
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
    or scale. With ``detail=True`` the returned :class:`GPDFit` names
    which.

    Three details that change the answer, pinned by the fixtures:

    * **The exceedance inequality is strict** (``s > thr``), so ties at the
      threshold are excluded and ``len(exc)`` can be below ``n_tail``.
    * **The variance is the sample variance**, denominator ``n - 1``. Both
      moment estimators are functions of ``mean^2 / var``, and the branch
      test ``xi <= 0`` sits at the operating point, so the divisor can flip
      the branch.
    * **The rescaling uses the nominal ``n_tail / B``**, not ``len(exc) / B``.

    The ``xi <= 0`` branch takes the exponential limit rather than the GPD
    form, because a GPD with negative shape has a hard upper bound beyond
    which the survival function is zero.

    The floor ``1 / (B * n_tail)`` is an honesty constraint, not a
    numerical guard: it is what ``B`` permutations and ``n_tail`` tail
    points can support.
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

    if exc.size < MIN_GPD_EXCEEDANCES:
        return _bail("bail_few_exceedances")
    if obs <= thr:
        return _bail("bail_obs_at_or_below_thr")

    mean_exc = float(exc.mean())
    var_exc = float(exc.var(ddof=1))            # sample variance, denominator n-1
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
    alternative: str = "greater",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Empirical p-values across genes, then selective GPD refinement.

    Returns ``(p, nexc, refined)``; ``refined`` marks the genes whose
    p-value came from the tail fit rather than the permutation count.

    A gene is refined when it has fewer than ``n_exc_min`` exceedances
    *and* ``B >= 2 * n_tail`` — with the defaults, ``B >= 500``, so below
    500 permutations no refinement ever happens and the minimum p-value is
    ``1 / (B + 1)``. Genes with enough exceedances keep their empirical
    p-value: a well-resolved p-value is never replaced by an extrapolated
    one.
    """
    obs = np.asarray(obs, dtype=np.float64)
    null = np.asarray(null, dtype=np.float64)
    n_perms = null.shape[1]
    nexc = exceedance_counts(obs, null, alternative)
    # The GPD refines the same oriented quantity the counts were taken on.
    obs, null = _orient(obs, null, alternative)
    p = empirical_p(nexc, n_perms)

    refined = np.zeros(obs.shape[0], dtype=bool)
    if n_perms >= 2 * n_tail:
        for i in np.flatnonzero(nexc < n_exc_min):
            fit = gpd_tail_p(obs[i], null[i], n_tail=n_tail, detail=True)
            p[i] = fit.p
            refined[i] = not fit.branch.startswith("bail")
    return p, nexc, refined


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up adjustment.

    Applied separately to each stage across all genes; the two stages are
    not pooled into one family. NaN is dropped, the adjustment uses the
    reduced count, and the NaN is put back in position (``wade()`` produces
    all-NaN p-value columns when ``nperms == 0``).
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

    order = np.argsort(-q, kind="stable")

    ranks = np.arange(lp, 0, -1, dtype=np.float64)
    adjusted = np.minimum(1.0, np.minimum.accumulate(lp / ranks * q[order]))
    inverse = np.empty(lp, dtype=np.intp)
    inverse[order] = np.arange(lp)
    out[ok] = adjusted[inverse]
    return out
