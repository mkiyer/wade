"""The permutation tail of a subset sum, computed rather than sampled.

Stage 1's statistic is the signed area between the two quantile functions,
and :math:`\\int_0^1 Q(p)\\,dp = E[X]` is an identity, so that area **is** the
difference of group means and the quantile grid was only ever a quadrature of
it. Computed exactly, stage 1 is a linear statistic:

.. code-block:: text

    T  =  A·(1/n1 + 1/n0) − V/n0        A = the case-subset sum, V = the total

so ``P(T >= t) = P(A >= a)``, a question about the sum of a simple random
sample drawn without replacement. That has a classical **double saddlepoint**
answer (Skovgaard 1987; Booth & Butler 1990) with relative error far into the
tail, at :math:`O(n)` per gene and no permutations at all — so there is no
resolution floor, no tail model and no extreme-value parameter to estimate.

The conditioning is what makes sampling *without* replacement tractable. Take
independent Bernoulli selectors :math:`Z_i` and condition on
:math:`\\sum Z_i = n_1`; the joint cumulant generating function is

.. code-block:: text

    K(s, t)  =  sum_i log(1 + exp(s·v_i + t))

and Skovgaard's formula turns the two saddlepoints of that into a tail
probability. Everything below is that formula, vectorized across genes.

It is not a distributional assumption about the data: the CGF is built from
the observed values and the probability computed is the exact permutation
probability. **Its accuracy does depend on the values not being an atom.** A
gene whose nonzero support is smaller than a group has a lattice-like
permutation distribution — most relabellings tie on the same handful of sums —
and the continuous approximation is then off by several-fold in either
direction. Exact ties at the extreme are counted exactly (the boundary is
combinatorial, not approximated); near-ties inside it are the limit of the
method, and it is why ``stage1="saddlepoint"`` is documented for cohorts
where genes are expressed, not for the sparse tail.

Validated against 1e8 brute-force permutations per design: median ratio to
truth 0.91–1.15 over four geometries, 1.00–1.02 on negative-binomial counts
through :func:`wade.wade`'s own normalization, and better at larger ``n``, as
an asymptotic method should be.
"""

from __future__ import annotations

from math import comb, exp, lgamma

import numpy as np

__all__ = ["saddlepoint_subset_sum", "mean_diff_saddlepoint_p"]

#: Newton iteration caps. Both solves exit early on a residual and the final
#: answer is checked against both saddlepoint equations before it is returned,
#: so a cap that turned out too small raises rather than returning a plausible
#: wrong number.
_INNER_ITERS = 40
_OUTER_ITERS = 12
_TOL = 1e-11


def _cgf(v, s, t):
    """``K``, its gradient and its Hessian, vectorized over genes.

    ``v`` is ``(g, n)``; ``s`` and ``t`` are ``(g,)``. Returns ``K``, ``K_t``,
    ``K_s``, ``K_ss``, ``K_st``, ``K_tt``. ``s·v`` reaches the hundreds in the
    far tail, so both the CGF and the logistic are written in the forms that
    do not overflow there.
    """
    z = s[:, None] * v + t[:, None]
    K = np.logaddexp(0.0, z).sum(axis=1)
    p = 0.5 * (1.0 + np.tanh(0.5 * z))            # the logistic, overflow-free
    q = p * (1.0 - p)
    return (K, p.sum(axis=1), (p * v).sum(axis=1),
            (q * v * v).sum(axis=1), (q * v).sum(axis=1), q.sum(axis=1))


def _bracket_t(v, s, n1):
    """A closed-form bracket for ``K_t(s, t) = n1``.

    ``K_t(t) = sum_i sigma(s·v_i + t)`` with ``sigma`` the logistic, so with
    ``c = log(n1/n0)``: at ``t = c - max_i(s·v_i)`` every term is at most
    ``n1/n`` and ``K_t <= n1``; at ``t = c - min_i(s·v_i)`` every term is at
    least that and ``K_t >= n1``. Both hold for either sign of ``s``.
    """
    n = v.shape[1]
    c = np.log(n1 / (n - n1))
    sv = s[:, None] * v
    return c - sv.max(axis=1), c - sv.min(axis=1)


def _solve_t(v, s, n1, iters=_INNER_ITERS):
    """``t`` with ``K_t(s, t) = n1``, per gene.

    ``K_t`` is a sum of ``n`` logistics, so it increases strictly from 0 to
    ``n`` and the root is unique for any ``0 < n1 < n``. Six bisections of the
    closed-form bracket, then Newton with the bracket carried alongside so a
    step that leaves it is replaced by the midpoint.
    """
    lo, hi = _bracket_t(v, s, n1)
    for _ in range(6):
        mid = 0.5 * (lo + hi)
        below = _cgf(v, s, mid)[1] < n1
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    t = 0.5 * (lo + hi)
    for _ in range(iters):
        _, Kt, _, _, _, Ktt = _cgf(v, s, t)
        r = Kt - n1
        if np.all(np.abs(r) < _TOL * max(n1, 1)):
            break
        lo = np.where(r < 0, t, lo)
        hi = np.where(r > 0, t, hi)
        with np.errstate(over="ignore", invalid="ignore"):
            step = np.where(Ktt > 1e-300, r / np.maximum(Ktt, 1e-300), 0.0)
        nxt = t - step
        bad = ~np.isfinite(nxt) | (nxt <= lo) | (nxt >= hi)
        t = np.where(bad, 0.5 * (lo + hi), nxt)
    return t


def _extreme_labellings(v, n1):
    """Per gene, how many of the ``C(n, n1)`` labellings attain the maximum
    subset sum. One when the top ``n1`` values are distinct from the rest;
    ``C(t, r)`` when ``t`` values tie at the cut and ``r`` of them are needed."""
    srt = np.sort(v, axis=1)[:, ::-1]
    cut = srt[:, n1 - 1][:, None]
    ties = (v == cut).sum(axis=1)
    above = (v > cut).sum(axis=1)
    return np.array([comb(int(t), int(n1 - a)) for t, a in zip(ties, above)], dtype=np.float64)


def saddlepoint_subset_sum(values, n1, a, *, upper=True):
    """``P(A >= a)`` for ``A`` the sum of ``n1`` of ``values``, drawn without
    replacement — one probability per row.

    Parameters
    ----------
    values
        ``(g, n)``. Each row is one gene's ``n`` sample values.
    n1
        The subset size. The complement is ``n - n1``.
    a
        ``(g,)``. The threshold, in the same units as ``values``.
    upper
        ``True`` for ``P(A >= a)``. ``False`` returns ``P(A <= a)``, which is
        the same computation with the sign of the statistic flipped.

    Returns
    -------
    ``(g,)`` probabilities in ``[1/C(n, n1), 1]``.

    Notes
    -----
    Skovgaard's second-order form is

    .. code-block:: text

        P  =  1 - Phi(w) - phi(w)·(1/w - 1/u)

        w  =  sign(s)·sqrt( 2·[ (K(0,t0) - t0·n1) - (K(s,t) - s·a - t·n1) ] )
        u  =  s·sqrt( |K''(s,t)| / K_tt(0,t0) )

    where ``(s, t)`` solve ``K_s = a`` and ``K_t = n1``, and ``t0`` solves
    ``K_t = n1`` at ``s = 0``. At ``s = 0`` every selector has the same
    probability ``n1/n``, so ``t0 = log(n1/n0)`` and
    ``K_tt(0, t0) = n1·n0/n`` in closed form.

    Three regions are handled exactly rather than approximated. ``a`` at or
    above the largest attainable sum has probability equal to the fraction of
    labellings that attain it (one, unless values tie at the cut); ``a`` at
    or below the smallest is certain; and ``w -> 0`` near the mean, where the
    formula is :math:`0/0`, falls back to the normal approximation, which is
    accurate there.
    """
    try:
        from scipy.stats import norm
    except ModuleNotFoundError as exc:      # pragma: no cover - see api.py
        raise ImportError(
            "the saddlepoint p-value needs SciPy, which WADE does not require "
            "otherwise. Install it, or use the default stage1='grid'."
        ) from exc

    v = np.ascontiguousarray(values, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError(f"values must be 2-D (genes x samples), got {v.shape}")
    g, n = v.shape
    n1 = int(n1)
    n0 = n - n1
    if not 0 < n1 < n:
        raise ValueError(f"n1 must be strictly between 0 and n={n}; got {n1}")
    a = np.asarray(a, dtype=np.float64)
    if a.shape != (g,):
        raise ValueError(f"a must have shape ({g},), got {a.shape}")
    if not upper:
        return saddlepoint_subset_sum(-v, n1, -a, upper=True)

    # The denominator saddlepoint, in closed form.
    t0 = np.full(g, np.log(n1 / n0))
    K0 = _cgf(v, np.zeros(g), t0)[0]
    Ktt0 = np.full(g, n1 * n0 / n)

    # The reachable range of A is [a_min, a_max]. At either end the answer is
    # combinatorial and exact; strictly inside, a finite saddlepoint exists.
    # The boundary is settled before any solving, with a tolerance relative to
    # the reachable width rather than to `a`, which may be negative.
    srt = np.sort(v, axis=1)
    a_min = srt[:, :n1].sum(axis=1)
    a_max = srt[:, -n1:].sum(axis=1)
    width = np.maximum(a_max - a_min, 1e-300)
    at_max = a >= a_max - 1e-12 * width
    at_min = a <= a_min + 1e-12 * width
    interior = ~(at_max | at_min)

    # A symmetric bracket [-S, S] for the numerator root. `K_s` increases
    # strictly in `s` from `a_min` to `a_max`, and `|s| * spread = 60`
    # saturates every logistic to within 1e-26 -- but saturation is governed
    # by the gaps between adjacent sorted values near the cut, which can be
    # arbitrarily small, so the bracket is widened until it straddles `a` at
    # both ends. Doubling terminates because `a` is strictly interior.
    S = 60.0 / np.maximum(v.max(axis=1) - v.min(axis=1), 1e-300)
    for _ in range(200):
        short = interior & ((a > _cgf(v, S, _solve_t(v, S, n1))[2])
                            | (a < _cgf(v, -S, _solve_t(v, -S, n1))[2]))
        if not short.any():
            break
        S = np.where(short, S * 2.0, S)

    # Bisection on `s`, re-solving `t` at every step, then a Newton polish
    # that is kept only where it reduces the residual.
    lo, hi = -S, S
    for _ in range(60):
        s = 0.5 * (lo + hi)
        t = _solve_t(v, s, n1)
        below = _cgf(v, s, t)[2] < a
        lo = np.where(below, s, lo)
        hi = np.where(below, hi, s)
    s = 0.5 * (lo + hi)
    t = _solve_t(v, s, n1)

    for _ in range(_OUTER_ITERS):
        _, _, Ks, Kss, Kst, Ktt = _cgf(v, s, t)
        r = Ks - a
        # d(K_s)/ds along the constraint K_t = n1 is the conditional variance
        # of A: `t` moves with `s`, so it is not `Kss`.
        deriv = Kss - Kst * Kst / np.maximum(Ktt, 1e-300)
        if np.all(np.abs(r) <= 1e-6 * np.maximum(np.sqrt(np.maximum(deriv, 0.0)), 1e-12)):
            break
        with np.errstate(over="ignore", invalid="ignore"):
            step = np.where(deriv > 1e-300, r / np.maximum(deriv, 1e-300), 0.0)
        cand = np.where(np.isfinite(step), s - step, s)
        t_cand = _solve_t(v, cand, n1)
        r_cand = _cgf(v, cand, t_cand)[2] - a
        better = np.abs(r_cand) < np.abs(r)
        s = np.where(better, cand, s)
        t = np.where(better, t_cand, t)
        if not better.any():
            break

    K1, Kt_chk, Ks_chk, Kss, Kst, Ktt = _cgf(v, s, t)

    # Both saddlepoint equations must hold: a silent non-convergence returns
    # a plausible number that is wrong by orders of magnitude in the tail.
    # The residual is judged against the conditional standard deviation of
    # A, the scale on which a `K_s` error moves the p-value.
    sd_cond = np.sqrt(np.maximum(Kss - Kst * Kst / np.maximum(Ktt, 1e-300), 0.0))
    tol_a = 1e-3 * np.maximum(sd_cond, 1e-12 * np.maximum(np.abs(a), 1.0))
    bad = interior & ((np.abs(Ks_chk - a) > tol_a) | (np.abs(Kt_chk - n1) > 1e-6 * n1))
    if bad.any():
        worst = float(np.max((np.abs(Ks_chk - a) / np.maximum(tol_a, 1e-300))[bad]))
        raise RuntimeError(
            f"saddlepoint did not converge for {int(bad.sum())} of {g} genes "
            f"(worst residual {worst:.1e}x tolerance). This is a bug, not a "
            f"data condition; please report the matrix shape and n1={n1}."
        )

    # Skovgaard.
    inner = 2.0 * ((K0 - t0 * n1) - (K1 - s * a - t * n1))
    w = np.sign(s) * np.sqrt(np.maximum(inner, 0.0))
    det = np.maximum(Kss * Ktt - Kst * Kst, 0.0)
    u = s * np.sqrt(det / Ktt0)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = norm.sf(w) - norm.pdf(w) * (1.0 / w - 1.0 / u)

    # Near the mean w -> 0 and the correction term is 0/0; the normal
    # approximation is accurate there.
    near = (np.abs(w) < 1e-4) | ~np.isfinite(p)
    if near.any():
        mu_a = n1 * v.sum(axis=1) / n
        sd_a = np.sqrt(np.maximum(sd_cond * sd_cond, 1e-300))
        p = np.where(near, norm.sf((a - mu_a) / sd_a), p)

    # The exact answers last, so nothing overrides them. `1 / C(n, n1)` is
    # formed in log space because C(2000, 1000) overflows a double, and the
    # result is floored at the smallest normal double so a p-value is never
    # exactly zero.
    log_c = lgamma(n + 1) - lgamma(n1 + 1) - lgamma(n0 + 1)
    one_labelling = max(exp(-log_c), np.finfo(float).tiny)
    p = np.where(at_max, _extreme_labellings(v, n1) * one_labelling, p)
    p = np.where(at_min, 1.0, p)
    return np.clip(p, one_labelling, 1.0)


def mean_diff_saddlepoint_p(x, cond, stat=None, *, alternative="two-sided"):
    """Stage 1's p-value with no permutations (``docs/method.md`` §6).

    ``x`` is the normalized ``genes x samples`` matrix, ``cond`` the label
    vector. The statistic is the exact difference of group means, which is the
    signed quantile area computed without quadrature error.

    ``alternative`` is handled the way :func:`wade.pvalues.perm_pvalues` does
    it: ``"greater"`` takes the upper tail, ``"less"`` the lower, and
    ``"two-sided"`` sums both — the exact two-sided permutation probability
    ``P(|T| >= |t|)`` rather than a doubling.
    """
    from .pvalues import ALTERNATIVES
    from .stats import split_groups

    x = np.ascontiguousarray(x, dtype=np.float64)
    cond = np.asarray(cond)
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}; got {alternative!r}")
    i1, i0 = split_groups(cond)
    n1, n0 = i1.size, i0.size
    if stat is None:
        stat = x[:, i1].mean(axis=1) - x[:, i0].mean(axis=1)
    stat = np.asarray(stat, dtype=np.float64)

    # T = A(1/n1 + 1/n0) - V/n0  ->  A = (T + V/n0) / (1/n1 + 1/n0)
    V = x.sum(axis=1)
    to_a = lambda t: (t + V / n0) / (1.0 / n1 + 1.0 / n0)  # noqa: E731

    if alternative == "greater":
        return saddlepoint_subset_sum(x, n1, to_a(stat), upper=True)
    if alternative == "less":
        return saddlepoint_subset_sum(x, n1, to_a(stat), upper=False)
    hi = saddlepoint_subset_sum(x, n1, to_a(np.abs(stat)), upper=True)
    lo = saddlepoint_subset_sum(x, n1, to_a(-np.abs(stat)), upper=False)
    return np.clip(hi + lo, 0.0, 1.0)
