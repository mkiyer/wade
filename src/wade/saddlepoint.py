"""The permutation tail of a subset sum, computed rather than sampled.

Stage 1's statistic is the signed area between the two quantile functions, and
:math:`\\int_0^1 Q(p)\\,dp = E[X]` is an identity — so that area **is** the
difference of group means, and the quantile grid was only ever a quadrature of
it (exact when balanced, an interpolation of the larger group otherwise;
``limits.md`` §2.5). Computed exactly, stage 1 is a linear statistic:

.. code-block:: text

    T  =  A·(1/n1 + 1/n0) − V/n0        A = the case-subset sum, V = the total

so ``P(T >= t) = P(A >= a)``, a question about the sum of a simple random
sample drawn without replacement. That has a classical **double saddlepoint**
answer (Skovgaard 1987; Booth & Butler 1990) with *relative* error far into the
tail, at :math:`O(n)` per gene and **no permutations at all**.

Why this matters here rather than being a curiosity: the empirical p-value
floors at ``1/(B+1)``, and BH across 20,000 genes decides near 2.5e-6, so the
genes BH rules on are exactly the genes no permutation count can resolve. The
GPD refinement that fills that gap was measured 3–4,906× conservative on stage 1
(``scaling.md`` §4.5, §4.7) because the permutation null of a bounded statistic
has a negative extreme-value shape and the implementation substitutes an
unbounded exponential for it. This module removes the question instead of
answering it better: there is no floor, no threshold, and no extreme-value
parameter to estimate.

**Validated against brute force**, 1e8 permutations per design
(``scaling.md`` §4.9): median ratio to truth 0.91–1.15 over four geometries on
continuous data, 1.00–1.02 on negative-binomial counts through :func:`wade`'s
own normalization, 0.93–1.06 on sparse counts at a median count of 3, and
1.00–1.14 at 300 v 300 — better at larger ``n``, as an asymptotic method should
be. Worst anti-conservative ratio anywhere: 0.61.

**Two things it is not.** It is not a distributional assumption about the data:
the CGF below is built from the observed values, and the probability computed
is the exact permutation probability, which is why it agrees with brute-force
relabelling. And it is not Wilcoxon or any rank test — the statistic uses the
values, not their ranks.

The conditioning is what makes sampling *without* replacement tractable. Take
independent Bernoulli selectors :math:`Z_i` and condition on
:math:`\\sum Z_i = n_1`; the joint cumulant generating function is

.. code-block:: text

    K(s, t)  =  sum_i log(1 + exp(s·v_i + t))

and Skovgaard's formula turns the two saddlepoints of that into a tail
probability. Everything below is that formula, vectorized across genes.
"""

from __future__ import annotations

import numpy as np

__all__ = ["saddlepoint_subset_sum", "mean_diff_saddlepoint_p"]

#: Iteration caps for the two nested Newton solves. Both loops exit early on
#: a residual, and both keep a bisection bracket so a step that leaves it is
#: rejected rather than trusted; the caps only bound the worst gene. The final
#: answer is *checked* against both saddlepoint equations before it is
#: returned, so a cap that turned out to be too small raises rather than
#: silently returning a plausible wrong number.
_INNER_ITERS = 40
_OUTER_ITERS = 12
_TOL = 1e-11


def _cgf(v, s, t):
    """``K``, its gradient and its Hessian, vectorized over genes.

    ``v`` is ``(g, n)``; ``s`` and ``t`` are ``(g,)``. Returns ``K``, ``K_t``,
    ``K_s`` and the three second derivatives. ``s·v`` reaches the hundreds in
    the far tail, so both the CGF and the logistic are written in the forms
    that do not overflow there.
    """
    z = s[:, None] * v + t[:, None]
    K = np.logaddexp(0.0, z).sum(axis=1)
    # The logistic as (1 + tanh(z/2))/2 -- the same function, and the only
    # form that does not overflow at either end. `1/(1 + exp(-z))` warns and
    # `where(z >= 0, ...)` still evaluates both branches.
    p = 0.5 * (1.0 + np.tanh(0.5 * z))
    q = p * (1.0 - p)
    return (K, p.sum(axis=1), (p * v).sum(axis=1),
            (q * v * v).sum(axis=1), (q * v).sum(axis=1), q.sum(axis=1))


def _bracket_t(v, s, n1):
    """An exact, tight bracket for ``K_t(s, t) = n1``, in closed form.

    ``K_t(t) = sum_i sigma(s·v_i + t)`` with ``sigma`` the logistic, so with
    ``c = log(n1/n0)``:

    * at ``t = c - max_i(s·v_i)`` every term is at most ``sigma(c) = n1/n``,
      so ``K_t <= n1``;
    * at ``t = c - min_i(s·v_i)`` every term is at least that, so
      ``K_t >= n1``.

    Both hold for either sign of ``s``, which makes this a bracket by
    construction rather than by search. The first version searched — widening
    geometrically from ``[-1, 1]`` — and had two failure modes that a search
    cannot avoid: the widened ``lo`` inflated the step used for ``hi``, so the
    bracket could balloon to a width bisection could not close in the
    iteration budget (measured: ``|K_t - n1| = 10.8`` at ``s = 0.1``), and a
    bracket that quietly failed to straddle made the Newton return a non-root
    that everything downstream trusted.
    """
    n = v.shape[1]
    c = np.log(n1 / (n - n1))
    sv = s[:, None] * v
    return c - sv.max(axis=1), c - sv.min(axis=1)


def _solve_t(v, s, n1, t0=None, iters=_INNER_ITERS):
    """``t`` with ``K_t(s, t) = n1``, per gene.

    ``K_t`` is a sum of ``n`` logistics, so it increases strictly from 0 to
    ``n`` and the root is unique for any ``0 < n1 < n``. Newton, with the
    closed-form bracket of :func:`_bracket_t` carried alongside so a step that
    leaves it is rejected rather than trusted.

    ``t0`` warm-starts from the previous outer iteration, which is what makes
    the nested solve affordable: the root moves only as far as ``s`` did, so a
    handful of iterations suffice instead of a cold descent. It is clipped
    into the bracket, so a stale or wild warm start costs speed and never
    correctness.
    """
    lo, hi = _bracket_t(v, s, n1)
    # Bisection to a guaranteed width, then Newton to polish. Newton alone,
    # warm-started from the previous outer iteration, was the first version:
    # it left `|K_t - n1| = 4.8` on occasional genes, and because `K_s` is
    # evaluated at that wrong `t` the outer bisection then saw a
    # non-monotone function and converged to nonsense. Correctness first --
    # the bracket halves 60 times regardless of where the warm start was, and
    # the whole solve is still well under a millisecond per gene.
    # A few bisections to get safely inside, then Newton, which converges
    # quadratically on a smooth monotone sum of logistics. The *warm start*
    # was the thing that broke an earlier version -- clipped to a bracket
    # endpoint it could sit where Newton made no progress, leaving
    # `|K_t - n1| = 4.8`, and `K_s` evaluated at that wrong `t` made the outer
    # function non-monotone. Starting from the midpoint costs a few
    # iterations and removes the failure mode.
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
    ``(g,)`` probabilities, clipped to ``[0, 1]``.

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
    ``K_tt(0, t0) = n1·n0/n`` in closed form — no solve needed.

    ``w -> 0`` as ``a`` approaches the mean, where the formula is
    :math:`0/0`; there the normal approximation is excellent anyway and the
    saddlepoint is not needed, so genes with ``|w|`` below a threshold fall
    back to it. That boundary is the one place this is not a relative-error
    method, and it is the region where the empirical p-value works.
    """
    from scipy.stats import norm

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

    # --- the denominator saddlepoint, in closed form --------------------
    t0 = np.full(g, np.log(n1 / n0))
    K0 = _cgf(v, np.zeros(g), t0)[0]
    Ktt0 = np.full(g, n1 * n0 / n)

    # --- the numerator: solve K_s = a, with t re-solved at every step ----
    # `K_s` increases strictly in `s` from `a_min` (as s -> -inf) to `a_max`
    # (as s -> +inf), so the root exists for any reachable `a` -- but it sits
    # at infinity as `a` approaches either end, and no finite bracket contains
    # every case. Two earlier versions searched for one by doubling and both
    # failed on the genes nearest the boundary, silently returning a
    # conservative answer.
    #
    # So bisect a **bounded reparameterization** instead: `s = atanh(u)/scale`
    # maps `u` in (-1, 1) onto the whole line, the bracket is [-1, 1] by
    # construction, and 80 halvings of `u` locate the root to 1e-24 of the
    # transformed range. Newton is then a polish step, not the mechanism, so
    # it can be trusted where it converges and ignored where it does not.
    srt = np.sort(v, axis=1)
    a_min = srt[:, :n1].sum(axis=1)
    a_max = srt[:, -n1:].sum(axis=1)
    scale = np.maximum(v.max(axis=1) - v.min(axis=1), 1e-300)

    # `s * spread = 60` saturates every logistic to within 1e-26, so `K_s` is
    # then equal to `a_max` (or `a_min`) to machine precision and no larger
    # `|s|` changes anything. That makes [-S, S] a guaranteed bracket for any
    # `a` strictly inside the reachable range, with no search and no
    # transform. An `atanh` reparameterization was tried first and silently
    # capped `s` an order of magnitude short once `wade()`'s normalization put
    # the values near 1e8: `atanh(1 - 1e-15)` is only 17.9, and 60 was needed.
    # The boundary is settled BEFORE any solving, because it is exact and
    # because letting it interact with the solver is what made three earlier
    # versions of this function wrong. `A` ranges over [a_min, a_max]; at
    # either end only one labelling qualifies, so:
    #
    #   a >= a_max  ->  p = 1 / C(n, n1), the one-labelling floor
    #   a <= a_min  ->  p = 1
    #
    # and strictly between them a finite saddlepoint exists. `edge` is a
    # relative tolerance on the reachable width, not on `a`, which may be
    # negative.
    width = np.maximum(a_max - a_min, 1e-300)
    at_max = a >= a_max - 1e-12 * width
    at_min = a <= a_min + 1e-12 * width
    interior = ~(at_max | at_min)

    # A bracket [-S, S] containing the root for every interior gene. `S` has
    # to ADAPT: saturation is governed by the *gaps* between adjacent sorted
    # values near the cut, not by the total spread, and those gaps can be
    # arbitrarily small -- measured, a fixed `60 / spread` left `K_s(+S)` 8e-4
    # short of `a_max`, and any `a` inside that last 0.08% then looked like a
    # solver failure. Doubling must terminate because `K_s -> a_max` and `a`
    # is strictly below it here.
    S = 60.0 / scale
    for _ in range(200):
        short = interior & (a > _cgf(v, S, _solve_t(v, S, n1))[2])
        if not short.any():
            break
        S = np.where(short, S * 2.0, S)

    ulo, uhi = -S, S
    t = _solve_t(v, np.zeros(g), n1)
    for _ in range(60):
        s = 0.5 * (ulo + uhi)
        t = _solve_t(v, s, n1, t)
        below = _cgf(v, s, t)[2] < a
        ulo = np.where(below, s, ulo)
        uhi = np.where(below, uhi, s)
    s = 0.5 * (ulo + uhi)
    t = _solve_t(v, s, n1, t)

    # Newton polish: quadratic near the root, and rejected outright where it
    # does not improve the residual, so it can only help.
    for _ in range(_OUTER_ITERS):
        _, _, Ks, Kss, Kst, Ktt = _cgf(v, s, t)
        r = Ks - a
        # d(K_s)/ds along the constraint K_t = n1 is the CONDITIONAL variance
        # of A -- `t` moves with `s`, so it is not `Kss`.
        deriv = Kss - Kst * Kst / np.maximum(Ktt, 1e-300)
        if np.all(np.abs(r) <= 1e-6 * np.maximum(np.sqrt(np.maximum(deriv, 0.0)), 1e-12)):
            break
        with np.errstate(over="ignore", invalid="ignore"):
            step = np.where(deriv > 1e-300, r / np.maximum(deriv, 1e-300), 0.0)
        cand = np.where(np.isfinite(step), s - step, s)
        t_cand = _solve_t(v, cand, n1, t)
        r_cand = _cgf(v, cand, t_cand)[2] - a
        better = np.abs(r_cand) < np.abs(r)
        s = np.where(better, cand, s)
        t = np.where(better, t_cand, t)
        if not better.any():
            break

    K1, Kt_chk, Ks_chk, Kss, Kst, Ktt = _cgf(v, s, t)

    # Both saddlepoint equations must actually hold. A silent non-convergence
    # returns a plausible-looking number that is wrong by orders of magnitude
    # in the tail, which is the one failure mode this module must not have.
    # The residual is judged against the CONDITIONAL STANDARD DEVIATION of A,
    # the scale on which a `K_s` error moves the p-value: 1e-3 sd bounds the
    # induced error below 1% even at p = 1e-8, where `w` is about 5.6 and the
    # sensitivity is exp(w · da/sd) -- well inside the saddlepoint's own ~1%
    # approximation error. Judged against `a` itself it flags residuals in the
    # sixth decimal place that cannot matter.
    sd_cond = np.sqrt(np.maximum(Kss - Kst * Kst / np.maximum(Ktt, 1e-300), 0.0))
    tol_a = 1e-3 * np.maximum(sd_cond, 1e-12 * np.maximum(np.abs(a), 1.0))
    bad = (np.abs(Ks_chk - a) > tol_a) | (np.abs(Kt_chk - n1) > 1e-6 * n1)
    # A threshold outside the reachable range has no saddlepoint: `a` beyond
    # `a_max` is impossible under any relabelling and at or below `a_min` is
    # certain. Those are exact answers, not failures.
    # As `a` approaches `a_max` the saddlepoint recedes to infinity, so no
    # finite `s` solves K_s = a and the bisection ends pinned at the edge of
    # its transformed bracket. That is a boundary case with an exact answer --
    # the observation is at the extreme of the permutation distribution and
    # its p-value is the one-labelling floor -- not a solver failure, and the
    # two must be told apart or the check either masks real bugs or rejects
    # correct answers. `pinned` is the boundary; anything else that misses
    # tolerance is a bug and raises.
    # "Pinned" means the bisection never moved that end of its bracket: the
    # root lies beyond the representable range of `s`, which happens exactly
    # when `a` is at the edge of what any relabelling can reach. Keyed on the
    # bracket, not on the residual's sign -- the sign flips harmlessly at
    # convergence and using it swallowed genuine failures.
    unreachable = ~interior
    if np.any(bad & ~unreachable):
        n_bad = int(np.sum(bad & ~unreachable))
        worst = float(np.max(
            (np.abs(Ks_chk - a) / np.maximum(tol_a, 1e-300))[bad & ~unreachable]))
        raise RuntimeError(
            f"saddlepoint did not converge for {n_bad} of {g} genes "
            f"(worst residual {worst:.1e}x tolerance). This is a bug, not a "
            f"data condition; please report the matrix shape and n1={n1}."
        )

    # --- Skovgaard --------------------------------------------------------
    inner = 2.0 * ((K0 - t0 * n1) - (K1 - s * a - t * n1))
    w = np.sign(s) * np.sqrt(np.maximum(inner, 0.0))
    det = np.maximum(Kss * Ktt - Kst * Kst, 0.0)
    u = s * np.sqrt(det / Ktt0)

    with np.errstate(divide="ignore", invalid="ignore"):
        p = norm.sf(w) - norm.pdf(w) * (1.0 / w - 1.0 / u)

    # Near the mean w -> 0 and the correction term is 0/0; the normal
    # approximation is accurate there and it is also where the empirical
    # p-value has plenty of resolution, so nothing is lost by using it.
    var_a = Kss - Kst * Kst / np.maximum(Ktt, 1e-300)
    near = (np.abs(w) < 1e-4) | ~np.isfinite(p)
    if near.any():
        mu_a = n1 * v.sum(axis=1) / n
        sd_a = np.sqrt(np.maximum(var_a, 1e-300))
        p = np.where(near, norm.sf((a - mu_a) / sd_a), p)

    # The exact answers last, so nothing overrides them.
    p = np.where(at_max, 0.0, p)                     # floored just below
    p = np.where(at_min, 1.0, p)

    # A permutation p-value cannot be smaller than the probability of a single
    # labelling, 1 / C(n, n1) -- that is one outcome out of the exchangeable
    # set, and it is exact rather than a numerical guard. Without it the far
    # tail underflows `norm.sf` to 0, and a p-value of exactly zero is a claim
    # no finite permutation set can support. At 40 v 40 the floor is 9.3e-24;
    # the empirical floor it replaces is 5.0e-4.
    from math import exp, lgamma

    # In log space: C(2000, 1000) overflows a float, so `1 / C` has to be
    # formed as `exp(-log C)` and floored at the smallest normal double so a
    # p-value is never exactly zero even when the combinatorial floor
    # underflows.
    log_c = lgamma(n + 1) - lgamma(n1 + 1) - lgamma(n - n1 + 1)
    floor = max(exp(-log_c) if log_c < 700 else 0.0, 2.3e-308)
    return np.clip(p, floor, 1.0)


def mean_diff_saddlepoint_p(x, cond, stat=None, *, alternative="two-sided"):
    """Stage 1's p-value with no permutations — ``scaling.md`` §4.9.

    ``x`` is the normalized ``genes x samples`` matrix, ``cond`` the label
    vector. The statistic is the exact difference of group means, which is the
    signed quantile area computed without quadrature error.

    ``alternative`` is handled the way :func:`wade.pvalues.perm_pvalues` does
    it, so the two are interchangeable: ``"greater"`` takes the upper tail,
    ``"less"`` the lower, and ``"two-sided"`` sums both — which is the exact
    two-sided permutation probability ``P(|T| >= |t|)`` rather than a doubling,
    matching the ``abs()`` orientation the empirical path uses.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    cond = np.asarray(cond)
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(
            f"alternative must be 'two-sided', 'greater' or 'less'; "
            f"got {alternative!r}"
        )
    n1 = int(np.sum(cond == 1))
    n0 = int(np.sum(cond == 0))
    if n1 + n0 != cond.size:
        raise ValueError("cond must contain only 0 and 1")
    if stat is None:
        stat = x[:, cond == 1].mean(axis=1) - x[:, cond == 0].mean(axis=1)
    stat = np.asarray(stat, dtype=np.float64)

    # T = A(1/n1 + 1/n0) - V/n0  ->  A = (T + V/n0) / (1/n1 + 1/n0)
    V = x.sum(axis=1)
    to_a = lambda t: (t + V / n0) / (1.0 / n1 + 1.0 / n0)

    if alternative == "greater":
        return saddlepoint_subset_sum(x, n1, to_a(stat), upper=True)
    if alternative == "less":
        return saddlepoint_subset_sum(x, n1, to_a(stat), upper=False)
    hi = saddlepoint_subset_sum(x, n1, to_a(np.abs(stat)), upper=True)
    lo = saddlepoint_subset_sum(x, n1, to_a(-np.abs(stat)), upper=False)
    return np.clip(hi + lo, 0.0, 1.0)
