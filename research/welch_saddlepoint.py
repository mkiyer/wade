"""A saddlepoint for the Welch-studentized permutation test of equal means (prototype).

The mean test's null is equal means. Its unstudentized permutation test is exact when the two
groups share a distribution, but with unequal spreads in unbalanced designs it rejects far too
often (docs/subset-null-plan.md, section 2.7). Studentizing by the unpooled standard error,

    T = (mean1 - mean0) / sqrt(var1 / n1 + var0 / n0),

keeps exactness under exchangeability and makes the test asymptotically valid for equal means
(Janssen 1997; Chung and Romano 2013). WADE computes the mean test without shuffles from a double
saddlepoint for the cases' total. T depends on the cases' total A and their sum of squares B, with
the number of cases fixed, so its tail needs the saddlepoint of a smooth function of a bivariate
sum drawn without replacement:

  - Independent Bernoulli selectors conditioned on their count give the joint cumulant transform
    K(s, r, t) = sum_i log(1 + exp(s v_i + r v_i^2 + t)) of (A, B, N).
  - The conditional rate of (A, B) given N = n1 is the Legendre transform of K less that of N.
  - Its minimum on the level curve T(a, b) = tau, found with a Lagrange multiplier lambda, gives
    w = sign(lambda) sqrt(2 rate); the curve's Laplace integral gives
    u = lambda |grad T| sqrt(det K'' / var N) sqrt(H), H the rate's curvature along the curve.
  - P(T >= tau) ~ 1 - Phi(w) + phi(w) (1/u - 1/w) (Lugannani and Rice).

When T depends on A alone the formula is WADE's double saddlepoint for the cases' total.

Sections:

  exact    the tail against exact enumeration of every labelling, small designs and shapes
  shuffles the tail against many shuffles at larger designs, down to small p-values
  solver   continuation against a global search along the level curve, on heavy-tailed genes

    PYTHONPATH=src:research python research/welch_saddlepoint.py [section ...]
"""
import sys
from itertools import combinations
from math import comb, erfc, sqrt

import numpy as np

SQRT2 = sqrt(2.0)


def welch(a, b, n1, n0, total, squares):
    """T from the cases' total a and sum of squares b (arrays), given the gene's total and sum of squares."""
    m1, m0 = a / n1, (total - a) / n0
    v1 = (b - a * a / n1) / (n1 - 1)
    v0 = (squares - b - (total - a) ** 2 / n0) / (n0 - 1)
    return (m1 - m0) / np.sqrt(v1 / n1 + v0 / n0)


def welch_derivatives(a, b, n1, n0, total, squares):
    """T and its first and second derivatives in (a, b)."""
    c = 1 / n1 + 1 / n0
    d = a / n1 - (total - a) / n0
    v = ((b - a * a / n1) / (n1 - 1)) / n1 + ((squares - b - (total - a) ** 2 / n0) / (n0 - 1)) / n0
    va = -2 * a / (n1 * n1 * (n1 - 1)) + 2 * (total - a) / (n0 * n0 * (n0 - 1))
    vb = 1 / (n1 * (n1 - 1)) - 1 / (n0 * (n0 - 1))
    vaa = -2 / (n1 * n1 * (n1 - 1)) - 2 / (n0 * n0 * (n0 - 1))
    root = np.sqrt(v)
    g = d / root
    ga = c / root - d * va / (2 * v * root)
    gb = -d * vb / (2 * v * root)
    gaa = -c * va / (v * root) - d * vaa / (2 * v * root) + 0.75 * d * va * va / (v * v * root)
    gab = -0.5 * c * vb / (v * root) + 0.75 * d * va * vb / (v * v * root)
    gbb = 0.75 * d * vb * vb / (v * v * root)
    return g, ga, gb, gaa, gab, gbb


def _moments(v, s, r, t):
    """Selector probabilities and the sums behind K', K'' at the tilt (s, r, t); rows are genes."""
    z = s[:, None] * v + r[:, None] * v * v + t[:, None]
    p = 0.5 * (1 + np.tanh(0.5 * z))                          # logistic, stable
    q = p * (1 - p)
    v2 = v * v
    grad = np.stack([(p * v).sum(1), (p * v2).sum(1), p.sum(1)], axis=1)
    cols = [v, v2, np.ones_like(v)]
    hess = np.empty((len(v), 3, 3))
    for i in range(3):
        for j in range(i, 3):
            hess[:, i, j] = hess[:, j, i] = (q * cols[i] * cols[j]).sum(1)
    return z, grad, hess


def _softplus(z):
    return np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))


def welch_tail(values, n1, tau, *, steps=16, iterations=60):
    """P(T >= tau) under uniformly random labellings with n1 cases, per row, by the saddlepoint.

    Rows are genes. Values are standardized first (T is unchanged by an affine map of the values).
    The constrained saddlepoint is found by continuation from the centre (tau = 0 at lambda = 0)
    to tau in `steps` stages, each solved by damped Newton on (s, r, t, lambda). Returns the tail
    and a flag marking rows where the solve failed (tau beyond reach, or no convergence).
    """
    values = np.asarray(values, float)
    genes, n = values.shape
    n0 = n - n1
    v = (values - values.mean(1, keepdims=True)) / values.std(1, keepdims=True)
    total, squares = np.zeros(genes), np.full(genes, float(n))
    p0 = n1 / n
    t0 = np.log(n1 / n0)
    tau = np.broadcast_to(np.asarray(tau, float), (genes,)).copy()
    s, r, t, lam = np.zeros(genes), np.zeros(genes), np.full(genes, t0), np.zeros(genes)
    ok = np.ones(genes, bool)
    for stage in range(1, steps + 1):
        target = tau * stage / steps
        for _ in range(iterations):
            z, grad, hess = _moments(v, s, r, t)
            a, b, count = grad[:, 0], grad[:, 1], grad[:, 2]
            g, ga, gb, gaa, gab, gbb = welch_derivatives(a, b, n1, n0, total, squares)
            f = np.stack([count - n1, s - lam * ga, r - lam * gb, g - target], axis=1)
            jac = np.zeros((genes, 4, 4))
            jac[:, 0, :3] = hess[:, 2, :]
            jac[:, 1, :3] = -lam[:, None] * (gaa[:, None] * hess[:, 0, :] + gab[:, None] * hess[:, 1, :])
            jac[:, 1, 0] += 1
            jac[:, 1, 3] = -ga
            jac[:, 2, :3] = -lam[:, None] * (gab[:, None] * hess[:, 0, :] + gbb[:, None] * hess[:, 1, :])
            jac[:, 2, 1] += 1
            jac[:, 2, 3] = -gb
            jac[:, 3, :3] = ga[:, None] * hess[:, 0, :] + gb[:, None] * hess[:, 1, :]
            try:
                step = np.linalg.solve(jac, -f[:, :, None])[:, :, 0]
            except np.linalg.LinAlgError:
                ok[:] = False
                break
            size = np.max(np.abs(f), axis=1)
            if np.all(size < 1e-11 * (1 + np.abs(target)) + 1e-9):
                break
            # damped: halve the step until the residual falls (per row)
            scale = np.ones(genes)
            for _ in range(30):
                s2, r2, t2, l2 = s + scale * step[:, 0], r + scale * step[:, 1], t + scale * step[:, 2], lam + scale * step[:, 3]
                _, g2, _ = _moments(v, s2, r2, t2)
                with np.errstate(invalid="ignore"):
                    gg = welch_derivatives(g2[:, 0], g2[:, 1], n1, n0, total, squares)
                f2 = np.stack([g2[:, 2] - n1, s2 - l2 * gg[1], r2 - l2 * gg[2], gg[0] - target], axis=1)
                worse = ~(np.max(np.abs(f2), axis=1) < size) & (size > 1e-12)
                worse |= ~np.isfinite(f2).all(axis=1)
                if not worse.any():
                    break
                scale = np.where(worse, scale / 2, scale)
            s, r, t, lam = s + scale * step[:, 0], r + scale * step[:, 1], t + scale * step[:, 2], lam + scale * step[:, 3]
    z, grad, hess = _moments(v, s, r, t)
    a, b = grad[:, 0], grad[:, 1]
    g, ga, gb, gaa, gab, gbb = welch_derivatives(a, b, n1, n0, total, squares)
    ok &= (np.abs(g - tau) < 1e-7 * (1 + np.abs(tau))) & np.isfinite(g)
    # the conditional rate, summed per selector as Bregman divergences of the softplus from t0
    delta = z - t0
    bregman = _softplus(z) - _softplus(np.full_like(z, t0)) - p0 * delta
    rate = s * (a - p0 * total) + r * (b - p0 * squares) - bregman.sum(1)
    w = np.sign(lam) * np.sqrt(np.maximum(2 * rate, 0))
    inverse = np.linalg.inv(hess)
    m = inverse[:, :2, :2]
    grad_norm = np.sqrt(ga * ga + gb * gb)
    tx, ty = -gb / grad_norm, ga / grad_norm                  # unit tangent of the level curve
    curvature = (tx * tx * (m[:, 0, 0] - lam * gaa) + 2 * tx * ty * (m[:, 0, 1] - lam * gab)
                 + ty * ty * (m[:, 1, 1] - lam * gbb))
    count_var = n1 * n0 / n
    u = lam * grad_norm * np.sqrt(np.linalg.det(hess) / count_var) * np.sqrt(np.maximum(curvature, 0))
    phi = np.exp(-0.5 * w * w) / sqrt(2 * np.pi)
    upper = 0.5 * np.array([erfc(x / SQRT2) for x in w])
    with np.errstate(divide="ignore", invalid="ignore"):
        tail = upper + phi * (1 / u - 1 / w)
    near = np.abs(w) < 1e-4                                   # at the centre the correction's limit
    tail = np.where(near, 0.5, tail)
    return tail, ok


def exact_tail(values, n1, tau):
    """P(T >= tau) by enumerating every labelling (small n)."""
    values = np.asarray(values, float)
    n = len(values)
    n0 = n - n1
    total, squares = values.sum(), (values ** 2).sum()
    idx = np.array(list(combinations(range(n), n1)))
    a = values[idx].sum(1)
    b = (values[idx] ** 2).sum(1)
    t = welch(a, b, n1, n0, total, squares)
    tau = np.atleast_1d(tau)
    return np.array([np.mean(t >= x - 1e-12 * max(1, abs(x))) for x in tau]), t


def exact(seed=1):
    print("\n== exact: the saddlepoint tail P(T >= tau) against exact enumeration of every labelling; tau at "
          "the exact tail's quantiles 0.1, 0.01, 0.001, 1e-4 (where reachable); ratio saddlepoint / exact")
    rng = np.random.default_rng(seed)
    shapes = {
        "normal": lambda k: rng.normal(size=k),
        "lognormal CV^2 2": lambda k: np.exp(rng.normal(0, np.sqrt(np.log(3)), k)),
        "Gamma CV^2 0.5": lambda k: rng.gamma(2, 1, k),
        "bimodal": lambda k: rng.normal(3, .3, k) + 2 * (rng.random(k) < .5),
    }
    for n1, n0 in ((6, 10), (8, 12), (10, 14), (5, 19)):
        n = n1 + n0
        for name, draw in shapes.items():
            cells = []
            for rep in range(3):
                values = draw(n)
                _, t = exact_tail(values, n1, [0.0])
                ts = np.sort(t)[::-1]
                taus = [ts[int(q * len(ts)) - 1] for q in (0.1, 0.01, 0.001, 1e-4) if q * len(ts) >= 1]
                ex = np.array([np.mean(t >= x - 1e-12 * max(1, abs(x))) for x in taus])
                sp, ok = welch_tail(np.tile(values, (len(taus), 1)), n1, np.array(taus))
                cells.append(" ".join(f"{e:.0e}:{s / e:.2f}{'' if o else '!'}" for e, s, o in zip(ex, sp, ok)))
            print(f"{n1}:{n0} {name:17s} (C = {comb(n, n1):,}) " + " | ".join(cells), flush=True)


def shuffles(seed=2, draws=2_000_000):
    print(f"\n== shuffles: the saddlepoint tail against {draws:,} shuffles; tau at the shuffled tail's quantiles "
          f"0.01, 1e-3, 1e-4, 1e-5; ratio saddlepoint / shuffled")
    rng = np.random.default_rng(seed)
    for n1, n0 in ((20, 60), (50, 150), (30, 300)):
        n = n1 + n0
        for name, values in (("lognormal CV^2 2", np.exp(rng.normal(0, np.sqrt(np.log(3)), n))),
                             ("Gamma CV^2 0.5", rng.gamma(2, 1, n)),
                             ("normal", rng.normal(size=n))):
            total, squares = values.sum(), (values ** 2).sum()
            t = np.empty(draws)
            for start in range(0, draws, 100_000):
                k = min(100_000, draws - start)
                pick = np.argsort(rng.random((k, n)), axis=1)[:, :n1]
                a = values[pick].sum(1)
                b = (values[pick] ** 2).sum(1)
                t[start:start + k] = welch(a, b, n1, n0, total, squares)
            ts = np.sort(t)[::-1]
            taus = [ts[int(q * draws) - 1] for q in (0.01, 1e-3, 1e-4, 1e-5)]
            sh = np.array([np.mean(t >= x) for x in taus])
            sp, ok = welch_tail(np.tile(values, (len(taus), 1)), n1, np.array(taus))
            print(f"{n1}:{n0} {name:17s} " + " | ".join(f"{e:.0e}:{s / e:.2f}{'' if o else '!'}"
                                                         for e, s, o in zip(sh, sp, ok)), flush=True)


def solver(seed=3, genes=200, draws=2_000_000):
    """Continuation against the global search on heavy-tailed genes, and competing minima against shuffles."""
    print(f"\n== solver: {genes} lognormal genes (CV^2 2), 50 v 450, tau = |observed T| + 0.5")
    rng = np.random.default_rng(seed)
    n1, n0 = 50, 450
    n = n1 + n0
    x = np.exp(rng.normal(0, 1.05, (genes, n)))
    cond = np.r_[np.ones(n1), np.zeros(n0)].astype(bool)
    t_obs = (x[:, cond].mean(1) - x[:, ~cond].mean(1)) / np.sqrt(x[:, cond].var(1, ddof=1) / n1
                                                                  + x[:, ~cond].var(1, ddof=1) / n0)
    tau = np.abs(t_obs) + 0.5
    with np.errstate(all="ignore"):
        p1, ok1 = welch_tail(x, n1, tau, steps=8)
        p2, ok2, rival = welch_tail_global(x, n1, tau)
    print(f"converged: continuation {ok1.mean():.1%}, global search {ok2.mean():.1%}")
    print(f"rival weight (a competing local minimum's exp(-rate difference)): median {np.median(rival):.3f}, "
          f"99th percentile {np.quantile(rival, .99):.3f}, largest {rival.max():.3f}; above 0.2 in {np.mean(rival > .2):.1%}")
    both = ok1 & ok2
    pick = np.flatnonzero(both & (np.abs(p2 / p1 - 1) > 0.1) & (p1 > 1e-5) & (p2 > 1e-5))
    trace = np.flatnonzero(~ok1)[:1]
    sh = np.random.default_rng(9)
    for g in pick:
        values = x[g]
        total, squares = values.sum(), (values ** 2).sum()
        hits = 0
        for _ in range(draws // 100_000):
            idx = np.argsort(sh.random((100_000, n)), axis=1)[:, :n1]
            hits += np.sum(welch(values[idx].sum(1), (values[idx] ** 2).sum(1), n1, n0, total, squares) >= tau[g])
        print(f"gene {g}: continuation {p1[g]:.3g}, global {p2[g]:.3g}, {draws:,} shuffles {hits / draws:.3g}; "
              f"rival weight {rival[g]:.2f}")
    for g in trace:
        cells = []
        for frac in np.linspace(0.1, 1, 10):
            with np.errstate(all="ignore"):
                pp, oo = welch_tail(x[g:g + 1], n1, np.array([tau[g] * frac]), steps=8)
            cells.append(f"{tau[g] * frac:.2f}: {pp[0]:.3g}{'' if oo[0] else ' (failed)'}")
        print(f"gene {g}, continuation at rising tau: " + "; ".join(cells))




# ---------------------------------------------------------------------------------------------
# A global search along the level curve (for branch switches)

def _inner(v, a, b, n1, s, r, t, iterations=60):
    """The tilt (s, r, t) at which (A, B, N) has mean (a, b, n1): Newton ascent on the concave
    dual s a + r b + t n1 - K(s, r, t), halving steps that fail to raise it. Rows whose point
    lies outside the attainable hull have no tilt; they come back flagged."""
    def dual(s, r, t):
        z = s[:, None] * v + r[:, None] * v * v + t[:, None]
        return s * a + r * b + t * n1 - _softplus(z).sum(1)
    value = dual(s, r, t)
    done = np.zeros(len(v), bool)
    for _ in range(iterations):
        _, grad, hess = _moments(v, s, r, t)
        gap = np.stack([a - grad[:, 0], b - grad[:, 1], n1 - grad[:, 2]], axis=1)
        done = np.max(np.abs(gap) / (1 + np.abs(np.stack([a, b, np.full_like(a, n1)], 1))), axis=1) < 1e-12
        if done.all():
            break
        ridge = 1e-12 * np.trace(hess, axis1=1, axis2=2)[:, None, None] * np.eye(3) + 1e-300 * np.eye(3)
        hess = np.where(np.isfinite(hess), hess, 0) + ridge
        step = np.linalg.solve(hess, np.where(np.isfinite(gap), gap, 0)[:, :, None])[:, :, 0]
        step = np.where(np.isfinite(step), step, 0)
        scale = np.where(done, 0.0, 1.0)
        for _ in range(40):
            new = dual(s + scale * step[:, 0], r + scale * step[:, 1], t + scale * step[:, 2])
            bad = ~(new >= value - 1e-12 * np.abs(value)) & (scale > 0)
            if not bad.any():
                break
            scale = np.where(bad, scale / 2, scale)
        s, r, t = s + scale * step[:, 0], r + scale * step[:, 1], t + scale * step[:, 2]
        value = dual(s, r, t)
    return s, r, t, done


def welch_tail_global(values, n1, tau, *, angles=48):
    """P(T >= tau) with the minimizer found by a global search along the level curve T = tau.

    In whitened coordinates around the centre (the null mean of the two sums), each ray crosses
    the curve at most once, because {T < tau} is convex and contains the centre. The rate is
    evaluated where each ray crosses, the smallest is refined by golden-section search between
    its neighbours, and the full system is then solved by Newton from there. Returns the tail, a
    convergence flag, and the relative weight of the best competing local minimum (the tail it
    would add if counted), so that a caller can send such genes to shuffles.
    """
    values = np.asarray(values, float)
    genes, n = values.shape
    n0 = n - n1
    v = (values - values.mean(1, keepdims=True)) / values.std(1, keepdims=True)
    p0, t0 = n1 / n, np.log(n1 / n0)
    total, squares = np.zeros(genes), np.full(genes, float(n))
    tau = np.broadcast_to(np.asarray(tau, float), (genes,)).copy()
    # the conditional covariance of (A, B) given N at the centre, and its Cholesky factor
    _, _, hess0 = _moments(v, np.zeros(genes), np.zeros(genes), np.full(genes, t0))
    cov = hess0[:, :2, :2] - hess0[:, :2, 2:3] * hess0[:, 2:3, :2] / hess0[:, 2:3, 2:3]
    chol = np.linalg.cholesky(cov)
    centre = np.stack([np.zeros(genes), np.full(genes, float(n1))], axis=1)

    def crossing(phi):
        """The point where the ray at angle phi (whitened) crosses T = tau, or NaN if it never does."""
        e = np.stack([np.cos(phi), np.sin(phi)], axis=-1)
        d = np.einsum("gij,gj->gi", chol, e)
        lo, hi = np.zeros(genes), np.full(genes, 1.0)
        def t_at(rho):
            p = centre + rho[:, None] * d
            with np.errstate(invalid="ignore", divide="ignore"):
                return welch(p[:, 0], p[:, 1], n1, n0, total, squares)
        for _ in range(60):                                   # grow hi until T >= tau or V fails
            val = t_at(hi)
            grow = np.isfinite(val) & (val < tau) & (hi < 1e6)
            if not grow.any():
                break
            hi = np.where(grow, hi * 2, hi)
        val = t_at(hi)
        reach = np.isfinite(val) & (val >= tau)
        # shrink hi to the last finite point where needed (V fails beyond the hull)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            vm = t_at(mid)
            above = np.isfinite(vm) & (vm >= tau)
            hi = np.where(reach & above, mid, hi)
            lo = np.where(reach & ~above, mid, lo)
        point = centre + hi[:, None] * d
        return np.where(reach[:, None], point, np.nan)

    def rate_at(point, guess):
        a, b = point[:, 0], point[:, 1]
        bad = ~np.isfinite(a)
        a, b = np.where(bad, 0, a), np.where(bad, n1, b)
        s, r, t, done = _inner(v, a, b, n1, *guess)
        z = s[:, None] * v + r[:, None] * v * v + t[:, None]
        bregman = _softplus(z) - _softplus(np.full_like(z, t0)) - p0 * (z - t0)
        rate = s * (a - p0 * total) + r * (b - p0 * squares) - bregman.sum(1)
        rate = np.where(bad | ~done, np.inf, rate)
        return rate, (s, r, t)

    grid = np.linspace(0, 2 * np.pi, angles, endpoint=False)
    rates = np.empty((genes, angles))
    zero = (np.zeros(genes), np.zeros(genes), np.full(genes, t0))
    for k, phi in enumerate(grid):
        rates[:, k], _ = rate_at(crossing(np.full(genes, phi)), zero)
    best = np.argmin(rates, axis=1)
    # competing local minima on the grid: rate of the best other local minimum
    left, right = np.roll(rates, 1, axis=1), np.roll(rates, -1, axis=1)
    local = (rates <= left) & (rates <= right) & np.isfinite(rates)
    local[np.arange(genes), best] = False
    rival = np.where(local.any(1), np.min(np.where(local, rates, np.inf), axis=1), np.inf)
    # golden-section refinement between the best's neighbours
    step = 2 * np.pi / angles
    lo, hi = grid[best] - step, grid[best] + step
    golden = (np.sqrt(5) - 1) / 2
    for _ in range(40):
        c, d = hi - golden * (hi - lo), lo + golden * (hi - lo)
        rc, _ = rate_at(crossing(c), zero)
        rd, _ = rate_at(crossing(d), zero)
        take_left = rc < rd
        hi = np.where(take_left, d, hi)
        lo = np.where(take_left, lo, c)
    phi = 0.5 * (lo + hi)
    point = crossing(phi)
    rate, (s, r, t) = rate_at(point, zero)
    a, b = point[:, 0], point[:, 1]
    g, ga, gb, gaa, gab, gbb = welch_derivatives(a, b, n1, n0, total, squares)
    lam = s / ga                                                # (s, r) = lambda grad T at the minimum
    _, grad, hess = _moments(v, s, r, t)
    w = np.sign(lam) * np.sqrt(np.maximum(2 * rate, 0))
    inverse = np.linalg.inv(hess)
    m = inverse[:, :2, :2]
    grad_norm = np.sqrt(ga * ga + gb * gb)
    tx, ty = -gb / grad_norm, ga / grad_norm
    curvature = (tx * tx * (m[:, 0, 0] - lam * gaa) + 2 * tx * ty * (m[:, 0, 1] - lam * gab)
                 + ty * ty * (m[:, 1, 1] - lam * gbb))
    u = lam * grad_norm * np.sqrt(np.linalg.det(hess) / (n1 * n0 / n)) * np.sqrt(np.maximum(curvature, 0))
    phi_w = np.exp(-0.5 * w * w) / sqrt(2 * np.pi)
    upper = 0.5 * np.array([erfc(x / SQRT2) for x in w])
    with np.errstate(divide="ignore", invalid="ignore"):
        tail = upper + phi_w * (1 / u - 1 / w)
    ok = np.isfinite(tail) & (curvature > 0) & np.isfinite(rate) & (np.abs(s - lam * ga) + np.abs(r - lam * gb) < 1e-6 * (1 + np.abs(s) + np.abs(r)))
    # the competing minimum's weight relative to the best: exp(-(rival - rate)) to leading order
    rival_weight = np.exp(-(rival - rate))
    return tail, ok, rival_weight


SECTIONS = dict(exact=exact, shuffles=shuffles, solver=solver)

if __name__ == "__main__":
    for section in sys.argv[1:] or list(SECTIONS):
        SECTIONS[section]()
