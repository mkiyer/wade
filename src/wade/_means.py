"""The area: each gene's signed area between the case and control quantile functions, and its test.

The signed area, the integral over u of Q_case(u) - Q_control(u), is exactly the difference in
the group means: an empirical quantile function is a step function whose integral is the mean,
whatever the group sizes. Its permutation test is therefore a question about the cases' total, a
sum of n1 values drawn without replacement: shuffling answers it by brute force, and the
saddlepoint of that sum's distribution answers it without shuffles.

The test runs on counts thinned to one sequencing depth (equalized): read counts carry Poisson
noise that shrinks with depth, so samples sequenced to different depths are not exchangeable at
low counts even when nothing differs between the groups. Thinning every sample to the shallowest
depth makes them exchangeable again, and washes out genes whose counts are too low to trust.
"""
from math import erfc, exp, lgamma, log, pi, sqrt

import numpy as np

from .permutation import draw_perms
from .pvalues import sequential_p

DIRECTIONS = ("both", "up", "down")
EPS, TINY = np.finfo(float).eps, np.finfo(float).tiny
TIE_RTOL = 100 * EPS
_ERFC = np.frompyfunc(erfc, 1, 1)


def jittered(counts, rows, seed):
    """Counts plus independent Uniform(0, 1) jitter, each gene's from its own stream (seed, row index).

    Counts are discrete, and many samples share one, above all zero. The subset test's null moves
    each case by a fold and would pull such ties apart, which the observed data never are;
    jittered values are all distinct. The jitter does not depend on the gene chunk.
    """
    out = np.array(counts, dtype=float)
    for row, index in zip(out, rows):
        row += np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(index,))).random(len(row))
    return out


def expression(counts, factors):
    """Normalized counts: count / size_factor."""
    with np.errstate(over="ignore", invalid="ignore"):
        x = counts / factors
    if not np.isfinite(x).all():
        raise ValueError("counts / size_factors exceeds the finite expression range")
    return x


def equalized(counts, rows, factors, seed):
    """Counts thinned binomially to the shallowest sample's depth, each gene from its own stream
    (seed, row index, 1).

    Sample j keeps each read with chance min(s) / s_j, s its size factor. A thinned Poisson count
    is Poisson at the lower depth, so every sample ends at one depth and a difference in depth
    between the groups cannot pass for a difference in expression. Counts are rounded to whole
    reads first. factors holds one size factor per sample, or one row of them per gene. The draws
    do not depend on the gene chunk.
    """
    keep = np.broadcast_to(factors.min(axis=-1, keepdims=True) / factors, np.shape(counts))
    out = np.empty(np.shape(counts), np.int64)
    for i, (row, index) in enumerate(zip(counts, rows)):
        out[i] = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(index, 1))).binomial(
            np.rint(row).astype(np.int64), keep[i])
    return out


def area_values(counts, rows, factors, seed):
    """The values the area test shuffles: equalized counts plus jitter, over the common depth."""
    return expression(jittered(equalized(counts, rows, factors, seed), rows, seed),
                      factors.min(axis=-1, keepdims=True))


def mean_test(x, cond, *, direction, stop, maximum, seed, batch=4096):
    """Permutation p-values for each gene's signed area, the difference in means.

    Rows are genes. direction "both" counts the shuffles whose difference
    lies at least as far from zero as the observed one, "up" those at least as high, and "down"
    those at least as low. Every gene sees the same shuffles and stops at its stop-th
    exceedance or at maximum (Besag-Clifford). Returns the p-values and the exceedance and
    shuffle counts behind them.
    """
    cond = np.asarray(cond)
    n1, n0 = int(cond.sum()), int((cond == 0).sum())
    observed = x[:, cond == 1].mean(axis=1) - x[:, cond == 0].mean(axis=1)
    total = x.sum(axis=1)
    rounding = TIE_RTOL * np.abs(x).sum(axis=1) / min(n1, n0)     # a shuffle within it ties
    stop = min(stop, maximum)
    exceed, tried = np.zeros(len(x), np.int64), np.zeros(len(x), np.int64)
    live, rng, drawn = np.ones(len(x), bool), np.random.default_rng(seed), 0
    while live.any():
        labels = draw_perms(cond, min(batch, maximum - drawn), rng=rng)
        drawn += len(labels)
        genes = np.flatnonzero(live)
        sums = x[genes] @ labels.T.astype(float)                  # the cases' total in each shuffle
        shuffled = sums / n1 - (total[genes, None] - sums) / n0
        mark, tie = observed[genes, None], rounding[genes, None]
        if direction == "up":
            hits = shuffled >= mark - tie
        elif direction == "down":
            hits = shuffled <= mark + tie
        else:
            hits = np.abs(shuffled) >= np.abs(mark) - tie
        count = exceed[genes, None] + np.cumsum(hits, axis=1)
        reached = count >= stop
        used = np.where(reached.any(axis=1), reached.argmax(axis=1) + 1, len(labels))
        exceed[genes], tried[genes] = count[np.arange(len(genes)), used - 1], tried[genes] + used
        live[genes] = (exceed[genes] < stop) & (tried[genes] < maximum)
    return sequential_p(exceed, tried, stop=stop, maximum=maximum), exceed, tried


def mean_saddlepoint(x, cond, *, direction):
    """Each gene's exact permutation p-value for the mean shift by Skovgaard's double saddlepoint,
    and that p-value's estimated relative error.

    The cases' total is a sum of n1 values drawn without replacement: independent Bernoulli
    selectors conditioned on their count, with the joint cumulant generating function
    K(s, t) = sum_i log(1 + exp(s x_i + t)). Its saddlepoint gives the tail probability from
    O(n) work per gene and no shuffles, and no p-value falls below the chance of the most
    extreme labelling. The error estimate is the larger of two: the saddlepoint's own next term
    (the second-order correction of the joint density at the saddlepoint, less that of the count
    alone), and the granularity of the exact distribution it smooths, one labelling among the
    L = p C(n, n1) at least as extreme. direction has the meaning it has in mean_test.
    """
    cond = np.asarray(cond)
    n1 = int(cond.sum())
    total = x[:, cond == 1].sum(axis=1)
    if direction == "up":
        p, error = _upper_tail(x, n1, total)
    elif direction == "down":
        p, error = _upper_tail(-x, n1, -total)
    else:
        centre = n1 * x.mean(axis=1)                          # the cases' total, averaged over labellings
        gap = np.abs(total - centre)
        (above, error_above), (below, error_below) = _upper_tail(x, n1, centre + gap), _upper_tail(-x, n1, gap - centre)
        p = np.minimum(above + below, 1.0)
        error = (above * error_above + below * error_below) / (above + below)
    labellings = _log_choose(x.shape[1], n1)
    with np.errstate(over="ignore"):
        granularity = np.exp(-(np.log(p) + labellings))       # 1/L for the L labellings in the tail
    return p, np.where(error > 0, np.maximum(error, granularity), 0.0)


def _upper_tail(values, n1, total):
    """P(A >= total) per row, for A the sum of n1 of the row's values drawn without replacement,
    and the saddlepoint's estimated relative error (0 where the answer is exact)."""
    n = values.shape[1]
    centre, spread = values.mean(axis=1), values.std(axis=1)
    flat = spread <= TIE_RTOL * np.abs(values).max(axis=1)       # every labelling gives the same total
    spread = np.where(flat, 1.0, spread)
    v = (values - centre[:, None]) / spread[:, None]
    a = (total - n1 * centre) / spread
    ordered = np.sort(v, axis=1)
    reach = TIE_RTOL * n * np.abs(v).max(axis=1)                  # the rounding of a sum of n values
    highest = ordered[:, -n1:].sum(axis=1)
    beyond = ~flat & (a > highest + reach)                        # no labelling gets there
    top = ~flat & ~beyond & (a >= highest - reach)
    inside = ~flat & ~beyond & ~top & (a > ordered[:, :n1].sum(axis=1) + reach)
    p, error = np.where(beyond, 0.0, 1.0), np.zeros(len(v))
    p[top] = [_most_extreme(row, n1) for row in ordered[top]]
    if inside.any():
        p[inside], error[inside] = _saddlepoint(v[inside], n1, a[inside])
    return p, error


def _most_extreme(ordered, n1):
    """The exact chance of the largest total: the labellings taking the n1 largest values, ties included."""
    n = len(ordered)
    last = ordered[n - n1]
    tied = np.abs(ordered - last) <= TIE_RTOL * np.abs(ordered).max()
    within = int(np.sum(tied[n - n1:]))
    return max(exp(_log_choose(int(tied.sum()), within) - _log_choose(n, n1)), TINY)


def _log_choose(n, k):
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def _saddlepoint(v, n1, a):
    """The double-saddlepoint tail and its estimated error, for standardized rows strictly inside the reachable range."""
    n = v.shape[1]
    n0 = n - n1
    s, t, _ = _solve(v, n1, a)
    # w^2 / 2 = phi(0, t0) - phi(s, t) = s a - sum_i B_i, with B_i each selector's Bregman
    # divergence of the softplus from t0. Summed per selector it keeps full precision; the
    # difference of the two sums of n softplus terms would lose it near the centre.
    p0, t0 = n1 / n, log(n1 / n0)
    delta = s[:, None] * v + (t - t0)[:, None]
    softplus = lambda z: np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))
    near = delta <= 1
    bregman = np.where(near, np.log1p(p0 * np.expm1(np.minimum(delta, 1))),
                       softplus(t0 + delta) - softplus(t0)) - p0 * delta
    hessian, third, fourth = _derivatives(v, s, t)
    w = np.sign(s) * np.sqrt(np.maximum(2 * (s * a - bregman.sum(axis=1)), 0.0))
    u = s * np.sqrt(np.maximum(np.linalg.det(hessian), 0.0) / (n1 * n0 / n))
    # 1/w - 1/u is 0/0 at the centre, where its limit is the count-conditioned skewness over 6;
    # the crossover sits where the rounding of the difference meets the limit's own error.
    q0 = p0 * (1 - p0)
    limit = q0 * (1 - 2 * p0) * (v ** 3).sum(axis=1) / (6 * (q0 * n) ** 1.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        correction = np.where(np.abs(s) < EPS ** (1 / 3), limit, 1 / w - 1 / u)
    tail = 0.5 * _ERFC(w / sqrt(2)).astype(float)
    p = tail - np.exp(-0.5 * w * w) / sqrt(2 * pi) * correction
    floor = max(exp(-_log_choose(n, n1)), TINY)             # no p-value is exactly zero
    # Second-order density terms (McCullagh's rho4, rho13^2, rho23^2) of the joint cumulants in
    # (s, t), less the count's own (Daniels' lambda4/8 - 5 lambda3^2/24).
    inverse = np.linalg.inv(hessian)
    rho4 = np.einsum("gabcd,gab,gcd->g", fourth, inverse, inverse)
    rho13 = np.einsum("gabc,gdef,gab,gcd,gef->g", third, third, inverse, inverse, inverse)
    rho23 = np.einsum("gabc,gdef,gad,gbe,gcf->g", third, third, inverse, inverse, inverse)
    lambda3, lambda4 = (1 - 2 * p0) / sqrt(q0 * n), (1 - 6 * q0) / (q0 * n)
    error = np.abs(rho4 / 8 - (2 * rho23 + 3 * rho13) / 24 - (lambda4 / 8 - 5 * lambda3 ** 2 / 24))
    return np.clip(p, floor, 1.0), error


def _cgf(v, s, t):
    """K(s, t) per row, with each selector's probability p and variance q at (s, t)."""
    z = s[:, None] * v + t[:, None]
    e = np.exp(-np.abs(z))
    return (np.maximum(z, 0) + np.log1p(e)).sum(axis=1), np.where(z >= 0, 1.0, e) / (1 + e), e / (1 + e) ** 2


def _derivatives(v, s, t):
    """The second, third and fourth derivatives of K in (s, t) per row: each s brings a factor x."""
    _, p, q = _cgf(v, s, t)
    third, fourth = q * (1 - 2 * p), q * (1 - 6 * q)
    powers = [np.ones_like(v), v, v * v, v ** 3, v ** 4]
    sums = {(w, k): (weights * powers[k]).sum(axis=1)
            for w, weights, top in (("q", q, 2), ("r", third, 3), ("f", fourth, 4)) for k in range(top + 1)}
    g = len(v)
    tensors = [np.empty((g,) + (2,) * order) for order in (2, 3, 4)]
    for order, name, tensor in zip((2, 3, 4), ("q", "r", "f"), tensors):
        for index in np.ndindex(*(2,) * order):
            tensor[(slice(None),) + index] = sums[(name, order - sum(index))]
    return tensors


def _solve(v, n1, a):
    """The minimum of phi(s, t) = K(s, t) - s a - t n1 per row by damped Newton, to machine precision: (s, t, phi)."""
    s, t = np.zeros(len(v)), np.full(len(v), log(n1 / (v.shape[1] - n1)))
    phi = _cgf(v, s, t)[0] - t * n1
    live = np.ones(len(v), bool)
    while live.any():
        rows = np.flatnonzero(live)
        vv, si, ti, ai = v[rows], s[rows], t[rows], a[rows]
        K, p, q = _cgf(vv, si, ti)
        gs, gt = (p * vv).sum(axis=1) - ai, p.sum(axis=1) - n1
        hss, hst, htt = (q * vv * vv).sum(axis=1), (q * vv).sum(axis=1), q.sum(axis=1)
        det = hss * htt - hst * hst
        ds, dt = (hst * gt - htt * gs) / det, (hst * gs - hss * gt) / det
        now = K - si * ai - ti * n1
        # Done once the Newton step's predicted decrease is below the rounding of phi itself.
        done = -(gs * ds + gt * dt) <= 4 * EPS * (np.abs(now) + 1)
        live[rows[done]] = False
        pending, step = np.flatnonzero(~done), np.ones(len(rows))
        while len(pending):                                    # halve each step until phi falls
            sn, tn = si[pending] + step[pending] * ds[pending], ti[pending] + step[pending] * dt[pending]
            new = _cgf(vv[pending], sn, tn)[0] - sn * ai[pending] - tn * n1
            fell = new < now[pending]
            moved = rows[pending[fell]]
            s[moved], t[moved], phi[moved] = sn[fell], tn[fell], new[fell]
            pending = pending[~fell]
            step[pending] /= 2
            stuck = step[pending] < EPS                        # no step lowers phi: the minimum, to rounding
            live[rows[pending[stuck]]] = False
            pending = pending[~stuck]
    return s, t, phi
