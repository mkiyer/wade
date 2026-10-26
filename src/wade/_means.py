"""The area: each gene's signed area between the case and control quantile functions, and its test.

The signed area, the integral over u of Q_case(u) - Q_control(u), is exactly the difference in
the group means: an empirical quantile function is a step function whose integral is the mean,
whatever the group sizes. Its permutation test is therefore a question about the cases' total, a
sum of n1 values drawn without replacement: shuffling answers it by brute force, and the
saddlepoint of that sum's distribution answers it without shuffles.

The test runs on counts thinned until every sample's size factor equals the smallest
(equalized). A size factor is a sample's depth as the normalizer measures it: RLE on all genes,
RLE on internal control genes, or factors supplied from elsewhere; raw totals only when the
library normalization is chosen. Read counts carry Poisson noise that shrinks with depth, so
samples at different size factors are not exchangeable at low counts even when nothing differs
between the groups. Thinning makes them exchangeable again, and washes out genes whose counts
are too low to trust.
"""
from math import erfc, exp, lgamma, log, pi, sqrt

import numpy as np

from .permutation import STREAM_BLOCK, stratum_columns, stream_block
from .pvalues import sequential_p, settled

DIRECTIONS = ("both", "up", "down")
EPS, TINY = np.finfo(float).eps, np.finfo(float).tiny
TIE_RTOL = 100 * EPS
_ERFC = np.frompyfunc(erfc, 1, 1)


def stream(seed, index, kind=None, draw=0):
    """One gene's generator for one purpose: spawn key (row index,) for its jitter, (row index, kind)
    for the rest (1 equalizes depth, 3 thins by the subset test's fold; the shuffles use (block, 2)).
    The subset test's further thinning draws d = 1, 2, ... append d, the jitter's kind then being 0."""
    key = (index,) if kind is None else (index, kind)
    if draw:
        key = (index, kind or 0, draw)
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=key))


def jittered(counts, rows, seed, draw=0):
    """Counts plus independent Uniform(0, 1) jitter, each gene's from its own stream (seed, row index).

    Counts are discrete, and many samples share one, above all zero. The subset test's null moves
    each case by a fold and would pull such ties apart, which the observed data never are;
    jittered values are all distinct. The jitter does not depend on the gene chunk.
    """
    out = np.array(counts, dtype=float)
    for row, index in zip(out, rows):
        row += stream(seed, index, draw=draw).random(len(row))
    return out


def expression(counts, factors):
    """Normalized counts: count / size_factor."""
    with np.errstate(over="ignore", invalid="ignore"):
        x = counts / factors
    if not np.isfinite(x).all():
        raise ValueError("counts / size_factors exceeds the finite expression range")
    return x


def floors(factors, code=None):
    """The size factor each sample is thinned to: the smallest, or with strata its stratum's smallest.
    factors holds one size factor per sample, or one row of them per gene."""
    factors = np.asarray(factors, float)
    if code is None:
        return factors.min(axis=-1, keepdims=True)
    out = np.empty(factors.shape)
    for idx in stratum_columns(code):
        out[..., idx] = factors[..., idx].min(axis=-1, keepdims=True)
    return out


def equalized(counts, rows, factors, seed, code=None, draw=0):
    """Counts thinned binomially until every sample's size factor equals the smallest, each gene
    from its own stream (seed, row index, 1).

    Sample j keeps each read with chance min(s) / s_j, s its size factor from the normalization.
    A thinned Poisson count is Poisson at the lower rate, so every sample ends at one size factor
    and a difference in depth between the groups cannot pass for a difference in expression. Counts are rounded to whole
    reads first. factors holds one size factor per sample, or one row of them per gene. The draws
    do not depend on the gene chunk. With strata (code), each stratum is thinned to its own smallest
    size factor: labels move only within strata, so depths need to be equal only there.
    """
    keep = np.broadcast_to(floors(factors, code) / factors, np.shape(counts))
    out = np.empty(np.shape(counts), np.int64)
    for i, (row, index) in enumerate(zip(counts, rows)):
        out[i] = stream(seed, index, 1, draw).binomial(np.rint(row).astype(np.int64), keep[i])
    return out


def area_values(counts, rows, factors, seed, code=None):
    """The values the area test shuffles: equalized counts plus jitter, over the common size factor
    (with strata, each stratum's own), so that values are in normalized units in every stratum."""
    return expression(jittered(equalized(counts, rows, factors, seed, code), rows, seed), floors(factors, code))


def centred(x, code):
    """Values less their stratum's mean, gene by gene."""
    out = np.array(x, dtype=float)
    for idx in stratum_columns(code):
        out[:, idx] -= out[:, idx].mean(axis=1, keepdims=True)
    return out


def _stratified(code):
    return code is not None and not np.all(np.asarray(code) == np.asarray(code)[0])


def advance_means(x, cond, exceed, tried, *, direction, stop, maximum, threshold, seed, code=None):
    """Shuffle each gene's area (rows of x) along the run's stream until it is settled at threshold.

    Each gene resumes at its own position, the number of shuffles it has tried, and counts the
    shuffles whose difference in means reaches the observed one: direction "both" counts those at
    least as far from zero, "up" those at least as high, "down" those at least as low. It stops at
    its stop-th exceedance, at maximum shuffles, or once its Besag–Clifford bound reaches threshold
    (pvalues.settled). The records exceed and tried are updated in place.

    With strata (code), labels are shuffled within strata and the values are first centred within
    them: the cases' total is then the Cochran–Mantel–Haenszel sum of within-stratum differences,
    centred at zero under the stratified null, as the two-sided test needs.
    """
    cond = np.asarray(cond)
    if _stratified(code):
        x = centred(x, code)
    n1, n0 = int(cond.sum()), int((cond == 0).sum())
    observed = x[:, cond == 1].mean(axis=1) - x[:, cond == 0].mean(axis=1)
    total = x.sum(axis=1)
    rounding = TIE_RTOL * np.abs(x).sum(axis=1) / min(n1, n0)     # a shuffle within it ties
    steps = np.arange(STREAM_BLOCK)
    while (live := ~settled(exceed, tried, stop=stop, maximum=maximum, threshold=threshold)).any():
        block = int(tried[live].min()) // STREAM_BLOCK
        first = block * STREAM_BLOCK
        genes = np.flatnonzero(live & (tried < first + STREAM_BLOCK))
        labels = stream_block(cond, block, seed, code)
        sums = x[genes] @ labels.T.astype(float)                   # the cases' total in each shuffle
        shuffled = sums / n1 - (total[genes, None] - sums) / n0
        mark, tie = observed[genes, None], rounding[genes, None]
        if direction == "up":
            hits = shuffled >= mark - tie
        elif direction == "down":
            hits = shuffled <= mark + tie
        else:
            hits = np.abs(shuffled) >= np.abs(mark) - tie
        start = tried[genes].astype(np.int64) - first               # each gene's next shuffle in this block
        mine = steps >= start[:, None]
        count = exceed[genes, None].astype(np.int64) + np.cumsum(hits & mine, axis=1)
        position = tried[genes, None].astype(np.int64) + steps - start[:, None] + 1
        done = mine & settled(count, np.maximum(position, count), stop=stop, maximum=maximum, threshold=threshold)
        last = np.where(done.any(axis=1), done.argmax(axis=1), STREAM_BLOCK - 1)
        rows = np.arange(len(genes))
        exceed[genes], tried[genes] = count[rows, last], position[rows, last]


def mean_test(x, cond, *, direction, stop, maximum, seed, code=None):
    """Permutation p-values for each gene's area, every gene run to the end of its Besag–Clifford
    rule: its stop-th exceedance, or maximum shuffles. Rows are genes; every gene sees the run's one
    stream of shuffles (permutation.stream_block), within strata if code is given. Returns the
    p-values and the exceedance and shuffle counts behind them.
    """
    stop = min(stop, maximum)
    exceed, tried = np.zeros(len(x), np.uint64), np.zeros(len(x), np.uint64)
    advance_means(x, cond, exceed, tried, direction=direction, stop=stop, maximum=maximum,
                  threshold=-np.inf, seed=seed, code=code)
    return sequential_p(exceed, tried, stop=stop, maximum=maximum), exceed, tried


def mean_saddlepoint(x, cond, *, direction, code=None):
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

    With strata (code), the selectors are conditioned on one count per stratum: an (S + 1)-
    dimensional double saddlepoint (see strata_saddlepoint).
    """
    if _stratified(code):
        return strata_saddlepoint(x, cond, code, direction=direction)
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


# ---------------------------------------------------------------------------------------------
# The stratified saddlepoint

def strata_saddlepoint(x, cond, code, *, direction):
    """Each gene's permutation p-value for the cases' total with labels exchanged within strata, by
    the stratified double saddlepoint, and that p-value's estimated relative error.

    The selectors are independent Bernoulli variables conditioned on one count per stratum, with the
    joint cumulant generating function K(s, t) = sum_j log(1 + exp(s x_j + t_{stratum of j})). Its
    Hessian is an arrowhead (s against each stratum's t, no stratum against another), so each Newton
    step, the tail and the second-order error term are sums over samples: O(n) per gene, whatever the
    number of strata. Values are centred within their stratum, so that the cases' total has null mean
    0. The error estimate is the larger of the joint second-order density term (less the strata
    counts' own) and the granularity of the exact distribution, one labelling among the
    L = p prod_s C(n_s, n1_s) at least as extreme. Strata with only one group cannot move and are left
    out. direction has the meaning it has in mean_test.
    """
    cond, code = np.asarray(cond), np.asarray(code)
    S = int(code.max()) + 1
    n1s = np.bincount(code, weights=cond, minlength=S).astype(np.int64)
    ns = np.bincount(code, minlength=S)
    moves = (n1s > 0) & (n1s < ns)
    keep = moves[code]
    x, cond = x[:, keep], cond[keep]
    code = (np.cumsum(moves) - 1)[code[keep]]
    n1s, n0s = n1s[moves], (ns - n1s)[moves]
    order = np.argsort(code, kind="stable")                    # strata as contiguous blocks
    x, cond, code = x[:, order], cond[order], code[order]
    starts = np.r_[0, np.flatnonzero(np.diff(code)) + 1]
    c = x - (np.add.reduceat(x, starts, axis=1) / (n1s + n0s))[:, code]
    total = (c * cond).sum(axis=1)                            # the cases' total less its null mean
    if direction == "up":
        p, error = _strata_tail(c, code, starts, n1s, n0s, total)
    elif direction == "down":
        p, error = _strata_tail(-c, code, starts, n1s, n0s, -total)
    else:
        gap = np.abs(total)
        (above, ea), (below, eb) = _strata_tail(c, code, starts, n1s, n0s, gap), _strata_tail(-c, code, starts, n1s, n0s, gap)
        p = np.minimum(above + below, 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            error = np.where(above + below > 0, (above * ea + below * eb) / (above + below), 0.0)
    labellings = sum(_log_choose(int(a + b), int(a)) for a, b in zip(n1s, n0s))
    with np.errstate(over="ignore", divide="ignore"):
        granularity = np.exp(-(np.log(p) + labellings))
    return p, np.where(error > 0, np.maximum(error, granularity), 0.0)


def _strata_tail(c, code, starts, n1s, n0s, total):
    """P(T >= total) per row for T the stratified cases' total of values c (centred within strata),
    and the saddlepoint's estimated relative error (0 where the answer is exact)."""
    G, n = c.shape
    spread = np.sqrt((c * c).mean(axis=1))
    flat = spread <= TIE_RTOL * np.abs(c).max(axis=1)        # every labelling gives the same total
    spread = np.where(flat, 1.0, spread)
    v = c / spread[:, None]
    a = total / spread
    reach = TIE_RTOL * n * np.abs(v).max(axis=1)              # the rounding of a sum of n values
    highest, lowest = np.zeros(G), np.zeros(G)
    for s, (lo, hi) in enumerate(zip(starts, np.r_[starts[1:], n])):
        block = np.sort(v[:, lo:hi], axis=1)
        highest += block[:, -n1s[s]:].sum(axis=1)
        lowest += block[:, :n1s[s]].sum(axis=1)
    beyond = ~flat & (a > highest + reach)                    # no labelling gets there
    top = ~flat & ~beyond & (a >= highest - reach)
    inside = ~flat & ~beyond & ~top & (a > lowest + reach)
    p, error = np.where(beyond, 0.0, 1.0), np.zeros(G)
    for g in np.flatnonzero(top):
        p[g] = _strata_most_extreme(v[g], starts, n1s)
    if inside.any():
        p[inside], error[inside] = _strata_inside(v[inside], code, starts, n1s, n0s, a[inside])
    return p, error


def _strata_most_extreme(v, starts, n1s):
    """The exact chance of the largest stratified total: every stratum takes its n1s largest values,
    ties included."""
    out = 0.0
    for s, (lo, hi) in enumerate(zip(starts, np.r_[starts[1:], len(v)])):
        ordered = np.sort(v[lo:hi])
        ns, k = len(ordered), int(n1s[s])
        last = ordered[ns - k]
        tied = np.abs(ordered - last) <= TIE_RTOL * np.abs(ordered).max()
        out += _log_choose(int(tied.sum()), int(np.sum(tied[ns - k:]))) - _log_choose(ns, k)
    return max(exp(out), TINY)


def _strata_state(v, code, theta, lam):
    """Each selector's natural parameter, probability p and variance q at (theta, lambda)."""
    z = theta[:, None] * v + lam[:, code]
    e = np.exp(-np.abs(z))
    return z, np.where(z >= 0, 1.0, e) / (1 + e), e / (1 + e) ** 2


def _strata_phi(v, code, theta, lam, a, n1s):
    z, _, _ = _strata_state(v, code, theta, lam)
    return (np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))).sum(axis=1) - theta * a - lam @ n1s


def _strata_solve(v, code, starts, n1s, n0s, a):
    """The minimum of phi(theta, lambda) = K - theta a - lambda . n1s per row, by damped Newton with the
    arrowhead Hessian, to machine precision: (theta, lambda)."""
    G = len(v)
    theta = np.zeros(G)
    lam = np.tile(np.log(n1s / n0s), (G, 1))
    phi = _strata_phi(v, code, theta, lam, a, n1s)
    live = np.ones(G, bool)
    while live.any():
        rows = np.flatnonzero(live)
        vv, th, la, aa = v[rows], theta[rows], lam[rows], a[rows]
        _, p, q = _strata_state(vv, code, th, la)
        gt = (p * vv).sum(axis=1) - aa
        gs = np.add.reduceat(p, starts, axis=1) - n1s
        b, d = np.add.reduceat(q * vv, starts, axis=1), np.add.reduceat(q, starts, axis=1)
        c = (q * vv * vv).sum(axis=1) - (b * b / d).sum(axis=1)   # the arrowhead's Schur complement
        dt = -(gt - (b * gs / d).sum(axis=1)) / c
        dl = -(gs + b * dt[:, None]) / d
        now = phi[rows]
        # Done once the Newton step's predicted decrease is below the rounding of phi itself.
        done = -(gt * dt + (gs * dl).sum(axis=1)) <= 4 * EPS * (np.abs(now) + 1)
        live[rows[done]] = False
        pending, step = np.flatnonzero(~done), np.ones(len(rows))
        while len(pending):                                    # halve each step until phi falls
            tn = th[pending] + step[pending] * dt[pending]
            ln = la[pending] + step[pending, None] * dl[pending]
            new = _strata_phi(vv[pending], code, tn, ln, aa[pending], n1s)
            fell = new < now[pending]
            moved = rows[pending[fell]]
            theta[moved], lam[moved], phi[moved] = tn[fell], ln[fell], new[fell]
            pending = pending[~fell]
            step[pending] /= 2
            stuck = step[pending] < EPS                        # no step lowers phi: the minimum, to rounding
            live[rows[pending[stuck]]] = False
            pending = pending[~stuck]
    return theta, lam


def _strata_inside(v, code, starts, n1s, n0s, a):
    """The stratified double-saddlepoint tail and its estimated error, for rows strictly inside the range."""
    ns = n1s + n0s
    theta, lam = _strata_solve(v, code, starts, n1s, n0s, a)
    p0, t0 = n1s / ns, np.log(n1s / n0s)
    q0 = p0 * (1 - p0)
    # w^2 / 2 = theta a - sum_j B_j, each selector's Bregman divergence of the softplus from its stratum's
    # centre t0_s (the constraint terms vanish there, sum_j p0_s = n1s). Summed per selector it keeps full
    # precision near the centre.
    delta = theta[:, None] * v + (lam - t0)[:, code]
    base = p0[code]
    softplus = lambda z: np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))
    bregman = np.where(delta <= 1, np.log1p(base * np.expm1(np.minimum(delta, 1))),
                       softplus(t0[code] + delta) - softplus(t0[code])) - base * delta
    w = np.sign(theta) * np.sqrt(np.maximum(2 * (theta * a - bregman.sum(axis=1)), 0.0))
    _, p, q = _strata_state(v, code, theta, lam)
    b, d = np.add.reduceat(q * v, starts, axis=1), np.add.reduceat(q, starts, axis=1)
    c = (q * v * v).sum(axis=1) - (b * b / d).sum(axis=1)
    # |K''(theta, lambda)| / |K''_lambda lambda(0, t0)| = c prod_s d_s / prod_s (n_s q0_s), in logs.
    u = theta * np.exp(0.5 * (np.log(np.maximum(c, TINY)) + (np.log(d) - np.log(ns * q0)).sum(axis=1)))
    # At the centre 1/w - 1/u is 0/0; its limit is the conditional skewness over 6.
    r0 = q0 * (1 - 2 * p0)
    limit = (r0 * np.add.reduceat(v ** 3, starts, axis=1)).sum(axis=1) / (
        6 * ((q0 * np.add.reduceat(v * v, starts, axis=1)).sum(axis=1)) ** 1.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        correction = np.where(np.abs(theta) < EPS ** (1 / 3), limit, 1 / w - 1 / u)
    tail = 0.5 * _ERFC(w / sqrt(2)).astype(float)
    pval = tail - np.exp(-0.5 * w * w) / sqrt(2 * pi) * correction
    floor = max(exp(-sum(_log_choose(int(k), int(m)) for k, m in zip(ns, n1s))), TINY)
    return np.clip(pval, floor, 1.0), _strata_second_order(v, code, starts, p, q, b, d, c, n1s, ns)


def _strata_second_order(v, code, starts, p, q, b, d, c, n1s, ns):
    """|joint second-order density term - the strata counts' own| from sums over samples. With
    e_j = (v_j, the unit vector of j's stratum), every cumulant tensor is sum_j k_j^(m) e_j^(x)m, and the
    inverse arrowhead Hessian gives e_j' M e_k = vt_j vt_k / c + [same stratum] / d_s, vt = v - b_s / d_s:
    McCullagh's rho4, rho13 and rho23 (as in _saddlepoint) without forming a tensor."""
    r, f = q * (1 - 2 * p), q * (1 - 6 * q)
    ratio = b / d
    vt = v - ratio[:, code]
    h = vt * vt / c[:, None] + 1 / d[:, code]                   # e_j' M e_j
    rho4 = (f * h * h).sum(axis=1)
    ut, us = (r * h * v).sum(axis=1), np.add.reduceat(r * h, starts, axis=1)
    rho13 = (ut - (ratio * us).sum(axis=1)) ** 2 / c + (us * us / d).sum(axis=1)
    rho23 = ((r * vt ** 3).sum(axis=1)) ** 2 / c ** 3 + (
        3 * np.add.reduceat(r * vt * vt, starts, axis=1) ** 2 / (c[:, None] ** 2 * d)
        + 3 * np.add.reduceat(r * vt, starts, axis=1) ** 2 / (c[:, None] * d * d)
        + np.add.reduceat(r, starts, axis=1) ** 2 / d ** 3).sum(axis=1)
    p0 = n1s / ns
    q0 = p0 * (1 - p0)
    lambda3, lambda4 = (1 - 2 * p0) / np.sqrt(q0 * ns), (1 - 6 * q0) / (q0 * ns)
    counts = float(np.sum(lambda4 / 8 - 5 * lambda3 ** 2 / 24))
    return np.abs(rho4 / 8 - (2 * rho23 + 3 * rho13) / 24 - counts)
