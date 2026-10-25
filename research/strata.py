"""Strata: a prototype of WADE's tests with labels exchanged only within strata, and its vetting.

Strata are the groups that one or more factors (study, site, batch) define. Under the stratified
null the labels are exchangeable within each stratum, whatever differs between strata, and every
test shuffles them only there. A stratum with only cases or only controls cannot move a label and
carries no information. This file prototypes the pieces outside the package and measures them;
nothing here changes wade() until the design is adopted.

The area test. The cases' total, shuffled within strata, is the Cochran-Mantel-Haenszel sum:
T - E(T) = sum_s (n1s n0s / ns)(mean1_s - mean0_s). So the statistic is unchanged and only the
shuffles and the saddlepoint need strata. The saddlepoint conditions independent Bernoulli
selectors on one count per stratum, an (S+1)-dimensional double saddlepoint whose Hessian is an
arrowhead; every quantity, including the second-order error term, is a sum over samples (O(n) per
gene, whatever S). Each stratum is thinned to its own smallest size factor, which is all that
exchangeability within the stratum needs.

The subset test. Two constructions are compared:
  pooled (S2)   register each stratum's cases onto its controls by the stratum's own W1 fold, align
                each stratum's registered pool to the cohort's, and walk the pooled cases against
                the pooled controls, as now, with labels shuffled within strata;
  combined (S1) walk each stratum on its own and add the strata's gaps with weights
                sqrt(n1s n0s / ns), labels shuffled within strata.

Sections (default: all):

  exact      the stratified saddlepoint and the stratified shuffles against exact enumeration of
             every within-stratum labelling, on small designs; one stratum reproduces the package
  accuracy   the stratified saddlepoint and its error estimate on realistic designs (the TCGA sites,
             two unbalanced studies, many small ones), against 20 million within-stratum labellings,
             on the confirmation's generator with batch effects and lumpy totals, as
             research/pvalue_accuracy.py does for one stratum
  model      the count model with studies: TCGA prostate tumours, each gene split at random into
             studies whose case shares differ (confounding), thinned binomially to a depth per study
             and a gene-specific batch scale per study (exact under the count model, as
             research/count_model.py). Both tests unstratified and stratified, under no fold, a 1.5x
             fold and a planted subset
  sites      TCGA prostate tumours relabelled at random within their tissue source sites, with case
             shares that differ between sites: real site effects, a known null
  gtex       TCGA and GTEx prostate normals relabelled at random within study, case shares that
             differ: real study effects, a known null; and TCGA normals against GTEx normals
  real       TCGA prostate tumours against normals stratified by tissue source site (the sites with
             both), against the unstratified test: calls and the fusion markers
  layout     the real contrast's five sites as a null: their tumours relabelled with the real case shares
  common     the subset test with one registration after a label-free alignment of the strata (S3), and
             with a two-way fit of stratum levels and one fold (S4), against S2 and the unstratified test,
             on every null with strata and the planted subsets
  fold       the subset test's null under a common 1.5x fold, where its registration is stressed: the
             model's designs and batch settings, and the TCGA sites
  thinned    the adopted subset construction within strata (each stratum thinned to its smallest size
             factor, the higher group thinned by one fold) on the strata's nulls, with bins down to 8 reads
  registration  why S1 fails on the TCGA sites: its statistic registered within each stratum and on raw
             values (exact without a fold)
  heterogeneous  folds that differ between strata: what the two-way construction, whose null has one
             fold, and the pooled one, with a fold per stratum, call
  speed      the stratified pieces' cost against the unstratified ones on TCGA prostate (24,246 genes,
             497 samples), measured back to back, with 1 to 128 strata drawn at random

    WADE_TCGA_CONFIG=.release/0.2.0/tcga-config.json RAYON_NUM_THREADS=6 PYTHONPATH=src:research \\
        python research/strata.py [section ...]
"""
import itertools
import sys
from math import comb, erfc, exp, lgamma, log, pi, sqrt

import numpy as np

from wade import _kernel
from wade._means import _log_choose, expression, jittered, mean_saddlepoint
from wade._subsets import split as package_split
from wade.permutation import STREAM_BLOCK
from wade.pvalues import sequential_p, settled

EPS, TINY = np.finfo(float).eps, np.finfo(float).tiny
TIE_RTOL = 100 * EPS
_ERFC = np.frompyfunc(erfc, 1, 1)


# ---------------------------------------------------------------------------------------------
# Strata

def codes(strata):
    """Each sample's stratum as 0..S-1, in order of first appearance; strata is one label per sample,
    or a list of factors (one label per sample each), crossed."""
    if isinstance(strata, (list, tuple)) and len(strata) and np.ndim(strata[0]) == 1:
        keys = list(zip(*[np.asarray(f).tolist() for f in strata]))
    else:
        keys = np.asarray(strata).tolist()
    seen = {}
    return np.array([seen.setdefault(k, len(seen)) for k in keys], dtype=np.int64)


def design(cond, code):
    """Per stratum: cases, controls, and whether it carries information (both groups present)."""
    cond, code = np.asarray(cond), np.asarray(code)
    S = int(code.max()) + 1
    n1s = np.bincount(code, weights=cond, minlength=S).astype(np.int64)
    ns = np.bincount(code, minlength=S)
    return n1s, ns - n1s, (n1s > 0) & (n1s < ns)


def stream_block(cond, code, block, seed):
    """Block `block` of the stratified stream: each stratum's labels permuted within it, the strata in
    order, all from the block's one generator (spawn key (block, 2), as the package's stream). With one
    stratum it is the package's stream exactly."""
    cond = np.asarray(cond, dtype=np.uint8)
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(block, 2)))
    rows = np.tile(cond, (STREAM_BLOCK, 1))
    S = int(np.max(code)) + 1
    if S == 1:
        return rng.permuted(rows, axis=1, out=rows)
    for s in range(S):
        idx = np.flatnonzero(code == s)
        if 0 < cond[idx].sum() < len(idx):
            rows[:, idx] = rng.permuted(rows[:, idx], axis=1)
    return rows


def equalized(counts, rows, factors, code, seed):
    """Counts thinned binomially until every sample's size factor equals its stratum's smallest, each gene
    from its own stream (seed, row index, 1), as the package's equalized(). Exchangeability within a
    stratum needs equal depths only there. factors: one per sample, or one row per gene."""
    factors = np.broadcast_to(factors, np.shape(counts)).astype(float)
    floor = np.empty_like(factors)
    for s in np.unique(code):
        idx = code == s
        floor[:, idx] = factors[:, idx].min(axis=1, keepdims=True)
    keep = floor / factors
    out = np.empty(np.shape(counts), np.int64)
    for i, (row, index) in enumerate(zip(counts, rows)):
        out[i] = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(index, 1))).binomial(
            np.rint(row).astype(np.int64), keep[i])
    return out, floor


def area_values(counts, rows, factors, code, seed):
    """The values the stratified area test shuffles: counts equalized within strata, jittered, over each
    stratum's common size factor (normalized units, comparable across strata)."""
    thinned, floor = equalized(counts, rows, factors, code, seed)
    return expression(jittered(thinned, rows, seed), floor)


# ---------------------------------------------------------------------------------------------
# The area test's p-value: shuffled, and by the stratified double saddlepoint

def shuffled_area(x, cond, code, *, direction="both", stop=25, maximum=9_999, seed=1):
    """Besag-Clifford p-values of the cases' total under within-stratum shuffles, every run finished:
    the package's advance_means with the stratified stream (one stratum: the package exactly).

    Values are centred within their stratum first. The cases' total is then centred at zero under the
    stratified null, which the two-sided test needs: with case shares that differ between strata, the
    plain difference in means has a null mean away from zero."""
    cond, code = np.asarray(cond), np.asarray(code)
    if len(np.unique(code)) > 1:
        means = np.zeros((len(x), int(code.max()) + 1))
        for s in np.unique(code):
            means[:, s] = x[:, code == s].mean(axis=1)
        x = x - means[:, code]
    n1, n0 = int(cond.sum()), int((cond == 0).sum())
    observed = x[:, cond == 1].mean(axis=1) - x[:, cond == 0].mean(axis=1)
    total = x.sum(axis=1)
    rounding = TIE_RTOL * np.abs(x).sum(axis=1) / min(n1, n0)
    exceed, tried = np.zeros(len(x), np.uint64), np.zeros(len(x), np.uint64)
    steps = np.arange(STREAM_BLOCK)
    while (live := ~settled(exceed, tried, stop=stop, maximum=maximum, threshold=-np.inf)).any():
        block = int(tried[live].min()) // STREAM_BLOCK
        first = block * STREAM_BLOCK
        genes = np.flatnonzero(live & (tried < first + STREAM_BLOCK))
        labels = stream_block(cond, code, block, seed)
        sums = x[genes] @ labels.T.astype(float)
        shuffled = sums / n1 - (total[genes, None] - sums) / n0
        mark, tie = observed[genes, None], rounding[genes, None]
        hits = (shuffled >= mark - tie if direction == "up" else shuffled <= mark + tie if direction == "down"
                else np.abs(shuffled) >= np.abs(mark) - tie)
        start = tried[genes].astype(np.int64) - first
        mine = steps >= start[:, None]
        count = exceed[genes, None].astype(np.int64) + np.cumsum(hits & mine, axis=1)
        position = tried[genes, None].astype(np.int64) + steps - start[:, None] + 1
        done = mine & settled(count, np.maximum(position, count), stop=stop, maximum=maximum, threshold=-np.inf)
        last = np.where(done.any(axis=1), done.argmax(axis=1), STREAM_BLOCK - 1)
        rows = np.arange(len(genes))
        exceed[genes], tried[genes] = count[rows, last], position[rows, last]
    return sequential_p(exceed, tried, stop=stop, maximum=maximum)


def saddlepoint_area(x, cond, code, *, direction="both"):
    """Each gene's p-value for the cases' total under within-stratum exchange, by the stratified double
    saddlepoint, and its estimated relative error (the second-order term, and one labelling among the
    L = p prod_s C(ns, n1s) in the tail). Strata without both groups are left out: they cannot move."""
    cond, code = np.asarray(cond), np.asarray(code)
    n1s, n0s, informative = design(cond, code)
    keep = informative[code]
    x, cond = x[:, keep], cond[keep]
    remap = np.cumsum(informative) - 1
    code = remap[code[keep]]
    n1s, n0s = n1s[informative], n0s[informative]
    order = np.argsort(code, kind="stable")                    # strata as contiguous blocks
    x, cond, code = x[:, order], cond[order], code[order]
    starts = np.r_[0, np.flatnonzero(np.diff(code)) + 1]
    means = np.add.reduceat(x, starts, axis=1) / (n1s + n0s)
    centred = x - means[:, code]
    total = (centred * cond).sum(axis=1)                      # the cases' total less its null mean
    if direction == "up":
        p, error = _tail(centred, code, starts, n1s, n0s, total)
    elif direction == "down":
        p, error = _tail(-centred, code, starts, n1s, n0s, -total)
    else:
        gap = np.abs(total)
        (above, ea), (below, eb) = (_tail(centred, code, starts, n1s, n0s, gap),
                                    _tail(-centred, code, starts, n1s, n0s, gap))
        p = np.minimum(above + below, 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            error = np.where(above + below > 0, (above * ea + below * eb) / (above + below), 0.0)
    labellings = sum(_log_choose(int(a + b), int(a)) for a, b in zip(n1s, n0s))
    with np.errstate(over="ignore", divide="ignore"):
        granularity = np.exp(-(np.log(p) + labellings))
    return p, np.where(error > 0, np.maximum(error, granularity), 0.0)


def _tail(c, code, starts, n1s, n0s, total):
    """P(T >= total) per row for T the stratified cases' total of values c (centred within strata), and
    the saddlepoint's estimated relative error (0 where exact)."""
    G, n = c.shape
    spread = np.sqrt((c * c).mean(axis=1))
    flat = spread <= TIE_RTOL * np.abs(c).max(axis=1)
    spread = np.where(flat, 1.0, spread)
    v = c / spread[:, None]
    a = total / spread
    reach = TIE_RTOL * n * np.abs(v).max(axis=1)
    highest, lowest = np.zeros(G), np.zeros(G)
    for s, (lo, hi) in enumerate(zip(starts, np.r_[starts[1:], n])):
        block = np.sort(v[:, lo:hi], axis=1)
        highest += block[:, -n1s[s]:].sum(axis=1)
        lowest += block[:, :n1s[s]].sum(axis=1)
    beyond = ~flat & (a > highest + reach)
    top = ~flat & ~beyond & (a >= highest - reach)
    inside = ~flat & ~beyond & ~top & (a > lowest + reach)
    p, error = np.where(beyond, 0.0, 1.0), np.zeros(G)
    for g in np.flatnonzero(top):
        p[g] = _most_extreme(v[g], starts, n1s)
    if inside.any():
        p[inside], error[inside] = _saddlepoint(v[inside], code, starts, n1s, n0s, a[inside])
    return p, error


def _most_extreme(v, starts, n1s):
    """The exact chance of the largest stratified total: every stratum takes its n1s largest, ties included."""
    out = 0.0
    for s, (lo, hi) in enumerate(zip(starts, np.r_[starts[1:], len(v)])):
        ordered = np.sort(v[lo:hi])
        ns, k = len(ordered), int(n1s[s])
        last = ordered[ns - k]
        tied = np.abs(ordered - last) <= TIE_RTOL * np.abs(ordered).max()
        out += _log_choose(int(tied.sum()), int(np.sum(tied[ns - k:]))) - _log_choose(ns, k)
    return max(exp(out), TINY)


def _sums(values, starts):
    return np.add.reduceat(values, starts, axis=1)


def _state(v, code, theta, lam):
    """Each selector's natural parameter, probability p and variance q at (theta, lambda)."""
    z = theta[:, None] * v + lam[:, code]
    e = np.exp(-np.abs(z))
    return z, np.where(z >= 0, 1.0, e) / (1 + e), e / (1 + e) ** 2


def _phi(v, code, theta, lam, a, n1s):
    z, _, _ = _state(v, code, theta, lam)
    return (np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))).sum(axis=1) - theta * a - lam @ n1s


def _solve(v, code, starts, n1s, n0s, a):
    """The minimum of phi(theta, lambda) = K - theta a - lambda . n1s per row, by damped Newton with the
    arrowhead Hessian (each step O(n)), to machine precision."""
    G = len(v)
    theta = np.zeros(G)
    lam = np.tile(np.log(n1s / n0s), (G, 1))
    phi = _phi(v, code, theta, lam, a, n1s)
    live = np.ones(G, bool)
    while live.any():
        rows = np.flatnonzero(live)
        vv, th, la, aa = v[rows], theta[rows], lam[rows], a[rows]
        _, p, q = _state(vv, code, th, la)
        gt = (p * vv).sum(axis=1) - aa
        gs = _sums(p, starts) - n1s
        b, d = _sums(q * vv, starts), _sums(q, starts)
        c = (q * vv * vv).sum(axis=1) - (b * b / d).sum(axis=1)
        dt = -(gt - (b * gs / d).sum(axis=1)) / c
        dl = -(gs + b * dt[:, None]) / d
        now = phi[rows]
        done = -(gt * dt + (gs * dl).sum(axis=1)) <= 4 * EPS * (np.abs(now) + 1)
        live[rows[done]] = False
        pending, step = np.flatnonzero(~done), np.ones(len(rows))
        while len(pending):
            tn = th[pending] + step[pending] * dt[pending]
            ln = la[pending] + step[pending, None] * dl[pending]
            new = _phi(vv[pending], code, tn, ln, aa[pending], n1s)
            fell = new < now[pending]
            moved = rows[pending[fell]]
            theta[moved], lam[moved], phi[moved] = tn[fell], ln[fell], new[fell]
            pending = pending[~fell]
            step[pending] /= 2
            stuck = step[pending] < EPS
            live[rows[pending[stuck]]] = False
            pending = pending[~stuck]
    return theta, lam


def _saddlepoint(v, code, starts, n1s, n0s, a):
    """The stratified double-saddlepoint tail and its estimated error, for rows strictly inside the range."""
    ns = n1s + n0s
    theta, lam = _solve(v, code, starts, n1s, n0s, a)
    p0, t0 = n1s / ns, np.log(n1s / n0s)
    q0 = p0 * (1 - p0)
    # w^2 / 2 = theta a - sum_j Bregman_j, each selector's Bregman divergence of the softplus from its
    # stratum's centre t0_s (the constraint terms vanish there, sum_j p0_s = n1s).
    delta = theta[:, None] * v + (lam - t0)[:, code]
    base = p0[code]
    softplus = lambda z: np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))
    bregman = np.where(delta <= 1, np.log1p(base * np.expm1(np.minimum(delta, 1))),
                       softplus(t0[code] + delta) - softplus(t0[code])) - base * delta
    w = np.sign(theta) * np.sqrt(np.maximum(2 * (theta * a - bregman.sum(axis=1)), 0.0))
    _, p, q = _state(v, code, theta, lam)
    b, d = _sums(q * v, starts), _sums(q, starts)
    c = (q * v * v).sum(axis=1) - (b * b / d).sum(axis=1)       # the arrowhead's Schur complement
    log_ratio = np.log(np.maximum(c, TINY)) + (np.log(d) - np.log(ns * q0)).sum(axis=1)
    u = theta * np.exp(0.5 * log_ratio)
    # At the centre 1/w - 1/u is 0/0; its limit is the conditional skewness over 6.
    r0 = q0 * (1 - 2 * p0)
    v2, v3 = _sums(v * v, starts), _sums(v ** 3, starts)
    limit = (r0 * v3).sum(axis=1) / (6 * ((q0 * v2).sum(axis=1)) ** 1.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        correction = np.where(np.abs(theta) < EPS ** (1 / 3), limit, 1 / w - 1 / u)
    tail = 0.5 * _ERFC(w / sqrt(2)).astype(float)
    pval = tail - np.exp(-0.5 * w * w) / sqrt(2 * pi) * correction
    floor = max(exp(-sum(_log_choose(int(k), int(m)) for k, m in zip(ns, n1s))), TINY)
    error = _second_order(v, code, starts, p, q, b, d, c, n1s, ns)
    return np.clip(pval, floor, 1.0), error


def _second_order(v, code, starts, p, q, b, d, c, n1s, ns):
    """|joint second-order density term - the strata counts' own|, from sums over samples: with
    e_j = (v_j, unit vector of j's stratum), every cumulant tensor is sum_j k_j^(m) e_j^(x)m, and the
    inverse arrowhead Hessian gives e_j' M e_k = vt_j vt_k / c + [same stratum] / d_s, vt = v - b_s / d_s."""
    r, f = q * (1 - 2 * p), q * (1 - 6 * q)
    ratio = b / d
    vt = v - ratio[:, code]
    h = vt * vt / c[:, None] + 1 / d[:, code]                   # e_j' M e_j
    rho4 = (f * h * h).sum(axis=1)
    ut, us = (r * h * v).sum(axis=1), _sums(r * h, starts)
    rho13 = (ut - (ratio * us).sum(axis=1)) ** 2 / c + (us * us / d).sum(axis=1)
    rho23 = ((r * vt ** 3).sum(axis=1)) ** 2 / c ** 3 + (
        3 * _sums(r * vt * vt, starts) ** 2 / (c[:, None] ** 2 * d)
        + 3 * _sums(r * vt, starts) ** 2 / (c[:, None] * d * d)
        + _sums(r, starts) ** 2 / d ** 3).sum(axis=1)
    p0 = n1s / ns
    q0 = p0 * (1 - p0)
    lambda3, lambda4 = (1 - 2 * p0) / np.sqrt(q0 * ns), (1 - 6 * q0) / (q0 * ns)
    counts = float(np.sum(lambda4 / 8 - 5 * lambda3 ** 2 / 24))
    return np.abs(rho4 / 8 - (2 * rho23 + 3 * rho13) / 24 - counts)


def area_p(z, cond, code, *, seed=1, stop=25, maximum=9_999, relative_error=0.2):
    """The stratified area test's p-values as wade() would compute them: the saddlepoint, shuffled within
    strata where its error estimate exceeds relative_error."""
    p, error = saddlepoint_area(z, cond, code)
    shuffle = error > relative_error
    if shuffle.any():
        p[shuffle] = shuffled_area(z[shuffle], cond, code, stop=stop, maximum=maximum, seed=seed)
    return p


def inverse_variance(z, code):
    """Values weighted by each stratum's inverse variance, computed without the labels."""
    var = np.ones((len(z), int(code.max()) + 1))
    for s in np.unique(code):
        var[:, s] = z[:, code == s].var(axis=1)
    return z / np.where(var > 0, var, 1.0)[:, code]


# ---------------------------------------------------------------------------------------------
# The subset test with strata

def pooled_values(x, cond, code):
    """The pooled construction (S2): the informative strata only; each stratum's cases carried back onto
    its controls by the stratum's own W1 fold (on count + 1), then each stratum moved, by a fold on
    count + 1, so that its median log(count + 1) is the cohort's. Returns the pool, its labels and codes."""
    n1s, n0s, informative = design(cond, code)
    keep = informative[code]
    x, cond, code = x[:, keep], cond[keep], code[keep]
    pooled = x.astype(float).copy()
    for s in np.unique(code):
        idx = np.flatnonzero(code == s)
        shift = package_split(np.ascontiguousarray(x[:, idx]), cond[idx])[4]
        cases = idx[cond[idx] == 1]
        pooled[:, cases] = (x[:, cases] + 1) / shift[:, None] - 1
    logs = np.log1p(pooled)
    centre = np.median(logs, axis=1)
    for s in np.unique(code):
        idx = code == s
        logs[:, idx] -= (np.median(logs[:, idx], axis=1) - centre)[:, None]
    return np.expm1(logs), cond, code


def common_values(x, cond, code):
    """The common construction (S3): the informative strata only; each stratum moved, by a fold on
    count + 1 computed without the labels, so that its median log(count + 1) is the cohort's; then one
    registration for every case, the W1 fold of the aligned cases onto the aligned controls, as wade()
    registers one cohort. Returns the pool, its labels and codes."""
    n1s, n0s, informative = design(cond, code)
    keep = informative[code]
    x, cond, code = x[:, keep], cond[keep], code[keep]
    logs = np.log1p(x.astype(float))
    centre = np.median(logs, axis=1)
    for s in np.unique(code):
        idx = code == s
        logs[:, idx] -= (np.median(logs[:, idx], axis=1) - centre)[:, None]
    aligned = np.expm1(logs)
    shift = package_split(np.ascontiguousarray(aligned), np.ascontiguousarray(cond, dtype=np.uint8))[4]
    pooled = np.where(cond == 1, (aligned + 1) / shift[:, None] - 1, aligned)
    return pooled, cond, code


def twoway_values(x, cond, code, rounds=50):
    """The two-way construction (S4): log(count + 1) = stratum level + one fold for every case, fitted
    together by alternating robust steps: each stratum's level the median of its fold-adjusted values
    (all its samples), the fold the W1 fold of the level-adjusted cases onto the level-adjusted controls,
    as wade() registers one cohort. A stratum's level then rests on all its samples, so that no group is
    centred on a few of its members, and the fold on every stratum. Returns the pool (strata moved to
    their levels, cases carried back by the fold), its labels and codes."""
    n1s, n0s, informative = design(cond, code)
    keep = informative[code]
    x, cond, code = x[:, keep], np.ascontiguousarray(cond[keep], dtype=np.uint8), code[keep]
    logs = np.log1p(x.astype(float))
    strata = np.unique(code)
    beta = np.zeros(len(x))
    level = np.zeros((len(x), int(code.max()) + 1))
    for _ in range(rounds):
        adjusted = logs - beta[:, None] * cond
        for s in strata:
            level[:, s] = np.median(adjusted[:, code == s], axis=1)
        aligned = np.expm1(logs - level[:, code])
        new = np.log(package_split(np.ascontiguousarray(aligned), cond)[4])
        if np.array_equal(new, beta):
            break
        beta = new
    aligned = np.expm1(logs - level[:, code])
    pooled = np.where(cond == 1, (aligned + 1) / np.exp(beta)[:, None] - 1, aligned)
    return pooled, cond, code


def onepass_values(x, cond, code):
    """The two-way construction in one pass (S4b): one fold for every case, the weighted median of the
    strata's own W1 folds on count + 1 (weights n1s n0s / ns; where the weights below a value make exactly
    half, the midpoint, as the registration's own median); each stratum's level the median of its
    log(count + 1) once that fold is taken out of its cases. Deterministic, about one registration's cost.
    Returns the pool (strata moved to their levels, cases carried back by the fold), its labels and codes."""
    n1s, n0s, informative = design(cond, code)
    keep = informative[code]
    x, cond, code = x[:, keep], np.ascontiguousarray(cond[keep], dtype=np.uint8), code[keep]
    strata = np.unique(code)
    folds = np.empty((len(x), len(strata)))
    for i, s in enumerate(strata):
        idx = np.flatnonzero(code == s)
        folds[:, i] = np.log(package_split(np.ascontiguousarray(x[:, idx]), cond[idx])[4])
    weights = (n1s * n0s / (n1s + n0s))[strata]
    order = np.argsort(folds, axis=1, kind="stable")
    sorted_folds = np.take_along_axis(folds, order, axis=1)
    below = np.cumsum(weights[order], axis=1)
    total = weights.sum()
    first = np.argmax(2 * below >= total * (1 - 1e-12), axis=1)       # first value at or past half the weight
    rows = np.arange(len(x))
    exact_half = np.abs(2 * below[rows, first] - total) <= 1e-12 * total
    after = np.minimum(first + 1, len(strata) - 1)
    beta = np.where(exact_half, 0.5 * (sorted_folds[rows, first] + sorted_folds[rows, after]), sorted_folds[rows, first])
    logs = np.log1p(x.astype(float))
    adjusted = logs - beta[:, None] * cond
    level = np.zeros((len(x), int(code.max()) + 1))
    for s in strata:
        level[:, s] = np.median(adjusted[:, code == s], axis=1)
    aligned = np.expm1(logs - level[:, code])
    pooled = np.where(cond == 1, (aligned + 1) / np.exp(beta)[:, None] - 1, aligned)
    return pooled, cond, code


def thinned_pool(counts, factors, cond, code, rows, seed=1):
    """S4b with thinning, the construction adopted on 25 October 2026: the strata with both groups only;
    each stratum thinned to its own smallest size factor (the area test's equalization, the same draw);
    one fold for every case, the weighted median (weights n1s n0s / ns) of the strata's own W1 folds on
    the equalized counts; the higher group thinned by it in every stratum (draw (seed, gene, 3)); then each
    stratum moved, by a fold on count + 1, to the cohort's median log(count + 1). With one stratum it is
    research/subset_range.py's "equalized, thinned". Returns the pool (jittered), its labels and codes."""
    n1s, n0s, informative = design(cond, code)
    keep = informative[code]
    factors = np.broadcast_to(factors, np.shape(counts))
    c, f, cond, code = counts[:, keep], factors[:, keep], np.ascontiguousarray(cond[keep], dtype=np.uint8), codes(code[keep])
    level, _ = equalized(c, rows, f, code, seed)
    ones = np.ones(level.shape[1])
    x = expression(jittered(level, rows, seed), ones)
    strata = np.unique(code)
    folds = np.empty((len(x), len(strata)))
    weights = np.empty(len(strata))
    for i, s_ in enumerate(strata):
        idx = np.flatnonzero(code == s_)
        a = int(cond[idx].sum())
        weights[i] = a * (len(idx) - a) / len(idx)
        folds[:, i] = np.log(package_split(np.ascontiguousarray(x[:, idx]), cond[idx])[4])
    order = np.argsort(folds, axis=1, kind="stable")
    ranked = np.take_along_axis(folds, order, axis=1)
    below = np.cumsum(weights[order], axis=1)
    total = weights.sum()
    g = np.arange(len(x))
    first = np.argmax(2 * below >= total * (1 - 1e-12), axis=1)
    half = np.abs(2 * below[g, first] - total) <= 1e-12 * total
    fold = np.exp(np.where(half, 0.5 * (ranked[g, first] + ranked[g, np.minimum(first + 1, len(strata) - 1)]),
                           ranked[g, first]))[:, None]
    keep_share = np.where(cond == 1, np.where(fold > 1, 1 / fold, 1.0), np.where(fold < 1, fold, 1.0))
    thinned = np.empty_like(level)
    for i, (row, index) in enumerate(zip(level, rows)):
        thinned[i] = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(int(index), 3))).binomial(row, keep_share[i])
    logs = np.log1p(expression(jittered(thinned, rows, seed), ones))
    centre = np.median(logs, axis=1, keepdims=True)
    for s_ in strata:
        idx = code == s_
        logs[:, idx] += centre - np.median(logs[:, idx], axis=1, keepdims=True)
    return np.expm1(logs), cond, code


def subset_thinned(counts, factors, cond, code, rows, *, direction="both", stop=25, maximum=9_999, seed=1):
    """The thinned S4b subset test: thinned_pool's pool walked with no further registration, labels shuffled
    within strata, the package's walk and Besag-Clifford stopping."""
    pooled, labels, inner = thinned_pool(counts, factors, cond, code, rows, seed)
    order = np.ascontiguousarray(np.argsort(pooled, axis=1, kind="stable"), dtype=np.int64)
    values = np.ascontiguousarray(np.take_along_axis(pooled, order, axis=1))
    observed = _kernel.score(values, order, direction, labels)
    exceed, tried = np.zeros(len(counts), np.uint64), np.zeros(len(counts), np.uint64)
    while (live := ~settled(exceed, tried, stop=stop, maximum=maximum, threshold=-np.inf)).any():
        block = int(tried[live].min()) // STREAM_BLOCK
        exceed[:], tried[:] = _kernel.tally(values, order, direction, stream_block(labels, inner, block, seed),
                                            observed, exceed, tried, stop, maximum, block * STREAM_BLOCK, -np.inf)
    p = sequential_p(exceed, tried, stop=stop, maximum=maximum)
    return np.where(observed > 0, p, 1.0)


def subset_pooled(x, cond, code, *, direction="both", stop=25, maximum=9_999, seed=1, values_of=None):
    """The pooled subset test (S2; S3 with values_of=common_values): one walk of the pooled cases against
    the pooled controls, labels shuffled within strata, the package's native walk and Besag-Clifford
    stopping."""
    pooled, labels, inner = (values_of or pooled_values)(x, cond, code)
    labels = np.ascontiguousarray(labels, dtype=np.uint8)
    order = np.ascontiguousarray(np.argsort(pooled, axis=1, kind="stable"), dtype=np.int64)
    values = np.ascontiguousarray(np.take_along_axis(pooled, order, axis=1))
    observed = _kernel.score(values, order, direction, labels)
    exceed, tried = np.zeros(len(x), np.uint64), np.zeros(len(x), np.uint64)
    while (live := ~settled(exceed, tried, stop=stop, maximum=maximum, threshold=-np.inf)).any():
        block = int(tried[live].min()) // STREAM_BLOCK
        exceed[:], tried[:] = _kernel.tally(values, order, direction, stream_block(labels, inner, block, seed),
                                            observed, exceed, tried, stop, maximum, block * STREAM_BLOCK, -np.inf)
    p = sequential_p(exceed, tried, stop=stop, maximum=maximum)
    return np.where(observed > 0, p, 1.0)


def subset_combined(x, cond, code, *, direction="both", shuffles=1_999, seed=1):
    """The combined subset test (S1): each stratum walked on its own (its cases carried back by its own
    fold), the strata's scores added with weights sqrt(n1s n0s / ns), labels shuffled within strata;
    p = (1 + shuffles at least as high) / (1 + shuffles)."""
    n1s, n0s, informative = design(cond, code)
    parts = []
    for s in np.flatnonzero(informative):
        idx = np.flatnonzero(code == s)
        sub = np.ascontiguousarray(x[:, idx], dtype=float)
        lab = np.ascontiguousarray(cond[idx], dtype=np.uint8)
        shift = package_split(sub, lab)[4]
        reg = np.where(lab == 1, (sub + 1) / shift[:, None] - 1, sub)
        order = np.ascontiguousarray(np.argsort(reg, axis=1, kind="stable"), dtype=np.int64)
        values = np.ascontiguousarray(np.take_along_axis(reg, order, axis=1))
        weight = sqrt(n1s[s] * n0s[s] / (n1s[s] + n0s[s]))
        parts.append((idx, values, order, weight))
    score = lambda labels: sum(w * _kernel.score(v, o, direction, np.ascontiguousarray(labels[i], dtype=np.uint8))
                               for i, v, o, w in parts)
    observed = score(np.asarray(cond))
    bar = observed - TIE_RTOL * np.abs(observed)
    hits = np.zeros(len(x))
    done, block = 0, 0
    while done < shuffles:
        rows = stream_block(cond, code, block, seed)[:shuffles - done]
        for row in rows:
            hits += score(row) >= bar
        done += len(rows)
        block += 1
    return np.where(observed > 0, (1 + hits) / (1 + shuffles), 1.0)


def exact_area(x, cond, code, *, direction="both"):
    """Exact p-values by enumerating every within-stratum labelling (small designs only)."""
    cond, code = np.asarray(cond), np.asarray(code)
    n1s, n0s, informative = design(cond, code)
    out = np.empty(len(x))
    for g, row in enumerate(x):
        totals = np.zeros(1)
        for s in np.flatnonzero(informative):
            idx = np.flatnonzero(code == s)
            sub = np.array([row[list(pick)].sum() for pick in itertools.combinations(idx, int(n1s[s]))])
            totals = (totals[:, None] + sub[None, :]).ravel()
        observed = row[(cond == 1) & informative[code]].sum()
        centre = totals.mean()
        tie = TIE_RTOL * np.abs(row).sum()
        if direction == "up":
            out[g] = np.mean(totals >= observed - tie)
        elif direction == "down":
            out[g] = np.mean(totals <= observed + tie)
        else:
            out[g] = np.mean(np.abs(totals - centre) >= abs(observed - centre) - tie)
    return out


# ---------------------------------------------------------------------------------------------
# Sections

def exact():
    rng = np.random.default_rng(2026102401)
    print("\n== exact: the stratified saddlepoint and the stratified stream against every within-stratum "
          "labelling", flush=True)
    # One stratum: the prototype reproduces the package's saddlepoint and stream.
    x = rng.gamma(2, 50, (300, 40)) + rng.random((300, 40))
    cond = np.r_[np.ones(15, int), np.zeros(25, int)]
    one = np.zeros(40, np.int64)
    p_pkg, e_pkg = mean_saddlepoint(x, cond, direction="both")
    p_new, e_new = saddlepoint_area(x, cond, one, direction="both")
    from wade.permutation import stream_block as package_stream
    same_stream = np.array_equal(stream_block(cond, one, 3, 7), package_stream(cond, 3, 7))
    print(f"  one stratum: largest relative difference from the package's saddlepoint p {np.max(np.abs(p_new / p_pkg - 1)):.1e}, "
          f"its error estimate {np.max(np.abs(e_new - e_pkg) / np.maximum(e_pkg, TINY)):.1e}; same stream: {same_stream}",
          flush=True)
    # Small stratified designs, every labelling enumerated.
    designs = {"three strata (5:3, 4:4, 2:6)": [(5, 3), (4, 4), (2, 6)],
               "two unbalanced (9:2, 2:8)": [(9, 2), (2, 8)],
               "four small (3:2, 2:3, 3:3, 1:4)": [(3, 2), (2, 3), (3, 3), (1, 4)],
               "with a cases-only stratum (5:3, 4:0, 3:5)": [(5, 3), (4, 0), (3, 5)]}
    collected = []
    for name, sizes in designs.items():
        cond = np.concatenate([np.r_[np.ones(a, int), np.zeros(b, int)] for a, b in sizes])
        code = np.concatenate([np.full(a + b, s) for s, (a, b) in enumerate(sizes)])
        n = len(cond)
        # Genes from tight to skewed, each with study shifts, so strata differ.
        shape = np.repeat([0.5, 2, 8], 100)
        x = rng.gamma(shape[:, None], 30, (300, n)) * np.exp(rng.normal(0, 0.7, (300, int(code.max()) + 1)))[:, code]
        x += rng.random(x.shape)
        x[:, cond == 1] *= np.exp(rng.normal(0, 0.4, (300, 1)))             # some genes shifted, some not
        exact_p = exact_area(x, cond, code)
        sp, err = saddlepoint_area(x, cond, code)
        mc = shuffled_area(x, cond, code, stop=400, maximum=99_999, seed=5)
        rel = np.abs(sp / exact_p - 1)
        flagged = err > 0.2
        n1s, n0s, inf = design(cond, code)
        space = np.prod([comb(int(a + b), int(a)) for a, b, i in zip(n1s, n0s, inf) if i])
        stops = exact_p >= 400 / 99_999 * 2                    # runs that end at their 400th exceedance
        mc_rel = np.abs(mc[stops] / exact_p[stops] - 1)
        print(f"  {name}: {space:,} labellings; saddlepoint relative error median {np.median(rel):.3f}, 95th "
              f"{np.quantile(rel, .95):.3f}, max {rel.max():.3f} (among p <= 0.05: max "
              f"{rel[exact_p <= .05].max() if np.any(exact_p <= .05) else np.nan:.3f}); error estimate > 0.2 in "
              f"{flagged.mean():.0%}, where the largest actual error is "
              f"{rel[flagged].max() if flagged.any() else np.nan:.3f}, against {rel[~flagged].max() if (~flagged).any() else np.nan:.3f} "
              f"where not", flush=True)
        print(f"    shuffles (stop 400, relative SE 0.05): relative deviation from exact median {np.median(mc_rel):.3f}, "
              f"share beyond 0.15 (3 SE) {np.mean(mc_rel > 0.15):.1%}, over {stops.sum()} genes", flush=True)
        ratio = sp / exact_p
        beyond = (ratio < 2 / 3) | (ratio > 1.5)
        print(f"    kept by the error estimate (<= 0.2) {np.mean(~flagged):.0%}; of those, beyond 0.67-1.5x of exact "
              f"{np.mean(beyond[~flagged]):.1%} ({int(beyond[~flagged].sum())} genes); of those sent to shuffles, "
              f"{np.mean(beyond[flagged]):.0%} would have been", flush=True)
        collected.append((err, ratio))
    err = np.concatenate([e for e, _ in collected])
    ratio = np.concatenate([r for _, r in collected])
    print("\n  estimated error | genes | median actual error | beyond 0.67-1.5x", flush=True)
    for lo, hi in [(0, .05), (.05, .1), (.1, .2), (.2, .5), (.5, np.inf)]:
        k = (err >= lo) & (err < hi)
        if k.any():
            print(f"  {lo:g}-{hi:g} | {k.sum()} | {np.median(np.abs(ratio[k] - 1)):.1%} | "
                  f"{np.mean((ratio[k] < 2 / 3) | (ratio[k] > 1.5)):.0%}", flush=True)


ACCURACY_DESIGNS = {
    "TCGA prostate sites (74:23, 50:13, 33:11, 28:4, 22:1)": [(74, 23), (50, 13), (33, 11), (28, 4), (22, 1)],
    "TCGA breast sites (78:11, 77:72, 47:15, 42:4, 34:8)": [(78, 11), (77, 72), (47, 15), (42, 4), (34, 8)],
    "two unbalanced studies (200:20, 30:150)": [(200, 20), (30, 150)],
    "twenty small studies (20 x 6:4)": [(6, 4)] * 20,
    "one large study and ten tiny ones (400:40, 10 x 3:2)": [(400, 40)] + [(3, 2)] * 10,
}


def reference(x, cond, code, labellings=20_000_000, block=20_000, seed=7):
    """Two-sided p-values from random within-stratum labellings (the stratified shuffles are exact)."""
    n1s, n0s, informative = design(cond, code)
    means = np.stack([x[:, code == s].mean(axis=1) for s in range(int(code.max()) + 1)], axis=1)
    c = x - means[:, code]
    c[:, ~informative[code]] = 0.0                             # strata that cannot move add nothing
    observed = (c * cond).sum(axis=1)
    reach = np.abs(observed) - 1e-9 * np.abs(c).sum(axis=1)
    rng = np.random.default_rng(seed)
    rows = np.tile(cond.astype(np.float64), (block, 1))
    groups = [np.flatnonzero(code == s) for s in np.flatnonzero(informative)]
    hits, drawn = np.zeros(len(x)), 0
    while drawn < labellings:
        for idx in groups:
            rows[:, idx] = rng.permuted(rows[:, idx], axis=1)
        hits += (np.abs(c @ rows.T) >= reach[:, None]).sum(axis=1)
        drawn += block
    return hits / drawn, hits


def accuracy(per_decade=10):
    import json
    from pathlib import Path
    import wade
    from wade_research.point_simulation import generate
    cfg = json.loads((Path(__file__).resolve().parent / "protocols/confirmation.json").read_text())
    print("\n== accuracy: the stratified saddlepoint on realistic designs against 20 million within-stratum "
          "labellings; the confirmation's generator, a gene-specific batch scale per study (lognormal, sd 0.5) "
          "and a depth per study, every case x 2-8 in a quarter of the genes and one or two cases x 20 in a "
          f"tenth; up to {per_decade} genes per decade of p from 0.1 to 1e-6", flush=True)
    bands = [(0, .05), (.05, .1), (.1, .2), (.2, .5), (.5, np.inf)]
    rows_all = []
    for d, (name, sizes) in enumerate(ACCURACY_DESIGNS.items()):
        n1, n0 = sum(a for a, _ in sizes), sum(b for _, b in sizes)
        data = generate(cfg, dict(n_case=n1, n_control=n0, panel="complete_null",
                                  biology=("gamma", "correlated_lognormal")[d % 2], seed=[2026102402, d, 0]))
        counts, cond0 = data["counts"].astype(float), data["cond"]
        # Lay the generator's cases and controls out into the strata.
        cases, controls = np.flatnonzero(cond0 == 1), np.flatnonzero(cond0 == 0)
        order, code, at1, at0 = [], [], 0, 0
        for s, (a, b) in enumerate(sizes):
            order += list(cases[at1:at1 + a]) + list(controls[at0:at0 + b])
            code += [s] * (a + b)
            at1, at0 = at1 + a, at0 + b
        counts, cond, code = counts[:, order], cond0[order], np.array(code)
        rng = np.random.default_rng([2026102402, d, 1])
        S = len(sizes)
        batch = np.exp(rng.normal(0, 0.5, (len(counts), S)))
        depth = np.exp(rng.normal(0, 0.5, S))
        counts = rng.poisson(counts * batch[:, code] * depth[code])
        shifted = rng.random(len(counts)) < 0.25
        counts[np.ix_(shifted, cond == 1)] *= rng.choice([2, 3, 4, 8], shifted.sum())[:, None]
        for g in np.flatnonzero(~shifted & (rng.random(len(counts)) < 0.1)):
            counts[g, rng.choice(np.flatnonzero(cond == 1), rng.integers(1, 3), replace=False)] *= 20
        keep = np.flatnonzero(wade.filter_counts(counts, min_count=3, min_samples=2))
        factors = wade.fit_normalization(counts).size_factors
        x = area_values(counts[keep], keep, factors, code, 1)
        p, error = saddlepoint_area(x, cond, code)
        pick = []
        for lo in range(1, 7):
            inside = np.flatnonzero((p <= 10.0 ** -(lo - 1)) & (p > 10.0 ** -lo))
            pick += list(rng.choice(inside, min(per_decade, len(inside)), replace=False))
        pick = np.array(sorted(pick))
        ref, hits = reference(x[pick], cond, code)
        ok = hits >= 100                                       # reference within about 10%
        ratio = p[pick] / np.where(ref > 0, ref, np.nan)
        out = (ratio < 2 / 3) | (ratio > 1.5)
        kept = error[pick] <= 0.2
        print(f"  {name}: {len(keep):,} genes, shuffled {np.mean(error > 0.2):.1%} of them; picked {len(pick)}, "
              f"{ok.sum()} with a reference of 100+ hits; kept {kept[ok].mean():.0%} of those, of which beyond "
              f"0.67-1.5x {np.mean(out[ok & kept]):.1%}; sent to shuffles that would have been beyond "
              f"{np.mean(out[ok & ~kept]) if (ok & ~kept).any() else np.nan:.0%}", flush=True)
        rows_all.append((error[pick][ok], ratio[ok]))
    err = np.concatenate([e for e, _ in rows_all])
    ratio = np.concatenate([r for _, r in rows_all])
    print("\n  estimated error | genes | median actual error | beyond 0.67-1.5x", flush=True)
    for lo, hi in bands:
        k = (err >= lo) & (err < hi)
        if k.any():
            print(f"  {lo:g}-{hi:g} | {k.sum()} | {np.median(np.abs(ratio[k] - 1)):.1%} | "
                  f"{np.mean((ratio[k] < 2 / 3) | (ratio[k] > 1.5)):.0%}", flush=True)


# ---------------------------------------------------------------------------------------------
# The count model with studies

MODEL_DESIGNS = {
    "two studies, case shares 80% and 20%, depths 1 and 1/4": ([(176, 44), (44, 176)], (1, 1 / 4)),
    "twenty studies of 22, case shares 68% and 32%, depths 1 and 1/4": ([(15, 7)] * 10 + [(7, 15)] * 10,
                                                                        (1, 1 / 4) * 10),
}
EDGES = (32, 128)


_RLE = {}


def _rle():
    """RLE of the tumours from each half of the genes (for the other half), and from all of them."""
    if not _RLE:
        from count_model import halves, tumours
        from wade.normalize import fit_normalization
        full, half = tumours(), halves()
        _RLE["halves"] = [fit_normalization(full[half == 1 - h]).size_factors for h in (0, 1)]
        _RLE["whole"] = fit_normalization(full).size_factors
    return _RLE["halves"], _RLE["whole"]


def model_dataset(sizes, depths, batch_sd, fold, subset, seed, genes, noise_sd=0.0):
    """One data set under the count model with studies. Each gene gets its own random assignment of
    the tumours to the design's slots (stratum by stratum, cases first), and is thinned binomially with
    chance depth_s x batch_gs (x 1/fold for controls, or for everyone outside a planted subset), all at
    most 1, so the counts follow the count model exactly. Returns the counts, each gene's size factors as
    the tests see them (RLE of the unthinned tumours from the other half of the genes, times depth_s), the
    labels, the strata, the genes' rows in the tumour matrix, and each gene's smallest expected reads per
    sample over the strata's groups. noise_sd > 0 adds technical noise to alternate studies: each of
    their counts thinned further by exp(-noise_sd |Z|), Z standard normal per sample and gene."""
    from count_model import halves, tumours
    raw, half = tumours(), halves()
    rows = np.asarray(genes)
    raw, half = raw[rows], half[rows]
    rng = np.random.default_rng([2026102403, len(sizes), round(100 * batch_sd), *np.round(4 * np.ravel(fold)).astype(int),
                                 int(subset), seed, round(100 * noise_sd)])
    cond = np.concatenate([np.r_[np.ones(a, np.uint8), np.zeros(b, np.uint8)] for a, b in sizes])
    code = np.concatenate([np.full(a + b, i) for i, (a, b) in enumerate(sizes)])
    n, S = len(cond), len(sizes)
    pick = np.argsort(rng.random(raw.shape), axis=1)[:, :n]
    latent = np.take_along_axis(raw, pick, axis=1)
    depth = np.asarray(depths, float)
    log_batch = rng.normal(0, batch_sd, (len(rows), S)) if batch_sd > 0 else np.zeros((len(rows), S))
    batch = np.exp(log_batch - log_batch.max(axis=1, keepdims=True))
    keep = depth[code] * batch[:, code]
    if noise_sd > 0:
        noisy = (code % 2 == 1)
        keep = keep * np.where(noisy, np.exp(-noise_sd * np.abs(rng.normal(size=keep.shape))), 1.0)
    if subset:                                                  # a tenth of each stratum's cases raised 8x
        raised = np.zeros((len(rows), n), bool)
        for s_, (a, b) in enumerate(sizes):
            start = int(np.flatnonzero(code == s_)[0])
            k = max(1, round(0.1 * a))
            for g in range(len(rows)):
                raised[g, start + rng.choice(a, k, replace=False)] = True
        keep = keep * np.where(raised, 1.0, 1 / 8)
    elif np.any(np.asarray(fold) != 1):
        per = np.broadcast_to(np.asarray(fold, float), (S,))      # one fold, or one per stratum
        keep = keep * np.where(cond == 1, 1.0, 1 / per[code])
    counts = rng.binomial(latent, keep)
    rle, whole = _rle()
    factors = np.where(half[:, None] == 0, rle[0][pick], rle[1][pick]) * depth[code]
    factors /= np.exp(np.log(factors).mean(axis=1, keepdims=True))
    level = (raw / whole).mean(axis=1)
    lowest = np.full(len(rows), np.inf)
    for s_ in range(S):
        share = keep[:, code == s_].mean(axis=1) if not subset else depth[s_] * batch[:, s_] / 8
        lowest = np.minimum(lowest, level * share)
    return counts, factors, cond, code, lowest


def calibration_line(name, p, lowest, alpha=0.01, edges=EDGES):
    bins = np.digitize(lowest, edges)
    labels = [f"<{edges[0]}"] + [f"{a}-{b}" for a, b in zip(edges[:-1], edges[1:])] + [f">={edges[-1]}"]
    parts = [f"{lab} {np.mean(p[bins == b] <= alpha):6.2%}" if np.sum(bins == b) >= 200 else f"{lab}     --"
             for b, lab in enumerate(labels)]
    return f"    {name:22s} " + " | ".join(parts) + f" || all {np.mean(p <= .05):.2%} / {np.mean(p <= .01):.2%} / {np.mean(p <= .001):.3%}"


def model(genes_per_set=3_000):
    from count_model import area_pvalues, tumours
    from wade._means import area_values as package_area_values
    from count_model import rescaled_subset_test as subset_test
    print("\n== model: the count model with studies; each gene split at random into the design's studies; "
          f"{genes_per_set:,} genes per data set (expressed, at random); share at p <= 0.01 by each gene's "
          "smallest expected reads per sample over the strata's groups, then over all genes at 0.05 / 0.01 / 0.001",
          flush=True)
    raw = tumours()
    expressed = np.flatnonzero((raw > 0).mean(axis=1) >= 0.5)
    pick = np.sort(np.random.default_rng(2026102404).choice(expressed, genes_per_set, replace=False))
    for name, (sizes, depths) in MODEL_DESIGNS.items():
        for batch_sd, noise_sd in ((0.0, 0.0), (0.5, 0.0), (0.5, 0.5)):
            for label, fold, subset in (("no fold", 1, False), ("a 1.5x fold", 1.5, False), ("a planted subset", 1, True)):
                if noise_sd > 0 and fold != 1:
                    continue
                counts, factors, cond, code, lowest = model_dataset(sizes, depths, batch_sd, fold, subset, 1, pick,
                                                                    noise_sd=noise_sd)
                rows = pick
                zU = package_area_values(counts, rows, factors, 1)
                zP = area_values(counts, rows, factors, code, 1)
                x = expression(jittered(counts, rows, 1), factors)
                out = dict()
                if not subset:
                    out["area, unstratified"] = area_pvalues(zU, cond, 1)
                    out["area, global thinning"] = area_p(zU, cond, code)
                    out["area, stratum thinning"] = area_p(zP, cond, code)
                    out["area, inverse variance"] = area_p(inverse_variance(zP, code), cond, code)
                if fold == 1:
                    out["subset, unstratified"] = subset_test(x, cond, direction="both", stop=25, maximum=9_999, seed=1)[0]
                    out["subset, pooled (S2)"] = subset_pooled(x, cond, code)
                    out["subset, combined (S1)"] = subset_combined(x, cond, code)
                noise = f"; extra noise sd {noise_sd:g} in alternate studies" if noise_sd else ""
                print(f"\n  {name}; batch sd {batch_sd:g}{noise}; {label}", flush=True)
                for key, p in out.items():
                    print(calibration_line(key, p, lowest), flush=True)


# ---------------------------------------------------------------------------------------------
# Real data: TCGA sites, and TCGA with GTEx

def relabelled(groups, share_of, rng, genes):
    """A random within-group labelling per gene: for each group (stratum), its samples in random order,
    the first round(share x size) the cases. Returns, per gene, the sample order (strata as blocks, cases
    first in each), and the labels and codes of that layout."""
    layout_cond, layout_code, blocks = [], [], []
    for s_, members in enumerate(groups):
        k = int(round(share_of(s_) * len(members)))
        layout_cond += [1] * k + [0] * (len(members) - k)
        layout_code += [s_] * len(members)
        blocks.append(np.asarray(members))
    orders = np.empty((genes, len(layout_cond)), np.int64)
    for g in range(genes):
        orders[g] = np.concatenate([rng.permutation(b) for b in blocks])
    return orders, np.array(layout_cond, np.uint8), np.array(layout_code)


def real_null(name, counts, factors, groups, share_of, genes_used, lowest_reads, seed=2026102405, fold=1.0,
              area=True, combined=True):
    """Both tests, unstratified and stratified, on real samples relabelled at random within strata. With
    fold > 1, every pseudo-case carries that fold (the pseudo-controls thinned by 1/fold, gene by gene):
    the subset test's null still holds, and only it is run."""
    from count_model import area_pvalues
    from wade._means import area_values as package_area_values
    from count_model import rescaled_subset_test as subset_test
    rng = np.random.default_rng(seed)
    orders, cond, code = relabelled(groups, share_of, rng, len(genes_used))
    c = np.take_along_axis(counts[genes_used], orders, axis=1).astype(np.int64)
    if fold != 1:
        c = np.random.default_rng([seed, 1]).binomial(c, np.where(cond == 1, 1.0, 1 / fold))
    f = factors[orders]
    rows = np.asarray(genes_used)
    zU = package_area_values(c, rows, f, 1)
    zP = area_values(c, rows, f, code, 1)
    x = expression(jittered(c, rows, 1), f)
    n1s, n0s, inf = design(cond, code)
    print(f"\n  {name}{f', every pseudo-case {fold:g}x' if fold != 1 else ''}: {len(groups)} strata, "
          f"{int(inf.sum())} with both groups; {int(cond.sum())} pseudo-cases, "
          f"{int((cond == 0).sum())} pseudo-controls; {len(rows):,} genes", flush=True)
    out = {}
    if area:
        out = {"area, unstratified": area_pvalues(zU, cond, 1), "area, stratum thinning": area_p(zP, cond, code)}
    out.update({"subset, unstratified": subset_test(x, cond, direction="both", stop=25, maximum=9_999, seed=1)[0],
                "subset, pooled (S2)": subset_pooled(x, cond, code),
                "subset, common (S3)": subset_pooled(x, cond, code, values_of=common_values),
                "subset, two-way (S4)": subset_pooled(x, cond, code, values_of=twoway_values),
                "subset, one pass (S4b)": subset_pooled(x, cond, code, values_of=onepass_values)})
    if combined:
        out["subset, combined (S1)"] = subset_combined(x, cond, code)
    for key, p in out.items():
        print(calibration_line(key, p, lowest_reads), flush=True)


def sites(genes_per_set=4_000):
    from count_model import tcga
    d = tcga("prostate")
    tumour = np.flatnonzero(d["cond"] == 1)
    site = np.array([s.split("-")[1] for s in d["samples"][tumour]])
    names, sizes = np.unique(site, return_counts=True)
    usable = names[sizes >= 4]
    groups = [tumour[site == s] for s in usable]
    raw = d["raw"].astype(np.int64)
    lower = (raw[:, tumour] / d["factors"][tumour]).mean(axis=1)
    pick = np.sort(np.random.default_rng(2026102406).choice(np.flatnonzero(lower >= 1), genes_per_set, replace=False))
    print(f"\n== sites: TCGA prostate tumours relabelled at random within their tissue source sites ({len(usable)} sites "
          f"with 4+ tumours); in alternate sites (by size) a quarter or three quarters are pseudo-cases. Share at "
          f"p <= 0.01 by the gene's mean normalized count, then overall at 0.05 / 0.01 / 0.001", flush=True)
    by_size = np.argsort(-np.array([len(g) for g in groups]))
    rank = np.empty(len(groups), int)
    rank[by_size] = np.arange(len(groups))
    real_null("prostate tumours by site", raw, d["factors"], groups, lambda s_: 0.25 if rank[s_] % 2 else 0.75,
              pick, lower[pick])


def _normals():
    """TCGA normal-adjacent and GTEx prostate normals from the examples' bundle: counts (all genes),
    study per sample, and the TCGA examples' test genes."""
    import json
    from pathlib import Path
    import polars as pl
    from count_model import ROOT
    sys.path.insert(0, str(ROOT / "tools"))
    import tempfile
    import tcga_examples as TC
    config = Path(__import__("os").environ["WADE_TCGA_CONFIG"]).resolve()
    bundle = config.parent / json.loads(config.read_text())["bundles"]["prostate"]
    samples = pl.read_parquet(bundle / "samples.parquet")
    chosen = samples.filter(((pl.col("study") == "TCGA") & (pl.col("sample_class") == "normal_adjacent"))
                            | ((pl.col("study") == "gtex") & (pl.col("sample_class") == "control")))
    frame = pl.read_ipc(bundle / "counts.feather", memory_map=False)
    counts = frame.select(chosen["library"].to_list()).to_numpy().astype(np.float64)
    with tempfile.TemporaryDirectory() as out:
        cohort = TC.load_cohort("prostate", out)
    return counts, chosen["study"].to_numpy(), cohort.test_genes, cohort.genes["gene_name"].to_numpy()


def gtex(genes_per_set=4_000):
    import wade
    counts, study, test_genes, names = _normals()
    tested = np.flatnonzero(test_genes & (counts.sum(axis=1) > 0))
    factors = wade.fit_normalization(counts).size_factors
    lower = (counts / factors).mean(axis=1)
    groups = [np.flatnonzero(study == "TCGA"), np.flatnonzero(study == "gtex")]
    pick = np.sort(np.random.default_rng(2026102407).choice(tested[lower[tested] >= 1], genes_per_set, replace=False))
    print(f"\n== gtex: prostate normals, {len(groups[0])} TCGA normal-adjacent and {len(groups[1])} GTEx, relabelled at "
          f"random within study, three quarters of TCGA's and a quarter of GTEx's pseudo-cases; RLE over all of them. "
          f"Share at p <= 0.01 by the gene's mean normalized count, then overall at 0.05 / 0.01 / 0.001", flush=True)
    real_null("TCGA and GTEx normals by study", counts, factors, groups, lambda s_: (0.75, 0.25)[s_], pick, lower[pick])
    # The bridge: the two studies' normals against each other, which strata cannot do.
    cond = (study == "TCGA").astype(int)
    mask = np.zeros(len(counts), bool)
    mask[tested] = True
    res = wade.wade(counts, cond, normalization=wade.Normalization.from_size_factors(counts, factors, source="normals RLE"),
                    test_genes=mask, seed=1)
    t = res.tested
    print(f"\n  the bridge, TCGA normal-adjacent (cases) against GTEx normals, wade() with its defaults on {t.sum():,} genes: "
          f"area called {np.sum(res.area_padj[t] <= .05):,} ({np.mean(res.area_padj[t] <= .05):.0%}), "
          f"{np.sum((res.area_padj[t] <= .05) & (np.abs(res.log2fc[t]) >= 1)):,} of them at least 2-fold; "
          f"subset called {np.sum(res.subset_padj[t] <= .05):,} ({np.mean(res.subset_padj[t] <= .05):.0%})", flush=True)


def real():
    from count_model import tcga
    from wade.pvalues import bh_adjust
    import wade
    d = tcga("prostate")
    cond_all, raw, factors, names = d["cond"], d["raw"], d["factors"], d["names"]
    site = np.array([s.split("-")[1] for s in d["samples"]])
    code_all = codes(site)
    n1s, n0s, informative = design(cond_all, code_all)
    keep = informative[code_all]
    rows = np.arange(len(raw))
    print(f"\n== real: TCGA prostate tumours against normals; stratified by tissue source site, {int(informative.sum())} "
          f"of {len(informative)} sites have both groups ({int(cond_all[keep].sum())} tumours, {int((cond_all[keep] == 0).sum())} "
          f"normals); N_eff {np.sum((n1s * n0s / (n1s + n0s))[informative]):.1f} against "
          f"{cond_all.sum() * (cond_all == 0).sum() / len(cond_all):.1f} unstratified; {len(raw):,} tested genes", flush=True)
    res = wade.wade(raw, cond_all, normalization=wade.Normalization.from_size_factors(raw, factors, source="TCGA primary RLE"),
                    gene_names=names, seed=1)
    c, f, cond, code = raw[:, keep], factors[keep], cond_all[keep], codes(code_all[keep])
    z = area_values(c, rows, np.broadcast_to(f, c.shape), code, 1)
    area = bh_adjust(area_p(z, cond, code))
    x = expression(jittered(c, rows, 1), f)
    subset = bh_adjust(subset_pooled(x, cond, code))
    twoway = bh_adjust(subset_pooled(x, cond, code, values_of=twoway_values))
    onepass = bh_adjust(subset_pooled(x, cond, code, values_of=onepass_values))
    thin = bh_adjust(subset_thinned(c, np.broadcast_to(f, c.shape), cond, code, rows))
    same = wade.wade(c, cond, normalization=wade.Normalization.from_size_factors(c, f, source="TCGA primary RLE"),
                     gene_names=names, seed=1)                  # the same samples, no strata
    print(f"  the same {keep.sum()} samples unstratified: area {np.sum(same.area_padj <= .05):,}, subset "
          f"{np.sum(same.subset_padj <= .05):,}", flush=True)
    print(f"  area called: unstratified {np.sum(res.area_padj <= .05):,}, stratified {np.sum(area <= .05):,}, both "
          f"{np.sum((res.area_padj <= .05) & (area <= .05)):,}", flush=True)
    print(f"  subset called: unstratified {np.sum(res.subset_padj <= .05):,}, stratified pooled (S2) {np.sum(subset <= .05):,}, "
          f"two-way (S4) {np.sum(twoway <= .05):,}, one pass (S4b) {np.sum(onepass <= .05):,}, S4b thinned "
          f"{np.sum(thin <= .05):,}; with the same samples unstratified, S4 shares "
          f"{np.sum((same.subset_padj <= .05) & (twoway <= .05)):,}, S4b {np.sum((same.subset_padj <= .05) & (onepass <= .05)):,} "
          f"and S4b thinned {np.sum((same.subset_padj <= .05) & (thin <= .05)):,}", flush=True)
    for gene in ("ERG", "ETV1", "ETV4", "SPINK1", "AMACR", "KRT5", "TP63"):
        i = np.flatnonzero(names == gene)
        if len(i):
            i = i[0]
            print(f"    {gene:7s} area q {res.area_padj[i]:.2g} -> {area[i]:.2g}; subset q {res.subset_padj[i]:.2g} "
                  f"(same samples {same.subset_padj[i]:.2g}) -> S2 {subset[i]:.2g}, S4 {twoway[i]:.2g}, S4b {onepass[i]:.2g}, "
                  f"S4b thinned {thin[i]:.2g}", flush=True)


def layout(genes_per_set=4_000):
    """The real contrast's layout as a null: in each of the five sites with both groups, its tumours
    relabelled at random with the real design's case share (one fewer pseudo-case where a site's
    tumours cannot fill it), so that lopsided sites (22 tumours : 1 normal) are tested."""
    from count_model import tcga
    d = tcga("prostate")
    site_all = np.array([s_.split("-")[1] for s_ in d["samples"]])
    tumour = np.flatnonzero(d["cond"] == 1)
    real = {}
    for s_ in np.unique(site_all):
        here = site_all == s_
        n1, n0 = int(np.sum(d["cond"][here] == 1)), int(np.sum(d["cond"][here] == 0))
        if n1 and n0:
            real[s_] = (n1, n0)
    groups = [tumour[site_all[tumour] == s_] for s_ in real]
    shares = [max(1, min(len(g) - 1, round(len(g) * a / (a + b)))) / len(g) for g, (a, b) in zip(groups, real.values())]
    raw = d["raw"].astype(np.int64)
    lower = (raw[:, tumour] / d["factors"][tumour]).mean(axis=1)
    pick = np.sort(np.random.default_rng(2026102409).choice(np.flatnonzero(lower >= 1), genes_per_set, replace=False))
    print(f"\n== layout: the five sites with both groups ({', '.join(f'{k} {a}:{b}' for k, (a, b) in real.items())}), "
          f"their tumours relabelled at random with those case shares", flush=True)
    real_null("prostate tumours in the real design's sites", raw, d["factors"], groups, lambda s_: shares[s_], pick,
              lower[pick])


def common(genes_per_set=3_000):
    """The common construction (S3) against the pooled one (S2) and the unstratified test, on every null
    that has strata and on the planted subsets: the real layout, the sites (with and without a fold),
    TCGA with GTEx, and the model's designs with batch effects."""
    from count_model import tcga, tumours
    from count_model import rescaled_subset_test as subset_test
    print("\n== common: the subset test with one registration after a label-free alignment of the strata "
          "(S3), against S2 and the unstratified test", flush=True)
    layout_sites = []
    d = tcga("prostate")
    site_all = np.array([s_.split("-")[1] for s_ in d["samples"]])
    tumour = np.flatnonzero(d["cond"] == 1)
    raw = d["raw"].astype(np.int64)
    lower = (raw[:, tumour] / d["factors"][tumour]).mean(axis=1)
    real = {}
    for s_ in np.unique(site_all):
        here = site_all == s_
        n1, n0 = int(np.sum(d["cond"][here] == 1)), int(np.sum(d["cond"][here] == 0))
        if n1 and n0:
            real[s_] = (n1, n0)
    groups = [tumour[site_all[tumour] == s_] for s_ in real]
    shares = [max(1, min(len(g) - 1, round(len(g) * a / (a + b)))) / len(g) for g, (a, b) in zip(groups, real.values())]
    pick = np.sort(np.random.default_rng(2026102409).choice(np.flatnonzero(lower >= 1), genes_per_set, replace=False))
    for factor in (1.0, 1.5):
        real_null("the real design's five sites", raw, d["factors"], groups, lambda s_: shares[s_], pick, lower[pick],
                  fold=factor, area=False, combined=False)
    site = site_all[tumour]
    names, sizes = np.unique(site, return_counts=True)
    usable = names[sizes >= 4]
    groups = [tumour[site == s_] for s_ in usable]
    by_size = np.argsort(-np.array([len(g) for g in groups]))
    rank = np.empty(len(groups), int)
    rank[by_size] = np.arange(len(groups))
    for factor in (1.0, 1.5):
        real_null("prostate tumours by site", raw, d["factors"], groups, lambda s_: 0.25 if rank[s_] % 2 else 0.75,
                  pick, lower[pick], fold=factor, area=False, combined=False)
    counts, study, test_genes, _ = _normals()
    import wade
    tested = np.flatnonzero(test_genes & (counts.sum(axis=1) > 0))
    factors = wade.fit_normalization(counts).size_factors
    low = (counts / factors).mean(axis=1)
    groups = [np.flatnonzero(study == "TCGA"), np.flatnonzero(study == "gtex")]
    pick = np.sort(np.random.default_rng(2026102407).choice(tested[low[tested] >= 1], genes_per_set, replace=False))
    real_null("TCGA and GTEx normals by study", counts, factors, groups, lambda s_: (0.75, 0.25)[s_], pick, low[pick],
              area=False, combined=False)
    raw = tumours()
    expressed = np.flatnonzero((raw > 0).mean(axis=1) >= 0.5)
    pick = np.sort(np.random.default_rng(2026102404).choice(expressed, genes_per_set, replace=False))
    for name, (sizes, depths) in MODEL_DESIGNS.items():
        for label, factor, subset in (("no fold", 1, False), ("every case 1.5x", 1.5, False), ("a planted subset", 1, True)):
            counts, factors, cond, code, lowest = model_dataset(sizes, depths, 0.5, factor, subset, 1, pick)
            x = expression(jittered(counts, pick, 1), factors)
            print(f"\n  {name}; batch sd 0.5; {label}", flush=True)
            for key, p in (("subset, unstratified", subset_test(x, cond, direction="both", stop=25, maximum=9_999, seed=1)[0]),
                           ("subset, pooled (S2)", subset_pooled(x, cond, code)),
                           ("subset, common (S3)", subset_pooled(x, cond, code, values_of=common_values)),
                           ("subset, two-way (S4)", subset_pooled(x, cond, code, values_of=twoway_values)),
                           ("subset, one pass (S4b)", subset_pooled(x, cond, code, values_of=onepass_values))):
                print(calibration_line(key, p, lowest), flush=True)


def registration(genes_per_set=2_000, shuffles=999):
    """Why the combined construction (S1) fails on the TCGA sites: the same statistic with each stratum
    registered onto its controls, and on its raw values (exact here, where no fold exists), over all sites
    and over the sites of 12 or more."""
    from count_model import tcga
    d = tcga("prostate")
    tumour = np.flatnonzero(d["cond"] == 1)
    site = np.array([s_.split("-")[1] for s_ in d["samples"][tumour]])
    names, sizes = np.unique(site, return_counts=True)
    groups = [tumour[site == s_] for s_ in names[sizes >= 4]]
    raw = d["raw"].astype(np.int64)
    lower = (raw[:, tumour] / d["factors"][tumour]).mean(axis=1)
    pick = np.sort(np.random.default_rng(2026102406).choice(np.flatnonzero(lower >= 1), genes_per_set, replace=False))
    by_size = np.argsort(-np.array([len(g) for g in groups]))
    rank = np.empty(len(groups), int)
    rank[by_size] = np.arange(len(groups))
    orders, cond, code = relabelled(groups, lambda s_: 0.25 if rank[s_] % 2 else 0.75, np.random.default_rng(2026102405),
                                    len(pick))
    x = expression(jittered(np.take_along_axis(raw[pick], orders, axis=1), pick, 1), d["factors"][orders])
    n1s, n0s, informative = design(cond, code)
    print(f"\n== registration: the combined construction (S1) on TCGA tumours relabelled within 20 sites (sizes "
          f"{sorted(int(np.sum(code == s_)) for s_ in range(code.max() + 1))}), {genes_per_set:,} genes, {shuffles} shuffles",
          flush=True)
    for register in (True, False):
        parts = []
        for s_ in np.flatnonzero(informative):
            idx = np.flatnonzero(code == s_)
            sub = np.ascontiguousarray(x[:, idx])
            lab = np.ascontiguousarray(cond[idx], dtype=np.uint8)
            reg = sub
            if register:
                shift = package_split(sub, lab)[4]
                reg = np.where(lab == 1, (sub + 1) / shift[:, None] - 1, sub)
            o = np.ascontiguousarray(np.argsort(reg, axis=1, kind="stable"), dtype=np.int64)
            parts.append((idx, np.ascontiguousarray(np.take_along_axis(reg, o, axis=1)), o,
                          sqrt(n1s[s_] * n0s[s_] / (n1s[s_] + n0s[s_])), len(idx)))
        for label, least in (("all sites", 0), ("sites of 12 or more", 12)):
            use = [q for q in parts if q[4] >= least]
            score = lambda lab: sum(w * _kernel.score(v, o, "both", np.ascontiguousarray(lab[i], dtype=np.uint8))
                                    for i, v, o, w, _ in use)
            observed = score(cond)
            bar = observed - TIE_RTOL * np.abs(observed)
            hits, done, block = np.zeros(len(x)), 0, 0
            while done < shuffles:
                rows = stream_block(cond, code, block, 1)[:shuffles - done]
                for row in rows:
                    hits += score(row) >= bar
                done += len(rows)
                block += 1
            p = np.where(observed > 0, (1 + hits) / (1 + shuffles), 1.0)
            print(f"  {'registered within each stratum' if register else 'raw values (exact without a fold)'}, {label}: "
                  f"p <= 0.05 {np.mean(p <= .05):.2%}, p <= 0.01 {np.mean(p <= .01):.2%}", flush=True)


def heterogeneous(genes_per_set=3_000):
    """Folds that differ between strata: 1.5x in alternate studies, none in the others. Not the null of
    the two-way construction (one fold for every stratum), which reads it as a subset; the pooled one (S2),
    with a fold per stratum, absorbs it. How often each calls it."""
    from count_model import tumours
    from count_model import rescaled_subset_test as subset_test
    print("\n== heterogeneous: a 1.5x fold in alternate studies and none in the others; batch sd 0.5", flush=True)
    raw = tumours()
    expressed = np.flatnonzero((raw > 0).mean(axis=1) >= 0.5)
    pick = np.sort(np.random.default_rng(2026102404).choice(expressed, genes_per_set, replace=False))
    for name, (sizes, depths) in MODEL_DESIGNS.items():
        folds = np.where(np.arange(len(sizes)) % 2 == 0, 1.5, 1.0)
        counts, factors, cond, code, lowest = model_dataset(sizes, depths, 0.5, folds, False, 1, pick)
        x = expression(jittered(counts, pick, 1), factors)
        print(f"\n  {name}", flush=True)
        for key, p in (("subset, unstratified", subset_test(x, cond, direction="both", stop=25, maximum=9_999, seed=1)[0]),
                       ("subset, pooled (S2)", subset_pooled(x, cond, code)),
                       ("subset, two-way (S4)", subset_pooled(x, cond, code, values_of=twoway_values)),
                       ("subset, one pass (S4b)", subset_pooled(x, cond, code, values_of=onepass_values))):
            print(calibration_line(key, p, lowest), flush=True)


def thinned(genes_per_set=3_000):
    """The adopted subset construction with strata (S4b, thinned) against S4b as first built (scaled), the
    unstratified thinned test and the unstratified package, on the strata's nulls and planted subsets,
    with bins down to 8 reads."""
    from count_model import area_pvalues, tcga, tumours
    from count_model import rescaled_subset_test as subset_test
    edges = (8, 32, 128)
    print("\n== thinned: the subset test with each stratum thinned to its smallest size factor and the higher group "
          "thinned by one fold (S4b, thinned), against S4b scaled, the unstratified thinned test and the package; share "
          "at p <= 0.01 by the smallest expected reads per sample (<8 | 8-32 | 32-128 | >=128), then overall at "
          "0.05 / 0.01 / 0.001", flush=True)

    def compare(counts, factors, cond, code, rows, lowest):
        x = expression(jittered(counts, rows, 1), factors)
        one = np.zeros(len(cond), np.int64)
        for key, p in (("package", subset_test(x, cond, direction="both", stop=25, maximum=9_999, seed=1)[0]),
                       ("thinned, unstratified", subset_thinned(counts, factors, cond, one, rows)),
                       ("S4b, scaled", subset_pooled(x, cond, code, values_of=onepass_values)),
                       ("S4b, thinned", subset_thinned(counts, factors, cond, code, rows))):
            print(calibration_line(key, p, lowest, edges=edges), flush=True)

    raw = tumours()
    expressed = np.flatnonzero((raw > 0).mean(axis=1) >= 0.5)
    pick = np.sort(np.random.default_rng(2026102404).choice(expressed, genes_per_set, replace=False))
    for name, (sizes, depths) in MODEL_DESIGNS.items():
        for batch_sd in (0.0, 0.5):
            for label, factor, subset in (("no fold", 1, False), ("every case 1.5x", 1.5, False),
                                          ("every case 3x", 3.0, False), ("a planted subset", 1, True)):
                counts, factors, cond, code, lowest = model_dataset(sizes, depths, batch_sd, factor, subset, 1, pick)
                print(f"\n  {name}; batch sd {batch_sd:g}; {label}", flush=True)
                compare(counts, factors, cond, code, pick, lowest)
    d = tcga("prostate")
    tumour = np.flatnonzero(d["cond"] == 1)
    raw_all = d["raw"].astype(np.int64)
    lower = (raw_all[:, tumour] / d["factors"][tumour]).mean(axis=1)
    site_all = np.array([s_.split("-")[1] for s_ in d["samples"]])
    site = site_all[tumour]
    names, sizes_ = np.unique(site, return_counts=True)
    groups = [tumour[site == s_] for s_ in names[sizes_ >= 4]]
    by_size = np.argsort(-np.array([len(g) for g in groups]))
    rank = np.empty(len(groups), int)
    rank[by_size] = np.arange(len(groups))
    real = {}
    for s_ in np.unique(site_all):
        here = site_all == s_
        a, b = int(np.sum(d["cond"][here] == 1)), int(np.sum(d["cond"][here] == 0))
        if a and b:
            real[s_] = (a, b)
    five = [tumour[site == s_] for s_ in real]
    shares = [max(1, min(len(g) - 1, round(len(g) * a / (a + b)))) / len(g) for g, (a, b) in zip(five, real.values())]
    pick = np.sort(np.random.default_rng(2026102406).choice(np.flatnonzero(lower >= 1), 4_000, replace=False))
    nulls = [("TCGA tumours within 20 sites", groups, lambda s_: 0.25 if rank[s_] % 2 else 0.75, raw_all, d["factors"], pick, lower),
             ("the real contrast's five sites", five, lambda s_: shares[s_], raw_all, d["factors"], pick, lower)]
    counts_n, study, test_genes, _ = _normals()
    import wade
    tested = np.flatnonzero(test_genes & (counts_n.sum(axis=1) > 0))
    factors_n = wade.fit_normalization(counts_n).size_factors
    low = (counts_n / factors_n).mean(axis=1)
    pick_n = np.sort(np.random.default_rng(2026102407).choice(tested[low[tested] >= 1], 4_000, replace=False))
    nulls.append(("TCGA and GTEx normals by study", [np.flatnonzero(study == "TCGA"), np.flatnonzero(study == "gtex")],
                  lambda s_: (0.75, 0.25)[s_], counts_n, factors_n, pick_n, low))
    for name, grps, share_of, counts_all, factors_all, genes_used, level in nulls:
        for factor in ((1.0, 1.5) if "GTEx" not in name else (1.0,)):
            rng = np.random.default_rng(2026102405)
            orders, cond, code = relabelled(grps, share_of, rng, len(genes_used))
            c = np.take_along_axis(counts_all[genes_used], orders, axis=1).astype(np.int64)
            if factor != 1:
                c = np.random.default_rng([2026102405, 1]).binomial(c, np.where(cond == 1, 1.0, 1 / factor))
            print(f"\n  {name}{f', every pseudo-case {factor:g}x' if factor != 1 else ''}; {len(genes_used):,} genes", flush=True)
            compare(c, factors_all[orders], cond, code, np.asarray(genes_used), level[genes_used])


def fold(genes_per_set=3_000, factor=1.5):
    from count_model import tcga, tumours
    from count_model import rescaled_subset_test as subset_test
    print(f"\n== fold: the subset test's null under a common {factor:g}x fold on every case (its registration "
          f"stressed); as model and sites otherwise", flush=True)
    raw = tumours()
    expressed = np.flatnonzero((raw > 0).mean(axis=1) >= 0.5)
    pick = np.sort(np.random.default_rng(2026102404).choice(expressed, genes_per_set, replace=False))
    for name, (sizes, depths) in MODEL_DESIGNS.items():
        for batch_sd, noise_sd in ((0.0, 0.0), (0.5, 0.0), (0.5, 0.5)):
            counts, factors, cond, code, lowest = model_dataset(sizes, depths, batch_sd, factor, False, 1, pick,
                                                                noise_sd=noise_sd)
            x = expression(jittered(counts, pick, 1), factors)
            noise = f"; extra noise sd {noise_sd:g} in alternate studies" if noise_sd else ""
            print(f"\n  {name}; batch sd {batch_sd:g}{noise}; every case {factor:g}x", flush=True)
            for key, p in (("subset, unstratified", subset_test(x, cond, direction="both", stop=25, maximum=9_999, seed=1)[0]),
                           ("subset, pooled (S2)", subset_pooled(x, cond, code)),
                           ("subset, combined (S1)", subset_combined(x, cond, code))):
                print(calibration_line(key, p, lowest), flush=True)
    d = tcga("prostate")
    tumour = np.flatnonzero(d["cond"] == 1)
    site = np.array([s_.split("-")[1] for s_ in d["samples"][tumour]])
    names, sizes = np.unique(site, return_counts=True)
    usable = names[sizes >= 4]
    groups = [tumour[site == s_] for s_ in usable]
    raw = d["raw"].astype(np.int64)
    lower = (raw[:, tumour] / d["factors"][tumour]).mean(axis=1)
    pick = np.sort(np.random.default_rng(2026102406).choice(np.flatnonzero(lower >= 1), 4_000, replace=False))
    by_size = np.argsort(-np.array([len(g) for g in groups]))
    rank = np.empty(len(groups), int)
    rank[by_size] = np.arange(len(groups))
    real_null("prostate tumours by site", raw, d["factors"], groups, lambda s_: 0.25 if rank[s_] % 2 else 0.75,
              pick, lower[pick], fold=factor, area=False)


def speed():
    import time
    from count_model import tcga
    from wade._means import area_values as package_area_values
    from count_model import rescaled_subset_test as subset_test
    from wade.permutation import stream_block as package_stream
    d = tcga("prostate")
    raw, cond, factors = d["raw"], d["cond"], d["factors"]
    rows = np.arange(len(raw))
    z = package_area_values(raw, rows, factors, 1)
    x = expression(jittered(raw, rows, 1), factors)
    clock = lambda f: (lambda t: (f(), time.perf_counter() - t)[1])(time.perf_counter())
    base = clock(lambda: mean_saddlepoint(z, cond, direction="both"))
    stream = clock(lambda: [package_stream(cond, b, 1) for b in range(50)])
    print(f"\n== speed: TCGA prostate, {len(raw):,} genes x {len(cond)} samples; the package's saddlepoint "
          f"{base:.1f} s, 50 stream blocks {stream:.2f} s", flush=True)
    rng = np.random.default_rng(2026102408)
    for S in (1, 8, 32, 128):
        code = np.zeros(len(cond), np.int64) if S == 1 else rng.integers(0, S, len(cond))
        sp = clock(lambda: saddlepoint_area(z, cond, code))
        st = clock(lambda: [stream_block(cond, code, b, 1) for b in range(50)])
        print(f"  {S:3d} strata: stratified saddlepoint {sp:.1f} s ({sp / base:.2f}x the package's); "
              f"50 stream blocks {st:.2f} s ({st / stream:.2f}x)", flush=True)
    genes = rows[:3_000]
    unstratified = clock(lambda: subset_test(x[genes], cond, direction="both", stop=25, maximum=9_999, seed=1))
    code = rng.integers(0, 8, len(cond))
    onepass = clock(lambda: subset_pooled(x[genes], cond, code, values_of=onepass_values))
    print(f"  the subset test on 3,000 genes, every run finished: unstratified {unstratified:.1f} s; one fold and a level "
          f"per stratum (S4b) with 8 random strata {onepass:.1f} s ({onepass / unstratified:.2f}x)", flush=True)
    build = clock(lambda: onepass_values(x, cond, code))
    plain = clock(lambda: package_split(np.ascontiguousarray(x), np.ascontiguousarray(cond, dtype=np.uint8)))
    print(f"  building the S4b pool for all {len(x):,} genes: {build:.1f} s, against one registration {plain:.1f} s",
          flush=True)
    # The alternating fit (S4) on the same genes: rounds until each gene's fold stops changing.
    labels = np.ascontiguousarray(cond, dtype=np.uint8)
    logs = np.log1p(x)
    beta, level = np.zeros(len(x)), np.zeros((len(x), int(code.max()) + 1))
    rounds, start = np.zeros(len(x), int), time.perf_counter()
    for r in range(50):
        adjusted = logs - beta[:, None] * labels
        for s_ in np.unique(code):
            level[:, s_] = np.median(adjusted[:, code == s_], axis=1)
        new = np.log(package_split(np.ascontiguousarray(np.expm1(logs - level[:, code])), labels)[4])
        moved = new != beta
        rounds[moved] += 1
        beta = new
        if not moved.any():
            break
    print(f"  the alternating fit (S4) for all genes: {time.perf_counter() - start:.1f} s over {r + 1} rounds; rounds per "
          f"gene median {np.median(rounds):.0f}; {int(moved.sum()):,} genes ({moved.mean():.0%}) still moving after 50",
          flush=True)


SECTIONS = dict(exact=exact, accuracy=accuracy, model=model, sites=sites, gtex=gtex, real=real, layout=layout,
                fold=fold, common=common, heterogeneous=heterogeneous, registration=registration, thinned=thinned,
                speed=speed)

if __name__ == "__main__":
    for section in sys.argv[1:] or list(SECTIONS):
        SECTIONS[section]()
