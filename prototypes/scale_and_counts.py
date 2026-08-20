"""Prototype: the subset test on count data at low expression.

Two findings motivated this (docs/method.md, "Scale"):

1. Stage 1's transform is a pure power choice (permutation is exact for any
   statistic), and the linear scale is the most powerful for rare strong
   subsets. Nothing to fix.
2. Stage 2 assumes a global fold change is a *multiplicative* shift of the
   whole distribution — true for continuous data, false for counts at low
   expression (the jitter floor; NB's Poisson component). Measured on the
   current implementation, a genuine 2x NB mean shift is called "not a global
   shift" 99% of the time at 2 counts and 83% of the time at 20 counts with
   1000 v 1000. The characterization's "global reads 1.0" anchor fails there
   too.

Two candidate fixes, each label-free so exchangeability under the null is
preserved, measured here side by side against the current method:

* THIN — a count-native shift correction. NB is closed under binomial thinning
  with the same dispersion, so thinning the higher group's *raw counts* by
  1/f gives an exactly exchangeable null under "a fold change in the count
  model", including zeros. f is the ratio of the two groups' interquartile
  means (robust to subsets below 25%, defined in the zero floor where
  median(R) is not).
* WEIGHT — inverse-variance weights on the log-ratio curve, w(p) =
  1 / Var_perm[R(p)], the node's variance under the permutations of the
  (corrected) matrix. Label-free and model-free. (A first draft used the
  delta-method 1/mu + phi at the pooled count quantile; it is far too generous
  at 1-3 counts, where a 1 against a jittered 0 is a log-ratio of ~7.6, and
  made things worse. Measured, so recorded.) Floor and transition nodes get
  ~0 weight; the bridge becomes S_k - (W_k / W_m) S_m, still identically
  zero for a flat curve, and affected_fraction the weighted participation
  ratio.

What was learned, all measured (tables at the bottom of this file's output
and in docs/method.md "Scale"):

* THIN fixes the test's level. With the observed statistic and the null both
  on the thinned matrix, a genuine 2x NB mean shift fires at ~0.05 at every
  expression level and sample size tried, where the current method fires at
  0.95-1.00 below ~5 counts and 0.72 at 20 counts with 1000 v 1000. Subset
  power is preserved above ~2 counts and lower at the extreme floor, where the
  current method's "power" came with the same artifact that fires on global
  shifts.
* f must be estimated unbiasedly AND robustly. The ratio of interquartile
  means is robust to subsets but biased for NB (the middle of a skewed
  distribution does not scale with its mean: 2.08 for a true 2), which at
  1000 v 1000 leaves ~0.07-0.15 false subsets; the ratio of means is unbiased
  but a 5% subset at 8x moves it to 1.35. The estimator that is both is the
  thinning factor that makes the two groups' interquartile means AGREE
  (``fhat_match``: bisection on log f with thinning inside the loop, common
  random numbers). It is unbiased by construction under an NB fold change and
  blind to subsets below 25%.
* WEIGHT does not work as a permutation-variance weight: the subset's own
  values inflate the null variance of the top nodes, so the signal region is
  down-weighted and the weighted affected_fraction reads ~0 for genuine
  subsets. It is label-free but not signal-free. Recorded; not recommended.
* The characterization below the count floor is genuinely interval-valued:
  on nodes whose pooled count quantile is >= 3, a global 2x at 2 counts reads
  0.86 (vs 0.17 on all nodes) but a 5% subset reads 0.18, because 5% of the
  samples is 18% of the 34% of the distribution that is above the floor.
  Rescaling by the measurable fraction flips which is right. Two numbers are
  needed (the fraction of the measurable distribution, and how much was
  measurable); see docs/method.md.

Further findings from the scenarios the user asked about (``--scenarios``):

* A large subset contaminates f-hat (33% at 2x: 1.24; 60%: 1.55) and it does
  not matter for detection — such genes are mixtures, not global shifts, and
  the thinned test calls them so (power 0.97, 0.97; the 80%-at-2x / 20%-unchanged
  gene 0.64 against 0.28 today). What matters is that f-hat be UNBIASED on a
  true global shift: forcing it 30% off on a genuine 2x gives 0.26-0.28 false
  subsets. The error is anti-conservative, not conservative (I had said the
  opposite; measured, retracted).
* An estimator-free profile — p = max over candidate f of the thinned p —
  is valid (0.00-0.01 on null and global) but too conservative to be the
  default (0.24 vs 0.64 on the 80/20 gene). ``p_profile`` is kept as the
  principled fallback.
* A SINGLE ZERO can halve affected_fraction at any n: at 20,000 v 20,000 one
  control zero (jittered to 0.005) against a case minimum of 5 is R ~ 10 at
  one node, R^4 = 10^4 against 20,000 nodes of 1, and the global 2x reads 0.39
  instead of 0.97. The tie-breaking jitter is being read as a measurement.
  Fix: a PSEUDOCOUNT OF ONE COUNT before the log for the R curve (a zero
  means "less than one"; its log is bounded). A half-count floor
  (max(x, 0.5)) was the first candidate; the user preferred log(x+1), and
  measured head to head they behave the same within noise, log(x+1) is
  slightly better at 5 counts (global 2x reads 0.86 vs 0.74) and is the
  convention, so log(x+1) it is (``log_floor`` below keeps both for the
  record). Inert above ~20 counts; fixes the single-zero case completely;
  moves the global anchor at 2 counts from 0.16 to 0.70; raises thinned
  stage-2 power at the floor (15% at 8x, mean 0.5: 0.32 -> 0.84) with level
  intact, because those spikes were noise in the bridge too. Under thinning
  any deterministic transform applied to observed and permuted data alike
  leaves the test exact. With a pseudocount the jitter is irrelevant to stage
  2 (same numbers with and without). Poisson(1) noise in place of the jitter
  was measured: costs power at the floor (0.84 -> 0.49), removes no zeros.
* Bootstrap CIs (resample within each group) for affected_fraction, direction
  and log2_fc are cheap and behave: 5% at 8x, 200 v 200, [0.03, 0.09]; 1000 v
  1000, [0.04, 0.07]; coverage ~1. They cover the ESTIMAND, which is biased
  for a blurred step (33% at 2x reads ~0.5, because a 2x step against 30%
  noise is a ramp) and is a single-level summary by construction (global 2x
  plus 5% at 8x reads 0.43). Both belong in the docs.

Two procedural rules that the first draft got wrong, both measured:

* Thinning must be applied to the OBSERVED statistic as well as the null. The
  division trick could correct only the null because the bridge is exactly
  invariant to division; it is not invariant to thinning, which is the point.
* The delta-method weight is not usable below a few counts; use the
  permutation variance.

Run:  conda activate wade && python prototypes/scale_and_counts.py              # the variant tables
      conda activate wade && python prototypes/scale_and_counts.py --scenarios  # large subsets, wrong f, floor, profile
Each takes a few minutes; they print the tables docs/method.md quotes.
"""

from __future__ import annotations

import sys
import time

import numpy as np

import wade
from wade.permutation import draw_perms, subset_null_backend
from wade.pvalues import perm_pvalues
from wade.quantiles import probability_grid, type7_quantiles
from wade.stats import split_groups
from wade.subset import affected_fraction, bridge, direction, log_ratio_curve, subset_test

PHI = 0.1          # NB dispersion of the simulated data (biological CV ~ 0.32)
B = 500
ALPHA = 0.05
JITTER = 0.01


# ---------------------------------------------------------------------------
# Data


def nb(rng, mu, size, phi=PHI):
    return rng.poisson(rng.gamma(1 / phi, phi * mu, size=size)).astype(float)


def scenario(rng, mu, alt, fold, n1, n0, reps, frac=0.05):
    """reps genes x (n1 + n0) raw counts; cases carry `alt`."""
    case = nb(rng, mu, (reps, n1))
    ctrl = nb(rng, mu, (reps, n0))
    if alt == "global":
        case = nb(rng, mu * fold, (reps, n1))
    elif alt == "subset":
        k = round(frac * n1)
        for i in range(reps):
            case[i, rng.choice(n1, k, replace=False)] = nb(rng, mu * fold, k)
    return np.c_[case, ctrl]


# ---------------------------------------------------------------------------
# The pieces


def midmean_ratio(x, cond):
    """Ratio of the groups' interquartile means, per gene. The fold change a
    pure global shift would have, estimated where a subset of < 25% cannot
    reach and where zeros do not break it."""
    i1, i0 = split_groups(cond)
    p = np.linspace(0.25, 0.75, 51)
    m1 = type7_quantiles(x[:, i1], p).mean(axis=1)
    m0 = type7_quantiles(x[:, i0], p).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = m1 / m0
    bad = ~np.isfinite(f) | (f <= 0)
    if bad.any():                                   # degenerate middle: fall back to means
        f[bad] = x[bad][:, i1].mean(axis=1) / x[bad][:, i0].mean(axis=1)
    return f


P_MID = np.linspace(0.25, 0.75, 51)


def _midmean_rows(a):
    return type7_quantiles(np.asarray(a, dtype=float), P_MID).mean(axis=1)


def fhat_match(counts, cond, seed=0, iters=14, max_log_f=np.log(16.0)):
    """The thinning factor that equalizes the groups' interquartile means.

    Bisection on log f with thinning inside the loop and common random
    numbers (the same seed at every evaluation, so the objective is monotone
    in f). Unbiased under an NB fold change by construction — the test is
    "after thinning, do the middles agree?", which does not assume the middle
    scales with the mean — and blind to subsets below 25%. Thins whichever
    group is higher; never scales anything up."""
    i1, i0 = split_groups(cond)
    c1 = counts[:, i1].astype(np.int64)
    c0 = counts[:, i0].astype(np.int64)
    up = _midmean_rows(c1) >= _midmean_rows(c0)
    g = counts.shape[0]
    lo = np.zeros(g)
    hi = np.full(g, max_log_f)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        q_thin = np.exp(-mid)[:, None]
        r = np.random.default_rng(seed)
        t1 = np.where(up[:, None], r.binomial(c1, q_thin), c1)
        r = np.random.default_rng(seed)
        t0 = np.where(up[:, None], c0, r.binomial(c0, q_thin))
        d = _midmean_rows(t1) - _midmean_rows(t0)
        thinned_too_little = np.where(up, d > 0, d < 0)
        lo = np.where(thinned_too_little, mid, lo)
        hi = np.where(thinned_too_little, hi, mid)
    f = np.exp(0.5 * (lo + hi))
    return np.where(up, f, 1.0 / f)


def thin(counts, cond, f, rng):
    """Binomial-thin the higher group's raw counts by the fitted fold change,
    gene by gene. Drawn once, before any permutation, like the jitter."""
    i1, i0 = split_groups(cond)
    out = counts.copy()
    for g in range(counts.shape[0]):
        if f[g] >= 1.0:
            out[g, i1] = rng.binomial(counts[g, i1].astype(np.int64), 1.0 / f[g])
        else:
            out[g, i0] = rng.binomial(counts[g, i0].astype(np.int64), f[g])
    return out


def node_weights(counts, q, phi=0.1):
    """w(p) = Q_pool(p) / (1 + phi Q_pool(p)): inverse delta-method variance of
    a log count, evaluated at the pooled (label-free) quantile."""
    qp = type7_quantiles(counts, q)
    return qp / (1.0 + phi * qp)


def bridge_w(r, w):
    s = np.cumsum(w * r, axis=1)
    W = np.cumsum(w, axis=1)
    return s - (W / W[:, -1:]) * s[:, -1:]


def affected_fraction_w(r, w):
    s2 = (w * r * r).sum(axis=1)
    s4 = (w * r ** 4).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(s4 > 0, s2 * s2 / (w.sum(axis=1) * s4), np.nan)


def weighted_subset_null(xs, b_obs, perms, q, w, alternative="two-sided"):
    """_subset_null_numpy with node weights. Slow, readable, prototype.
    ``w`` may be None (unweighted)."""
    g, m = b_obs.shape
    n_perms = perms.shape[0]
    s1 = np.zeros((g, m)); s2 = np.zeros((g, m))
    bridges = np.empty((n_perms, g, m))
    for b in range(n_perms):
        i1, i0 = split_groups(perms[b])
        r = log_ratio_curve(type7_quantiles(xs[:, i1], q), type7_quantiles(xs[:, i0], q))
        bb = bridge(r) if w is None else bridge_w(r, w)
        bridges[b] = bb
        s1 += bb; s2 += bb * bb
    mu = s1 / n_perms
    sd = np.sqrt(np.maximum(s2 / n_perms - mu * mu, 0.0))
    usable = sd > 0; usable[:, -1] = False
    safe = np.where(usable, sd, np.inf)
    red = (lambda z: z) if alternative == "greater" else ((lambda z: -z) if alternative == "less" else np.abs)
    null = np.empty((g, n_perms))
    for b in range(n_perms):
        null[:, b] = red((bridges[b] - mu) / safe).max(axis=1)
    stat = red((b_obs - mu) / safe).max(axis=1)
    return stat, null


def empirical_weights(xs, perms, q, n_use=200):
    """w(p) = 1 / Var_perm[R(p)]: how informative each node is, read off the
    permutation distribution of the log-ratio curve itself. Label-free, no
    noise model, no formula. Nodes whose R is constant under permutation
    carry nothing and get weight 0."""
    g = xs.shape[0]; m = q.size
    s1 = np.zeros((g, m)); s2 = np.zeros((g, m))
    for b in range(min(n_use, perms.shape[0])):
        i1, i0 = split_groups(perms[b])
        r = log_ratio_curve(type7_quantiles(xs[:, i1], q), type7_quantiles(xs[:, i0], q))
        s1 += r; s2 += r * r
    nb_ = min(n_use, perms.shape[0])
    var = np.maximum(s2 / nb_ - (s1 / nb_) ** 2, 0.0)
    with np.errstate(divide="ignore"):
        w = np.where(var > 0, 1.0 / var, 0.0)
    return w / np.maximum(w.mean(axis=1, keepdims=True), 1e-300)


def curve(x, cond, q):
    i1, i0 = split_groups(cond)
    return log_ratio_curve(type7_quantiles(x[:, i1], q), type7_quantiles(x[:, i0], q))


# ---------------------------------------------------------------------------
# The variants, each returning (p_subset, affected_fraction)
#
# current       : division-corrected null, unweighted (what the package does)
# thin(midmean) : observed AND null on the thinned matrix, f = interquartile-mean ratio
# thin          : same, f = fhat_match (recommended)
# div+w     : division-corrected null, empirical weights on bridge and pi
# thin+w    : thinned observed and null, empirical weights; pi on the
#             unthinned curve with the same weights (the characterization
#             must still read a global shift as 1.0)


def run_variants(counts, cond, perms, rng, alternative="two-sided"):
    jitter = rng.uniform(0, JITTER, counts.shape)
    x = counts + jitter                                       # "normalized": unit libraries
    i1, i0 = split_groups(cond)
    q = probability_grid(min(i1.size, i0.size))
    r_obs = curve(x, cond, q)
    out = {}

    sub = subset_test(x, cond, perms, alternative=alternative)
    p, _, _ = perm_pvalues(sub.statistic, sub.null, alternative="greater")
    out["current"] = (p, sub.affected_fraction)

    f_mid = midmean_ratio(x, cond)
    xs_mid = thin(counts, cond, f_mid, rng) + jitter
    stat, null, _, _, _ = subset_null_backend(xs_mid, bridge(curve(xs_mid, cond, q)), perms, q, alternative=alternative)
    p, _, _ = perm_pvalues(stat, null, alternative="greater")
    out["thin(midmean)"] = (p, sub.affected_fraction)

    f = fhat_match(counts, cond)
    xs_thin = thin(counts, cond, f, rng) + jitter
    r_thin = curve(xs_thin, cond, q)
    stat, null, _, _, _ = subset_null_backend(xs_thin, bridge(r_thin), perms, q, alternative=alternative)
    p, _, _ = perm_pvalues(stat, null, alternative="greater")
    out["thin"] = (p, sub.affected_fraction)

    shift = 2.0 ** np.median(r_obs, axis=1)
    xs_div = x.copy(); xs_div[:, i1] /= shift[:, None]
    w_div = empirical_weights(xs_div, perms, q)
    stat, null = weighted_subset_null(xs_div, bridge_w(r_obs, w_div), perms, q, w_div, alternative)
    p, _, _ = perm_pvalues(stat, null, alternative="greater")
    out["div+w"] = (p, affected_fraction_w(r_obs, w_div))

    w_thin = empirical_weights(xs_thin, perms, q)
    stat, null = weighted_subset_null(xs_thin, bridge_w(r_thin, w_thin), perms, q, w_thin, alternative)
    p, _, _ = perm_pvalues(stat, null, alternative="greater")
    out["thin+w"] = (p, affected_fraction_w(r_obs, w_thin))
    return out


def log_floor(x, c=0.5, mode="pseudocount"):
    """The bottom of the log scale. ``mode="pseudocount"``: x + c (the adopted
    rule, c = 1 count). ``mode="floor"``: max(x, c) (the first candidate, kept
    for the record). A count of zero means 'less than one'; against the
    tie-breaking jitter it would otherwise read as a log-ratio of 7-10."""
    if not c:
        return x
    return x + c if mode == "pseudocount" else np.maximum(x, c)


def bootstrap_ci(case, ctrl, nboot=300, seed=0, c=None):
    """Percentile 95% CIs for (affected_fraction, direction, log2_fc), resampling
    samples within each group. Returns (lo, hi), each (3, genes)."""
    r = np.random.default_rng(seed)
    n1, n0 = case.shape[1], ctrl.shape[1]
    q = probability_grid(min(n1, n0))
    out = np.empty((3, nboot, case.shape[0]))
    for b in range(nboot):
        a = case[:, r.integers(0, n1, n1)]
        d = ctrl[:, r.integers(0, n0, n0)]
        rr = log_ratio_curve(type7_quantiles(log_floor(a, c), q), type7_quantiles(log_floor(d, c), q))
        out[0, b] = affected_fraction(rr)
        out[1, b] = direction(rr)
        out[2, b] = np.log2(a.mean(1) / d.mean(1))
    lo, hi = np.percentile(out, [2.5, 97.5], axis=1)
    return lo, hi


def p_profile(counts, cond, perms, q, rng, jitter, F=7):
    """Estimator-free: p = max over candidate f of the thinned test's p. The
    candidates span the range of quantile ratios over the middle half (where
    both groups are >= 1 count), widened by 10%. Valid by construction; too
    conservative to be the default (measured)."""
    x = counts + jitter
    i1, i0 = split_groups(cond)
    pm = np.linspace(0.25, 0.75, 26)
    Q1 = type7_quantiles(x[:, i1], pm); Q0 = type7_quantiles(x[:, i0], pm)
    ok = (Q1 >= 1) & (Q0 >= 1)
    ratio = np.where(ok, Q1 / np.maximum(Q0, 1e-9), np.nan)
    with np.errstate(all="ignore"):
        lo = np.nanmin(ratio, axis=1) / 1.1; hi = np.nanmax(ratio, axis=1) * 1.1
    mr = x[:, i1].mean(1) / x[:, i0].mean(1)
    bad = ~np.isfinite(lo) | ~np.isfinite(hi)
    lo[bad] = mr[bad] / 1.5; hi[bad] = mr[bad] * 1.5
    grid = np.exp(np.linspace(0, 1, F)[None, :] * (np.log(hi) - np.log(lo))[:, None] + np.log(lo)[:, None])
    pmax = np.zeros(counts.shape[0])
    for j in range(F):
        xs = thin(counts, cond, grid[:, j], rng) + jitter
        stat, null, *_ = subset_null_backend(xs, bridge(curve(xs, cond, q)), perms, q, alternative="two-sided")
        pmax = np.maximum(pmax, perm_pvalues(stat, null, alternative="greater")[0])
    return pmax


def scenarios(n1=200, n0=200, reps=100, mus=(20, 2), seed=3, floor=1.0):
    """The large-subset scenarios, the wrong-f check, the profile, the floor."""
    from wade.permutation import null_statistics
    from wade.subset import affected_fraction as aff, direction as dirn
    rng = np.random.default_rng(seed)
    cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
    perms = draw_perms(cond, B, seed=1)
    q = probability_grid(min(n1, n0))
    i1, i0 = split_groups(cond)

    def make(mu, kind):
        case = nb(rng, mu, (reps, n1)); ctrl = nb(rng, mu, (reps, n0))
        if kind == "global x2":
            case = nb(rng, 2 * mu, (reps, n1))
        elif kind == "80% x2, 20% same":
            k = round(.8 * n1)
            for i in range(reps): case[i, rng.choice(n1, k, replace=False)] = nb(rng, 2 * mu, k)
        elif kind == "global x2 + 5% x8":
            case = nb(rng, 2 * mu, (reps, n1))
            for i in range(reps): case[i, rng.choice(n1, round(.05 * n1), replace=False)] = nb(rng, 16 * mu, round(.05 * n1))
        elif kind != "null":
            frac, fold = {"5% x8": (.05, 8), "33% x2": (.33, 2), "60% x2": (.6, 2)}[kind]
            k = round(frac * n1)
            for i in range(reps): case[i, rng.choice(n1, k, replace=False)] = nb(rng, mu * fold, k)
        return np.c_[case, ctrl]

    def rate_thin(counts, f, jit, c=None):
        xs = thin(counts, cond, f, rng) + jit
        xf = log_floor(xs, c)
        stat, null, *_ = subset_null_backend(xf, bridge(curve(xf, cond, q)), perms, q, alternative="two-sided")
        return float(np.mean(perm_pvalues(stat, null, alternative="greater")[0] < ALPHA))

    KINDS = ["null", "global x2", "5% x8", "33% x2", "60% x2", "80% x2, 20% same", "global x2 + 5% x8"]
    for mu in mus:
        print(f"\n{n1} v {n0}, mean count {mu}, {reps} genes per row. Rates at alpha = {ALPHA}.")
        print(f"{'gene':>18} | {'p_mean':>6} | {'cur':>5} {'thin':>5} {'thin+floor':>10} {'profile':>7} | {'f mean':>6} {'f match':>7} | {'aff':>5} {'aff+floor':>9} {'dir':>5} | wrong f on global x1.3 /1.3")
        for kind in KINDS:
            counts = make(mu, kind); jit = rng.uniform(0, JITTER, counts.shape); x = counts + jit
            p1 = perm_pvalues(wade.wade_stats(x, cond).mean_shift, null_statistics(x, perms), alternative="two-sided")[0]
            sub = subset_test(x, cond, perms)
            pc = perm_pvalues(sub.statistic, sub.null, alternative="greater")[0]
            f = fhat_match(counts, cond)
            pt = rate_thin(counts, f, jit); ptf = rate_thin(counts, f, jit, floor)
            pp = p_profile(counts, cond, perms, q, rng, jit)
            fm = x[:, i1].mean(1) / x[:, i0].mean(1)
            rf = log_ratio_curve(type7_quantiles(log_floor(x, floor)[:, i1], q), type7_quantiles(log_floor(x, floor)[:, i0], q))
            extra = ""
            if kind == "global x2":
                extra = f"{rate_thin(counts, f * 1.3, jit):>12.2f} {rate_thin(counts, f / 1.3, jit):>5.2f}"
            print(f"{kind:>18} | {np.mean(p1 < ALPHA):>6.2f} | {np.mean(pc < ALPHA):>5.2f} {pt:>5.2f} {ptf:>10.2f} {np.mean(pp < ALPHA):>7.2f} | "
                  f"{np.median(fm):>6.2f} {np.median(f):>7.2f} | {np.median(sub.affected_fraction):>5.2f} {np.median(aff(rf)):>9.2f} {np.median(sub.direction):>+5.2f} |{extra}", flush=True)


VARIANTS = ("current", "thin(midmean)", "thin", "div+w", "thin+w")


def table(n1, n0, reps, mus, alts, seed=0, alternative="two-sided"):
    rng = np.random.default_rng(seed)
    cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
    perms = draw_perms(cond, B, seed=1)
    print(f"\n{n1} v {n0}, {reps} genes per cell, B = {B}, alternative = {alternative}")
    print(f"{'mu':>5} {'gene':>14} | " + " ".join(f"{v:>12}" for v in VARIANTS)
          + "   ||  median affected_fraction: " + " ".join(f"{v:>11}" for v in VARIANTS))
    rows = {}
    for mu in mus:
        for alt, fold, frac in alts:
            t = time.time()
            counts = scenario(rng, mu, alt, fold, n1, n0, reps, frac)
            res = run_variants(counts, cond, perms, rng, alternative)
            rates = [float((res[v][0] < ALPHA).mean()) for v in VARIANTS]
            affs = [float(np.nanmedian(res[v][1])) for v in VARIANTS]
            label = alt if alt != "subset" else f"subset {int(frac*100)}%x{fold:g}"
            if alt == "global":
                label = f"global x{fold:g}"
            rows[(mu, label)] = (rates, affs)
            print(f"{mu:>5} {label:>14} | " + " ".join(f"{r:>12.2f}" for r in rates)
                  + "   ||  " + " ".join(f"{a:>11.2f}" for a in affs)
                  + f"   ({time.time()-t:.0f}s)", flush=True)
    return rows


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    if "--scenarios" in sys.argv:
        scenarios(reps=40 if quick else 100)
        sys.exit(0)
    reps = 40 if quick else 100
    ALTS = [("null", 1, 0.05), ("global", 2.0, 0.05), ("subset", 8.0, 0.05), ("subset", 8.0, 0.15)]
    print("Rate at which the subset test fires at alpha = 0.05: LEVEL on null and global, POWER on subsets.")
    table(200, 200, reps, (0.5, 2, 20, 500), ALTS)
    print("\nLarge n: the regime WADE is for. (global x2 only — the false-subset rate that grew with n)")
    table(1000, 1000, 20 if quick else 40, (2, 20, 100), [("global", 2.0, 0.05), ("subset", 8.0, 0.05)])
    print("\nOne-sided 'greater' (the cancer-outlier direction), 200 v 200:")
    table(200, 200, reps, (0.5, 2, 20), [("global", 2.0, 0.05), ("subset", 8.0, 0.05)], alternative="greater")
