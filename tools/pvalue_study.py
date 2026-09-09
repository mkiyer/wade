"""How far into the tail can each p-value estimator be trusted? (`docs/pvalue-review.md`)

Pick a design small enough that **brute force is the ground truth**, then hold
every candidate to it at true p-values spanning 1e-3 to 1e-8. A study, not part
of WADE.

    python tools/pvalue_study.py calibrate 40 40 1     # find a p ladder
    python tools/pvalue_study.py truth     40 40 1 1e9
    python tools/pvalue_study.py compare   40 40 1
    python tools/pvalue_study.py stress                # every geometry x stage

The trailing `1` or `2` is the stage. Geometry is `n1 n0`, and **`n1 + n0` is
held at 80** across the sweep, so a difference between geometries is a
difference in balance rather than in permutation cost.

**Four things this had to get right**, each measured rather than assumed and
recorded because none was obvious in advance.

*Memory, not speed, stops brute force.* `draw_perms` materializes a `(B, n)`
label matrix and the nulls are `(genes, B)`: at B = 1e8 that is 64 GB and
51 GB. Permutations are streamed and only exceedance counts kept, so B is
unbounded and the footprint constant.

*The vectorized draw is bitwise identical to the shipped one.*
`rng.permuted(broadcast_to(cond, (B, n)), axis=1)` reproduces a loop of
`rng.permutation(cond)` exactly — arrays and final RNG state, over 108
configurations of group size, B and seed — and is 3x faster. Used here and
deliberately **not** in `wade.draw_perms`: it is undocumented numpy behaviour,
and on a real run drawing is already a rounding error against the kernel.

*Stage 2's statistic is not a fixed function of a label assignment.* It is
`max_k (B_k - mu_k)/sigma_k` with the moments estimated from the same
permutations the tail is read from, so the observed value moves with the
permutation sample: 3% at B = 2,000, 0.09% at 500,000. Brute force therefore
streams blocks of 500,000, each carrying its own observed value; and anything
that samples a *conditioned* distribution — multilevel splitting — must freeze
the moments first, or it estimates the tail of a different statistic.

*Stage 2's ladder is `k`, not the effect size.* At a fixed `k = 8` of 40,
subsets of 2x through 25x all returned p between 2.5e-4 and 5e-4: the gene had
reached the combinatorial floor for that `k`, and no effect size buys depth a
design has not got.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np

import wade
from wade.permutation import null_statistics, split_groups, subset_null_backend
from wade.pvalues import gpd_tail_p, perm_pvalues
from wade.stats import wade_stats

G_PER_RUNG = 8
BLOCK = 500_000
PHI = 0.1


@dataclass(frozen=True)
class Design:
    """A geometry, and what every routine needs from it."""

    n1: int
    n0: int

    @property
    def n(self) -> int:
        return self.n1 + self.n0

    @property
    def cond(self) -> np.ndarray:
        return np.r_[np.ones(self.n1, int), np.zeros(self.n0, int)]

    def perms(self, rng, b) -> np.ndarray:
        return rng.permuted(np.broadcast_to(self.cond, (b, self.n)), axis=1)

    def __str__(self) -> str:
        return f"{self.n1}v{self.n0}"


# ---------------------------------------------------------------------------
# cohorts


def cohort1(d: Design, rungs, seed=0):
    """Stage 1: log-normal expression, a global shift per rung.

    Continuous and tie-free, the cleanest setting in which to measure the tail
    of a permutation null. Counts belong to stage 2, whose null is built by
    thinning reads.
    """
    rng = np.random.default_rng(seed)
    rows, truth = [], []
    for mult in rungs:
        for _ in range(G_PER_RUNG):
            rows.append(np.r_[rng.lognormal(3.0, 0.6, d.n1) * mult,
                              rng.lognormal(3.0, 0.6, d.n0)])
            truth.append(mult)
    return np.array(rows), np.array(truth, dtype=float)


def cohort2(d: Design, rungs, fold=8.0, seed=0):
    """Stage 2: NB counts with `k` affected cases, `k` being the rung."""
    rng = np.random.default_rng(seed)
    nb = lambda mu, size: rng.poisson(rng.gamma(1 / PHI, PHI * mu, size=size)).astype(float)
    rows, truth = [], []
    for k in rungs:
        for _ in range(G_PER_RUNG):
            mu = 10 ** rng.uniform(1.6, 2.4)
            case, ctrl = nb(mu, d.n1), nb(mu, d.n0)
            case[:int(k)] = nb(mu * fold, int(k))
            rows.append(np.r_[case, ctrl])
            truth.append(k)
    return np.array(rows), np.array(truth, dtype=float)


# ---------------------------------------------------------------------------
# brute force


def empirical_p(nexc, n_perms):
    return (1.0 + nexc) / (n_perms + 1.0)


def observed1(d: Design, x):
    return wade_stats(x, d.cond).mean_shift


def brute1(d: Design, x, obs, n_perms, block=BLOCK, seed=7, report=None):
    rng = np.random.default_rng(seed)
    nexc = np.zeros(x.shape[0], dtype=np.int64)
    done, t0 = 0, time.perf_counter()
    while done < n_perms:
        b = int(min(block, n_perms - done))
        nexc += (null_statistics(x, d.perms(rng, b)) >= obs[:, None]).sum(axis=1)
        done += b
        if report and (done // block) % report == 0:
            el = time.perf_counter() - t0
            print(f"    {done:>13,}/{n_perms:,.0f}  {el/60:5.1f} min  "
                  f"eta {el*(n_perms/done-1)/60:5.1f}", flush=True)
    return nexc, done


def stage2_block(d: Design, counts, perms):
    """One self-consistent block: its own observed value and its own null."""
    r = wade.wade(counts, np.ones(counts.shape[0]), d.cond, lib_sizes=np.ones(d.n),
                  perms=perms, nperms=len(perms), seed=1, alternative="greater",
                  keep_null=True)
    return r.subset.statistic, (r.subset.null >= r.subset.statistic[:, None]).sum(axis=1)


def brute2(d: Design, counts, n_perms, block=BLOCK, seed=7, report=None):
    rng = np.random.default_rng(seed)
    nexc = np.zeros(counts.shape[0], dtype=np.int64)
    done, t0 = 0, time.perf_counter()
    while done < n_perms:
        b = int(min(block, n_perms - done))
        nexc += stage2_block(d, counts, d.perms(rng, b))[1]
        done += b
        if report and (done // block) % report == 0:
            el = time.perf_counter() - t0
            print(f"    {done:>13,}/{n_perms:,.0f}  {el/60:5.1f} min  "
                  f"eta {el*(n_perms/done-1)/60:5.1f}", flush=True)
    return nexc, done


# ---------------------------------------------------------------------------
# candidate: the GPD fitted by maximum likelihood rather than by moments


def gpd_mle_p(obs, null, n_tail=250):
    """`gpd_tail_p`'s refinement with the shape fitted by ML.

    Wanted because moments are poorly behaved for
    `xi > 0.5`. §4.5 then found the opposite failure — a *negative* fitted
    shape on every deep-tail stage-1 gene, sending all of them down the
    `xi <= 0` exponential branch, which is a far heavier tail than a bounded
    permutation null. Everything else is held identical (threshold, the strict
    exceedance inequality, the `n_tail/B` rescaling, the floor) so a
    difference is attributable to the estimator alone.

    Where the fitted GPD has negative shape its support ends at `-sigma/xi`
    and `genpareto.sf` returns 0 past it — the machine-epsilon collapse
    `gpd_tail_p` avoids by substituting the exponential. That is left in and
    floored here, because the question is which of the two failures is
    smaller.
    """
    from scipy.stats import genpareto

    null = np.asarray(null, dtype=np.float64)
    n_perms = null.shape[-1]
    n_tail_used = int(min(n_tail, n_perms // 2))
    floor = 1.0 / (n_perms * n_tail_used)
    out = np.empty(null.shape[0])
    for i in range(null.shape[0]):
        s = np.sort(null[i])[::-1]
        thr = float(s[n_tail_used])
        exc = s[s > thr] - thr
        emp = float((1 + int(np.sum(null[i] >= obs[i]))) / (n_perms + 1))
        if exc.size < 10 or obs[i] <= thr:
            out[i] = emp
            continue
        try:
            xi, _, sigma = genpareto.fit(exc, floc=0.0)
        except Exception:
            out[i] = emp
            continue
        if not np.isfinite(sigma) or sigma <= 0:
            out[i] = emp
            continue
        tail = float(genpareto.sf(obs[i] - thr, xi, loc=0.0, scale=sigma))
        out[i] = max((n_tail_used / n_perms) * tail, floor)
    return out


# ---------------------------------------------------------------------------
# candidate: the double saddlepoint (linear statistics only)


def saddlepoint_applies(d: Design) -> bool:
    """`mean_shift` is the mean difference only on a balanced design.

    `T = A(1/n1 + 1/n0) - V/n0` for **any** group sizes, so balance is not a
    saddlepoint requirement. What is required is that `mean_shift` *be* that
    quantity: unbalanced it is the grid quadrature over `m = min(n1, n0)`
    nodes, a weighted sum of the larger group's order statistics — an
    L-statistic with no reduction to a subset sum (`docs/method.md` §2).
    """
    return d.n1 == d.n0


def _linear_target(d: Design, x, obs):
    V = x.sum(axis=1)
    return x, (obs + V / d.n0) / (1.0 / d.n1 + 1.0 / d.n0)


def _K(v, s, t):
    z = s * v + t
    p = 1.0 / (1.0 + np.exp(-z))
    q = p * (1.0 - p)
    return np.logaddexp(0.0, z).sum(), (p @ v, p.sum()), (q @ (v * v), q @ v, q.sum())


def _solve_t(v, s, target, iters=90):
    lo, hi = -60.0, 60.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _K(v, s, mid)[1][1] < target else (lo, mid)
    return 0.5 * (lo + hi)


def saddlepoint_p(d: Design, x, obs):
    """Skovgaard's double-saddlepoint tail probability. No permutations."""
    from scipy.stats import norm

    vals, a_all = _linear_target(d, x, obs)
    out = np.empty(len(obs))
    for i in range(len(obs)):
        v = np.ascontiguousarray(vals[i], dtype=float)
        a = float(a_all[i])
        t0 = float(np.log(d.n1 / d.n0))
        K0, _, (_, _, Ktt0) = _K(v, 0.0, t0)
        lim = 50.0 / max(np.abs(v).max(), 1e-12)
        lo, hi = -lim, lim
        for _ in range(90):
            s = 0.5 * (lo + hi)
            lo, hi = (s, hi) if _K(v, s, _solve_t(v, s, d.n1))[1][0] < a else (lo, s)
        s_hat = 0.5 * (lo + hi)
        t_hat = _solve_t(v, s_hat, d.n1)
        K1, _, (Kss, Kst, Ktt) = _K(v, s_hat, t_hat)
        w = np.sign(s_hat) * np.sqrt(max(2.0 * ((K0 - t0 * d.n1)
                                                - (K1 - s_hat * a - t_hat * d.n1)), 0.0))
        u = s_hat * np.sqrt(max(Kss * Ktt - Kst * Kst, 0.0) / Ktt0)
        out[i] = 0.5 if abs(w) < 1e-8 or abs(u) < 1e-12 else float(
            norm.sf(w) - norm.pdf(w) * (1.0 / w - 1.0 / u))
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# candidate: fgsea's adaptive multilevel splitting


def _log_p(k, rem, n_sample):
    """fgsea's Beta correction. Each level is a *sample* median, whose
    exceedance probability is a Beta order statistic, so the rounds accumulate
    `E[log U] = psi(a) - psi(a+b)` rather than `log E[U]` — which is what makes
    `log p` approximately unbiased instead of systematically optimistic."""
    from scipy.special import digamma

    half = n_sample // 2
    return (k * (digamma(half) - digamma(n_sample + 1))
            + digamma(rem + 1) - digamma(n_sample + 1))


def multilevel1(d: Design, x, obs, n_sample=1000, sweeps=6, seed=5, max_rounds=4000):
    """Stage 1 multilevel splitting, on the statistic WADE actually computes.

    **The population is scored with `null_statistics`, the same kernel brute
    force uses**, and not with a subset-sum shortcut. The shortcut was the
    first version and it was wrong off balance: `mean_shift` is the difference
    of group means only when `n1 == n0`, and otherwise it is the grid
    quadrature over `m = min(n1, n0)` nodes — a weighted sum of the larger
    group's order statistics. Scoring a subset sum while thresholding at a
    quadrature value estimates the tail of a different statistic, and the
    damage was not subtle: medians of 0.00 to 0.13 at 70 v 10, with the sign
    of the error following which group was larger. It looked like multilevel
    failing on unbalanced designs. It was the harness.

    The cost of doing it properly is that **the O(1) update is gone**, which
    was the whole reason stage 1 was milliseconds and stage 2 seconds. A swap
    changes one value in each group, but the quadrature re-reads order
    statistics, so there is nothing to update incrementally. What rescues it is
    that the population can be scored in one batched call per sweep rather
    than one per element: ~`sweeps * rounds` kernel calls per gene, each on a
    `(1, n)` matrix with `n_sample` permutations.
    """
    rng = np.random.default_rng(seed)
    half = n_sample // 2
    rows = np.arange(n_sample)
    out, rounds = np.empty(len(obs)), np.empty(len(obs), dtype=int)

    for i in range(len(obs)):
        xi = np.ascontiguousarray(x[i:i + 1])
        score = lambda pop: null_statistics(xi, pop)[0]
        pop = d.perms(rng, n_sample).copy()
        s = score(pop)
        a, k = float(obs[i]), 0
        while k < max_rounds:
            srt = np.argsort(s)
            level = float(s[srt[half - 1]])
            if level >= a:
                break
            pick = np.r_[srt[half:], srt[half:]][:n_sample]
            pop, s = pop[pick].copy(), s[pick].copy()
            for _ in range(sweeps):
                ones = np.argsort(rng.random((n_sample, d.n)) - pop, axis=1)[:, 0]
                zeros = np.argsort(rng.random((n_sample, d.n)) + pop, axis=1)[:, 0]
                prop = pop.copy()
                prop[rows, ones], prop[rows, zeros] = 0, 1
                sp = score(prop)
                ok = sp >= level
                pop[ok], s[ok] = prop[ok], sp[ok]
            k += 1
        out[i] = float(np.exp(_log_p(k, int((s >= a).sum()), n_sample)))
        rounds[i] = k
    return out, rounds


def capture_stage2(d: Design, counts, b_probe=2000, seed=1):
    """`(xs, b_obs, q)` — stage 2's real inputs, captured rather than rebuilt.

    `subset_test` assembles `xs` from six things (`wade()`'s jitter stream,
    `tpm_like`, the fitted fold change, `thin_counts`, `one_count` and a seed
    derivation). Rebuilding them here would be six chances to diverge silently
    and return a confidently wrong answer, so this wraps `subset_null_backend`
    for one call and takes what it was handed. `subset_test` imports it inside
    its own body, so a runtime patch is seen. Verified against the kernel to
    0.000e+00 on the observed labels and on a random permutation.
    """
    import wade.permutation as WP

    grabbed, original = {}, WP.subset_null_backend

    def spy(xs, b_obs, perms, q, **kw):
        grabbed.setdefault("xs", xs)
        grabbed.setdefault("b_obs", b_obs)
        grabbed.setdefault("q", q)
        return original(xs, b_obs, perms, q, **kw)

    WP.subset_null_backend = spy
    try:
        rng = np.random.default_rng(seed)
        wade.wade(counts, np.ones(counts.shape[0]), d.cond, lib_sizes=np.ones(d.n),
                  perms=d.perms(rng, b_probe), nperms=b_probe, seed=1,
                  alternative="greater")
    finally:
        WP.subset_null_backend = original
    return grabbed["xs"], grabbed["b_obs"], grabbed["q"]


def frozen_moments(d: Design, xs, b_obs, q, n_ref=200_000, seed=3):
    """`(mu, safe)` — the bridge's null moments, estimated once and fixed.

    What stage 2 needs before any tail method can be applied to it. Splitting
    the two estimates is also better than not doing so: `mu` and `sigma` are
    bulk quantities converging as `1/sqrt(B)` while the tail is a rare-event
    quantity, so estimating them separately untangles two error sources that
    are currently conflated.
    """
    rng = np.random.default_rng(seed)
    _, _, mu, sd, _ = subset_null_backend(xs, b_obs, d.perms(rng, n_ref), q,
                                          alternative="greater")
    usable = sd > 0
    usable[:, -1] = False                  # B_m is identically zero
    return mu, np.where(usable, sd, np.inf)


def frozen_T(xs, q, mu, safe, labels, gene=None):
    from wade.permutation import _bridge_from

    sl = slice(None) if gene is None else slice(gene, gene + 1)
    return np.max((_bridge_from(xs[sl], labels, q) - mu[sl]) / safe[sl], axis=1)


def multilevel2(d: Design, xs, q, mu, safe, t_obs, genes,
                n_sample=400, sweeps=3, seed=5, max_rounds=2000):
    """Stage 2: no O(1) update — a swap changes the whole quantile function."""
    rng = np.random.default_rng(seed)
    half = n_sample // 2
    out, rounds = np.empty(len(genes)), np.empty(len(genes), dtype=int)
    for j, g in enumerate(genes):
        pop = d.perms(rng, n_sample).copy()
        s = np.array([frozen_T(xs, q, mu, safe, pop[r], gene=g)[0]
                      for r in range(n_sample)])
        a, k = float(t_obs[g]), 0
        while k < max_rounds:
            srt = np.argsort(s)
            level = float(s[srt[half - 1]])
            if level >= a:
                break
            pick = np.r_[srt[half:], srt[half:]][:n_sample]
            pop, s = pop[pick].copy(), s[pick].copy()
            for _ in range(sweeps):
                for r in range(n_sample):
                    ones, zeros = np.flatnonzero(pop[r]), np.flatnonzero(pop[r] == 0)
                    i1 = ones[rng.integers(len(ones))]
                    i0 = zeros[rng.integers(len(zeros))]
                    pop[r, i1], pop[r, i0] = 0, 1
                    t = frozen_T(xs, q, mu, safe, pop[r], gene=g)[0]
                    if t >= level:
                        s[r] = t
                    else:
                        pop[r, i1], pop[r, i0] = 1, 0
            k += 1
        out[j] = float(np.exp(_log_p(k, int((s >= a).sum()), n_sample)))
        rounds[j] = k
    return out, rounds


# ---------------------------------------------------------------------------
# stage 2 endpoints (pvalue-review.md §8, question 2)


def reviewer_xmax(d: Design, xs, q, mu, safe, g):
    """The proposed stage-2 endpoint: the observed `R` curve, sorted descending.

    **Not a bound.** It treats the multiset `{R_i}` as invariant under
    relabelling, but a permutation rebuilds *both* quantile functions rather
    than reordering one curve, so permutations reach `R` values outside the
    observed set and carry a different `sum(R)`. Measured, permutations exceed
    it on 16/24 genes at 40v40 and 24/24 at 60v20, up to 30% of draws. Kept
    because reproducing the refutation is the point.
    """
    from wade.quantiles import type7_quantiles
    from wade.subset import log_ratio_curve

    m = len(q)
    i1, i0 = split_groups(d.cond)
    R = log_ratio_curve(type7_quantiles(xs[g:g + 1, i1], q),
                        type7_quantiles(xs[g:g + 1, i0], q))[0]
    k = np.arange(1, m)
    bk = np.cumsum(np.sort(R)[::-1])[:m - 1] - (k / m) * R.sum()
    return float(np.max((bk - mu[g, :m - 1]) / safe[g, :m - 1]))


def valid_xmax(d: Design, xs, q, mu, safe, g):
    """A relaxation that **is** a bound: 0 violations in 96e6 gene-permutations.

    `B_k = sum_i w_i (L1_i - L0_i)` with `w_i = 1{i<=k} - k/m` and `L = log2 Q`.
    Type-7 quantiles are elementwise monotone in the sorted sample, so every
    grid position of either group is bracketed by the grid of the extreme
    subset (that group taking the largest, or the smallest, pooled values);
    bound each position in the direction its weight wants. Prefix sums give
    every `k` in O(m).

    Valid and useless: stage 2's endpoint lands at about twice the largest of
    2e6 null draws, so a fit anchored there is anti-conservative to 0.007 on a
    gene whose true tail shape is +0.31. See `pvalue-review.md` §8.
    """
    from wade.quantiles import type7_quantiles

    m = len(q)
    v = np.sort(xs[g])[::-1]
    hi1 = type7_quantiles(v[None, :d.n1], q)[0]
    lo1 = type7_quantiles(v[None, -d.n1:], q)[0]
    hi0 = type7_quantiles(v[None, :d.n0], q)[0]
    lo0 = type7_quantiles(v[None, -d.n0:], q)[0]
    upper = np.log2(hi1) - np.log2(lo0)          # w_i > 0: the head
    lower = np.log2(lo1) - np.log2(hi0)          # w_i < 0: the tail
    k = np.arange(1, m)
    bk = ((1 - k / m) * np.cumsum(upper)[:m - 1]
          - (k / m) * (lower.sum() - np.cumsum(lower)[:m - 1]))
    return float(np.max((bk - mu[g, :m - 1]) / safe[g, :m - 1]))


def fixed_endpoint_p(obs, null, xmax, n_tail=250):
    """`S(y) = (1 - (y-u)/(xmax-u))**alpha`, the shape by ML in closed form.

    With the endpoint fixed the GPD has one free parameter and
    `alpha = -n / sum(log(1 - z))` exactly, `z` the exceedances rescaled onto
    `[0, 1)`. Equivalent to a GPD with `xi = -1/alpha`, `sigma = (xmax-u)/alpha`.
    Returns `None` where refinement does not fire, and `0.0` when the
    observation is at or past the claimed endpoint -- which is the outcome that
    matters, and is what the proposed endpoint produces on 20 of 23 genes.
    """
    s = np.sort(null)[::-1]
    thr = float(s[n_tail])
    exc = s[:n_tail][s[:n_tail] > thr]
    if exc.size < 10 or obs <= thr or not xmax > thr:
        return None
    width = xmax - thr
    z = np.clip((exc - thr) / width, 0.0, 1 - 1e-15)
    alpha = -len(z) / np.log1p(-z).sum()
    z_obs = (obs - thr) / width
    if z_obs >= 1.0:
        return 0.0
    return (n_tail / len(s)) * (1 - z_obs) ** alpha


# ---------------------------------------------------------------------------
# drivers


#: Effect ladders. Stage 1's is a multiplicative shift; log10 p falls about
#: 5.4 per unit at 40 v 40. Stage 2's is the number of affected cases.
RUNGS1 = (1.45, 1.60, 1.77, 1.95, 2.13, 2.32, 2.51, 2.70)
RUNGS2 = (6, 8, 10, 12, 14, 16, 18, 20)


def _tag(d: Design, stage: int) -> str:
    return f"pvalue_{d}_s{stage}.npz"


def _calibrate(d, stage, b=1_000_000):
    rungs = RUNGS1 if stage == 1 else RUNGS2
    print(f"{d}, stage {stage}, B = {b:,}\n")
    if stage == 1:
        x, truth = cohort1(d, rungs)
        nexc, done = brute1(d, x, observed1(d, x), b)
    else:
        counts, truth = cohort2(d, rungs)
        nexc, done = brute2(d, counts, b)
    p = empirical_p(nexc, done)
    lab = "shift" if stage == 1 else "affected k"
    print(f"{lab:>11} {'floor':>11} {'median p':>12} {'min p':>12}")
    for r in rungs:
        m = truth == r
        fl = (wade.detectability_floor(d.n1, d.n0, int(r)) if stage == 2
              else float("nan"))
        print(f"{r:>11} {fl:11.2e} {np.median(p[m]):12.3e} {p[m].min():12.3e}")
    print(f"\nfloor at this B: {1/(done+1):.2e}")


def _truth(d, stage, n_perms):
    rungs = RUNGS1 if stage == 1 else RUNGS2
    print(f"{d} stage {stage}: {n_perms:,.0f} permutations, blocks of {BLOCK:,}",
          flush=True)
    if stage == 1:
        x, truth = cohort1(d, rungs)
        nexc, done = brute1(d, x, observed1(d, x), int(n_perms), report=20)
    else:
        counts, truth = cohort2(d, rungs)
        nexc, done = brute2(d, counts, int(n_perms), report=20)
    np.savez(_tag(d, stage), nexc=nexc, n_perms=done, truth=truth)
    p = empirical_p(nexc, done)
    print(f"\n  resolved (>=10 exceedances): {(nexc >= 10).sum()} of {len(nexc)}"
          f"   median p {np.median(p):.2e}   min {p.min():.2e}")


def _estimates(d, stage, b_small=2000, seed=11):
    """Every candidate's p-value on the same `b_small` permutations."""
    rungs = RUNGS1 if stage == 1 else RUNGS2
    rng = np.random.default_rng(seed)
    perms = d.perms(rng, b_small)
    est, extra = {}, {}
    if stage == 1:
        x, _ = cohort1(d, rungs)
        obs = observed1(d, x)
        null = null_statistics(x, perms)
        est["multilevel"] = multilevel1(d, x, obs)[0]
        if saddlepoint_applies(d):
            est["saddlepoint"] = saddlepoint_p(d, x, obs)
    else:
        counts, _ = cohort2(d, rungs)
        r = wade.wade(counts, np.ones(counts.shape[0]), d.cond,
                      lib_sizes=np.ones(d.n), perms=perms, nperms=b_small,
                      seed=1, alternative="greater", keep_null=True)
        obs, null = r.subset.statistic, r.subset.null
    est["empirical"] = empirical_p((null >= obs[:, None]).sum(axis=1), b_small)
    est["GPD moments"] = perm_pvalues(obs, null, alternative="greater")[0]
    est["GPD mle"] = gpd_mle_p(obs, null)
    extra["fits"] = [gpd_tail_p(o, null[i], detail=True) for i, o in enumerate(obs)]
    return est, extra, obs, null


def _compare(d, stage):
    z = np.load(_tag(d, stage))
    nexc, n_big = z["nexc"], int(z["n_perms"])
    true_p = empirical_p(nexc, n_big)
    ok = nexc >= 10
    est, extra, _, _ = _estimates(d, stage)

    order = ["empirical", "GPD moments", "GPD mle", "multilevel", "saddlepoint"]
    names = [m for m in order if m in est]
    print(f"\n=== {d}, stage {stage} ===  brute force {n_big:,}, "
          f"{ok.sum()} of {len(ok)} genes resolved")
    print(f"{'true p':>13} {'n':>4}  " + "".join(f"{m:>22}" for m in names))
    print(f"{'':>13} {'':>4}  " + "".join(f"{'med   worst-lo/hi':>22}" for _ in names))
    edges = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    worst_lo = {k: 1.0 for k in names}          # most anti-conservative seen
    for lo, hi in zip(edges[1:], edges[:-1]):
        m = ok & (true_p < hi) & (true_p >= lo)
        if not m.any():
            continue
        cells = ""
        for k in names:
            r = est[k][m] / true_p[m]
            worst_lo[k] = min(worst_lo[k], float(r.min()))
            cells += f"{np.median(r):>10.2f}{r.min():>6.2f}/{r.max():<5.0f}"
        print(f"{lo:.0e}-{hi:.0e} {int(m.sum()):>4}  {cells}")
    print("\n  worst ANTI-conservative ratio (below 1.0 means the p-value is "
          "too small):")
    for k in names:
        flag = "  <- unsafe" if worst_lo[k] < 0.5 else ""
        print(f"    {k:<14} {worst_lo[k]:.3f}{flag}")
    deep = ok & (true_p < 1e-5)
    if deep.any():
        xi = np.array([f.xi for f in extra["fits"]])
        expo = np.array([f.branch == "exponential" for f in extra["fits"]])
        print(f"  below 1e-5 ({deep.sum()} genes): shape negative "
              f"{(xi[deep] < 0).mean():.0%} (median {np.median(xi[deep]):+.2f}), "
              f"exponential branch {expo[deep].mean():.0%}")
    return {k: (est[k], true_p, ok) for k in names}


GEOMETRIES = (Design(40, 40), Design(60, 20), Design(70, 10), Design(20, 60))


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0] if a else "stress"
    if cmd == "stress":
        for d in GEOMETRIES:
            for stage in (1, 2):
                try:
                    _compare(d, stage)
                except FileNotFoundError:
                    print(f"\n=== {d}, stage {stage} === no truth file; run "
                          f"`truth {d.n1} {d.n0} {stage} <B>` first")
    else:
        d, stage = Design(int(a[1]), int(a[2])), int(a[3])
        if cmd == "calibrate":
            _calibrate(d, stage)
        elif cmd == "truth":
            _truth(d, stage, float(a[4]) if len(a) > 4 else 2e8)
        elif cmd == "compare":
            _compare(d, stage)
        else:
            raise SystemExit(f"unknown command {cmd!r}")
