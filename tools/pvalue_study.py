"""`docs/scaling.md` §4.5 — how far into the tail can each method be trusted?

§4.5 specifies the experiment: pick a design small enough that **brute force is
the ground truth**, then compare each candidate against it at true p-values
spanning 1e-3 to 1e-9. This is the harness. It is a study, not part of WADE.

    python tools/pvalue_study.py calibrate     # find effects giving a p ladder
    python tools/pvalue_study.py truth 1e9     # the brute-force column
    python tools/pvalue_study.py compare       # the §4.5 table

**Design.** 40 v 40, which §4.5 chose because it is small enough to permute
enormously and balanced, so stage 1's statistic is exactly the difference of
group means — a *linear* statistic, which is what the saddlepoint candidate
needs (§4.4). Stage 1 only, for now: stage 2's statistic is a maximum over
widths of a standardized bridge and has no saddlepoint, so it belongs to the
multilevel half of the comparison.

**Two things this had to get right.**

*Memory, not speed, is what stops brute force.* `draw_perms` materializes a
`(B, n)` label matrix and `null_statistics` returns `(genes, B)`: at B = 1e8
that is 64 GB and 51 GB. So permutations are **streamed** in blocks and only
the exceedance count is kept, which makes B unbounded and the footprint a
constant. Measured: drawing costs 1.4x scoring at 64 genes, so neither can be
ignored, and 64 genes is roughly where the two balance.

*The vectorized draw is bitwise identical to the shipped one.*
`rng.permuted(broadcast_to(cond, (B, n)), axis=1)` reproduces a loop of
`rng.permutation(cond)` exactly — arrays and final RNG state, verified over
108 configurations of group size, B and seed on numpy 2.5.2 — and is 3x
faster. It is used **here and not in `wade.draw_perms`**, because it is an
undocumented implementation detail of numpy, and on a real run (20,000 genes,
B = 2,000) drawing is already a rounding error against the kernel. Fragility
in the shipped inference path buys nothing there; here it buys a third off a
multi-hour run.
"""

from __future__ import annotations

import sys
import time

import numpy as np

import wade
from wade.permutation import null_statistics
from wade.pvalues import gpd_tail_p, perm_pvalues
from wade.stats import wade_stats

N1 = N0 = 40                      # §4.5's design: balanced, brute-forceable
N = N1 + N0
COND = np.r_[np.ones(N1, int), np.zeros(N0, int)]
G_PER_RUNG = 8                    # genes per effect size
BLOCK = 500_000                   # permutations per streamed block

#: Effect sizes, as a multiplicative shift applied to every case sample.
#: Calibrated on 2e6 permutations: log10 p falls about 5.4 per unit of shift
#: here, so these land the true p-value roughly a decade apart from 1e-2 to
#: 1e-9. Scatter within a rung is wanted, not a nuisance — it fills the range
#: in continuously instead of leaving eight discrete points. The last rung is
#: deliberately past what brute force can resolve, so the table shows where
#: the ground truth itself runs out.
RUNGS = (1.45, 1.60, 1.77, 1.95, 2.13, 2.32, 2.51, 2.70)


def cohort(seed=0):
    """`(x, truth)` — log-normal expression with a global shift per rung.

    Log-normal rather than negative binomial on purpose: this experiment is
    about the *tail of a permutation null*, and a continuous statistic with no
    ties is the cleanest setting to measure that in. The count-specific
    machinery (thinning, the pseudocount) belongs to stage 2 and to the
    benchmark, not here.
    """
    rng = np.random.default_rng(seed)
    rows, truth = [], []
    for mult in RUNGS:
        for _ in range(G_PER_RUNG):
            case = rng.lognormal(3.0, 0.6, N1) * mult
            rows.append(np.r_[case, rng.lognormal(3.0, 0.6, N0)])
            truth.append(mult)
    return np.array(rows), np.array(truth)


def observed(x):
    return wade_stats(x, COND).mean_shift


def brute_force(x, obs, n_perms, block=BLOCK, seed=7, report=None):
    """Exceedance counts over `n_perms` streamed permutations.

    Returns `(nexc, n_done)`. Nothing but the running count is retained, so
    the footprint is one block regardless of `n_perms` — which is the whole
    reason this can go past 1e8 at all.
    """
    rng = np.random.default_rng(seed)
    nexc = np.zeros(x.shape[0], dtype=np.int64)
    done = 0
    t0 = time.perf_counter()
    while done < n_perms:
        b = int(min(block, n_perms - done))
        perms = rng.permuted(np.broadcast_to(COND, (b, N)), axis=1)
        nexc += (null_statistics(x, perms) >= obs[:, None]).sum(axis=1)
        done += b
        if report and (done // block) % report == 0:
            el = time.perf_counter() - t0
            print(f"    {done:>14,} / {n_perms:,}   {el / 60:5.1f} min   "
                  f"eta {el * (n_perms / done - 1) / 60:5.1f} min", flush=True)
    return nexc, done


def empirical_p(nexc, n_perms):
    return (1.0 + nexc) / (n_perms + 1.0)


def gpd_p(x, obs, n_perms, n_tail=250, seed=11):
    """What the shipped refinement returns from `n_perms` permutations."""
    rng = np.random.default_rng(seed)
    perms = rng.permuted(np.broadcast_to(COND, (n_perms, N)), axis=1)
    null = null_statistics(x, perms)
    return np.array([gpd_tail_p(o, null[i], n_tail=n_tail)
                     for i, o in enumerate(obs)])


# ---------------------------------------------------------------------------


def _calibrate():
    x, truth = cohort()
    obs = observed(x)
    B = 2_000_000
    print(f"calibrating on {x.shape[0]} genes, B = {B:,}\n")
    nexc, done = brute_force(x, obs, B)
    p = empirical_p(nexc, done)
    print(f"{'shift':>7} {'median p':>12} {'min p':>12} {'resolved?':>10}")
    for mult in RUNGS:
        m = truth == mult
        med = np.median(p[m])
        print(f"{mult:>7.2f} {med:12.3e} {p[m].min():12.3e} "
              f"{'yes' if nexc[m].min() >= 10 else 'AT THE FLOOR':>10}")
    print(f"\nfloor at this B: {1 / (done + 1):.2e}")


def _truth(n_perms):
    x, truth = cohort()
    obs = observed(x)
    print(f"brute force: {x.shape[0]} genes, {n_perms:,.0f} permutations, "
          f"blocks of {BLOCK:,}", flush=True)
    nexc, done = brute_force(x, obs, int(n_perms), report=20)
    np.savez("pvalue_truth.npz", nexc=nexc, n_perms=done, truth=truth, obs=obs)
    p = empirical_p(nexc, done)
    print(f"\n{'shift':>7} {'median p':>12} {'exceedances':>13}")
    for mult in RUNGS:
        m = truth == mult
        print(f"{mult:>7.2f} {np.median(p[m]):12.3e} {int(np.median(nexc[m])):13,}")



# ---------------------------------------------------------------------------
# candidate 1: the double saddlepoint (scaling.md 4.4)


def _linear_target(x, obs):
    """`(values, a)` — the statistic recast as a subset sum, per gene.

    `mean_shift` on a balanced design is the difference of group means, and
    for ANY split it is an increasing affine function of `A`, the sum over the
    case subset::

        T = A/n1 - (V - A)/n0 = A (1/n1 + 1/n0) - V/n0

    so `P(T >= t) = P(A >= a)` with `a = (t + V/n0) / (1/n1 + 1/n0)`. That is
    the whole content of "the statistic must be linear": what the saddlepoint
    needs is that the statistic be a monotone function of a **subset sum**,
    and it is one here whatever the group sizes. **Balance is not a
    saddlepoint requirement** — it is a requirement for `mean_shift` to equal
    the mean difference in the first place (`scaling.md` 3.1), because the
    grid quadrature over `m = min(n1, n0)` nodes over-weights the larger
    group's extremes (`limits.md` 2.5). On an unbalanced design WADE's stage 1
    is an L-statistic, which has no such reduction, and that is what actually
    rules the saddlepoint out there.
    """
    V = x.sum(axis=1)
    return x, (obs + V / N0) / (1.0 / N1 + 1.0 / N0)


def _K(v, s, t):
    """CGF of `(sum Z_i v_i, sum Z_i)` for independent Bernoulli selectors, and
    its first two derivatives. Conditioning `sum Z_i = n1` recovers sampling
    without replacement, which is what a permutation does."""
    z = s * v + t
    # log(1 + e^z) stably
    K = np.logaddexp(0.0, z).sum()
    p = 1.0 / (1.0 + np.exp(-z))
    q = p * (1.0 - p)
    return K, (p @ v, p.sum()), (q @ (v * v), q @ v, q.sum())


def _solve_t(v, s, target, lo=-60.0, hi=60.0, iters=200):
    """`t` with `K_t(s, t) = target`. Monotone increasing in `t`, so bisect."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _K(v, s, mid)[1][1] < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def saddlepoint_p(x, obs):
    """Skovgaard's double-saddlepoint tail probability, no permutations at all.

    `P(A >= a | sum Z = n1)` for the subset sum, via the standard two-variable
    saddlepoint with the conditioning handled by the second variable. Returns
    one p-value per gene at `O(n)` per Newton step and no sampling, so it has
    no resolution floor of any kind.
    """
    from scipy.stats import norm

    vals, a_all = _linear_target(x, obs)
    out = np.empty(len(obs))
    for i in range(len(obs)):
        v = np.ascontiguousarray(vals[i], dtype=float)
        a = float(a_all[i])
        # denominator saddlepoint: s = 0, all selectors equal, t0 = log(n1/n0)
        t0 = float(np.log(N1 / N0))
        K0, _, (_, _, Ktt0) = _K(v, 0.0, t0)

        # numerator: solve K_s = a with t re-solved for K_t = n1 at each step
        lo, hi = -50.0 / max(np.abs(v).max(), 1e-12), 50.0 / max(np.abs(v).max(), 1e-12)
        for _ in range(200):
            s = 0.5 * (lo + hi)
            if _K(v, s, _solve_t(v, s, N1))[1][0] < a:
                lo = s
            else:
                hi = s
        s_hat = 0.5 * (lo + hi)
        t_hat = _solve_t(v, s_hat, N1)
        K1, (Ks, Kt), (Kss, Kst, Ktt) = _K(v, s_hat, t_hat)

        num = 2.0 * ((K0 - t0 * N1) - (K1 - s_hat * a - t_hat * N1))
        w = np.sign(s_hat) * np.sqrt(max(num, 0.0))
        det = Kss * Ktt - Kst * Kst
        u = s_hat * np.sqrt(max(det, 0.0) / Ktt0)
        if abs(w) < 1e-8 or abs(u) < 1e-12:
            out[i] = 0.5
        else:
            out[i] = float(norm.sf(w) - norm.pdf(w) * (1.0 / w - 1.0 / u))
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# candidate 2: fgsea's adaptive multilevel splitting (scaling.md 4.3)


def multilevel_p(x, obs, n_sample=1000, sweeps=6, seed=5, max_rounds=4000):
    """fgsea's scheme, with its gene swap replaced by a case/control label swap.

    Keep `n_sample` label vectors. Each round: discard the lower half by
    statistic, duplicate the upper half, then move every survivor by swapping
    one case label with one control label, accepting only while the statistic
    stays at or above the current level. The level ratchets to the population
    median, so `P ~ (1/2)^k` after `k` rounds and 1e-100 is ~332 rounds rather
    than 1e100 draws.

    The estimator is **not** `2^-k`. Each level is a *sample* median, whose
    exceedance probability is a Beta order statistic, so the rounds accumulate
    `E[log U] = psi(a) - psi(a + b)` rather than `log E[U]`; that is what makes
    `log p` approximately unbiased instead of systematically optimistic.

    The population is carried as the *indices* of each element's case subset
    rather than a boolean mask, so a swap is two fancy-index writes across the
    whole population at once and the subset sum updates in O(1) — a case value
    leaves, a control value enters. Written with a `rng.choice` per element it
    was a Python loop of `n_sample * sweeps` calls per round, and unusable.
    """
    from scipy.special import digamma

    rng = np.random.default_rng(seed)
    vals, a_all = _linear_target(x, obs)
    half = n_sample // 2
    rows = np.arange(n_sample)
    out = np.empty(len(obs))
    rounds = np.empty(len(obs), dtype=int)

    for i in range(len(obs)):
        v = np.ascontiguousarray(vals[i], dtype=float)
        a = float(a_all[i])
        # `inn` holds each element's case indices, `outp` the rest
        order = np.argsort(rng.random((n_sample, N)), axis=1)
        inn, outp = order[:, :N1].copy(), order[:, N1:].copy()
        s = v[inn].sum(axis=1)

        k = 0
        while k < max_rounds:
            srt = np.argsort(s)
            level = float(s[srt[half - 1]])       # the largest discarded
            if level >= a:
                break
            keep = srt[half:]
            pick = np.r_[keep, keep][:n_sample]
            inn, outp, s = inn[pick].copy(), outp[pick].copy(), s[pick].copy()
            for _ in range(sweeps):
                ci = rng.integers(N1, size=n_sample)
                co = rng.integers(N0, size=n_sample)
                gi, go = inn[rows, ci], outp[rows, co]
                delta = v[go] - v[gi]
                ok = (s + delta) >= level
                r = rows[ok]
                inn[r, ci[ok]], outp[r, co[ok]] = go[ok], gi[ok]
                s[r] += delta[ok]
            k += 1

        rem = int((s >= a).sum())
        log_p = (k * (digamma(half) - digamma(n_sample + 1))
                 + digamma(rem + 1) - digamma(n_sample + 1))
        out[i], rounds[i] = float(np.exp(log_p)), k
    return out, rounds


# ---------------------------------------------------------------------------
# the 4.5 table


def _compare(path="pvalue_truth.npz", b_small=2000):
    """Fill in §4.5's table: every method's error against brute force."""
    d = np.load(path)
    nexc, n_big, truth = d["nexc"], int(d["n_perms"]), d["truth"]
    x, _ = cohort()
    obs = observed(x)
    true_p = empirical_p(nexc, n_big)
    resolved = nexc >= 10                     # where brute force is itself sound

    rng = np.random.default_rng(11)
    perms = rng.permuted(np.broadcast_to(COND, (b_small, N)), axis=1)
    null = null_statistics(x, perms)

    t = time.perf_counter()
    emp = empirical_p((null >= obs[:, None]).sum(axis=1), b_small)
    t_emp = time.perf_counter() - t
    # the SHIPPED path, not raw gpd_tail_p: perm_pvalues keeps the empirical
    # value for a gene with >= n_exc_min exceedances and refines only the rest.
    t = time.perf_counter()
    gpd, _, refined_mask = perm_pvalues(obs, null, alternative="greater")
    t_gpd = time.perf_counter() - t
    branches = [gpd_tail_p(o, null[i], detail=True) for i, o in enumerate(obs)]
    t = time.perf_counter()
    sad = saddlepoint_p(x, obs)
    t_sad = time.perf_counter() - t
    t = time.perf_counter()
    mul, rounds = multilevel_p(x, obs)
    t_mul = time.perf_counter() - t

    print(f"brute force: {n_big:,} permutations, {resolved.sum()} of {len(obs)} "
          f"genes resolved (>= 10 exceedances); floor {1 / (n_big + 1):.1e}\n")
    print(f"{'true p':>10} {'n':>4}   " + "".join(f"{m:>21}" for m in
          ("empirical B=2k", "GPD B=2k", "saddlepoint", "multilevel")))
    print(f"{'':>10} {'':>4}   " + "".join(f"{'median  worst |log10|':>21}"
                                           for _ in range(4)))
    edges = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    for lo, hi in zip(edges[1:], edges[:-1]):
        m = resolved & (true_p < hi) & (true_p >= lo)
        if not m.any():
            continue
        cells = ""
        for est in (emp, gpd, sad, mul):
            r = np.log10(np.maximum(est[m], 1e-300) / true_p[m])
            cells += f"{10 ** np.median(r):>11.2f}x{np.abs(r).max():>9.2f}"
        print(f"{lo:.0e}-{hi:.0e} {int(m.sum()):>4}   {cells}")
    xi = np.array([b.xi for b in branches])
    expo = np.array([b.branch == "exponential" for b in branches])
    deep = resolved & (true_p < 1e-5)
    print(f"\n  on the {deep.sum()} genes with true p < 1e-5: the fitted shape is "
          f"negative for {(xi[deep] < 0).mean():.0%} of them (median xi "
          f"{np.median(xi[deep]):+.2f}), so {expo[deep].mean():.0%} take the "
          f"xi<=0 exponential branch.")

    print(f"\n{'method':>16} {'cost for 64 genes':>20} {'resolution':>22}")
    for name, tt, res in (("empirical B=2k", t_emp, f"1/(B+1) = {1/(b_small+1):.1e}"),
                          ("GPD B=2k", t_gpd, f"1/(B*n_tail) = {1/(b_small*250):.1e}"),
                          ("saddlepoint*", t_sad, "none - no sampling"),
                          ("multilevel", t_mul, "none - error ~ sqrt(k)")):
        print(f"{name:>16} {tt:>19.2f}s {res:>22}")
    print(f"\nmultilevel rounds: median {int(np.median(rounds))}, max {rounds.max()}")
    print("* saddlepoint cost is an UNOPTIMIZED nested bisection, 200x200 "
          "iterations per gene.\n  Newton on both variables would be orders "
          "faster; nothing here depends on its speed.")



# ---------------------------------------------------------------------------
# stage 2, which is a different problem


PHI = 0.1
BLOCK2 = 500_000


def cohort2(seed=0):
    """NB counts with a planted subset of growing strength, cases first.

    Counts, not log-normals: stage 2's null is built by binomial **thinning**
    of reads, so the study has to feed it the thing it was designed for.
    """
    rng = np.random.default_rng(seed)
    nb = lambda mu, size: rng.poisson(rng.gamma(1 / PHI, PHI * mu, size=size)).astype(float)
    rows, truth = [], []
    for k in RUNGS2:
        for _ in range(G_PER_RUNG):
            mu = 10 ** rng.uniform(1.6, 2.4)
            case, ctrl = nb(mu, N1), nb(mu, N0)
            case[:k] = nb(mu * FOLD2, k)
            rows.append(np.r_[case, ctrl])
            truth.append(k)
    return np.array(rows), np.array(truth)


#: The ladder for stage 2 is the **number of affected cases**, not the effect
#: size, and finding that out was the first result here. At a fixed k = 8 of
#: 40, subsets of 2x through 25x all returned p between 2.5e-4 and 5e-4: the
#: gene had reached the combinatorial floor for k = 8 (2.7e-3 by
#: `detectability_floor`, and p sits about a decade under it, per
#: `limits.md` §4.1) and a larger effect cannot buy depth the design has not
#: got. Varying k at a fixed strong fold spans true p from about 1e-2 to
#: 1e-8, which is the range 4.5 asks for.
RUNGS2 = (6, 8, 10, 12, 14, 16, 18, 20)
FOLD2 = 8.0


def _stage2_block(counts, perms):
    """One block: `(observed T, exceedance count)`, standardized consistently.

    The statistic is `max_k (B_k - mu_k) / sigma_k` with **mu and sigma
    estimated from the permutations themselves** (`method.md` §3), so unlike
    stage 1 it is *not* a fixed function of a label vector — the observed
    value moves with the permutation sample (measured: 3% relative at
    B = 2,000). Blocks are therefore self-consistent rather than shared: each
    takes its own observed value and its own null, both standardized by the
    same moments, and the estimate converges as the block grows. Measured
    across four independent blocks of 500,000: the observed T spreads 0.09%
    (max 0.38%) and the exceedance rate 0.4% (max 5.6%, on genes with few
    exceedances). At 20,000 it is 0.46% and 2.4%, which is why the block is
    large.
    """
    r = wade.wade(counts, np.ones(counts.shape[0]), COND, lib_sizes=np.ones(N),
                  perms=perms, nperms=len(perms), seed=1,
                  alternative="greater", keep_null=True)
    return r.subset.statistic, (r.subset.null >= r.subset.statistic[:, None]).sum(axis=1)


def brute_force_stage2(counts, n_perms, block=BLOCK2, seed=7, report=None):
    rng = np.random.default_rng(seed)
    nexc = np.zeros(counts.shape[0], dtype=np.int64)
    done = 0
    t0 = time.perf_counter()
    while done < n_perms:
        b = int(min(block, n_perms - done))
        perms = rng.permuted(np.broadcast_to(COND, (b, N)), axis=1)
        _, e = _stage2_block(counts, perms)
        nexc += e
        done += b
        if report and (done // block) % report == 0:
            el = time.perf_counter() - t0
            print(f"    {done:>14,} / {n_perms:,}   {el / 60:5.1f} min   "
                  f"eta {el * (n_perms / done - 1) / 60:5.1f} min", flush=True)
    return nexc, done


def _calibrate2():
    counts, truth = cohort2()
    B = 500_000
    print(f"stage 2: {counts.shape[0]} genes, B = {B:,}\n")
    nexc, done = brute_force_stage2(counts, B)
    p = empirical_p(nexc, done)
    print(f"{'affected k':>11} {'floor':>11} {'median p':>12} {'min p':>12}")
    for f in RUNGS2:
        m = truth == f
        print(f"{f:>11} {wade.detectability_floor(N1, N0, int(f)):11.2e} "
              f"{np.median(p[m]):12.3e} {p[m].min():12.3e}")
    print(f"\nfloor at this B: {1 / (done + 1):.2e}")


def _truth2(n_perms):
    counts, truth = cohort2()
    print(f"stage 2 brute force: {counts.shape[0]} genes, {n_perms:,.0f} "
          f"permutations, blocks of {BLOCK2:,}", flush=True)
    nexc, done = brute_force_stage2(counts, int(n_perms), report=20)
    np.savez("pvalue_truth2.npz", nexc=nexc, n_perms=done, truth=truth)
    p = empirical_p(nexc, done)
    print(f"\n{'affected k':>11} {'floor':>11} {'median p':>12} {'exceedances':>13}")
    for f in RUNGS2:
        m = truth == f
        print(f"{f:>11} {wade.detectability_floor(N1, N0, int(f)):11.2e} "
              f"{np.median(p[m]):12.3e} {int(np.median(nexc[m])):13,}")


def _compare2(path="pvalue_truth2.npz", b_small=2000):
    d = np.load(path)
    nexc, n_big, truth = d["nexc"], int(d["n_perms"]), d["truth"]
    counts, _ = cohort2()
    true_p = empirical_p(nexc, n_big)
    resolved = nexc >= 10

    rng = np.random.default_rng(11)
    perms = rng.permuted(np.broadcast_to(COND, (b_small, N)), axis=1)
    r = wade.wade(counts, np.ones(counts.shape[0]), COND, lib_sizes=np.ones(N),
                  perms=perms, nperms=b_small, seed=1, alternative="greater",
                  keep_null=True)
    obs, null = r.subset.statistic, r.subset.null
    emp = empirical_p((null >= obs[:, None]).sum(axis=1), b_small)
    gpd, _, _ = perm_pvalues(obs, null, alternative="greater")
    fits = [gpd_tail_p(o, null[i], detail=True) for i, o in enumerate(obs)]

    print(f"stage 2. brute force: {n_big:,} permutations, {resolved.sum()} of "
          f"{len(obs)} genes resolved; floor {1 / (n_big + 1):.1e}\n")
    print(f"{'true p':>13} {'n':>4}   {'empirical B=2k':>22}{'GPD B=2k':>22}")
    print(f"{'':>13} {'':>4}   " + "".join(f"{'median  worst |log10|':>22}" for _ in range(2)))
    edges = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    for lo, hi in zip(edges[1:], edges[:-1]):
        m = resolved & (true_p < hi) & (true_p >= lo)
        if not m.any():
            continue
        cells = ""
        for est in (emp, gpd):
            rr = np.log10(np.maximum(est[m], 1e-300) / true_p[m])
            cells += f"{10 ** np.median(rr):>12.2f}x{np.abs(rr).max():>9.2f}"
        print(f"{lo:.0e}-{hi:.0e} {int(m.sum()):>4}   {cells}")
    xi = np.array([f.xi for f in fits])
    expo = np.array([f.branch == "exponential" for f in fits])
    deep = resolved & (true_p < 1e-5)
    if deep.any():
        print(f"\n  on the {deep.sum()} genes with true p < 1e-5: shape negative for "
              f"{(xi[deep] < 0).mean():.0%} (median xi {np.median(xi[deep]):+.2f}), "
              f"{expo[deep].mean():.0%} on the exponential branch")



# ---------------------------------------------------------------------------
# multilevel for stage 2, and why it needs the moments frozen


def capture_stage2(counts, b_probe=2000, seed=1):
    """`(xs, b_obs, q)` — stage 2's actual inputs, captured rather than rebuilt.

    `subset_test` builds `xs` (the thinned, normalized, pseudocounted matrix)
    from six things `wade()` assembles: the jitter stream, `tpm_like`, the
    fitted fold change, `thin_counts`, `one_count` and a seed derivation. All
    six are reachable, and reconstructing them here would be six chances to
    diverge silently from the shipped path and get a confidently wrong answer.
    So this wraps `subset_null_backend` for one call and takes what it was
    handed — `subset_test` imports it inside its own body, so a runtime patch
    is seen.
    """
    import wade.permutation as WP

    grabbed = {}
    original = WP.subset_null_backend

    def spy(xs, b_obs, perms, q, **kw):
        grabbed.setdefault("xs", xs)
        grabbed.setdefault("b_obs", b_obs)
        grabbed.setdefault("q", q)
        return original(xs, b_obs, perms, q, **kw)

    WP.subset_null_backend = spy
    try:
        rng = np.random.default_rng(seed)
        perms = rng.permuted(np.broadcast_to(COND, (b_probe, N)), axis=1)
        wade.wade(counts, np.ones(counts.shape[0]), COND, lib_sizes=np.ones(N),
                  perms=perms, nperms=b_probe, seed=1, alternative="greater")
    finally:
        WP.subset_null_backend = original
    return grabbed["xs"], grabbed["b_obs"], grabbed["q"]


def frozen_moments(xs, b_obs, q, n_ref=200_000, seed=3):
    """`(mu, safe)` — the null moments of the bridge, estimated once and fixed.

    **This is what stage 2 needs before any tail method can be applied to it.**
    Its statistic is `max_k (B_k - mu_k) / sigma_k` with the moments estimated
    from the same permutations the tail is read from (`method.md` §3), so it
    is not a fixed function of a label assignment — measured, the observed
    value moves 3% between permutation samples at B = 2,000. Multilevel
    splitting samples from the distribution *conditioned* on exceeding a
    level, whose moments are not the null's, so applying it directly would
    standardize by the wrong numbers and silently estimate the tail of a
    different statistic.

    Splitting the two estimates is also better than what stage 1 does today.
    `mu` and `sigma` are bulk quantities converging as `1/sqrt(B)`; the tail
    is a rare-event quantity. Estimating them from one modest uniform sample
    and the tail by a method that can reach costs nothing and separates two
    error sources that are currently entangled.
    """
    from wade.permutation import subset_null_backend

    rng = np.random.default_rng(seed)
    ref = rng.permuted(np.broadcast_to(COND, (n_ref, N)), axis=1)
    _, _, mu, sd, _ = subset_null_backend(xs, b_obs, ref, q, alternative="greater")
    usable = sd > 0
    usable[:, -1] = False              # B_m is identically zero: no information
    return mu, np.where(usable, sd, np.inf)


def frozen_T(xs, q, mu, safe, labels, gene=None):
    """The stage-2 statistic as a fixed function of a label assignment."""
    from wade.subset import _bridge_from

    x = xs if gene is None else xs[gene:gene + 1]
    m = mu if gene is None else mu[gene:gene + 1]
    sf = safe if gene is None else safe[gene:gene + 1]
    return np.max((_bridge_from(x, labels, q) - m) / sf, axis=1)


def multilevel_stage2(xs, q, mu, safe, t_obs, genes, n_sample=1000, sweeps=4,
                      seed=5, max_rounds=2000):
    """fgsea's scheme on stage 2's statistic, with the moments held fixed.

    Unlike stage 1 there is no O(1) update: a label swap changes the whole
    quantile function, so every move costs a fresh bridge. That is the price
    of stage 2 not being a subset sum, and it is why this is seconds per gene
    rather than milliseconds.
    """
    from scipy.special import digamma

    rng = np.random.default_rng(seed)
    half = n_sample // 2
    out, rounds = np.empty(len(genes)), np.empty(len(genes), dtype=int)

    for j, g in enumerate(genes):
        pop = rng.permuted(np.broadcast_to(COND, (n_sample, N)), axis=1).copy()
        s = np.array([frozen_T(xs, q, mu, safe, pop[r], gene=g)[0]
                      for r in range(n_sample)])
        a, k = float(t_obs[g]), 0
        while k < max_rounds:
            srt = np.argsort(s)
            level = float(s[srt[half - 1]])
            if level >= a:
                break
            keep = srt[half:]
            pick = np.r_[keep, keep][:n_sample]
            pop, s = pop[pick].copy(), s[pick].copy()
            for _ in range(sweeps):
                for r in range(n_sample):
                    ones = np.flatnonzero(pop[r] == 1)
                    zeros = np.flatnonzero(pop[r] == 0)
                    i1 = ones[rng.integers(len(ones))]
                    i0 = zeros[rng.integers(len(zeros))]
                    pop[r, i1], pop[r, i0] = 0, 1
                    t = frozen_T(xs, q, mu, safe, pop[r], gene=g)[0]
                    if t >= level:
                        s[r] = t
                    else:
                        pop[r, i1], pop[r, i0] = 1, 0
            k += 1
        rem = int((s >= a).sum())
        out[j] = float(np.exp(k * (digamma(half) - digamma(n_sample + 1))
                              + digamma(rem + 1) - digamma(n_sample + 1)))
        rounds[j] = k
    return out, rounds


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "calibrate"
    if cmd == "calibrate":
        _calibrate()
    elif cmd == "truth":
        _truth(float(sys.argv[2]) if len(sys.argv) > 2 else 1e9)
    elif cmd == "compare":
        _compare(*sys.argv[2:3])
    elif cmd == "calibrate2":
        _calibrate2()
    elif cmd == "truth2":
        _truth2(float(sys.argv[2]) if len(sys.argv) > 2 else 1e8)
    elif cmd == "compare2":
        _compare2(*sys.argv[2:3])
    else:
        raise SystemExit(f"unknown command {cmd!r}")
