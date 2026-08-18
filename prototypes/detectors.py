"""Detector prototype: which threshold-free statistic detects a subset signal?

Six detectors, all given permutation p-values from the SAME permutation set, so
they are on equal footing and any difference is the statistic, not the calibration.

    mean       difference of sample means                  (the floor)
    wade_bulk  diff_mean, WADE's bulk axis                 (grid area)
    wade_tail  tail_mean at k=ceil(0.10*m)                 (THE INCUMBENT)
    maxz       max_i standardized D[i]                     (simplest sparse detector)
    scan       max_k standardized partial sum from the top (threshold-free by maximizing)
    hc         Higher Criticism over the m positions       (Donoho & Jin)

Standardization uses permutation-null moments per (gene, position), so every
detector is scale-free and comparable across genes. Two passes over the
permutations: one to estimate the moments, one to build each detector's null.
"""
import numpy as np
from scipy.special import ndtr
from math import lgamma
import wade

def lchoose(n, r): return lgamma(n+1) - lgamma(r+1) - lgamma(n-r+1)

def comb_floor(n1, n, k):
    """P(a random relabelling puts all k signal-carrying samples in the case group).
    No label-permutation test can return a p-value below this. A property of the
    DESIGN: no detector, effect size or permutation count moves it."""
    if k < 1 or k > n1: return 1.0
    return float(np.exp(lchoose(n1, k) - lchoose(n, k)))

def hc_stat(Z, alpha0=0.5):
    """Higher Criticism over the m positions of one gene's standardized curve."""
    m = Z.shape[1]
    p = ndtr(-Z)                               # one-sided upper p per position
    p = np.sort(p, axis=1)                     # ascending
    p = np.clip(p, 1e-15, 1 - 1e-15)
    i = np.arange(1, m + 1)
    hc = np.sqrt(m) * (i / m - p) / np.sqrt(p * (1 - p))
    cut = max(1, int(alpha0 * m))
    return hc[:, :cut].max(axis=1)

def detector_stats(X, cond, q, k_tail, mu_D, sd_D, mu_S, sd_S):
    """All six statistics for one labelling. Returns a dict of length-g arrays."""
    i1 = np.flatnonzero(cond == 1); i0 = np.flatnonzero(cond == 0)
    st = wade.wade_stats(X, cond)
    D = st.D
    S = np.cumsum(D, axis=1)                   # partial sums from the TOP of the grid
    out = {
        "mean":      X[:, i1].mean(1) - X[:, i0].mean(1),
        "wade_bulk": D.sum(1) / D.shape[1],
        "wade_tail": D[:, :k_tail].mean(1),
    }
    argmax = None
    if mu_D is not None:
        Z = (D - mu_D) / sd_D
        ZS = (S - mu_S) / sd_S
        out["maxz"] = Z.max(1)
        out["scan"] = ZS.max(1)
        out["hc"]   = hc_stat(Z)
        # the scan's argmax is the grid width it chose -- a fraction estimate
        # that falls out of DETECTION, at no extra cost
        argmax = ZS.argmax(1) + 1
    return out, D, S, argmax

def run(n1, n0, fracs, boost=8.0, n_null=200, n_per=100, B=400, seed=0, sdlog=0.6):
    rng = np.random.default_rng(seed)
    n = n1 + n0
    cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
    m = min(n1, n0)
    k_tail = max(1, int(np.ceil(0.10 * m)))

    rows, lab = [], []
    for _ in range(n_null):
        rows.append(np.r_[rng.lognormal(3, sdlog, n1), rng.lognormal(3, sdlog, n0)])
        lab.append(0.0)
    for f in fracs:
        kk = max(1, int(round(f * n1)))
        for _ in range(n_per):
            x = np.r_[rng.lognormal(3, sdlog, n1), rng.lognormal(3, sdlog, n0)]
            x[rng.choice(n1, kk, replace=False)] *= boost
            rows.append(x); lab.append(f)
    X = np.array(rows); lab = np.array(lab); g = X.shape[0]
    perms = wade.draw_perms(cond, B, seed=seed + 1)
    q = wade.probability_grid(m)

    # pass 1 -- permutation-null moments per (gene, position)
    sD = np.zeros((g, m)); sD2 = np.zeros((g, m))
    sS = np.zeros((g, m)); sS2 = np.zeros((g, m))
    for cb in perms:
        _, D, S, _ = detector_stats(X, cb, q, k_tail, None, None, None, None)
        sD += D; sD2 += D * D; sS += S; sS2 += S * S
    mu_D = sD / B; sd_D = np.sqrt(np.maximum(sD2 / B - mu_D**2, 0))
    mu_S = sS / B; sd_S = np.sqrt(np.maximum(sS2 / B - mu_S**2, 0))
    sd_D[sd_D <= 0] = np.inf; sd_S[sd_S <= 0] = np.inf

    # observed
    obs, _, _, argmax = detector_stats(X, cond, q, k_tail, mu_D, sd_D, mu_S, sd_S)
    names = list(obs)
    # pass 2 -- each detector's full permutation null, retained
    null = {nm: np.empty((B, g)) for nm in names}
    for b, cb in enumerate(perms):
        st, _, _, _ = detector_stats(X, cb, q, k_tail, mu_D, sd_D, mu_S, sd_S)
        for nm in names:
            null[nm][b] = st[nm]
    pvals = {nm: (1 + (null[nm] >= obs[nm]).sum(0)) / (B + 1) for nm in names}
    return dict(lab=lab, pvals=pvals, names=names, pi_scan=argmax / m,
                obs=obs, null=null, n1=n1, n0=n0, n=n, m=m,
                k_tail=k_tail, fracs=fracs, B=B)

def report(r, alpha=0.05):
    lab, pv, names = r["lab"], r["pvals"], r["names"]
    print(f"\n{'='*100}")
    print(f"  {r['n1']} cases vs {r['n0']} controls   m={r['m']}   incumbent k={r['k_tail']}"
          f"   B={r['B']}   power at alpha={alpha}")
    print(f"{'='*100}")
    print(f"  {'affected':>9s} {'k samp':>7s} {'comb floor':>11s} " + "".join(f"{nm:>11s}" for nm in names))
    t1 = [f"{np.mean(pv[nm][lab == 0] <= alpha):>11.3f}" for nm in names]
    print(f"  {'NULL (t1)':>9s} {'-':>7s} {'-':>11s} " + "".join(t1))
    for f in r["fracs"]:
        kk = max(1, int(round(f * r['n1'])))
        fl = comb_floor(r['n1'], r['n'], kk)
        sel = lab == f
        cells = "".join(f"{np.mean(pv[nm][sel] <= alpha):>11.3f}" for nm in names)
        mark = "  <- floor > alpha, undetectable by ANY label-permutation test" if fl > alpha else ""
        print(f"  {f:>8.0%} {kk:>7d} {fl:>11.3g} " + cells + mark)


def combine(r, parts, name="combo"):
    """One detection p-value from several detectors.

    Each is z-scored against its OWN permutation null so the scales are
    commensurable, the maximum is taken, and that maximum is calibrated against
    the null of the same maximum. The multiplicity of looking at more than one
    statistic is absorbed by the permutation null rather than paid for with a
    correction, which is the same trick the scan uses across k.
    """
    zo, zn = [], []
    for nm in parts:
        mu = r["null"][nm].mean(0); sd = r["null"][nm].std(0, ddof=1)
        sd[sd <= 0] = np.inf
        zo.append((r["obs"][nm] - mu) / sd)
        zn.append((r["null"][nm] - mu) / sd)
    zo = np.max(zo, axis=0); zn = np.max(zn, axis=0)
    B = zn.shape[0]
    r["pvals"][name] = (1 + (zn >= zo).sum(0)) / (B + 1)
    if name not in r["names"]:
        r["names"].append(name)
    return r
