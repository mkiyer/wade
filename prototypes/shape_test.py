"""Two nearly-orthogonal questions instead of one omnibus answer.

    STAGE 1  mean test   : is there ANY difference?  (the most sensitive
                           detector for a global shift -- the user's "null
                           hypothesis" that a plain DE method already answers)
    STAGE 2  shape test  : is a SUBSET a better explanation than a global shift?

Stage 2 scans the Brownian bridge of the log-ratio curve,
    R(p) = log2 Q_case(p) - log2 Q_ctrl(p),  S_k = cumsum(R),
    B_k = S_k - (k/m) S_m,
which is identically zero under a pure global fold change and positive when the
difference is concentrated at the top. B is orthogonal to the total by
construction, so stage 2 is not just re-testing stage 1.

The whole point is SPECIFICITY: stage 2 must stay at its nominal level on a
genuinely global change, or "subset" means nothing.
"""
import numpy as np
import wade

def curves(X, cond):
    st = wade.wade_stats(X, cond)
    R = np.log2(st.Q1) - np.log2(st.Q0)      # log scale: global fold change is FLAT
    S = np.cumsum(R, axis=1)
    m = R.shape[1]
    k = np.arange(1, m + 1)
    B = S - (k / m) * S[:, -1][:, None]      # departure from proportional growth
    return R, S, B

def run(n1, n0, classes, B_perm=400, n_per=200, seed=0, sdlog=0.6, shift_null=True):
    """shift_null=True builds stage 2's null under 'a pure global shift' rather
    than under 'no difference'.

    This is the user's framing made literal: the mean shift IS the null
    hypothesis for the shape test. Permuting the raw data tests against no
    difference at all, which is the wrong question -- and it is measurably the
    wrong question, because the permuted groups become mixtures of shifted and
    unshifted samples, whose spread does not match the shift model's, so the
    denominator of the standardization comes out too small and the test fires
    on genuine global changes.

    The fix uses the fact that B_k is EXACTLY invariant to a global fold change
    (R -> R - c sends S_k -> S_k - kc, and B_k = S_k - (k/m)S_m is unchanged).
    So the observed statistic needs no adjustment; only the null does. Divide
    the case columns by the estimated fold change and the two groups become
    exchangeable under H0, and permuting THAT is the right null.

    The estimate is the MEDIAN of the log-ratio curve, not the mean: a subset
    signal moves the top quantiles and leaves the median alone, so estimating
    the shift this way does not quietly remove the very signal being tested for.
    """
    rng = np.random.default_rng(seed)
    cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
    m = min(n1, n0)
    rows, lab = [], []
    for name, fn in classes.items():
        for _ in range(n_per):
            rows.append(fn(rng, n1, n0, sdlog)); lab.append(name)
    X = np.array(rows); lab = np.array(lab); g = X.shape[0]
    perms = wade.draw_perms(cond, B_perm, seed=seed + 1)

    # build the matrix the stage-2 null is generated from
    if shift_null:
        R0, _, _ = curves(X, cond)
        shift = 2.0 ** np.median(R0, axis=1)          # robust to a subset signal
        Xs = X.copy()
        Xs[:, cond == 1] /= shift[:, None]            # groups now exchangeable under H0
    else:
        Xs = X

    # pass 1: null moments for the bridge, per (gene, width)
    sB = np.zeros((g, m)); sB2 = np.zeros((g, m))
    for cb in perms:
        _, _, Bp = curves(Xs, cb)
        sB += Bp; sB2 += Bp * Bp
    muB = sB / B_perm; sdB = np.sqrt(np.maximum(sB2 / B_perm - muB**2, 0))
    sdB[sdB <= 0] = np.inf

    def stats(cb, M):
        R, S, Bb = curves(M, cb)
        i1 = np.flatnonzero(cb == 1); i0 = np.flatnonzero(cb == 0)
        return {"mean_shift": M[:, i1].mean(1) - M[:, i0].mean(1),
                "log_fc":     R.mean(1),
                "shape":      ((Bb - muB) / sdB).max(1)}

    # stage 1 reads the data as given; stage 2's observed value is identical on
    # X and Xs because B is shift-invariant, but its NULL comes from Xs.
    obs = stats(cond, X); names = list(obs)
    null = {nm: np.empty((B_perm, g)) for nm in names}
    for b, cb in enumerate(perms):
        st1 = stats(cb, X); st2 = stats(cb, Xs)
        for nm in names:
            null[nm][b] = st2[nm] if nm == "shape" else st1[nm]
    pv = {nm: (1 + (null[nm] >= obs[nm]).sum(0)) / (B_perm + 1) for nm in names}

    # pi_hat, the characterization, from the same curve
    R, _, _ = curves(X, cond)
    pi = ((R**2).sum(1)**2) / (m * (R**4).sum(1))
    return dict(lab=lab, pv=pv, names=names, null=null, obs=obs, pi=pi,
                n1=n1, n0=n0, m=m)

def report(r, alpha=0.05):
    print(f"\n{'='*94}")
    print(f"  {r['n1']} v {r['n0']}   m={r['m']}   rate at alpha={alpha}")
    print(f"{'='*94}")
    print(f"  {'gene class':<22s} {'mean_shift':>12s} {'log_fc':>10s} "
          f"{'SHAPE':>10s} {'EITHER':>9s} {'pi_hat median':>15s}")
    for c in dict.fromkeys(r["lab"]):
        s = r["lab"] == c
        either = np.mean((r['pv']['mean_shift'][s] <= alpha/2) |
                         (r['pv']['shape'][s] <= alpha/2))
        print(f"  {c:<22s} {np.mean(r['pv']['mean_shift'][s] <= alpha):>12.3f} "
              f"{np.mean(r['pv']['log_fc'][s] <= alpha):>10.3f} "
              f"{np.mean(r['pv']['shape'][s] <= alpha):>10.3f} "
              f"{either:>9.3f} {np.median(r['pi'][s]):>15.3f}")
    # independence of the two stages under the null
    n = r["null"]
    for a, b in (("mean_shift", "shape"), ("log_fc", "shape")):
        cc = [np.corrcoef(n[a][:, i], n[b][:, i])[0, 1]
              for i in range(0, n[b].shape[1], 37)]
        print(f"  correlation({a:>10s}, shape) under the null: median {np.median(cc):+.3f}"
              f"  [{np.percentile(cc,5):+.3f}, {np.percentile(cc,95):+.3f}]")
