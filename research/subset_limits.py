"""Single-gene limits of upward detection: where does area work, and is 4x detectable at all?

The confirmation generator's biology (Gamma- or lognormal-Poisson, exposure
log-sd 0.35, baseline mean 50, 250:250), normalized by true exposure. One-sided
upward p-values, Besag-Clifford h=20, all statistics on shared draws:

  linear / log : area_up on count/exposure or its log2(x+1)
  tail         : max over top-k pooled samples of the standardized excess of
                 cases; the cutoff search is repeated in every permutation
  oracle       : the best single cutoff chosen WITH HINDSIGHT (min_k
                 hypergeometric p). Anti-conservative on purpose: if even this
                 cannot reach the threshold, no cutoff-type upward test can.
  wilcoxon     : one-sided rank-sum, cases greater

    PYTHONPATH=src:research python research/subset_limits.py gamma 8
"""
import sys
import numpy as np
from scipy.stats import hypergeom, mannwhitneyu

MAXP, H, N1, N0 = 199_999, 20, 250, 250      # floor 5e-6; p <= 1e-4 needs >= 2e5 draws
FOLDS, CV2 = (4, 8, 16, 32, 128, 1024), (0.2, 0.5, 1.0)


def simulate(rng, cv2, bulk, fold, biology, frac=0.10, mean=50):
    n = N1 + N0
    exposure = np.exp(rng.normal(0, 0.35, n))
    exposure /= np.exp(np.log(exposure).mean())
    if biology == "gamma":
        latent = rng.gamma(1 / cv2, mean * cv2, n)
    else:
        v = np.log1p(cv2)
        latent = mean * np.exp(np.sqrt(v) * rng.normal(size=n) - v / 2)
    multiplier = np.full(N1, float(bulk))
    multiplier[rng.permutation(N1)[:round(frac * N1)]] = fold
    latent[:N1] *= multiplier
    return rng.poisson(latent * exposure) / exposure


def area_up(x):
    grid, order = N1 * np.arange(1, N1 + N0), np.argsort(x, kind="stable")
    gaps = np.diff(x[order])
    return lambda lab: np.maximum(grid - (N1 + N0) * np.cumsum(lab[:, order], axis=1)[:, :-1], 0) @ gaps


def tail_excess(x):
    n, p1 = N1 + N0, N1 / (N1 + N0)
    order, k = np.argsort(-x, kind="stable")[: n // 2], np.arange(1, n // 2 + 1)
    mu, sd = k * p1, np.sqrt(k * p1 * (1 - p1) * (n - k) / (n - 1))
    return lambda lab: ((np.cumsum(lab[:, order], axis=1) - mu) / sd).max(axis=1)


def pvalues(statistics, cond, rng):
    observed = [(f, f(cond[None])[0] * (1 - 100 * np.finfo(float).eps)) for f in statistics]
    k, t, done, base = np.zeros(len(statistics), int), np.zeros(len(statistics), int), 0, np.tile(cond, (4096, 1))
    while done < MAXP and np.any(k < H):
        b = min(4096, MAXP - done)
        labels = rng.permuted(base[:b], axis=1)
        for i, (f, threshold) in enumerate(observed):
            if k[i] < H:
                hits = f(labels) >= threshold
                crossing = np.flatnonzero(hits)
                used = crossing[H - k[i] - 1] + 1 if len(crossing) >= H - k[i] else b
                t[i] += used
                k[i] += int(hits[:used].sum())
        done += b
    return np.where(k == H, H / np.maximum(t, 1), (k + 1) / (MAXP + 1))


def oracle(x, cond):
    top = cond[np.argsort(-x, kind="stable")]
    return float(hypergeom.sf(np.cumsum(top) - 1, N1 + N0, N1, np.arange(1, N1 + N0 + 1)).min())


def gene(rng, cond, x):
    linear, log, tail = pvalues([area_up(x), area_up(np.log2(x + 1)), tail_excess(x)], cond, rng)
    control = x[cond == 0]
    return dict(linear=linear, log=log, tail=tail, oracle=oracle(x, cond),
                wilcoxon=mannwhitneyu(x[cond == 1], control, alternative="greater").pvalue,
                beyond_p99=float(np.mean(x[cond == 1] > np.quantile(control, .99))))


def main(biology="gamma", reps=8, seed=2026092701):
    rng, cond = np.random.default_rng(seed), np.r_[np.ones(N1, np.int64), np.zeros(N0, np.int64)]
    keys = ("linear", "log", "tail", "oracle", "wilcoxon")
    print(f"{biology}, {N1}:{N0}, 10% subset, {reps} genes/cell: share of genes with upward p <= 1e-4 "
          f"[median p]; beyond_p99 = cases above the control 99th percentile")
    print(f"{'bulk':>5} {'cv2':>4} {'fold':>5} | " + " ".join(f"{k:>16}" for k in keys) + " | beyond_p99")
    for bulk in (0.5, 1.0):
        for cv2 in CV2:
            for fold in FOLDS:
                rows = [gene(rng, cond, simulate(rng, cv2, bulk, fold, biology)) for _ in range(reps)]
                cells = " ".join(f"{np.mean([r[k] <= 1e-4 for r in rows]):4.0%} [{np.median([r[k] for r in rows]):7.1e}]"
                                 for k in keys)
                print(f"{bulk:5} {cv2:4} {fold:5} | {cells} | {np.mean([r['beyond_p99'] for r in rows]):5.1%}", flush=True)
    print("\nWhole-cohort shifts (fold 1 is the null; its column reports p <= 0.05):")
    for cv2 in CV2:
        for fold in (1.0, 1.25, 1.5, 2.0):
            rows = [gene(rng, cond, simulate(rng, cv2, fold, fold, biology)) for _ in range(reps * (5 if fold == 1 else 1))]
            cut = 0.05 if fold == 1 else 1e-4
            print(f"{cv2:4} {fold:5} | " + " ".join(f"{k} {np.mean([r[k] <= cut for r in rows]):4.0%}"
                                                  for k in ("linear", "log", "tail", "wilcoxon")), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gamma", int(sys.argv[2]) if len(sys.argv) > 2 else 8)
