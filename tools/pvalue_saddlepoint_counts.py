"""The saddlepoint on the data model users actually have: NB counts, through
wade()'s own normalization and jitter, at the sizes real cohorts have."""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np, wade, pvalue_study as S

n1, n0, B = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
d = S.Design(n1, n0); PHI = 0.1
rng = np.random.default_rng(3)
nb = lambda mu, size: rng.poisson(rng.gamma(1/PHI, PHI*mu, size=size)).astype(float)
# global shifts on counts, a ladder tuned to the design's power
shifts = np.geomspace(1.15, 1.15 + 1.0 * (80 / d.n) ** 0.5, 8)
rows = []
for f in shifts:
    for _ in range(8):
        mu = 10 ** rng.uniform(1.0, 2.5)
        rows.append(np.r_[nb(mu * f, n1), nb(mu, n0)])
counts = np.array(rows)
# what stage 1 actually sees: wade()'s normalized, jittered matrix
res = wade.wade(counts, np.ones(len(counts)), d.cond, lib_sizes=np.ones(d.n),
                nperms=50, seed=1, subset=False)
x = res.tpm
obs = x[:, :n1].mean(1) - x[:, n1:].mean(1)
print(f"{d}, NB counts: zeros {np.mean(counts == 0):.1%}, "
      f"median count {np.median(counts):.0f}")

t0 = time.perf_counter(); nexc = np.zeros(len(obs), np.int64); done = 0
prng = np.random.default_rng(7)
while done < B:
    b = int(min(S.BLOCK, B - done)); P = d.perms(prng, b).astype(float)
    null = (x @ P.T) / n1 - (x @ (1 - P).T) / n0
    nexc += (null >= obs[:, None]).sum(1); done += b
p = S.empirical_p(nexc, done); ok = nexc >= 10
print(f"  brute force {done:,.0f} in {(time.perf_counter()-t0)/60:.1f} min, "
      f"{ok.sum()} of {len(ok)} resolved, min p {p.min():.1e}")

t0 = time.perf_counter(); sad = S.saddlepoint_p(d, x, obs); dt = time.perf_counter() - t0
print(f"  saddlepoint: {dt/len(obs)*1000:.0f} ms/gene (unoptimized bisection)")
edges = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
print(f"  {'true p':>13} {'n':>3} {'median':>8} {'worst-lo':>9} {'worst-hi':>9}")
worst = 1.0
for lo, hi in zip(edges[1:], edges[:-1]):
    m = ok & (p < hi) & (p >= lo)
    if not m.any(): continue
    r = sad[m] / p[m]; worst = min(worst, r.min())
    print(f"  {lo:.0e}-{hi:.0e} {int(m.sum()):>3} {np.median(r):>8.2f} {r.min():>9.2f} {r.max():>9.2f}")
print(f"  worst anti-conservative: {worst:.3f}")
