"""Brute-force truth for the PLAIN MEAN DIFFERENCE on unbalanced designs.

The reviewer's first recommendation is to redefine stage 1 as x1bar - x0bar,
which is a subset sum at every geometry. The existing truth files are for the
quadrature, so this builds ground truth for the redefined statistic, as a GEMM
over streamed permutation blocks, and then holds the saddlepoint to it.
"""
import sys, time
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np, pvalue_study as S

d = S.Design(int(sys.argv[1]), int(sys.argv[2]))
B = float(sys.argv[3]) if len(sys.argv) > 3 else 1e8
x, truth = S.cohort1(d, S.RUNGS1)
obs = x[:, :d.n1].mean(1) - x[:, d.n1:].mean(1)          # the redefined statistic

rng = np.random.default_rng(7)
nexc = np.zeros(len(obs), dtype=np.int64); done = 0; t0 = time.perf_counter()
while done < B:
    b = int(min(S.BLOCK, B - done))
    P = d.perms(rng, b).astype(float)                     # (b, n) labels
    null = (x @ P.T) / d.n1 - (x @ (1 - P).T) / d.n0      # (genes, b) mean differences
    nexc += (null >= obs[:, None]).sum(1); done += b
np.savez(f"meandiff_{d}.npz", nexc=nexc, n_perms=done, obs=obs)
p = S.empirical_p(nexc, done); ok = nexc >= 10
print(f"{d}: {done:,.0f} perms in {(time.perf_counter()-t0)/60:.1f} min, "
      f"{ok.sum()} of {len(ok)} resolved")

sad = S.saddlepoint_p(d, x, obs)                          # any n1, n0
edges = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
print(f"  {'true p':>13} {'n':>3} {'saddlepoint med':>16} {'worst-lo':>9} {'worst-hi':>9}")
worst = 1.0
for lo, hi in zip(edges[1:], edges[:-1]):
    m = ok & (p < hi) & (p >= lo)
    if not m.any(): continue
    r = sad[m] / p[m]; worst = min(worst, r.min())
    print(f"  {lo:.0e}-{hi:.0e} {int(m.sum()):>3} {np.median(r):>16.2f} {r.min():>9.2f} {r.max():>9.2f}")
print(f"  worst anti-conservative: {worst:.3f}")
