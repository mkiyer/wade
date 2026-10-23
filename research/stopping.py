"""Decision-based stopping on real cohorts: does it reject exactly what finished runs reject, and
how much faster is it?

wade(fdr=...) shuffles each gene only until its BH decision at that level is settled (pvalues.settle).
On the TCGA examples' independent designs (prostate 445 tumours v 52 normals, breast 971 v 112;
the examples' tested genes and RLE size factors), per cancer:

  exact   a random 3,000-gene family run both ways, fdr=0.05 and fdr=None (every run finished):
          the decisions of both tests at BH levels 0.05, 0.1 and 0.2, the p- and q-values of the genes
          not rejected, the bounds of those rejected, and the shuffles each gene used
  full    the whole tested transcriptome with fdr=0.05: wall time, shuffles and passes
  whole   the whole tested transcriptome both ways (prostate only, by default): the exactness checks
          of `exact` on every gene, and the display markers' q-values, bounded and finished

    WADE_TCGA_CONFIG=.release/0.2.0/tcga-config.json RAYON_NUM_THREADS=6 PYTHONPATH=src:research \\
        python research/stopping.py [prostate breast] [exact full]
"""
import sys
import time

import numpy as np

import wade
from count_model import tcga

GENES = 3_000


def run(d, rows, fdr):
    raw = d["raw"][rows]
    norm = wade.Normalization.from_size_factors(raw, d["factors"], source="TCGA primary RLE")
    start = time.perf_counter()
    res = wade.wade(raw, d["cond"], normalization=norm, gene_names=d["names"][rows], seed=1, fdr=fdr)
    return res, time.perf_counter() - start


def exact(cancer):
    d = tcga(cancer)
    rows = np.sort(np.random.default_rng(2026102301).choice(len(d["raw"]), GENES, replace=False))
    fast, t_fast = run(d, rows, 0.05)
    full, t_full = run(d, rows, None)
    print(f"\n== exact: {cancer}, {int(d['cond'].sum())} v {int((d['cond'] == 0).sum())}, a random {GENES:,}-gene "
          f"family; fdr=0.05 in {t_fast:.0f}s, every run finished in {t_full:.0f}s", flush=True)
    for test in ("area", "subset"):
        p, q = getattr(fast, test + "_pvalue"), getattr(fast, test + "_padj")
        pf, qf = getattr(full, test + "_pvalue"), getattr(full, test + "_padj")
        called = qf <= 0.05
        same = all(np.array_equal(q <= a, qf <= a) for a in (0.05, 0.1, 0.2))
        exact_rest = np.array_equal(p[~called], pf[~called]) and np.array_equal(q[~called], qf[~called])
        bounds = bool(np.all(p[called] >= pf[called]) and np.all(q[called] <= 0.05))
        s_fast, s_full = getattr(fast, test + "_permutations"), getattr(full, test + "_permutations")
        print(f"  {test:6s} called {called.sum():,} of {full.tested.sum():,}; same decisions at 0.05/0.1/0.2: {same}; "
              f"p and q exact where not called: {exact_rest}; bounds where called: {bounds}; "
              f"bounded p-values {fast.bounded[test].sum():,}; shuffles {s_fast.sum():.3e} against {s_full.sum():.3e} "
              f"({s_full.sum() / max(s_fast.sum(), 1):.0f}x fewer), never more per gene: {bool(np.all(s_fast <= s_full))}; "
              f"passes {fast.params['passes'][test]}", flush=True)


def full(cancer):
    d = tcga(cancer)
    rows = np.arange(len(d["raw"]))
    res, t = run(d, rows, 0.05)
    print(f"\n== full: {cancer}, {int(d['cond'].sum())} v {int((d['cond'] == 0).sum())}, {res.tested.sum():,} tested "
          f"genes, fdr=0.05: {t:.0f}s", flush=True)
    for test in ("area", "subset"):
        s = getattr(res, test + "_permutations")
        print(f"  {test:6s} called {(getattr(res, test + '_padj') <= 0.05).sum():,}; shuffles {s.sum():.3e} "
              f"(max per gene {s.max():,}); bounded p-values {res.bounded[test].sum():,}; "
              f"passes {res.params['passes'][test]}", flush=True)


PANEL = ["ERG", "ETV1", "ETV4", "FLI1", "SPINK1"]


def whole(cancer):
    d = tcga(cancer)
    rows = np.arange(len(d["raw"]))
    fast, t_fast = run(d, rows, 0.05)
    full_, t_full = run(d, rows, None)
    print(f"\n== whole: {cancer}, {int(d['cond'].sum())} v {int((d['cond'] == 0).sum())}, {fast.tested.sum():,} genes "
          f"both ways; fdr=0.05 in {t_fast:.0f}s, every run finished in {t_full:.0f}s", flush=True)
    for test in ("area", "subset"):
        p, q = getattr(fast, test + "_pvalue"), getattr(fast, test + "_padj")
        pf, qf = getattr(full_, test + "_pvalue"), getattr(full_, test + "_padj")
        called = qf <= 0.05
        same = all(np.array_equal(q <= a, qf <= a) for a in (0.05, 0.1, 0.2))
        exact_rest = np.array_equal(p[~called], pf[~called]) and np.array_equal(q[~called], qf[~called])
        s_fast, s_full = getattr(fast, test + "_permutations"), getattr(full_, test + "_permutations")
        print(f"  {test:6s} called {called.sum():,}; same decisions at 0.05/0.1/0.2: {same}; exact where not called: "
              f"{exact_rest}; shuffles {s_fast.sum():.3e} against {s_full.sum():.3e} "
              f"({s_full.sum() / max(s_fast.sum(), 1):.0f}x fewer)", flush=True)
    names = np.asarray(fast.gene)
    for gene in PANEL:
        i = np.flatnonzero(names == gene)
        if len(i):
            i = i[0]
            print(f"  {gene:7s} subset q {fast.subset_padj[i]:.4g} (bounded: {bool(fast.bounded['subset'][i])}) "
                  f"against finished {full_.subset_padj[i]:.4g}; area q {fast.area_padj[i]:.4g} against "
                  f"{full_.area_padj[i]:.4g}", flush=True)


SECTIONS = dict(exact=exact, full=full, whole=whole)

if __name__ == "__main__":
    args = sys.argv[1:]
    cancers = [a for a in args if a in ("prostate", "breast")] or ["prostate", "breast"]
    sections = [a for a in args if a in SECTIONS] or ["exact", "full"]
    for cancer in cancers:
        for section in sections:
            SECTIONS[section](cancer)
