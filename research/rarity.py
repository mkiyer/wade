"""Rarity: does subset_rarity rank rare, strong subsets first, and does the sample load find
contaminated libraries?

On the TCGA examples' independent designs (prostate, 445 tumours v 52 normals; breast, 971 v 112;
the examples' tested genes and RLE size factors, seed 1, fdr=0.05). Ranks are within each gene's
rarity group: the subset test's validated range (lower group mean >= params["subset_validated_reads"],
32 reads since 25 October 2026) or below it.

Sections (default: all):

  biology        the markers' ranks, by rarity and by the gap alone; rarity against expression and
                 subset size; the rarest subsets, with the listed gene sharing most of their members;
                 the heaviest sample loads, with their drivers
  planted        subsets planted by exact binomial thinning into real prostate tumours: k of 445 tumours
                 (1, 3, 12, 30) raised or lowered by a fold f (1.5, 2, 4, 8), in 1% of the validated genes per
                 replicate (raised: every other sample thinned by 1/f; lowered: the k tumours thinned by 1/f).
                 The planted genes' rarity rank against their rank by the gap alone, the split's recovery of
                 the planted tumours, the subset test's calls, and the share of labellings that leave the
                 planted tumours all among the cases
  cnv            each gene's own DNA alteration (GISTIC high-level amplification for a raised subset, deep
                 deletion for a lowered one; TCGA PanCancer Atlas via cBioPortal, .release/0.2.0/tcga-cna-20261023)
                 in its subset's members against the other tumours, by rarity rank
  contamination  a breast tumour's library mixed into one prostate tumour as 0.5-5% of its reads: the
                 contaminated library's rank by sample load, with the load's weights -log(rarity) and their
                 excess over the null mean, max(0, -log(rarity) - 1)
  strong         the planted design at folds 16 and 32, the size of real driver subsets
  stability      rarity under seeds 1 and 2

    WADE_TCGA_CONFIG=.release/0.2.0/tcga-config.json RAYON_NUM_THREADS=6 PYTHONPATH=src:research \\
        python research/rarity.py [section ...]
"""
import sys
from math import comb
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

import wade
from count_model import tcga
from wade._rarity import memberships

ROOT = Path(__file__).resolve().parents[1]
CNA = ROOT / ".release/0.2.0/tcga-cna-20261023"
STUDIES = dict(prostate="prad_tcga_pan_can_atlas_2018", breast="brca_tcga_pan_can_atlas_2018")
MARKERS = dict(prostate=["ETV4", "ETV1", "ETV5", "ERG", "FLI1", "SPINK1", "TP63"],
               breast=["ERBB2", "GRB7", "STARD3", "PGAP3", "ESR1", "FOXA1", "KRT5", "MKI67"])


def analyze(d, counts=None, seed=1):
    raw = d["raw"] if counts is None else counts
    norm = wade.Normalization.from_size_factors(raw, d["factors"], source="TCGA primary RLE")
    return wade.wade(raw, d["cond"], normalization=norm, gene_names=d["names"], seed=seed)


def ranks(res):
    """Each gene's rarity rank within its group (1 the rarest; ties share the better rank), and the groups."""
    _, _, groups = res._subsets()
    rank = np.full(len(res.gene), np.nan)
    for group in (0, 1):
        idx = np.flatnonzero(groups == group)
        r = res.subset_rarity[idx]
        rank[idx] = np.searchsorted(np.sort(r), r, side="left") + 1
    return rank, groups


def gap_ranks(res, groups):
    """Each gene's rank by its gap alone within its group, the comparison without size matching."""
    rank = np.full(len(res.gene), np.nan)
    for group in (0, 1):
        idx = np.flatnonzero(groups == group)
        g = -res.lorenz_gap[idx]
        rank[idx] = np.searchsorted(np.sort(g), g, side="left") + 1
    return rank


def loads(res, excess):
    """The sample load with weights -log(rarity), or their excess over the null mean, max(0, -log(rarity) - 1)."""
    size, side, _ = res._subsets()
    genes = np.flatnonzero((size > 0) & np.isfinite(res.subset_rarity))
    weight = -np.log(res.subset_rarity[genes])
    if excess:
        weight = np.maximum(weight - 1, 0)
    load = np.zeros(int(res.cond.sum()))
    for g, w, members in zip(genes, weight, memberships(lambda rows: res.expression[rows], res.cond, size, side, genes)):
        load[members] += w / size[g]
    return load / (weight.sum() / len(load))


def biology():
    for cancer in ("prostate", "breast"):
        d = tcga(cancer)
        res = analyze(d)
        rank, groups = ranks(res)
        gap = gap_ranks(res, groups)
        size, side, _ = res._subsets()
        names = np.asarray(res.gene)
        sizes = res.params["rarity_groups"]
        print(f"\n== biology: {cancer}, {int(d['cond'].sum())} v {int((d['cond'] == 0).sum())}; rarity groups: "
              f"{sizes['validated']:,} genes in the validated range, {sizes['below']:,} below", flush=True)
        for gene in MARKERS[cancer]:
            i = np.flatnonzero(names == gene)
            if len(i):
                i = i[0]
                group = "validated" if groups[i] == 0 else "below"
                print(f"  {gene:7s} rank {int(rank[i]):6,} of the {group} group ({int(gap[i]):,} by the gap alone); "
                      f"subset {size[i]} tumours "
                      f"{'up' if side[i] > 0 else 'down' if side[i] < 0 else 'none'}; rarity {res.subset_rarity[i]:.2e}; "
                      f"subset q {res.subset_padj[i]:.3g}", flush=True)
        lower = np.log1p(np.minimum(res.case_mean, res.control_mean))
        for group, label in ((0, "validated"), (1, "below")):
            k = (groups == group) & (size > 0)
            print(f"  {label}: Spearman of rarity with log lower-group mean {spearmanr(res.subset_rarity[k], lower[k])[0]:+.2f}, "
                  f"with subset size {spearmanr(res.subset_rarity[k], size[k])[0]:+.2f} ({k.sum():,} genes with a subset)",
                  flush=True)
        rare = res.rare_subsets(n=20)
        row = {g: i for i, g in enumerate(names)}
        print("  20 rarest (validated): side and size, rarity, rank by the gap alone, and the listed gene sharing the "
              "most members", flush=True)
        for g, s, n, r, shared in zip(rare["gene"], rare["side"], rare["size"], rare["rarity"], rare["shared"]):
            partner = f"{shared[0][1]} of {n} with {shared[0][0]}" if shared else "no members with the others"
            print(f"    {g:16s} {'+' if s == 'up' else '-'}{n:<4d} {r:.1e}  gap rank {int(gap[row[g]]):6,}  shares {partner}",
                  flush=True)
        load = res.sample_load(drivers=6)
        top = np.argsort(-load["load"])[:5]
        print(f"  sample load: median {np.median(load['load']):.2f}, 99th percentile {np.quantile(load['load'], .99):.2f}; "
              f"heaviest:", flush=True)
        for j in top:
            print(f"    {load['load'][j]:5.1f} in {load['subsets'][j]:,} subsets: {', '.join(load['drivers'][j])}", flush=True)


GRID = [(k, f, up) for k in (1, 3, 12, 30) for f in (1.5, 2, 4, 8) for up in (True, False)]
STRONG = [(k, f, up) for k in (1, 3, 12, 30) for f in (16, 32) for up in (True, False)]


def planted(replicates=8, grid=GRID):
    d = tcga("prostate")
    raw, cond = d["raw"].astype(np.int64), d["cond"]
    cases = np.flatnonzero(cond == 1)
    base = analyze(d)
    _, base_groups = ranks(base)
    lower = np.minimum(base.case_mean, base.control_mean)
    reads = base.params["subset_validated_reads"]              # planted genes stay in the validated range
    pool = np.flatnonzero((base_groups == 0) & (lower >= reads * max(f for _, f, _ in grid)))
    per = round(0.01 * np.sum(base_groups == 0))
    print(f"\n== planted: prostate, {len(cases)} tumours; {per} planted genes per replicate (1% of the validated "
          f"range), drawn from the {len(pool):,} validated genes whose lower group keeps >= {reads} reads after a "
          f"thinning by {max(f for _, f, _ in grid):g}; {replicates} replicates; folds {sorted({f for _, f, _ in grid})}; "
          f"ranks are percentiles of the validated group", flush=True)
    n, n0 = raw.shape[1], int(np.sum(cond == 0))
    print(f"  share of the labellings of {n} samples that leave k given tumours all among the cases, where a shuffle "
          f"repeats the planted split: " + ", ".join(f"k={k} {comb(n - k, n0) / comb(n, n0):.3f}"
                                                   for k in sorted({k for k, _, _ in grid})), flush=True)
    rows = []
    for rep in range(replicates):
        rng = np.random.default_rng([2026102302, rep, len(grid)])
        hosts = rng.choice(pool, per, replace=False)
        counts = raw.copy()
        truth = {}
        for i, host in enumerate(hosts):
            k, f, up = grid[(i + rep) % len(grid)]
            members = rng.choice(cases, k, replace=False)
            if up:
                others = np.setdiff1d(np.arange(raw.shape[1]), members)
                counts[host, others] = rng.binomial(counts[host, others], 1 / f)
            else:
                counts[host, members] = rng.binomial(counts[host, members], 1 / f)
            truth[host] = (k, f, up, set(members.tolist()))
        res = analyze(d, counts)
        rank, groups = ranks(res)
        gap = gap_ranks(res, groups)
        m = np.sum(groups == 0)
        size, side, _ = res._subsets()
        for host, (k, f, up, members) in truth.items():
            found = set()
            if size[host] > 0:
                found = {cases[j] for j in memberships(lambda rows: res.expression[rows], res.cond, size, side, [host])[0]}
            right = size[host] > 0 and (side[host] > 0) == up
            hit = len(found & members)
            rows.append(dict(k=k, f=f, up=up, group=int(groups[host]), rarity=rank[host] / m, gap=gap[host] / m,
                             called=res.subset_padj[host] <= 0.05, right=right,
                             precision=hit / len(found) if found and right else np.nan,
                             recall=hit / k if right else 0.0))
        print(f"  replicate {rep + 1}: {len(truth)} planted", flush=True)
    t = pl.DataFrame(rows)
    print(f"\n  {'k':>3} {'fold':>4} {'side':>5} | rarity: median pct, top 1% | gap alone: median pct, top 1% | "
          f"called | side right | precision | recall", flush=True)
    for (k, f, up), g in sorted(t.group_by(["k", "f", "up"]), key=lambda x: (x[0][0], x[0][2] is False, x[0][1])):
        c = {name: g[name].to_numpy() for name in g.columns}
        print(f"  {k:3d} {f:4g} {'up' if up else 'down':>5} | {np.median(c['rarity']):6.1%} {np.mean(c['rarity'] <= .01):6.0%} | "
              f"{np.median(c['gap']):6.1%} {np.mean(c['gap'] <= .01):6.0%} | {np.mean(c['called']):5.0%} | "
              f"{np.mean(c['right']):5.0%} | {np.nanmedian(c['precision']) if np.isfinite(c['precision']).any() else np.nan:4.2f} | "
              f"{np.median(c['recall']):4.2f}", flush=True)
    wrong = t.filter(pl.col("group") != 0).height
    print(f"  planted genes that left the validated group: {wrong}", flush=True)


def cnv():
    for cancer in ("prostate", "breast"):
        d = tcga(cancer)
        res = analyze(d)
        rank, groups = ranks(res)
        size, side, _ = res._subsets()
        table = pl.read_csv(CNA / STUDIES[cancer] / "data_cna.txt", separator="\t", infer_schema_length=0)
        symbols = table["Hugo_Symbol"].to_list()
        cases = d["samples"][d["cond"] == 1]
        covered = [c for c in cases if c in table.columns]
        where = {c: j for j, c in enumerate(cases)}
        row = {}
        for i, s in enumerate(symbols):
            row.setdefault(s, i)
        values = table.select([pl.col(c).cast(pl.Float64, strict=False) for c in covered]).to_numpy()
        names = [str(n).split("|")[0] for n in res.gene]
        genes = [g for g in np.flatnonzero((groups == 0) & (size > 0)) if names[g] in row]
        bins = [(1, 100), (101, 500), (501, 2000), (2001, 5000), (5001, 10 ** 9)]
        tally = {b: np.zeros(5) for b in bins}
        detail = []
        counted = 0
        index = np.array([where[c] for c in covered])
        for g in genes:
            calls = values[row[names[g]]]
            ok = np.isfinite(calls)
            event = calls == (2 if side[g] > 0 else -2)
            if not event.any():
                continue
            counted += 1
            member = np.zeros(len(cases), bool)
            member[memberships(lambda rows: res.expression[rows], res.cond, size, side, [g])[0]] = True
            m = member[index] & ok
            o = ~member[index] & ok
            a, b, c, e = event[m].sum(), m.sum(), event[o].sum(), o.sum()
            for lo, hi in bins:
                if lo <= rank[g] <= hi:
                    tally[(lo, hi)] += [a, b, c, e, a > 0]
            if rank[g] <= 30:
                detail.append((int(rank[g]), names[g], "up" if side[g] > 0 else "down", int(size[g]), int(a), int(b), int(c), int(e)))
        print(f"\n== cnv: {cancer}, {len(covered)} of {len(cases)} tumours with GISTIC calls; validated genes with a subset "
              f"and at least one tumour carrying their own alteration (amplification for up, deep deletion for down): "
              f"{counted:,}, by rarity rank", flush=True)
        for (lo, hi), (a, b, c, e, any_) in tally.items():
            if b and e:
                rr = (a / b) / (c / e) if c else np.inf
                print(f"  rarity rank {lo:>5,}-{hi if hi < 10**9 else '':<6}: altered members {int(a):5,} of {int(b):6,} "
                      f"({a / b:6.1%}), other tumours {int(c):6,} of {int(e):8,} ({c / e:6.2%}); relative risk {rr:6.1f}; "
                      f"genes with an altered member {int(any_):,}", flush=True)
        for r, gene, s, k, a, b, c, e in sorted(detail):
            print(f"    #{r:<3} {gene:12s} {s:4s} {k:4d} tumours: {a}/{b} members altered, {c}/{e} others", flush=True)


def contamination(fractions=(0.005, 0.01, 0.02, 0.05), tumours=5):
    p, b = tcga("prostate"), tcga("breast")
    donor_rows = {n: i for i, n in enumerate(b["names"])}
    shared = np.array([donor_rows.get(n, -1) for n in p["names"]])
    donor = b["raw"][:, np.flatnonzero(b["cond"] == 1)[0]].astype(np.int64)
    aligned = np.where(shared >= 0, donor[np.maximum(shared, 0)], 0)
    raw = p["raw"].astype(np.int64)
    cases = np.flatnonzero(p["cond"] == 1)
    rng = np.random.default_rng(2026102303)
    hosts = rng.choice(cases, tumours, replace=False)
    print(f"\n== contamination: one breast tumour's library mixed into one of {tumours} prostate tumours "
          f"({np.mean(shared >= 0):.0%} of the prostate genes shared); the contaminated case's rank by load (1 the "
          f"heaviest of {len(cases)}) and its load, weights -log(rarity) | excess over the null mean", flush=True)
    base = analyze(p)
    for name, excess in (("-log rarity", False), ("excess", True)):
        clean = loads(base, excess)
        at = [int(np.flatnonzero(cases == h)[0]) for h in hosts]
        print(f"  uncontaminated, {name}: hosts' ranks {[int(np.sum(clean > clean[j]) + 1) for j in at]} "
              f"(loads {', '.join(f'{clean[j]:.1f}' for j in at)})", flush=True)
    for rho in fractions:
        out = {False: [], True: []}
        drivers = None
        for h in hosts:
            counts = raw.copy()
            j = int(h)
            q = min(1.0, rho * counts[:, j].sum() / max(aligned.sum(), 1))
            counts[:, j] = rng.binomial(counts[:, j], 1 - rho) + rng.binomial(aligned, q)
            res = analyze(p, counts)
            pos = int(np.flatnonzero(cases == h)[0])
            for excess in (False, True):
                load = loads(res, excess)
                out[excess].append((int(np.sum(load > load[pos]) + 1), load[pos]))
            if drivers is None:
                drivers = res.sample_load(drivers=6)["drivers"][pos]
        print(f"  {rho:5.1%} of reads: -log rarity ranks {[r for r, _ in out[False]]} (loads "
              f"{', '.join(f'{v:.1f}' for _, v in out[False])}) | excess ranks {[r for r, _ in out[True]]} (loads "
              f"{', '.join(f'{v:.1f}' for _, v in out[True])}); first host's drivers: {', '.join(drivers)}", flush=True)


def stability():
    d = tcga("prostate")
    a, b = analyze(d, seed=1), analyze(d, seed=2)
    ra, ga = ranks(a)
    rb, gb = ranks(b)
    keep = (ga == 0) & (gb == 0) & (a.subset_fraction > 0) & (b.subset_fraction > 0)
    top = lambda r, n: set(np.flatnonzero(keep & (r <= n)))
    print(f"\n== stability: prostate, seeds 1 and 2 (jitter, thinning and shuffles); {keep.sum():,} validated genes with a "
          f"subset under both: Spearman of rarity {spearmanr(a.subset_rarity[keep], b.subset_rarity[keep])[0]:.3f}; "
          f"the 100 rarest share {len(top(ra, 100) & top(rb, 100))}, the 1,000 rarest {len(top(ra, 1000) & top(rb, 1000))}",
          flush=True)
    names = np.asarray(a.gene)
    for gene in MARKERS["prostate"]:
        i = np.flatnonzero(names == gene)
        if len(i):
            print(f"  {gene:7s} rank {int(ra[i[0]]):,} and {int(rb[i[0]]):,}", flush=True)


def strong():
    planted(grid=STRONG)


SECTIONS = dict(biology=biology, planted=planted, strong=strong, cnv=cnv, contamination=contamination,
                stability=stability)

if __name__ == "__main__":
    for section in sys.argv[1:] or list(SECTIONS):
        SECTIONS[section]()
