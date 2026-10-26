"""Pooled thinning draws for the subset test: do K draws keep it calibrated and steady its calls?

The thinned subset test as first built (25 October 2026) thinned each gene once (wade._subsets.thinned
with draws=1): every sample to the smallest size factor, then the higher group by the fold. Which reads
are kept is random, and near BH's threshold one draw decides the call: on TCGA prostate two seeds agree
on 70% of the subset calls, against 93% when only the shuffles change (research/subset_range.py, section
seeds). This study combines K independent draws of the thinned pool, every draw walked under the
package's one shuffle stream:

  single       one draw (draw 0 is the first of the package's draws)
  mean p (K)   each draw's Besag-Clifford p-value under the same shuffles, averaged over the K draws.
               Each draw alone is the one-draw test. An average of valid p-values is valid within a
               factor of 2 under any dependence (Rüschendorf 1982; Vovk and Wang 2020); it estimates the
               p-value averaged over the thinning
  mean gap (K) the walk's score averaged over the K draws, in one Besag-Clifford run. Given its level, a
               sample's draws are correlated by the share of its reads it keeps, which differs between
               samples and between the groups, so these shuffles need not be exact

Genes without a subset on their normalized counts have p = 1, as in the package.

The package adopted the mean gap of 4 draws on 25 October 2026, natively (wade._subsets.DRAWS: the
kernel averages each shuffle's walks over a gene's draws). The constructions run by default are the
package's own test with 1, 4 and 8 draws ("one draw" is the test of the 25th, bit for bit); --prototype
runs the five constructions above instead, as first measured (subset-pooled-*-20261025.txt; the extra
draws seeded [seed, TAG, draw], the mean gap walked from Python).

Sections (default: all):

  grid       the count model's 48 data sets (subset_range.py, section grid): share at p <= 0.01 by the
             lower group's expected reads per sample, then overall
  bimodal    real zero-inflated genes under random splits of the tumours (subset_range.py, section bimodal)
  power      subsets planted by thinning at low counts (subset_range.py, section power)
  seeds      TCGA prostate with its own labels (445 tumours v 52 normals): the calls at BH q <= 0.05 under
             seeds 1 and 2, by the lower group's mean, and each construction's time. Every run finishes at
             9,999 shuffles: BH's threshold sits near p = 0.01, which 25 exceedances reach in about 2,500,
             so the calls are those of the default budget; the shipped construction is checked against
             wade() (with --prototype, the single draw against the package's one-draw test)

    WADE_TCGA_CONFIG=.release/0.2.0/tcga-config.json RAYON_NUM_THREADS=6 PYTHONPATH=src:research \\
        python research/subset_pooled.py [section ...] [--prototype]
"""
import sys
import time

import numpy as np

import subset_range
import wade
from count_model import tcga
from wade import _kernel
from wade._means import TIE_RTOL, expression, jittered
from wade._subsets import DRAWS, advance, split, subset_test, thinned, walks
from wade.permutation import STREAM_BLOCK, stream_block
from wade.pvalues import sequential_p

STOP, MAXIMUM = 25, 9_999
TAG = 2026102601                                   # the extra draws' streams: [seed, TAG, draw]
CHUNK = 4_000                                      # genes walked at once; p-values do not depend on it


def pools(counts, factors, cond, rows, seed, k):
    """K independent draws of the package's thinned pool, gene by gene: draw 0 is the package's own."""
    return [thinned(counts, factors, cond, rows, seed if j == 0 else [seed, TAG, j])[0] for j in range(k)]


def mean_p(draws, cond, seed, maximum=MAXIMUM):
    """Each draw's finished Besag-Clifford p-value under the one stream, averaged over the draws."""
    k, g = len(draws), len(draws[0])
    pooled, order, observed = walks(np.concatenate(draws), cond, np.ones(k * g), direction="both")
    exceed, tried = np.zeros(k * g, np.uint64), np.zeros(k * g, np.uint64)
    advance(pooled, order, observed, cond, exceed, tried, direction="both", stop=STOP, maximum=maximum,
            threshold=-np.inf, seed=seed)
    return sequential_p(exceed, tried, stop=STOP, maximum=maximum).reshape(k, g).mean(axis=0)


def mean_gap(draws, cond, seed, maximum=MAXIMUM):
    """The walk's score averaged over the draws: one finished Besag-Clifford run per gene, every gene
    reading the one stream from its start."""
    k, g = len(draws), len(draws[0])
    pooled, order, observed = walks(np.concatenate(draws), cond, np.ones(k * g), direction="both")
    target = observed.reshape(k, g).mean(axis=0)
    bar = target - TIE_RTOL * np.abs(target)                    # this close ties with it, as in the tally
    exceed, tried = np.zeros(g, np.int64), np.zeros(g, np.int64)
    active, position, labels, current = np.arange(g), 0, None, -1
    while position < maximum:
        active = active[(exceed[active] < STOP) & (tried[active] < maximum)]
        if not len(active):
            break
        rows = (np.arange(k)[:, None] * g + active).ravel()
        values, at = np.ascontiguousarray(pooled[rows]), np.ascontiguousarray(order[rows])
        while position < maximum:
            block, offset = divmod(position, STREAM_BLOCK)
            if block != current:
                labels, current = stream_block(cond, block, seed), block
            score = _kernel.score(values, at, "both", labels[offset]).reshape(k, len(active)).mean(axis=0)
            live = (exceed[active] < STOP) & (tried[active] < maximum)
            tried[active[live]] += 1
            exceed[active[live]] += score[live] >= bar[active[live]]
            position += 1
            if np.sum((exceed[active] < STOP) & (tried[active] < maximum)) < 0.75 * len(active):
                break                                            # slice the running genes afresh
    return sequential_p(exceed, tried, stop=STOP, maximum=maximum)


PROTOTYPE = (("single", mean_p, 1), ("mean p (4)", mean_p, 4), ("mean p (8)", mean_p, 8),
             ("mean gap (4)", mean_gap, 4), ("mean gap (8)", mean_gap, 8))
NATIVE = (("one draw", 1), (f"{DRAWS} draws (shipped)", DRAWS), ("8 draws", 8))
SHIPPED = NATIVE[1][0]
PROTOTYPING = "--prototype" in sys.argv


def construct(counts, factors, cond, rows, fold=None, seed=1, maximum=MAXIMUM, timing=None):
    """Every construction's p-values for genes `rows` (factors one row per gene, or one per sample); fold
    is unused (subset_range's sections pass the true fold for its oracle). timing, if given, collects each
    construction's seconds: its draws and its walks."""
    cond = np.ascontiguousarray(cond, dtype=np.uint8)
    rows = np.asarray(rows)
    if not PROTOTYPING:
        out = {}
        for name, k in NATIVE:
            start = time.perf_counter()
            out[name] = subset_test(counts, factors, cond, rows, direction="both", stop=STOP, maximum=maximum,
                                    seed=seed, draws=k)[0]
            if timing is not None:
                timing[name] = timing.get(name, 0.0) + time.perf_counter() - start
        return out
    described = expression(jittered(counts, rows, seed), factors)       # the subset wade() reports
    ran = np.flatnonzero(split(np.ascontiguousarray(described), cond)[3] > 0)
    out = {name: np.ones(len(counts)) for name, _, _ in PROTOTYPE}
    most = max(k for _, _, k in PROTOTYPE)
    for at in range(0, len(ran), CHUNK):
        part = ran[at:at + CHUNK]
        own = factors[part] if np.ndim(factors) == 2 else factors
        start = time.perf_counter()
        draws = pools(counts[part], own, cond, rows[part], seed, most)
        per_draw = (time.perf_counter() - start) / most
        for name, combine, k in PROTOTYPE:
            start = time.perf_counter()
            out[name][part] = combine(draws[:k], cond, seed, maximum)
            if timing is not None:
                timing[name] = timing.get(name, 0.0) + time.perf_counter() - start + k * per_draw
    return out


def grid():
    subset_range.grid(construct=construct)


def bimodal():
    subset_range.bimodal(construct=construct)


def power():
    subset_range.power(construct=construct)


def seeds(seeds_=(1, 2), alpha=0.05, markers=("ERG", "ETV1", "ETV4", "FLI1", "SPINK1")):
    d = tcga("prostate")
    raw, cond, factors, names = d["raw"], d["cond"].astype(np.uint8), d["factors"], list(d["names"])
    rows = np.arange(len(raw))
    lower = np.minimum(*[(raw[:, cond == c] / factors[cond == c]).mean(axis=1) for c in (1, 0)])
    marked = [(g, names.index(g)) for g in markers if g in names]
    print(f"\n== seeds: TCGA prostate, {int(cond.sum())} tumours v {int((cond == 0).sum())} normals, {len(raw):,} "
          f"tested genes; subset calls at BH q <= {alpha} under seeds {seeds_[0]} and {seeds_[1]} (each draws the "
          f"jitter, the thinning and the shuffles), every run finished at {MAXIMUM:,} shuffles", flush=True)
    q, seconds = {}, {}
    for s in seeds_:
        timing = {}
        p = construct(raw, factors, cond, rows, seed=s, timing=timing)
        for name, v in p.items():
            q.setdefault(name, []).append(wade.bh_adjust(v))
            seconds.setdefault(name, []).append(timing[name])
    names_run = [c[0] for c in (PROTOTYPE if PROTOTYPING else NATIVE)]
    if PROTOTYPING:
        one = subset_test(raw, factors, cond, rows, direction="both", stop=STOP, maximum=MAXIMUM, seed=seeds_[0],
                          draws=1)[0]
        print(f"  the single draw's calls under seed {seeds_[0]} are the package's one-draw test's: "
              f"{np.array_equal(q['single'][0] <= alpha, wade.bh_adjust(one) <= alpha)}", flush=True)
    else:
        norm = wade.Normalization.from_size_factors(raw, factors, source="TCGA primary RLE")
        package = wade.wade(raw, cond, normalization=norm, gene_names=d["names"], seed=seeds_[0], fdr=alpha)
        print(f"  the {SHIPPED} construction's calls under seed {seeds_[0]} are wade()'s: "
              f"{np.array_equal(q[SHIPPED][0] <= alpha, package.subset_padj <= alpha)}", flush=True)
    base = np.mean(seconds[names_run[0]])
    for name in names_run:
        a, b = (v <= alpha for v in q[name])
        either = a | b
        cells = []
        for lo, hi in zip((0, 8, 32, 128), (8, 32, 128, np.inf)):
            k = either & (lower >= lo) & (lower < hi)
            cells.append(f"[{lo:g}, {hi:g}) {np.sum(a & b & k) / max(k.sum(), 1):.1%} of {k.sum():,}")
        print(f"\n  {name}: calls {a.sum():,} and {b.sum():,}, {np.sum(a & b):,} in both "
              f"({np.sum(a & b) / either.sum():.1%} of those called by either); {np.mean(seconds[name]):.0f} s a "
              f"run, {np.mean(seconds[name]) / base:.1f}x the single draw", flush=True)
        print("    called by both, of those called by either: " + " | ".join(cells), flush=True)
        print("    " + "; ".join(f"{g} q {q[name][0][i]:.3g} / {q[name][1][i]:.3g}" for g, i in marked), flush=True)


SECTIONS = dict(grid=grid, bimodal=bimodal, power=power, seeds=seeds)

if __name__ == "__main__":
    for section in [a for a in sys.argv[1:] if not a.startswith("--")] or list(SECTIONS):
        SECTIONS[section]()
