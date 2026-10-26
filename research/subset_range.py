"""The subset test's validated range: can it reach below 128 reads per sample?

Until 25 October 2026 the subset test registered the cases onto the controls by the Wasserstein-1
fold, a fold on count + 1 held fixed through the shuffles. Under the count model it was calibrated
only where the sparser group expects at least 128 reads per sample: at low counts a fold, or a difference in depth
between the groups, changes Poisson noise and zeros as well as levels, and that reads as a subset
(docs/validation.md). Thinning keeps the count model exact: a thinned mixed-Poisson count is mixed
Poisson at the lower rate, whatever the latent level. This study measures constructions that thin
instead of scaling, on the count model's reference (research/count_model.py: TCGA prostate tumours, a
random split per gene, thinned binomially by design), at the arms that set the 128-read bound:

  package                the test until 25 October 2026: W1 fold on count + 1, unthinned normalized
                         counts (count_model.rescaled_subset_test)
  equalized, scaled      every sample thinned to the smallest size factor (as the area test), then the
                         shipped W1 registration
  equalized, thinned     every sample thinned to the smallest size factor; the higher group thinned
                         further by the W1 fold of the equalized counts; no registration in the shuffles
                         (the test since 25 October 2026, validated at 32 reads)
  equalized, fixed point the same with the fold by the W1 fixed point (thin, register, multiply, five
                         rounds; docs/research-ideas.md, section 4.2)
  oracle                 every sample thinned to the smallest exposure given by its size factor and the
                         true fold, no registration: exact (section 3.1)

Sections (default: all):

  grid       designs 222:223, 40:405, 405:40; depths 1 and 1/8; no fold, folds 2, 4 and 0.5, depth gaps
             of 2 either way, the TCGA prostate design's own imbalance, a fold of 2 with a depth gap.
             Share at p <= 0.01 by the sparser group's expected reads per sample, and overall at
             0.05 / 0.01 / 0.001; every 4th gene
  bimodal    real zero-inflated genes (GSTM1, ERAP2, PSPHP1, UGT2B17) under random splits of the tumours
  power      subsets planted by thinning (a tenth of the cases 8x, a fifth 4x) at low counts, by level
  seeds      the subset test on TCGA prostate with its own labels (445 tumours v 52 normals): how its
             calls move between seeds 1 and 2, by the lower group's mean. The package at relative_error
             0.2 and 0.1 (each seed draws the jitter, the thinning and the shuffles); seed 1's thinned pool
             under each seed's shuffles (the shuffles' share); and the rescaled test of before 25 October

    WADE_TCGA_CONFIG=.release/0.2.0/tcga-config.json RAYON_NUM_THREADS=6 PYTHONPATH=src:research \\
        python research/subset_range.py [section ...]
"""
import sys
import time

import numpy as np

import wade
from count_model import EDGES, LABELS, cohort_ratio, dataset, levels, share, shares, tcga
from wade import _kernel
from wade._means import equalized as package_equalized, expression, jittered
from count_model import rescaled_subset_test as subset_test
from wade._subsets import advance as package_advance, split, thinned as package_thinned, walks
from wade.permutation import STREAM_BLOCK, stream_block
from wade.pvalues import sequential_p, settled

MAXIMUM, STOP = 9_999, 25
CONSTRUCTIONS = ("package", "equalized, scaled", "equalized, thinned", "equalized, fixed point", "oracle")


def plain(x, cond, seed=1, direction="both"):
    """The subset statistic shuffled over the values as they are, with no registration: the package's
    walk, stream and Besag-Clifford stopping. p = 1 for genes without a subset."""
    cond = np.ascontiguousarray(cond, dtype=np.uint8)
    ran = np.flatnonzero(split(np.ascontiguousarray(x), cond)[3] > 0)
    p = np.ones(len(x))
    if not len(ran):
        return p
    order = np.ascontiguousarray(np.argsort(x[ran], axis=1, kind="stable"), dtype=np.int64)
    pooled = np.ascontiguousarray(np.take_along_axis(x[ran], order, axis=1))
    observed = _kernel.score(pooled, order, direction, cond)
    exceed, tried = np.zeros(len(ran), np.uint64), np.zeros(len(ran), np.uint64)
    while (live := ~settled(exceed, tried, stop=STOP, maximum=MAXIMUM, threshold=-np.inf)).any():
        block = int(tried[live].min()) // STREAM_BLOCK
        exceed[:], tried[:] = _kernel.tally(pooled, order, direction, stream_block(cond, block, seed), observed,
                                            exceed, tried, STOP, MAXIMUM, block * STREAM_BLOCK, -np.inf)
    p[ran] = sequential_p(exceed, tried, stop=STOP, maximum=MAXIMUM)
    return p


def thin_higher(counts, cond, fold, rng):
    """Thin the higher group by the fold: the cases by 1/fold where fold > 1, the controls by fold where
    fold < 1 (one fold per gene)."""
    fold = np.asarray(fold, float)[:, None]
    keep = np.where(cond == 1, np.where(fold > 1, 1 / fold, 1.0), np.where(fold < 1, fold, 1.0))
    return rng.binomial(counts, keep)


def w1_fold(counts, cond, rows, seed):
    """The package's registration (the W1 fold on count + 1) of jittered counts at one depth."""
    x = expression(jittered(counts, rows, seed), np.ones(counts.shape[1]))
    return split(np.ascontiguousarray(x), np.ascontiguousarray(cond, dtype=np.uint8))[4]


def fixed_point(counts, cond, rows, seed, rounds=5):
    """The W1 fixed point: thin the higher group by the current fold, register the thinned counts, multiply."""
    fold = w1_fold(counts, cond, rows, seed)
    for r in range(rounds):
        fold = fold * w1_fold(thin_higher(counts, cond, fold, np.random.default_rng([seed, 11, r])), cond, rows, seed)
    return fold


def oracle_equalize(counts, factors, cond, fold, rng):
    """Every sample thinned to its gene's smallest exposure: size factor, times the true fold for a case."""
    e = factors * np.where(cond == 1, fold, 1.0)
    return rng.binomial(counts, e.min(axis=1, keepdims=True) / e)


def constructions(counts, factors, cond, rows, fold, seed=1, which=CONSTRUCTIONS):
    """Each construction's p-values for genes `rows` (factors one row per gene, as the tests see them)."""
    out = {}
    ones = np.ones(counts.shape[1])
    rng = np.random.default_rng([seed, 29, int(rows[0]) if len(rows) else 0])
    level = package_equalized(counts, rows, factors, seed)              # every sample at the smallest size factor
    if "package" in which:
        out["package"] = subset_test(expression(jittered(counts, rows, seed), factors), cond, direction="both",
                                     stop=STOP, maximum=MAXIMUM, seed=seed)[0]
    if "equalized, scaled" in which:
        out["equalized, scaled"] = subset_test(expression(jittered(level, rows, seed), ones), cond, direction="both",
                                               stop=STOP, maximum=MAXIMUM, seed=seed)[0]
    if "equalized, thinned" in which:
        thinned = thin_higher(level, cond, w1_fold(level, cond, rows, seed), rng)
        out["equalized, thinned"] = plain(expression(jittered(thinned, rows, seed), ones), cond, seed)
    if "equalized, fixed point" in which:
        thinned = thin_higher(level, cond, fixed_point(level, cond, rows, seed), rng)
        out["equalized, fixed point"] = plain(expression(jittered(thinned, rows, seed), ones), cond, seed)
    if "oracle" in which:
        out["oracle"] = plain(expression(jittered(oracle_equalize(counts, factors, cond, fold, rng), rows, seed), ones),
                              cond, seed)
    return out


def line(name, p, bins, alpha=0.01):
    cells = [f"{share(p[bins == b], alpha):6.2%}" if np.sum(bins == b) >= 200 else f"{'--':>6s}" for b in range(len(LABELS))]
    return (f"  {name:23s} " + " | ".join(cells)
            + f" || {share(p, .05):.2%} / {share(p, .01):.2%} / {share(p, .001):.3%}")


def grid(designs=((222, 223), (40, 405), (405, 40)), depths=(1, 1 / 8), every=4, construct=None):
    ratio = cohort_ratio("prostate")
    arms = ((1, 1), (2, 1), (4, 1), (0.5, 1), (1, 0.5), (1, 2), (1, ratio), (2, 0.5))
    level = levels()
    print(f"\n== grid: the count model's reference (TCGA prostate tumours, a random split per gene, thinned by "
          f"design), every {every}th gene, up to {MAXIMUM:,} shuffles. Share at p <= 0.01 by the sparser group's "
          f"expected reads per sample {' | '.join(LABELS)}, then overall at 0.05 / 0.01 / 0.001", flush=True)
    for design in designs:
        for depth in depths:
            for fold, exposure in arms:
                start = time.time()
                p1, p0 = shares(fold, exposure)
                counts, factors, cond = dataset(design, depth, fold, exposure, 1)
                rows = np.flatnonzero(counts.any(axis=1))
                rows = rows[rows % every == 0]
                bins = np.digitize(level[rows] * depth * min(p1, p0), EDGES)
                p = (construct or constructions)(counts[rows], factors[rows], cond, rows, fold)
                print(f"\n{design[0]}:{design[1]}  depth 1/{round(1 / depth)}  f {fold:g}  d {exposure:.3g}  "
                      f"genes per bin {' | '.join(str(int(np.sum(bins == b))) for b in range(len(LABELS)))}  "
                      f"[{time.time() - start:.0f}s]", flush=True)
                for name, v in p.items():
                    print(line(name, v, bins), flush=True)


def bimodal(splits=2_000, designs=((20, 20), (10, 40), (40, 10), (100, 100)),
            genes=("GSTM1", "ERAP2", "PSPHP1", "UGT2B17"), construct=None):
    d = tcga("prostate")
    tumour = d["cond"] == 1
    print(f"\n== bimodal: real zero-inflated or bimodal genes, the TCGA prostate tumours split at random "
          f"({splits:,} splits per design, each a row); share of splits at p <= 0.05 / 0.01 / 0.001", flush=True)
    for gene in genes:
        raw = d["raw"][d["names"] == gene][0][tumour].astype(np.int64)
        factors = d["factors"][tumour]
        print(f"{gene}: counts {raw.min()} to {raw.max()}, median {np.median(raw):.0f}, zero in "
              f"{np.mean(raw == 0):.0%} of tumours", flush=True)
        for design in designs:
            rng = np.random.default_rng([2026102501, *design])
            pick = np.stack([rng.choice(len(raw), sum(design), replace=False) for _ in range(splits)])
            c, sf = raw[pick], factors[pick]
            sf = sf / np.exp(np.log(sf).mean(axis=1, keepdims=True))
            cond = np.r_[np.ones(design[0], np.uint8), np.zeros(design[1], np.uint8)]
            p = (construct or constructions)(c, sf, cond, np.arange(splits), 1.0)
            print(f"  {design[0]:3d}:{design[1]:<3d} " + " | ".join(
                f"{name} {share(v, .05):.2%} / {share(v, .01):.2%} / {share(v, .001):.3%}" for name, v in p.items()),
                flush=True)


def power(design=(222, 223), depths=(1, 1 / 8), every=4, construct=None):
    """Subsets planted by thinning: in each gene, a share of the cases keeps its reads and every other
    sample keeps 1/fold (a tenth at 8x, a fifth at 4x), under no common fold; share called at p <= 0.01
    by the sparser group's expected reads per sample (counted outside the subset)."""
    from count_model import halves, tumours
    from wade.normalize import fit_normalization
    level = levels()
    raw, half = tumours(), halves()
    rle = [fit_normalization(raw[half == 1 - h]).size_factors for h in (0, 1)]
    print(f"\n== power: subsets planted by thinning, {design[0]}:{design[1]}, every {every}th gene; share called at "
          f"p <= 0.01 by the expected reads per sample outside the subset {' | '.join(LABELS)}, then overall", flush=True)
    for depth in depths:
        for share_of, fold in ((0.1, 8), (0.2, 4)):
            rng = np.random.default_rng([2026102502, round(64 * depth), round(10 * share_of), fold])
            n1, n0 = design
            pick = np.argsort(rng.random(raw.shape), axis=1)[:, :n1 + n0]
            cond = np.r_[np.ones(n1, np.uint8), np.zeros(n0, np.uint8)]
            k = max(1, round(share_of * n1))
            raised = np.zeros(pick.shape, bool)
            raised[:, :k] = True                                     # the first k cases of each gene's own split
            counts = rng.binomial(np.take_along_axis(raw, pick, axis=1), depth * np.where(raised, 1.0, 1 / fold))
            factors = np.where(half[:, None] == 0, rle[0][pick], rle[1][pick])
            factors /= np.exp(np.log(factors).mean(axis=1, keepdims=True))
            rows = np.flatnonzero(counts.any(axis=1))
            rows = rows[rows % every == 0]
            bins = np.digitize(level[rows] * depth / fold, EDGES)
            p = (construct or constructions)(counts[rows], factors[rows], cond, rows, 1.0)
            print(f"\n  depth 1/{round(1 / depth)}: {share_of:.0%} of the cases at {fold}x", flush=True)
            for name, v in p.items():
                print(line(name, v, bins), flush=True)


def seeds(seeds_=(1, 2), alpha=0.05, markers=("ERG", "ETV1", "ETV4", "FLI1", "SPINK1")):
    d = tcga("prostate")
    raw, cond, factors, names = d["raw"], d["cond"].astype(np.uint8), d["factors"], list(d["names"])
    rows = np.arange(len(raw))
    norm = wade.Normalization.from_size_factors(raw, factors, source="TCGA primary RLE")
    marked = [(g, names.index(g)) for g in markers if g in names]
    print(f"\n== seeds: the subset test on TCGA prostate, {int(cond.sum())} tumours v {int((cond == 0).sum())} normals, "
          f"{len(raw):,} tested genes, BH q <= {alpha}; calls under seeds {seeds_[0]} and {seeds_[1]}, by the lower "
          f"group's mean", flush=True)

    def report(name, q, lower):
        a, b = (v <= alpha for v in q)
        either = a | b
        print(f"  {name}\n    calls {a.sum():,} and {b.sum():,}, {np.sum(a & b):,} in both ({np.sum(a & b) / either.sum():.1%} "
              f"of those called by either)", flush=True)
        cells = []
        for lo, hi in zip((0, 8, 32, 128), (8, 32, 128, np.inf)):
            k = either & (lower >= lo) & (lower < hi)
            cells.append(f"[{lo:g}, {hi:g}) {np.sum(a & b & k) / max(k.sum(), 1):.1%} of {k.sum():,}")
        print("    called by both, of those called by either: " + " | ".join(cells), flush=True)
        print("    " + "; ".join(f"{g} q {q[0][i]:.3g} / {q[1][i]:.3g}" for g, i in marked), flush=True)

    lower = first = None
    for precision in (0.2, 0.1):
        res = [wade.wade(raw, cond, normalization=norm, gene_names=d["names"], seed=s, fdr=alpha,
                         relative_error=precision) for s in seeds_]
        if lower is None:
            first, lower = res[0], np.minimum(res[0].case_mean, res[0].control_mean)
        report(f"the package, relative_error {precision}: each seed draws the jitter, the thinning and the shuffles",
               [r.subset_padj for r in res], lower)

    stop, maximum = 25, wade.DEFAULT_MAX_PERMUTATIONS                        # the defaults (relative_error 0.2)
    described = expression(jittered(raw, rows, seeds_[0]), factors)        # the package's subset_test, its stages apart
    ran = np.flatnonzero(split(described, cond)[3] > 0)
    pool, _ = package_thinned(raw[ran], factors, cond, rows[ran], seeds_[0])
    pooled, order, observed = walks(pool, cond, np.ones(len(ran)), direction="both")
    finished = []
    for s in seeds_:
        exceed, tried = np.zeros(len(ran), np.uint64), np.zeros(len(ran), np.uint64)
        package_advance(pooled, order, observed, cond, exceed, tried, direction="both", stop=stop, maximum=maximum,
                        threshold=-np.inf, seed=s)
        pv = np.ones(len(raw))
        pv[ran] = sequential_p(exceed, tried, stop=stop, maximum=maximum)
        finished.append(wade.bh_adjust(pv))
    print(f"  finished runs reproduce the package's seed {seeds_[0]} calls: "
          f"{np.array_equal(finished[0] <= alpha, first.subset_padj <= alpha)}", flush=True)
    report(f"the jitter and thinning of seed {seeds_[0]}, the shuffles of each seed (every run finished)", finished, lower)

    rescaled = [wade.bh_adjust(subset_test(expression(jittered(raw, rows, s), factors), cond, direction="both",
                                           stop=stop, maximum=maximum, seed=s)[0]) for s in seeds_]
    report("the rescaled test (until 25 October 2026): each seed draws the jitter and the shuffles", rescaled, lower)


SECTIONS = dict(grid=grid, bimodal=bimodal, power=power, seeds=seeds)

if __name__ == "__main__":
    for section in sys.argv[1:] or list(SECTIONS):
        SECTIONS[section]()
