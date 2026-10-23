"""The count model: an exact reference for WADE's tests, built from real counts, and their calibration under it.

The null that RNA-seq data need is the count model,

    C_j | Λ_j ~ Poisson(s_j f^{L_j} Λ_j),   the latent levels Λ_j drawn independently from one G,

for sample j's count of a gene, with s_j its exposure (sequencing depth), L_j its label (1 for a
case) and f the gene's fold. The area test's null is f = 1; the subset test's is any f (every case
carries one fold).

The reference. The 445 TCGA prostate tumours are split at random into pseudo-cases and
pseudo-controls, and thinned binomially, each sample with a probability fixed by the design alone:
q for depth, and group shares with p1 / p0 = f d, the larger 1. Thinning a count by p leaves it
mixed Poisson with its mean times p, whatever the latent level, so the thinned counts follow the
count model exactly, with the tumours' own depths and biology as the latent levels. The tests get
each sample's size factor (RLE of the unthinned tumours, from the other half of the genes, so
independent of the tested gene) times d for the cases.

Each gene gets its own random split. One split shared by every gene makes the share of genes
rejected swing with that split, because genes respond together to where a few unusual tumours
land; with a split per gene the share over genes is a binomial estimate of each test's size. Each
gene's samples are ordered with its cases first, so the package's functions run unchanged, with
one row of size factors per gene (tests() below mirrors wade(), and the section checks that it
matches wade() exactly on a shared split).

A gene's level is its mean count over all the tumours at the median depth; its bin is that level
times q times the sparser group's share: the expected reads per sample in the sparser group.

Sections (default: all):

  shipped      the area test (counts thinned to the shallowest depth) and, for contrast, the same
               test on the unthinned counts: under no fold across designs, depths and exposure
               imbalances (a 2x gap and the TCGA prostate and breast designs' own), and under a
               1.5x fold (what thinning washes out)
  composition  the normalization when a share of genes changes, RLE refitted on each data set
  real         the area test on TCGA prostate with its own labels: the depth kept, the calls
               against the unthinned test by level, and how the calls move with the thinning seed

The subset test is unchanged by the 30 September 2026 build, so its count-model measurements stand;
they, and the constructions tried and set aside (thinning registrations, the Laplace fold, rank and
count-exact score tests, depth choices), are in git at commit e4fd67f and recorded in
docs/research-ideas.md.

    WADE_TCGA_CONFIG=.release/0.2.0/tcga-config.json RAYON_NUM_THREADS=6 PYTHONPATH=src:research \\
        python research/count_model.py [section ...]
"""
import inspect
import sys
import tempfile
import time
from math import ceil
from pathlib import Path

import numpy as np

import wade
from wade._means import area_values, expression, jittered, mean_saddlepoint, mean_test
from wade._subsets import subset_test
from wade.normalize import fit_normalization

ROOT = Path(__file__).resolve().parents[1]
MAXIMUM = 9_999
RELATIVE_ERROR = inspect.signature(wade.wade).parameters["relative_error"].default
STOP = ceil(1 / RELATIVE_ERROR ** 2)
EDGES = (0.5, 2, 8, 32, 128)                  # expected reads per sample, each a factor of 4 above the last
LABELS = [f"[{lo:g},{hi:g})" for lo, hi in zip((0,) + EDGES, EDGES + (np.inf,))]
DESIGNS = ((222, 223), (40, 405), (405, 40))
CHUNK = 4096


# ---------------------------------------------------------------------------------------------
# Data

_COHORTS = {}


def tcga(cancer):
    """A TCGA cohort's independent design: raw counts of the tested genes, labels, size factors, names."""
    if cancer not in _COHORTS:
        sys.path.insert(0, str(ROOT / "tools"))
        import tcga_examples as TC
        with tempfile.TemporaryDirectory() as out:
            cohort = TC.load_cohort(cancer, out)
        idx = cohort.independent
        meta = cohort.meta[idx.tolist()]
        cond = (meta["group"].to_numpy() == "tumor").astype(int)
        factors = cohort.normalizations["primary"].select_samples(meta["library"].to_numpy()).size_factors
        counts = cohort.counts[:, idx]
        rows = np.flatnonzero(cohort.test_genes & (counts.sum(axis=1) > 0))
        _COHORTS[cancer] = dict(raw=counts[rows], cond=cond, factors=factors,
                                names=cohort.genes["gene_name"].to_numpy()[rows])
    return _COHORTS[cancer]


def tumours():
    """The prostate tumours' raw counts (integral floats in the bundle)."""
    d = tcga("prostate")
    return d["raw"][:, d["cond"] == 1].astype(np.int64)


def halves():
    return (np.random.default_rng(2026092907).random(len(tcga("prostate")["raw"])) < 0.5).astype(int)


def levels():
    """Each gene's mean count over all the tumours at the median depth."""
    d = tcga("prostate")
    tumour = d["cond"] == 1
    return (d["raw"][:, tumour] / d["factors"][tumour]).mean(axis=1)


def cohort_ratio(cancer):
    """A TCGA cohort's median tumour size factor over its median normal one: the exposure imbalance its
    own design carries."""
    d = tcga(cancer)
    return float(np.median(d["factors"][d["cond"] == 1]) / np.median(d["factors"][d["cond"] == 0]))


def shares(fold, exposure):
    """The shares of their reads the cases and the controls keep before depth: p1 / p0 = f d, the larger 1."""
    ratio = fold * exposure
    return (1.0, 1 / ratio) if ratio >= 1 else (ratio, 1.0)


def dataset(design, depth, fold, exposure, seed):
    """One data set: each gene's own random split of the tumours, its cases first, thinned by the
    design. Returns the counts, each gene's size factors as given to the tests, and the labels."""
    raw, half = tumours(), halves()
    rle = [fit_normalization(raw[half == 1 - h]).size_factors for h in (0, 1)]
    rng = np.random.default_rng([2026092905, *design, round(64 * depth), round(4 * fold), round(4 * exposure), seed])
    pick = np.argsort(rng.random(raw.shape), axis=1)[:, :sum(design)]
    cond = np.r_[np.ones(design[0], np.uint8), np.zeros(design[1], np.uint8)]
    p1, p0 = shares(fold, exposure)
    counts = rng.binomial(np.take_along_axis(raw, pick, axis=1), depth * np.where(cond == 1, p1, p0))
    factors = np.where(half[:, None] == 0, rle[0][pick], rle[1][pick]) * np.where(cond == 1, exposure, 1.0)
    factors /= np.exp(np.log(factors).mean(axis=1, keepdims=True))
    return counts, factors, cond


def share(p, alpha):
    ok = np.isfinite(p)
    return np.mean(p[ok] <= alpha) if ok.any() else np.nan


# ---------------------------------------------------------------------------------------------
# The package's tests, as wade() runs them

def area_pvalues(z, cond, seed):
    """The area test's p-values from its values z, as wade() computes them: the saddlepoint, and shuffles
    where its error estimate exceeds relative_error."""
    p, error = mean_saddlepoint(z, cond, direction="both")
    shuffle = error > RELATIVE_ERROR
    if shuffle.any():
        p[shuffle] = mean_test(z[shuffle], cond, direction="both", stop=STOP, maximum=MAXIMUM, seed=seed)[0]
    return p


def tests(counts, factors, cond, rows, seed=1):
    """The area test (on counts thinned to the shallowest depth), the same test on the unthinned
    normalized counts (the pre-build DE test), and the subset test, for genes `rows`."""
    x = expression(jittered(counts, rows, seed), factors)
    return dict(area=area_pvalues(area_values(counts, rows, factors, seed), cond, seed),
                unthinned=area_pvalues(x, cond, seed),
                subset=subset_test(x, cond, direction="both", stop=STOP, maximum=MAXIMUM, seed=seed)[0])


def matches_wade(genes=500, seed=1):
    """tests() reproduces wade() exactly on a shared split with one size factor per sample."""
    raw = tumours()[:genes]
    cond = np.r_[np.ones(222, np.uint8), np.zeros(223, np.uint8)]
    sf = fit_normalization(raw).size_factors
    res = wade.wade(raw, cond, normalization=wade.Normalization.from_size_factors(raw, sf, source="check"),
                    max_permutations=MAXIMUM, seed=seed)
    rows = np.flatnonzero(res.tested)
    mine = tests(raw[rows], np.broadcast_to(res.normalization.size_factors, (len(rows), raw.shape[1])), cond, rows, seed)
    return np.array_equal(mine["area"], res.area_pvalue[rows]) and np.array_equal(mine["subset"], res.subset_pvalue[rows])


# ---------------------------------------------------------------------------------------------
# Sections

def shipped(designs=DESIGNS + ((20, 20),), depths=(1, 1 / 8, 1 / 64)):
    print(f"\n== shipped: the area test on counts thinned to the shallowest sample's depth, against the same test "
          f"on the unthinned counts; a random split per gene, up to {MAXIMUM:,} shuffles. Share at p <= 0.01 by the "
          f"sparser group's expected reads per sample, then over all tested genes at 0.05 / 0.01 / 0.001", flush=True)
    print(f"tests() reproduces wade() exactly: {matches_wade()}", flush=True)
    level = levels()
    arms = ((1, 1), (1, cohort_ratio("prostate")), (1, cohort_ratio("breast")), (1, 0.5), (1, 2), (1.5, 1))
    for design in designs:
        for depth in (depths if design != (20, 20) else (1 / 8,)):
            for fold, exposure in arms:
                start = time.time()
                p1, p0 = shares(fold, exposure)
                counts, factors, cond = dataset(design, depth, fold, exposure, 1)
                rows = np.flatnonzero(counts.any(axis=1))
                bins = np.digitize(level[rows] * depth * min(p1, p0), EDGES)
                p = {k: np.full(len(rows), np.nan) for k in ("area", "unthinned")}
                for at in range(0, len(rows), CHUNK):
                    block = rows[at:at + CHUNK]
                    x = expression(jittered(counts[block], block, 1), factors[block])
                    p["area"][at:at + CHUNK] = area_pvalues(area_values(counts[block], block, factors[block], 1), cond, 1)
                    p["unthinned"][at:at + CHUNK] = area_pvalues(x, cond, 1)
                kind = "no fold" if fold == 1 else f"a {fold:g}x fold: power"
                print(f"\n{design[0]}:{design[1]}  depth 1/{round(1 / depth)}  {kind}, d {exposure:.3g}  "
                      f"[{time.time() - start:.0f}s]", flush=True)
                for name, v in p.items():
                    print(f"  {name:9s} " + " | ".join(f"{lab} {share(v[bins == b], .01):5.1%}" if np.sum(bins == b) >= 200
                                                       else f"{lab}    --" for b, lab in enumerate(LABELS))
                          + f" || {share(v, .05):.2%} / {share(v, .01):.2%} / {share(v, .001):.3%}", flush=True)


def composition(shares_=(0.1, 0.3), splits=4, fold=2.0, depth=1 / 8, design=(222, 223), every=8):
    """The normalization under changes of composition. One split shared by every gene (a sample's size
    factor depends on its label across genes), a share of the genes changed by `fold` (all up in the
    cases, or half up and half down), 1/8 depth; the area test on every `every`-th unchanged gene, with RLE
    refitted on the data set against the design's own size factors (RLE of the unthinned tumours)."""
    raw = tumours()
    base = fit_normalization(raw).size_factors
    cond = np.r_[np.ones(design[0], np.uint8), np.zeros(design[1], np.uint8)]
    print(f"\n== composition: {design[0]}:{design[1]}, one split shared by every gene ({splits} splits), depth 1/"
          f"{round(1 / depth)}, a share of genes changed {fold:g}x; the area test on every {every}th unchanged gene; share "
          f"at p <= 0.05 / 0.01 pooled over the splits", flush=True)
    for share_ in shares_:
        for balanced in (False, True):
            p = {"design factors": [], "refitted RLE": []}
            bias = []
            for split_index in range(splits):
                rng = np.random.default_rng([2026093005, round(100 * share_), balanced, split_index])
                pick = rng.permutation(raw.shape[1])[:sum(design)]
                counts = rng.binomial(raw[:, pick], depth)
                changed = rng.random(len(raw)) < share_
                up = changed & ((rng.random(len(raw)) < 0.5) if balanced else True)
                down = changed & ~up
                counts[np.ix_(up, cond == 0)] = rng.binomial(counts[np.ix_(up, cond == 0)], 1 / fold)
                counts[np.ix_(down, cond == 1)] = rng.binomial(counts[np.ix_(down, cond == 1)], 1 / fold)
                design_sf = base[pick] / np.exp(np.log(base[pick]).mean())
                refit = fit_normalization(counts).size_factors
                ratio = refit / design_sf
                bias.append(np.median(ratio[cond == 1]) / np.median(ratio[cond == 0]))
                rows = np.flatnonzero(~changed & counts.any(axis=1))
                rows = rows[rows % every == 0]
                for label, sf in (("design factors", design_sf), ("refitted RLE", refit)):
                    grid = np.broadcast_to(sf, (len(rows), len(sf)))
                    p[label].append(area_pvalues(area_values(counts[rows], rows, grid, 1), cond, 1))
            a, b = (np.concatenate(v) for v in p.values())
            print(f"{share_:.0%} of genes changed, {'half up, half down' if balanced else 'all up'}: refitted over design "
                  f"size factors, cases over controls, {', '.join(f'{v:.3f}' for v in bias)}; unchanged genes called with "
                  f"design factors {share(a, .05):.1%} / {share(a, .01):.1%}, refitted {share(b, .05):.1%} / "
                  f"{share(b, .01):.1%}; paired difference {share(b, .05) - share(a, .05):+.1%} / "
                  f"{share(b, .01) - share(a, .01):+.1%}", flush=True)


def real(seeds_=(1, 2), alpha=0.05):
    """The area test on TCGA prostate with its own labels (445 tumours v 52 normals): the depth kept, the
    calls against the same test on the unthinned counts by level, and how the calls move with the seed
    that draws the thinning and the jitter."""
    d = tcga("prostate")
    raw, cond, factors = d["raw"], d["cond"].astype(np.uint8), d["factors"]
    rows = np.arange(len(raw))
    lower = np.minimum(*[(raw[:, cond == c] / factors[cond == c]).mean(axis=1) for c in (1, 0)])
    print(f"\n== real: the area test on TCGA prostate, {int(cond.sum())} tumours v {int((cond == 0).sum())} normals, "
          f"{len(raw):,} tested genes; the shallowest sample keeps its reads, the median sample "
          f"{factors.min() / np.median(factors):.3f} of its own; calls at BH q <= {alpha}", flush=True)

    def pvalues(values, seed):
        return np.concatenate([area_pvalues(values(slice(at, at + CHUNK)), cond, seed) for at in range(0, len(raw), CHUNK)])

    p = {s: pvalues(lambda b, s=s: area_values(raw[b], rows[b], factors, s), s) for s in seeds_}
    p["unthinned"] = pvalues(lambda b: expression(jittered(raw[b], rows[b], 1), factors), 1)
    q = {k: wade.bh_adjust(v) for k, v in p.items()}
    edges = (0, 8, 32, 128, np.inf)
    for lo, hi in zip(edges, edges[1:]):
        k = (lower >= lo) & (lower < hi)
        print(f"  lower group mean [{lo:g}, {hi:g}) reads, {k.sum():,} genes: called unthinned {np.sum(q['unthinned'][k] <= alpha):,}, "
              + ", ".join(f"seed {s} {np.sum(q[s][k] <= alpha):,}" for s in seeds_), flush=True)
    a, b = (q[s] <= alpha for s in seeds_)
    either = a | b
    top = [set(np.argsort(p[s], kind="stable")[:200]) for s in seeds_]
    order = [np.argsort(np.argsort(p[s][either], kind="stable")) for s in seeds_]
    print(f"  seeds {seeds_[0]} and {seeds_[1]}: {a.sum():,} and {b.sum():,} calls, {np.sum(a & b):,} in both "
          f"({np.sum(a & b) / either.sum():.1%} of those called by either); Spearman of p among them "
          f"{np.corrcoef(*order)[0, 1]:.3f}; the 200 smallest p-values share {len(top[0] & top[1])} genes", flush=True)
    for lo, hi in zip(edges, edges[1:]):
        k = either & (lower >= lo) & (lower < hi)
        if k.any():
            print(f"    [{lo:g}, {hi:g}) reads: {np.sum(a & b & k):,} of {k.sum():,} called by either are called by both", flush=True)


SECTIONS = dict(shipped=shipped, composition=composition, real=real)

if __name__ == "__main__":
    for section in sys.argv[1:] or list(SECTIONS):
        SECTIONS[section]()
