"""The count model: an exact reference for WADE's two tests, built from real counts.

Step 1 of docs/subset-null-plan.md. The null that RNA-seq data need is the count model,

    C_j | Λ_j ~ Poisson(s_j f^{L_j} Λ_j),   the latent levels Λ_j drawn independently from one G,

for sample j's count of a gene, with s_j its exposure, L_j its label (1 for a case) and f the
gene's fold. The subset test's null is this model at any f (every case carries one fold); the
DE test's is f = 1. The tests work on normalized counts, and a fold of the latent level is not
a fold of the observed count + 1: Poisson noise and the pseudocount do not scale with it.

The reference. The 445 TCGA prostate tumours are split at random into pseudo-cases and
pseudo-controls and thinned binomially, each sample with a probability fixed by the design
alone. Thinning a count by p leaves it mixed Poisson with its mean times p, whatever the latent
level, so the thinned counts follow the count model exactly, with the tumours' own depths and
biology as the latent levels. No estimate enters a thinning probability, so nothing couples the
thinning to the counts it thins.

Each gene gets its own random split. One split shared by every gene makes the share of genes
rejected swing with that split, because genes respond together to where a few unusual tumours
land (section `splits`); the share over genes then measures the split, not the test. With a
split per gene it is a binomial estimate of each test's size. Each gene's samples are ordered
with its cases first, so the package's tests run unchanged, with one row of size factors per gene.

Each data set's design:

  q  depth: every sample keeps a share q of its reads (TCGA prostate: median 47M reads)
  d  the cases' exposure over the controls', as the size factors given to the tests say
  f  every gene's fold, cases over controls, relative to those exposures

The cases keep q p1 of their reads and the controls q p0, with p1 / p0 = f d and the larger
share 1. The tests are given each sample's size factor (RLE of the unthinned tumours) times d
for the cases, centred to geometric mean one as the package centres them. A gene's size factors
come from the other half of the genes (a fixed random half), so they do not depend on its counts.

A gene's level is its mean count over all the tumours at the median depth (normalized by the
cohort's size factors). Its bin is that level times q times the sparser group's share: the
expected count per sample in the sparser group, which does not depend on the splits.

Tests, each shuffled with the package's stopping rule up to 9,999 shuffles, on the genes the
package tests (any count in the design):

  mean             the package's mean test (its saddlepoint where precise enough)
  rank             the prototype rank test: Brunner and Munzel's studentized statistic
  subset           the package's subset test
  thinned          the subset statistic with no registration, after thinning the higher group
                   by the true fold (an oracle; exact when d = 1)
  fixed            the same after thinning by the fixed-point fold instead: thin by the fold,
                   register the thinned counts, multiply, five rounds (plan, section 3.5)
  equalized        the subset statistic with no registration, after thinning every sample to the
                   smallest exposure given by its size factor and the true fold (an oracle)
  rank, equalized  the rank test after thinning every sample to the smallest exposure

The count model's own registration (prototypes). A sample thinned to exposure e shows no reads
with the chance (1 - e/s_j)^C_j, whose mean over a group is the latent levels' Laplace transform
at e f, whatever their distribution and the samples' exposures. The exposures at which the cases'
and the controls' mean zero chance reach one level therefore have the ratio 1/f: the Laplace
fold, exact for any latent distribution, and little moved by a few extreme samples, whose zero
chance is near 0 whatever the fold.

  laplace             the subset statistic with no registration, after thinning the higher
                      group by the Laplace fold
  laplace, equalized  the same after thinning every sample to the smallest exposure given by its
                      size factor and the Laplace fold
  zero chance         a DE test: each sample's zero chance at the exposure where the pooled mean is
                      1/2 (under f = 1 every sample's has the same mean, whatever its exposure),
                      compared by the Welch-studentized difference in group means over shuffles

The DE tests (mean, rank, zero chance) run only at f = 1, their null; at any other fold every
gene is changed.

A count-exact rank (prototype, section `scores`): sample j's mid-rank score is its chance that its
count, thinned to the gene's smallest exposure, exceeds another sample's thinned count, plus half
the chance of a tie, averaged over the other samples. Under f = 1 every score has mean 1/2 whatever
the exposures, and the scores do not change under shuffles; the test compares their group means,
Welch-studentized.

Sections (default: splits and reference, in this order):

  splits     one split shared by every gene against a split per gene, at full depth
  reference  every design, depth and arm: shares at p <= 0.05 / 0.01 by level, and over all
             tested genes at 0.05 / 0.01 / 0.001
  laplace    the same grid for the Laplace-fold constructions (not in the default run)
  power      subsets planted by thinning: the package's subset test against the Laplace constructions
             (not in the default run)
  scores     the DE tests where they part, below 8 expected counts: the jittered rank test, the rank
             test after equalizing, the zero chance and count-exact mid-rank scores (not in the default run)
  realistic  the grid's tests at the exposure imbalance of the TCGA prostate and breast designs
             (not in the default run)
  depthchoice  the equalized and optimized-depth subset constructions: calibration and power
             (not in the default run)
  seeds      how much the thinned subset test's calls depend on its thinning draw (not in the
             default run)
  pairwise   mid-rank scores at the smallest depth against each pair at its own depth (not in
             the default run)
  small      the adopted constructions in small designs, with and without jitter (not in the default run)
  bimodal    review 2's bimodal null under the count model (not in the default run)
  realbimodal  GSTM1 and ERAP2 under random splits (not in the default run)
  robust     the W1 fixed point on equalized counts against the Laplace construction (not in the
             default run)
  composition  the normalization when a share of genes changes, RLE refitted (not in the default run)
  summary    the saved outputs of both grids condensed by group of arms (not in the default run)

    WADE_TCGA_CONFIG=.release/0.2.0/tcga-config.json RAYON_NUM_THREADS=6 PYTHONPATH=src:research \\
        python research/count_model.py [section ...]
"""
import inspect
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import bdtr
from scipy.stats import binom

import wade
from wade import _kernel
from wade._means import expression, jittered, mean_saddlepoint, mean_test
from wade._subsets import split, subset_test
from wade.normalize import fit_normalization
from wade.permutation import draw_perms
from wade.pvalues import sequential_p
from subset_null_study import PATTERNS, STOP, tcga, thin

MAXIMUM = 9_999
RELATIVE_ERROR = inspect.signature(wade.wade).parameters["relative_error"].default
EDGES = (0.5, 2, 8, 32, 128)                  # expected counts per sample, each a factor of 4 above the last
DESIGNS = ((222, 223), (40, 405), (405, 40))
DEPTHS = (1, 1 / 8, 1 / 64)
ARMS = ((1, 1), (1.5, 1), (2, 1), (4, 1), (0.5, 1), (1, 0.5), (1, 2), (2, 0.5), (2, 2))    # (f, d)
TESTS = ("mean", "rank", "subset", "thinned", "fixed", "equalized", "rank, equalized")
LAPLACE = ("laplace", "laplace, equalized", "zero chance")
DE = ("mean", "rank", "rank, equalized", "zero chance")
CHUNK = 4096
ROOT = Path(__file__).resolve().parents[1]
TIE_RTOL = 100 * np.finfo(float).eps           # as the package: statistics this close tie


# ---------------------------------------------------------------------------------------------
# Tests beside the package's

def shuffled(tally, observed, cond, seed):
    """Besag-Clifford p-values: `tally` counts a batch of shuffles for the genes still running."""
    exceed, tried = np.zeros(len(observed), np.uint64), np.zeros(len(observed), np.uint64)
    rng, size = np.random.default_rng(seed), 32
    while (live := (exceed < STOP) & (tried < MAXIMUM)).any():
        labels = draw_perms(cond, int(min(size, MAXIMUM - tried[live].max())), rng=rng)
        exceed, tried = tally(labels, observed, exceed, tried, STOP, MAXIMUM)
        size = min(2 * size, 4096)
    return sequential_p(exceed, tried, stop=STOP, maximum=MAXIMUM)


def mean_pvalues(x, cond, seed):
    """The package's mean test as wade() runs it: the saddlepoint, shuffles where its error is too large."""
    p, error = mean_saddlepoint(x, cond, direction="both")
    shuffle = error > RELATIVE_ERROR
    if shuffle.any():
        p[shuffle] = mean_test(x[shuffle], cond, direction="both", stop=STOP, maximum=MAXIMUM, seed=seed)[0]
    return p


def rank_test(x, cond, seed, direction="both"):
    """The prototype rank test: Brunner and Munzel's statistic over shuffles (values untied)."""
    cond = np.ascontiguousarray(cond, np.uint8)
    order = np.ascontiguousarray(np.argsort(x, axis=1, kind="stable"), np.int64)
    observed = _kernel.rank_score_study(order, direction, cond)
    return shuffled(lambda *a: _kernel.rank_tally_study(order, direction, *a), observed, cond, seed)


def plain_subset_test(x, cond, seed, direction="both"):
    """The package's subset statistic shuffled over the values as they are, with no registration."""
    cond = np.ascontiguousarray(cond, np.uint8)
    ran = np.flatnonzero(split(x, cond, "linear")[4] > 0)
    order = np.ascontiguousarray(np.argsort(x[ran], axis=1, kind="stable"), np.int64)
    pooled = np.ascontiguousarray(np.take_along_axis(x[ran], order, axis=1))
    observed = _kernel.score(pooled, order, True, direction, False, cond)
    p = np.ones(len(x))
    p[ran] = shuffled(lambda *a: _kernel.tally(pooled, order, True, direction, False, *a), observed, cond, seed)
    return p


def fixed_point(counts, cond, factors, rows, seed, rounds=5):
    """The fold of subset_null_study.thinned_fold, with the package's registration: thin the higher
    group by the current fold, register the thinned counts, multiply."""
    fold = split(expression(jittered(counts, rows, seed), factors, "linear"), cond, "linear")[5]
    for r in range(rounds):
        thinned = thin(counts, cond, fold, np.random.default_rng([seed, 11, r]))
        fold = fold * split(expression(jittered(thinned, rows, seed), factors, "linear"), cond, "linear")[5]
    return fold


def equalize(counts, factors, cond, fold, rng):
    """Thin every sample to its gene's smallest exposure: sample j keeps min(e) / e_j of its reads,
    with e_j its size factor, times the fold (a number or one per gene) for a case."""
    e = factors * np.where(cond == 1, np.reshape(fold, (-1, 1)), 1.0)
    with np.errstate(invalid="ignore"):
        keep = e.min(axis=-1, keepdims=True) / e
    return rng.binomial(counts, np.where(e > 0, keep, 0.0))      # a fold of 0 or infinity: a group without reads


def optimized(counts, factors, cond, fold, rng):
    """Thin every kept sample to its gene's common exposure a, the one keeping the most expected reads:
    a maximizes a x #{j : e_j >= a} over the samples' own exposures e_j (size factor, times the fold
    for a case), and the samples below it are set aside. The kept samples' counts are then mixed Poisson
    at exposure a, but the null stays exact only if exposure is independent of the latent levels. Real
    size factors are not (composition moves both), and with a fold the cut sets aside different depth
    ranges in the two groups: section `depthchoice` measures this construction anti-conservative. Returns
    the thinned counts and which samples each gene keeps."""
    e = factors * np.where(cond == 1, np.reshape(fold, (-1, 1)), 1.0)
    ordered = -np.sort(-e, axis=1)
    with np.errstate(invalid="ignore"):
        value = np.where(np.isfinite(ordered), ordered * np.arange(1, e.shape[1] + 1), -np.inf)
    a = ordered[np.arange(len(e)), np.argmax(value, axis=1)]
    kept = (e >= a[:, None]) & np.isfinite(e) & (e > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        keep = np.where(kept, a[:, None] / e, 0.0)
    return rng.binomial(counts, keep), kept


def kept_subset_test(x, cond, kept, seed):
    """plain_subset_test on each gene's kept samples (genes grouped by their kept group sizes); p = 1
    where a group keeps fewer than two samples."""
    p = np.ones(len(x))
    n1, n0 = (kept & (cond == 1)).sum(axis=1), (kept & (cond == 0)).sum(axis=1)
    for a, b in sorted(set(zip(n1.tolist(), n0.tolist()))):
        rows = np.flatnonzero((n1 == a) & (n0 == b))
        if a < 2 or b < 2:
            continue
        order = np.argsort(np.where(kept[rows], np.where(cond == 1, 0, 1), 2), axis=1, kind="stable")[:, :a + b]
        p[rows] = plain_subset_test(np.take_along_axis(x[rows], order, axis=1),
                                    np.r_[np.ones(a, np.uint8), np.zeros(b, np.uint8)], seed)
    return p


def zero_chance(counts, factors, e):
    """Each sample's chance of no reads when thinned to exposure e (at most its own): (1 - e/s)^C."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(counts == 0, 1.0, np.exp(counts * np.log(1 - np.minimum(e / factors, 1))))


def exposure_at(counts, factors, level, steps=60):
    """Each gene's exposure at which its samples' mean zero chance equals `level` (one per gene),
    bisected in log exposure below the smallest one; that smallest one where the mean stays above."""
    top = np.log(factors.min(axis=1))
    lo, hi = top - 50, top.copy()
    for _ in range(steps):
        mid = (lo + hi) / 2
        above = zero_chance(counts, factors, np.exp(mid)[:, None]).mean(axis=1) > level
        lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    return np.exp(hi)


def laplace_fold(counts, factors, cond):
    """The Laplace fold, cases over controls: the ratio of the exposures at which the controls' and the
    cases' mean zero chance reach one level, 1/2 or, where either group's stays higher down to its
    smallest exposure, that higher value. 0 where the cases have no reads, infinite where the controls
    have none."""
    groups = [(counts[:, cond == g], factors[:, cond == g]) for g in (1, 0)]
    level = np.maximum.reduce([np.full(len(counts), 0.5)] + [
        zero_chance(c, s, s.min(axis=1, keepdims=True)).mean(axis=1) for c, s in groups])
    (c1, s1), (c0, s0) = groups
    with np.errstate(divide="ignore", invalid="ignore"):
        fold = exposure_at(c0, s0, level) / exposure_at(c1, s1, level)
    return np.where(c1.sum(axis=1) == 0, 0.0, np.where(c0.sum(axis=1) == 0, np.inf, fold))


def welch_test(y, cond, seed):
    """Two-sided shuffles of the Welch-studentized difference in means, with the package's stopping rule.

    The values (in [0, 1]) are first put on a grid of 2^-k, k = floor((53 - log2 n) / 2), on which
    every sum of n values and of their squares is exact in float64, whatever the order of summation:
    labellings with the same cases then give the same statistic, so ties count as ties. Without it,
    matrix products round sums differently from shuffle to shuffle, and genes with many equal values
    (sparse genes) lose ties and reject too often in the far tail."""
    cond = np.ascontiguousarray(cond, np.uint8)
    n1, n0 = int(cond.sum()), int((cond == 0).sum())
    grid = 2.0 ** ((53 - np.log2(len(cond))) // 2)
    y = np.round(np.asarray(y, float) * grid) / grid
    y2, total, total2 = y * y, y.sum(axis=1), (y * y).sum(axis=1)

    def statistic(rows, labels):
        chosen = labels.T.astype(float)
        s1, q1 = y[rows] @ chosen, y2[rows] @ chosen
        m1, m0 = s1 / n1, (total[rows, None] - s1) / n0
        v1 = np.maximum(q1 - n1 * m1 ** 2, 0) / (n1 - 1)
        v0 = np.maximum(total2[rows, None] - q1 - n0 * m0 ** 2, 0) / (n0 - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.abs(m1 - m0) / np.sqrt(v1 / n1 + v0 / n0)
        return np.where(np.isnan(t), 0.0, t)

    def tally(labels, observed, exceed, tried, stop, maximum):
        live = np.flatnonzero((exceed < stop) & (tried < maximum))
        mark = observed[live, None]
        hits = statistic(live, labels) >= np.where(np.isfinite(mark), mark - TIE_RTOL * mark, mark)
        running = exceed[live, None].astype(np.int64) + np.cumsum(hits, axis=1)
        first = np.argmax(running >= stop, axis=1)
        used = np.where(running[np.arange(len(live)), first] >= stop, first + 1, labels.shape[0])
        used = np.minimum(used, (maximum - tried[live]).astype(np.int64))
        exceed, tried = exceed.copy(), tried.copy()
        exceed[live] = running[np.arange(len(live)), used - 1]
        tried[live] += used.astype(np.uint64)
        return exceed, tried

    return shuffled(tally, statistic(np.arange(len(y)), cond[None, :])[:, 0], cond, seed)


def midrank_scores(counts, factors):
    """Count-exact mid-rank scores: sample j's chance that its count, thinned to its gene's smallest
    exposure, exceeds another sample's thinned count, plus half the chance of a tie, averaged over the
    other samples. Under f = 1 every score has mean 1/2 whatever the exposures; the scores do not change
    under shuffles. Exact binomial sums, one gene at a time (research speed)."""
    n = counts.shape[1]
    out = np.empty(counts.shape)
    for g, (c, s) in enumerate(zip(counts, factors)):
        pmf = binom.pmf(np.arange(c.max() + 1)[None, :], c[:, None], (s.min() / s)[:, None])
        below = np.cumsum(pmf, axis=1) - pmf
        out[g] = (pmf * (below.sum(0) - below + 0.5 * (pmf.sum(0) - pmf))).sum(1) / (n - 1)
    return out


def pairwise_scores(counts, factors, block=64):
    """Count-exact mid-rank scores with each pair compared at its own common depth: the shallower sample's
    count against the deeper one's thinned to the shallower exposure (one binomial per pair), ties half.
    Under f = 1 each comparison has mean 1/2 whatever the exposures; a sample's score is its mean over
    the other samples, and does not change under shuffles. With B the thinned count's distribution
    function, the shallower sample beats the deeper with chance (B(C - 1) + B(C)) / 2."""
    def distribution(k, trials, keep):                                      # P(Binomial(trials, keep) <= k)
        return np.where(k < 0, 0.0, np.where(k >= trials, 1.0, bdtr(np.clip(k, 0, None), trials, keep)))

    n = counts.shape[1]
    out = np.empty(counts.shape)
    for start in range(0, len(counts), block):
        c, s = counts[start:start + block].astype(np.int64), factors[start:start + block]
        shallow = s[:, :, None] <= s[:, None, :]                            # i no deeper than j
        ratio = np.where(shallow, s[:, :, None] / s[:, None, :], 1.0)
        mine = np.broadcast_to(c[:, :, None], shallow.shape)
        deep = np.broadcast_to(c[:, None, :], shallow.shape)
        beats = 0.5 * (distribution(mine - 1, deep, ratio) + distribution(mine, deep, ratio))   # i beats j where shallower
        h = np.where(shallow, beats, 1 - np.swapaxes(beats, 1, 2))
        out[start:start + block] = (h.sum(axis=2) - 0.5) / (n - 1)
    return out


def zero_chance_test(counts, factors, cond, seed):
    """The zero-chance DE test: exposures where the pooled mean zero chance is 1/2 (the smallest one
    where it stays above), chosen without the labels."""
    e = exposure_at(counts, factors, np.full(len(counts), 0.5))
    return welch_test(zero_chance(counts, factors, e[:, None]), cond, seed)


# ---------------------------------------------------------------------------------------------
# The reference

def tumours():
    """The tumours' raw counts (integral floats in the bundle)."""
    d = tcga("prostate")
    return d["raw"][:, d["cond"] == 1].astype(np.int64)


def halves():
    return (np.random.default_rng(2026092907).random(len(tcga("prostate")["raw"])) < 0.5).astype(int)


def levels():
    """Each gene's mean count over all the tumours at the median depth."""
    d = tcga("prostate")
    tumour = d["cond"] == 1
    return (d["raw"][:, tumour] / d["factors"][tumour]).mean(axis=1)


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


def run(design, depth, fold, exposure, seed=1):
    """Every test's p-values on one data set (NaN for genes the package would not test), and the
    median log2 ratio of the groups' normalized means over the genes of the top level bin."""
    counts, factors, cond = dataset(design, depth, fold, exposure, seed)
    p1, p0 = shares(fold, exposure)
    top = levels() * depth * min(p1, p0) >= EDGES[-1]
    p = {t: np.full(len(counts), np.nan) for t in TESTS}
    ratios = []
    tested = np.flatnonzero(counts.any(axis=1))
    for start in range(0, len(tested), CHUNK):
        rows = tested[start:start + CHUNK]
        c, sf = counts[rows], factors[rows]
        norm = c / sf
        ratios.append(np.log2(norm[:, cond == 1].mean(1) / norm[:, cond == 0].mean(1))[top[rows]])
        x = expression(jittered(c, rows, seed), sf, "linear")
        p["subset"][rows] = subset_test(x, cond, "linear", direction="both", stop=STOP, maximum=MAXIMUM, seed=seed)[0]
        local = np.random.default_rng([seed, 7, start])
        variants = (("thinned", thin(c, cond, np.full(len(rows), fold), local), sf),
                    ("fixed", thin(c, cond, fixed_point(c, cond, sf, rows, seed), local), sf),
                    ("equalized", equalize(c, sf, cond, fold, local), np.ones(len(cond))))
        for name, thinned, scale in variants:
            p[name][rows] = plain_subset_test(expression(jittered(thinned, rows, seed), scale, "linear"), cond, seed)
        if fold == 1:
            p["mean"][rows] = mean_pvalues(x, cond, seed)
            p["rank"][rows] = rank_test(x, cond, seed)
            equalized = expression(jittered(equalize(c, sf, cond, 1.0, local), rows, seed), np.ones(len(cond)), "linear")
            p["rank, equalized"][rows] = rank_test(equalized, cond, seed)
    return p, float(np.median(np.concatenate(ratios))), None


def run_laplace(design, depth, fold, exposure, seed=1):
    """The Laplace-fold constructions on run()'s data sets: their p-values, the same check, and each
    gene's Laplace fold."""
    counts, factors, cond = dataset(design, depth, fold, exposure, seed)
    p1, p0 = shares(fold, exposure)
    top = levels() * depth * min(p1, p0) >= EDGES[-1]
    p = {t: np.full(len(counts), np.nan) for t in LAPLACE}
    folds, ratios = np.full(len(counts), np.nan), []
    tested = np.flatnonzero(counts.any(axis=1))
    for start in range(0, len(tested), CHUNK):
        rows = tested[start:start + CHUNK]
        c, sf = counts[rows], factors[rows]
        norm = c / sf
        ratios.append(np.log2(norm[:, cond == 1].mean(1) / norm[:, cond == 0].mean(1))[top[rows]])
        folds[rows] = f = laplace_fold(c, sf, cond)
        local = np.random.default_rng([seed, 13, start])
        variants = (("laplace", thin(c, cond, f, local), sf),
                    ("laplace, equalized", equalize(c, sf, cond, f, local), np.ones(len(cond))))
        for name, thinned, scale in variants:
            p[name][rows] = plain_subset_test(expression(jittered(thinned, rows, seed), scale, "linear"), cond, seed)
        if fold == 1:
            p["zero chance"][rows] = zero_chance_test(c, sf, cond, seed)
    return p, float(np.median(np.concatenate(ratios))), folds


def share(p, alpha):
    ok = np.isfinite(p)
    return np.mean(p[ok] <= alpha) if ok.any() else np.nan


def reference(designs=DESIGNS, depths=DEPTHS, arms=ARMS, runner=run, tests=TESTS, name="reference"):
    labels = [f"[{lo:g},{hi:g})" for lo, hi in zip((0,) + EDGES, EDGES + (np.inf,))]
    print(f"\n== {name}: the TCGA prostate tumours, a random split per gene, thinned by design; up to {MAXIMUM:,} "
          f"shuffles. Shares at p <= 0.05 / 0.01 by the sparser group's expected count per sample, then over all "
          f"tested genes at 0.05 / 0.01 / 0.001 (binomial standard errors over 24,000 genes: 0.14%, 0.064%, 0.020%)",
          flush=True)
    level = levels()
    for design in designs:
        for depth in depths:
            for fold, exposure in arms:
                start = time.time()
                p1, p0 = shares(fold, exposure)
                bins = np.digitize(level * depth * min(p1, p0), EDGES)
                p, check, folds = runner(design, depth, fold, exposure)
                tested = np.isfinite(p[tests[0]])
                print(f"\n{design[0]}:{design[1]}  depth 1/{round(1 / depth)}  f {fold:g}  d {exposure:g}  "
                      f"(cases keep {p1:g}, controls {p0:g}; median log2 mean ratio at level >= {EDGES[-1]:g}: "
                      f"{check:.3f}, design {np.log2(fold):.3f})  [{time.time() - start:.0f}s]", flush=True)
                print(f"  {'genes':18s} " + " | ".join(f"{lab} {np.sum(tested & (bins == b))}" for b, lab in enumerate(labels))
                      + f" | all {int(tested.sum())}", flush=True)
                if folds is not None:
                    print(f"  {'median log2 fold':18s} " + " | ".join(
                        f"{np.median(np.log2(folds[tested & (bins == b)])):+12.3f}" if np.any(tested & (bins == b))
                        else f"{'--':>12s}" for b in range(len(labels))), flush=True)
                for test in tests:
                    if not np.isfinite(p[test]).any():
                        continue
                    cells = [(share(p[test][bins == b], 0.05), share(p[test][bins == b], 0.01)) for b in range(len(labels))]
                    overall = " / ".join(f"{share(p[test], a):.2%}" if a > 0.001 else f"{share(p[test], a):.3%}"
                                         for a in (0.05, 0.01, 0.001))
                    print(f"  {test:18s} " + " | ".join(f"{'--':>12s}" if np.isnan(a) else f"{a:5.1%} /{b:5.1%}"
                                                         for a, b in cells) + f" | all {overall}", flush=True)


def splits(count=4):
    print(f"\n== splits: the TCGA prostate tumours at full depth, unthinned, the genes of one half; {count} random "
          f"222:223 splits each shared by every gene, then a random split per gene. With no thinning both tests are exact "
          f"permutation tests. Share at p <= 0.05 / 0.01 for the subset statistic (no registration) | the rank test",
          flush=True)
    raw, half = tumours(), halves()
    rows = np.flatnonzero((half == 0) & raw.any(axis=1))
    rle = fit_normalization(raw[half == 1]).size_factors
    cond = np.r_[np.ones(222, np.uint8), np.zeros(223, np.uint8)]
    picks = [np.tile(np.random.default_rng([2026092908, s]).permutation(raw.shape[1]), (len(rows), 1)) for s in range(count)]
    picks.append(np.argsort(np.random.default_rng([2026092908, count]).random((len(rows), raw.shape[1])), axis=1))
    for s, pick in enumerate(picks):
        x = expression(jittered(np.take_along_axis(raw[rows], pick, axis=1), rows, 1), rle[pick], "linear")
        a, b = plain_subset_test(x, cond, 1), rank_test(x, cond, 1)
        label = f"shared split {s + 1}" if s < count else "a split per gene"
        print(f"{label:18s} {len(rows)} genes: {np.mean(a <= .05):.2%} / {np.mean(a <= .01):.2%} | "
              f"{np.mean(b <= .05):.2%} / {np.mean(b <= .01):.2%}", flush=True)


def laplace():
    reference(runner=run_laplace, tests=LAPLACE, name="laplace")


def power(designs=DESIGNS, depths=(1, 1 / 8), every=4):
    """Subsets planted by thinning, so the alternative is itself the count model: sample j keeps its
    pattern multiplier over the largest of its reads (the study's patterns: the subset's cases at
    `members`, the other cases at `majority`, the controls at 1). Every planted gene runs all its
    shuffles, so only every `every`-th gene is tested."""
    print(f"\n== power: subsets planted by thinning the TCGA prostate tumours (a split per gene; every {every}th gene); "
          f"share at p <= 0.001 (up to {MAXIMUM:,} shuffles) by the controls' expected count per sample", flush=True)
    level = levels()
    labels = [f"[{lo:g},{hi:g})" for lo, hi in zip((0,) + EDGES, EDGES + (np.inf,))]
    for design in designs:
        for label, share, members, majority in PATTERNS:
            for depth in depths:
                start = time.time()
                counts, factors, cond = dataset(design, depth, 1.0, 1.0, 5)
                k = max(1, round(share * design[0]))
                mult = np.where(cond == 1, majority, 1.0)
                mult[:k] = members                              # each gene's first k cases: random tumours
                counts = np.random.default_rng([2026092909, *design, k]).binomial(counts, mult / mult.max())
                rows = np.flatnonzero(counts.any(axis=1))
                rows = rows[rows % every == 0]
                bins = np.digitize(level[rows] * depth / mult.max(), EDGES)
                c, sf = counts[rows], factors[rows]
                rng = np.random.default_rng([2026092910, *design, k])
                x = expression(jittered(c, rows, 1), sf, "linear")
                f = laplace_fold(c, sf, cond)
                p = dict(subset=subset_test(x, cond, "linear", direction="both", stop=STOP, maximum=MAXIMUM, seed=1)[0],
                         laplace=plain_subset_test(expression(jittered(thin(c, cond, f, rng), rows, 1), sf, "linear"), cond, 1))
                p["laplace, equalized"] = plain_subset_test(
                    expression(jittered(equalize(c, sf, cond, f, rng), rows, 1), np.ones(len(cond)), "linear"), cond, 1)
                print(f"\n{design[0]}:{design[1]}  depth 1/{round(1 / depth)}  {label} ({k} cases)  [{time.time() - start:.0f}s]", flush=True)
                print(f"  {'genes':18s} " + " | ".join(f"{lab} {np.sum(bins == b)}" for b, lab in enumerate(labels)), flush=True)
                for name, v in p.items():
                    print(f"  {name:18s} " + " | ".join(f"{np.mean(v[bins == b] <= 1e-3):12.1%}" if np.any(bins == b)
                                                        else f"{'--':>12s}" for b in range(len(labels))), flush=True)


def scores(designs=DESIGNS, depth=1 / 8, arms=((1, 1), (1, 0.5), (1, 2), (1.5, 1), (2, 1))):
    """The DE tests on the genes whose sparser group expects under 8 counts per sample, where they part:
    under no fold with equal and unequal exposures (their null), and under folds with equal exposures,
    where the jittered rank test is valid too (power)."""
    print(f"\n== scores: the DE tests at depth 1/{round(1 / depth)} on the genes expecting under {EDGES[2]:g} counts per "
          f"sample in the sparser group (a split per gene); share at p <= 0.05 / 0.01 / 0.001, up to {MAXIMUM:,} shuffles",
          flush=True)
    level = levels()
    for design in designs:
        for fold, exposure in arms:
            start = time.time()
            p1, p0 = shares(fold, exposure)
            counts, factors, cond = dataset(design, depth, fold, exposure, 1)
            rows = np.flatnonzero(counts.any(axis=1) & (level * depth * min(p1, p0) < EDGES[2]))
            c, sf = counts[rows], factors[rows]
            x = expression(jittered(c, rows, 1), sf, "linear")
            thinned = equalize(c, sf, cond, 1.0, np.random.default_rng([2026092911, *design]))
            p = {"rank": rank_test(x, cond, 1),
                 "rank, equalized": rank_test(expression(jittered(thinned, rows, 1), np.ones(len(cond)), "linear"), cond, 1),
                 "zero chance": zero_chance_test(c, sf, cond, 1),
                 "mid-rank scores": welch_test(midrank_scores(c, sf), cond, 1)}
            print(f"{design[0]}:{design[1]}  f {fold:g}  d {exposure:.3g}  {len(rows)} genes  [{time.time() - start:.0f}s]  " + " | ".join(
                f"{name} {share(v, .05):.1%} / {share(v, .01):.1%} / {share(v, .001):.2%}" for name, v in p.items()), flush=True)


def depth_variants(c, sf, cond, rows, rng):
    """The Laplace-fold subset constructions that differ in how they treat depth: the higher group
    only, every sample equalized, and the optimized common depth. Also the share of samples kept."""
    f = laplace_fold(c, sf, cond)
    ones = np.ones(len(cond))
    thinned, kept = optimized(c, sf, cond, f, rng)
    return ({"laplace": plain_subset_test(expression(jittered(thin(c, cond, f, rng), rows, 1), sf, "linear"), cond, 1),
             "laplace, equalized": plain_subset_test(
                 expression(jittered(equalize(c, sf, cond, f, rng), rows, 1), ones, "linear"), cond, 1),
             "laplace, optimized": kept_subset_test(expression(jittered(thinned, rows, 1), ones, "linear"), cond, kept, 1)},
            kept.mean(axis=1))


def depthchoice(designs=((222, 223), (40, 405)), every=4):
    """How the Laplace-fold subset test should treat depth: calibration of the equalized and optimized
    constructions across exposure arms (the higher-group version fails when exposures differ), then
    power on subsets planted by thinning at equal exposures and at the prostate design's imbalance."""
    ratio = cohort_ratio("prostate")
    labels = [f"[{lo:g},{hi:g})" for lo, hi in zip((0,) + EDGES, EDGES + (np.inf,))]
    level = levels()
    print(f"\n== depthchoice, calibration: a split per gene, every {every}th gene, 1/8 depth; share at p <= 0.05 / 0.01 "
          f"by the sparser group's expected count, then overall at 0.05 / 0.01 / 0.001; median share of samples kept "
          f"by the optimized depth", flush=True)
    for design in DESIGNS:
        for fold, exposure in ((1, 1), (2, 1), (1, ratio), (2, ratio), (1, 0.5), (2, 0.5)):
            start = time.time()
            p1, p0 = shares(fold, exposure)
            counts, factors, cond = dataset(design, 1 / 8, fold, exposure, 1)
            rows = np.flatnonzero(counts.any(axis=1))
            rows = rows[rows % every == 0]
            bins = np.digitize(level[rows] / 8 * min(p1, p0), EDGES)
            p, kept = depth_variants(counts[rows], factors[rows], cond, rows, np.random.default_rng([2026093001, *design]))
            print(f"{design[0]}:{design[1]}  f {fold:g}  d {exposure:.3g}  {len(rows)} genes, optimized depth keeps a "
                  f"median {np.median(kept):.0%} of samples  [{time.time() - start:.0f}s]", flush=True)
            for name, v in p.items():
                print(f"  {name:18s} " + " | ".join(f"{share(v[bins == b], .05):5.1%} /{share(v[bins == b], .01):5.1%}"
                                                    if np.any(bins == b) else f"{'--':>12s}" for b in range(len(labels)))
                      + f" | all {share(v, .05):.2%} / {share(v, .01):.2%} / {share(v, .001):.3%}", flush=True)
    print(f"\n== depthchoice, power: subsets planted by thinning (a split per gene, every {every}th gene); share at "
          f"p <= 0.001 by the controls' expected count; 'laplace' (the higher group only) is exact only at d = 1", flush=True)
    for design in designs:
        for exposure in (1.0, ratio):
            for label, share_, members, majority in PATTERNS:
                for depth in (1, 1 / 8):
                    start = time.time()
                    counts, factors, cond = dataset(design, depth, 1.0, exposure, 5)
                    k = max(1, round(share_ * design[0]))
                    mult = np.where(cond == 1, majority, 1.0)
                    mult[:k] = members
                    p0 = shares(1.0, exposure)[1]
                    counts = np.random.default_rng([2026092909, *design, k]).binomial(counts, mult / mult.max())
                    rows = np.flatnonzero(counts.any(axis=1))
                    rows = rows[rows % every == 0]
                    bins = np.digitize(level[rows] * depth * p0 / mult.max(), EDGES)
                    p, kept = depth_variants(counts[rows], factors[rows], cond, rows,
                                             np.random.default_rng([2026092910, *design, k]))
                    print(f"\n{design[0]}:{design[1]}  d {exposure:.3g}  depth 1/{round(1 / depth)}  {label} ({k} cases); "
                          f"optimized depth keeps a median {np.median(kept):.0%} of samples  [{time.time() - start:.0f}s]", flush=True)
                    print(f"  {'genes':18s} " + " | ".join(f"{lab} {np.sum(bins == b)}" for b, lab in enumerate(labels)), flush=True)
                    for name, v in p.items():
                        print(f"  {name:18s} " + " | ".join(f"{np.mean(v[bins == b] <= 1e-3):12.1%}" if np.any(bins == b)
                                                            else f"{'--':>12s}" for b in range(len(labels))), flush=True)


def seeds(design=(222, 223), every=4):
    """How much a thinned test's calls depend on its thinning draw: the equalized Laplace-fold subset test
    run twice with different thinning seeds, on a null and on planted subsets."""
    print(f"\n== seeds: the equalized Laplace-fold subset test with two thinning seeds on the same data ({design[0]}:"
          f"{design[1]}, a split per gene, every {every}th gene). Share called at p <= 0.01 and 0.001 by each seed; the "
          f"share of genes called by either seed that both call; Spearman correlation of log p over all genes", flush=True)
    from scipy.stats import spearmanr
    cases = [("null", None), *[(label, pattern) for label, *pattern in PATTERNS]]
    for depth in (1, 1 / 8):
        for label, pattern in cases:
            counts, factors, cond = dataset(design, depth, 1.0, 1.0, 5)
            if pattern is not None:
                share_, members, majority = pattern
                k = max(1, round(share_ * design[0]))
                mult = np.where(cond == 1, majority, 1.0)
                mult[:k] = members
                counts = np.random.default_rng([2026092909, *design, k]).binomial(counts, mult / mult.max())
            rows = np.flatnonzero(counts.any(axis=1))
            rows = rows[rows % every == 0]
            c, sf = counts[rows], factors[rows]
            f = laplace_fold(c, sf, cond)
            p = [plain_subset_test(expression(jittered(equalize(c, sf, cond, f, np.random.default_rng([2026093002, draw])),
                                                        rows, 1), np.ones(len(cond)), "linear"), cond, 1) for draw in (1, 2)]
            cells = []
            for alpha in (0.01, 0.001):
                a, b = p[0] <= alpha, p[1] <= alpha
                both = np.sum(a & b) / max(np.sum(a | b), 1)
                cells.append(f"p <= {alpha:g}: {a.mean():.1%} and {b.mean():.1%}, both {both:.0%}")
            rho = spearmanr(np.log(p[0]), np.log(p[1])).statistic
            print(f"depth 1/{round(1 / depth)}  {label:36s} " + " | ".join(cells) + f" | rho {rho:.3f}", flush=True)


def pairwise(designs=DESIGNS, depth=1 / 8, every=4):
    """Mid-rank scores compared at the smallest depth against each pair at its own depth, with the
    jittered rank test for reference: every 4th gene at every level (the smallest-depth scores only
    below 8 expected counts, where they are cheap). Share at p <= 0.05 / 0.01 / 0.001."""
    ratio = cohort_ratio("prostate")
    labels = [f"[{lo:g},{hi:g})" for lo, hi in zip((0,) + EDGES, EDGES + (np.inf,))]
    level = levels()
    print(f"\n== pairwise: DE scores at depth 1/{round(1 / depth)}, a split per gene, every {every}th gene; share at "
          f"p <= 0.05 / 0.01 / 0.001 below and at or above {EDGES[2]:g} expected counts in the sparser group", flush=True)
    for design in designs:
        for fold, exposure in ((1, 1), (1, ratio), (1, 0.5), (1, 2), (1.5, 1), (2, 1)):
            start = time.time()
            p1, p0 = shares(fold, exposure)
            counts, factors, cond = dataset(design, depth, fold, exposure, 1)
            rows = np.flatnonzero(counts.any(axis=1))
            rows = rows[rows % every == 0]
            low = (level[rows] * depth * min(p1, p0)) < EDGES[2]
            c, sf = counts[rows], factors[rows]
            p = {"rank": rank_test(expression(jittered(c, rows, 1), sf, "linear"), cond, 1),
                 "scores, pairwise": welch_test(pairwise_scores(c, sf), cond, 1),
                 "scores, smallest": np.full(len(rows), np.nan)}
            p["scores, smallest"][low] = welch_test(midrank_scores(c[low], sf[low]), cond, 1)
            print(f"{design[0]}:{design[1]}  f {fold:g}  d {exposure:.3g}  {low.sum()} + {(~low).sum()} genes  "
                  f"[{time.time() - start:.0f}s]  " + " || ".join(
                      f"{name} " + " | ".join(f"{share(v[k], .05):.1%} / {share(v[k], .01):.1%} / {share(v[k], .001):.2%}"
                                              if np.isfinite(v[k]).any() else "--" for k in (low, ~low))
                      for name, v in p.items()), flush=True)


def adopted(c, sf, cond, rows, seed, fold=None, de=True):
    """The adopted constructions (plan, decisions 10-13) on one gene block: the subset test thins every
    sample to its gene's shallowest effective depth by the Laplace fold (or `fold`) and shuffles the Lorenz
    statistic, with and without jitter; the DE test shuffles the Welch-studentized pairwise mid-rank scores."""
    f = laplace_fold(c, sf, cond) if fold is None else np.broadcast_to(np.asarray(fold, float), len(c))
    thinned = equalize(c, sf, cond, f, np.random.default_rng([seed, 17]))
    ones = np.ones(len(cond))
    p = {"subset": plain_subset_test(expression(jittered(thinned, rows, seed), ones, "linear"), cond, seed),
         "subset, no jitter": plain_subset_test(thinned.astype(float), cond, seed)}
    if de:
        p["DE scores"] = welch_test(pairwise_scores(c, sf), cond, seed)
    return p


def small(designs=((20, 20), (50, 50), (10, 100), (100, 10)), depths=(1, 1 / 8)):
    """The adopted constructions in small designs on the real reference, across exposure arms. The DE
    scores are shown at f = 1 only (their null)."""
    ratio = cohort_ratio("prostate")
    labels = [f"[{lo:g},{hi:g})" for lo, hi in zip((0,) + EDGES, EDGES + (np.inf,))]
    level = levels()
    print(f"\n== small: the adopted constructions (Laplace fold, every sample equalized; pairwise DE scores) in small "
          f"designs, a split per gene; share at p <= 0.05 / 0.01 by the sparser group's expected count, then overall at "
          f"0.05 / 0.01 / 0.001", flush=True)
    for design in designs:
        for depth in depths:
            for fold, exposure in ((1, 1), (2, 1), (1, ratio), (2, ratio), (1, 0.5), (2, 0.5)):
                start = time.time()
                p1, p0 = shares(fold, exposure)
                counts, factors, cond = dataset(design, depth, fold, exposure, 1)
                rows = np.flatnonzero(counts.any(axis=1))
                bins = np.digitize(level[rows] * depth * min(p1, p0), EDGES)
                p = adopted(counts[rows], factors[rows], cond, rows, 1, de=fold == 1)
                print(f"{design[0]}:{design[1]}  depth 1/{round(1 / depth)}  f {fold:g}  d {exposure:.3g}  {len(rows)} genes  "
                      f"[{time.time() - start:.0f}s]", flush=True)
                for name, v in p.items():
                    print(f"  {name:18s} " + " | ".join(f"{share(v[bins == b], .05):5.1%} /{share(v[bins == b], .01):5.1%}"
                                                        if np.sum(bins == b) >= 200 else f"{'--':>12s}" for b in range(len(labels)))
                          + f" | all {share(v, .05):.2%} / {share(v, .01):.2%} / {share(v, .001):.3%}", flush=True)


def bimodal(designs=((20, 20), (50, 50), (250, 250)), genes=50_000, sd=0.3, separation=2.0, mean=3.0):
    """Review 2's bimodal null under the count model: latent levels an equal mixture of lognormal(mean, sd^2)
    and lognormal(mean + separation, sd^2), Poisson counts at size factor 1, labels exchangeable. Every
    permutation test without registration is exact here; the question is whether estimating the Laplace
    fold (and thinning by it) reintroduces the registered test's excess (plan, section 3.4). The oracle
    thins by the true fold, 1: no thinning at all."""
    print(f"\n== bimodal: {genes:,} genes per design, latent levels an equal mixture of lognormal({mean:g}, {sd:g}^2) and "
          f"lognormal({mean + separation:g}, {sd:g}^2), Poisson counts; share at p <= 0.05 / 0.01 / 0.001", flush=True)
    for design in designs:
        start = time.time()
        rng = np.random.default_rng([2026093003, *design])
        n = sum(design)
        modes = rng.random((genes, n)) < 0.5
        counts = rng.poisson(np.exp(mean + separation * modes + sd * rng.normal(size=(genes, n))))
        cond = np.r_[np.ones(design[0], np.uint8), np.zeros(design[1], np.uint8)]
        ones = np.ones((genes, n))
        rows = np.arange(genes)
        p = adopted(counts, ones, cond, rows, 1, de=False)
        p["subset, oracle fold"] = adopted(counts, ones, cond, rows, 1, fold=1.0, de=False)["subset"]
        x = expression(jittered(counts, rows, 1), np.ones(n), "linear")
        p["package subset"] = subset_test(x, cond, "linear", direction="both", stop=STOP, maximum=MAXIMUM, seed=1)[0]
        print(f"{design[0]}:{design[1]}  [{time.time() - start:.0f}s]  " + " | ".join(
            f"{name} {share(v, .05):.2%} / {share(v, .01):.2%} / {share(v, .001):.3%}" for name, v in p.items()), flush=True)


def w1_equalized(c, sf, cond, rows, seed, rounds=5):
    """A robust alternative: thin every sample to its gene's shallowest exposure by its size factor alone,
    then find the fold on the equalized counts by the W1 fixed point (thin the higher group by the current
    fold, register the thinned counts by W1, multiply), thin by it, and shuffle the Lorenz statistic."""
    rng = np.random.default_rng([seed, 19])
    ones = np.ones(len(cond))
    level = equalize(c, sf, cond, 1.0, rng)
    fold = fixed_point(level, cond, ones, rows, seed, rounds=rounds)
    return plain_subset_test(expression(jittered(thin(level, cond, fold, rng), rows, seed), ones, "linear"), cond, seed)


def robust(designs=((10, 40), (20, 20), (100, 100)), splits=5000, genes=("GSTM1", "UGT2B17", "PSPHP1", "SOX14", "SMN2")):
    """The W1 fixed point on equalized counts against the adopted Laplace construction: zero-inflated real
    genes under random splits, review 2's bimodal simulation, and the reference's low-count fold cells."""
    d = tcga("prostate")
    tumour = d["cond"] == 1
    print(f"\n== robust: share at p <= 0.05 / 0.01 / 0.001 for the adopted construction (Laplace fold, equalized) | the W1 "
          f"fixed point on equalized counts | the oracle (true fold)", flush=True)
    for gene in genes:
        raw = d["raw"][d["names"] == gene][0][tumour].astype(np.int64)
        factors = d["factors"][tumour]
        for design in designs:
            rng = np.random.default_rng([2026093004, *design])
            pick = np.stack([rng.choice(len(raw), sum(design), replace=False) for _ in range(splits)])
            c, sf = raw[pick], factors[pick]
            sf = sf / np.exp(np.log(sf).mean(axis=1, keepdims=True))
            cond = np.r_[np.ones(design[0], np.uint8), np.zeros(design[1], np.uint8)]
            rows = np.arange(splits)
            p = [adopted(c, sf, cond, rows, 1, de=False)["subset"], w1_equalized(c, sf, cond, rows, 1),
                 adopted(c, sf, cond, rows, 1, fold=1.0, de=False)["subset"]]
            print(f"{gene:8s} {design[0]:3d}:{design[1]:<3d} {splits:,} splits  " + " | ".join(
                f"{share(v, .05):.2%} / {share(v, .01):.2%} / {share(v, .001):.3%}" for v in p), flush=True)
    for design in ((20, 20), (50, 50), (250, 250)):
        rng = np.random.default_rng([2026093003, *design])
        n, genes_ = sum(design), 50_000
        modes = rng.random((genes_, n)) < 0.5
        counts = rng.poisson(np.exp(3.0 + 2.0 * modes + 0.3 * rng.normal(size=(genes_, n))))
        cond = np.r_[np.ones(design[0], np.uint8), np.zeros(design[1], np.uint8)]
        ones, rows = np.ones((genes_, n)), np.arange(genes_)
        p = [adopted(counts, ones, cond, rows, 1, de=False)["subset"], w1_equalized(counts, ones, cond, rows, 1),
             adopted(counts, ones, cond, rows, 1, fold=1.0, de=False)["subset"]]
        print(f"bimodal  {design[0]:3d}:{design[1]:<3d} {genes_:,} genes  " + " | ".join(
            f"{share(v, .05):.2%} / {share(v, .01):.2%} / {share(v, .001):.3%}" for v in p), flush=True)
    level, labels = levels(), [f"[{lo:g},{hi:g})" for lo, hi in zip((0,) + EDGES, EDGES + (np.inf,))]
    ratio = cohort_ratio("prostate")
    for design, depth, fold, exposure in (((222, 223), 1 / 8, 2, 1), ((222, 223), 1 / 8, 4, 1), ((40, 405), 1 / 8, 2, ratio),
                                          ((222, 223), 1 / 64, 2, 0.5), ((405, 40), 1, 2, 1)):
        p1, p0 = shares(fold, exposure)
        counts, factors, cond = dataset(design, depth, fold, exposure, 1)
        rows = np.flatnonzero(counts.any(axis=1))
        rows = rows[rows % 4 == 0]
        bins = np.digitize(level[rows] * depth * min(p1, p0), EDGES)
        c, sf = counts[rows], factors[rows]
        p = [adopted(c, sf, cond, rows, 1, de=False)["subset"], w1_equalized(c, sf, cond, rows, 1),
             adopted(c, sf, cond, rows, 1, fold=fold, de=False)["subset"]]
        print(f"reference {design[0]}:{design[1]} depth 1/{round(1 / depth)} f {fold:g} d {exposure:.3g}, every 4th gene; "
              f"share at p <= 0.01 by level {' | '.join(labels)}", flush=True)
        for name, v in zip(("adopted", "W1 fixed point", "oracle"), p):
            print(f"  {name:15s} " + " | ".join(f"{share(v[bins == b], .01):6.1%}" if np.sum(bins == b) >= 200 else f"{'--':>6s}"
                                               for b in range(len(labels))) + f" | all {share(v, .05):.2%} / {share(v, .01):.2%} / {share(v, .001):.3%}", flush=True)


def realbimodal(splits=5000, designs=((20, 20), (10, 40), (40, 10), (50, 50), (100, 100))):
    """Real genes made bimodal by germline genotype (GSTM1, a common whole-gene deletion; ERAP2, a common
    allele whose transcripts are degraded) under random splits of the TCGA prostate tumours, each split a
    row: the adopted construction, its oracle (every sample equalized by its size factor alone: exact),
    and the package's subset test. Share of splits at p <= 0.05 / 0.01 / 0.001."""
    d = tcga("prostate")
    tumour = d["cond"] == 1
    print(f"\n== realbimodal: TCGA prostate tumours, {splits:,} random splits per design; share of splits at "
          f"p <= 0.05 / 0.01 / 0.001", flush=True)
    for gene in ("GSTM1", "ERAP2"):
        raw = d["raw"][d["names"] == gene][0][tumour].astype(np.int64)
        factors = d["factors"][tumour]
        print(f"{gene}: counts over the tumours from {raw.min()} to {raw.max()}, median {np.median(raw):.0f}", flush=True)
        for design in designs:
            rng = np.random.default_rng([2026093004, *design])
            pick = np.stack([rng.choice(len(raw), sum(design), replace=False) for _ in range(splits)])
            c, sf = raw[pick], factors[pick]
            sf = sf / np.exp(np.log(sf).mean(axis=1, keepdims=True))
            cond = np.r_[np.ones(design[0], np.uint8), np.zeros(design[1], np.uint8)]
            rows = np.arange(splits)
            p = {"adopted": adopted(c, sf, cond, rows, 1, de=False)["subset"],
                 "oracle": adopted(c, sf, cond, rows, 1, fold=1.0, de=False)["subset"],
                 "package": subset_test(expression(jittered(c, rows, 1), sf, "linear"), cond, "linear", direction="both",
                                        stop=STOP, maximum=MAXIMUM, seed=1)[0]}
            print(f"  {design[0]:3d}:{design[1]:<3d} " + " | ".join(
                f"{name} {share(v, .05):.2%} / {share(v, .01):.2%} / {share(v, .001):.3%}" for name, v in p.items()), flush=True)


def composition(shares_=(0.1, 0.3), splits=4, fold=2.0, depth=1 / 8, design=(222, 223), every=8):
    """The normalization under changes of composition. One split shared by every gene (a sample's size
    factor depends on its label across genes), a share of the genes changed by `fold` (all up in the cases,
    or half up and half down), 1/8 depth; the DE scores and the package's subset test on every `every`-th
    unchanged gene, with RLE refitted on the data set against the design's own size factors (RLE of the
    unthinned tumours). Shares at p <= 0.05 / 0.01, pooled over the splits, and the median case-over-control
    ratio of refitted to design size factors (1 = no composition bias)."""
    raw, level = tumours(), levels()
    base = fit_normalization(raw).size_factors
    cond = np.r_[np.ones(design[0], np.uint8), np.zeros(design[1], np.uint8)]
    print(f"\n== composition: {design[0]}:{design[1]}, one split shared by every gene ({splits} splits), depth 1/"
          f"{round(1 / depth)}, a share of genes changed {fold:g}x; every {every}th unchanged gene; share at p <= 0.05 / 0.01 "
          f"pooled over the splits", flush=True)
    for share_ in shares_:
        for balanced in (False, True):
            start = time.time()
            p = {k: [] for k in ("scores, design factors", "scores, refitted RLE", "subset, design factors",
                                  "subset, refitted RLE")}
            bias, big = [], []
            for split_index in range(splits):
                rng = np.random.default_rng([2026093005, round(100 * share_), balanced, split_index])
                pick = rng.permutation(raw.shape[1])[:sum(design)]
                counts = rng.binomial(raw[:, pick], depth)
                changed = rng.random(len(raw)) < share_
                up = changed & ((rng.random(len(raw)) < 0.5) if balanced else True)
                down = changed & ~up
                assert up.dtype == bool and down.dtype == bool
                counts[np.ix_(up, cond == 0)] = rng.binomial(counts[np.ix_(up, cond == 0)], 1 / fold)
                counts[np.ix_(down, cond == 1)] = rng.binomial(counts[np.ix_(down, cond == 1)], 1 / fold)
                design_sf = base[pick] / np.exp(np.log(base[pick]).mean())
                refit = fit_normalization(counts).size_factors
                ratio = refit / design_sf
                bias.append(np.median(ratio[cond == 1]) / np.median(ratio[cond == 0]))
                rows = np.flatnonzero(~changed & counts.any(axis=1))
                rows = rows[rows % every == 0]
                big.append(level[rows] * depth >= 128)
                for label, sf in (("design factors", design_sf), ("refitted RLE", refit)):
                    grid = np.broadcast_to(sf, (len(rows), len(sf)))
                    p[f"scores, {label}"].append(welch_test(pairwise_scores(counts[rows], grid), cond, 1))
                    x = expression(jittered(counts[rows], rows, 1), sf, "linear")
                    p[f"subset, {label}"].append(subset_test(x, cond, "linear", direction="both", stop=STOP,
                                                             maximum=MAXIMUM, seed=1)[0])
            big = np.concatenate(big)
            print(f"{share_:.0%} of genes changed, {'half up, half down' if balanced else 'all up'}: refitted over design "
                  f"size factors, cases over controls, {', '.join(f'{b:.3f}' for b in bias)}  [{time.time() - start:.0f}s]", flush=True)
            for name, v in p.items():
                v = np.concatenate(v)
                subset_note = f" | at >= 128 expected reads {share(v[big], .05):.1%} / {share(v[big], .01):.1%}" if name.startswith("subset") else ""
                print(f"  {name:24s} all unchanged genes {share(v, .05):.1%} / {share(v, .01):.1%}{subset_note}", flush=True)
            for test in ("scores", "subset"):
                a, b = np.concatenate(p[f"{test}, design factors"]), np.concatenate(p[f"{test}, refitted RLE"])
                print(f"  {test}: refitted minus design, paired on the same genes and splits: "
                      f"{share(b, .05) - share(a, .05):+.1%} / {share(b, .01) - share(a, .01):+.1%}", flush=True)


def cohort_ratio(cancer):
    """A TCGA cohort's median tumour size factor over its median normal one: the exposure imbalance its
    own design carries."""
    d = tcga(cancer)
    return float(np.median(d["factors"][d["cond"] == 1]) / np.median(d["factors"][d["cond"] == 0]))


def realistic(designs=((222, 223), (40, 405)), depths=(1 / 8, 1 / 64)):
    """The grid's tests at the exposure imbalance of the TCGA prostate and breast designs."""
    arms = tuple((fold, cohort_ratio(cancer)) for cancer in ("prostate", "breast") for fold in (1, 2))
    print(f"\n== realistic: d is the TCGA prostate and breast designs' median tumour over median normal size factor "
          f"({', '.join(f'{a[1]:.3f}' for a in arms[::2])})", flush=True)
    reference(designs=designs, depths=depths, arms=arms, name="realistic")
    reference(designs=designs, depths=depths, arms=arms, runner=run_laplace, tests=LAPLACE, name="realistic, laplace")
    scores(designs=designs, depth=min(depths), arms=arms[::2])


def cells_in(path):
    """The cells of a saved reference or laplace output: design, depth, f, d, genes by level, and per
    test its shares at 0.05 / 0.01 by level and overall at 0.05 / 0.01 / 0.001 (percent)."""
    cells = []
    for line in open(path):
        head = re.match(r"^(\d+):(\d+)\s+depth 1/(\d+)\s+f ([\d.]+)\s+d ([\d.]+)", line)
        if head:
            cells.append(dict(design=head[1] + ":" + head[2], depth=int(head[3]), f=float(head[4]), d=float(head[5]), rows={}))
        elif cells and line.strip().startswith("genes"):
            cells[-1]["genes"] = [int(v) for v in re.findall(r"\) (\d+)", line)]
        elif cells and "| all " in line and "%" in line:
            segments = line.rstrip("\n").split("|")
            first = re.search(r"([\d.]+%\s*/\s*[\d.]+%|--)\s*$", segments[0])
            levels = [None if "--" in c else tuple(float(v.strip(" %")) for v in c.split("/"))
                      for c in [segments[0][first.start():]] + segments[1:-1]]
            overall = segments[-1].replace("all", "")
            cells[-1]["rows"][segments[0][:first.start()].strip()] = (
                levels, [float(v.strip().split()[0].rstrip("%")) for v in overall.split("/")])
    return cells


def summary(paths=(".release/0.2.0/count-model-20260929.txt", ".release/0.2.0/count-model-laplace-20260929.txt"),
            minimum=200):
    """The saved grids condensed: per group of arms and test, the range over cells of the share at
    p <= 0.01 in each level bin holding at least `minimum` genes (binomial standard error at most 0.7%),
    and of the overall share at 0.001."""
    cells = [c for path in paths for c in cells_in(ROOT / path)]
    merged = {}
    for c in cells:
        merged.setdefault((c["design"], c["depth"], c["f"], c["d"]), dict(c, rows={}))["rows"].update(c["rows"])
    groups = (("no fold, equal exposures (f 1, d 1: a random split)", lambda c: c["f"] == 1 and c["d"] == 1),
              ("a fold, equal exposures (f 1.5, 2, 4, 0.5; d 1)", lambda c: c["f"] != 1 and c["d"] == 1),
              ("no fold, unequal exposures (f 1; d 0.5, 2)", lambda c: c["f"] == 1 and c["d"] != 1),
              ("a fold and unequal exposures (f 2; d 0.5, 2)", lambda c: c["f"] != 1 and c["d"] != 1))
    labels = [f"[{lo:g},{hi:g})" for lo, hi in zip((0,) + EDGES, EDGES + (np.inf,))]
    print(f"\n== summary of {', '.join(paths)}: per test, the range over cells (3 designs x 3 depths x the arms) of "
          f"the share at p <= 0.01 by level (bins with >= {minimum} genes), then of the overall share at 0.001. "
          f"Levels: {' | '.join(labels)}")
    for label, chosen in groups:
        print(f"\n{label}")
        for test in TESTS + LAPLACE:
            rows = [c for c in merged.values() if chosen(c) and test in c["rows"]]
            if not rows:
                continue
            spans = []
            for b in range(len(EDGES) + 1):
                v = [c["rows"][test][0][b][1] for c in rows if c["rows"][test][0][b] is not None and c["genes"][b] >= minimum]
                spans.append(f"{min(v):4.1f}-{max(v):4.1f}" if v else f"{'--':>9s}")
            top = [c["rows"][test][1][2] for c in rows]
            print(f"  {test:18s} " + " | ".join(spans) + f" | 0.001: {min(top):.3f}-{max(top):.3f}% ({len(rows)} cells)")


SECTIONS = dict(splits=splits, reference=reference, laplace=laplace, power=power, scores=scores, realistic=realistic, depthchoice=depthchoice, seeds=seeds, pairwise=pairwise, small=small, bimodal=bimodal, realbimodal=realbimodal, robust=robust, composition=composition,
                summary=summary)
DEFAULT = ("splits", "reference")

if __name__ == "__main__":
    for section in sys.argv[1:] or DEFAULT:
        SECTIONS[section]()
