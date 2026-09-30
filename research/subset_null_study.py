"""The subset test's null: the released construction, the derived one, and what each part does.

Supports docs/subset-null-derivation.md; every number there comes from this script's output.
Each construction pairs a registration (the fold that divides the cases before they are pooled
with the controls) with a statistic (the walk's depth, or the depth in units of the median case):

  exact              no registration, depth: plain label permutation, exact when labels are exchangeable
  released           the mean-shift fold, depth: the construction in wade 0.2.0 as built
  W1 + depth         the Wasserstein-1 fold, depth
  mean + scaled      the mean-shift fold, depth in median-case units
  log-mean + scaled  the Wasserstein-2 (geometric-mean) fold, depth in median-case units
  proposed           the Wasserstein-1 fold, depth in median-case units
  lorenz             the Wasserstein-1 fold, depth over the case mean + 1 (the Lorenz-curve gap)
  (later entries: other normalizers and registrations, and 'exact, lorenz', the same-statistic oracle at fold 1)

Sections (default: all, in this order):

  checks      numerical checks of the invariance the derivation relies on
  foldform    between tumour and normal, does a gene's spread scale with its level? (TCGA)
  realnull    random labels on TCGA prostate, three designs, every construction
  genes       single genes' values under 200 random labellings
  simnull     simulated nulls: designs x distributions x uniform folds
  mechanism   registration by the true fold, by the true fold with independent noise, and by estimates
  power       the confirmation generator's planted patterns on fresh seeds, BH q <= 0.05
  truelabels  TCGA prostate and breast with their real labels
  thinning    a uniform 2x change made by binomial thinning of real tumour counts
  lowcount    simulated uniform folds at low counts
  normalizers review 1, recommendation A: other invariant normalizers (not in the default run)
  thinnednull review 1, recommendation B: binomial thinning to a common level (not in the default run)
  thinfixed   thinning by a count-consistent fixed-point fold (not in the default run)
  review2     review 2: bimodal nulls and remedies, small real designs, the Lorenz bootstrap, the mean test
  lorenz      review 2: the Lorenz normalizer (depth over the case mean + 1): power, simulated and real nulls
  bimodalgenes review 2: real genes made bimodal by common germline variants, under random splits
  bergerboos  review 2: the Berger-Boos bound over a rank confidence interval for the registration
  registrations review 2: W1 against the log-median difference and the Hodges-Lehmann shift
  matched     review 2: power at matched actual size, linear and log2
  mechanism2  review 2: which bimodal-null configurations the registration miscalibrates, and why
  budget      review 2: the confirmation generator at 50 v 500 with the package's permutation budget (999,999)
  large250    review 3: the bimodal null at 250 v 250 over 200,000 genes, paired with the same-statistic oracle

    WADE_TCGA_CONFIG=.release/0.2.0/tcga-config.json PYTHONPATH=src:research \\
        python research/subset_null_study.py [section ...]
"""
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import wade
from wade import _kernel
from wade._means import expression, jittered
from wade._subsets import _carry, _sorted
from wade.permutation import draw_perms
from wade.pvalues import bh_adjust, sequential_p
from wade_research.point_simulation import datasets, generate

ROOT = Path(__file__).resolve().parents[1]
STOP = 25                                   # the stopping rule at the default relative_error = 0.2
CONSTRUCTIONS = {
    "exact": ("none", "depth"),
    "released": ("mean", "depth"),
    "W1 + depth": ("w1", "depth"),
    "mean + scaled": ("mean", "scaled"),
    "log-mean + scaled": ("logmean", "scaled"),
    "proposed": ("w1", "scaled"),
    # Review 1, recommendation A and a more efficient scale: the same registration, other invariant normalizers.
    "proposed, geometric-mean scale": ("w1", "geo"),
    "proposed, control-anchored scale": ("w1", "anchored"),
    # Review 1, recommendation B: thin the higher group's counts to the lower's level, then no registration.
    "thinned": ("none", "scaled"),
    # Review 2: the depth over the case mean + 1 is the one-sided Lorenz-curve discrepancy (the package's
    # construction since 29 September 2026). W1 takes the middle of the weighted-median interval, which keeps
    # the registration symmetric under swapping cases and controls; the lower end is kept for comparison.
    "lorenz": ("w1", "lorenz"),
    "proposed, lower-median registration": ("w1low", "scaled"),
    "lorenz, lower-median registration": ("w1low", "lorenz"),
    # Review 2, question 2: other registrations; and the same statistic with no registration, the oracle at fold 1.
    "lorenz, log-median registration": ("logmedian", "lorenz"),
    "lorenz, Hodges-Lehmann registration": ("hl", "lorenz"),
    "exact, lorenz": ("none", "lorenz"),
}
MODES = {"depth": 0, "scaled": 1, "geo": 3, "anchored": 4, "lorenz": 5}


# ---------------------------------------------------------------------------------------------
# The test under each construction

def registration(case, control, scale, how):
    """Each gene's registration: a fold on count + 1 (linear scale) or a difference (log2). Rows sorted."""
    linear = scale == "linear"
    if how == "none":
        return np.ones(len(case)) if linear else np.zeros(len(case))
    if how == "mean":
        m1, m0 = case.mean(axis=1), control.mean(axis=1)
        return (m1 + 1) / (m0 + 1) if linear else m1 - m0
    if how == "logmean":
        if not linear:
            return case.mean(axis=1) - control.mean(axis=1)
        return np.exp(np.log1p(case).mean(axis=1) - np.log1p(control).mean(axis=1))
    if how == "logmedian":
        t = (np.median(np.log1p(case), axis=1) - np.median(np.log1p(control), axis=1) if linear
             else np.median(case, axis=1) - np.median(control, axis=1))
        return np.exp(t) if linear else t
    if how == "hl":                     # the median of all case-minus-control differences of log(x + 1)
        lc, lk = (np.log1p(case), np.log1p(control)) if linear else (case, control)
        t = np.empty(len(case))
        for s in range(0, len(case), 32):
            d = lc[s:s + 32, :, None] - lk[s:s + 32, None, :]
            t[s:s + 32] = np.median(d.reshape(len(d), -1), axis=1)
        return np.exp(t) if linear else t
    if how == "w1":
        return w1_registration(case, control, linear, midpoint=True)
    if how == "w1low":
        return w1_registration(case, control, linear)
    raise ValueError(how)


def w1_registration(case, control, linear, chunk=256, midpoint=False):
    """The registration minimizing the area between the two log(x + 1) (log2: x) quantile functions.

    Both quantile functions are steps; on the merged breakpoints their log ratio a(u) is constant on
    each segment, and the minimizers of the integral of |a(u) - t| over u in (0, 1) are the weighted
    medians of a, weighted by segment length. Breakpoints and weights are integers in units of
    1/(n1 n0), so an exact half split is found exactly: the lower end of the median interval is
    taken, or with `midpoint` its middle (the package's rule, rust/src/lib.rs). Before 29 September
    2026 this used float weights, which at an exact half split picked either end or the middle by
    rounding; any of them minimizes the area.
    """
    n1, n0 = case.shape[1], control.shape[1]
    total = n1 * n0
    edges = np.unique(np.r_[np.arange(1, n1) * n0, np.arange(1, n0) * n1, total])
    width = np.diff(np.r_[0, edges])
    middle = 2 * edges - width                              # twice each segment's midpoint, in units of 1/(n1 n0)
    i, j = middle // (2 * n0), middle // (2 * n1)
    out = np.empty(len(case))
    for s in range(0, len(case), chunk):
        c, k = case[s:s + chunk], control[s:s + chunk]
        a = np.log1p(c[:, i]) - np.log1p(k[:, j]) if linear else c[:, i] - k[:, j]
        order = np.argsort(a, axis=1, kind="stable")
        cum = np.cumsum(width[order], axis=1)
        at = np.argmax(2 * cum >= total, axis=1)
        sorted_a = np.take_along_axis(a, order, axis=1)
        rows = np.arange(len(a))
        value = sorted_a[rows, at]
        if midpoint:
            tie = 2 * cum[rows, at] == total
            nxt = sorted_a[rows, np.minimum(at + 1, sorted_a.shape[1] - 1)]
            value = np.where(tie, 0.5 * (value + nxt), value)
        out[s:s + chunk] = value
    return np.exp(out) if linear else out


def statistic(case, control, scale, scaled):
    """The observed walk: depth, or depth over an invariant normalizer (count + 1 units). Rows sorted.
    `scaled` is False, True (the median case), or a normalizer name from MODES."""
    _, fold, depth = _kernel.split_study(case, control, scale == "linear")
    stat = "scaled" if scaled is True else scaled or "depth"
    if scale == "linear" and stat == "scaled":
        depth = depth / (np.median(case, axis=1) + 1)
    elif scale == "linear" and stat == "geo":
        depth = depth / np.exp(np.log1p(case).mean(axis=1))
    elif scale == "linear" and stat == "anchored":
        depth = depth / (w1_registration(case, control, True) * (np.median(control, axis=1) + 1))
    elif scale == "linear" and stat == "lorenz":
        depth = depth / (case.mean(axis=1) + 1)
    return fold, depth


def subset_pvalues(x, cond, scale, construction, *, maximum, seed, direction="both", fold=None):
    """The subset test's p-values under one construction; Besag-Clifford stopping as in wade 0.2.0.
    `fold`, one per gene, replaces the construction's registration (the mechanism section's oracles)."""
    register, stat = CONSTRUCTIONS[construction]
    cond = np.asarray(cond)
    linear = scale == "linear"
    mode = MODES[stat] if linear else 0
    case, control = _sorted(x, cond)
    side, depth = statistic(case, control, scale, stat if linear else False)
    ran = np.flatnonzero(depth > 0)
    signed = np.where(side > 0, depth, -depth)[ran]
    observed = np.ascontiguousarray({"both": depth[ran], "up": signed, "down": -signed}[direction])
    g = registration(case[ran], control[ran], scale, register) if fold is None else np.asarray(fold, float)[ran]
    g = np.ascontiguousarray(g, dtype=float)
    pooled = np.where(cond == 1, _carry(x[ran], g[:, None], scale, undo=True), x[ran])
    order = np.ascontiguousarray(np.argsort(pooled, axis=1, kind="stable"), dtype=np.int64)
    pooled = np.ascontiguousarray(np.take_along_axis(pooled, order, axis=1))
    exceed, tried = np.zeros(len(ran), np.uint64), np.zeros(len(ran), np.uint64)
    rng, size = np.random.default_rng(seed), 32
    while (live := (exceed < STOP) & (tried < maximum)).any():
        labels = draw_perms(cond, int(min(size, maximum - tried[live].max())), rng=rng)
        exceed, tried = _kernel.tally_study(pooled, order, g, linear, direction, labels, observed, exceed, tried,
                                      STOP, maximum, mode)
        size = min(2 * size, 4096)
    p = np.ones(len(x))
    p[ran] = sequential_p(exceed, tried, stop=STOP, maximum=maximum)
    return p


def lorenz_at_case_grid(z, w):
    """Both groups' empirical Lorenz curves (of count + 1) at the case boundaries k/n1, k = 1..n1-1. Rows sorted."""
    g, n1 = z.shape
    n0 = w.shape[1]
    k = np.arange(1, n1)
    lz = np.cumsum(z, axis=1)[:, :-1] / z.sum(axis=1, keepdims=True)
    cw = np.concatenate([np.zeros((g, 1)), np.cumsum(w, axis=1)], axis=1)
    padded = np.concatenate([w, np.zeros((g, 1))], axis=1)
    whole, part = k * n0 // n1, (k * n0 % n1) / n1
    lw = (cw[:, whole] + part * padded[:, whole]) / w.sum(axis=1, keepdims=True)
    return lz, lw


def lorenz_bootstrap_pvalues(x, cond, *, replicates=9_999, seed=1):
    """Review 2's centered bootstrap of the two Lorenz processes: no registration and no pooling. The
    statistic is the largest gap of the control over the case Lorenz curve (the depth over the case mean + 1);
    each replicate resamples each group within itself and takes the same gap of the centered processes."""
    cond = np.asarray(cond)
    z, w = np.sort(x[:, cond == 1] + 1, axis=1), np.sort(x[:, cond == 0] + 1, axis=1)
    lz, lw = lorenz_at_case_grid(z, w)
    observed = np.maximum((lw - lz).max(axis=1), 0)
    rng = np.random.default_rng(seed)
    n1, n0 = z.shape[1], w.shape[1]
    exceed = np.zeros(len(x))
    for _ in range(replicates):
        zs = np.sort(z[:, rng.integers(0, n1, n1)], axis=1)
        ws = np.sort(w[:, rng.integers(0, n0, n0)], axis=1)
        lzs, lws = lorenz_at_case_grid(zs, ws)
        star = np.maximum(((lws - lw) - (lzs - lz)).max(axis=1), 0)
        exceed += star >= observed * (1 - 1e-12)
    return (exceed + 1) / (replicates + 1)


def welch_permutation_pvalues(x, cond, *, replicates=9_999, seed=1):
    """A two-sided permutation test of equal means with the Welch-studentized difference (Chung and Romano)."""
    cond = np.asarray(cond)

    def welch(values, labels):
        a, b = values[:, labels == 1], values[:, labels == 0]
        return (a.mean(axis=1) - b.mean(axis=1)) / np.sqrt(a.var(axis=1, ddof=1) / a.shape[1]
                                                            + b.var(axis=1, ddof=1) / b.shape[1])
    observed = np.abs(welch(x, cond))
    rng = np.random.default_rng(seed)
    exceed = np.zeros(len(x))
    for _ in range(replicates):
        exceed += np.abs(welch(x, rng.permutation(cond))) >= observed * (1 - 1e-12)
    return (exceed + 1) / (replicates + 1)


def rates(p, levels=(0.05, 0.01, 0.001)):
    return " / ".join(f"{np.mean(p <= a):.2%}" if a >= 0.01 else f"{np.mean(p <= a):.3%}" for a in levels)


def heavy(x):
    """Genes whose largest sample is at least 4x the next on count + 1."""
    s = np.sort(x, axis=1)
    return (s[:, -1] + 1) / (s[:, -2] + 1) >= 4


# ---------------------------------------------------------------------------------------------
# Data

_COHORTS = {}


def tcga(cancer):
    """A TCGA cohort's primary design as the example notebooks run it: jittered linear expression of
    the tested genes (seed 1), labels, raw counts and size factors."""
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
        _COHORTS[cancer] = dict(
            x=expression(jittered(counts[rows], rows, 1), factors, "linear"), cond=cond, raw=counts[rows],
            factors=factors, names=cohort.genes["gene_name"].to_numpy()[rows])
    return _COHORTS[cancer]


def simulate(rng, genes, n1, n0, model, fold, mean=50):
    """Poisson counts from a latent level, jittered, size factors 1. Cases carry `fold` alike."""
    return expression(jittered(simulate_counts(rng, genes, n1, n0, model, fold, mean), range(genes), 1),
                      np.ones(n1 + n0), "linear")


def simulate_counts(rng, genes, n1, n0, model, fold, mean=50, pattern=None):
    """The raw counts behind simulate(). `pattern`, if given, maps each gene's case multipliers
    (a callable of (rng, genes, n1)) instead of a uniform fold."""
    n = n1 + n0
    if model == "Gamma":
        latent = rng.gamma(2, mean / 2, (genes, n))                        # CV^2 = 0.5
    elif model == "lognormal":
        s2 = np.log1p(1.0)                                               # CV^2 = 1
        latent = mean * np.exp(np.sqrt(s2) * rng.normal(size=(genes, n)) - s2 / 2)
    elif model.startswith("Gamma + "):
        share, times = {"Gamma + 1% x100": (0.01, 100), "Gamma + 0.5% x500": (0.005, 500)}[model]
        latent = rng.gamma(2, mean / 2, (genes, n)) * np.where(rng.random((genes, n)) < share, times, 1)
    else:
        raise ValueError(model)
    latent[:, :n1] *= fold if pattern is None else pattern(rng, genes, n1)
    return rng.poisson(latent)


def planted(share, members, majority):
    """A pattern for simulate_counts: every case at `majority` times the baseline, and a random `share` of them at
    `members` times the baseline instead, as in the confirmation generator (wade_research/point_simulation.py).
    Until 29 September 2026 the members were set to `members` x `majority`, so "every case x1/2, a tenth x8
    instead" put the tenth at 4x the baseline where the generator puts it at 8x (found in review 3)."""
    def draw(rng, genes, n1):
        mult = np.full((genes, n1), majority)
        k = max(1, round(share * n1))
        for g in range(genes):
            mult[g, rng.choice(n1, k, replace=False)] = members
        return mult
    return draw


PATTERNS = (("every case x1/2, a tenth x8 instead", 0.1, 8, 0.5), ("a tenth x16", 0.1, 16, 1.0),
            ("a fifth x4", 0.2, 4, 1.0))                  # label, share, members' multiplier, majority's multiplier


def thin(counts, cond, fold, rng):
    """Bring both groups to the lower one's latent level: binomially thin the higher group's counts by
    each gene's fold (cases by 1/fold where fold >= 1, controls by fold where fold < 1). For any Poisson
    mixture, thinning by the true fold makes the two groups identically distributed under H0."""
    out = np.array(counts, dtype=np.int64)
    fold = np.asarray(fold, float)
    for rows, cols, keep in ((np.flatnonzero(fold >= 1), cond == 1, 1 / fold[fold >= 1]),
                             (np.flatnonzero(fold < 1), cond == 0, fold[fold < 1])):
        if len(rows):
            block = np.ix_(rows, np.flatnonzero(cols))
            out[block] = rng.binomial(out[block], keep[:, None])
    return out


def thinned_fold(counts, cond, factors, *, rounds=5, seed=1):
    """A count-consistent registration: start from the W1 fold of the unthinned data, thin the higher group
    by the current fold, and multiply the fold by the W1 registration of the thinned data. Once both groups
    sit at one level their registration is unbiased, so the fixed point is the latent fold."""
    rows = range(len(counts))
    case, control = _sorted(expression(jittered(counts, rows, 1), factors, "linear"), cond)
    fold = registration(case, control, "linear", "w1")
    for r in range(rounds):
        thinned = thin(counts, cond, fold, np.random.default_rng([seed, 11, r]))
        case, control = _sorted(expression(jittered(thinned, rows, 1), factors, "linear"), cond)
        fold = fold * registration(case, control, "linear", "w1")
    return fold


def thinned_pvalues(counts, cond, factors, fold, *, maximum, seed=1):
    """Thin by `fold` (an array; "w1" for the registration of the unthinned data; "fixed" for the
    count-consistent fixed point), then permute labels over the thinned data with no registration,
    testing the median-scaled depth."""
    x = expression(jittered(counts, range(len(counts)), 1), factors, "linear")
    if isinstance(fold, str) and fold == "fixed":
        fold = thinned_fold(counts, cond, factors, seed=seed)
    elif isinstance(fold, str):
        case, control = _sorted(x, cond)
        fold = registration(case, control, "linear", "w1")
    thinned = thin(counts, cond, fold, np.random.default_rng([seed, 7]))
    xt = expression(jittered(thinned, range(len(counts)), 1), factors, "linear")
    return subset_pvalues(xt, cond, "linear", "thinned", maximum=maximum, seed=seed)


# ---------------------------------------------------------------------------------------------
# Sections

def checks():
    """The invariance of the scaled statistic, and that re-applying the registration to shuffled cases
    cannot change it."""
    print("\n== checks")
    rng = np.random.default_rng(0)
    x = simulate(rng, 200, 40, 60, "Gamma + 1% x100", 1.0)
    cond = np.r_[np.ones(40, int), np.zeros(60, int)]
    case, control = _sorted(x, cond)
    worst_depth, worst_scaled = 0.0, 0.0
    for c in (0.1, 0.5, 3.0, 40.0):
        moved = np.ascontiguousarray((case + 1) * c - 1)
        _, d0 = statistic(case, control, "linear", False)
        _, d1 = statistic(moved, control, "linear", False)
        _, s0 = statistic(case, control, "linear", True)
        _, s1 = statistic(moved, control, "linear", True)
        k = d0 > 0
        worst_depth = max(worst_depth, np.max(np.abs(d1[k] - c * d0[k]) / (c * d0[k])))
        worst_scaled = max(worst_scaled, np.max(np.abs(s1[k] - s0[k]) / s0[k]))
    print(f"depth(fold c applied to cases) = c x depth: largest relative error {worst_depth:.1e}")
    print(f"scaled depth unchanged by a fold on the cases: largest relative error {worst_scaled:.1e}")
    ran = np.flatnonzero(statistic(case, control, "linear", True)[1] > 0)
    g = registration(case[ran], control[ran], "linear", "w1")
    pooled = np.where(cond == 1, _carry(x[ran], g[:, None], "linear", undo=True), x[ran])
    order = np.ascontiguousarray(np.argsort(pooled, axis=1, kind="stable"), dtype=np.int64)
    pooled = np.ascontiguousarray(np.take_along_axis(pooled, order, axis=1))
    observed = np.ascontiguousarray(statistic(case, control, "linear", True)[1][ran])
    labels = draw_perms(cond, 999, rng=np.random.default_rng(1))
    zero = np.zeros(len(ran), np.uint64)
    with_g = _kernel.tally_study(pooled, order, g, True, "both", labels, observed, zero, zero, 10**6, 999, 1)[0]
    without = _kernel.tally_study(pooled, order, np.ones(len(ran)), True, "both", labels, observed, zero, zero, 10**6, 999, 1)[0]
    print(f"shuffled exceedances with and without re-applying the registration to new cases: "
          f"{int(np.sum(with_g != without))} of {len(ran)} genes differ")
    for stat, mode in MODES.items():
        if stat == "depth":
            continue
        d_obs = statistic(case, control, "linear", stat)[1]
        keep = np.flatnonzero(d_obs > 0)
        g = registration(case[keep], control[keep], "linear", "w1")
        pooled = np.where(cond == 1, _carry(x[keep], g[:, None], "linear", undo=True), x[keep])
        order = np.ascontiguousarray(np.argsort(pooled, axis=1, kind="stable"), dtype=np.int64)
        pooled = np.ascontiguousarray(np.take_along_axis(pooled, order, axis=1))
        zero = np.zeros(len(keep), np.uint64)
        same = _kernel.tally_study(pooled, order, g, True, "both", cond[None, :].astype(np.uint8),
                             np.ascontiguousarray(d_obs[keep]), zero, zero, 10, 1, mode)[0]
        print(f"normalizer {stat}: the observed labelling replayed by the kernel reaches the observed score in "
              f"{int(same.sum())} of {len(keep)} genes")
    lz, lw = lorenz_at_case_grid(case + 1, control + 1)
    direct = np.maximum((lw - lz).max(axis=1), 0)
    _, via = statistic(case, control, "linear", "lorenz")
    print(f"depth / (case mean + 1) against the largest gap between the control and case Lorenz curves at the case "
          f"boundaries: largest absolute difference {np.max(np.abs(direct - via)):.1e}")
    a, b = _sorted(np.log2(x + 1), cond)
    _, _, depth_log2 = _kernel.split_study(a, b, False)
    k = np.arange(1, a.shape[1])
    n0 = b.shape[1]
    absolute_case = np.cumsum(a, axis=1)[:, :-1] / a.shape[1] - k / a.shape[1] * a.mean(axis=1, keepdims=True)
    cb = np.concatenate([np.zeros((len(b), 1)), np.cumsum(b, axis=1)], axis=1)
    padded = np.concatenate([b, np.zeros((len(b), 1))], axis=1)
    whole, part = k * n0 // a.shape[1], (k * n0 % a.shape[1]) / a.shape[1]
    absolute_control = (cb[:, whole] + part * padded[:, whole]) / n0 - k / a.shape[1] * b.mean(axis=1, keepdims=True)
    gap = np.maximum((absolute_control - absolute_case).max(axis=1), 0)
    print(f"log2 depth against the largest gap between the control and case absolute Lorenz curves of log2(x + 1): "
          f"largest absolute difference {np.max(np.abs(gap - depth_log2)):.1e}")


def foldform():
    print("\n== foldform: log2 IQR ratio against log2 median ratio, genes with median >= 50 in both groups")
    rng = np.random.default_rng(0)
    for cancer in ("prostate", "breast"):
        d = tcga(cancer)
        x, cond = d["x"], d["cond"]
        tumours = rng.permutation(np.flatnonzero(cond == 1))
        half = len(tumours) // 2
        splits = {"tumour v normal": (cond == 1, cond == 0),
                  "two random halves of the tumours": (np.isin(np.arange(len(cond)), tumours[:half]),
                                                       np.isin(np.arange(len(cond)), tumours[half:]))}
        for label, (a, b) in splits.items():
            qa, qb = np.percentile(x[:, a], [25, 50, 75], axis=1), np.percentile(x[:, b], [25, 50, 75], axis=1)
            iqr_a, iqr_b = qa[2] - qa[0], qb[2] - qb[0]
            keep = (qa[1] >= 50) & (qb[1] >= 50) & (iqr_a > 0) & (iqr_b > 0)
            level, spread = np.log2(qa[1][keep] / qb[1][keep]), np.log2(iqr_a[keep] / iqr_b[keep])
            slope, intercept = np.polyfit(level, spread, 1)
            cells = []
            for lo, hi in ((-4, -2), (-2, -1), (-1, -0.5), (-0.5, 0.5), (0.5, 1), (1, 2), (2, 4)):
                k = (level >= lo) & (level < hi)
                if k.sum() >= 20:
                    cells.append(f"[{lo}, {hi}): {np.median(spread[k]):+.2f} (n={k.sum()})")
            print(f"{cancer:8s} {label:32s} genes {keep.sum():6d}  slope {slope:.2f}  intercept {intercept:+.2f}")
            print(f"         median log2 IQR ratio by log2 median ratio: " + "  ".join(cells))


def realnull(labellings=8, maximum=9_999):
    print(f"\n== realnull: TCGA prostate genes under random labels ({labellings} labellings per design, "
          f"up to {maximum:,} shuffles); share of genes at p <= 0.05 / 0.01 / 0.001")
    d = tcga("prostate")
    x, n = d["x"], d["x"].shape[1]
    big = heavy(x)
    print(f"{x.shape[0]} genes x {n} samples; {big.sum()} genes whose top sample is >= 4x the next")
    for scale, names in (("linear", list(CONSTRUCTIONS)), ("log2", ["exact", "released", "W1 + depth"])):
        values = x if scale == "linear" else np.log2(x + 1)
        for n1 in (445, 248, 52):
            labels = []
            for r in range(labellings):
                lab = np.zeros(n, int)
                lab[np.random.default_rng(2000 + r).choice(n, n1, replace=False)] = 1
                labels.append(lab)
            for name in names:
                p = np.array([subset_pvalues(values, lab, scale, name, maximum=maximum, seed=r)
                              for r, lab in enumerate(labels)])
                print(f"{scale:6s} {n1}:{n - n1}  {name:18s} all genes {rates(p)}   heavy genes {rates(p[:, big])}",
                      flush=True)


def genes(labellings=200, maximum=99_999):
    print(f"\n== genes: one gene's values under {labellings} random 445:52 labellings (up to {maximum:,} "
          f"shuffles); observed p with the real labels, then P(p <= 0.05 / 0.01 / 0.001)")
    d = tcga("prostate")
    x, cond, names = d["x"], d["cond"], d["names"]
    n, n1 = len(cond), int(cond.sum())
    rng = np.random.default_rng(7)
    labels = np.zeros((labellings, n), int)
    for r in range(labellings):
        labels[r, rng.choice(n, n1, replace=False)] = 1
    for gene in ("MC4R", "DRGX", "AMBN", "IRS4", "ETV4", "ETV1", "ERG"):
        i = int(np.flatnonzero(names == gene)[0])
        row = x[i:i + 1]
        top = np.sort(x[i, cond == 1])[::-1][:4].round().astype(int).tolist()
        ctl = int(round(x[i, cond == 0].max()))
        for name in ("exact", "released", "proposed"):
            observed = subset_pvalues(row, cond, "linear", name, maximum=maximum, seed=1)[0]
            p = np.array([subset_pvalues(row, labels[r], "linear", name, maximum=maximum, seed=r)[0]
                          for r in range(labellings)])
            print(f"{gene:5s} {name:9s} real labels p = {observed:.1e}   random labels {rates(p)}   "
                  f"(top cases {top}, control max {ctl})", flush=True)


def simnull(genes_per=1500, maximum=9_999):
    print(f"\n== simnull: {genes_per} genes per cell, baseline mean 50, up to {maximum:,} shuffles; "
          f"share at p <= 0.05 / 0.01")
    rng = np.random.default_rng(2026092902)
    for n1, n0 in ((10, 100), (50, 500), (20, 20), (100, 100), (200, 20), (500, 50)):
        cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
        for model in ("Gamma", "lognormal", "Gamma + 1% x100", "Gamma + 0.5% x500"):
            for fold in (1.0, 2.0, 0.5):
                x = simulate(rng, genes_per, n1, n0, model, fold)
                cells = [f"{name} {rates(subset_pvalues(x, cond, 'linear', name, maximum=maximum, seed=1), (0.05, 0.01))}"
                         for name in CONSTRUCTIONS]
                print(f"{n1:3d}:{n0:<3d} {model:17s} x{fold:<3g} " + " | ".join(cells), flush=True)


def mechanism(genes_per=3000, maximum=9_999):
    print(f"\n== mechanism: registration by the true fold, by the true fold times independent noise as large as "
          f"the W1 estimate's error, and by estimates; {genes_per} genes per cell, share at p <= 0.05 / 0.01")
    rng = np.random.default_rng(2026092903)
    for n1, n0 in ((20, 200), (50, 500), (200, 20)):
        cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
        for model in ("Gamma", "lognormal"):
            for truth in (1.0, 2.0):
                x = simulate(rng, genes_per, n1, n0, model, truth)
                case, control = _sorted(x, cond)
                spread = np.std(np.log(registration(case, control, "linear", "w1")))
                noisy = truth * np.exp(rng.normal(0, spread, genes_per))
                cells = []
                for stat, name in (("depth", "W1 + depth"), ("scaled", "proposed")):
                    cells.append(f"{stat}: true fold {rates(subset_pvalues(x, cond, 'linear', name, maximum=maximum, seed=1, fold=np.full(genes_per, truth)), (0.05, 0.01))}"
                                 f", true x noise {rates(subset_pvalues(x, cond, 'linear', name, maximum=maximum, seed=1, fold=noisy), (0.05, 0.01))}"
                                 f", W1 {rates(subset_pvalues(x, cond, 'linear', name, maximum=maximum, seed=1), (0.05, 0.01))}")
                released = rates(subset_pvalues(x, cond, "linear", "released", maximum=maximum, seed=1), (0.05, 0.01))
                print(f"{n1:3d}:{n0:<3d} {model:9s} x{truth:<3g} sd(log W1 fold) {spread:.3f} | " + " | ".join(cells)
                      + f" | released (mean fold, depth) {released}", flush=True)


def power(replicates=6, maximum=99_999, names=("released", "W1 + depth", "proposed"), designs=None):
    cfg = json.loads((ROOT / "research/protocols/confirmation.json").read_text())
    cfg["seed_root"], cfg["replicates_per_cell"] = 2026092901, replicates      # fresh seeds
    print(f"\n== power: confirmation generator (protocol v{cfg['version']} patterns), {replicates} matrices "
          f"per design and panel, seed root {cfg['seed_root']}, up to {maximum:,} shuffles, BH q <= 0.05")
    called, false, calls, nulls, share = {}, {}, {}, {}, {}
    for setting in datasets(cfg):
        design = f"{setting['n_case']}:{setting['n_control']}"
        if designs is not None and design not in designs:
            continue
        data = generate(cfg, setting)
        counts, cond = data["counts"], data["cond"]
        keep = np.flatnonzero(wade.filter_counts(counts, min_count=3, min_samples=2) & (counts.sum(axis=1) > 0))
        x = expression(jittered(counts[keep], keep, 1), wade.fit_normalization(counts).size_factors, "linear")
        profile, base = data["profile"][keep], data["baseline_mean"][keep]
        planted = data["altered"][keep] & ~np.isin(profile, ["bulk_up_1.5x", "bulk_down_two_thirds"])
        for name in names:
            q = bh_adjust(subset_pvalues(x, cond, "linear", name, maximum=maximum, seed=1))
            if setting["panel"] == "complete_null":
                nulls.setdefault((design, name), []).append(int(np.sum(q <= 0.05)))
                continue
            c = q <= 0.05
            calls.setdefault((design, name), []).append(int(c.sum()))
            false.setdefault((design, name), []).append(np.sum(c & ~planted) / max(c.sum(), 1))
            for pattern in np.unique(profile[data["altered"][keep] & (base >= 5)]):
                k = (profile == pattern) & (base >= 5)
                called.setdefault((design, name, pattern), []).append(np.mean(c[k]))
        print(f"  {setting['id']} done", file=sys.stderr, flush=True)
    for design in dict.fromkeys(k[0] for k in calls):
        print(f"\n{design}: complete-null matrices with any call / mean calls; mixed matrices: mean false "
              f"discovery proportion (mean calls)")
        for name in names:
            z = nulls[(design, name)]
            print(f"  {name:12s} complete nulls {sum(v > 0 for v in z)} of {len(z)} ({np.mean(z):.1f})   "
                  f"mixed {np.mean(false[(design, name)]):.1%} ({np.mean(calls[(design, name)]):.1f})")
        print(f"  share of planted genes (baseline mean >= 5) called, by pattern: " + " / ".join(names))
        for pattern in sorted({k[2] for k in called if k[0] == design}):
            print(f"    {pattern:34s} " + " / ".join(f"{np.mean(called[(design, name, pattern)]):.0%}"
                                                    for name in names))


def truelabels(maximum=99_999):
    print(f"\n== truelabels: TCGA with the real labels, up to {maximum:,} shuffles; subset calls at q <= 0.05")
    panels = {"prostate": ("ERG", "ETV1", "ETV4", "ETV5", "FLI1", "SPINK1", "MC4R", "AMBN", "IRS4", "DEFA5"),
              "breast": ("ERBB2", "GRB7", "STARD3", "PGAP3", "MIEN1", "FGFR1", "CCND1")}
    for cancer, show in panels.items():
        d = tcga(cancer)
        x, cond, names = d["x"], d["cond"], d["names"]
        for name in ("released", "proposed"):
            t = time.perf_counter()
            p = subset_pvalues(x, cond, "linear", name, maximum=maximum, seed=1)
            q = bh_adjust(p)
            cells = " ".join(f"{g} {p[names == g][0]:.1e} ({q[names == g][0]:.2g})" for g in show)
            print(f"{cancer:8s} {int(cond.sum())}:{int((cond == 0).sum())} {name:9s} calls {int(np.sum(q <= 0.05)):6d}"
                  f"  [{time.perf_counter() - t:.0f} s]  p (q): {cells}", flush=True)


def thinning(maximum=9_999):
    print("\n== thinning: TCGA prostate tumours split at random; in one group half the genes are thinned "
          "binomially by 1/2, so every case carries the same 2x change; share at p <= 0.01 by the gene's mean "
          "over all tumours before thinning (a level that depends on neither the split nor the thinning)")
    d = tcga("prostate")
    raw, cond, factors = d["raw"], d["cond"], d["factors"]
    tumours = np.flatnonzero(cond == 1)
    level = (raw[:, tumours] / factors[tumours]).mean(axis=1)
    rng = np.random.default_rng(3)
    for n1 in (222, 45):
        pick = rng.permutation(tumours)
        a, b = pick[:n1], pick[n1:]
        shifted = rng.random(len(raw)) < 0.5
        counts = raw[:, np.r_[a, b]].astype(np.int64)
        lab = np.r_[np.ones(len(a), int), np.zeros(len(b), int)]
        thin = np.ix_(shifted, lab == 0)
        counts[thin] = rng.binomial(counts[thin], 0.5)
        x = expression(jittered(counts, range(len(raw)), 1), factors[np.r_[a, b]], "linear")
        for name in ("exact", "released", "proposed"):
            p = subset_pvalues(x, lab, "linear", name, maximum=maximum, seed=1)
            cells = []
            for lo, hi in ((0, 2), (2, 5), (5, 20), (20, 100), (100, np.inf)):
                k = (level >= lo) & (level < hi)
                cells.append(f"[{lo}, {hi:g}): 2x {np.mean(p[k & shifted] <= 0.01):.1%} of {np.sum(k & shifted)}, "
                             f"unchanged {np.mean(p[k & ~shifted] <= 0.01):.1%} of {np.sum(k & ~shifted)}")
            print(f"{n1}:{len(b)} {name:9s} " + " | ".join(cells), flush=True)


def lowcount(genes_per=3000, maximum=9_999):
    print(f"\n== lowcount: simulated Gamma-Poisson genes (CV^2 0.5) at low baselines, every case carrying the "
          f"same fold of its latent level; {genes_per} genes per cell, share at p <= 0.05 / 0.01")
    rng = np.random.default_rng(2026092904)
    for n1, n0 in ((100, 100), (250, 250), (50, 500), (200, 20)):
        cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
        for mean in (1, 2, 5, 20):
            for fold in (1.0, 2.0):
                x = simulate(rng, genes_per, n1, n0, "Gamma", fold, mean=mean)
                cells = [f"{name} {rates(subset_pvalues(x, cond, 'linear', name, maximum=maximum, seed=1), (0.05, 0.01))}"
                         for name in ("exact", "released", "proposed")]
                print(f"{n1:3d}:{n0:<3d} baseline {mean:3d} x{fold:<3g} " + " | ".join(cells), flush=True)


def normalizers(maximum=99_999):
    """Review 1, recommendation A: do other invariant normalizers recover the power lost when cases are few?"""
    names = ("proposed", "proposed, geometric-mean scale", "proposed, control-anchored scale")
    power(names=names, maximum=maximum)
    print("\n== normalizers, validity: simulated sparse-tail and light-tail nulls (1,500 genes per cell, up to "
          "9,999 shuffles); share at p <= 0.05 / 0.01")
    rng = np.random.default_rng(2026093001)
    for n1, n0 in ((10, 100), (50, 500), (20, 20), (100, 100), (500, 50)):
        cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
        for model in ("Gamma", "Gamma + 1% x100", "Gamma + 0.5% x500"):
            for fold in (1.0, 2.0):
                x = simulate(rng, 1500, n1, n0, model, fold)
                cells = [f"{name} {rates(subset_pvalues(x, cond, 'linear', name, maximum=9_999, seed=1), (0.05, 0.01))}"
                         for name in names]
                print(f"{n1:3d}:{n0:<3d} {model:17s} x{fold:<3g} " + " | ".join(cells), flush=True)
    print("\n== normalizers, real genes under random labels (8 labellings, up to 9,999 shuffles); share at "
          "p <= 0.05 / 0.01 / 0.001")
    d = tcga("prostate")
    x, n = d["x"], d["x"].shape[1]
    big = heavy(x)
    for n1 in (445, 52):
        labels = []
        for r in range(8):
            lab = np.zeros(n, int)
            lab[np.random.default_rng(2000 + r).choice(n, n1, replace=False)] = 1
            labels.append(lab)
        for name in names:
            p = np.array([subset_pvalues(x, lab, "linear", name, maximum=9_999, seed=r) for r, lab in enumerate(labels)])
            print(f"{n1}:{n - n1}  {name:34s} all genes {rates(p)}   heavy genes {rates(p[:, big])}", flush=True)


def thinnednull(genes_per=3000, maximum=9_999):
    """Review 1, recommendation B: binomial thinning to a common level, by the true fold (an oracle) and
    by the estimated registration fold."""
    print(f"\n== thinnednull: Gamma-Poisson (CV^2 0.5), every case carrying the same fold; {genes_per} genes per "
          f"cell, share at p <= 0.05 / 0.01 for proposed | thinned by the true fold | thinned by the estimated fold")
    rng = np.random.default_rng(2026093002)
    for n1, n0 in ((100, 100), (250, 250), (50, 500), (200, 20)):
        cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
        ones = np.ones(n1 + n0)
        for mean in (1, 2, 5, 20):
            for fold in (1.0, 2.0):
                counts = simulate_counts(rng, genes_per, n1, n0, "Gamma", fold, mean)
                x = expression(jittered(counts, range(genes_per), 1), ones, "linear")
                cells = [rates(subset_pvalues(x, cond, "linear", "proposed", maximum=maximum, seed=1), (0.05, 0.01)),
                         rates(thinned_pvalues(counts, cond, ones, np.full(genes_per, fold), maximum=maximum), (0.05, 0.01)),
                         rates(thinned_pvalues(counts, cond, ones, "w1", maximum=maximum), (0.05, 0.01))]
                print(f"{n1:3d}:{n0:<3d} baseline {mean:3d} x{fold:<3g} " + " | ".join(cells), flush=True)
    print("\n== thinnednull, sparse tails at baseline 50 (1,500 genes per cell): proposed | thinned by the estimated fold")
    for n1, n0 in ((10, 100), (20, 20), (100, 100), (500, 50)):
        cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
        ones = np.ones(n1 + n0)
        for model in ("Gamma + 1% x100", "Gamma + 0.5% x500"):
            for fold in (1.0, 2.0):
                counts = simulate_counts(rng, 1500, n1, n0, model, fold)
                x = expression(jittered(counts, range(1500), 1), ones, "linear")
                print(f"{n1:3d}:{n0:<3d} {model:17s} x{fold:<3g} "
                      f"{rates(subset_pvalues(x, cond, 'linear', 'proposed', maximum=maximum, seed=1), (0.05, 0.01))} | "
                      f"{rates(thinned_pvalues(counts, cond, ones, 'w1', maximum=maximum), (0.05, 0.01))}", flush=True)
    print("\n== thinnednull, power: share of planted genes at p <= 0.001 (up to 99,999 shuffles, 400 genes per "
          "cell): proposed | thinned by the estimated fold")
    for n1, n0 in ((250, 250), (50, 500)):
        cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
        ones = np.ones(n1 + n0)
        for label, share, members, majority in PATTERNS:
            for mean in (5, 50):
                counts = simulate_counts(rng, 400, n1, n0, "Gamma", 1.0, mean, pattern=planted(share, members, majority))
                x = expression(jittered(counts, range(400), 1), ones, "linear")
                a = np.mean(subset_pvalues(x, cond, "linear", "proposed", maximum=99_999, seed=1) <= 1e-3)
                b = np.mean(thinned_pvalues(counts, cond, ones, "w1", maximum=99_999) <= 1e-3)
                print(f"{n1:3d}:{n0:<3d} {label:36s} baseline {mean:3d}  {a:.0%} | {b:.0%}", flush=True)


def thinfixed(genes_per=3000, maximum=9_999):
    """The thinning null with the count-consistent fixed-point fold (thinned_fold)."""
    print(f"\n== thinfixed: Gamma-Poisson (CV^2 0.5), every case carrying the same fold; {genes_per} genes per cell. "
          f"Median fold estimate (unthinned W1 -> fixed point), then share at p <= 0.05 / 0.01 thinned by the fixed point")
    rng = np.random.default_rng(2026093003)
    for n1, n0 in ((100, 100), (250, 250), (50, 500), (200, 20)):
        cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
        ones = np.ones(n1 + n0)
        for mean in (1, 2, 5, 20):
            for fold in (1.0, 2.0):
                counts = simulate_counts(rng, genes_per, n1, n0, "Gamma", fold, mean)
                start = np.median(thinned_fold(counts, cond, ones, rounds=0))
                fixed = np.median(thinned_fold(counts, cond, ones))
                p = thinned_pvalues(counts, cond, ones, "fixed", maximum=maximum)
                print(f"{n1:3d}:{n0:<3d} baseline {mean:3d} x{fold:<3g} fold {start:.2f} -> {fixed:.2f}   "
                      f"{rates(p, (0.05, 0.01))}", flush=True)
    print("\n== thinfixed, sparse tails at baseline 50 (1,500 genes per cell); share at p <= 0.05 / 0.01")
    for n1, n0 in ((10, 100), (100, 100), (500, 50)):
        cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
        ones = np.ones(n1 + n0)
        for model in ("Gamma + 1% x100", "Gamma + 0.5% x500"):
            for fold in (1.0, 2.0):
                counts = simulate_counts(rng, 1500, n1, n0, model, fold)
                print(f"{n1:3d}:{n0:<3d} {model:17s} x{fold:<3g} "
                      f"{rates(thinned_pvalues(counts, cond, ones, 'fixed', maximum=maximum), (0.05, 0.01))}", flush=True)


def bimodal_matrix(rng, genes, n, sd, separation):
    """Review 2's bimodal null: an equal mixture of lognormal(3, sd^2) and lognormal(3 + separation, sd^2)."""
    return np.exp(rng.normal(3, sd, (genes, n)) + separation * (rng.random((genes, n)) < .5))


def review2(maximum=9_999):
    """Review 2: the bimodal null and remedies, small real designs, the Lorenz bootstrap, the mean test."""
    names = ("exact", "proposed", "lorenz", "proposed, lower-median registration", "lorenz, lower-median registration")
    print(f"\n== review2, bimodal null: log X ~ an equal mixture of N(3, sd^2) and N(3 + separation, sd^2), both "
          f"groups; share at p <= 0.05 / 0.01 / 0.001 (up to {maximum:,} shuffles). 'oracle' registers by the true "
          f"fold, 1")
    for (n1, n0, genes, seeds) in ((20, 20, 30000, (921173, 583241)), (50, 50, 6000, (1,)), (100, 100, 6000, (2,)),
                                   (250, 250, 3000, (3,)), (20, 200, 6000, (4,)), (200, 20, 6000, (5,))):
        cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
        for sd, separation in ((0.3, 2), (0.03, 5)):
            for seed in seeds:
                x = bimodal_matrix(np.random.default_rng(seed), genes, n1 + n0, sd, separation)
                cells = [f"{name} {rates(subset_pvalues(x, cond, 'linear', name, maximum=maximum, seed=seed + 11))}"
                         for name in names]
                for name in ("proposed", "lorenz"):
                    p = subset_pvalues(x, cond, "linear", name, maximum=maximum, seed=seed + 11, fold=np.ones(genes))
                    cells.append(f"{name}, oracle {rates(p)}")
                print(f"{n1:3d}:{n0:<3d} sd {sd} separation {separation} seed {seed}: " + " | ".join(cells), flush=True)

    print(f"\n== review2, real genes at small designs: TCGA prostate tumours only, split at random 10 times per design "
          f"(all genes null); share at p <= 0.05 / 0.01 / 0.001, all genes / heavy genes")
    d = tcga("prostate")
    x_all, cond_all = d["x"], d["cond"]
    tumours = np.flatnonzero(cond_all == 1)
    for n1, n0 in ((20, 20), (10, 40), (40, 10)):
        ps = {name: [] for name in ("exact", "proposed", "lorenz", "proposed, lower-median registration")}
        bigs = []
        for r in range(10):
            pick = np.random.default_rng(3000 + r).choice(tumours, n1 + n0, replace=False)
            x = x_all[:, pick]
            keep = np.flatnonzero(x.max(axis=1) > x.min(axis=1))
            x = np.ascontiguousarray(x[keep])
            cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
            bigs.append(heavy(x))
            for name in ps:
                ps[name].append(subset_pvalues(x, cond, "linear", name, maximum=maximum, seed=r))
        big = np.concatenate(bigs)
        for name, parts in ps.items():
            p = np.concatenate(parts)
            print(f"{n1}:{n0}  {name:32s} all genes {rates(p)}   heavy genes ({big.sum()}) {rates(p[big])}", flush=True)

    print("\n== review2, the centered Lorenz bootstrap (9,999 replicates) against the proposed test; share at "
          "p <= 0.05 / 0.01 / 0.001")
    rng = np.random.default_rng(2026093004)
    for n1, n0, genes in ((20, 20, 10000), (100, 100, 4000), (20, 200, 4000), (200, 20, 4000)):
        cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
        for model in ("bimodal sd 0.3 separation 2", "Gamma", "Gamma + 1% x100"):
            if model.startswith("bimodal"):
                x = bimodal_matrix(rng, genes, n1 + n0, 0.3, 2)
            else:
                x = simulate(rng, genes, n1, n0, model, 1.0)
            boot = lorenz_bootstrap_pvalues(x, cond, seed=7)
            prop = subset_pvalues(x, cond, "linear", "proposed", maximum=maximum, seed=7)
            print(f"{n1:3d}:{n0:<3d} {model:28s} bootstrap {rates(boot)} | proposed {rates(prop)}", flush=True)

    print("\n== review2, the mean test when means are equal but spreads differ (lognormal-Poisson at 50 counts; the "
          "cases' CV^2 against the controls'); share at p <= 0.05 / 0.01, the released permutation mean test and a "
          "Welch-studentized permutation test (9,999 shuffles)")
    from wade._means import mean_test
    for n1, n0 in ((50, 500), (500, 50), (250, 250)):
        cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
        for cv_case, cv_control in ((2.0, 0.25), (0.25, 2.0)):
            genes = 3000
            lat = np.empty((genes, n1 + n0))
            for cols, cv2 in ((cond == 1, cv_case), (cond == 0, cv_control)):
                s2 = np.log1p(cv2)
                lat[:, cols] = 50 * np.exp(np.sqrt(s2) * rng.normal(size=(genes, cols.sum())) - s2 / 2)
            x = expression(jittered(rng.poisson(lat), range(genes), 1), np.ones(n1 + n0), "linear")
            released, _, _ = mean_test(x, cond, direction="both", stop=25, maximum=maximum, seed=1)
            welch = welch_permutation_pvalues(x, cond, seed=1)
            print(f"{n1:3d}:{n0:<3d} case CV^2 {cv_case} control CV^2 {cv_control}: released {rates(released, (0.05, 0.01))}"
                  f" | Welch {rates(welch, (0.05, 0.01))}", flush=True)



def mann_whitney_cutoff(n1, n0, beta):
    """The largest c with P(U <= c - 1) <= beta / 2 for the Mann-Whitney count U of exchangeable, untied
    samples: the rank-inversion interval [d_(c), d_(N - c + 1)] of the N = n1 n0 sorted pairwise differences
    then covers a location shift with probability at least 1 - beta. U's generating function is
    prod_{i <= n1} (1 - q^(n0 + i)) / (1 - q^i); only the lower half is needed, and each factor touches only
    lower coefficients, so truncating there is exact."""
    half = n1 * n0 // 2 + 1
    poly = np.zeros(half + 1)
    poly[0] = 1.0
    for i in range(1, n1 + 1):
        m = n0 + i
        if m <= half:
            poly[m:] = poly[m:] - poly[:-m]
        for r in range(i):
            poly[r::i] = np.cumsum(poly[r::i])
    cdf = np.cumsum(poly) / math.comb(n1 + n0, n1)
    return int(np.searchsorted(cdf, beta / 2, side="right"))


def shift_interval(x, cond, beta, chunk=32):
    """Each gene's (1 - beta) rank-inversion interval for the shift of log(x + 1), cases minus controls.
    Under H0 (a common fold on count + 1) the shift model holds exactly, so the coverage is exact."""
    cond = np.asarray(cond)
    lz, lw = np.log1p(x[:, cond == 1]), np.log1p(x[:, cond == 0])
    n1, n0 = lz.shape[1], lw.shape[1]
    c, total = mann_whitney_cutoff(n1, n0, beta), n1 * n0
    if c < 1:
        raise ValueError(f"no finite {1 - beta} interval at {n1}:{n0}")
    lo, hi = np.empty(len(x)), np.empty(len(x))
    for s in range(0, len(x), chunk):
        d = (lz[s:s + chunk, :, None] - lw[s:s + chunk, None, :]).reshape(-1, total)
        part = np.partition(d, (c - 1, total - c), axis=1)
        lo[s:s + chunk], hi[s:s + chunk] = part[:, c - 1], part[:, total - c]
    return lo, hi


def bb_pvalues(x, cond, construction, *, beta, points=9, maximum, seed):
    """Berger and Boos's p-value: beta plus the largest permutation p-value over registrations in the
    (1 - beta) shift interval. The sup is taken over `points` evenly spaced registrations (so it is
    approximate), each with the same shuffles."""
    lo, hi = shift_interval(x, cond, beta)
    worst = np.zeros(len(x))
    for t in np.linspace(0, 1, points):
        p = subset_pvalues(x, cond, "linear", construction, maximum=maximum, seed=seed, fold=np.exp(lo + t * (hi - lo)))
        worst = np.maximum(worst, p)
    return np.minimum(1, beta + worst)


def planted_matrix(rng, n1, n0, share, members, majority, genes=400, mean=50):
    counts = simulate_counts(rng, genes, n1, n0, "Gamma", 1.0, mean, pattern=planted(share, members, majority))
    return expression(jittered(counts, range(genes), 1), np.ones(n1 + n0), "linear")


def null_matrix(rng, genes, n1, n0, model):
    if model.startswith("bimodal"):
        sd, separation = {"bimodal sd 0.3 separation 2": (0.3, 2), "bimodal sd 0.03 separation 5": (0.03, 5)}[model]
        return bimodal_matrix(rng, genes, n1 + n0, sd, separation)
    return simulate(rng, genes, n1, n0, model, 1.0)


def lorenz(maximum=9_999):
    """Review 2: the depth over the case mean + 1 is the largest gap between the controls' and the cases'
    Lorenz curves. Its power on the confirmation generator, its validity on simulated and real nulls, and
    the real-label panel, against the median-case normalizer."""
    names = ("proposed", "lorenz")
    power(names=("released", "proposed", "lorenz"), maximum=99_999)
    print(f"\n== lorenz, validity: simulated nulls (1,500 genes per cell, up to {maximum:,} shuffles); share at "
          f"p <= 0.05 / 0.01 / 0.001")
    rng = np.random.default_rng(2026093005)
    for n1, n0 in ((10, 100), (50, 500), (20, 20), (100, 100), (200, 20), (500, 50)):
        cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
        for model in ("Gamma", "lognormal", "Gamma + 1% x100", "Gamma + 0.5% x500"):
            for fold in (1.0, 2.0, 0.5):
                x = simulate(rng, 1500, n1, n0, model, fold)
                cells = [f"{name} {rates(subset_pvalues(x, cond, 'linear', name, maximum=maximum, seed=1))}"
                         for name in names]
                print(f"{n1:3d}:{n0:<3d} {model:17s} x{fold:<3g} " + " | ".join(cells), flush=True)
    print(f"\n== lorenz, TCGA prostate genes under random labels (8 labellings per design, up to {maximum:,} "
          f"shuffles); share at p <= 0.05 / 0.01 / 0.001")
    d = tcga("prostate")
    x, n = d["x"], d["x"].shape[1]
    big = heavy(x)
    for n1 in (445, 248, 52):
        labels = []
        for r in range(8):
            lab = np.zeros(n, int)
            lab[np.random.default_rng(2000 + r).choice(n, n1, replace=False)] = 1
            labels.append(lab)
        for name in ("exact",) + names:
            p = np.array([subset_pvalues(x, lab, "linear", name, maximum=maximum, seed=r) for r, lab in enumerate(labels)])
            print(f"{n1}:{n - n1}  {name:9s} all genes {rates(p)}   heavy genes {rates(p[:, big])}", flush=True)
    print("\n== lorenz, TCGA prostate with the real labels (up to 99,999 shuffles): p (BH q) for the panel, "
          "subset calls at q <= 0.05")
    cond, gene_names = d["cond"], d["names"]
    show = ("ERG", "ETV1", "ETV4", "ETV5", "SPINK1", "MC4R", "AMBN", "IRS4", "DEFA5")
    for name in ("released",) + names:
        p = subset_pvalues(x, cond, "linear", name, maximum=99_999, seed=1)
        q = bh_adjust(p)
        cells = " ".join(f"{g} {p[gene_names == g][0]:.1e} ({q[gene_names == g][0]:.2g})" for g in show)
        print(f"prostate {name:9s} calls {int(np.sum(q <= 0.05)):5d}  {cells}", flush=True)


def bimodalgenes(splits=5000, maximum=9_999):
    """Review 2's bimodal null in real data: GSTM1 (a common whole-gene deletion) and ERAP2 (a common allele
    whose transcripts are degraded) are bimodal across TCGA prostate tumours because of germline genotype.
    Random splits of the tumours are exact nulls; each split is tested as its own row."""
    print(f"\n== bimodalgenes: TCGA prostate tumours, {splits:,} random splits per design (up to {maximum:,} "
          f"shuffles); share of splits at p <= 0.05 / 0.01 / 0.001")
    d = tcga("prostate")
    x, cond, names = d["x"], d["cond"], d["names"]
    for gene in ("GSTM1", "ERAP2"):
        v = x[names == gene][0][cond == 1]
        low = np.log2(v + 1) < 0.5 * (np.log2(v + 1).min() + np.log2(v + 1).max())
        print(f"{gene}: {low.mean():.0%} of {len(v)} tumours in the lower mode (below the midpoint of the log2 range); "
              f"median {np.median(v[low]):.0f} there, {np.median(v[~low]):.0f} in the upper mode")
        for n1, n0 in ((20, 20), (10, 40), (40, 10), (50, 50), (100, 100)):
            rng = np.random.default_rng([4000, n1, n0])
            rows = np.stack([v[rng.choice(len(v), n1 + n0, replace=False)] for _ in range(splits)])
            lab = np.r_[np.ones(n1, int), np.zeros(n0, int)]
            cells = [f"{name} {rates(subset_pvalues(rows, lab, 'linear', name, maximum=maximum, seed=1))}"
                     for name in ("exact", "proposed", "lorenz")]
            print(f"  {n1:3d}:{n0:<3d} " + " | ".join(cells), flush=True)


def bergerboos(beta=1e-4, maximum=9_999):
    """Review 2, item 2: Berger and Boos's bound over the registration's rank interval, against the plain tests."""
    print(f"\n== bergerboos, validity: beta = {beta:g}, 9 registrations across each gene's {1 - beta:.2%} shift "
          f"interval, up to {maximum:,} shuffles; share at p <= 0.05 / 0.01 / 0.001")
    rng = np.random.default_rng(2026093006)
    for n1, n0, genes in ((20, 20, 6000), (100, 100, 3000)):
        cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
        print(f"{n1}:{n0}: Mann-Whitney cutoff c = {mann_whitney_cutoff(n1, n0, beta)} of {n1 * n0} pairwise differences")
        for model in ("bimodal sd 0.3 separation 2", "bimodal sd 0.03 separation 5", "Gamma", "Gamma + 1% x100"):
            x = null_matrix(rng, genes, n1, n0, model)
            cells = [f"{name} {rates(subset_pvalues(x, cond, 'linear', name, maximum=maximum, seed=1))}"
                     for name in ("proposed", "lorenz")]
            cells += [f"{name}, Berger-Boos {rates(bb_pvalues(x, cond, name, beta=beta, maximum=maximum, seed=1))}"
                      for name in ("proposed", "lorenz")]
            print(f"{n1:3d}:{n0:<3d} {model:28s} " + " | ".join(cells), flush=True)
    print(f"\n== bergerboos, power: Gamma-Poisson (CV^2 0.5) at baseline 50, 400 genes per cell, up to {maximum:,} "
          f"shuffles; share of planted genes at p <= 0.01 / 0.001")
    for n1, n0 in ((20, 20), (50, 500), (250, 250)):
        cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
        for label, share, members, majority in PATTERNS:
            x = planted_matrix(rng, n1, n0, share, members, majority)
            cells = [f"{name} {rates(subset_pvalues(x, cond, 'linear', name, maximum=maximum, seed=1), (0.01, 0.001))}"
                     for name in ("proposed", "lorenz")]
            cells += [f"{name}, Berger-Boos "
                      f"{rates(bb_pvalues(x, cond, name, beta=beta, maximum=maximum, seed=1), (0.01, 0.001))}"
                      for name in ("proposed", "lorenz")]
            print(f"{n1:3d}:{n0:<3d} {label:36s} " + " | ".join(cells), flush=True)



def bimodal_splits(gene, n1, n0, splits, seed=4000):
    """A TCGA prostate gene's tumour values under random splits, one split per row."""
    d = tcga("prostate")
    v = d["x"][d["names"] == gene][0][d["cond"] == 1]
    rng = np.random.default_rng([seed, n1, n0])
    return np.stack([v[rng.choice(len(v), n1 + n0, replace=False)] for _ in range(splits)])


def registrations(maximum=9_999):
    """Review 2, question 2: the W1 registration against the difference of log-medians and the Hodges-Lehmann
    shift (the median of all pairwise case-minus-control differences of log(x + 1)), each with the Lorenz
    statistic; 'oracle' is the same statistic with no registration, exact at fold 1."""
    names = {"lorenz": "W1", "lorenz, log-median registration": "log-median",
             "lorenz, Hodges-Lehmann registration": "Hodges-Lehmann"}
    rng = np.random.default_rng(2026093007)
    print("\n== registrations, error: sd of the log registration at fold 1 (1,500 genes per cell)")
    for n1, n0 in ((20, 20), (50, 500), (500, 50), (250, 250)):
        cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
        for model in ("Gamma", "Gamma + 1% x100", "bimodal sd 0.3 separation 2"):
            case, control = _sorted(null_matrix(rng, 1500, n1, n0, model), cond)
            cells = [f"{short} {np.std(np.log(registration(case, control, 'linear', CONSTRUCTIONS[name][0]))):.3f}"
                     for name, short in names.items()]
            print(f"{n1:3d}:{n0:<3d} {model:28s} " + " | ".join(cells), flush=True)

    print(f"\n== registrations, validity: simulated nulls at fold 1 (up to {maximum:,} shuffles); share at "
          f"p <= 0.05 / 0.01 / 0.001")
    for n1, n0, genes in ((20, 20, 6000), (50, 50, 6000), (50, 500, 3000), (500, 50, 3000)):
        cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
        for model in ("bimodal sd 0.3 separation 2", "bimodal sd 0.03 separation 5", "Gamma", "Gamma + 1% x100",
                      "Gamma + 0.5% x500"):
            x = null_matrix(rng, genes, n1, n0, model)
            cells = [f"{short} {rates(subset_pvalues(x, cond, 'linear', name, maximum=maximum, seed=1))}"
                     for name, short in names.items()]
            cells.append(f"oracle {rates(subset_pvalues(x, cond, 'linear', 'exact, lorenz', maximum=maximum, seed=1))}")
            print(f"{n1:3d}:{n0:<3d} {model:28s} " + " | ".join(cells), flush=True)

    print(f"\n== registrations, real bimodal genes: TCGA prostate tumours, 5,000 random splits per design; share at "
          f"p <= 0.05 / 0.01 / 0.001")
    for gene in ("GSTM1", "ERAP2"):
        for n1, n0 in ((20, 20), (50, 50), (100, 100)):
            rows, lab = bimodal_splits(gene, n1, n0, 5000), np.r_[np.ones(n1, int), np.zeros(n0, int)]
            cells = [f"{short} {rates(subset_pvalues(rows, lab, 'linear', name, maximum=maximum, seed=1))}"
                     for name, short in names.items()]
            cells.append(f"oracle {rates(subset_pvalues(rows, lab, 'linear', 'exact, lorenz', maximum=maximum, seed=1))}")
            print(f"{gene:5s} {n1:3d}:{n0:<3d} " + " | ".join(cells), flush=True)

    print(f"\n== registrations, TCGA prostate genes under random labels (8 labellings, up to {maximum:,} shuffles); "
          f"share at p <= 0.05 / 0.01 / 0.001, all genes / heavy genes")
    d = tcga("prostate")
    x, n = d["x"], d["x"].shape[1]
    big = heavy(x)
    for n1 in (445, 52):
        labels = []
        for r in range(8):
            lab = np.zeros(n, int)
            lab[np.random.default_rng(2000 + r).choice(n, n1, replace=False)] = 1
            labels.append(lab)
        for name, short in list(names.items()) + [("exact, lorenz", "oracle")]:
            p = np.array([subset_pvalues(x, lab, "linear", name, maximum=maximum, seed=r) for r, lab in enumerate(labels)])
            print(f"{n1}:{n - n1}  {short:15s} all genes {rates(p)}   heavy genes {rates(p[:, big])}", flush=True)

    print(f"\n== registrations, power: Gamma-Poisson at baseline 50, 400 planted genes per cell (up to {maximum:,} "
          f"shuffles); share at p <= 0.01 / 0.001; 'oracle' registers by the majority's latent fold")
    for n1, n0 in ((50, 500), (250, 250)):
        cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
        for label, share, members, majority in PATTERNS:
            x = planted_matrix(rng, n1, n0, share, members, majority)
            cells = [f"{short} {rates(subset_pvalues(x, cond, 'linear', name, maximum=maximum, seed=1), (0.01, 0.001))}"
                     for name, short in names.items()]
            p = subset_pvalues(x, cond, "linear", "lorenz", maximum=maximum, seed=1, fold=np.full(len(x), majority))
            cells.append(f"oracle {rates(p, (0.01, 0.001))}")
            print(f"{n1:3d}:{n0:<3d} {label:36s} " + " | ".join(cells), flush=True)


def matched(maximum=9_999, alpha=0.01):
    """Review 2, questions 3 and 5: power at matched actual size. A gene's null shape is unknown, so each test's
    cut is the largest p-value threshold that keeps every simulated null family (four shapes, folds 1 and 2)
    at or below alpha; power is read at that cut and at alpha itself."""
    tests = (("released", "linear"), ("proposed", "linear"), ("lorenz", "linear"), ("W1 + depth", "log2"))
    label = {t: t[0] if t[1] == "linear" else f"{t[0]} (log2)" for t in tests}
    families = [(m, f) for m in ("Gamma", "lognormal", "Gamma + 1% x100", "Gamma + 0.5% x500") for f in (1.0, 2.0)]
    rng = np.random.default_rng(2026093008)
    print(f"\n== matched: 3,000 null genes per family, 400 planted genes per pattern (Gamma-Poisson, baseline 50), up to "
          f"{maximum:,} shuffles; alpha = {alpha}")
    for n1, n0 in ((50, 500), (500, 50), (250, 250)):
        cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
        cut = {t: (1.0, None) for t in tests}
        for model, fold in families:
            x = simulate(rng, 3000, n1, n0, model, fold)
            for t in tests:
                p = np.sort(subset_pvalues(x if t[1] == "linear" else np.log2(x + 1), cond, t[1], t[0],
                                           maximum=maximum, seed=1))
                values = np.unique(p)
                ok = values[np.searchsorted(p, values, side="right") <= alpha * len(p)]
                c = ok.max() if len(ok) else 0.0
                if c < cut[t][0]:
                    cut[t] = (c, f"{model} x{fold:g}")
        print(f"{n1}:{n0} cuts: " + " | ".join(f"{label[t]} {c:.2g} ({fam})" for t, (c, fam) in cut.items()), flush=True)
        for name, share, members, majority in PATTERNS:
            x = planted_matrix(rng, n1, n0, share, members, majority)
            cells = []
            for t in tests:
                p = subset_pvalues(x if t[1] == "linear" else np.log2(x + 1), cond, t[1], t[0], maximum=maximum, seed=1)
                cells.append(f"{label[t]} {np.mean(p <= alpha):.0%} -> {np.mean(p <= cut[t][0]):.0%}")
            print(f"  {name:36s} power at alpha -> at the cut: " + " | ".join(cells), flush=True)



def mechanism2(genes=30000, maximum=9_999):
    """Review 2's bimodal null (20:20, sd 0.3, separation 2, the first seed): rejections by the groups'
    difference in upper-mode counts, by direction, and the registration's dependence on that difference."""
    n1 = n0 = 20
    rng = np.random.default_rng(921173)
    high = rng.random((genes, n1 + n0)) < .5
    x = np.exp(rng.normal(3, 0.3, (genes, n1 + n0)) + 2 * high)
    cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
    case, control = _sorted(x, cond)
    logfold = np.log(registration(case, control, "linear", "w1"))
    imbalance = high[:, :n1].sum(axis=1) - high[:, n1:].sum(axis=1)
    p = {}
    for name in ("proposed", "lorenz"):
        for direction in ("both", "up", "down"):
            p[name, direction] = subset_pvalues(x, cond, "linear", name, maximum=maximum, seed=11, direction=direction)
            p[name, direction, "oracle"] = subset_pvalues(x, cond, "linear", name, maximum=maximum, seed=11,
                                                          direction=direction, fold=np.ones(genes))
    print(f"\n== mechanism2: bimodal null, 20:20, sd 0.3, separation 2, {genes:,} genes (seed 921173), up to {maximum:,} "
          f"shuffles. Imbalance = cases' minus controls' upper-mode count")
    print(f"correlation of the log W1 fold with the imbalance {np.corrcoef(logfold, imbalance)[0, 1]:.2f}; "
          f"sd of the log W1 fold {logfold.std():.3f} (within-mode sd 0.3)")
    print("share at p <= 0.01 by |imbalance|: estimated registration / true fold")
    for a in range(0, 12, 2):
        k = (np.abs(imbalance) >= a) & (np.abs(imbalance) < a + 2)
        cells = [f"{name} {np.mean(p[name, 'both'][k] <= .01):.1%} / {np.mean(p[name, 'both', 'oracle'][k] <= .01):.1%}"
                 for name in ("proposed", "lorenz")]
        print(f"  [{a:2d}, {a + 2:2d}): {k.sum():6d} genes, mean |log fold| {np.abs(logfold[k]).mean():.2f}   "
              + " | ".join(cells))
    print("by direction, share at p <= 0.05 / 0.01 / 0.001: estimated registration | true fold")
    for name in ("proposed", "lorenz"):
        for direction in ("up", "down"):
            print(f"  {name:8s} {direction:4s} {rates(p[name, direction])} | {rates(p[name, direction, 'oracle'])}")
    lx = np.log1p(x)
    pooled = np.where(cond == 1, lx - logfold[:, None], lx)

    def modes(v):
        up, low = np.where(high, v, np.nan), np.where(~high, v, np.nan)
        return (np.nanmean(up, axis=1) - np.nanmean(low, axis=1),
                np.sqrt((np.nanvar(up, axis=1) + np.nanvar(low, axis=1)) / 2))
    gap, spread = modes(lx)
    gap_pooled, spread_pooled = modes(pooled)
    print("the registered pool's shape on log(x + 1): gap between the mode means, and the within-mode sd, "
          "before -> after registration")
    for a, b in ((-20, -6), (-6, -2), (-2, 3), (3, 7), (7, 21)):
        k = (imbalance >= a) & (imbalance < b)
        print(f"  imbalance [{a:3d}, {b:3d}): {k.sum():6d} genes, gap {gap[k].mean():.2f} -> {gap_pooled[k].mean():.2f}, "
              f"sd {spread[k].mean():.2f} -> {spread_pooled[k].mean():.2f}")
    only = (p["proposed", "both"] <= .01) & (p["proposed", "both", "oracle"] > .01)
    print(f"rejected at p <= 0.01 with the estimated registration but not the true fold: {only.sum()} genes; "
          f"cases held fewer upper-mode samples in {np.mean(imbalance[only] < 0):.1%}, and the median case sat in the "
          f"lower mode in {np.mean(high[only, :n1].sum(axis=1) < n1 / 2):.1%}; mean log W1 fold {logfold[only].mean():.2f}")



def budget():
    """The study's power runs cap shuffles at 99,999 (p >= 1e-5). Five raised cases among 50 v 500 can reach
    C(50, 5) / C(550, 5) = 5e-6, so the cap may censor the registered tests' power; rerun at the package default."""
    power(names=("released", "proposed", "lorenz"), maximum=999_999, designs=("50:500",))



def large250(batch=2500, maximum=9_999):
    """Review 3: the first bimodal null at 250 v 250 over 200,000 genes, the package construction (Lorenz, midpoint
    W1) against the same statistic without registration (exact here, the true fold being 1), each batch with one
    shared shuffle stream so that the two are paired. The reviewer's design and data seeds."""
    n = 250
    levels = np.array([.05, .01, .001])
    cond = np.r_[np.ones(n, int), np.zeros(n, int)]
    names = ("lorenz", "exact, lorenz")
    counts, discord, rows = np.zeros((2, 3), int), np.zeros((2, 3), int), []
    for data_seed, genes, stream in ((927202601, 50000, 991), (927202602, 150000, 1991)):
        rng = np.random.default_rng(data_seed)
        for b in range(genes // batch):
            x = bimodal_matrix(rng, batch, 2 * n, .3, 2)
            hit = np.array([subset_pvalues(x, cond, "linear", name, maximum=maximum, seed=stream + b)[:, None] <= levels
                            for name in names])
            counts += hit.sum(axis=1)
            discord += np.array([(hit[0] & ~hit[1]).sum(axis=0), (hit[1] & ~hit[0]).sum(axis=0)])
            rows.append(hit.sum(axis=1) / batch)
    rates = np.array(rows)                                           # batches x constructions x levels
    total = len(rows) * batch
    se = rates.std(axis=0, ddof=1) / np.sqrt(len(rows))
    diff = rates[:, 0] - rates[:, 1]
    print(f"\n== large250: bimodal null (sd 0.3, separation 2), {n} v {n}, {total:,} genes in {len(rows)} batches of "
          f"{batch:,}, up to {maximum:,} shuffles; share at p <= 0.05 / 0.01 / 0.001 (standard error across batches)")
    for k, name in enumerate(("W1 midpoint + Lorenz (the package)", "no registration + Lorenz (oracle)")):
        print(f"  {name:36s} " + " / ".join(f"{c / total:.4%} ({e:.4%})" for c, e in zip(counts[k], se[k])))
    print("  paired difference            " + " / ".join(f"{d.mean():+.4%} ({d.std(ddof=1) / np.sqrt(len(rows)):.4%})"
                                                         for d in diff.T))
    print("  rejected by one only (package / oracle): " + " | ".join(f"{a} / {b}" for a, b in zip(*discord)))


SECTIONS = dict(checks=checks, foldform=foldform, realnull=realnull, genes=genes, simnull=simnull,
                mechanism=mechanism, power=power, truelabels=truelabels, thinning=thinning, lowcount=lowcount,
                normalizers=normalizers, thinnednull=thinnednull, thinfixed=thinfixed, review2=review2,
                lorenz=lorenz, bimodalgenes=bimodalgenes, bergerboos=bergerboos, registrations=registrations,
                matched=matched, mechanism2=mechanism2, budget=budget, large250=large250)
EXTRA = ("normalizers", "thinnednull", "thinfixed", "review2", "lorenz", "bimodalgenes", "bergerboos", "registrations",
         "matched", "mechanism2", "budget", "large250")

if __name__ == "__main__":
    for section in sys.argv[1:] or [k for k in SECTIONS if k not in EXTRA]:
        SECTIONS[section]()
