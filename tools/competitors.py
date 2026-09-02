"""The subset-detection methods WADE is measured against, and one harness.

`notebooks/benchmark.qmd` is the deliverable; this file is the part of it that
wants tests rather than prose. **Nothing here is part of WADE** — it is not
imported by `wade`, it ships in no wheel, and it exists so that a comparison
can be made against the literature rather than against a description of it.

The five outlier statistics were published between 2005 and 2010 for
log-scale microarray data, and none has a maintained implementation. They are
reimplemented here from their papers, each with its formula in the docstring
so a reader can check the code against the source, and `verify()` holds each
to a value derived independently of the code — mostly its exact expectation
under a standard normal null, which every one of them has in closed form.

**The comparison is of statistics, not of inference.** `score_all()` computes
every statistic, every tuning value and both tails in one pass over the data,
and `permutation_test()` runs that one pass over a shared permutation matrix,
giving all of them the same empirical p-value with the same GPD refinement,
the same BH and the same ranking z — WADE's own, from `wade.pvalues` and
`wade.api`. What differs between two rows of a results table is then the
statistic and nothing else.

Each statistic also exists as a **standalone function** carrying its paper's
formula in its docstring, which is what `verify()` checks; a final check holds
the fused pass to those, because otherwise the verified code and the executed
code would be free to drift apart.

**Two-sidedness is imposed uniformly.** COPA, OS, ORT and MOST were defined to
find *elevated* outliers; WADE's default is two-sided. Running a one-sided
statistic against a two-sided one on a scenario containing downward subsets
would be scoring the scenario, not the method, so each is made two-sided the
way OS, ORT and MOST's own authors did it: take the larger of the statistic
and the statistic on the negated data.

**Scale.** These methods assume roughly symmetric, homoscedastic data, which
for expression means the log scale. `benchmark.qmd` hands every competitor
`log2(tpm + 1)` — the same normalized matrix WADE built internally, on the
scale the competitors' authors wrote for. WADE itself gets the raw counts,
because that is its documented entry point.
"""

from __future__ import annotations

import numpy as np

__all__ = ["TUNED", "NOT_A_TEST", "score_all", "permutation_test", "waddr",
           "verify",
           "copa", "outlier_sum", "ort", "most", "lsoss", "t_stat", "wilcoxon"]


# ---------------------------------------------------------------------------
# shared pieces


def _mad(a, center=None, scale=1.4826):
    """Median absolute deviation along the last axis, kept 2-D.

    ``scale=1.4826`` makes it a consistent estimator of sigma for normal data,
    which is what COPA and OS assume when they call their transform a z-score,
    and they are its only callers. ORT and MOST need a *pooled* deviation —
    each group about its own median — which is not this function; they compute
    it inline, ORT unscaled and MOST scaled, for the reason given in `most`.
    """
    if center is None:
        center = np.median(a, axis=1, keepdims=True)
    return scale * np.median(np.abs(a - center), axis=1, keepdims=True)


def _two_sided(fn):
    """The larger of the statistic and the statistic on negated data.

    OS, ORT and MOST define their two-sided forms exactly this way (a lower
    tail mirrored onto the upper one and the maximum taken). Applying the same
    rule to COPA keeps the comparison about the statistic rather than about
    which tail each author happened to write down.
    """
    def wrapped(X, cond, *args, two_sided=True, **kw):
        up = fn(X, cond, *args, **kw)
        if not two_sided:
            return up
        return np.maximum(up, fn(-X, cond, *args, **kw))
    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped


# ---------------------------------------------------------------------------
# the five outlier statistics


@_two_sided
def copa(X, cond, r=75.0):
    """Cancer Outlier Profile Analysis — Tomlins et al., *Science* 310 (2005).

        z_ij = (x_ij - med_j(x)) / mad_j(x)          over ALL samples
        COPA_j = the r-th percentile of { z_ij : i in cases }

    The median and MAD are taken over both groups, so a gene whose cases are
    globally shifted moves its own centre — the property the later methods
    were written to remove. **`r` is a tuning parameter**: Tomlins reports 75,
    90 and 95 and the choice bounds the smallest subset findable, since the
    r-th percentile of the case group cannot see a subset smaller than
    `1 - r/100` of it.
    """
    med = np.median(X, axis=1, keepdims=True)
    Z = (X - med) / _mad(X, center=med)
    return np.percentile(Z[:, cond == 1], r, axis=1)


@_two_sided
def outlier_sum(X, cond, k=1.0):
    """Outlier Sum — Tibshirani & Hastie, *Biostatistics* 8 (2007).

        z as in COPA (median and MAD over all samples)
        q25, q75, IQR  of z over ALL samples
        OS_j = sum over cases of z_ij * 1(z_ij > q75 + k*IQR)

    The improvement over COPA is summing the outliers rather than reading one
    percentile, so evidence accumulates with the number of affected samples.
    **`k` is a tuning parameter**; Tibshirani and Hastie use 1.
    """
    med = np.median(X, axis=1, keepdims=True)
    Z = (X - med) / _mad(X, center=med)
    q25, q75 = np.percentile(Z, [25, 75], axis=1, keepdims=True)
    C = Z[:, cond == 1]
    return np.where(C > q75 + k * (q75 - q25), C, 0.0).sum(axis=1)


@_two_sided
def ort(X, cond, k=1.0):
    """Outlier Robust T-statistic — Wu, *Biostatistics* 8 (2007).

        med_n = median of controls,  med_t = median of cases
        s = median{ |x - med_n| over controls, |x - med_t| over cases }
        q25, q75, IQR  of the CONTROLS only
        ORT_j = sum over cases of (x_ij - med_n) / s * 1(x_ij > q75 + k*IQR)

    Wu's objection to OS is that centring on all samples lets the outliers
    move their own reference point, so both the centre and the outlier
    threshold here come from the controls alone. The scale is pooled, each
    group about its own median, and is **not** rescaled by 1.4826 — Wu's form,
    and harmless here because a constant multiple of a statistic cancels out
    of a permutation p-value. `most` cannot take the same liberty; see there.
    """
    case, ctrl = X[:, cond == 1], X[:, cond == 0]
    med_n = np.median(ctrl, axis=1, keepdims=True)
    s = np.median(np.abs(np.concatenate(
        [ctrl - med_n, case - np.median(case, axis=1, keepdims=True)], axis=1)),
        axis=1, keepdims=True)
    q25, q75 = np.percentile(ctrl, [25, 75], axis=1, keepdims=True)
    hit = case > q75 + k * (q75 - q25)
    return np.where(hit, (case - med_n) / s, 0.0).sum(axis=1)


#: Replicates behind the MOST order-statistic tabulation. Named because
#: `verify()` needs it to state its own tolerance: the Monte Carlo standard
#: error on `mu_k` is `sqrt(n / _MOST_REPS)`.
_MOST_REPS = 20_000


def _most_moments(n, reps=_MOST_REPS, seed=20260902, _cache={}):
    """Mean and sd of the partial sums of descending N(0,1) order statistics.

    Lian's `T_k` standardizes the sum of the k largest case values by the
    moments of that sum under the null. They have no usable closed form, so
    Lian tabulates them by simulation and so does this — once per group size,
    at a fixed seed, cached. 20,000 replicates put the Monte Carlo error on
    each moment near 1e-2 of a standard deviation, an order below the
    differences the benchmark reads.
    """
    if n not in _cache:
        z = np.random.default_rng(seed).standard_normal((reps, n))
        c = np.cumsum(-np.sort(-z, axis=1), axis=1)
        _cache[n] = (c.mean(axis=0), c.std(axis=0, ddof=1))
    return _cache[n]


@_two_sided
def most(X, cond):
    """Maximum Ordered Subset T-statistic — Lian, *Biostatistics* 9 (2008).

        z_ij = (x_ij - med_n) / s        centre and scale as in ORT
        y_(1) >= ... >= y_(n1)           the cases' z, descending
        T_k = ( sum_{i<=k} y_(i) - mu_k ) / sigma_k
        MOST_j = max over k of T_k

    with `mu_k`, `sigma_k` the null moments of that partial sum (see
    `_most_moments`). **MOST has no tuning parameter** — maximizing over `k`
    is what replaces choosing one — which is the property it shares with WADE
    and the reason it is the most interesting competitor here.

    Unlike ORT's, this scale estimate must be **consistent for sigma**, hence
    the 1.4826: `mu_k` and `sigma_k` are absolute constants, so an inflated
    `z` is not merely rescaled, it changes which `k` attains the maximum. ORT
    can use the raw median deviation because a constant multiple cancels out
    of a permutation p-value; here it does not. Caught by `verify()` — the
    unscaled version gave `z` an sd of 1.50 and `T_1` a mean of 3.3.
    """
    case, ctrl = X[:, cond == 1], X[:, cond == 0]
    med_n = np.median(ctrl, axis=1, keepdims=True)
    s = 1.4826 * np.median(np.abs(np.concatenate(
        [ctrl - med_n, case - np.median(case, axis=1, keepdims=True)], axis=1)),
        axis=1, keepdims=True)
    csum = np.cumsum(-np.sort(-(case - med_n) / s, axis=1), axis=1)
    mu, sd = _most_moments(case.shape[1])
    return ((csum - mu) / sd).max(axis=1)


@_two_sided
def lsoss(X, cond):
    """Least Sum of Ordered Subset Square — Wang & Rekaya, *BMC Bioinf.* 11 (2010).

    Sort the cases descending and split them at the `k` minimizing the
    within-part sum of squares,

        SSE(k) = SS(y_(1..k)) + SS(y_(k+1..n1)),   k = 1 .. n1-1

    then compare the "activated" part against the controls with a t-statistic
    whose variance is pooled over all three groups (controls, activated,
    unaffected) on `n0 + n1 - 3` degrees of freedom:

        LSOSS_j = (mean(activated) - mean(controls)) / (sigma * sqrt(1/k + 1/n0))

    Unlike MOST it estimates the subset boundary rather than maximizing over
    it, which makes `k` an output — the one competitor that reports anything
    comparable to `affected_fraction`, and `benchmark.qmd` reads it as such.
    **No tuning parameter.**
    """
    return _lsoss_full(X, cond)[0]


def _lsoss_full(X, cond):
    """`(statistic, k_hat)` — the split point is wanted on its own as well."""
    case, ctrl = X[:, cond == 1], X[:, cond == 0]
    n1, n0 = case.shape[1], ctrl.shape[1]
    y = -np.sort(-case, axis=1)                       # descending

    cs, cs2 = np.cumsum(y, axis=1), np.cumsum(y * y, axis=1)
    tot, tot2 = cs[:, -1:], cs2[:, -1:]
    k = np.arange(1, n1)                              # split after k cases
    top = cs2[:, :-1] - cs[:, :-1] ** 2 / k           # SS of the top k
    bot = (tot2 - cs2[:, :-1]) - (tot - cs[:, :-1]) ** 2 / (n1 - k)
    kbest = np.argmin(top + bot, axis=1)              # index into k
    g = np.arange(len(y))
    khat = kbest + 1

    sse_case = (top + bot)[g, kbest]
    mean_top = cs[g, kbest] / khat
    mean_ctrl = ctrl.mean(axis=1)
    sse_ctrl = ((ctrl - mean_ctrl[:, None]) ** 2).sum(axis=1)
    sigma = np.sqrt((sse_case + sse_ctrl) / (n0 + n1 - 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = (mean_top - mean_ctrl) / (sigma * np.sqrt(1.0 / khat + 1.0 / n0))
    return np.nan_to_num(stat, nan=0.0, posinf=0.0, neginf=0.0), khat


# ---------------------------------------------------------------------------
# the two floors


def t_stat(X, cond, two_sided=True):
    """Pooled two-sample t. The first-moment floor every DE method clears.

    Reported here because a subset is the case it is *worst* at, and for a
    reason worth seeing: the affected samples raise the case group's variance
    as well as its mean, so the effect enters the numerator once and the
    denominator too.
    """
    case, ctrl = X[:, cond == 1], X[:, cond == 0]
    n1, n0 = case.shape[1], ctrl.shape[1]
    sp = np.sqrt((case.var(axis=1, ddof=1) * (n1 - 1)
                  + ctrl.var(axis=1, ddof=1) * (n0 - 1)) / (n1 + n0 - 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (case.mean(axis=1) - ctrl.mean(axis=1)) / (sp * np.sqrt(1 / n1 + 1 / n0))
    t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return np.abs(t) if two_sided else t


def wilcoxon(X, cond, two_sided=True):
    """Standardized rank-sum. The distribution-free location floor.

    Ranks run within each gene across all samples with ties averaged, and the
    case-group sum is standardized by its exact null moments, so genes are
    comparable before the permutation ever runs. `scipy.stats.rankdata` does
    the tie averaging along an axis in C; the same thing written as a Python
    loop over genes cost 79 ms against 4 ms per call at 1,000 genes, and this
    function is called once per permutation.
    """
    from scipy.stats import rankdata

    n = X.shape[1]
    n1 = int((cond == 1).sum())
    n0 = n - n1
    W = rankdata(X, axis=1)[:, cond == 1].sum(axis=1)
    z = (W - n1 * (n + 1) / 2.0) / np.sqrt(n1 * n0 * (n + 1) / 12.0)
    return np.abs(z) if two_sided else z


# ---------------------------------------------------------------------------
# one pass computes everything


#: The parameter each method makes you choose. MOST, LSOSS and both WADE
#: stages are absent from this table on purpose — that absence is the figure.
TUNED = {"COPA": ("r", (75.0, 90.0, 95.0)),
         "OS":   ("k", (0.5, 1.0, 2.0)),
         "ORT":  ("k", (0.5, 1.0, 2.0))}


def _upper(X, cond):
    """Every statistic's upper-tail form, sharing what they share.

    One function rather than eight because the permutation loop calls this
    2,000 times: COPA and OS want the same all-sample z, ORT and MOST the same
    control-centred scale, MOST and LSOSS the same descending sort. Computing
    each method independently repeated all three. The whole tuning grid comes
    out of the same pass too — three COPA percentiles are one `np.percentile`
    call, not three.
    """
    case_i, ctrl_i = cond == 1, cond == 0
    n1, n0 = int(case_i.sum()), int(ctrl_i.sum())
    case, ctrl = X[:, case_i], X[:, ctrl_i]
    out = {}

    # --- pooled t and the rank sum: signed, so the wrapper's max gives |.| --
    sp = np.sqrt((case.var(axis=1, ddof=1) * (n1 - 1)
                  + ctrl.var(axis=1, ddof=1) * (n0 - 1)) / (n1 + n0 - 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["t-test"] = np.nan_to_num(
            (case.mean(axis=1) - ctrl.mean(axis=1)) / (sp * np.sqrt(1 / n1 + 1 / n0)),
            nan=0.0, posinf=0.0, neginf=0.0)

    from scipy.stats import rankdata
    W = rankdata(X, axis=1)[:, case_i].sum(axis=1)
    out["Wilcoxon"] = ((W - n1 * (X.shape[1] + 1) / 2.0)
                       / np.sqrt(n1 * n0 * (X.shape[1] + 1) / 12.0))

    # --- COPA and OS share the all-sample z --------------------------------
    med = np.median(X, axis=1, keepdims=True)
    Z = (X - med) / _mad(X, center=med)
    Zc = Z[:, case_i]
    rs = TUNED["COPA"][1]
    for r, v in zip(rs, np.percentile(Zc, rs, axis=1)):
        out[f"COPA r={r:g}"] = v
    q25, q75 = np.percentile(Z, [25, 75], axis=1, keepdims=True)
    for k in TUNED["OS"][1]:
        out[f"OS k={k:g}"] = np.where(Zc > q75 + k * (q75 - q25), Zc, 0.0).sum(axis=1)

    # --- ORT and MOST share the control centre and the pooled scale --------
    med_n = np.median(ctrl, axis=1, keepdims=True)
    dev = np.median(np.abs(np.concatenate(
        [ctrl - med_n, case - np.median(case, axis=1, keepdims=True)], axis=1)),
        axis=1, keepdims=True)
    E = (case - med_n) / dev                          # Wu's unscaled form
    cq25, cq75 = np.percentile(ctrl, [25, 75], axis=1, keepdims=True)
    for k in TUNED["ORT"][1]:
        out[f"ORT k={k:g}"] = np.where(
            case > cq75 + k * (cq75 - cq25), E, 0.0).sum(axis=1)

    # MOST needs the sigma-consistent scale; see `most`.
    srt = -np.sort(-E, axis=1) / 1.4826
    csum = np.cumsum(srt, axis=1)
    mu, sd = _most_moments(n1)
    out["MOST"] = ((csum - mu) / sd).max(axis=1)

    # --- LSOSS reuses that descending sort ---------------------------------
    y = -np.sort(-case, axis=1)
    cs, cs2 = np.cumsum(y, axis=1), np.cumsum(y * y, axis=1)
    tot, tot2 = cs[:, -1:], cs2[:, -1:]
    kk = np.arange(1, n1)
    top = cs2[:, :-1] - cs[:, :-1] ** 2 / kk
    bot = (tot2 - cs2[:, :-1]) - (tot - cs[:, :-1]) ** 2 / (n1 - kk)
    kbest = np.argmin(top + bot, axis=1)
    g = np.arange(len(y))
    khat = kbest + 1
    mean_ctrl = ctrl.mean(axis=1)
    sigma = np.sqrt(((top + bot)[g, kbest]
                     + ((ctrl - mean_ctrl[:, None]) ** 2).sum(axis=1)) / (n0 + n1 - 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["LSOSS"] = np.nan_to_num(
            (cs[g, kbest] / khat - mean_ctrl) / (sigma * np.sqrt(1.0 / khat + 1.0 / n0)),
            nan=0.0, posinf=0.0, neginf=0.0)
    out["LSOSS k"] = khat.astype(float)               # carried, never tested
    return out


#: Statistics `_upper` computes that are descriptive rather than evidence.
NOT_A_TEST = ("LSOSS k",)


def score_all(X, cond):
    """`{name: score per gene}` for every statistic, both tails, one pass.

    Each method appears twice: under its own name, in the **upper-tail form
    its authors published**, and suffixed `" 2s"` for the two-sided form —
    the larger of the statistic and the statistic on negated data, which is
    how OS, ORT and MOST's own authors define theirs, and which for the
    signed `t` and rank sum is exactly the absolute value.

    Both are reported because neither alone is a fair comparison. WADE is
    two-sided by default, so the `2s` column is the like-for-like one; but a
    reader is entitled to see that the competitors keep their home-turf form
    on the upward scenarios they were written for, and that WADE's margin
    survives it.
    """
    up, dn = _upper(X, cond), _upper(-X, cond)
    out = {}
    for name, v in up.items():
        if name in NOT_A_TEST:
            out[name] = v
            continue
        out[name] = v
        out[name + " 2s"] = np.maximum(v, dn[name])
    return out


def permutation_test(X, cond, perms, *, scorer=score_all, n_exc_min=10, n_tail=250):
    """`{name: (p, padj, stat, z)}` for every statistic, through WADE's inference.

    One sweep over `perms` scores everything, so two methods' p-values differ
    only where their statistics do — same permutations, same empirical p with
    the same GPD refinement, same BH, all from `wade.pvalues`.

    Each **row of `perms` is a permuted label vector**, not an index vector
    (`wade.draw_perms`' docstring says so, and it is what R's `sample(cond)`
    returns). Indexing `cond` by a row instead would silently produce a
    degenerate split, so the shape and group sizes are checked once here.

    The fourth element is the **permutation z**, `(obs - mean(null)) / sd(null)`
    against the gene's own null — `wade.api._perm_z`, imported rather than
    rewritten so the two cannot drift. It is what ranking has to use. A raw
    COPA score or outlier sum is not comparable between genes, because each
    carries its own gene's median absolute deviation in its units; measured on
    the benchmark cohort, ranking by the raw statistic put COPA's sensitivity
    at 0.00 while its power was 67%, which says nothing about COPA and
    everything about the scale. WADE's own ranking column is a permutation z
    (`method.md` 4, 6), so every method gets one.
    """
    from wade.api import _perm_z            # the same z WADE ranks by
    from wade.pvalues import bh_adjust, perm_pvalues

    perms = np.asarray(perms)
    n1 = int((cond == 1).sum())
    if perms.ndim != 2 or perms.shape[1] != X.shape[1]:
        raise ValueError(f"perms must be (B, n_samples); got {perms.shape} "
                         f"for {X.shape[1]} samples")
    if not np.all(perms.sum(axis=1) == n1):
        raise ValueError("perms rows must be permuted LABELS with the group "
                         "sizes preserved, not index vectors")

    obs = scorer(X, cond)
    names = [n for n in obs if n not in NOT_A_TEST]
    null = {n: np.empty((X.shape[0], perms.shape[0])) for n in names}
    for b in range(perms.shape[0]):
        sb = scorer(X, perms[b])
        for n in names:
            null[n][:, b] = sb[n]

    out = {}
    for n in names:
        p, _, _ = perm_pvalues(obs[n], null[n], n_exc_min=n_exc_min,
                               n_tail=n_tail, alternative="greater")
        out[n] = (p, bh_adjust(p), obs[n], _perm_z(obs[n], null[n]))
    for n in NOT_A_TEST:
        out[n] = (None, None, obs[n], None)
    return out


# ---------------------------------------------------------------------------
# waddR, the nearest relative, through R


#: The R this shells out to. waddR is a Bioconductor package, so there is no
#: Python equivalent to reimplement from and no reason to try: it is the one
#: competitor whose real implementation can simply be run.
_WADDR_R = r"""
suppressMessages(library(waddR))
a <- commandArgs(trailingOnly = TRUE)
X <- as.matrix(read.table(a[1]))
cond <- scan(a[2], quiet = TRUE)
permnum <- as.integer(a[3])
set.seed(as.integer(a[4]))
keep <- c("pval", "d.wass", "location", "size", "shape",
          "perc.loc", "perc.size", "perc.shape")
res <- t(apply(X, 1, function(v)
    waddR::wasserstein.test(v[cond == 1], v[cond == 0],
                            method = "SP", permnum = permnum)[keep]))
colnames(res) <- keep
write.table(res, a[5], sep = "\t", quote = FALSE, row.names = FALSE)
"""


def waddr(X, cond, *, permnum=2000, seed=1, rscript="Rscript"):
    """waddR's two-sample Wasserstein test, per gene. Returns a dict of arrays.

    Schefzik, Flesch & Goncalves, *Nat. Commun.* 12 (2021) — the nearest
    published relative to WADE, and the only competitor here that is **run
    rather than reimplemented**, because it is a maintained Bioconductor
    package. Install with `BiocManager::install("waddR")`.

    Its `method="SP"` is a semi-parametric permutation test with a
    generalized-Pareto tail refinement — the same inference shape as WADE's,
    arrived at independently — so it is excluded from `permutation_test`'s
    common harness and run as published. That is the honest comparison: it is
    a method, not a statistic looking for one.

    The keys are waddR's own: `pval`, `d.wass`, and the decomposition of the
    squared distance into `location`, `size` and `shape` with their
    percentages. The decomposition is the interesting column — it is waddR's
    answer to the question WADE's two stages answer, and `benchmark.qmd` reads
    the two side by side.

    Raises `RuntimeError` if R or waddR is missing, so a notebook can skip the
    section rather than fail the render.
    """
    import shutil
    import subprocess
    import tempfile

    if shutil.which(rscript) is None:
        raise RuntimeError(f"{rscript!r} not on PATH; waddR needs R")
    X = np.asarray(X, dtype=float)
    with tempfile.TemporaryDirectory() as d:
        f = lambda n: f"{d}/{n}"
        np.savetxt(f("x.tsv"), X)
        np.savetxt(f("cond.txt"), np.asarray(cond, dtype=int), fmt="%d")
        with open(f("run.R"), "w") as fh:
            fh.write(_WADDR_R)
        r = subprocess.run([rscript, f("run.R"), f("x.tsv"), f("cond.txt"),
                            str(permnum), str(seed), f("out.tsv")],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("waddR failed:\n" + (r.stderr or r.stdout)[-2000:])
        tab = np.genfromtxt(f("out.tsv"), names=True, delimiter="\t")
    # genfromtxt strips the dots out of waddR's own column names, so put them
    # back as underscores: a caller should read `perc_shape`, not `percshape`,
    # and the mangling is numpy's business rather than waddR's.
    names = ["pval", "d_wass", "location", "size", "shape",
             "perc_loc", "perc_size", "perc_shape"]
    return {new: np.atleast_1d(tab[old])
            for new, old in zip(names, tab.dtype.names)}


# ---------------------------------------------------------------------------
# verification


def verify(n=1500, n1=2000, seed=7, tol=0.03):
    """Hold each statistic to a value derived without reference to this code.

    Every check below is an **exact** null quantity — a normal percentile, a
    closed-form tail expectation, a variance — so a transcription error in a
    centring, a scale or a threshold moves it. Returns
    `(name, expected, observed, ok)` rows.

    **Each row carries its own tolerance, in its own units.** `mu_n` is a
    Monte Carlo estimate of an exact 0, and the estimator's standard error is
    `sqrt(n / reps)` = 0.32 — so a blanket 3% relative tolerance was demanding
    twenty times more precision than the number can carry, and reported a
    1.6-sigma draw as a failure. The same mistake in a different costume as
    `tests/test_stage1_gemm.py`'s: a tolerance is a claim about the units of
    the thing it bounds.

    Two more things this had to get right to be worth running. **The group is large
    (2,000 a side).** The tail statistics threshold at `q75 + IQR` *estimated
    from the data*, and the tail sum is convex in that threshold, so at 200 a
    side Jensen's inequality inflates OS by 4% and ORT by 9.5% above the
    closed form — real, reproducible, and nothing to do with the code. The
    estimate is consistent, so the discrepancy vanishes with n; the check
    lives where the closed form is actually the right answer. **And MOST is
    checked in two independent pieces** rather than end to end: its order
    statistic moments against values known exactly (the full sum has mean 0
    and sd sqrt(n)), and its standardization against the sd of 1 those moments
    presuppose. Checking the composite would have let an error in one cancel
    an error in the other — and the composite is exactly where the bug was.

    Run it directly — ``python tools/competitors.py`` — or from the notebook,
    which prints the table so a reader can see the competitors were checked
    rather than take it on trust.
    """
    from math import erf, exp, pi, sqrt
    from scipy.special import erfinv

    q = lambda p: float(sqrt(2) * erfinv(2 * p - 1))
    phi = lambda z: exp(-z * z / 2) / sqrt(2 * pi)

    rng = np.random.default_rng(seed)
    cond = np.r_[np.ones(n1, int), np.zeros(n1, int)]
    X = rng.standard_normal((n, 2 * n1))               # pure null, unit scale
    out = []

    # COPA(r) reads the r-th percentile of a standard normal. No threshold is
    # estimated, so this one is sharp at any n.
    for r in (75.0, 90.0, 95.0):
        out.append((f"COPA-{r:.0f}  =  z_{r / 100:.2f}", q(r / 100),
                    float(np.median(copa(X, cond, r=r, two_sided=False)))))

    # For a standard normal, E[Z 1(Z>c)] = phi(c) exactly, so a sum over n1
    # cases past c = q75 + IQR has expectation n1 phi(c).
    c = q(.75) + (q(.75) - q(.25))
    out.append(("OS  =  n1 phi(q75+IQR)", n1 * phi(c),
                float(np.mean(outlier_sum(X, cond, two_sided=False)))))

    # ORT is the same sum centred on the control median and divided by the
    # unscaled pooled median deviation, which converges to q(0.75) = 0.6745.
    out.append(("ORT  =  n1 phi(c) / 0.6745", n1 * phi(c) / q(.75),
                float(np.mean(ort(X, cond, two_sided=False)))))

    # MOST, piece one: the k = n partial sum is the sum of ALL n standard
    # normals, so its moments are known exactly and pin the tabulation.
    mu, sd = _most_moments(n1)
    out.append(("MOST mu_n  =  0", 0.0, float(mu[-1]),
                4 * sqrt(n1 / _MOST_REPS)))          # 4 Monte Carlo sigma
    out.append((f"MOST sigma_n  =  sqrt({n1})", sqrt(n1), float(sd[-1]), None))

    # MOST, piece two: those moments are the moments of STANDARD normals, so
    # the statistic's own z must have unit sd. This is what the bug broke.
    case, ctrl = X[:, cond == 1], X[:, cond == 0]
    med_n = np.median(ctrl, axis=1, keepdims=True)
    s = 1.4826 * np.median(np.abs(np.concatenate(
        [ctrl - med_n, case - np.median(case, axis=1, keepdims=True)], axis=1)),
        axis=1, keepdims=True)
    out.append(("MOST z sd  =  1", 1.0, float(((case - med_n) / s).std())))

    # The standardized rank sum is N(0,1) under the null by construction, and
    # a pooled t on df degrees of freedom has variance df/(df-2).
    out.append(("Wilcoxon sd  =  1", 1.0,
                float(np.std(wilcoxon(X, cond, two_sided=False)))))
    df = 2 * n1 - 2
    out.append((f"t sd  =  sqrt(df/(df-2))", sqrt(df / (df - 2)),
                float(np.std(t_stat(X, cond, two_sided=False)))))

    # LSOSS has no null closed form. It is checked on what it claims to
    # estimate instead -- the split point -- which nothing else here reports.
    Y = rng.standard_normal((300, 2 * n1))
    planted = n1 // 8
    Y[:, :planted] += 6.0
    out.append((f"LSOSS k-hat  =  planted {planted}", float(planted),
                float(np.median(_lsoss_full(Y, cond)[1]))))

    # And the check that makes all of the above mean something: the functions
    # verified here are the READABLE ones, each carrying its paper's formula,
    # while the permutation loop runs the fused `_upper`. If the two ever
    # drift, every row above is verifying code nothing executes. Worst
    # relative disagreement over all of them, on signal-bearing data:
    Y = rng.standard_normal((200, 2 * n1))
    Y[:60, :n1 // 10] += 5.0
    ref = {"t-test": t_stat(Y, cond, two_sided=False),
           "Wilcoxon": wilcoxon(Y, cond, two_sided=False),
           "MOST": most(Y, cond, two_sided=False),
           "LSOSS": lsoss(Y, cond, two_sided=False)}
    for r in TUNED["COPA"][1]:
        ref[f"COPA r={r:g}"] = copa(Y, cond, r=r, two_sided=False)
    for k in TUNED["OS"][1]:
        ref[f"OS k={k:g}"] = outlier_sum(Y, cond, k=k, two_sided=False)
    for k in TUNED["ORT"][1]:
        ref[f"ORT k={k:g}"] = ort(Y, cond, k=k, two_sided=False)
    fused = _upper(Y, cond)
    worst = max(float(np.max(np.abs(fused[nm] - v) / np.maximum(np.abs(v), 1e-12)))
                for nm, v in ref.items())
    out.append((f"_upper == the {len(ref)} reference forms", 0.0, worst, 1e-12))

    return [(name, e, o, abs(o - e) <= (a if a is not None else tol * max(abs(e), 1.0)))
            for name, e, o, a in (r + (None,) * (4 - len(r)) for r in out)]


if __name__ == "__main__":
    rows = verify()
    w = max(len(r[0]) for r in rows)
    print(f"{'check':<{w}}  {'expected':>12}  {'observed':>12}")
    for name, e, o, ok in rows:
        print(f"{name:<{w}}  {e:>12.4f}  {o:>12.4f}  {'ok' if ok else 'FAIL'}")
    raise SystemExit(0 if all(r[3] for r in rows) else 1)
