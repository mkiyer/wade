"""The driver: normalize, observe, permute, refine, adjust.

The entry point takes **raw counts**, not a normalized matrix
(``ROADMAP.md`` S2). The continuity jitter is applied at
count precision before division, so a pre-normalized matrix cannot
reproduce it; :func:`wade_from_matrix` exists for callers who have one
anyway and says plainly what it costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace as _dc_replace

import numpy as np

from . import normalize as _normalize
from .permutation import draw_perms, null_statistics, validate_perms
from .pvalues import (
    DEFAULT_N_EXC_MIN,
    DEFAULT_N_TAIL,
    bh_adjust,
    perm_pvalues,
)
from .scores import wade_score
from .shape import ShapeResult, shape_test
from .stats import DEFAULT_TAIL_Q, WadeStats, split_groups, tail_concentration, wade_stats

__all__ = ["WadeResult", "wade", "wade_from_matrix", "wade_contrast", "DEFAULT_NPERMS",
           "DEFAULT_TAIL_CONC_MAX_FACTOR"]

#: The reference has two different defaults — 1000 in the core driver and
#: 2000 in the convenience wrapper — and they are not interchangeable: B
#: sets both the empirical floor 1/(B+1) and the extrapolation floor
#: 1/(B*n_tail), so it changes every small p-value, not just the runtime.
#: One default is picked here and stated: 2000, the source project's
#: production constant. At B < 500 the GPD refinement gate never opens at
#: all.
DEFAULT_NPERMS = 2000

#: The conditioning bound for ``tail_conc``. Because the guard is
#: ``sum(|D|) / |sum(D)| <= F``, it also bounds the reported statistic at
#: ``|tail_conc| <= F`` — the threshold and the guarantee are the same
#: number, which is why it is a documented promise rather than a tuning
#: knob, and why raising it later widens what users were told to expect.
#: See ``ROADMAP.md`` O1; this default is a starting
#: position, not a measurement on real data.
DEFAULT_TAIL_CONC_MAX_FACTOR = 3.0


@dataclass
class WadeResult:
    """Per-gene results. Arrays are parallel and gene-ordered."""

    gene: np.ndarray
    diff_mean: np.ndarray
    diff_frac: np.ndarray
    w1: np.ndarray
    tail_mean: np.ndarray
    tail_conc: np.ndarray
    tail_conc_ok: np.ndarray
    fc: np.ndarray
    cond1_mean: np.ndarray
    cond0_mean: np.ndarray
    tot_mean: np.ndarray
    p_diff: np.ndarray
    p_tail: np.ndarray
    padj_diff: np.ndarray
    padj_tail: np.ndarray

    nexc_diff: np.ndarray
    nexc_tail: np.ndarray
    refined_diff: np.ndarray
    refined_tail: np.ndarray

    stats: WadeStats
    tpm: np.ndarray
    jitter: np.ndarray
    perms: np.ndarray | None
    params: dict = field(default_factory=dict)

    #: Populated only when ``keep_null=True``. Two ``g x nperms`` float64
    #: matrices: about 71 MB at 2,219 genes and 2,000 permutations, but
    #: 3.2 GB at 20,000 genes and 10,000, which is why it is opt-in.
    null_diff: np.ndarray | None = None
    null_tail: np.ndarray | None = None

    scores: dict | None = None

    #: The shape test and the characterization (docs/method.md sections 3-4).
    #: None when nperms == 0, when the shape test is switched off, or when
    #: min(n_case, n_ctrl) < 3 and the bridge has no interior.
    shape: ShapeResult | None = None
    p_shape: np.ndarray | None = None
    padj_shape: np.ndarray | None = None

    @property
    def mean_shift(self) -> np.ndarray:
        """The ordinary mean-difference test — WADE's stage 1.

        Same array as ``diff_mean``, under the name ``docs/method.md`` uses.
        When the groups are equal-sized this is *exactly* the difference of
        the two sample means; on unequal groups it is a grid quadrature that
        over-weights the larger group's extremes.

        Named plainly because that is what it is: the test any conventional
        DE method already performs, and measurably the best detector for weak
        diffuse effects. WADE's claim is to add sensitivity it lacks, not to
        improve on it.
        """
        return self.diff_mean

    @property
    def p_shift(self) -> np.ndarray:
        """Stage 1 p-value: is there a difference at all?"""
        return self.p_diff

    @property
    def padj_shift(self) -> np.ndarray:
        return self.padj_diff

    @property
    def pi_hat(self) -> np.ndarray | None:
        """Effective affected fraction; 1.0 means a global change."""
        return None if self.shape is None else self.shape.pi_hat

    @property
    def up_share(self) -> np.ndarray | None:
        """Share of distributional movement that is upward, in [0, 1]."""
        return None if self.shape is None else self.shape.up_share

    def interpret(self, alpha: float = 0.05) -> np.ndarray:
        """The 2x2 of ``docs/method.md`` section 5, as a label per gene.

        =========  =========  ================================================
        p_shift    p_shape    label
        =========  =========  ================================================
        sig        --         ``global`` -- an ordinary DE method finds this
        sig        sig        ``subset`` -- concentrated; pi_hat says how much
        --         sig        ``shape-only`` -- no net mean shift
        --         --         ``none``
        =========  =========  ================================================
        """
        g = self.diff_mean.shape[0]
        out = np.full(g, "none", dtype=object)
        if self.p_diff is None:
            return out
        shift = np.nan_to_num(self.p_diff, nan=1.0) <= alpha
        shape = (np.zeros(g, bool) if self.p_shape is None
                 else np.nan_to_num(self.p_shape, nan=1.0) <= alpha)
        out[shift & ~shape] = "global"
        out[shift & shape] = "subset"
        out[~shift & shape] = "shape-only"
        return out

    @property
    def nprobs(self) -> int:
        """The quantile-grid resolution, ``min(n_case, n_ctrl)``.

        A property of the design, not a tunable parameter: it cannot be
        raised by any argument.
        """
        return self.stats.nprobs

    @property
    def k(self) -> int:
        """The number of order statistics the tail window actually averages."""
        return self.stats.k

    def columns(self) -> dict[str, np.ndarray]:
        """The result frame as a plain dict of equal-length arrays."""
        cols = {
            "gene": self.gene,
            "diff_mean": self.diff_mean,
            "diff_frac": self.diff_frac,
            "w1": self.w1,
            "tail_mean": self.tail_mean,
            "tail_conc": self.tail_conc,
            "tail_conc_ok": self.tail_conc_ok,
            "fc": self.fc,
            "cond1_mean": self.cond1_mean,
            "cond0_mean": self.cond0_mean,
            "tot_mean": self.tot_mean,
            "p_diff": self.p_diff,
            "p_tail": self.p_tail,
            "padj_diff": self.padj_diff,
            "padj_tail": self.padj_tail,
        }
        if self.shape is not None:
            cols.update({
                "shape_stat": self.shape.statistic,
                "p_shape": self.p_shape,
                "padj_shape": self.padj_shape,
                "pi_hat": self.shape.pi_hat,
                "up_share": self.shape.up_share,
            })
        cols.update(self.scores or {})
        return cols

    def to_pandas(self):
        """Optional convenience. pandas is not a dependency of this package."""
        import pandas as pd

        return pd.DataFrame(self.columns())


def tail_window_report(n_case: int, n_ctrl: int, tail_q: float = DEFAULT_TAIL_Q) -> dict:
    """How many order statistics a given design buys, without running the test.

    A caller needs this *before* deciding whether the subset axis means
    anything, and the reference makes every caller re-derive it. The
    source project's own thresholds were 10 as the floor to run at all
    (``ceil(0.10 * 10) = 1`` — still a single point) and 20 to believe the
    subset axis (20 gives a two-point window, 30 gives three). Those are
    consumer decisions, not method parameters, but the arithmetic behind
    them generalizes.
    """
    from .quantiles import tail_window_size

    nprobs = min(n_case, n_ctrl)
    k = tail_window_size(nprobs, tail_q)
    return {
        "nprobs": nprobs,
        "k": k,
        "tail_q": tail_q,
        "single_point_tail": k == 1,
        "note": (
            "The tail 'mean' averages a single order statistic."
            if k == 1
            else f"The tail mean averages {k} order statistics."
        ),
    }


def _resolve_gene_names(gene_names, g: int) -> np.ndarray:
    """Synthesize positional identifiers rather than drop the column.

    R defaults ``gene_names`` to ``rownames(counts)``, which is ``NULL``
    for an unnamed matrix, and ``tibble()`` **drops** a ``NULL`` column
    rather than erroring — so the frame silently loses its identifier
    while every other column is present and correct, and anything
    downstream that joins on it breaks somewhere else
    (``docs/implementation-notes.md`` section 7).
    """
    if gene_names is None:
        return np.array([f"gene{i}" for i in range(g)], dtype=object)
    gene_names = np.asarray(gene_names, dtype=object)
    if gene_names.shape != (g,):
        raise ValueError(
            f"gene_names must have one entry per gene: expected {g}, "
            f"got {gene_names.shape}"
        )
    return gene_names


def _run(
    x: np.ndarray,
    cond: np.ndarray,
    *,
    nperms: int,
    tail_q: float,
    log2_scale: bool,
    weight: float,
    perms,
    seed,
    perm_rng,
    n_exc_min: int,
    n_tail: int,
    tail_conc_max_factor: float,
    allow_single_sample_group: bool,
    keep_null: bool,
    gene_names,
    compute_scores: bool,
    backend: str,
    shape: bool,
    jitter: np.ndarray,
    tpm: np.ndarray,
) -> WadeResult:
    cond = np.asarray(cond)
    g = x.shape[0]
    obs = wade_stats(
        x, cond, tail_q=tail_q, log2_scale=log2_scale, weight=weight,
        allow_single_sample_group=allow_single_sample_group,
    )

    if nperms > 0:
        if perms is None:
            perms = draw_perms(cond, nperms, seed=seed, rng=perm_rng)
        perms = validate_perms(perms, cond, nperms)
        null_d, null_t = null_statistics(
            x, perms, tail_q=tail_q, log2_scale=log2_scale, weight=weight,
            backend=backend,
        )
        p_diff, nexc_d, ref_d = perm_pvalues(
            obs.diff_mean, null_d, n_exc_min=n_exc_min, n_tail=n_tail
        )
        p_tail, nexc_t, ref_t = perm_pvalues(
            obs.tail_mean, null_t, n_exc_min=n_exc_min, n_tail=n_tail
        )
    else:
        perms = None
        null_d = null_t = None
        p_diff = np.full(g, np.nan)
        p_tail = np.full(g, np.nan)
        nexc_d = np.zeros(g, dtype=np.int64)
        nexc_t = np.zeros(g, dtype=np.int64)
        ref_d = np.zeros(g, dtype=bool)
        ref_t = np.zeros(g, dtype=bool)

    sh = p_shape = padj_shape = None
    if nperms > 0 and shape and obs.nprobs >= 3:
        sh = shape_test(x, cond, perms, backend=backend)
        p_shape, _, _ = perm_pvalues(
            sh.statistic, sh.null, n_exc_min=n_exc_min, n_tail=n_tail
        )
        padj_shape = bh_adjust(p_shape)
        if not keep_null:
            sh = _dc_replace(sh, null=None)

    tconc, tconc_ok = tail_concentration(
        obs.tail_num, obs.tail_den, tail_conc_max_factor, np.abs(obs.D).sum(axis=1)
    )

    result = WadeResult(
        gene=_resolve_gene_names(gene_names, g),
        diff_mean=obs.diff_mean,
        diff_frac=obs.diff_frac,
        w1=obs.w1,
        tail_mean=obs.tail_mean,
        tail_conc=tconc,
        tail_conc_ok=tconc_ok,
        fc=obs.fc,
        cond1_mean=obs.cond1_mean,
        cond0_mean=obs.cond0_mean,
        tot_mean=obs.tot_mean,
        p_diff=p_diff,
        p_tail=p_tail,
        padj_diff=bh_adjust(p_diff),
        padj_tail=bh_adjust(p_tail),
        nexc_diff=nexc_d,
        nexc_tail=nexc_t,
        refined_diff=ref_d,
        refined_tail=ref_t,
        stats=obs,
        tpm=tpm,
        jitter=jitter,
        perms=perms,
        null_diff=null_d if keep_null else None,
        null_tail=null_t if keep_null else None,
        shape=sh, p_shape=p_shape, padj_shape=padj_shape,
        params=dict(
            nperms=nperms, tail_q=tail_q, log2_scale=log2_scale, weight=weight,
            n_exc_min=n_exc_min, n_tail=n_tail,
            tail_conc_max_factor=tail_conc_max_factor, seed=seed,
            nprobs=obs.nprobs, k=obs.k, n1=obs.n1, n0=obs.n0,
            backend=backend,
        ),
    )
    if compute_scores:
        result.scores = wade_score(
            obs.fc, obs.cond1_mean, obs.cond0_mean, obs.tail_mean, obs.diff_frac
        )
    return result


def wade(
    counts: np.ndarray,
    normalizer,
    cond: np.ndarray,
    *,
    lib_sizes: np.ndarray | None = None,
    nperms: int = DEFAULT_NPERMS,
    tail_q: float = DEFAULT_TAIL_Q,
    noise: float = _normalize.DEFAULT_NOISE,
    norm_factor: float = _normalize.DEFAULT_NORM_FACTOR,
    seed: int | None = 1,
    jitter: np.ndarray | None = None,
    perms: np.ndarray | None = None,
    log2_scale: bool = False,
    weight: float = 1.0,
    gene_names=None,
    n_exc_min: int = DEFAULT_N_EXC_MIN,
    n_tail: int = DEFAULT_N_TAIL,
    tail_conc_max_factor: float = DEFAULT_TAIL_CONC_MAX_FACTOR,
    allow_single_sample_group: bool = False,
    keep_null: bool = False,
    compute_scores: bool = False,
    backend: str = "auto",
    shape: bool = True,
) -> WadeResult:
    """Run WADE on a raw count matrix. The primary entry point.

    Parameters
    ----------
    counts
        Genes x samples raw counts.
    normalizer
        Either a per-gene vector of length ``n_genes`` (broadcast across
        samples — e.g. intron count, giving "sjTPM") or a full genes x
        samples matrix (e.g. per-library effective length, giving standard
        TPM). The arithmetic is identical; only the pair changes.
    cond
        Binary vector, 1 = case and 0 = control, one entry per column.
    lib_sizes
        Per-sample size factors. Computed from ``counts`` and
        ``normalizer`` when omitted. Note this depends on which genes are
        in the matrix — change the gene set and every normalized value
        changes — so pass it explicitly if that matters to you.
    nperms
        Permutations. See :data:`DEFAULT_NPERMS`. ``0`` returns effect
        sizes with all four p-value columns ``nan``, which is a supported
        mode.
    jitter, perms
        Supplied randomness, **on the production argument path**. ``jitter``
        is a genes x samples array used in place of the internal draw;
        ``perms`` is an ``(nperms, n_samples)`` matrix of label vectors.
        These exist because R's Mersenne-Twister and NumPy's PCG64 cannot
        agree on a shared seed, so exact cross-language parity is only
        achievable by passing the realised matrices as data. They are
        accepted here, and not on a separate test-only path, because a
        fixture path that bypasses production code validates code nobody
        runs (``docs/implementation-notes.md`` hazard 2).
    seed
        Seeds the jitter and the permutations on **separate, independent
        streams** (via ``SeedSequence.spawn``), mirroring the reference's
        use of an offset stream so the two never share state. ``None``
        draws from ambient entropy without touching any global RNG.
    keep_null
        Retain the two ``g x nperms`` null matrices on the result.
    compute_scores
        Add the rank scores. Off by default — they are nomination
        heuristics, not inference. See :mod:`wade.scores`.

    Returns
    -------
    WadeResult
    """
    counts = np.asarray(counts, dtype=np.float64)
    cond = np.asarray(cond)
    if cond.shape[0] != counts.shape[1]:
        raise ValueError(
            f"cond has {cond.shape[0]} entries but counts has {counts.shape[1]} samples"
        )
    split_groups(cond)  # fail fast, before any normalization work

    if seed is None:
        jitter_rng = np.random.default_rng()
        perm_rng = np.random.default_rng()
    else:
        js, ps = np.random.SeedSequence(seed).spawn(2)
        jitter_rng = np.random.default_rng(js)
        perm_rng = np.random.default_rng(ps)

    # Draw the jitter here rather than inside the normalizer, so the result
    # always carries the exact noise draw it was conditioned on. Inference is
    # conditional on that one draw, so a result that cannot report it is not
    # reproducible.
    if jitter is None:
        jitter = _normalize.draw_jitter(counts.shape, noise=noise, rng=jitter_rng)
    else:
        jitter = np.asarray(jitter, dtype=np.float64)

    tpm = _normalize.tpm_like(
        counts, normalizer, lib_sizes,
        noise=noise, norm_factor=norm_factor, jitter=jitter,
    )

    return _run(
        tpm, cond, nperms=nperms, tail_q=tail_q, log2_scale=log2_scale, weight=weight,
        perms=perms, seed=seed, perm_rng=perm_rng, n_exc_min=n_exc_min, n_tail=n_tail,
        tail_conc_max_factor=tail_conc_max_factor,
        allow_single_sample_group=allow_single_sample_group, keep_null=keep_null,
        gene_names=gene_names, compute_scores=compute_scores, backend=backend,
        shape=shape, jitter=jitter, tpm=tpm,
    )


def wade_from_matrix(
    x: np.ndarray,
    cond: np.ndarray,
    *,
    nperms: int = DEFAULT_NPERMS,
    tail_q: float = DEFAULT_TAIL_Q,
    seed: int | None = 1,
    perms: np.ndarray | None = None,
    log2_scale: bool = False,
    weight: float = 1.0,
    gene_names=None,
    n_exc_min: int = DEFAULT_N_EXC_MIN,
    n_tail: int = DEFAULT_N_TAIL,
    tail_conc_max_factor: float = DEFAULT_TAIL_CONC_MAX_FACTOR,
    allow_single_sample_group: bool = False,
    keep_null: bool = False,
    compute_scores: bool = False,
    backend: str = "auto",
    shape: bool = True,
) -> WadeResult:
    """Run WADE on a matrix that is **already on a comparable scale**.

    For callers with an existing pipeline. Be explicit about what this
    costs: **no continuity jitter is applied**, because it cannot be. The
    jitter is added at count precision before division, and once counts
    have been divided by a normalizer and a library size the information
    needed to reconstruct that perturbation is gone — a jitter applied
    afterwards is a different perturbation and produces different numbers.

    So this is a **variant of the test without tie-breaking**. On sparse,
    zero-heavy data that matters: many samples share a count of zero, the
    quantile grid degenerates into flat runs, and the statistic reads
    those runs as genuine agreement. It also loses the strict positivity
    the jitter provides, so ``fc`` and ``log2(fc)`` can be non-finite for
    a gene with an all-zero group.

    Use :func:`wade` with raw counts unless you have a specific reason not
    to.
    """
    x = np.asarray(x, dtype=np.float64)
    cond = np.asarray(cond)
    if cond.shape[0] != x.shape[1]:
        raise ValueError(
            f"cond has {cond.shape[0]} entries but the matrix has {x.shape[1]} samples"
        )
    split_groups(cond)
    perm_rng = np.random.default_rng(seed if seed is None else np.random.SeedSequence(seed).spawn(2)[1])
    return _run(
        x, cond, nperms=nperms, tail_q=tail_q, log2_scale=log2_scale, weight=weight,
        perms=perms, seed=seed, perm_rng=perm_rng, n_exc_min=n_exc_min, n_tail=n_tail,
        tail_conc_max_factor=tail_conc_max_factor,
        allow_single_sample_group=allow_single_sample_group, keep_null=keep_null,
        gene_names=gene_names, compute_scores=compute_scores, backend=backend,
        shape=shape, jitter=np.zeros_like(x), tpm=x,
    )


def wade_contrast(
    counts: np.ndarray,
    normalizer: np.ndarray,
    case_samples,
    ctrl_samples,
    *,
    gene_names=None,
    nperms: int = DEFAULT_NPERMS,
    compute_scores: bool = True,
    **kwargs,
) -> WadeResult:
    """Run one contrast from two lists of sample column indices.

    Port of the contract R's ``wade_run()`` defines — the entry point the
    historical call sites actually used. It does five things a caller
    would otherwise have to get right:

    1. Subsets both matrices to the selected columns, cases first.
    2. Codes the condition vector to match that order.
    3. **Sizes libraries on the subset**, not on the full matrix. This is
       the important one: a library's size factor then depends only on the
       genes and samples handed in, so changing the gene set changes every
       normalized value. The reference behaves the same way and the source
       project's cache key included the gene count for exactly this reason.
    4. Computes the rank scores (on by default here, matching the R).
    5. Records the group sizes.

    One asymmetry in the reference is resolved rather than inherited:
    ``wade_run()`` accepts only the matrix normalizer form, because it
    subsets with two indices, while ``wade()`` accepts both. Here both
    forms are accepted, and a per-gene vector is simply not subset by
    column.
    """
    counts = np.asarray(counts, dtype=np.float64)
    case_samples = np.asarray(case_samples, dtype=np.intp)
    ctrl_samples = np.asarray(ctrl_samples, dtype=np.intp)
    overlap = np.intersect1d(case_samples, ctrl_samples)
    if overlap.size:
        raise ValueError(f"samples appear in both groups: {overlap.tolist()}")

    cols = np.concatenate([case_samples, ctrl_samples])
    cond = np.concatenate([np.ones(case_samples.size, dtype=int),
                           np.zeros(ctrl_samples.size, dtype=int)])
    sub_counts = counts[:, cols]

    normalizer = np.asarray(normalizer, dtype=np.float64)
    sub_norm = normalizer[:, cols] if normalizer.ndim == 2 else normalizer

    result = wade(
        sub_counts, sub_norm, cond, nperms=nperms, gene_names=gene_names,
        compute_scores=compute_scores, **kwargs,
    )
    result.params["n_case"] = int(case_samples.size)
    result.params["n_ctrl"] = int(ctrl_samples.size)
    result.params["columns"] = cols
    return result
