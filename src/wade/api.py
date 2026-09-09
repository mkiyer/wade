"""The driver: normalize, observe, permute, refine, adjust.

WADE takes raw counts, and only raw counts. The continuity jitter is applied
at count precision before division and the subset stage's null is built by
thinning reads, so a pre-normalized matrix can reproduce neither, and there is
no entry point for one.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import normalize as _normalize
from .io import Condition, as_counts, result_columns
from .permutation import draw_perms, mean_diff_weights, null_statistics, validate_perms
from .pvalues import (ALTERNATIVES, DEFAULT_N_EXC_MIN, DEFAULT_N_TAIL, bh_adjust,
                      exceedance_counts, perm_pvalues)
from .quantiles import capped_nprobs
from .stats import WadeStats, split_groups, wade_stats
from .subset import SubsetResult, characterization_ci, subset_test
from .thinning import fit_fold_change, gene_chunks, one_count, thin_counts

__all__ = ["WadeResult", "wade", "wade_contrast",
           "DEFAULT_NPERMS", "DEFAULT_MAX_PROBS"]

#: Permutations. Sets both the empirical p-value floor ``1/(B+1)`` and the
#: GPD extrapolation floor ``1/(B * n_tail)``, so it changes every small
#: p-value rather than only the runtime. Below 500 the refinement never fires.
DEFAULT_NPERMS = 2000

#: Cap on the quantile grid (``docs/method.md`` §1). Designs at or under
#: 2,000 per group are untouched; above it the grid stops growing with the
#: cohort and ``affected_fraction``'s resolution is ``1/max_probs``.
DEFAULT_MAX_PROBS = 2000


@dataclass
class WadeResult:
    """Per-gene results. Arrays are parallel and gene-ordered.

    Two questions, answered separately (``docs/method.md`` §5):

    ``p_mean_shift``
        Is average expression different? The ordinary test, which any
        conventional DE method also performs.
    ``p_subset``
        Is the difference confined to a subset of samples, rather than
        spread across all of them?

    and two numbers describing *what kind* of difference it is:

    ``affected_fraction``
        The effective fraction of samples that differ. ``1.0`` is a global
        change; ``0.05`` is a 5% subset.
    ``direction``
        ``+1`` all up, ``-1`` all down, ``0`` two-sided.
    """

    gene: np.ndarray
    mean_shift: np.ndarray
    w1: np.ndarray
    fc: np.ndarray
    log2_fc: np.ndarray
    case_mean: np.ndarray
    ctrl_mean: np.ndarray

    p_mean_shift: np.ndarray
    padj_mean_shift: np.ndarray
    nexc_mean_shift: np.ndarray
    refined_mean_shift: np.ndarray

    stats: WadeStats
    tpm: np.ndarray
    #: The jitter and the pseudocount are both ``genes x samples`` and both
    #: reconstructible, so a result keeps the small inputs and rebuilds them
    #: on access — see the ``jitter`` and ``pseudocount`` properties.
    _jitter: np.ndarray | None
    _pc: tuple | None
    perms: np.ndarray | None
    params: dict = field(default_factory=dict)

    #: Two ``g x nperms`` float64 matrices when retained: about 71 MB at 2,219
    #: genes and 2,000 permutations, 3.2 GB at 20,000 by 10,000, hence opt-in.
    null_mean_shift: np.ndarray | None = None

    #: The subset test and the characterization. ``None`` when ``nperms == 0``,
    #: when the subset test is switched off, or when ``min(n_case, n_ctrl) < 3``
    #: and the bridge has no interior.
    subset: SubsetResult | None = None
    p_subset: np.ndarray | None = None
    padj_subset: np.ndarray | None = None

    #: Stage 2's exceedance count and GPD-refinement flag, the stage-1 pair
    #: for the subset test. A gene *at* the resolution floor has not been
    #: measured there; it has been censored there.
    nexc_subset: np.ndarray | None = None
    refined_subset: np.ndarray | None = None

    #: Permutation z-scores, ``(observed - mean(null)) / sd(null)`` per gene —
    #: the GSEA-NES analogue, for ranking past the p-value floor (see
    #: :func:`_perm_z`). ``z_mean_shift`` is NaN when there were no
    #: permutations; ``z_subset`` is ``None`` when the subset stage did not run.
    z_mean_shift: np.ndarray | None = None
    z_subset: np.ndarray | None = None

    #: The condition vector the contrast was run on, one entry per column of
    #: ``tpm``. Kept so a single gene can be re-derived from the result alone
    #: (:meth:`gene_detail`, and the plotting layer).
    cond: np.ndarray | None = None

    #: Sample labels, one per column of ``tpm`` — from the input frame, the
    #: ``sample_names`` argument, or positional.
    sample_names: np.ndarray | None = None

    #: Per-gene metadata the input carried — a symbol, a biotype, coordinates.
    #: Read by no statistic; figures and tables can show it.
    gene_meta: dict = field(default_factory=dict)

    #: Bootstrap 95% intervals, ``(2, genes)`` each, when ``n_boot > 0``.
    ci_affected_fraction: np.ndarray | None = None
    ci_direction: np.ndarray | None = None
    ci_subset_log2_fc: np.ndarray | None = None
    ci_log2_fc: np.ndarray | None = None
    ci_mean_shift: np.ndarray | None = None

    @property
    def affected_fraction(self) -> np.ndarray | None:
        return None if self.subset is None else self.subset.affected_fraction

    @property
    def direction(self) -> np.ndarray | None:
        return None if self.subset is None else self.subset.direction

    @property
    def subset_log2_fc(self) -> np.ndarray | None:
        """Log2 fold change **within the affected fraction** — the subset's
        magnitude, read off the log-ratio curve over the region
        ``affected_fraction`` names (:attr:`wade.SubsetResult.subset_log2_fc`).
        The number to rank subset findings by once ``p_subset`` saturates."""
        return None if self.subset is None else self.subset.subset_log2_fc

    @property
    def fitted_fold_change(self) -> np.ndarray | None:
        """The global fold change the subset stage's null was built under."""
        return None if self.subset is None else self.subset.shift

    @property
    def nprobs(self) -> int:
        """The realized quantile-grid resolution, ``min(n_case, n_ctrl, max_probs)``.

        The resolution limit on the affected fraction — nothing finer than
        ``1/nprobs`` is estimable. No argument raises it beyond the design's
        ``min(n_case, n_ctrl)``; ``max_probs`` (``docs/method.md`` §1) can cap
        it below that at large cohorts, and when it does, the cap and the
        realized value are both recorded here and in the manifest.
        """
        return self.stats.nprobs

    def gene_index(self, gene) -> int:
        """Resolve a gene name or a positional index to a row number.

        Names are matched exactly against ``gene``; an integer is taken as a
        position. Either way the error names what was asked for, because a
        silent fallback from name to position is how the wrong gene gets
        plotted.
        """
        if isinstance(gene, (int, np.integer)) and not isinstance(gene, bool):
            i = int(gene)
            g = self.gene.shape[0]
            if not -g <= i < g:
                raise IndexError(f"gene index {i} is out of range for {g} genes")
            return i % g
        hits = np.flatnonzero(self.gene == gene)
        if hits.size == 0:
            raise KeyError(f"no gene named {gene!r} in the result")
        if hits.size > 1:
            raise KeyError(f"gene name {gene!r} is not unique ({hits.size} rows); use an index")
        return int(hits[0])

    def gene_detail(self, gene):
        """The single-gene curves for ``gene`` (a name or an index).

        Re-derives :func:`wade.wade_gene` from the normalized matrix the test
        ran on, so the curves are exactly those the reported statistics were
        computed from — not a recomputation from counts with a fresh jitter.
        """
        if self.cond is None:
            raise ValueError("this result does not carry its condition vector; "
                             "pass the row and cond to wade_gene() directly")
        from .diagnostics import wade_gene
        i = self.gene_index(gene)
        return wade_gene(self.tpm[i], self.cond, pseudocount=self.pseudocount_row(i),
                         max_probs=self.params.get("max_probs"))

    @property
    def jitter(self) -> np.ndarray:
        """The continuity jitter the run used, ``genes x samples``.

        Rebuilt from the seed rather than stored: the RNG streams are spawned
        deterministically from one ``SeedSequence``, so regenerating the first
        reproduces the draw exactly. A result keeps the array only when it
        cannot be rebuilt (``seed=None``, or a caller-supplied jitter).
        """
        if self._jitter is not None:
            return self._jitter
        js = np.random.SeedSequence(self.params["seed"]).spawn(4)[0]
        return _normalize.draw_jitter(self.tpm.shape, noise=self.params["noise"],
                                      rng=np.random.default_rng(js))

    @property
    def pseudocount(self) -> np.ndarray | None:
        """One count in each sample's normalized units, ``genes x samples``;
        ``None`` when the pseudocount is switched off.

        Rebuilt on access from the normalizer and library sizes rather than
        stored, so bind it once rather than indexing it in a loop;
        :meth:`pseudocount_row` is the cheap way to get one gene.
        """
        if self._pc is None:
            return None
        normalizer, lib, norm_factor, scale = self._pc
        return one_count(normalizer, lib, self.tpm.shape,
                         norm_factor=norm_factor) * scale

    def pseudocount_row(self, i: int) -> np.ndarray | None:
        """One gene's pseudocount, without building the whole matrix."""
        if self._pc is None:
            return None
        normalizer, lib, norm_factor, scale = self._pc
        norm = normalizer[i][None, :] if normalizer.ndim == 2 else normalizer[i:i + 1]
        return one_count(norm, lib, (1, self.tpm.shape[1]),
                         norm_factor=norm_factor)[0] * scale

    def columns(self) -> dict[str, np.ndarray]:
        """The result table as a dict of equal-length arrays, in the written
        column order (:data:`wade.RESULT_COLUMNS`), with the bootstrap
        intervals as ``*_lo`` / ``*_hi`` pairs at the end."""
        return result_columns(self)

    def to_frame(self):
        """:meth:`columns` as a polars DataFrame (polars is imported here)."""
        import polars as pl

        return pl.DataFrame(self.columns())


def _describe_normalizer(normalizer) -> str:
    """A one-line description for the manifest. The normalizer decides every
    value the test sees, so provenance without it cannot reproduce a run."""
    a = np.asarray(normalizer, dtype=np.float64)
    if a.ndim == 2:
        return f"matrix {a.shape}"
    if a.size == 0:
        return "empty"
    if np.all(a == a.flat[0]):
        return f"scalar {float(a.flat[0]):g}"
    return f"per-gene vector (n={a.size}, median {float(np.median(a)):g})"


def _fit_alpha(normalizer, lib, norm_factor, n_samples):
    """The fold-change fit's per-cell scale: jitter-free ``tpm_like`` is
    exactly ``counts * alpha`` with ``alpha = norm_factor / (normalizer * lib)``,
    served by row slice so a chunked fit never materializes more than a chunk."""
    def alpha(rows=slice(None)):
        nrm = normalizer[rows]
        nrm = _normalize._broadcast_normalizer(nrm, (nrm.shape[0], n_samples))
        with np.errstate(divide="ignore", invalid="ignore"):
            return norm_factor / (nrm * lib[None, :])
    return alpha


def _check_libraries(lib: np.ndarray, sample_names, n_genes: int, allow: bool) -> None:
    """Refuse a library with nothing in it, by name.

    An all-zero column has no count scale: every gene in it normalizes to
    exactly ``norm_factor`` (``docs/method.md`` §7), and that constant takes
    part in every quantile the contrast reads. Dropping the sample is the
    caller's decision, so this raises rather than deciding quietly. An empty
    matrix (no genes) has no libraries to judge and passes.
    """
    lib = np.asarray(lib, dtype=np.float64)
    if allow or n_genes == 0:
        return
    empty = np.flatnonzero(~(lib > 0))
    if empty.size == 0:
        return
    names = (empty.tolist() if sample_names is None
             else np.asarray(sample_names)[empty].tolist())
    raise ValueError(
        f"{empty.size} sample(s) have a zero library size, so every gene in them "
        f"normalizes to the norm_factor constant rather than to an abundance: "
        f"{names[:8]}"
        + (f" and {len(names) - 8} more. " if len(names) > 8 else ". ")
        + "Drop them from the matrix and the condition, or pass "
          "allow_empty_samples=True to run anyway. wade.library_qc() reports "
          "depth, complexity and concentration for every library."
    )


def _validate_stage1(stage1: str, cond, fit_backend: str = "numpy") -> None:
    if fit_backend not in ("numpy", "rust"):
        raise ValueError(f"fit_backend must be 'numpy' or 'rust'; got {fit_backend!r}")
    if stage1 not in ("grid", "gemm", "saddlepoint"):
        raise ValueError(
            f"stage1 must be 'grid', 'gemm' or 'saddlepoint'; got {stage1!r}"
        )
    if stage1 == "saddlepoint":
        # Fail here, not four minutes into the permutation loop. SciPy is the
        # one dependency the statistic does not otherwise have, and this is
        # the only path that needs it.
        from importlib.util import find_spec

        if find_spec("scipy") is None:
            raise ImportError(
                "stage1='saddlepoint' needs SciPy, which WADE does not "
                "require otherwise. Install it (`pip install scipy` or "
                "`conda install scipy`), or use the default stage1='grid', "
                "which needs only NumPy (docs/method.md §6)."
            )
    if stage1 == "gemm":
        cond = np.asarray(cond)
        n1, n0 = int(np.sum(cond == 1)), int(np.sum(cond == 0))
        if n1 != n0:
            raise ValueError(
                f"stage1='gemm' needs a balanced design: the mean-difference "
                f"statistic equals the grid statistic only when the groups are "
                f"equal-sized, and this design is {n1} v {n0}. Use the default "
                f"stage1='grid' or stage1='saddlepoint', which works at any balance."
            )


def _resolve_counts(counts, normalizer, gene_names, sample_names):
    """``counts`` through :func:`wade.as_counts`; ``normalizer`` may be the name
    of a numeric column carried alongside the counts, a scalar, a per-gene
    vector or a genes x samples matrix."""
    c = as_counts(counts, gene_names=gene_names, sample_names=sample_names,
                  exclude=(normalizer,) if isinstance(normalizer, str) else ())
    if isinstance(normalizer, str):
        normalizer = c.normalizer(normalizer)
    elif np.isscalar(normalizer):
        normalizer = np.full(c.values.shape[0], float(normalizer), dtype=np.float64)
    return c, np.asarray(normalizer, dtype=np.float64)


def _perm_z(stat: np.ndarray, null: np.ndarray) -> np.ndarray:
    """The permutation z-score, ``(observed - mean(null)) / sd(null)`` per
    gene: the analogue of GSEA's normalized enrichment score, for ranking
    past the p-value floor. Not a calibrated tail probability
    (``docs/method.md`` §6).
    """
    mu = null.mean(axis=1)
    sd = null.std(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, (stat - mu) / sd, np.nan)


def _finish(*, obs, null, perms, sub, ci, cond, nperms, seed, n_exc_min, n_tail,
            alternative, keep_null, gene_names, tpm, retain_jitter, pc,
            n_boot, sample_names, condition_meta, max_probs,
            gene_chunk=None, stage1_stat=None, stage1="grid",
            fit_backend="numpy", strata=None, gene_meta=None) -> WadeResult:
    """P-values, BH and the result object, from assembled per-gene arrays.

    ``stage1_stat`` is the observed stage-1 vector when it is not the grid
    quadrature — the exact mean difference of ``stage1="gemm"`` and
    ``"saddlepoint"`` — and is then both what the p-value compares against
    the null and what the result reports as ``mean_shift``.
    """
    g = obs.mean_shift.shape[0]
    stat = obs.mean_shift if stage1_stat is None else stage1_stat
    if stage1 == "saddlepoint":
        # No permutation enters the p-value. The null is still built when
        # permutations were drawn for stage 2, so `z_mean_shift` and
        # `nexc_mean_shift` keep their meaning; `refined_mean_shift` is False
        # throughout, there being no floor to refine past.
        from .saddlepoint import mean_diff_saddlepoint_p

        p_mean = mean_diff_saddlepoint_p(tpm, cond, stat, alternative=alternative)
        refined = np.zeros(g, dtype=bool)
        if null is not None:
            nexc = exceedance_counts(stat, null, alternative=alternative)
            z_mean = _perm_z(stat, null)
        else:
            nexc = np.zeros(g, dtype=np.int64)
            z_mean = np.full(g, np.nan)
    elif null is not None:
        p_mean, nexc, refined = perm_pvalues(
            stat, null, n_exc_min=n_exc_min, n_tail=n_tail,
            alternative=alternative,
        )
        z_mean = _perm_z(stat, null)
    else:
        p_mean = np.full(g, np.nan)
        nexc = np.zeros(g, dtype=np.int64)
        refined = np.zeros(g, dtype=bool)
        z_mean = np.full(g, np.nan)

    p_subset = padj_subset = None
    z_subset = nexc_subset = refined_subset = None
    if sub is not None:
        # The subset statistic is already a maximum over widths, oriented by
        # `alternative` inside the scan, so its p-value is an upper-tail one.
        p_subset, nexc_subset, refined_subset = perm_pvalues(
            sub.statistic, sub.null, n_exc_min=n_exc_min, n_tail=n_tail,
            alternative="greater",
        )
        padj_subset = bh_adjust(p_subset)
        z_subset = _perm_z(sub.statistic, sub.null)
        if not keep_null:
            sub = replace(sub, null=None)

    return WadeResult(
        gene=gene_names,
        mean_shift=stat, w1=obs.w1, fc=obs.fc, log2_fc=obs.log2_fc,
        case_mean=obs.case_mean, ctrl_mean=obs.ctrl_mean,
        p_mean_shift=p_mean, padj_mean_shift=bh_adjust(p_mean),
        nexc_mean_shift=nexc, refined_mean_shift=refined,
        stats=obs, tpm=tpm, _jitter=retain_jitter, _pc=pc, perms=perms,
        null_mean_shift=null if keep_null else None,
        subset=sub, p_subset=p_subset, padj_subset=padj_subset,
        nexc_subset=nexc_subset, refined_subset=refined_subset,
        z_mean_shift=z_mean, z_subset=z_subset,
        cond=cond, sample_names=sample_names, gene_meta=dict(gene_meta or {}),
        ci_affected_fraction=ci.get("affected_fraction"),
        ci_direction=ci.get("direction"),
        ci_subset_log2_fc=ci.get("subset_log2_fc"),
        ci_log2_fc=ci.get("log2_fc"),
        ci_mean_shift=ci.get("mean_shift"),
        params=dict(nperms=nperms, seed=seed, alternative=alternative,
                    n_exc_min=n_exc_min, n_tail=n_tail,
                    max_probs=max_probs, n_case=obs.n1, n_ctrl=obs.n0,
                    n_boot=n_boot, gene_chunk=gene_chunk, stage1=stage1,
                    fit_backend=fit_backend,
                    strata=None if strata is None else np.asarray(strata),
                    **(condition_meta or {})),
    )


def _concat_stats(parts: list[WadeStats]) -> WadeStats:
    first = parts[0]
    if len(parts) == 1:
        return first
    cat = lambda name: np.concatenate([getattr(p, name) for p in parts], axis=0)  # noqa: E731
    return WadeStats(
        q=first.q, nprobs=first.nprobs, n1=first.n1, n0=first.n0,
        Q1=cat("Q1"), Q0=cat("Q0"), D=cat("D"),
        mean_shift=cat("mean_shift"), w1=cat("w1"), fc=cat("fc"),
        case_mean=cat("case_mean"), ctrl_mean=cat("ctrl_mean"),
    )


def _concat_subsets(parts: list[SubsetResult]) -> SubsetResult:
    first = parts[0]
    if len(parts) == 1:
        return first
    cat = lambda name: np.concatenate([getattr(p, name) for p in parts], axis=0)  # noqa: E731
    return SubsetResult(
        statistic=cat("statistic"), null=cat("null"), r=cat("r"),
        affected_fraction=cat("affected_fraction"), direction=cat("direction"),
        argmax_k=cat("argmax_k"), shift=cat("shift"),
    )


def _run(*, counts, normalizer, cond, lib, jitter, retain_jitter,
         noise, norm_factor,
         nperms, perms, seed, perm_rng, thin_rng, boot_rng,
         n_exc_min, n_tail, alternative, allow_single_sample_group,
         keep_null, gene_names, backend, subset, pseudocount,
         n_boot, sample_names, condition_meta, max_probs,
         gene_chunk, stage1, fit_backend, strata, boot_level, gene_meta) -> WadeResult:
    """The driver proper, over gene chunks (one chunk when ``gene_chunk`` is
    ``None``). The block size is a memory layout, never a numerical choice:
    the jitter is drawn once and indexed per chunk, library sizes are one
    full-matrix pass, and the fold-change fit and the thinning consume their
    random streams in gene order across chunks, so the result is bit-identical
    whatever the block size (``tests/test_chunking.py``).
    """
    g, n = counts.shape
    chunks = gene_chunks(g, gene_chunk)
    i1, i0 = split_groups(cond)
    m_real = capped_nprobs(i1.size, i0.size, max_probs)

    def normalize_chunk(c, rows, jit):
        return _normalize.tpm_like(c, normalizer[rows], lib, noise=noise,
                                   norm_factor=norm_factor, jitter=jit)

    do_subset = nperms > 0 and subset and m_real >= 3

    # The count-native shift correction runs before the chunk loop: its two
    # random streams are consumed in gene order across chunks, and the
    # thinned matrix is held as int32 (counts are integers, so the values
    # are exact) at half the footprint of a float64 resident.
    shift = thinned = None
    if do_subset:
        c_int = np.round(counts)
        shift = fit_fold_change(c_int, cond,
                                alpha=_fit_alpha(normalizer, lib, norm_factor, n),
                                seed=int(thin_rng.integers(2**31)),
                                gene_chunk=gene_chunk, backend=fit_backend)
        # The guard tests the *rounded* values (what the buffer will hold —
        # an unrounded 2**31 - 0.4 rounds past int32) and tolerates an empty
        # matrix via `initial`.
        dtype = (np.int32 if c_int.max(initial=0.0) <= np.iinfo(np.int32).max
                 else np.float64)
        thinned = np.empty(counts.shape, dtype=dtype)
        thin_counts(c_int, cond, shift, thin_rng, gene_chunk=gene_chunk,
                    out=thinned)
        del c_int

    # The pseudocount is not retained by the result: it is `norm_factor /
    # (norm * lib)` scaled, so the result keeps the small inputs and rebuilds
    # it on access. It is materialized once here for the run itself.
    pc = ((normalizer, lib, norm_factor, float(pseudocount))
          if pseudocount > 0 else None)
    pc_full = (one_count(normalizer, lib, counts.shape, norm_factor=norm_factor)
               * pseudocount if pseudocount > 0 else None)

    if nperms > 0:
        if perms is None:
            perms = draw_perms(cond, nperms, seed=seed, rng=perm_rng, strata=strata)
        perms = validate_perms(perms, cond, nperms, strata=strata)
    else:
        perms = None

    W_null = w_obs = stage1_stat = None
    if stage1 in ("gemm", "saddlepoint"):
        w_obs = mean_diff_weights(cond)
        stage1_stat = np.empty(g, dtype=np.float64)
        if perms is not None:
            W_null = np.ascontiguousarray(mean_diff_weights(perms).T)

    tpm = np.empty((g, n), dtype=np.float64)
    null = np.empty((g, nperms), dtype=np.float64) if nperms > 0 else None
    stats_parts: list[WadeStats] = []
    sub_parts: list[SubsetResult] = []
    for ch in chunks:
        jit = jitter[ch]
        x_ch = normalize_chunk(counts[ch], ch, jit)
        tpm[ch] = x_ch
        stats_parts.append(wade_stats(
            x_ch, cond, allow_single_sample_group=allow_single_sample_group,
            max_probs=max_probs))
        if stage1 in ("gemm", "saddlepoint"):
            stage1_stat[ch] = x_ch @ w_obs
            if W_null is not None:
                null[ch] = x_ch @ W_null
        elif nperms > 0:
            null[ch] = null_statistics(x_ch, perms, backend=backend,
                                       max_probs=max_probs)
        if do_subset:
            corrected_ch = normalize_chunk(
                np.asarray(thinned[ch], dtype=np.float64), ch, jit)
            sub_parts.append(subset_test(
                x_ch, cond, perms, corrected_ch, shift[ch],
                alternative=alternative, backend=backend,
                pseudocount=None if pc_full is None else pc_full[ch],
                max_probs=max_probs))
    obs = _concat_stats(stats_parts)
    sub = _concat_subsets(sub_parts) if do_subset else None

    ci = {}
    if n_boot > 0:
        ci = characterization_ci(tpm, cond, pseudocount=pc_full, n_boot=n_boot,
                                 rng=boot_rng, max_probs=max_probs,
                                 gene_chunk=gene_chunk, level=boot_level)
        if sub is None:
            # No interval for a statistic the table does not contain.
            ci = {k: ci[k] for k in ("log2_fc", "mean_shift")}


    return _finish(
        obs=obs, null=null, perms=perms, sub=sub, ci=ci, cond=cond,
        nperms=nperms, seed=seed, n_exc_min=n_exc_min, n_tail=n_tail,
        alternative=alternative, keep_null=keep_null, gene_names=gene_names,
        tpm=tpm, retain_jitter=retain_jitter, pc=pc, n_boot=n_boot,
        sample_names=sample_names, condition_meta=condition_meta,
        max_probs=max_probs, gene_chunk=gene_chunk,
        stage1_stat=stage1_stat, stage1=stage1, fit_backend=fit_backend,
        strata=strata, gene_meta=gene_meta,
    )


def wade(
    counts,
    normalizer,
    cond,
    *,
    nperms: int = DEFAULT_NPERMS,
    seed: int | None = 1,
    alternative: str = "two-sided",
    lib_sizes: np.ndarray | None = None,
    strata=None,
    n_boot: int = 0,
    boot_level: float = 0.95,
    subset: bool = True,
    pseudocount: float = 1.0,
    max_probs: int | None = DEFAULT_MAX_PROBS,
    gene_chunk: int | None = None,
    stage1: str = "grid",
    fit_backend: str = "numpy",
    backend: str = "auto",
    gene_names=None,
    sample_names=None,
    noise: float = _normalize.DEFAULT_NOISE,
    norm_factor: float = _normalize.DEFAULT_NORM_FACTOR,
    n_exc_min: int = DEFAULT_N_EXC_MIN,
    n_tail: int = DEFAULT_N_TAIL,
    jitter: np.ndarray | None = None,
    perms: np.ndarray | None = None,
    keep_null: bool = False,
    allow_single_sample_group: bool = False,
    allow_empty_samples: bool = False,
) -> WadeResult:
    """Run WADE on a raw count matrix.

    Parameters
    ----------
    counts
        Genes x samples **raw counts**: a NumPy array, a polars or pandas
        DataFrame, a SciPy sparse matrix, or a :class:`wade.Counts`. Anything
        but an array goes through :func:`wade.as_counts` with its defaults;
        call that yourself for a transposed matrix or an unusual column
        layout. WADE reads no files.
    normalizer
        A per-gene vector (gene length gives TPM-like values), a genes x
        samples matrix, a scalar (``1.0`` gives CPM), or the name of a
        numeric column carried alongside the counts.
    cond
        ``1`` for case and ``0`` for control, one entry per sample, or a
        :class:`wade.Condition` from :func:`wade.condition`, aligned to the
        sample labels by name.
    nperms
        Permutations. Sets the p-value floors: ``1/(nperms+1)`` empirical,
        ``1/(nperms * n_tail)`` after GPD refinement. ``0`` skips inference.
    seed
        Seeds the jitter, the permutations, the thinning and the bootstrap;
        a seed reproduces a result exactly. ``None`` is not reproducible.
    alternative
        ``"two-sided"`` (default), ``"greater"`` (elevation in cases only) or
        ``"less"``. Applies to both stages.
    lib_sizes
        Per-sample library sizes; computed from the counts and the
        normalizer when omitted. Use it to pass RLE or other size factors.
    strata
        Per-sample stratum labels (study, batch, protocol). Labels are then
        permuted only within each stratum, and the manifest records the
        permutation space that leaves (:func:`wade.permutation_space`).
    n_boot, boot_level
        Bootstrap replicates for intervals on the descriptors, resampled
        within groups; ``0`` (default) skips them. ``boot_level`` is the
        interval's coverage.
    subset
        Run the subset stage and the characterization. Needs at least 3
        grid points, so it is skipped when ``min(n_case, n_ctrl) < 3``.
    pseudocount
        In counts, added to every cell in that sample's normalized units
        before the log-ratio curve is taken (``docs/method.md`` §9.4).
        ``0`` disables.
    max_probs
        Cap on the quantile grid, ``min(n_case, n_ctrl, max_probs)``
        (``docs/method.md`` §1). The realized grid is ``result.nprobs``.
        Keep it at or above ``2.5 / (smallest fraction of interest)``;
        ``None`` removes the cap.
    gene_chunk
        Process genes in blocks of this many to bound peak memory. Purely a
        memory layout: the result is bit-identical whatever the block size.
    stage1
        ``"grid"`` (default): the quantile-grid quadrature and its
        permutation p-value. ``"gemm"``: on a balanced design, the exact
        mean difference with its null as one matrix product, much faster on
        large cohorts. ``"saddlepoint"``: the exact mean difference with its
        permutation p-value computed in closed form, no floor, any balance;
        needs SciPy and costs about 60x the stage-1 permutation loop
        (``docs/method.md`` §6). Both alternatives change the reported
        ``mean_shift`` off the grid quadrature, so they are opt-in.
    fit_backend
        ``"numpy"`` (default) or ``"rust"`` for the fold-change fit's
        bisection. Opt-in because the two backends draw their thinning
        differently and so fit slightly different fold changes for the same
        seed; both are deterministic.
    backend
        ``"auto"`` (default), ``"rust"`` or ``"numpy"`` for the two
        permutation loops. The kernels are bitwise-checked against NumPy, so
        this never changes an answer.
    gene_names, sample_names
        Labels, when ``counts`` is a bare array; otherwise positional.
    noise, norm_factor
        The continuity jitter's width (in counts) and the normalization's
        scale factor (``docs/method.md`` §7).
    n_exc_min, n_tail
        The GPD tail refinement fires for a gene with fewer than
        ``n_exc_min`` exceedances and fits the top ``n_tail`` null draws
        (``docs/method.md`` §6).
    jitter, perms
        Supplied randomness in place of the seeded draws. ``perms`` is a
        ``(nperms, samples)`` matrix of permuted labels.
    keep_null
        Retain the two ``genes x nperms`` null matrices on the result.
    allow_single_sample_group
        Allow a group of one sample (stage 1 only).
    allow_empty_samples
        Run even though some library has a zero size; refused by default,
        naming the samples.

    Returns
    -------
    WadeResult
        Per-gene arrays. The headline columns are ``p_mean_shift`` and
        ``padj_mean_shift`` (stage 1), ``p_subset`` and ``padj_subset``
        (stage 2), and the descriptors ``affected_fraction``, ``direction``
        and ``subset_log2_fc``; :meth:`WadeResult.columns` has the full
        table.
    """
    data, normalizer = _resolve_counts(counts, normalizer, gene_names, sample_names)
    cond_meta = {}
    if isinstance(cond, Condition):
        cond_meta = dict(condition_column=cond.column,
                        case_label=str(cond.case), control_label=str(cond.control))
        cond = cond.vector(data.sample_names)
    cond = np.asarray(cond)
    cond_meta.update(noise=noise, norm_factor=norm_factor, subset=subset,
                     pseudocount=pseudocount,
                     normalizer=_describe_normalizer(normalizer),
                     lib_sizes_supplied=lib_sizes is not None,
                     perms_supplied=perms is not None)
    counts = data.values
    gene_names, sample_names = data.gene_names, data.sample_names
    if cond.shape[0] != counts.shape[1]:
        raise ValueError(
            f"cond has {cond.shape[0]} entries but counts has {counts.shape[1]} samples"
        )
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}; got {alternative!r}")
    if pseudocount < 0:
        raise ValueError(f"pseudocount must be non-negative, got {pseudocount}")
    if perms is not None:
        perms = np.asarray(perms)
        if perms.ndim != 2:
            raise ValueError(f"perms must be a (nperms, samples) matrix, got shape {perms.shape}")
        nperms = perms.shape[0]
    if nperms < 0:
        raise ValueError(f"nperms must be non-negative, got {nperms}")
    split_groups(cond)
    _validate_stage1(stage1, cond, fit_backend)
    if strata is not None:
        from .permutation import strata_indices
        strata_indices(strata, counts.shape[1])          # shape/length check

    # Four independent streams from one seed: jitter, permutations, thinning,
    # bootstrap.
    if seed is None:
        jitter_rng, perm_rng, thin_rng, boot_rng = (np.random.default_rng() for _ in range(4))
    else:
        streams = np.random.SeedSequence(seed).spawn(4)
        jitter_rng, perm_rng, thin_rng, boot_rng = (np.random.default_rng(s) for s in streams)

    user_jitter = jitter is not None
    if jitter is None:
        jitter = _normalize.draw_jitter(counts.shape, noise=noise, rng=jitter_rng)
    else:
        jitter = _normalize._resolve_jitter(jitter, counts.shape, noise, None, None)

    # Library sizes are fixed from the original counts and reused for every
    # thinned matrix: thinning one gene is a counterfactual about that gene,
    # not about the library. Always a full-matrix pass: a column sum's
    # association depends on how it is blocked.
    lib = (_normalize.library_sizes(counts, normalizer) if lib_sizes is None
           else np.asarray(lib_sizes, dtype=np.float64))
    _check_libraries(lib, sample_names, counts.shape[0], allow_empty_samples)

    # The result rebuilds the jitter from the seed rather than carrying it,
    # so keep the array only when the draw is not reproducible.
    retain_jitter = jitter if (user_jitter or seed is None) else None

    return _run(
        counts=counts, normalizer=normalizer, cond=cond, lib=lib,
        jitter=jitter, retain_jitter=retain_jitter,
        noise=noise, norm_factor=norm_factor,
        nperms=nperms, perms=perms, seed=seed, perm_rng=perm_rng,
        thin_rng=thin_rng, boot_rng=boot_rng, n_exc_min=n_exc_min,
        n_tail=n_tail, alternative=alternative,
        allow_single_sample_group=allow_single_sample_group,
        keep_null=keep_null, gene_names=gene_names, backend=backend,
        subset=subset, pseudocount=pseudocount, n_boot=n_boot,
        sample_names=sample_names, condition_meta=cond_meta,
        max_probs=max_probs, gene_chunk=gene_chunk, stage1=stage1,
        fit_backend=fit_backend, strata=strata, boot_level=boot_level,
        gene_meta=data.meta)


def wade_contrast(
    counts,
    normalizer,
    case_samples,
    ctrl_samples,
    *,
    gene_names=None,
    sample_names=None,
    **kwargs,
) -> WadeResult:
    """Run one contrast from two groups of samples, **by name or by index**.

    ``case_samples`` and ``ctrl_samples`` may be sample labels (matched against
    the matrix's sample names, which a DataFrame input supplies) or integer
    column positions. Every other keyword is passed to :func:`wade`.

    Sizes libraries **on the subset**, not on the full matrix, so a library's
    size factor depends only on the genes and samples handed in — change the
    sample set and every normalized value changes.
    """
    data, normalizer = _resolve_counts(counts, normalizer, gene_names, sample_names)
    case_idx = _resolve_samples(case_samples, data.sample_names, "case_samples")
    ctrl_idx = _resolve_samples(ctrl_samples, data.sample_names, "ctrl_samples")

    overlap = np.intersect1d(case_idx, ctrl_idx)
    if overlap.size:
        raise ValueError(
            f"samples appear in both groups: {data.sample_names[overlap].tolist()}")

    cols = np.concatenate([case_idx, ctrl_idx])
    cond = np.concatenate([np.ones(case_idx.size, dtype=int),
                           np.zeros(ctrl_idx.size, dtype=int)])
    sub_norm = normalizer[:, cols] if normalizer.ndim == 2 else normalizer
    kwargs = _reindex_sample_kwargs(kwargs, cols, data.values.shape[1])

    result = wade(data.values[:, cols], sub_norm, cond,
                  gene_names=data.gene_names,
                  sample_names=data.sample_names[cols], **kwargs)
    result.params.update(columns=cols)
    return result


#: Arguments of :func:`wade` whose values are indexed by **sample** along
#: their last axis. :func:`wade_contrast` reorders the matrix into
#: ``[cases..., controls...]``, so each of these has to be carried through
#: the same reordering — passing them positionally onto the reordered
#: columns applies them to the wrong samples, and when the lengths happen to
#: match it does so *silently*.
_SAMPLE_AXIS_KWARGS = ("strata", "lib_sizes", "jitter", "perms")


def _reindex_sample_kwargs(kwargs: dict, cols: np.ndarray, n_samples: int) -> dict:
    """Reorder every per-sample argument onto ``cols``.

    Refuses an argument whose sample axis does not match the *full* matrix:
    a pre-subset array cannot be reindexed and guessing would be the silent
    error this function exists to prevent.
    """
    out = dict(kwargs)
    for name in _SAMPLE_AXIS_KWARGS:
        val = out.get(name)
        if val is None:
            continue
        arr = np.asarray(val)
        if arr.shape[-1] != n_samples:
            raise ValueError(
                f"{name}= must cover every sample of the matrix handed to "
                f"wade_contrast ({n_samples}), because the contrast reorders the "
                f"columns and {name} has to be reordered with them; got "
                f"{arr.shape[-1]}. Pass the full-length {name}, not one already "
                f"subset to the contrast."
            )
        out[name] = arr[..., cols]
    return out


def _resolve_samples(selection, sample_names, what: str) -> np.ndarray:
    """Sample labels or integer positions to column indices, strictly.

    A selection is treated as labels unless every entry is an integer that is
    *not* also a sample name — so a matrix whose samples are literally named
    ``0, 1, 2`` still resolves by name rather than by position.
    """
    sel = list(selection)
    if not sel:
        raise ValueError(f"{what} is empty")
    names = list(sample_names)
    index = {name: i for i, name in enumerate(names)}

    looks_positional = all(
        isinstance(v, (int, np.integer)) and not isinstance(v, bool) and v not in index
        for v in sel)
    if looks_positional:
        idx = np.asarray(sel, dtype=np.intp)
        bad = [int(i) for i in idx if not -len(names) <= i < len(names)]
        if bad:
            raise IndexError(f"{what}: column index out of range for {len(names)} samples: {bad}")
        return idx % len(names)

    missing = [v for v in sel if v not in index]
    if missing:
        raise KeyError(
            f"{what}: {len(missing)} name(s) are not samples of this matrix: "
            f"{missing[:8]}" + (f" and {len(missing) - 8} more" if len(missing) > 8 else "")
            + f". Known samples begin {names[:4]}."
        )
    return np.asarray([index[v] for v in sel], dtype=np.intp)
