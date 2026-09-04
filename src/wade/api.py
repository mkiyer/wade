"""The driver: normalize, observe, permute, refine, adjust.

**WADE takes raw counts, and only raw counts.** The continuity jitter is
applied at count precision before division and the subset stage's null is
built by thinning reads, so a pre-normalized matrix can reproduce neither:
its ties are never broken and its stage-2 null falls back to a division
correction measured wrong at low expression (``docs/method.md`` §10.2). There
was once a ``wade_from_matrix`` entry point for callers who had only a
normalized matrix; it was removed 2026-08-21 because what it offered was a
stage-2 test known to be broken in the regime this package exists for, and
because deleting it leaves exactly one driver here instead of two.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import normalize as _normalize
from .io import Condition, as_counts, result_columns
from .permutation import draw_perms, null_statistics, validate_perms
from .pvalues import (ALTERNATIVES, DEFAULT_N_EXC_MIN, DEFAULT_N_TAIL, bh_adjust,
                      exceedance_counts, perm_pvalues)
from .stats import WadeStats, split_groups, wade_stats
from .subset import SubsetResult, characterization_ci, subset_test
from .thinning import fit_fold_change, one_count, thin_counts

__all__ = ["WadeResult", "wade", "wade_contrast",
           "DEFAULT_NPERMS", "DEFAULT_MAX_PROBS"]

#: Permutations. Sets both the empirical p-value floor ``1/(B+1)`` and the
#: GPD extrapolation floor ``1/(B * n_tail)``, so it changes every small
#: p-value rather than only the runtime. Below 500 the refinement never fires.
DEFAULT_NPERMS = 2000

#: Cap on the quantile grid (``docs/method.md`` §1). Designs at or under
#: 2,000 per group — every design the method was developed on — are
#: untouched; above it the grid stops growing with the cohort and
#: ``affected_fraction``'s resolution is ``1/max_probs``, faithful to a 0.1%
#: subset at half this value (``docs/scaling.md`` §2.1).
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
    jitter: np.ndarray
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

    #: Stage 2's exceedance count and GPD-refinement flag — the stage-1 pair
    #: (``nexc_mean_shift`` / ``refined_mean_shift``) computed for the subset
    #: test too. ``docs/limits.md`` §5 makes a reading rule out of these: a
    #: gene *at* the resolution floor has not been measured there, it has been
    #: censored there, and until now that rule could only be applied to
    #: stage 1.
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
    #: **Read by nothing here.** It exists so figures and tables can optionally
    #: show it (``docs/plotting.md``); dropping it would change no number.
    gene_meta: dict = field(default_factory=dict)

    #: The pseudocount added before the log-ratio curve, per cell — one count
    #: in each sample's normalized units for :func:`wade` (``method.md``
    #: §10.4), or ``None`` when ``pseudocount=0``. Kept so
    #: :meth:`gene_detail` draws the curve the statistics were read from.
    pseudocount: np.ndarray | None = None

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
        """The global fold change the subset stage's null was built under —
        by thinning by default, by division under ``thin=False``
        (``subset.correction`` says which)."""
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
        pc = None if self.pseudocount is None else self.pseudocount[i]
        return wade_gene(self.tpm[i], self.cond, pseudocount=pc,
                         max_probs=self.params.get("max_probs"))

    def columns(self) -> dict[str, np.ndarray]:
        """The result frame as a plain dict of equal-length arrays."""
        cols = {
            "gene": self.gene,
            "mean_shift": self.mean_shift,
            "log2_fc": self.log2_fc,
            "w1": self.w1,
            "case_mean": self.case_mean,
            "ctrl_mean": self.ctrl_mean,
            "p_mean_shift": self.p_mean_shift,
            "padj_mean_shift": self.padj_mean_shift,
        }
        if self.z_mean_shift is not None:
            cols["z_mean_shift"] = self.z_mean_shift
        cols["nexc_mean_shift"] = self.nexc_mean_shift
        cols["refined_mean_shift"] = self.refined_mean_shift
        if self.subset is not None:
            cols.update({
                "subset_stat": self.subset.statistic,
                "p_subset": self.p_subset,
                "padj_subset": self.padj_subset,
                "affected_fraction": self.subset.affected_fraction,
                "subset_log2_fc": self.subset.subset_log2_fc,
                "direction": self.subset.direction,
            })
            if self.z_subset is not None:
                cols["z_subset"] = self.z_subset
            if self.nexc_subset is not None:
                cols["nexc_subset"] = self.nexc_subset
                cols["refined_subset"] = self.refined_subset
        for name, ci in (("affected_fraction", self.ci_affected_fraction),
                         ("direction", self.ci_direction),
                         ("subset_log2_fc", self.ci_subset_log2_fc),
                         ("log2_fc", self.ci_log2_fc),
                         ("mean_shift", self.ci_mean_shift)):
            if ci is not None:
                cols[f"{name}_lo"] = ci[0]
                cols[f"{name}_hi"] = ci[1]
        return cols

    def to_frame(self):
        """The result as a polars DataFrame in the written column order — the
        same table :func:`wade.write_results` writes; see
        :data:`wade.io.RESULT_COLUMNS`. polars is not a dependency of this
        package and is imported here.
        """
        from .io import to_frame

        return to_frame(self)

    def report(self) -> dict:
        """The written column order as a plain dict of arrays. No polars."""
        return result_columns(self)


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


def _check_libraries(lib: np.ndarray, sample_names, allow: bool) -> None:
    """Refuse a library with nothing in it, by name.

    An all-zero column — a failed sample left in the matrix, which is routine
    — has no count scale, so ``tpm_like``'s numerator and denominator are both
    the jitter and **every gene in it normalizes to exactly** ``norm_factor``
    (``docs/method.md`` §8 names the artefact). Those values are not small or
    noisy, they are a constant, and they take part in every quantile the
    contrast reads. Left undetected this is the silent-wrong-answer shape of
    error, so it raises and says which samples: dropping a sample is the
    caller's decision to make, not a default to be applied quietly.
    """
    lib = np.asarray(lib, dtype=np.float64)
    if allow or lib.size == 0 or np.all(lib == 0.0):
        # All-zero across the board means there are no genes at all (a
        # degenerate but well-defined empty matrix), not a failed sample.
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
                f"stage1='grid' (docs/scaling.md §3.1)."
            )


def _resolve_gene_names(gene_names, g: int) -> np.ndarray:
    """Synthesize positional identifiers rather than drop the column."""
    if gene_names is None:
        return np.array([f"gene{i}" for i in range(g)], dtype=object)
    gene_names = np.asarray(gene_names, dtype=object)
    if gene_names.shape != (g,):
        raise ValueError(
            f"gene_names must have one entry per gene: expected {g}, got {gene_names.shape}"
        )
    return gene_names


def _resolve_input(counts, normalizer, cond, gene_names, sample_names):
    """Accept whatever the caller brought, and return arrays the statistic can use.

    ``counts`` goes through :func:`wade.as_counts` (array, polars/pandas frame,
    sparse, or an existing ``Counts``); ``normalizer`` may additionally be the
    name of a numeric column carried alongside the counts, or a scalar; ``cond``
    may be a :class:`wade.Condition`, which is aligned to the sample labels by
    name and raises on any mismatch.
    """
    c = as_counts(counts, gene_names=gene_names, sample_names=sample_names,
                  exclude=(normalizer,) if isinstance(normalizer, str) else ())
    g = c.values.shape[0]

    if isinstance(normalizer, str):
        normalizer = c.normalizer(normalizer)
    elif np.isscalar(normalizer):
        normalizer = np.full(g, float(normalizer), dtype=np.float64)

    meta = {}
    if isinstance(cond, Condition):
        meta = dict(condition_column=cond.column,
                    case_label=str(cond.case), control_label=str(cond.control))
        cond = cond.vector(c.sample_names)
    return c, np.asarray(normalizer, dtype=np.float64), np.asarray(cond), meta


def _perm_z(stat: np.ndarray, null: np.ndarray) -> np.ndarray:
    """The permutation z-score: ``(observed - mean(null)) / sd(null)``,
    per gene, against the gene's own permutation null.

    The analogue of GSEA's normalized enrichment score, in z form because
    WADE's stage-1 statistic is signed (its null mean is ~0, so a ratio to
    the mean would be unstable where NES's ratio to the positive-ES mean is
    not). It exists for **ranking**: the empirical p-value floors at
    ``1/(B+1)`` (GPD-refined, ``1/(B * n_tail)``), so on large cohorts
    thousands of genes tie at the floor while their separations from the
    null differ by orders of magnitude — this number keeps ordering them,
    at no extra cost, because the null matrix is already in hand. It is
    not a calibrated tail probability and must not be converted into one
    via the normal CDF; calibrated resolution is ``docs/scaling.md`` §4.
    """
    mu = null.mean(axis=1)
    sd = null.std(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, (stat - mu) / sd, np.nan)


def _finish(*, obs, null, perms, sub, ci, cond, nperms, seed, n_exc_min, n_tail,
            alternative, keep_null, gene_names, tpm, jitter, pseudocount,
            n_boot, sample_names, condition_meta, max_probs,
            gene_chunk=None, stage1_stat=None, stage1="grid",
            fit_backend="numpy", strata=None, gene_meta=None) -> WadeResult:
    """P-values, BH and the result object, from assembled per-gene arrays.

    Shared by the one-pass path (:func:`_run`) and the gene-chunked driver
    (:func:`_wade_chunked`), so the two cannot drift on anything downstream
    of the per-gene statistics.

    ``stage1_stat`` is the observed stage-1 vector when it is not the grid
    quadrature — the ``stage1="gemm"`` exact mean difference — and it is then
    both what the p-value compares against the null and what the result
    reports as ``mean_shift``, because the two must be one statistic.
    """
    g = obs.mean_shift.shape[0]
    stat = obs.mean_shift if stage1_stat is None else stage1_stat
    if stage1 == "saddlepoint":
        # No permutation enters the p-value: the exact permutation tail is
        # computed from the values themselves (`wade.saddlepoint`,
        # `scaling.md` §4.9). The null is still built when permutations were
        # drawn for stage 2 -- it is one matrix product -- so `z_mean_shift`
        # and `nexc_mean_shift` keep their meaning as ranking and diagnostic
        # columns. `refined_mean_shift` is False everywhere by construction:
        # there is nothing to refine when there is no floor.
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
            from dataclasses import replace
            sub = replace(sub, null=None)

    return WadeResult(
        gene=_resolve_gene_names(gene_names, g),
        mean_shift=stat, w1=obs.w1, fc=obs.fc, log2_fc=obs.log2_fc,
        case_mean=obs.case_mean, ctrl_mean=obs.ctrl_mean,
        p_mean_shift=p_mean, padj_mean_shift=bh_adjust(p_mean),
        nexc_mean_shift=nexc, refined_mean_shift=refined,
        stats=obs, tpm=tpm, jitter=jitter, perms=perms,
        null_mean_shift=null if keep_null else None,
        subset=sub, p_subset=p_subset, padj_subset=padj_subset,
        nexc_subset=nexc_subset, refined_subset=refined_subset,
        z_mean_shift=z_mean, z_subset=z_subset,
        cond=cond, sample_names=sample_names, gene_meta=dict(gene_meta or {}),
        pseudocount=pseudocount,
        ci_affected_fraction=ci.get("affected_fraction"),
        ci_direction=ci.get("direction"),
        ci_subset_log2_fc=ci.get("subset_log2_fc"),
        ci_log2_fc=ci.get("log2_fc"),
        ci_mean_shift=ci.get("mean_shift"),
        params=dict(nperms=nperms, seed=seed, alternative=alternative,
                    n_exc_min=n_exc_min, n_tail=n_tail,
                    nprobs=obs.nprobs, max_probs=max_probs, n1=obs.n1, n0=obs.n0,
                    n_boot=n_boot, gene_chunk=gene_chunk, stage1=stage1,
                    fit_backend=fit_backend,
                    strata=None if strata is None else np.asarray(strata),
                    correction=None if sub is None else sub.correction,
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
        statistic=cat("statistic"), null=cat("null"), r=cat("r"), b=cat("b"),
        affected_fraction=cat("affected_fraction"), direction=cat("direction"),
        argmax_k=cat("argmax_k"), shift=cat("shift"), r_test=cat("r_test"),
        correction=first.correction,
    )


def _wade_chunked(*, counts, normalizer, cond, lib, jitter, noise, norm_factor,
                  nperms, perms, seed, perm_rng, thin_rng, boot_rng,
                  n_exc_min, n_tail, alternative, allow_single_sample_group,
                  keep_null, gene_names, backend, subset, thin, pseudocount,
                  n_boot, sample_names, condition_meta, max_probs,
                  gene_chunk, stage1="grid", fit_backend="numpy",
                  strata=None, boot_level=0.95, gene_meta=None) -> WadeResult:
    """The gene-chunked driver — ``docs/scaling.md`` §2.2.

    **Bit-identical to the unchunked path**, by construction rather than by
    tolerance, and pinned by ``tests/test_chunking.py``:

    * the jitter is drawn once for the whole matrix and *indexed* per chunk;
    * library sizes are one full-matrix pass, fixed before any chunking;
    * the fold-change fit and the thinning consume their random streams in
      gene order across chunks (see :func:`wade.thinning.fit_fold_change` and
      :func:`wade.thinning.thin_counts` for how);
    * everything else — normalization, the observed statistics, both
      permutation nulls, the bootstrap — is per-gene arithmetic on shared
      permutations, so chunk boundaries cannot reach it.

    Peak memory is the handful of full ``genes x samples`` residents (counts,
    jitter, ``tpm``, the pseudocount, the thinned counts as int32) plus a few
    chunk-sized transients, instead of the ~8 full-matrix transients of the
    one-pass path.
    """
    from .quantiles import capped_nprobs
    from .thinning import gene_chunks

    g, n = counts.shape
    chunks = gene_chunks(g, gene_chunk)
    i1, i0 = split_groups(cond)
    m_real = capped_nprobs(i1.size, i0.size, max_probs)

    def normalize_chunk(c, rows, jit):
        return _normalize.tpm_like(c, normalizer[rows], lib, noise=noise,
                                   norm_factor=norm_factor, jitter=jit)

    # The count-native shift correction runs before the chunk loop: its two
    # random streams are consumed in gene order across chunks, and the
    # thinned matrix is held as int32 (counts are integers, so the values
    # are exact) at half the footprint of a float64 resident.
    shift = thinned = None
    if thin and nperms > 0 and subset and min(i1.size, i0.size) >= 3:
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

    pc_full = (one_count(normalizer, lib, counts.shape, norm_factor=norm_factor)
               * pseudocount if pseudocount > 0 else None)

    if nperms > 0:
        if perms is None:
            perms = draw_perms(cond, nperms, seed=seed, rng=perm_rng, strata=strata)
        perms = validate_perms(perms, cond, nperms, strata=strata)
    else:
        perms = None
    do_subset = nperms > 0 and subset and m_real >= 3

    W_null = w_obs = stage1_stat = None
    if stage1 in ("gemm", "saddlepoint"):
        from .permutation import _mean_diff_weights
        w_obs = _mean_diff_weights(cond)
        stage1_stat = np.empty(g, dtype=np.float64)
        if perms is not None:
            W_null = np.ascontiguousarray(_mean_diff_weights(perms).T)

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
            pc_ch = None if pc_full is None else pc_full[ch]
            corrected_ch = shift_ch = None
            if thinned is not None:
                corrected_ch = normalize_chunk(
                    np.asarray(thinned[ch], dtype=np.float64), ch, jit)
                shift_ch = shift[ch]
            sub_parts.append(subset_test(
                x_ch, cond, perms, alternative=alternative, backend=backend,
                pseudocount=pc_ch, corrected=corrected_ch, shift=shift_ch,
                max_probs=max_probs))
    obs = _concat_stats(stats_parts)
    sub = _concat_subsets(sub_parts) if do_subset else None

    ci = {}
    # Gated on the subset stage, not merely on n_boot: an interval for a
    # statistic the same table does not contain reads as a corrupt table.
    if n_boot > 0 and sub is not None:
        ci = characterization_ci(tpm, cond, pseudocount=pc_full, n_boot=n_boot,
                                 rng=boot_rng, max_probs=max_probs,
                                 gene_chunk=gene_chunk, level=boot_level)

    return _finish(
        obs=obs, null=null, perms=perms, sub=sub, ci=ci, cond=cond,
        nperms=nperms, seed=seed, n_exc_min=n_exc_min, n_tail=n_tail,
        alternative=alternative, keep_null=keep_null, gene_names=gene_names,
        tpm=tpm, jitter=jitter, pseudocount=pc_full, n_boot=n_boot,
        sample_names=sample_names, condition_meta=condition_meta,
        max_probs=max_probs, gene_chunk=gene_chunk,
        stage1_stat=stage1_stat, stage1=stage1, fit_backend=fit_backend,
        strata=strata, gene_meta=gene_meta,
    )


def wade(
    counts: np.ndarray,
    normalizer,
    cond: np.ndarray,
    *,
    lib_sizes: np.ndarray | None = None,
    nperms: int = DEFAULT_NPERMS,
    noise: float = _normalize.DEFAULT_NOISE,
    norm_factor: float = _normalize.DEFAULT_NORM_FACTOR,
    seed: int | None = 1,
    jitter: np.ndarray | None = None,
    perms: np.ndarray | None = None,
    gene_names=None,
    n_exc_min: int = DEFAULT_N_EXC_MIN,
    n_tail: int = DEFAULT_N_TAIL,
    alternative: str = "two-sided",
    allow_single_sample_group: bool = False,
    keep_null: bool = False,
    backend: str = "auto",
    subset: bool = True,
    thin: bool = True,
    pseudocount: float = 1.0,
    n_boot: int = 0,
    sample_names=None,
    max_probs: int | None = DEFAULT_MAX_PROBS,
    gene_chunk: int | None = None,
    stage1: str = "grid",
    fit_backend: str = "numpy",
    strata=None,
    allow_empty_samples: bool = False,
    boot_level: float = 0.95,
) -> WadeResult:
    """Run WADE on a raw count matrix. The primary entry point.

    Parameters
    ----------
    counts
        Genes x samples raw counts: a NumPy array, a polars or pandas
        DataFrame, a sparse matrix, or a :class:`wade.Counts`. Anything but an
        array goes through :func:`wade.as_counts` with its defaults — call that
        yourself for a transposed matrix (``genes="columns"``) or an unusual
        column layout. **WADE reads no files**; your reader does that
        (``ROADMAP.md`` §1).
    normalizer
        A per-gene vector, a full genes x samples matrix, a scalar, or the
        **name of a numeric column** carried alongside the counts (a
        featureCounts ``"Length"``).
    cond
        Binary vector, 1 = case and 0 = control, one entry per column — or a
        :class:`wade.Condition` from :func:`wade.condition`, which is aligned
        to the sample labels **by name** and raises on any mismatch.
    alternative
        ``"two-sided"`` (default) detects differences in either direction;
        ``"greater"`` only elevation in cases, ``"less"`` only reduction.
        Applies to both stages, so they cannot disagree about what counts as
        a finding. Direction is carried by ``mean_shift``'s sign and by
        ``direction``.
    jitter, perms
        Supplied randomness, **on the production argument path**, because R's
        Mersenne-Twister and NumPy's PCG64 cannot agree on a shared seed and a
        fixture path that bypassed production code would validate code nobody
        runs.
    n_exc_min, n_tail
        The GPD tail refinement (``docs/method.md`` §6). Refinement fires for
        a gene with fewer than ``n_exc_min`` exceedances, fitting the top
        ``n_tail`` null draws — **so together with** ``nperms`` **these set
        the smallest p-value the run can report**, ``1/(nperms * n_tail)``.
        That floor is an honesty constraint, not a numerical guard; raising
        resolution means more permutations, not a smaller floor.
    subset
        Run the subset test and the characterization. Requires
        ``min(n_case, n_ctrl) >= 3``.
    thin
        Build the subset stage's null by **binomial thinning** of the raw
        counts under the fitted global fold change (``docs/method.md``
        §10.3), which is exact for counts where the division it replaces is
        not. ``False`` falls back to that division — kept as the comparison
        that shows why thinning exists, not as an analysis option.
    pseudocount
        In **counts**. Added to every cell, in that sample's normalized
        units, before the log-ratio curve is taken (``§10.4``): a zero means
        "less than one", and without this the tie-breaking jitter turns it
        into a log-ratio of 7–10. Default one count; ``0`` disables.
    boot_level
        Coverage of the bootstrap intervals; ``0.95`` by default.
    n_boot
        Bootstrap replicates for intervals on ``affected_fraction``,
        ``direction``, ``subset_log2_fc``, ``log2_fc`` and ``mean_shift``
        (``§10.5``); ``0`` (default) skips them.
        Resampled within groups. A few hundred is enough; the cost is a few
        quantile grids per replicate.
    max_probs
        Cap on the quantile grid (``docs/method.md`` §1). The realized grid is
        ``min(n_case, n_ctrl, max_probs)``, reported as ``result.nprobs`` and
        recorded in the manifest. Designs at or under the default 2,000 per
        group are untouched; above it the cap bounds every per-gene array and
        sets ``affected_fraction``'s resolution to ``1/max_probs`` — keep
        ``max_probs >= 2.5 / (smallest fraction of interest)``. ``None``
        removes the cap.
    gene_chunk
        Process the genes in blocks of this many, so peak memory stops
        depending on the number of genes (``docs/scaling.md`` §2.2). **Not a
        numerical choice**: the jitter is indexed rather than redrawn and
        every random stream is consumed in gene order, so the block size
        cannot change a number — ``None`` (the default) is simply one block
        over the whole matrix, and it is the *same* code path. A few thousand
        genes is a good block on a large cohort.
    stage1
        How the stage-1 statistic and its null are computed. ``"grid"`` (the
        default) is the quantile-grid quadrature, parity-pinned against the R
        reference. ``"gemm"`` — **balanced designs only** — computes the
        statistic as the exact difference of group means and the entire null
        as one matrix product (``docs/scaling.md`` §3.1; measured 139–185×
        faster at large n). On a balanced, uncapped design the two agree to
        ~1e-9 relative — not bitwise, because BLAS reassociates — and under a
        ``max_probs`` cap the GEMM statistic is arguably the better number:
        it is the exact mean difference where the capped quadrature drifts.
        ``result.mean_shift`` then reports the statistic the p-value actually
        tested; the grid quadrature stays available as
        ``result.stats.mean_shift``. Stage 2 is unaffected either way.

        ``"saddlepoint"`` — **any geometry** — is the same exact mean
        difference with its p-value computed rather than sampled: the
        permutation tail of a subset sum has a double-saddlepoint form
        (:mod:`wade.saddlepoint`, ``docs/scaling.md`` §4.9), so stage 1 gets
        **no resolution floor at all**. That is the point of it. The empirical
        p-value stops at ``1/(B+1)`` and the GPD refinement that fills the gap
        was measured 3–4,906× conservative on stage 1, which costs power
        exactly where BH decides; the saddlepoint agrees with 1e8-permutation
        brute force to 0.99–1.00 in the median and is never worse than 0.61
        anti-conservative. It is **opt-in and changes reported numbers**, so
        it is named rather than inferred. ``z_mean_shift`` and
        ``nexc_mean_shift`` still come from the permutation null when one was
        drawn for stage 2; ``refined_mean_shift`` is False throughout, there
        being no floor to refine past. Needs SciPy, imported only when used.

        **Cost scales with the cohort, not the gene count**: for 20,000 genes,
        0.7 min at 40 v 40, 4.6 at 300 v 300, 17.6 at 1,000 v 1,000 and 50 at
        3,000 v 3,000 — about 60× the stage-1 permutation loop it replaces, at
        every size, since both are ``O(n)`` per gene. Worth it where the
        permutation floor is what limits you; at tens of thousands of samples
        it is not, and there the floor is not the binding constraint anyway.
    fit_backend
        ``"numpy"`` (default) or ``"rust"`` for the fold-change fit's
        bisection (``docs/scaling.md`` §3.3). **Opt-in, and unlike**
        ``backend`` **it changes the realized fit**: no two binomial samplers
        consume randomness alike, so the fitted fold changes for a given seed
        differ between the two at the resolution of the bisection — which is
        why the kernel is never dispatched automatically. Both are
        deterministic given the seed and chunk-invariant; the kernel is
        parallel over genes and several times faster.
    strata
        Per-sample stratum labels (study, batch, protocol, donor). When
        given, the permutation null is **restricted**: labels are shuffled
        only *within* each stratum, so batch structure is held fixed instead
        of being tested as if it were biology (``docs/limits.md`` §2.2). The
        price is permutation space — a stratum with only one class present
        contributes no freedom at all — so the realized space and its
        implied p-value floor are computed and recorded
        (:func:`wade.permutation_space`, and the manifest's
        ``design.permutation_space``). Supplying your own ``perms`` alongside
        ``strata`` validates them against the stratum counts.
    allow_empty_samples
        Run even though some library has a zero size. Refused by default:
        every gene in such a sample normalizes to the ``norm_factor``
        constant, which then takes part in every quantile the contrast reads
        (:func:`wade.library_qc` reports per-library depth and complexity).
    """
    data, normalizer, cond, cond_meta = _resolve_input(
        counts, normalizer, cond, gene_names, sample_names)
    cond_meta.update(entry_point="wade", noise=noise, norm_factor=norm_factor,
                     thin=thin, subset=subset,
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
    split_groups(cond)
    _validate_stage1(stage1, cond, fit_backend)
    if strata is not None:
        from .permutation import strata_indices
        strata_indices(strata, counts.shape[1])          # shape/length check

    # Four independent streams from one seed: jitter, permutations, thinning,
    # bootstrap. Spawned children are deterministic by index, so adding the
    # third and fourth left the first two — and every stage-1 number — as they
    # were.
    if seed is None:
        jitter_rng, perm_rng, thin_rng, boot_rng = (np.random.default_rng() for _ in range(4))
    else:
        js, ps, ts, bs = np.random.SeedSequence(seed).spawn(4)
        jitter_rng, perm_rng, thin_rng, boot_rng = (np.random.default_rng(s_) for s_ in (js, ps, ts, bs))

    if jitter is None:
        jitter = _normalize.draw_jitter(counts.shape, noise=noise, rng=jitter_rng)
    else:
        # Validated here, against the full matrix, so both drivers refuse a
        # mis-shaped jitter identically — the chunked driver only ever hands
        # tpm_like row slices, which would let a wrong shape through.
        jitter = _normalize._resolve_jitter(jitter, counts.shape, noise, None, None)

    # Library sizes are fixed from the original counts and reused for every
    # thinned matrix: thinning one gene is a counterfactual about that gene,
    # not about the library. Computed on the full matrix in both drivers —
    # a column sum's association depends on how it is blocked, so chunking
    # it would move every normalized value by an ulp.
    lib = (_normalize.library_sizes(counts, normalizer) if lib_sizes is None
           else np.asarray(lib_sizes, dtype=np.float64))
    _check_libraries(lib, sample_names, allow_empty_samples)

    return _wade_chunked(
        counts=counts, normalizer=normalizer, cond=cond, lib=lib,
        jitter=jitter, noise=noise, norm_factor=norm_factor,
        nperms=nperms, perms=perms, seed=seed, perm_rng=perm_rng,
        thin_rng=thin_rng, boot_rng=boot_rng, n_exc_min=n_exc_min,
        n_tail=n_tail, alternative=alternative,
        allow_single_sample_group=allow_single_sample_group,
        keep_null=keep_null, gene_names=gene_names, backend=backend,
        subset=subset, thin=thin, pseudocount=pseudocount, n_boot=n_boot,
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
    nperms: int = DEFAULT_NPERMS,
    **kwargs,
) -> WadeResult:
    """Run one contrast from two groups of samples, **by name or by index**.

    ``case_samples`` and ``ctrl_samples`` may be sample labels (matched against
    the matrix's sample names, which a DataFrame input supplies) or integer
    column positions. Names are the point: integer positions are what the
    caller almost never has.

    Sizes libraries **on the subset**, not on the full matrix, so a library's
    size factor depends only on the genes and samples handed in — change the
    sample set and every normalized value changes.
    """
    data = as_counts(counts, gene_names=gene_names, sample_names=sample_names,
                     exclude=(normalizer,) if isinstance(normalizer, str) else ())
    case_idx = _resolve_samples(case_samples, data.sample_names, "case_samples")
    ctrl_idx = _resolve_samples(ctrl_samples, data.sample_names, "ctrl_samples")

    overlap = np.intersect1d(case_idx, ctrl_idx)
    if overlap.size:
        raise ValueError(
            f"samples appear in both groups: {data.sample_names[overlap].tolist()}")

    cols = np.concatenate([case_idx, ctrl_idx])
    cond = np.concatenate([np.ones(case_idx.size, dtype=int),
                           np.zeros(ctrl_idx.size, dtype=int)])
    if isinstance(normalizer, str):
        normalizer = data.normalizer(normalizer)
    elif np.isscalar(normalizer):
        normalizer = np.full(data.values.shape[0], float(normalizer), dtype=np.float64)
    normalizer = np.asarray(normalizer, dtype=np.float64)
    sub_norm = normalizer[:, cols] if normalizer.ndim == 2 else normalizer
    kwargs = _reindex_sample_kwargs(kwargs, cols, data.values.shape[1])

    result = wade(data.values[:, cols], sub_norm, cond, nperms=nperms,
                  gene_names=data.gene_names,
                  sample_names=data.sample_names[cols], **kwargs)
    result.params.update(n_case=int(case_idx.size),
                         n_ctrl=int(ctrl_idx.size), columns=cols)
    return result


#: Arguments of :func:`wade` whose values are indexed by **sample**, and which
#: axis of the value the samples run along. :func:`wade_contrast` reorders the
#: matrix into ``[cases..., controls...]``, so each of these has to be carried
#: through the same reordering — passing them positionally onto the reordered
#: columns applies them to the wrong samples, and when the lengths happen to
#: match it does so *silently*.
_SAMPLE_AXIS_KWARGS = {"strata": -1, "lib_sizes": -1, "jitter": 1, "perms": 1}


def _reindex_sample_kwargs(kwargs: dict, cols: np.ndarray, n_samples: int) -> dict:
    """Reorder every per-sample argument onto ``cols``.

    Refuses an argument whose sample axis does not match the *full* matrix:
    a pre-subset array cannot be reindexed and guessing would be the silent
    error this function exists to prevent.
    """
    out = dict(kwargs)
    for name, axis in _SAMPLE_AXIS_KWARGS.items():
        val = out.get(name)
        if val is None:
            continue
        arr = np.asarray(val)
        got = arr.shape[axis] if arr.ndim > (axis if axis >= 0 else 0) else arr.shape[-1]
        if got != n_samples:
            raise ValueError(
                f"{name}= must cover every sample of the matrix handed to "
                f"wade_contrast ({n_samples}), because the contrast reorders the "
                f"columns and {name} has to be reordered with them; got {got}. "
                f"Pass the full-length {name}, not one already subset to the "
                f"contrast."
            )
        out[name] = arr[..., cols] if axis != -1 or arr.ndim > 1 else arr[cols]
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
