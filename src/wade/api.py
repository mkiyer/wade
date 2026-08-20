"""The driver: normalize, observe, permute, refine, adjust.

The entry point takes **raw counts**, not a normalized matrix
(``docs/method.md`` §8). The continuity jitter is applied at count precision
before division, so a pre-normalized matrix cannot reproduce it;
:func:`wade_from_matrix` exists for callers who have one anyway and says
plainly what it costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import normalize as _normalize
from .io import Condition, as_counts, result_columns
from .permutation import draw_perms, null_statistics, validate_perms
from .pvalues import ALTERNATIVES, DEFAULT_N_EXC_MIN, DEFAULT_N_TAIL, bh_adjust, perm_pvalues
from .stats import WadeStats, split_groups, wade_stats
from .subset import SubsetResult, characterization_ci, subset_test
from .thinning import fit_fold_change, one_count, thin_counts

__all__ = ["WadeResult", "wade", "wade_from_matrix", "wade_contrast", "DEFAULT_NPERMS"]

#: Permutations. Sets both the empirical p-value floor ``1/(B+1)`` and the
#: GPD extrapolation floor ``1/(B * n_tail)``, so it changes every small
#: p-value rather than only the runtime. Below 500 the refinement never fires.
DEFAULT_NPERMS = 2000


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

    #: The condition vector the contrast was run on, one entry per column of
    #: ``tpm``. Kept so a single gene can be re-derived from the result alone
    #: (:meth:`gene_detail`, and the plotting layer).
    cond: np.ndarray | None = None

    #: Sample labels, one per column of ``tpm`` — from the input frame, the
    #: ``sample_names`` argument, or positional.
    sample_names: np.ndarray | None = None

    #: The pseudocount added before the log-ratio curve, per cell — one count
    #: in each sample's normalized units for :func:`wade` (``method.md``
    #: §10.4), whatever the caller passed for :func:`wade_from_matrix`, or
    #: ``None``. Kept so :meth:`gene_detail` draws the curve the statistics
    #: were read from.
    pseudocount: np.ndarray | None = None

    #: Bootstrap 95% intervals, ``(2, genes)`` each, when ``n_boot > 0``.
    ci_affected_fraction: np.ndarray | None = None
    ci_direction: np.ndarray | None = None
    ci_log2_fc: np.ndarray | None = None

    @property
    def affected_fraction(self) -> np.ndarray | None:
        return None if self.subset is None else self.subset.affected_fraction

    @property
    def direction(self) -> np.ndarray | None:
        return None if self.subset is None else self.subset.direction

    @property
    def fitted_fold_change(self) -> np.ndarray | None:
        """The global fold change the subset stage's null was built under —
        by thinning for :func:`wade`, by division for :func:`wade_from_matrix`
        (``subset.correction`` says which)."""
        return None if self.subset is None else self.subset.shift

    @property
    def nprobs(self) -> int:
        """The quantile-grid resolution, ``min(n_case, n_ctrl)``.

        A property of the design, not a parameter: no argument raises it, and
        it is the resolution limit on the affected fraction — nothing finer
        than ``1/nprobs`` is estimable.
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
        return wade_gene(self.tpm[i], self.cond, pseudocount=pc)

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
        if self.subset is not None:
            cols.update({
                "subset_stat": self.subset.statistic,
                "p_subset": self.p_subset,
                "padj_subset": self.padj_subset,
                "affected_fraction": self.subset.affected_fraction,
                "direction": self.subset.direction,
            })
        for name, ci in (("affected_fraction", self.ci_affected_fraction),
                         ("direction", self.ci_direction), ("log2_fc", self.ci_log2_fc)):
            if ci is not None:
                cols[f"{name}_lo"] = ci[0]
                cols[f"{name}_hi"] = ci[1]
        return cols

    def to_polars(self):
        """The raw :meth:`columns` dict as a polars DataFrame.

        :meth:`to_frame` is the *report* frame — the written column order, with
        ``neglog10_p_*``. This one is the in-memory column set, unordered.
        polars is not a dependency of this package.
        """
        import polars as pl

        return pl.DataFrame(self.columns())

    def to_frame(self):
        """The result as a polars DataFrame in the written column order.

        The same table :func:`wade.write_results` writes; see
        :data:`wade.io.RESULT_COLUMNS`.
        """
        from .io import to_frame

        return to_frame(self)

    def report(self) -> dict:
        """The written column order as a plain dict of arrays. No polars."""
        return result_columns(self)


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


def _run(x, cond, *, nperms, perms, seed, perm_rng, n_exc_min, n_tail,
         alternative, allow_single_sample_group, keep_null, gene_names,
         backend, subset, jitter, tpm, pseudocount=None, corrected=None,
         shift=None, n_boot=0, boot_rng=None, sample_names=None,
         condition_meta=None) -> WadeResult:
    cond = np.asarray(cond)
    g = x.shape[0]
    obs = wade_stats(x, cond, allow_single_sample_group=allow_single_sample_group)

    if nperms > 0:
        if perms is None:
            perms = draw_perms(cond, nperms, seed=seed, rng=perm_rng)
        perms = validate_perms(perms, cond, nperms)
        null = null_statistics(x, perms, backend=backend)
        p_mean, nexc, refined = perm_pvalues(
            obs.mean_shift, null, n_exc_min=n_exc_min, n_tail=n_tail,
            alternative=alternative,
        )
    else:
        perms = None
        null = None
        p_mean = np.full(g, np.nan)
        nexc = np.zeros(g, dtype=np.int64)
        refined = np.zeros(g, dtype=bool)

    sub = p_subset = padj_subset = None
    if nperms > 0 and subset and obs.nprobs >= 3:
        sub = subset_test(x, cond, perms, alternative=alternative, backend=backend,
                          pseudocount=pseudocount, corrected=corrected, shift=shift)
        # The subset statistic is already a maximum over widths, oriented by
        # `alternative` inside the scan, so its p-value is an upper-tail one.
        p_subset, _, _ = perm_pvalues(
            sub.statistic, sub.null, n_exc_min=n_exc_min, n_tail=n_tail,
            alternative="greater",
        )
        padj_subset = bh_adjust(p_subset)
        if not keep_null:
            from dataclasses import replace
            sub = replace(sub, null=None)

    ci = {}
    if n_boot > 0 and obs.nprobs >= 2:
        ci = characterization_ci(x, cond, pseudocount=pseudocount, n_boot=n_boot, rng=boot_rng)

    return WadeResult(
        gene=_resolve_gene_names(gene_names, g),
        mean_shift=obs.mean_shift, w1=obs.w1, fc=obs.fc, log2_fc=obs.log2_fc,
        case_mean=obs.case_mean, ctrl_mean=obs.ctrl_mean,
        p_mean_shift=p_mean, padj_mean_shift=bh_adjust(p_mean),
        nexc_mean_shift=nexc, refined_mean_shift=refined,
        stats=obs, tpm=tpm, jitter=jitter, perms=perms,
        null_mean_shift=null if keep_null else None,
        subset=sub, p_subset=p_subset, padj_subset=padj_subset,
        cond=cond, sample_names=sample_names, pseudocount=pseudocount,
        ci_affected_fraction=ci.get("affected_fraction"),
        ci_direction=ci.get("direction"),
        ci_log2_fc=ci.get("log2_fc"),
        params=dict(nperms=nperms, seed=seed, alternative=alternative,
                    n_exc_min=n_exc_min, n_tail=n_tail,
                    nprobs=obs.nprobs, n1=obs.n1, n0=obs.n0, n_boot=n_boot,
                    correction=None if sub is None else sub.correction,
                    **(condition_meta or {})),
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
    subset
        Run the subset test and the characterization. Requires
        ``min(n_case, n_ctrl) >= 3``.
    thin
        Build the subset stage's null by **binomial thinning** of the raw
        counts under the fitted global fold change (``docs/method.md``
        §10.3), which is exact for counts where the division it replaces is
        not. ``False`` falls back to the division, which is what
        :func:`wade_from_matrix` has to do without counts.
    pseudocount
        In **counts**. Added to every cell, in that sample's normalized
        units, before the log-ratio curve is taken (``§10.4``): a zero means
        "less than one", and without this the tie-breaking jitter turns it
        into a log-ratio of 7–10. Default one count; ``0`` disables.
    n_boot
        Bootstrap replicates for 95% intervals on ``affected_fraction``,
        ``direction`` and ``log2_fc`` (``§10.5``); ``0`` (default) skips them.
        Resampled within groups. A few hundred is enough; the cost is a few
        quantile grids per replicate.
    """
    data, normalizer, cond, cond_meta = _resolve_input(
        counts, normalizer, cond, gene_names, sample_names)
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
        jitter = np.asarray(jitter, dtype=np.float64)

    # Library sizes are fixed from the original counts and reused for every
    # thinned matrix: thinning one gene is a counterfactual about that gene,
    # not about the library.
    lib = (_normalize.library_sizes(counts, normalizer) if lib_sizes is None
           else np.asarray(lib_sizes, dtype=np.float64))
    normalize = lambda c: _normalize.tpm_like(  # noqa: E731
        c, normalizer, lib, noise=noise, norm_factor=norm_factor, jitter=jitter)
    tpm = normalize(counts)
    # The fold-change fit compares the groups' middles on the normalized scale
    # *without* the jitter: a hundredth of a count has no business in a
    # fold-change estimate, and on an all-zero gene it would otherwise decide
    # which group is "higher" and keep deciding it at every bisection step.
    no_jitter = np.zeros(counts.shape)
    normalize_for_fit = lambda c: _normalize.tpm_like(  # noqa: E731
        c, normalizer, lib, noise=noise, norm_factor=norm_factor, jitter=no_jitter)

    pc = one_count(normalizer, lib, counts.shape, norm_factor=norm_factor) * pseudocount \
        if pseudocount > 0 else None

    corrected = shift = None
    if thin and nperms > 0 and subset and min(np.sum(cond == 1), np.sum(cond == 0)) >= 3:
        # Thinning draws reads, so it works on integers. Estimated counts
        # (salmon, kallisto) are not integers; they are rounded for the
        # thinning only — the observed matrix and the characterization use
        # the values as given.
        c_int = np.round(counts)
        shift = fit_fold_change(c_int, cond, normalize_for_fit, seed=int(thin_rng.integers(2**31)))
        corrected = normalize(thin_counts(c_int, cond, shift, thin_rng))

    return _run(
        tpm, cond, nperms=nperms, perms=perms, seed=seed, perm_rng=perm_rng,
        n_exc_min=n_exc_min, n_tail=n_tail, alternative=alternative,
        allow_single_sample_group=allow_single_sample_group, keep_null=keep_null,
        gene_names=gene_names, backend=backend, subset=subset,
        jitter=jitter, tpm=tpm, pseudocount=pc, corrected=corrected, shift=shift,
        n_boot=n_boot, boot_rng=boot_rng, sample_names=sample_names,
        condition_meta=cond_meta,
    )


def wade_from_matrix(
    x: np.ndarray,
    cond: np.ndarray,
    *,
    nperms: int = DEFAULT_NPERMS,
    seed: int | None = 1,
    perms: np.ndarray | None = None,
    gene_names=None,
    n_exc_min: int = DEFAULT_N_EXC_MIN,
    n_tail: int = DEFAULT_N_TAIL,
    alternative: str = "two-sided",
    allow_single_sample_group: bool = False,
    keep_null: bool = False,
    backend: str = "auto",
    subset: bool = True,
    pseudocount: float = 0.0,
    n_boot: int = 0,
    sample_names=None,
) -> WadeResult:
    """Run WADE on a matrix that is **already on a comparable scale**.

    Be explicit about what this costs: **no continuity jitter is applied**,
    because it cannot be. The jitter is added at count precision before
    division, and once counts have been divided by a normalizer and a library
    size the information needed to reconstruct that perturbation is gone.

    So this is a **variant of the test without tie-breaking**. On sparse,
    zero-heavy data many samples share a count of zero, the quantile grid
    degenerates into flat runs, and the statistic reads those runs as genuine
    agreement. It also loses the strict positivity the jitter provides, which
    the log-ratio curve needs.

    And the subset stage **cannot thin** what it cannot count: its null is
    built by the division correction, which at low expression is not exact
    (``docs/method.md`` §10.2). ``pseudocount`` is in the matrix's own units
    (there is no "one count" to convert) and defaults to none.
    """
    data, _, cond, cond_meta = _resolve_input(x, 1.0, cond, gene_names, sample_names)
    x = data.values
    gene_names, sample_names = data.gene_names, data.sample_names
    if cond.shape[0] != x.shape[1]:
        raise ValueError(
            f"cond has {cond.shape[0]} entries but the matrix has {x.shape[1]} samples"
        )
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}; got {alternative!r}")
    split_groups(cond)
    if pseudocount < 0:
        raise ValueError(f"pseudocount must be non-negative, got {pseudocount}")
    if seed is None:
        perm_rng, boot_rng = np.random.default_rng(), np.random.default_rng()
    else:
        children = np.random.SeedSequence(seed).spawn(4)
        perm_rng, boot_rng = np.random.default_rng(children[1]), np.random.default_rng(children[3])
    pc = np.full(x.shape, float(pseudocount)) if pseudocount > 0 else None
    return _run(
        x, cond, nperms=nperms, perms=perms, seed=seed, perm_rng=perm_rng,
        n_exc_min=n_exc_min, n_tail=n_tail, alternative=alternative,
        allow_single_sample_group=allow_single_sample_group, keep_null=keep_null,
        gene_names=gene_names, backend=backend, subset=subset,
        jitter=np.zeros_like(x), tpm=x, pseudocount=pc,
        n_boot=n_boot, boot_rng=boot_rng, sample_names=sample_names,
        condition_meta=cond_meta,
    )


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

    result = wade(data.values[:, cols], sub_norm, cond, nperms=nperms,
                  gene_names=data.gene_names,
                  sample_names=data.sample_names[cols], **kwargs)
    result.params.update(n_case=int(case_idx.size),
                         n_ctrl=int(ctrl_idx.size), columns=cols)
    return result


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
