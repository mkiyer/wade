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
from .permutation import draw_perms, null_statistics, validate_perms
from .pvalues import ALTERNATIVES, DEFAULT_N_EXC_MIN, DEFAULT_N_TAIL, bh_adjust, perm_pvalues
from .stats import WadeStats, split_groups, wade_stats
from .subset import SubsetResult, subset_test

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

    @property
    def affected_fraction(self) -> np.ndarray | None:
        return None if self.subset is None else self.subset.affected_fraction

    @property
    def direction(self) -> np.ndarray | None:
        return None if self.subset is None else self.subset.direction

    @property
    def nprobs(self) -> int:
        """The quantile-grid resolution, ``min(n_case, n_ctrl)``.

        A property of the design, not a parameter: no argument raises it, and
        it is the resolution limit on the affected fraction — nothing finer
        than ``1/nprobs`` is estimable.
        """
        return self.stats.nprobs

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
        return cols

    def to_polars(self):
        """Optional convenience. polars is not a dependency of this package."""
        import polars as pl

        return pl.DataFrame(self.columns())


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


def _run(x, cond, *, nperms, perms, seed, perm_rng, n_exc_min, n_tail,
         alternative, allow_single_sample_group, keep_null, gene_names,
         backend, subset, jitter, tpm) -> WadeResult:
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
        sub = subset_test(x, cond, perms, alternative=alternative, backend=backend)
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

    return WadeResult(
        gene=_resolve_gene_names(gene_names, g),
        mean_shift=obs.mean_shift, w1=obs.w1, fc=obs.fc, log2_fc=obs.log2_fc,
        case_mean=obs.case_mean, ctrl_mean=obs.ctrl_mean,
        p_mean_shift=p_mean, padj_mean_shift=bh_adjust(p_mean),
        nexc_mean_shift=nexc, refined_mean_shift=refined,
        stats=obs, tpm=tpm, jitter=jitter, perms=perms,
        null_mean_shift=null if keep_null else None,
        subset=sub, p_subset=p_subset, padj_subset=padj_subset,
        params=dict(nperms=nperms, seed=seed, alternative=alternative,
                    n_exc_min=n_exc_min, n_tail=n_tail,
                    nprobs=obs.nprobs, n1=obs.n1, n0=obs.n0),
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
) -> WadeResult:
    """Run WADE on a raw count matrix. The primary entry point.

    Parameters
    ----------
    counts
        Genes x samples raw counts.
    normalizer
        A per-gene vector, or a full genes x samples matrix.
    cond
        Binary vector, 1 = case and 0 = control, one entry per column.
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
    """
    counts = np.asarray(counts, dtype=np.float64)
    cond = np.asarray(cond)
    if cond.shape[0] != counts.shape[1]:
        raise ValueError(
            f"cond has {cond.shape[0]} entries but counts has {counts.shape[1]} samples"
        )
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}; got {alternative!r}")
    split_groups(cond)

    if seed is None:
        jitter_rng = np.random.default_rng()
        perm_rng = np.random.default_rng()
    else:
        js, ps = np.random.SeedSequence(seed).spawn(2)
        jitter_rng = np.random.default_rng(js)
        perm_rng = np.random.default_rng(ps)

    if jitter is None:
        jitter = _normalize.draw_jitter(counts.shape, noise=noise, rng=jitter_rng)
    else:
        jitter = np.asarray(jitter, dtype=np.float64)

    tpm = _normalize.tpm_like(
        counts, normalizer, lib_sizes, noise=noise, norm_factor=norm_factor, jitter=jitter,
    )
    return _run(
        tpm, cond, nperms=nperms, perms=perms, seed=seed, perm_rng=perm_rng,
        n_exc_min=n_exc_min, n_tail=n_tail, alternative=alternative,
        allow_single_sample_group=allow_single_sample_group, keep_null=keep_null,
        gene_names=gene_names, backend=backend, subset=subset,
        jitter=jitter, tpm=tpm,
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
    """
    x = np.asarray(x, dtype=np.float64)
    cond = np.asarray(cond)
    if cond.shape[0] != x.shape[1]:
        raise ValueError(
            f"cond has {cond.shape[0]} entries but the matrix has {x.shape[1]} samples"
        )
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}; got {alternative!r}")
    split_groups(cond)
    perm_rng = (np.random.default_rng() if seed is None
                else np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1]))
    return _run(
        x, cond, nperms=nperms, perms=perms, seed=seed, perm_rng=perm_rng,
        n_exc_min=n_exc_min, n_tail=n_tail, alternative=alternative,
        allow_single_sample_group=allow_single_sample_group, keep_null=keep_null,
        gene_names=gene_names, backend=backend, subset=subset,
        jitter=np.zeros_like(x), tpm=x,
    )


def wade_contrast(
    counts: np.ndarray,
    normalizer: np.ndarray,
    case_samples,
    ctrl_samples,
    *,
    gene_names=None,
    nperms: int = DEFAULT_NPERMS,
    **kwargs,
) -> WadeResult:
    """Run one contrast from two lists of sample column indices.

    Sizes libraries **on the subset**, not on the full matrix, so a library's
    size factor depends only on the genes and samples handed in — change the
    gene set and every normalized value changes.
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
    normalizer = np.asarray(normalizer, dtype=np.float64)
    sub_norm = normalizer[:, cols] if normalizer.ndim == 2 else normalizer

    result = wade(counts[:, cols], sub_norm, cond, nperms=nperms,
                  gene_names=gene_names, **kwargs)
    result.params.update(n_case=int(case_samples.size),
                         n_ctrl=int(ctrl_samples.size), columns=cols)
    return result
