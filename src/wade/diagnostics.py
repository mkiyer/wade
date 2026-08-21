"""Diagnostics: the single-gene curves, per-library QC, and subset attribution.

R's ``wade_gene()``. The downstream reading of this panel is what makes
the statistic legible — on the log-ratio curve ``r`` a global fold change
is flat, a rare high-expressing subset sits at zero and then climbs near
``p = 1``, and a single-outlier gene stays flat and then spikes at the
last node. The second and third are not distinguishable from this curve
alone; separating them is the permutation p-value's job.
:func:`wade.plot_gene` draws it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stats import wade_stats

__all__ = ["GeneDetail", "wade_gene", "library_qc", "subset_drivers"]


@dataclass(frozen=True)
class GeneDetail:
    """One gene's grid, in **ascending** probability order."""

    p: np.ndarray      # (m,) ascending, 0 -> 1
    y1: np.ndarray     # (m,) case quantiles, reordered to match p
    y0: np.ndarray     # (m,) control quantiles
    cum: np.ndarray    # (m,) cumulative signed area; cum[-1] == mean_shift
    r: np.ndarray      # (m,) the log-ratio curve, ascending
    nprobs: int


def wade_gene(
    row: np.ndarray,
    cond: np.ndarray,
    *,
    allow_single_sample_group: bool = False,
    pseudocount=None,
    max_probs: int | None = None,
) -> GeneDetail:
    """Per-gene detail from one **already-normalized** row.

    Takes normalized values, not counts, and not the matrix. ``pseudocount``
    (a scalar, or one value per sample) is added before the log-ratio curve
    ``r`` is taken, as the subset stage does (``docs/method.md`` §10.4); the
    quantile functions ``y1``/``y0`` and the cumulative area stay raw.
    ``max_probs`` caps the grid as the test does (``§1``); pass the run's
    value — :meth:`wade.WadeResult.gene_detail` does — so the curves here are
    the ones the statistics were read from.

    Why the reversal happens *after* the statistic, not before
    ----------------------------------------------------------
    Two reasons, and only the second is about floating point.

    The statistics are computed on the **descending** grid the test uses;
    the reversal here is for display only, so the curve reads left to right
    in increasing quantile. Doing it in this order is what keeps the plotted
    curve consistent with the numbers the test produced.

    Second, reversing changes the summation order. The curve is built so
    that ``cum[-1] == mean_shift``, and in exact arithmetic that is
    trivially true — but ``mean_shift`` sums ``D`` forwards while the
    endpoint accumulates it backwards. Measured in the R over 2,000 random
    rows, the two differ bitwise in about two thirds of genes, at a worst
    relative difference of 3.5e-14. **So the endpoint identity is a
    tolerance assertion, not an exact one.** It is still the cheapest
    correctness test the implementation has and it is asserted in the test
    suite.

    Note the division is by ``nprobs`` — the full grid size — at every
    point, not by the number of terms accumulated so far.
    """
    row = np.asarray(row, dtype=np.float64)
    if row.ndim != 1:
        raise ValueError(f"expected a single gene's row (1-D), got shape {row.shape}")

    st = wade_stats(
        row[None, :], cond, allow_single_sample_group=allow_single_sample_group,
        max_probs=max_probs,
    )
    y1 = st.Q1[0][::-1]
    y0 = st.Q0[0][::-1]
    if pseudocount is None or np.all(np.asarray(pseudocount) == 0):
        ry1, ry0 = y1, y0
    else:
        pc = np.broadcast_to(np.asarray(pseudocount, dtype=np.float64), row.shape)
        stp = wade_stats((row + pc)[None, :], cond,
                         allow_single_sample_group=allow_single_sample_group,
                         max_probs=max_probs)
        ry1, ry0 = stp.Q1[0][::-1], stp.Q0[0][::-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log2(ry1) - np.log2(ry0)
    return GeneDetail(
        p=st.q[::-1].copy(),
        y1=y1.copy(),
        y0=y0.copy(),
        cum=np.cumsum(y1 - y0) / st.nprobs,
        r=r,
        nprobs=st.nprobs,
    )


# ---------------------------------------------------------------------------
# Per-library QC, and who drives a subset


def library_qc(counts: np.ndarray, *, top_n: int = 10, normalizer=None) -> dict:
    """Per-library depth, complexity and concentration.

    Three numbers per sample, and they have the same status as the
    combinatorial floor in ``docs/limits.md``: properties of the **data** to
    look at *before* trusting a run, not knobs.

    ``depth``
        Column total. Library size on the count scale.
    ``n_detected`` / ``detected_fraction``
        How many genes are non-zero. **Complexity.**
    ``top_share``
        The fraction of the library held by its ``top_n`` largest genes.
        **Concentration.**

    Why they matter to *this* test: a library that detects half as many genes
    as its peers, and holds a third of its mass in ten of them, is an extreme
    outlier in every gene it does detect — and a distributional test will
    faithfully report a subset in each of those genes. Measured on real plasma
    cfRNA, a handful of such libraries topped the subset ranking of thousands
    of genes at once and produced a thoroughly convincing false story
    (``notebooks/rna100k.qmd``, "Who drives a subset?"). Complexity was what
    separated them from a genuine finding; cross-gene recurrence was not.

    ``normalizer`` (a per-gene vector or genes x samples matrix) computes the
    shares on the rate scale ``counts / normalizer`` instead of on raw counts,
    matching what the statistic actually sees.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError(f"expected a 2-D genes x samples array, got shape {counts.shape}")
    g, n = counts.shape
    if normalizer is None:
        rate = counts
    else:
        from .normalize import _broadcast_normalizer
        rate = counts / _broadcast_normalizer(normalizer, counts.shape)
    total = rate.sum(axis=0)
    k = min(max(int(top_n), 1), g)
    # Selection, not a full sort: only the top k matter.
    top = np.partition(rate, g - k, axis=0)[g - k:].sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        top_share = np.where(total > 0, top / total, np.nan)
    detected = (counts > 0).sum(axis=0)
    return {
        "depth": counts.sum(axis=0),
        "n_detected": detected,
        "detected_fraction": detected / g,
        "top_share": top_share,
    }


def subset_drivers(res, gene, *, k: int | None = None) -> dict:
    """Which samples drive one gene's subset call.

    The affected region is the gene's own ``affected_fraction`` — the top
    ``ceil(affected_fraction * nprobs)`` case samples by normalized value, or
    the bottom ones when ``direction`` is negative — so no threshold is
    introduced here either. ``k`` overrides that width for exploration.

    Returns ``columns`` (positions in ``res.tpm``), ``samples`` (their names),
    ``values`` (their normalized values, descending in magnitude for an upward
    subset), and ``share`` — each driver's value as a fraction of that
    library's total. **``share`` is the discriminating number**: a gene taking
    12% of a driver's library is telling you about the library, while one
    taking a fraction of a percent is telling you about the gene.

    Pair it with :func:`library_qc` on the same columns: drivers that are
    ordinary libraries which happen to share a diagnosis are the finding;
    drivers that are low-complexity outliers are the artefact.
    """
    if res.subset is None:
        raise ValueError("this result has no subset stage, so there are no drivers "
                         "(nperms=0, subset=False, or min(n_case, n_ctrl) < 3)")
    if res.cond is None:
        raise ValueError("this result does not carry its condition vector")
    i = res.gene_index(gene)
    cond = np.asarray(res.cond)
    case_cols = np.flatnonzero(cond == 1)
    row = np.asarray(res.tpm)[i, case_cols]
    frac = res.subset.affected_fraction[i]
    if k is None:
        frac = 1.0 if not np.isfinite(frac) else float(frac)
        k = int(np.clip(np.ceil(frac * res.nprobs), 1, case_cols.size))
    else:
        k = int(np.clip(k, 1, case_cols.size))
    down = np.nan_to_num(res.subset.direction[i]) < 0
    order = np.argsort(row if down else -row)[:k]
    cols = case_cols[order]
    lib_total = np.asarray(res.tpm)[:, cols].sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(lib_total > 0, row[order] / lib_total, np.nan)
    names = (None if res.sample_names is None
             else np.asarray(res.sample_names)[cols])
    return {"gene": res.gene[i], "columns": cols, "samples": names,
            "values": row[order], "share": share,
            "direction": float(np.nan_to_num(res.subset.direction[i])),
            "affected_fraction": float(res.subset.affected_fraction[i])}
