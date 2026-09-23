"""One distribution test per gene, followed by descriptive effects."""
from dataclasses import dataclass
import numpy as np

from ._descriptions import describe
from ._wasserstein import STOP_EXCEEDANCES, infer, log_expression, resolve_backend
from .io import Condition, Counts, as_counts, result_columns
from .normalize import Normalization, _fingerprint, _indices, _named, fit_normalization
from .permutation import permutation_space, strata_indices, validate_labels
from .pvalues import bh_adjust

DEFAULT_MAX_PERMUTATIONS = 499_999


@dataclass
class WadeResult:
    """A gene's q-value tests its distribution, not the sign of a description.

    bulk_shift is the case minus control median of log2(count/size_factor+1).
    upper_tail/lower_tail are the corresponding upper/lower decile mean
    differences minus bulk_shift. Both are signed descriptive contrasts.
    No tail p-values or sample outlier probabilities are inferred.
    """
    gene: np.ndarray
    statistic: np.ndarray
    pvalue: np.ndarray
    padj: np.ndarray
    status: np.ndarray
    bulk_shift: np.ndarray
    upper_tail: np.ndarray
    lower_tail: np.ndarray
    n_permutations: np.ndarray
    n_exceedances: np.ndarray
    expression: np.ndarray
    cond: np.ndarray
    sample_names: np.ndarray
    normalization: Normalization
    params: dict
    gene_meta: dict

    @property
    def tested(self):
        return self.status == "tested"

    def __repr__(self):
        return (f"WadeResult(genes={len(self.gene)}, tested={int(self.tested.sum())}, "
                f"case={int(self.cond.sum())}, control={int((self.cond == 0).sum())})")

    def columns(self):
        return result_columns(self)

    def to_frame(self):
        """Return a Polars table; the core API only requires NumPy."""
        try:
            import polars as pl
        except ImportError as e:
            raise ImportError("to_frame requires polars; install wade-rnaseq[io]") from e
        return pl.DataFrame(self.columns())

    def gene_index(self, gene):
        matches = np.flatnonzero(self.gene == gene)
        if len(matches):
            return int(matches[0])
        if isinstance(gene, (int, np.integer)) and not isinstance(gene, bool):
            if -len(self.gene) <= gene < len(self.gene):
                return int(gene % len(self.gene))
            raise IndexError("gene position out of range")
        raise KeyError(f"unknown gene {gene!r}")

    def gene_detail(self, gene):
        from .diagnostics import gene_detail
        return gene_detail(self.expression[self.gene_index(gene)], self.cond)

    def plot_gene(self, gene, **kwargs):
        from .plotting import plot_gene
        return plot_gene(self, gene, **kwargs)

    def plot_volcano(self, **kwargs):
        from .plotting import plot_volcano
        return plot_volcano(self, **kwargs)

    def plot_patterns(self, **kwargs):
        from .plotting import plot_patterns
        return plot_patterns(self, **kwargs)


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def wade(counts, cond, *, normalization="rle", test_genes=None,
         max_permutations=DEFAULT_MAX_PERMUTATIONS, seed=1, strata=None,
         gene_names=None, sample_names=None, gene_chunk=None, backend="auto"):
    """Test distribution differences on raw counts (genes x samples).

    The default fits RLE before excluding all-zero genes from testing. Pass
    test_genes as gene IDs or a pooled Boolean mask to define a smaller family
    without changing normalization. A fitted Normalization supports validated
    reference genes, external factors, and reuse across contrasts.

    Exact empirical W1 uses log2(count/size_factor+1). Independent random label
    shuffles preserve group sizes and optional strata. A gene stops at its
    20th inclusive exceedance or max_permutations; BH covers the entire tested
    family once. The minimum Monte Carlo p-value is 1/(max_permutations+1).
    Small budgets are useful for exploration but limit transcriptome discovery.

    Samples must be independent exchangeable units under the null, conditional
    on any strata. Normalization does not remove batch/depth confounding.
    gene_chunk and backend affect computation, not the statistical procedure.
    """
    maximum = _positive_int(max_permutations, "max_permutations")
    selected_backend = resolve_backend(backend)
    data = as_counts(counts, gene_names=gene_names, sample_names=sample_names)
    labels = cond.vector(data.sample_names) if isinstance(cond, Condition) else cond
    labels = validate_labels(labels, data.shape[1])
    if min(int(labels.sum()), int((labels == 0).sum())) < 2:
        raise ValueError("WADE requires at least two independent samples in each group")
    if strata is not None:
        strata = np.asarray(strata).copy()
        strata_indices(strata, data.shape[1])
    space = permutation_space(labels, strata)
    if space["log10_space"] == 0:
        raise ValueError("the specified strata permit no label exchanges")
    if isinstance(normalization, str):
        fitted = fit_normalization(data, method=normalization)
    elif isinstance(normalization, Normalization):
        fitted = normalization._align(data, named_samples=_named(counts, sample_names))
    else:
        raise TypeError("normalization must be 'rle', 'library', or a fitted Normalization")
    if not np.all(data.values.sum(axis=0) > 0):
        raise ValueError("remove empty samples before analysis")
    g, n = data.shape
    selected = np.ones(g, bool)
    if test_genes is not None:
        selected[:] = False
        selected[_indices(test_genes, data.gene_names, "test genes")] = True
    nonzero = np.any(data.values > 0, axis=1)
    tested = selected & nonzero
    status = np.where(~nonzero, "all_zero", np.where(selected, "tested", "filtered"))
    # Bounded sorting scratch and shuffle blocks; no genes x permutations bank.
    chunk = max(1, (128 * 2**20) // max(n * 8 * 8, 1)) if gene_chunk is None else _positive_int(gene_chunk, "gene_chunk")
    batch = max(1, min(4096, (32 * 2**20) // max(n, 1)))
    # Store a concrete seed even when the caller requests a fresh random run.
    if seed is None:
        seed = np.random.SeedSequence().entropy
    # Validate now, even when no gene will be tested. A sequence supports frozen protocols.
    np.random.default_rng(seed)
    if isinstance(seed, (np.random.Generator, np.random.BitGenerator, np.random.SeedSequence)):
        raise TypeError("seed must be an integer, an integer sequence, or None")
    seed = np.asarray(seed).tolist()
    expression = np.empty(data.shape, float)
    statistic, pvalue = np.full(g, np.nan), np.full(g, np.nan)
    exceed, tried = np.zeros(g, np.uint64), np.zeros(g, np.uint64)
    bulk, upper, lower = np.empty(g), np.empty(g), np.empty(g)
    for start in range(0, g, chunk):
        rows = slice(start, min(start + chunk, g))
        x = log_expression(data.values[rows], fitted.size_factors)
        expression[rows] = x
        bulk[rows], upper[rows], lower[rows] = describe(x, labels)
        local = np.flatnonzero(tested[rows])
        if len(local):
            idx = start + local
            statistic[idx], pvalue[idx], exceed[idx], tried[idx] = infer(
                x[local], labels, maximum=maximum, seed=seed, strata=strata,
                backend=selected_backend, batch=batch)
    params = dict(algorithm="log-w1-besag-clifford-v1", expression="log2(count / size_factor + 1)",
                  max_permutations=maximum, stop_exceedances=min(STOP_EXCEEDANCES, maximum),
                  minimum_pvalue=1 / (maximum + 1), seed=seed, strata=None if strata is None else strata.tolist(),
                  backend=selected_backend, gene_chunk=chunk, permutation_batch=batch,
                  counts_sha256=_fingerprint(data), permutation_space=space,
                  test_selection="nonzero" if test_genes is None else "explicit; all-zero genes excluded",
                  description="median shift; signed decile mean differences minus median shift")
    if isinstance(cond, Condition):
        params.update(case_label=cond.case, control_label=cond.control, condition_column=cond.column)
    return WadeResult(data.gene_names.copy(), statistic, pvalue, bh_adjust(pvalue), status,
                      bulk, upper, lower, tried, exceed, expression, labels.copy(),
                      data.sample_names.copy(), fitted, params, dict(data.meta))


def wade_contrast(counts, case_samples, control_samples, *, gene_names=None, sample_names=None, **kwargs):
    """Select two groups by sample ID; preserve a previously fitted normalization.

    A method name fits on the selected samples. A fitted object retains its
    original reference cohort and centering. strata, when supplied, must cover
    the entire input matrix and is selected alongside the samples.
    """
    data = as_counts(counts, gene_names=gene_names, sample_names=sample_names)
    i1 = _indices(case_samples, data.sample_names, "case samples")
    i0 = _indices(control_samples, data.sample_names, "control samples")
    if np.intersect1d(i1, i0).size:
        raise ValueError("samples appear in both groups")
    idx = np.r_[i1, i0]
    if kwargs.get("strata") is not None:
        strata = np.asarray(kwargs["strata"])
        if strata.shape != (data.shape[1],):
            raise ValueError("strata must cover the full input matrix")
        kwargs["strata"] = strata[idx]
    selected = Counts(data.values[:, idx], data.gene_names, data.sample_names[idx], data.meta)
    return wade(selected, np.r_[np.ones(len(i1), int), np.zeros(len(i0), int)], **kwargs)
