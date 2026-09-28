"""Upward and downward area tests per gene, followed by descriptive effects."""
from dataclasses import dataclass
import numpy as np

from ._descriptions import describe, subsets
from ._wasserstein import DIRECTIONS, SCALES, STOP_EXCEEDANCES, expression, infer, resolve_backend
from .io import Condition, Counts, as_counts, result_columns
from .normalize import Normalization, _fingerprint, _indices, _named, fit_normalization
from .permutation import permutation_space, strata_indices, validate_labels
from .pvalues import bh_adjust

# Floor 1e-6: an isolated BH discovery at q <= 0.05 stays possible for up to
# 50,000 gene-direction tests, i.e. a 25,000-gene direction="both" analysis.
DEFAULT_MAX_PERMUTATIONS = 999_999
_ACTIVE = {"both": (True, True), "up": (True, False), "down": (False, True)}


@dataclass(frozen=True)
class Subset:
    """The cases one gene's walk places in its upward or downward subset."""
    direction: str
    samples: np.ndarray    # case sample names, most extreme first
    boundary: float        # expression at the subset's inner edge (params["scale"]); NaN if empty
    fraction: float        # share of cases in the subset
    log2fc: float          # subset median minus control median, log2(count/size_factor+1)


@dataclass
class WadeResult:
    """Each gene's upward and downward areas, with the declared hypothesis's q-values.

    area_up integrates case-minus-control quantile differences where cases are
    higher; area_down where they are lower; both on params["scale"]. padj_up and
    padj_down share one BH family for direction="both", or cover the tested
    genes for one direction (the other is NaN). bulk_shift is the case minus
    control median of log2(count/size_factor+1). The subset columns describe
    the enriched subsets each gene's walk learns (see subset()): the share of
    cases in them and their median minus the control median, in log2 units.
    Descriptions carry no p-values, and no sample outlier probabilities are inferred.
    """
    gene: np.ndarray
    status: np.ndarray
    area_up: np.ndarray
    area_down: np.ndarray
    pvalue_up: np.ndarray
    pvalue_down: np.ndarray
    padj_up: np.ndarray
    padj_down: np.ndarray
    bulk_shift: np.ndarray
    subset_up_fraction: np.ndarray
    subset_up_log2fc: np.ndarray
    subset_down_fraction: np.ndarray
    subset_down_log2fc: np.ndarray
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
                f"case={int(self.cond.sum())}, control={int((self.cond == 0).sum())}, "
                f"direction={self.params['direction']!r}, scale={self.params['scale']!r})")

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
        """Exact quantile intervals on the test scale; their areas are area_up/area_down."""
        from .diagnostics import gene_detail
        return gene_detail(self.expression[self.gene_index(gene)], self.cond)

    def subset(self, gene, direction="up"):
        """The cases the gene's walk places in its upward or downward subset.

        The walk steps down the pooled ranking of samples. At each cut it takes
        the fraction of cases above minus the fraction of controls above, divided
        by that difference's standard deviation under random labelling. The
        upward subset is the cases at or above the cut where this peaks; the
        downward subset, the cases at or below its lowest point. A subset whose
        median is not on its own side of the control median is empty. Nothing
        is chosen in advance, and the walk depends only on ranks, so the scale
        does not change it. It describes where cases concentrate; read it with
        the direction's q-value, since a walk on noise also has a peak.
        """
        if direction not in DIRECTIONS:
            raise ValueError("direction must be 'up' or 'down'")
        i = self.gene_index(gene)
        x = self.expression[i]
        # Walk on the same log2 values as the table, so both give the same subset on either scale.
        walked = x if self.params["scale"] == "log" else np.log2(x + 1)
        up_cut, up_cases, _, down_cut, down_cases, _, values = subsets(walked[None], self.cond)
        fraction = float(getattr(self, f"subset_{direction}_fraction")[i])
        log2fc = float(getattr(self, f"subset_{direction}_log2fc")[i])
        if (up_cases if direction == "up" else down_cases)[0] == 0:
            return Subset(direction, self.sample_names[:0], float("nan"), fraction, log2fc)
        case = self.cond == 1
        if direction == "up":
            members = case & (walked >= values[0, up_cut[0]])
            order, boundary = np.argsort(-x[members], kind="stable"), x[members].min()
        else:
            members = case & (walked <= values[0, down_cut[0] - 1])
            order, boundary = np.argsort(x[members], kind="stable"), x[members].max()
        return Subset(direction, self.sample_names[members][order], float(boundary), fraction, log2fc)

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


def wade(counts, cond, *, direction="both", scale="linear", normalization="rle", test_genes=None,
         max_permutations=DEFAULT_MAX_PERMUTATIONS, seed=1, strata=None,
         gene_names=None, sample_names=None, gene_chunk=None, backend="auto"):
    """Test upward and downward expression differences on raw counts (genes x samples).

    area_up integrates the case-minus-control quantile differences where cases
    are higher. Quantiles where cases are lower add nothing to it instead of
    cancelling it, so a subset raised above a lowered bulk keeps its evidence;
    area_down is the mirror image, and area_up - area_down is the mean difference.

    direction="both" tests both areas in one pass with shared label draws: a
    discovery hypothesis with one BH family over every tested gene-direction
    pair. direction="up" or "down" tests one declared direction, and BH covers
    the tested genes. scale="linear" uses count/size_factor; "log" uses
    log2(count/size_factor+1). Choose both before looking at results.

    The default fits RLE before excluding all-zero genes from testing. Pass
    test_genes as gene IDs or a pooled Boolean mask to define a smaller family
    without changing normalization. A fitted Normalization supports validated
    reference genes, external factors, and reuse across contrasts.

    Independent random label shuffles preserve group sizes and optional strata.
    Each direction stops at its 20th inclusive exceedance or max_permutations.
    The minimum Monte Carlo p-value is 1/(max_permutations+1). Small budgets
    are useful for exploration but limit transcriptome discovery.

    Samples must be independent exchangeable units under the null, conditional
    on any strata. Normalization does not remove batch/depth confounding; the
    run record reports median size factors per group. gene_chunk and backend
    affect computation, not the statistical procedure.
    """
    if direction not in _ACTIVE:
        raise ValueError("direction must be 'both', 'up' or 'down'")
    if scale not in SCALES:
        raise ValueError("scale must be 'linear' or 'log'")
    active = _ACTIVE[direction]
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
    values = np.empty(data.shape, float)
    area, pvalue = np.full((g, 2), np.nan), np.full((g, 2), np.nan)
    exceed, tried = np.zeros((g, 2), np.uint64), np.zeros((g, 2), np.uint64)
    described = np.empty((5, g))
    for start in range(0, g, chunk):
        rows = slice(start, min(start + chunk, g))
        x = expression(data.values[rows], fitted.size_factors, scale)
        values[rows] = x
        described[:, rows] = describe(x if scale == "log" else np.log2(x + 1), labels)
        local = np.flatnonzero(tested[rows])
        if len(local):
            idx = start + local
            area[idx], pvalue[idx], exceed[idx], tried[idx] = infer(
                x[local], labels, active=active, maximum=maximum, seed=seed, strata=strata,
                backend=selected_backend, batch=batch)
    padj = np.full((g, 2), np.nan)
    columns = [d for d, on in enumerate(active) if on]
    padj[:, columns] = bh_adjust(pvalue[:, columns].ravel()).reshape(g, len(columns))
    factors = fitted.size_factors
    params = dict(algorithm="directional-area-besag-clifford-v1", direction=direction, scale=scale,
                  expression="count / size_factor" if scale == "linear" else "log2(count / size_factor + 1)",
                  family="every tested gene-direction pair" if direction == "both" else f"tested genes, {direction} only",
                  hypotheses=int(len(columns) * tested.sum()),
                  max_permutations=maximum, stop_exceedances=min(STOP_EXCEEDANCES, maximum),
                  minimum_pvalue=1 / (maximum + 1), seed=seed, strata=None if strata is None else strata.tolist(),
                  backend=selected_backend, gene_chunk=chunk, permutation_batch=batch,
                  counts_sha256=_fingerprint(data), permutation_space=space,
                  size_factor_median=dict(case=float(np.median(factors[labels == 1])),
                                          control=float(np.median(factors[labels == 0]))),
                  test_selection="nonzero" if test_genes is None else "explicit; all-zero genes excluded",
                  description="log2 median shift; standardized running-sum subsets up and down")
    if isinstance(cond, Condition):
        params.update(case_label=cond.case, control_label=cond.control, condition_column=cond.column)
    return WadeResult(data.gene_names.copy(), status, area[:, 0], area[:, 1], pvalue[:, 0], pvalue[:, 1],
                      padj[:, 0], padj[:, 1], *described, tried, exceed, values, labels.copy(),
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
