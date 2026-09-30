"""Each gene's mean shift, and the subset of cases that departs from a common fold, with a permutation test for each."""
from dataclasses import dataclass
import numpy as np

from math import ceil, isfinite

from ._means import DIRECTIONS, SCALES, expression, jittered, mean_saddlepoint, mean_test
from ._subsets import split, subset_test
from .io import Condition, Counts, as_counts, result_columns
from .normalize import Normalization, _fingerprint, _indices, _named, fit_normalization
from .permutation import permutation_space, validate_labels
from .pvalues import bh_adjust

# Floor 1e-6: an isolated BH discovery at q <= 0.05 stays possible among up to 50,000 genes.
DEFAULT_MAX_PERMUTATIONS = 999_999
PVALUES = ("auto", "permutation")
SUBSETS = ("cases", "either")


@dataclass(frozen=True)
class Subset:
    """The samples that depart from the fold every case would share; see WadeResult.subset()."""
    side: str              # "up" above the rest of their group, "down" below it, "none" if every case carries one fold
    group: str             # "case" or "control" (only when both groups are tested), "none" without a subset
    samples: np.ndarray    # the subset's sample names, furthest out first
    fraction: float        # the subset's share of its group
    log2fc: float          # the subset's fold over what the common fold predicts for it
    pvalue: float          # against the null that every case carries one common fold
    padj: float


@dataclass
class WadeResult:
    """Each gene's mean shift and subset, each with a permutation test.

    mean_log2fc is the signed area between the case and control quantile functions, which is
    the difference in means, as a fold: log2 of (mean case + 1) / (mean control + 1) on the
    linear scale, the difference in mean log2 expression on the log2 scale. mean_pvalue tests
    it against the declared direction, and mean_padj adjusts it by BH over the tested genes.
    alignment_log2fc is the fold that best aligns the cases' quantile function of log(count + 1)
    with the controls' (their Wasserstein-1 registration; on the log2 scale, a difference): the
    common fold the subset test holds every case to. Under an alternative it is an alignment, not
    necessarily any subgroup's fold. lorenz_gap is the subset test's statistic: the largest amount
    by which one group's Lorenz curve of count + 1 falls below the other's (on log2, of absolute
    Lorenz curves). The subset is the samples that depart from the common fold (see subset()):
    subset_group says whether they are cases or, when both groups are tested, controls;
    subset_side whether they are the upper ("up") or lower ("down") part of that group;
    subset_fraction their share of it; and subset_log2fc their fold over what the alignment
    predicts for their quantiles. The fold can point against the side in genes whose extreme
    values dominate both groups: read the side from subset_side. subset_pvalue tests the null
    that every case carries one common fold, against the declared direction, and subset_padj
    adjusts it by BH over the tested genes. Every fold on the linear scale is of count + 1, a
    pseudocount-regularized ratio. A gene-level p-value does not validate any one sample's
    membership: no sample outlier probabilities are inferred.
    """
    gene: np.ndarray
    status: np.ndarray
    mean_log2fc: np.ndarray
    mean_pvalue: np.ndarray
    mean_padj: np.ndarray
    alignment_log2fc: np.ndarray
    lorenz_gap: np.ndarray
    subset_group: np.ndarray
    subset_side: np.ndarray
    subset_fraction: np.ndarray
    subset_log2fc: np.ndarray
    subset_pvalue: np.ndarray
    subset_padj: np.ndarray
    mean_permutations: np.ndarray
    mean_exceedances: np.ndarray
    subset_permutations: np.ndarray
    subset_exceedances: np.ndarray
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
        """Exact quantile intervals on the test scale; their signed area is the difference in means."""
        from .diagnostics import gene_detail
        return gene_detail(self.expression[self.gene_index(gene)], self.cond)

    def subset(self, gene):
        """The samples that depart from the one fold every case would share.

        If every case carried the same fold (on the log2 scale, the same difference), the
        cases' Lorenz curve of count + 1 would match the controls'. Walking up the case
        quantiles, the running gap between them is lowest where the cases that carry less than
        their share of the cases' total give way to those that carry more. The smaller side of
        that split, which departs further per case, is the subset: above the rest of the cases
        or below them. When both groups are tested (subsets="either"), the controls are walked
        against the cases too, and the subset may be controls. Nothing is chosen in advance; a
        walk on noise also has a lowest point, so read the subset with its p-value.

        The side is the subset's part of its group, which the one-sided tests ask about. Its
        log2fc is measured against what the alignment predicts for its quantiles; in genes whose
        extreme values dominate both groups' means the two can point opposite ways.
        """
        i = self.gene_index(gene)
        x = self.expression[i]
        both = self.params["subsets"] == "either"
        (size,), (side,), (group,), (fold,), _, _ = split(x[None], self.cond, self.params["scale"], both_groups=both,
                                                          direction=self.params["direction"])
        members = self.cond == group
        values, names = x[members], self.sample_names[members]
        order = np.argsort(-values if side > 0 else values, kind="stable")[:size]
        part = "none" if size == 0 else "up" if side > 0 else "down"
        label = "none" if size == 0 else "case" if group == 1 else "control"
        return Subset(part, label, names[order], int(size) / len(values), float(fold),
                      float(self.subset_pvalue[i]), float(self.subset_padj[i]))

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


def wade(counts, cond, *, direction="both", scale="linear", subsets="cases", normalization="rle", test_genes=None,
         pvalues="auto", relative_error=0.2, max_permutations=DEFAULT_MAX_PERMUTATIONS, seed=1, strata=None,
         gene_names=None, sample_names=None, gene_chunk=None):
    """Test each gene's mean shift, and whether a subset of cases departs from a common fold (raw counts, genes x samples).

    The mean shift is the signed area between the case and control quantile functions: where
    cases are higher it adds, where they are lower it subtracts. It is exactly the difference
    in means on the tested scale. scale="linear" uses count/size_factor, which favours raised
    subsets; "log2" uses log2(count/size_factor+1), which treats a halving like a doubling. Each
    count first gets independent Uniform(0, 1) jitter, seeded like the shuffles, so that no two
    values tie: the subset test's null moves cases by a fold, which would split the ties of
    discrete counts (above all zeros) that the observed data keep.

    The subset is the cases that depart from the one fold every case would share, above the
    rest of the cases or below them (see WadeResult.subset()). Its test asks whether every case
    could carry one common fold: it registers the cases onto the controls by the fold that best
    aligns their quantile functions of log(count + 1) (the Wasserstein-1 fold), pools the two,
    and walks label shuffles of the pool. Its statistic, the largest gap between the cases' and
    the controls' Lorenz curves of count + 1 (on log2, of their absolute Lorenz curves), is
    unchanged by any fold on the cases. subsets="cases" asks for a subset among the cases: are
    the cases more unequal than the controls? subsets="either" also walks the controls against
    the cases and keeps the larger gap: do the two groups' shapes differ at all?

    Samples must be independent: strata, paired samples and repeated measures are not supported
    until a construction is validated for them (docs/subset-null-plan.md, section 4.9), and
    passing strata is an error.

    direction="both" tests both two-sided. "up" asks whether cases are higher, as a whole or in
    a subset: a mean shift up, and a subset of cases above the rest (with subsets="either", or
    a subset of controls below the rest). "down" asks the reverse.
    Choose the direction and the scale before looking at results.

    The default fits RLE before excluding all-zero genes from testing. Pass test_genes as gene
    IDs or a pooled Boolean mask to define a smaller family without changing normalization. A
    fitted Normalization supports validated reference genes, external factors, and reuse
    across contrasts.

    Every p-value is estimated to about relative_error. A shuffled p-value draws independent
    label shuffles that preserve group sizes, and stops at its
    ceil(1/relative_error^2)-th exceedance (the 25th at the default 0.2), whose count estimates p
    to that relative standard error, or at max_permutations shuffles, below which it cannot go.
    The subset test is always shuffled. With pvalues="auto" the mean test is computed instead:
    the exact permutation distribution of the cases' total has a double-saddlepoint tail with
    no floor above one labelling in C(n, n1). Its error is estimated as the larger of the
    saddlepoint's second-order term and one labelling among the L = p C(n, n1) in the tail. A
    gene whose estimate exceeds relative_error is shuffled.
    pvalues="permutation" shuffles every p-value. BH covers the tested genes. Samples must be independent exchangeable units under the null.
    Normalization does not remove batch or depth confounding; the
    run record reports median size factors per group. gene_chunk affects computation, not the
    statistical procedure.
    """
    if direction not in DIRECTIONS:
        raise ValueError("direction must be 'both', 'up' or 'down'")
    if scale not in SCALES:
        raise ValueError("scale must be 'linear' or 'log2'")
    if pvalues not in PVALUES:
        raise ValueError("pvalues must be 'auto' or 'permutation'")
    if subsets not in SUBSETS:
        raise ValueError("subsets must be 'cases' or 'either'")
    both = subsets == "either"
    if isinstance(relative_error, bool) or not isinstance(relative_error, (int, float, np.number)) \
            or not isfinite(relative_error) or not 0 < relative_error < 1:
        raise ValueError("relative_error must be a number between 0 and 1")
    stop = ceil(1 / relative_error ** 2)
    maximum = _positive_int(max_permutations, "max_permutations")
    data = as_counts(counts, gene_names=gene_names, sample_names=sample_names)
    labels = cond.vector(data.sample_names) if isinstance(cond, Condition) else cond
    labels = validate_labels(labels, data.shape[1])
    if min(int(labels.sum()), int((labels == 0).sum())) < 2:
        raise ValueError("WADE requires at least two independent samples in each group")
    if strata is not None:
        raise ValueError("strata are not supported: WADE tests independent samples until a stratified "
                         "construction is validated (docs/subset-null-plan.md, section 4.9)")
    space = permutation_space(labels)
    computed = pvalues == "auto"
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
    # Bounded scratch per chunk and per shuffle block; no genes x permutations bank.
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
    shift, pvalue, subset_p = np.empty(g), np.full(g, np.nan), np.full(g, np.nan)
    size, side, group = np.zeros(g, int), np.zeros(g, np.int8), np.ones(g, np.uint8)
    fold, alignment, gap = np.full(g, np.nan), np.full(g, np.nan), np.zeros(g)
    exceed, tried = np.zeros(g, np.uint64), np.zeros(g, np.uint64)
    subset_exceed, subset_tried = np.zeros(g, np.uint64), np.zeros(g, np.uint64)
    for start in range(0, g, chunk):
        rows = slice(start, min(start + chunk, g))
        x = expression(jittered(data.values[rows], range(rows.start, rows.stop), seed), fitted.size_factors, scale)
        values[rows] = x
        shift[rows] = _mean_log2fc(x, labels, scale)
        size[rows], side[rows], group[rows], fold[rows], gap[rows], registered = split(
            x, labels, scale, both_groups=both, direction=direction)
        alignment[rows] = np.log2(registered) if scale == "linear" else registered
        local = np.flatnonzero(tested[rows])
        if len(local):
            idx = start + local
            shuffled = np.ones(len(local), bool)
            if computed:
                pvalue[idx], error = mean_saddlepoint(x[local], labels, direction=direction)
                shuffled = error > relative_error                    # less precise than shuffling
            if shuffled.any():
                pvalue[idx[shuffled]], exceed[idx[shuffled]], tried[idx[shuffled]] = mean_test(
                    x[local[shuffled]], labels, direction=direction, stop=stop, maximum=maximum,
                    seed=seed, batch=batch)
            subset_p[idx], subset_exceed[idx], subset_tried[idx] = subset_test(
                x[local], labels, scale, direction=direction, stop=stop, maximum=maximum, seed=seed,
                both_groups=both, batch=batch)
    factors = fitted.size_factors
    params = dict(algorithm="signed-area-and-subset-v3", direction=direction, scale=scale, subsets=subsets,
                  subset_statistic="largest gap between the case and control Lorenz curves of count + 1"
                  if scale == "linear" else "largest gap between the case and control absolute Lorenz curves",
                  subset_registration="Wasserstein-1 fold on log(count + 1), the middle of any median interval"
                  if scale == "linear" else "Wasserstein-1 difference, the middle of any median interval",
                  expression="(count + jitter) / size_factor" if scale == "linear"
                  else "log2((count + jitter) / size_factor + 1)", jitter="Uniform(0, 1) per count, seeded by seed and gene row",
                  family="tested genes", hypotheses=int(tested.sum()), pvalues=pvalues,
                  mean_pvalue="saddlepoint unless its error estimate exceeds relative_error"
                  if computed else "shuffled", relative_error=relative_error, max_permutations=maximum,
                  stop_exceedances=min(stop, maximum), minimum_shuffled_pvalue=1 / (maximum + 1),
                  seed=seed,
                  gene_chunk=chunk, permutation_batch=batch,
                  counts_sha256=_fingerprint(data), permutation_space=space,
                  size_factor_median=dict(case=float(np.median(factors[labels == 1])),
                                          control=float(np.median(factors[labels == 0]))),
                  test_selection="nonzero" if test_genes is None else "explicit; all-zero genes excluded")
    if isinstance(cond, Condition):
        params.update(case_label=cond.case, control_label=cond.control, condition_column=cond.column)
    n_group = np.where(group == 1, int(labels.sum()), int((labels == 0).sum()))
    label = np.where(size == 0, "none", np.where(group == 1, "case", "control"))
    part = np.where(size == 0, "none", np.where(side > 0, "up", "down"))
    return WadeResult(data.gene_names.copy(), status, shift, pvalue, bh_adjust(pvalue), alignment, gap, label, part,
                      size / n_group, fold, subset_p, bh_adjust(subset_p),
                      tried, exceed, subset_tried, subset_exceed, values, labels.copy(),
                      data.sample_names.copy(), fitted, params, dict(data.meta))


def _mean_log2fc(x, cond, scale):
    """The mean shift as a fold: of the group means on count + 1, or of mean log2 expression."""
    mean1, mean0 = x[:, cond == 1].mean(axis=1), x[:, cond == 0].mean(axis=1)
    return np.log2((mean1 + 1) / (mean0 + 1)) if scale == "linear" else mean1 - mean0


def wade_contrast(counts, case_samples, control_samples, *, gene_names=None, sample_names=None, **kwargs):
    """Select two groups by sample ID; preserve a previously fitted normalization.

    A method name fits on the selected samples. A fitted object retains its
    original reference cohort and centering.
    """
    data = as_counts(counts, gene_names=gene_names, sample_names=sample_names)
    i1 = _indices(case_samples, data.sample_names, "case samples")
    i0 = _indices(control_samples, data.sample_names, "control samples")
    if np.intersect1d(i1, i0).size:
        raise ValueError("samples appear in both groups")
    idx = np.r_[i1, i0]
    selected = Counts(data.values[:, idx], data.gene_names, data.sample_names[idx], data.meta)
    return wade(selected, np.r_[np.ones(len(i1), int), np.zeros(len(i0), int)], **kwargs)
