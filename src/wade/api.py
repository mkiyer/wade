"""Each gene's area between cases and controls, and the subset of cases that departs from a common fold, with a permutation test for each."""
import warnings
from dataclasses import dataclass
import numpy as np

from math import ceil, isfinite

from ._means import DIRECTIONS, advance_means, area_values, expression, jittered, mean_saddlepoint
from ._rarity import rare_subsets, rarity, sample_load
from ._means import floors
from ._subsets import DRAWS, advance, aligned, split, thinned, walks
from .io import Condition, Counts, as_counts, result_columns
from .normalize import Normalization, _fingerprint, _indices, _named, fit_normalization
from .permutation import permutation_space, strata_codes, stratum_columns, validate_labels
from .permutation import STREAM_BLOCK
from .pvalues import TIE_RTOL, bh_adjust, settle

# Floor 1e-6: an isolated BH discovery at q <= 0.05 stays possible among up to 50,000 genes.
DEFAULT_MAX_PERMUTATIONS = 999_999
DEFAULT_FDR = 0.05
PVALUES = ("auto", "permutation")
# The subset test is calibrated where the lower group averages at least this many reads per sample
# (docs/validation.md; measured by research/subset_range.py): a measured bound on where the test is
# valid, not a setting of the method.
SUBSET_VALIDATED_READS = 32


@dataclass(frozen=True)
class Subset:
    """The cases that depart from the fold every case would share; see WadeResult.subset()."""
    side: str              # "up" above the rest of the cases, "down" below them, "none" if every case carries one fold
    samples: np.ndarray    # the subset's sample names, furthest out first
    fraction: float        # the subset's share of the cases
    log2fc: float          # the subset's fold over what the common fold predicts for it
    pvalue: float          # against the null that every case carries one common fold
    padj: float


@dataclass
class WadeResult:
    """Each gene's area and subset, each with a permutation test.

    case_mean and control_mean are the groups' mean normalized counts (count / size factor: reads
    per sample at the cohort's typical depth); with strata, the Mantel-Haenszel standardized means
    over the strata holding both groups, sum_s w_s mean_s / sum_s w_s with w_s = n1s n0s / ns. area
    is their difference, which without strata is exactly the signed area between the case and
    control quantile functions, and with them the strata's areas weighted by w_s. area_fraction is that area over
    the total area under both curves, (case_mean - control_mean) / (case_mean + control_mean):
    between -1 and 1. log2fc is the fold of the group means, log2(case_mean / control_mean), with
    no pseudocount: +inf where the controls have no reads, -inf where the cases have none. The
    two carry the same information, area_fraction = (fc - 1) / (fc + 1). area_pvalue tests the
    area against the declared direction on counts thinned until every sample's size factor
    equals the smallest (see wade()), and area_padj adjusts it by BH over the tested genes.

    alignment_log2fc is the fold that best aligns the cases' quantile function of log(count + 1)
    with the controls' (their Wasserstein-1 registration): the common fold every case is held to
    (with strata, the weighted median of the strata's own). Under an alternative it is an
    alignment, not necessarily any subgroup's fold. lorenz_gap is the subset statistic read on
    normalized counts: the largest amount by which the cases' Lorenz curve of count + 1 falls below
    the controls'; the test computes it on counts made exchangeable by thinning, averaged over four
    draws (see wade()). The
    subset is the cases that depart from the common fold (see subset()): subset_side says whether
    they are the upper ("up") or lower ("down") part of the cases, subset_fraction their share of
    the cases, and subset_log2fc their fold over what the alignment predicts for their quantiles
    (of count + 1). The fold can point against the side in genes whose extreme values dominate both
    groups: read the side from subset_side. subset_pvalue tests the null that every case carries
    one common fold, against the declared direction, and subset_padj adjusts it by BH over the
    tested genes; a gene without a subset has p = 1. It is validated where the lower group mean is
    at least 32 reads per sample. A gene-level p-value does not validate any one sample's
    membership: no sample outlier probabilities are inferred.

    subset_rarity is the share of genes whose subset is at least as rare: among the genes of its
    group (those in the subset test's validated range, or those below it), the share whose gap at
    this gene's cut, a subset of the same size and side, is at least this gene's gap, ranked across
    the group. It is 1 for a gene without a subset. It describes and ranks, and is not a test: never
    select genes by it before BH. sample_load() and rare_subsets() show which samples and genes
    carry the rarest subsets.
    """
    gene: np.ndarray
    status: np.ndarray
    case_mean: np.ndarray
    control_mean: np.ndarray
    area: np.ndarray
    area_fraction: np.ndarray
    log2fc: np.ndarray
    area_pvalue: np.ndarray
    area_padj: np.ndarray
    alignment_log2fc: np.ndarray
    lorenz_gap: np.ndarray
    subset_side: np.ndarray
    subset_fraction: np.ndarray
    subset_log2fc: np.ndarray
    subset_pvalue: np.ndarray
    subset_padj: np.ndarray
    subset_rarity: np.ndarray
    area_permutations: np.ndarray
    area_exceedances: np.ndarray
    subset_permutations: np.ndarray
    subset_exceedances: np.ndarray
    expression: np.ndarray
    cond: np.ndarray
    sample_names: np.ndarray
    normalization: Normalization
    params: dict
    gene_meta: dict
    strata: np.ndarray = None      # each sample's stratum (0 to S - 1), -1 if left out; None without strata

    @property
    def tested(self):
        return self.status == "tested"

    @property
    def bounded(self):
        """Which shuffled p-values are bounds rather than finished Besag–Clifford values, per test.

        With fdr set, a gene's shuffles stop once BH at that level must reject it; its p-value is
        then an upper bound, at most the BH threshold, and its q-value at most fdr. Every other
        gene's p-value is the one its finished run gives. Returns {"area": mask, "subset": mask}.
        """
        stop, maximum = self.params["stop_exceedances"], self.params["max_permutations"]
        cut = lambda hits, draws: (draws > 0) & (hits < stop) & (draws < maximum)
        return dict(area=cut(self.area_exceedances, self.area_permutations),
                    subset=cut(self.subset_exceedances, self.subset_permutations))

    def __repr__(self):
        return (f"WadeResult(genes={len(self.gene)}, tested={int(self.tested.sum())}, "
                f"case={int(self.cond.sum())}, control={int((self.cond == 0).sum())}, "
                f"direction={self.params['direction']!r})")

    def columns(self):
        return result_columns(self)

    def to_frame(self):
        """Return a Polars table; the core API only requires NumPy."""
        try:
            import polars as pl
        except ImportError as e:
            raise ImportError("to_frame requires polars; install wade-rnaseq[io]") from e
        return pl.DataFrame(self.columns())

    def _walked(self, genes):
        """The values the subset walk reads for genes (rows), with the labels and names of the samples it
        reads: the expression, or with strata the aligned pool of the samples in strata with both groups."""
        genes = np.atleast_1d(np.asarray(genes, dtype=np.int64))
        cond, names = self._walked_samples()
        if self.strata is None:
            return self.expression[genes], cond, names
        keep = self.strata >= 0
        if self.strata.max() == 0:                                   # one stratum with both groups
            return self.expression[genes][:, keep], cond, names
        return aligned(self.expression[genes][:, keep], cond, self.strata[keep])[0], cond, names

    def _walked_samples(self):
        """The labels and names of the samples the subset walk reads: with strata, those in strata with both groups."""
        if self.strata is None:
            return self.cond, self.sample_names
        keep = self.strata >= 0
        return self.cond[keep], self.sample_names[keep]

    def _subsets(self):
        """Each gene's subset size and side (1 above, -1 below, 0 none), and its rarity group
        (0 the validated range, 1 below it, -1 not tested)."""
        n1 = int(self.cond.sum() if self.strata is None else self.cond[self.strata >= 0].sum())
        size = np.rint(np.nan_to_num(self.subset_fraction) * n1).astype(np.int64)
        side = np.where(self.subset_side == "up", 1, np.where(self.subset_side == "down", -1, 0))
        lower = np.minimum(self.case_mean, self.control_mean)
        groups = np.where(self.tested, np.where(lower >= self.params["subset_validated_reads"], 0, 1), -1)
        return size, side, groups

    def sample_load(self, drivers=5):
        """Which cases carry the rarest subsets.

        Every gene with a subset spreads its weight, -log(subset_rarity), over the cases in its
        subset, a singleton giving all of it to one case. A case's load is the weight it receives over
        what random membership would give it, so the loads average exactly 1. The heaviest libraries
        carry another tissue's programme or an artefact; a light contamination may not stand out, and
        a subtype carries many co-regulated genes too, so read the drivers before acting. Returns
        {"sample", "load", "subsets", "drivers"}: each case, its load, the number of subsets it
        belongs to, and its top driving genes, most weight first.
        """
        size, side, _ = self._subsets()
        cond, names = self._walked_samples()
        load, joined, top = sample_load(lambda genes: self._walked(genes)[0], cond, size, side, self.subset_rarity,
                                        self.gene, drivers=drivers)
        return dict(sample=names[cond == 1], load=load, subsets=joined, drivers=top)

    def rare_subsets(self, n=25, validated=True):
        """The n rarest subsets, with their members and the listed genes they share members with.

        validated=True lists genes in the subset test's validated range, False those below it. Genes
        whose members coincide are one programme: another tissue, a subtype or an artefact. A driver
        can share members with genes it regulates, so read how many are shared. Returns {"gene",
        "rarity", "side", "size", "members", "shared"}, shared listing (gene, members in common) for
        each.
        """
        size, side, groups = self._subsets()
        cond, names = self._walked_samples()
        return rare_subsets(lambda genes: self._walked(genes)[0], cond, size, side, self.subset_rarity, groups,
                            0 if validated else 1, self.gene, names, n=n)

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
        """Exact quantile intervals of normalized expression; their signed area is the difference in means."""
        from .diagnostics import gene_detail
        return gene_detail(self.expression[self.gene_index(gene)], self.cond)

    def subset(self, gene):
        """The cases that depart from the one fold every case would share.

        If every case carried the same fold on count + 1, the cases' Lorenz curve of count + 1
        would match the controls'. Walking up the case quantiles, the running gap between them is
        lowest where the cases that carry less than their share of the cases' total give way to
        those that carry more. The smaller side of that split, which departs further per case, is
        the subset: above the rest of the cases or below them. Nothing is chosen in advance; a
        walk on noise also has a lowest point, so read the subset with its p-value.

        The side is the subset's part of the cases, which the one-sided tests ask about. Its
        log2fc is measured against what the alignment predicts for its quantiles; in genes whose
        extreme values dominate both groups' means the two can point opposite ways.
        """
        i = self.gene_index(gene)
        walked, cond, sample_names = self._walked([i])
        (size,), (side,), (fold,), _, _ = split(walked, cond)
        cases = cond == 1
        values, names = walked[0][cases], sample_names[cases]
        order = np.argsort(-values if side > 0 else values, kind="stable")[:size]
        part = "none" if size == 0 else "up" if side > 0 else "down"
        return Subset(part, names[order], int(size) / len(values), float(fold),
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


def wade(counts, cond, *, direction="both", normalization="rle", test_genes=None, pvalues="auto",
         fdr=DEFAULT_FDR, relative_error=0.2, max_permutations=DEFAULT_MAX_PERMUTATIONS, seed=1, strata=None,
         gene_names=None, sample_names=None, gene_chunk=None):
    """Test each gene's area between cases and controls, and whether a subset of cases departs from a common fold (raw counts, genes x samples).

    Every normalization reduces to one size factor per sample: its depth as the normalizer
    measures it, from RLE on all genes (the default), RLE on internal control genes
    (fit_normalization(..., reference_genes=...)), or factors fitted elsewhere
    (Normalization.from_size_factors). Raw library totals are used only with
    normalization="library".

    The area is the signed area between the case and control quantile functions of normalized
    counts (count / size factor): where cases are higher it adds, where they are lower it
    subtracts. It is exactly the difference in the group means; over the total area under both
    curves it is the area fraction, and their ratio is the fold, log2fc. Its test runs on counts
    thinned binomially until every sample's size factor equals the smallest: sample j keeps each
    read with chance min(s) / s_j. Read counts carry Poisson noise that shrinks with depth, so
    samples at different size factors are not exchangeable at low counts even when nothing
    differs between the groups; after thinning they are. Genes with too few reads to trust are
    washed out rather than called, and well-expressed genes barely change. Every sample is
    brought down to the smallest size factor, so remove failed or unusually shallow libraries
    first; params["depth_equalized"] records the share of its reads the median sample kept.

    The subset is the cases that depart from the one fold every case would share, above the rest
    of the cases or below them (see WadeResult.subset()), read on normalized counts. Its statistic,
    the largest gap between the cases' and the controls' Lorenz curves of count + 1, asks: are the
    cases more unequal than the controls? Its test asks whether every case could carry one common
    fold. It runs on counts made exchangeable by thinning, which keeps the count model exact:
    every sample is thinned to the smallest size factor, as for the area test, and the fold that
    best aligns the two groups' quantile functions of log(count + 1) (the Wasserstein-1 fold of
    those counts) is taken out of the higher group by thinning it further, not by rescaling it.
    This is done in four independent draws, and labels are shuffled over the four pools at once,
    each shuffle scored by the mean of their walks, so that one draw's chance thinning does not
    decide a call. Each count first gets independent Uniform(0, 1)
    jitter, seeded like the shuffles, so that no two values tie. The subset test is validated where
    the lower group averages at least 32 reads per sample; below that it can be anti-conservative
    when every case carries a fold, and wade() warns how many tested genes fall there.

    strata, if given, are one label per sample (a study, a site, a batch), or a list of such
    factors whose combinations are the strata; a Condition built with condition(..., strata=...)
    carries them by sample name. Labels are then exchanged only within strata, and both tests are
    exact under the null that labels are exchangeable within each stratum, whatever differs
    between strata. The area test thins each stratum to its own smallest size factor and tests the
    cases' total centred within strata (the Cochran-Mantel-Haenszel sum), its p-value by a
    stratified saddlepoint or by shuffles within strata; the effects are Mantel-Haenszel
    standardized means. The subset test thins each stratum the same way, takes one fold out of
    every case (the weighted median of the strata's own folds), and moves each stratum to the
    cohort's level. A stratum holding only cases or only controls cannot move a label: its
    samples are left out of both tests, with a warning, and an error is raised if no stratum holds
    both groups. params["strata"] records the design. Strata cannot separate a study effect from
    a condition that coincides with it. Paired samples and repeated measures are not yet
    validated.

    direction="both" tests both two-sided. "up" asks whether cases are higher: a larger area, and
    a subset of cases above the rest. "down" asks the reverse. Choose the direction before
    looking at results.

    The default fits RLE before excluding all-zero genes from testing. Pass test_genes as gene
    IDs or a pooled Boolean mask to define a smaller family without changing normalization. A
    fitted Normalization supports validated reference genes, external factors, and reuse across
    contrasts.

    A shuffled p-value draws label shuffles that preserve group sizes, from one seeded stream that
    every gene sees, and follows Besag and Clifford's rule: it stops at its
    ceil(1/relative_error^2)-th exceedance (the 25th at the default 0.2), whose count estimates p
    to that relative standard error, or at max_permutations shuffles, below which it cannot go.
    The subset test is always shuffled. With pvalues="auto" the area test is computed instead: the
    exact permutation distribution of the cases' total has a double-saddlepoint tail with no floor
    above one labelling in C(n, n1). Its error is estimated as the larger of the saddlepoint's
    second-order term and one labelling among the L = p C(n, n1) in the tail. A gene whose
    estimate exceeds relative_error is shuffled. pvalues="permutation" shuffles every p-value.
    Each test has one BH family over the tested genes.

    fdr is the false discovery rate at which BH decides, chosen before looking: shuffles run only
    as far as each gene's decision needs. A gene's Besag–Clifford run has a bound that holds
    whatever its remaining shuffles would show; once that bound is below BH's threshold, the gene
    stops. BH then rejects exactly the genes it would reject had every run finished, on every run,
    and the false discovery rate is the same; the shuffles fall from up to max_permutations per
    gene to about stop / threshold (Fischer, Barry and Ramdas, Electron. J. Statist. 2026). The
    genes BH rejects report their bound as the p-value (at most the threshold; result.bounded marks
    them) and a q-value of at most fdr; every other gene's p- and q-value is exactly the finished
    one. BH at any level of fdr or above gives the same decisions; a stricter level needs a new
    run. fdr=None finishes every run. Normalization does not remove batch confounding; the run
    record reports median size factors per group. gene_chunk affects computation, not the
    statistical procedure.
    """
    if direction not in DIRECTIONS:
        raise ValueError("direction must be 'both', 'up' or 'down'")
    if pvalues not in PVALUES:
        raise ValueError("pvalues must be 'auto' or 'permutation'")
    if isinstance(relative_error, bool) or not isinstance(relative_error, (int, float, np.number)) \
            or not isfinite(relative_error) or not 0 < relative_error < 1:
        raise ValueError("relative_error must be a number between 0 and 1")
    if fdr is not None and (isinstance(fdr, bool) or not isinstance(fdr, (int, float, np.number))
                            or not isfinite(fdr) or not 0 < fdr < 1):
        raise ValueError("fdr must be a number between 0 and 1, or None")
    maximum = _positive_int(max_permutations, "max_permutations")
    stop = min(ceil(1 / relative_error ** 2), maximum)
    data = as_counts(counts, gene_names=gene_names, sample_names=sample_names)
    labels = cond.vector(data.sample_names) if isinstance(cond, Condition) else cond
    labels = validate_labels(labels, data.shape[1])
    if min(int(labels.sum()), int((labels == 0).sum())) < 2:
        raise ValueError("WADE requires at least two independent samples in each group")
    if isinstance(cond, Condition) and cond.strata is not None:
        if strata is not None:
            raise ValueError("strata are given twice: in the condition and as strata=")
        strata = [list(column) for column in zip(*(cond.strata[name] for name in data.sample_names))]
    design = None if strata is None else _strata_design(strata_codes(strata, data.shape[1]), labels)
    keep = slice(None) if design is None else design["keep"]
    # The informative samples' strata; one stratum with both groups is the unstratified test on its samples.
    code = None if design is None or design["record"]["with_both_groups"] == 1 else design["code"]
    tlabels = labels[keep]                                          # the labels the tests exchange
    if min(int(tlabels.sum()), int((tlabels == 0).sum())) < 2:
        raise ValueError("the strata with both groups hold fewer than two cases or two controls")
    space = permutation_space(labels) if design is None else permutation_space(labels, design["all"])
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
    # Store a concrete seed even when the caller requests a fresh random run.
    if seed is None:
        seed = np.random.SeedSequence().entropy
    # Validate now, even when no gene will be tested. A sequence supports frozen protocols.
    np.random.default_rng(seed)
    if isinstance(seed, (np.random.Generator, np.random.BitGenerator, np.random.SeedSequence)):
        raise TypeError("seed must be an integer, an integer sequence, or None")
    seed = np.asarray(seed).tolist()
    factors = fitted.size_factors
    values = np.empty(data.shape, float)
    case_mean, control_mean = np.empty(g), np.empty(g)
    pvalue, subset_p = np.full(g, np.nan), np.full(g, np.nan)
    size, side = np.zeros(g, int), np.zeros(g, np.int8)
    fold, alignment, gap = np.full(g, np.nan), np.full(g, np.nan), np.zeros(g)
    exceed, tried = np.zeros(g, np.uint64), np.zeros(g, np.uint64)
    subset_exceed, subset_tried = np.zeros(g, np.uint64), np.zeros(g, np.uint64)
    shift, observed = np.ones(g), np.zeros(g)                     # each gene's registration and observed gap score
    area_live, subset_live = np.zeros(g, bool), np.zeros(g, bool)  # the genes each test shuffles

    def area_step(z, genes, threshold):
        hits, shuffles = exceed[genes], tried[genes]
        advance_means(z, tlabels, hits, shuffles, direction=direction, stop=stop, maximum=maximum,
                      threshold=threshold, seed=seed, code=code)
        exceed[genes], tried[genes] = hits, shuffles

    def subset_step(pooled, order, genes, threshold):
        hits, shuffles = subset_exceed[genes], subset_tried[genes]
        advance(pooled, order, observed[genes], tlabels, hits, shuffles, direction=direction, stop=stop,
                maximum=maximum, threshold=threshold, seed=seed, code=code, draws=DRAWS)
        subset_exceed[genes], subset_tried[genes] = hits, shuffles

    def area_input(counts_rows, genes):
        return area_values(counts_rows[:, keep], genes, factors[keep], seed, code)

    def walked(x):
        """The values the subset's description reads, and each gene's registration: with strata,
        aligned's pool."""
        if code is None:
            return x[:, keep], None
        return aligned(x[:, keep], tlabels, code)

    def tested_pool(counts_rows, genes):
        """The values the subset test shuffles: thinned's DRAWS pools of each gene, labels exchanged
        within strata, each shuffle scored by the mean of the pools' walks."""
        return walks(thinned(counts_rows[:, keep], factors[keep], tlabels, genes, seed, code, DRAWS)[0], tlabels,
                     np.ones(len(genes) * DRAWS), direction=direction, draws=DRAWS)

    # The first pass runs every shuffled gene until BH at its top threshold, fdr itself, is settled
    # (pvalues.settle): certified below it, or finished. fdr=None finishes every run here.
    first = -np.inf if fdr is None else fdr * (1 - TIE_RTOL)
    for start in range(0, g, chunk):
        rows = slice(start, min(start + chunk, g))
        block = data.values[rows]
        x = expression(jittered(block, range(rows.start, rows.stop), seed), factors)
        values[rows] = x
        plain = expression(block, factors)                       # the effects need no jitter
        case_mean[rows], control_mean[rows] = _group_means(plain, labels, design if code is not None else None, keep)
        pool, registration = walked(x)
        size[rows], side[rows], fold[rows], gap[rows], shift[rows] = split(pool, tlabels)
        # With strata the pool is registered already, by the strata's one fold.
        alignment[rows] = np.log2(shift[rows] if registration is None else registration)
        local = np.flatnonzero(tested[rows])
        if len(local):
            idx = start + local
            z = area_input(block[local], idx)
            shuffled = np.ones(len(local), bool)
            if computed:
                pvalue[idx], error = mean_saddlepoint(z, tlabels, direction=direction, code=code)
                shuffled = error > relative_error                    # less precise than shuffling
            area_live[idx[shuffled]] = True
            area_step(z[shuffled], idx[shuffled], first)
            ran = idx[gap[idx] > 0]                                  # genes without a subset have p = 1
            subset_live[ran] = True
            pooled, order, observed[ran] = tested_pool(block[ran - start], ran)
            subset_step(pooled, order, ran, first)

    # Later passes lower BH's threshold, revisiting only genes not yet settled at it.
    def area_more(genes, threshold):
        for at in range(0, len(genes), chunk):
            part = genes[at:at + chunk]
            area_step(area_input(data.values[part], part), part, threshold)

    def subset_more(genes, threshold):
        for at in range(0, len(genes), chunk):
            part = genes[at:at + chunk]
            pooled, order, _ = tested_pool(data.values[part], part)
            subset_step(pooled, order, part, threshold)

    pvalue, area_passes = settle(np.where(area_live, np.nan, pvalue), area_live, exceed, tried, area_more,
                                 fdr=fdr, stop=stop, maximum=maximum)
    subset_p, subset_passes = settle(np.where(tested & ~subset_live, 1.0, np.nan), subset_live, subset_exceed,
                                     subset_tried, subset_more, fdr=fdr, stop=stop, maximum=maximum)
    lower = np.minimum(case_mean, control_mean)
    groups = np.where(tested, np.where(lower >= SUBSET_VALIDATED_READS, 0, 1), -1)
    subset_rarity = rarity(lambda part: walked(values[part])[0], tlabels, size, side, groups)
    area = case_mean - control_mean
    total = case_mean + control_mean
    fraction = np.divide(area, total, out=np.full(g, np.nan), where=total > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log2fc = np.log2(case_mean / control_mean)                # +-inf where one group has no reads
    below = tested & (np.minimum(case_mean, control_mean) < SUBSET_VALIDATED_READS)
    if below.any():
        warnings.warn(f"{int(below.sum()):,} of {int(tested.sum()):,} tested genes average fewer than "
                      f"{SUBSET_VALIDATED_READS} reads per sample in their lower group, below the range where the "
                      f"subset test is validated: their subset p-values can be anti-conservative. The area test "
                      f"holds at every expression level.", UserWarning, stacklevel=2)
    kept = float(factors.min() / np.median(factors)) if design is None else \
        float(np.median(floors(factors[keep], code) / factors[keep]))
    params = dict(algorithm="area-and-subset-v5", direction=direction,
                  area_test="difference in means of counts thinned binomially until every size factor equals the smallest"
                  if design is None else "the cases' total of counts thinned within each stratum to its smallest size "
                  "factor and centred within it (the Cochran-Mantel-Haenszel sum), labels exchanged within strata",
                  depth_equalized=kept,
                  subset_statistic="largest gap between the case and control Lorenz curves of count + 1",
                  subset_registration="counts thinned to the smallest size factor (as the area test), then the "
                  "higher group thinned by the Wasserstein-1 fold of log(count + 1) of those counts (the middle of any "
                  "median interval)" if design is None else "each stratum thinned to its smallest size factor (as the "
                  "area test), then the higher group thinned by one fold, the weighted median (weights n1s n0s / ns) "
                  "of the strata's own Wasserstein-1 folds of those counts, and each stratum moved to the cohort's "
                  "median log(count + 1); labels exchanged within strata",
                  subset_draws=DRAWS,
                  subset_score=f"the walk's gap averaged over {DRAWS} independent thinning draws",
                  subset_description="the subset, its side, share, fold and gap read on normalized counts" + (
                  "" if design is None else ", each stratum moved to the cohort's level and the cases carried back "
                  "by the strata's one fold"),
                  subset_validated_reads=SUBSET_VALIDATED_READS,
                  subset_validated_genes=int((tested & ~below).sum()),
                  expression="(count + jitter) / size_factor",
                  effects="count / size_factor" if design is None else "Mantel-Haenszel standardized group means of "
                  "count / size_factor, weights n1s n0s / ns over the strata with both groups",
                  jitter="Uniform(0, 1) per count, seeded by seed and gene row",
                  thinning="Binomial per count, sample j keeping min(s) / s_j of its reads (s: the normalization's "
                           "size factors" + ("" if design is None else ", the minimum within its stratum") +
                           "), seeded by seed and gene row, counts rounded to whole reads",
                  strata=None if design is None else design["record"],
                  family="tested genes", hypotheses=int(tested.sum()), pvalues=pvalues,
                  area_pvalue="saddlepoint unless its error estimate exceeds relative_error"
                  if computed else "shuffled", relative_error=relative_error, max_permutations=maximum,
                  stop_exceedances=stop, minimum_shuffled_pvalue=1 / (maximum + 1),
                  fdr=fdr, stopping="each shuffled gene until its BH decision at fdr is settled (its "
                  "Besag-Clifford bound below BH's threshold, or its run finished)" if fdr is not None
                  else "every shuffled gene to the end of its Besag-Clifford run",
                  passes=dict(area=area_passes, subset=subset_passes),
                  rarity="share of the genes in its group whose gap at this gene's cut is at least its own, "
                         "ranked across the group; 1 without a subset",
                  rarity_groups=dict(validated=int(np.sum(groups == 0)), below=int(np.sum(groups == 1))),
                  seed=seed,
                  gene_chunk=chunk, stream_block=STREAM_BLOCK,
                  counts_sha256=_fingerprint(data), permutation_space=space,
                  size_factor_median=dict(case=float(np.median(factors[labels == 1])),
                                          control=float(np.median(factors[labels == 0]))),
                  test_selection="nonzero" if test_genes is None else "explicit; all-zero genes excluded")
    if isinstance(cond, Condition):
        params.update(case_label=cond.case, control_label=cond.control, condition_column=cond.column)
    part = np.where(size == 0, "none", np.where(side > 0, "up", "down"))
    return WadeResult(data.gene_names.copy(), status, case_mean, control_mean, area, fraction, log2fc, pvalue,
                      bh_adjust(pvalue),
                      alignment, gap, part, size / int(tlabels.sum()), fold, subset_p, bh_adjust(subset_p),
                      subset_rarity,
                      tried, exceed, subset_tried, subset_exceed, values, labels.copy(),
                      data.sample_names.copy(), fitted, params, dict(data.meta),
                      strata=None if design is None else design["stored"])


def _strata_design(codes, labels):
    """The strata's design: which strata hold both groups (only those can move a label), the informative
    samples and their strata renumbered 0 to S - 1, the run record, and every sample's stratum as stored
    in the result (-1 for a sample left out). Warns about the samples left out; an error if no stratum
    holds both groups."""
    S = int(codes.max()) + 1
    n1s = np.bincount(codes, weights=labels, minlength=S).astype(np.int64)
    ns = np.bincount(codes, minlength=S)
    moves = (n1s > 0) & (n1s < ns)
    if not moves.any():
        raise ValueError("no stratum holds both cases and controls: within-stratum shuffles cannot move a label")
    keep = moves[codes]
    if not keep.all():
        warnings.warn(f"{int((~moves).sum())} of {S} strata hold only cases or only controls; their "
                      f"{int((~keep).sum())} samples cannot move a label and are left out of both tests.",
                      UserWarning, stacklevel=3)
    code = (np.cumsum(moves) - 1)[codes[keep]]
    n1, n0 = n1s[moves], (ns - n1s)[moves]
    record = dict(strata=S, with_both_groups=int(moves.sum()), samples_left_out=int((~keep).sum()),
                  cases_tested=int(n1.sum()), controls_tested=int(n0.sum()),
                  effective_size=float(np.sum(n1 * n0 / (n1 + n0))),
                  effective_size_unstratified=float(labels.sum() * (len(labels) - labels.sum()) / len(labels)),
                  cases_by_stratum=n1.tolist(), controls_by_stratum=n0.tolist())
    stored = np.full(len(codes), -1, np.int64)
    stored[keep] = code
    return dict(keep=keep, code=code, all=codes, record=record, stored=stored)


def _group_means(plain, labels, design, keep=slice(None)):
    """The groups' mean normalized counts over the tested samples; with strata, the Mantel-Haenszel
    standardized means over the strata with both groups, sum_s w_s mean_gs / sum_s w_s with
    w_s = n1s n0s / ns."""
    if design is None:
        x, cond = plain[:, keep], labels[keep]
        return x[:, cond == 1].mean(axis=1), x[:, cond == 0].mean(axis=1)
    x, cond = plain[:, design["keep"]], labels[design["keep"]]
    case, control, total = 0.0, 0.0, 0.0
    for idx in stratum_columns(design["code"]):
        here = cond[idx]
        n1 = int(here.sum())
        w = n1 * (len(idx) - n1) / len(idx)
        case = case + w * x[:, idx][:, here == 1].mean(axis=1)
        control = control + w * x[:, idx][:, here == 0].mean(axis=1)
        total += w
    return case / total, control / total


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
