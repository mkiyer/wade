# Changes in WADE 0.2.0

0.2.0 replaces the 0.1.x two-stage test with two questions per gene, each
tested: how far apart the cases and controls lie as a whole (the area), and
whether a subset of cases departs from the one fold every case would share.
This is an intentional breaking release with no compatibility aliases.

## The area and the subset

- `wade(counts, cond)` measures each gene's **area**, the signed area between
  the case and control quantile functions of normalized counts. It is exactly
  the difference in means, in reads per sample. It also reports the **area
  fraction**, the area over the total area under both curves, between −1 and
  1.
- The **fold**, `log2fc`, is log2(case mean / control mean), with no
  pseudocount (±inf where a group has no reads).
- Every normalization ends as one size factor per sample: RLE on all genes,
  RLE on internal control genes, or factors fitted elsewhere. Before testing,
  both tests thin every sample binomially until all size factors equal the
  smallest.
  - Samples sequenced to different depths are then exchangeable under the
    null at every expression level.
  - Genes with too few reads to trust are washed out rather than called.
  - The effects use every read.
- WADE finds each gene's **subset**: the cases above or below the point where
  the cases' Lorenz curve of count + 1 falls furthest below the controls'.
  Nothing about the subset's size is chosen in advance.
- **The subset test** asks whether every case could carry one common fold of
  the controls' distribution.
  - It thins the depth-equalized counts of the higher group by the
    Wasserstein-1 fold of log(count + 1), rather than rescaling them: a
    thinned count keeps the count model exact.
  - It then shuffles labels over the thinned counts.
  - Its validated range is genes whose sparser group averages at least 32
    reads per sample, and `wade()` warns how many tested genes fall below it.
  - Its calls near BH's threshold depend on the thinning draw, which the
    seed sets.
- `direction="both"` (default) tests both two-sided. `"up"` or `"down"`
  declares a side for both: an area in that direction, and a subset of cases
  beyond the rest on that side.
- Each count receives seeded Uniform(0, 1) jitter, so that no two values tie,
  above all zeros.
- **Strata.** `strata=` takes one label per sample (study, site, batch), or a
  list of factors whose combinations are the strata; `condition(...,
  strata=...)` reads them from metadata columns.
  - Labels are exchanged only within strata, and each stratum is thinned to
    its own smallest size factor.
  - The area test sums within-stratum differences with
    Cochran–Mantel–Haenszel weights, its p-value from a stratified
    saddlepoint or from shuffles within strata.
  - The subset test takes one fold out of every case and a level per
    stratum.
  - case_mean and control_mean are Mantel–Haenszel standardized means.
  - Strata holding only one group are left out of both tests, with a
    warning.
  - `params["strata"]` records the design.
  - Pairing and repeated measures are not validated yet.

## P-values

- `relative_error=0.2` sets the precision of every p-value. Shuffled p-values
  follow Besag–Clifford's sequential rule, stopping at the
  ⌈1/relative_error²⌉-th exceedance (the 25th by default) or at
  `max_permutations` shuffles (default 999,999, so an isolated discovery stays
  possible among 50,000 genes).
- `pvalues="auto"` (default) computes the area test from the exact permutation
  distribution of the cases' total by Skovgaard's double saddlepoint, with no
  floor above one labelling. It shuffles only the genes whose saddlepoint
  error estimate exceeds `relative_error`. `pvalues="permutation"` shuffles
  every p-value. The subset test is always shuffled.
- `fdr=0.05` (default) is the level at which BH decides. Shuffles stop once
  each gene's decision is settled: once a bound on its finished p-value falls
  below BH's threshold.
  - BH rejects exactly the genes it would reject had every run finished, so
    the false discovery rate is unchanged.
  - Strongly significant genes need hundreds of shuffles instead of the full
    budget.
  - Rejected genes report the bound as their p-value; `result.bounded` marks
    them. `fdr=None` finishes every run.
- Every gene, in both tests, sees one seeded stream of shuffles that can be
  regenerated from any position, so a gene can stop and resume exactly.
- The subset walk and its registration are native code, one implementation for
  the observed split and every shuffle.

## Results

One row per input gene, with these columns: `gene`, `status`, `case_mean`,
`control_mean`, `area`, `area_fraction`, `log2fc`, `area_pvalue`, `area_padj`,
`alignment_log2fc`, `lorenz_gap`, `subset_side`, `subset_fraction`,
`subset_log2fc`, `subset_pvalue`, `subset_padj` and `subset_rarity`. Each test has its own BH
family over the tested genes.

- `result.subset(gene)` returns the subset's cases, its side, share, fold,
  p-value and q-value.
- `subset_rarity` ranks each gene's subset among genes with a subset of the
  same size and side: the share of genes whose subset is at least as rare,
  within the subset test's validated range or below it.
  - `result.rare_subsets(n)` lists the rarest subsets, with the genes that
    share their members.
  - `result.sample_load()` shows which cases carry the rarest subsets, and
    their driving genes.
- `gene_detail(gene)` returns exact quantile intervals and their signed area.
- `plot_volcano` maps area fractions against area q-values, `plot_patterns`
  each gene's subset share against its fold, and `plot_gene` shows log2
  expression.
- The manifest records the hypothesis families, the p-value settings, the
  depth kept, group size-factor medians and every stopping record.

**Removed from 0.1.x.**

- The `p_mean_shift`/`p_subset` stages and their q-values, the fitted
  bulk-shift null, GPD tail extrapolation and bootstrap intervals.
- The arguments `normalizer`, `nperms`, `max_probs`, `stage1`, `fit_backend`,
  `backend`, `n_exc_min` and `jitter`.
- The names `ALTERNATIVES`, `DEFAULT_MAX_PROBS`, `DEFAULT_NPERMS`,
  `affected_fraction`, `detectability_floor`, the `direction` function,
  `draw_jitter`, `library_sizes`, `null_statistics`, `perm_pvalues`,
  `plot_drivers`, `plot_linked`, `plot_stages`, `probability_grid`,
  `subset_drivers`, `subset_log2_fc`, `tpm_like`, `wade_gene` and
  `wade_stats`.

**Also removed before release:** the `scale="log2"` option and the
`subsets="either"` option. They are recorded in
[research ideas](research-ideas.md).

**New names:** `DEFAULT_MAX_PERMUTATIONS`, `Normalization`, `Subset`,
`fit_normalization`, `filter_counts`, `cpm`, `tpm` and `plot_patterns`. New
arguments: `wade(..., strata=)` and `condition(..., strata=)`.

## Normalization and filtering

- `fit_normalization(counts)` fits RLE once; `reference_genes=` restricts it to
  validated controls; `normalization="library"` uses total counts;
  `Normalization.from_size_factors(...)` imports external exposures.
- `filter_counts` and `test_genes=` choose the tested genes without refitting.
  Filtered and all-zero genes stay in the table with a status and no test.
- `cpm` and `tpm` convert single samples for reporting; they are not test inputs.

## Evidence

The area test was measured under an exact count-model reference built from
real tumours, and the subset test's validated range under the same reference.
Strata were measured against exact enumeration, on simulated studies with
batch effects, and on real relabelling nulls from TCGA and GTEx; see
[validation](validation.md).
