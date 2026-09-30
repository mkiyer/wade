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
- The area test thins every sample binomially to the shallowest sample's depth
  before testing. Samples sequenced to different depths are then exchangeable
  under the null at every expression level, and genes with too few reads to
  trust are washed out rather than called. The effects use every read.
- WADE finds each gene's **subset**: the cases above or below the point where
  the cases' Lorenz curve of count + 1 falls furthest below the controls'.
  Nothing about the subset's size is chosen in advance. The subset test asks
  whether every case could carry one common fold of the controls'
  distribution. It registers the cases by the Wasserstein-1 fold of
  log(count + 1) and shuffles labels over the pooled values. Its validated
  range is genes whose sparser group averages at least 128 reads per sample,
  and `wade()` warns how many tested genes fall below it.
- `direction="both"` (default) tests both two-sided. `"up"` or `"down"`
  declares a side for both: an area in that direction, and a subset of cases
  beyond the rest on that side.
- Each count receives seeded Uniform(0, 1) jitter, so that the subset test's
  null cannot pull apart ties, above all zeros.
- Samples must be independent. `strata` is not supported in this version, and
  passing it is an error.

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
- The subset walk and its registration are native code, one implementation for
  the observed split and every shuffle.

## Results

One row per input gene, with these columns: `gene`, `status`, `case_mean`,
`control_mean`, `area`, `area_fraction`, `area_pvalue`, `area_padj`,
`alignment_log2fc`, `lorenz_gap`, `subset_side`, `subset_fraction`,
`subset_log2fc`, `subset_pvalue` and `subset_padj`. Each test has its own BH
family over the tested genes.

- `result.subset(gene)` returns the subset's cases, its side, share, fold,
  p-value and q-value.
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
`fit_normalization`, `filter_counts`, `cpm`, `tpm` and `plot_patterns`.

## Normalization and filtering

- `fit_normalization(counts)` fits RLE once; `reference_genes=` restricts it to
  validated controls; `normalization="library"` uses total counts;
  `Normalization.from_size_factors(...)` imports external exposures.
- `filter_counts` and `test_genes=` choose the tested genes without refitting.
  Filtered and all-zero genes stay in the table with a status and no test.
- `cpm` and `tpm` convert single samples for reporting; they are not test inputs.

## Evidence

The area test was measured under an exact count-model reference built from
real tumours, and the subset test's validated range under the same reference;
see [validation](validation.md).
