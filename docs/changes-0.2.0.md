# Changes in WADE 0.2.0

0.2.0 replaces the 0.1.x two-stage test with two questions per gene, each
tested: how far the cases shift as a whole, and which cases depart from that
shift. This is an intentional breaking release with no compatibility aliases.

## The mean shift and the subset

- `wade(counts, cond)` measures each gene's **mean shift**, the signed area
  between the case and control quantile functions, which is exactly the
  difference in means, and tests it against label exchange.
- It finds each gene's **subset**, the cases that depart from the mean shift,
  above it or below it, by walking the case quantiles against the prediction
  that every case carries the shift. Nothing about the subset's size is chosen
  in advance. The subset test asks whether every case could carry the mean
  shift. A gene can shift one way as a whole and have a subset that departs
  the other way.
- `direction="both"` (default) tests both two-sided; `"up"` or `"down"`
  declares a side for both: a mean shift in that direction, and a subset beyond
  the mean shift on that side.
- `scale="linear"` (default) tests `count / size_factor`; `scale="log2"` tests
  `log2(count / size_factor + 1)`.
- Each count receives seeded Uniform(0, 1) jitter before normalization, so that
  the subset test's null cannot pull apart ties, above all zeros.

## P-values

- `relative_error=0.2` sets the precision of every p-value. Shuffled p-values
  follow Besag–Clifford's sequential rule, stopping at the
  ⌈1/relative_error²⌉-th exceedance (the 25th by default) or at
  `max_permutations` shuffles (default 999,999, so an isolated discovery stays
  possible among 50,000 genes).
- `pvalues="auto"` (default) computes the mean test from the exact permutation
  distribution of the cases' total by Skovgaard's double saddlepoint, with no
  floor above one labelling, and shuffles only the genes whose saddlepoint error
  estimate exceeds `relative_error`, or every gene when `strata` restrict the
  shuffles. `pvalues="permutation"` shuffles every p-value. The subset test is
  always shuffled.
- The subset walk is native code, one implementation for the observed split
  and every shuffle.

## Results

`gene`, `status`, `mean_log2fc`, `mean_pvalue`, `mean_padj`,
`subset_fraction`, `subset_log2fc`, `subset_pvalue` and `subset_padj`, one row
per input gene; each test has its own BH family over the tested genes.
`result.subset(gene)` returns the subset's cases, its side, share, fold,
p-value and q-value. `gene_detail(gene)` returns exact quantile intervals and
their signed area. `plot_volcano` maps mean shifts, `plot_patterns` each gene's
subset share against its fold, and `plot_gene` shows log2 expression. The
manifest records the hypothesis families, the p-value settings, group
size-factor medians and every stopping record.

Removed from 0.1.x: the `p_mean_shift`/`p_subset` stages and their q-values,
the fitted bulk-shift null, read thinning, GPD tail extrapolation, bootstrap
intervals, and the `normalizer`, `nperms`, `max_probs`, `stage1`, `fit_backend`,
`backend`, `n_exc_min` and `jitter` arguments. Removed names: `ALTERNATIVES`,
`DEFAULT_MAX_PROBS`, `DEFAULT_NPERMS`, `affected_fraction`,
`detectability_floor`, the `direction` function, `draw_jitter`, `library_sizes`,
`null_statistics`, `perm_pvalues`, `plot_drivers`, `plot_linked`,
`plot_stages`, `probability_grid`, `subset_drivers`, `subset_log2_fc`,
`tpm_like`, `wade_gene` and `wade_stats`. New names:
`DEFAULT_MAX_PERMUTATIONS`, `Normalization`, `Subset`, `fit_normalization`,
`filter_counts`, `cpm`, `tpm` and `plot_patterns`.

## Normalization and filtering

- `fit_normalization(counts)` fits RLE once; `reference_genes=` restricts it to
  validated controls; `normalization="library"` uses total counts;
  `Normalization.from_size_factors(...)` imports external exposures.
- `filter_counts` and `test_genes=` choose the tested genes without refitting.
  Filtered and all-zero genes stay in the table with a status and no test.
- `cpm` and `tpm` convert single samples for reporting; they are not test inputs.

## Evidence

The procedure was confirmed on 120 fresh simulated transcriptomes; see
[validation](validation.md).
