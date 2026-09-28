# Changes in WADE 0.2.0

0.2.0 replaces the 0.1.x two-stage test with one directional question per
gene: is case expression raised above, or lowered below, the controls? This
is an intentional breaking release with no compatibility aliases.

## One test per direction

- `wade(counts, cond)` measures the upward and downward areas between the
  case and control quantile functions and tests each by label permutation.
  A raised subset above a lowered bulk is an upward discovery; in 0.1.x its
  mean difference could read as a decrease.
- `direction="both"` (default) tests both areas in one pass with one BH family
  over every tested gene-direction pair. `direction="up"` or `"down"` tests a
  declared direction over the tested genes and skips the other.
- `scale="linear"` (default) tests `count / size_factor`; `scale="log"` tests
  `log2(count / size_factor + 1)`.
- p-values use a sequential Besag–Clifford rule per direction (20 exceedances,
  or the cap). The default cap of 999,999 keeps an isolated discovery possible
  among 50,000 hypotheses.

Removed from 0.1.x: the `p_mean_shift`/`p_subset` stages and their q-values,
the fitted bulk-shift null, read thinning, continuity jitter, GPD tail
extrapolation, the saddlepoint and bootstrap intervals. Removed names:
`ALTERNATIVES`, `DEFAULT_MAX_PROBS`, `DEFAULT_NPERMS`, `affected_fraction`,
`detectability_floor`, `direction`, `draw_jitter`, `library_sizes`,
`null_statistics`, `perm_pvalues`, `plot_drivers`, `plot_linked`,
`plot_stages`, `probability_grid`, `subset_drivers`, `subset_log2_fc`,
`tpm_like`, `wade_gene` and `wade_stats`. New names: `DEFAULT_MAX_PERMUTATIONS`,
`Normalization`, `fit_normalization`, `filter_counts`, `cpm`, `tpm` and
`plot_patterns`.

## Results

`area_up`, `area_down`, `pvalue_up`, `pvalue_down`, `padj_up`, `padj_down`,
`bulk_shift`, `subset_up_fraction`, `subset_up_log2fc`, `subset_down_fraction`,
`subset_down_log2fc` and `status`, one row per input gene. The subsets are
learned from each gene: a standardized running sum of cases over controls down
the pooled ranking, whose peak bounds the upward subset and whose lowest point
bounds the downward one. `result.subset(gene, direction)` returns the members
and boundary; with `cutoff=` it returns the cases beyond any log2 level.
`plot_patterns` maps each gene's subset share against its fold.
`gene_detail(gene)` returns exact quantile intervals with their two areas.
Volcano and pattern plots take `direction=`; gene plots show log2 expression.
The manifest records the hypothesis family, group size-factor medians and each
direction's stopping records.

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
