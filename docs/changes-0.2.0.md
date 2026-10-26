# Changes in WADE 0.2.0

0.2.0 replaces 0.1.0's two-stage test (`p_mean_shift`, then `p_subset`) with
two tested quantities per gene: the **area** between cases and controls, and
the **subset** of cases that departs from the one fold every case would
share. It is a breaking release with no compatibility aliases. This note maps
0.1.0, the published release (tag `v0.1.0`), onto 0.2.0. The
[manual](manual.md) documents every argument and column, and the
[method](method.md) gives the definitions.

## What the tests now do

- **The area** is exactly the difference in group means, in reads per sample:
  the signed area between the quantile functions, with no quantile grid. Its
  **area fraction**, the area over the total area under both curves, lies
  between −1 and 1, and `log2fc` is log2(case mean / control mean).
- **Normalization** is one size factor per sample, RLE by default. Means are
  count / size factor: reads per sample at the cohort's typical depth, not
  per million.
- **Depth equalization.** Before both tests every sample is thinned
  binomially until all size factors equal the smallest. Depth cannot then
  pass for expression, and genes with too few reads are washed out rather
  than called. The effects use every read.
- **The subset** is the cases above or below the point where the cases'
  Lorenz curve of count + 1 falls furthest below the controls'. Its test asks
  whether every case could carry one common fold, and averages its walk over
  four thinning draws. It is validated where the lower group averages at
  least 32 reads per sample, and `wade()` warns how many tested genes fall
  below that.
- **P-values.** `relative_error` sets the precision of every p-value. The
  area test's p-value is a saddlepoint approximation to its exact
  permutation distribution, shuffled where the approximation's error
  estimate exceeds `relative_error`; the subset test is always shuffled.
  With `fdr` set, shuffles stop once each gene's BH decision is settled, and
  the decisions are exactly those of finished runs.
- **Strata** exchange labels only within each stratum and thin each stratum
  to its own smallest size factor. The area test combines strata with
  Cochran–Mantel–Haenszel weights.

## `wade()`

```python
# 0.1.0
wade(counts, normalizer, cond, *, nperms=2000, seed=1, alternative="two-sided",
     lib_sizes=None, strata=None, n_boot=0, boot_level=0.95, subset=True,
     pseudocount=1.0, max_probs=2000, gene_chunk=None, stage1="grid",
     fit_backend="numpy", backend="auto", gene_names=None, sample_names=None,
     noise=0.01, norm_factor=1e6, n_exc_min=10, n_tail=250, jitter=None,
     perms=None, keep_null=False, allow_single_sample_group=False,
     allow_empty_samples=False)

# 0.2.0
wade(counts, cond, *, direction="both", normalization="rle", test_genes=None,
     pvalues="auto", fdr=0.05, relative_error=0.2, max_permutations=999_999,
     seed=1, strata=None, gene_names=None, sample_names=None, gene_chunk=None)
```

The second positional argument is now the condition: `wade(counts,
normalizer, cond)` becomes `wade(counts, cond)`.

| 0.1.0 | 0.2.0 |
| --- | --- |
| `normalizer` (positional), `lib_sizes`, `norm_factor` | `normalization=`: `"rle"` (default), `"library"` (total counts), or a `Normalization` from `fit_normalization(...)` or `Normalization.from_size_factors(...)` |
| `alternative="two-sided"`, `"greater"`, `"less"` | `direction="both"`, `"up"`, `"down"`, for both tests |
| `nperms` | `relative_error`, `max_permutations` and `fdr` |
| `stage1="saddlepoint"` | `pvalues="auto"` (the default) |
| `seed`, `strata`, `gene_names`, `sample_names`, `gene_chunk` | unchanged names |
| none | `test_genes=`, a mask such as `filter_counts(...)` returns |

**Removed with no counterpart:** `n_boot`, `boot_level`, `subset` (the
subset test always runs), `pseudocount`, `max_probs`, `fit_backend`,
`backend`, `noise`, `jitter`, `perms`, `n_exc_min`, `n_tail`, `keep_null`,
`allow_single_sample_group` and `allow_empty_samples`. The jitter is
Uniform(0, 1) per count and the shuffles come from one stream, both seeded
by `seed`. Each group needs at least two samples, and `wade()` refuses
samples with no reads: remove them first.

## Result columns

| 0.1.0 | 0.2.0 | Note |
| --- | --- | --- |
| `gene` | `gene` | |
| none | `status` | `tested`, `filtered` (left out by `test_genes`) or `all_zero` |
| `case_mean`, `ctrl_mean` | `case_mean`, `control_mean` | Reads per sample at the cohort's typical depth |
| `mean_shift` | `area` | Exactly the difference in means |
| none | `area_fraction` | |
| `log2_fc` | `log2fc` | log2(case_mean / control_mean): ±inf where a group has no reads |
| `p_mean_shift`, `padj_mean_shift` | `area_pvalue`, `area_padj` | Tested on depth-equalized counts |
| `fitted_fold_change` (an attribute) | `alignment_log2fc` | The common fold, as log2 |
| `subset_stat` | `lorenz_gap` | A different statistic: the largest gap between the Lorenz curves |
| `p_subset`, `padj_subset` | `subset_pvalue`, `subset_padj` | The null that every case carries one common fold |
| `affected_fraction` | `subset_fraction` | The subset's share of the cases |
| `direction`, a score from −1 to 1 | `subset_side` | `"up"`, `"down"` or `"none"` |
| `subset_log2_fc` | `subset_log2fc` | The subset's fold over what the common fold predicts |
| none | `subset_rarity` | A size-matched rank of the subset |
| `w1`, `neglog10_p_*`, `z_*`, `nexc_*`, `refined_*`, bootstrap `*_lo` / `*_hi` | removed | |

Each test has its own BH family over the tested genes. `result.bounded` marks
the p-values that are bounds because their shuffles stopped at the decision.

## Other changes to the API

- `wade_contrast(counts, normalizer, case_samples, ctrl_samples, ...)` is now
  `wade_contrast(counts, case_samples, control_samples, ...)`. Groups are
  chosen by sample ID or Boolean mask; integer column positions are no
  longer accepted. Its `strata=` follows the counts' sample order, as
  before.
- `condition(..., strata=)` reads strata from sample metadata columns.
- `library_qc(counts, *, top_n=10)` drops `normalizer=`.
- `plot_gene(result, gene)` and `plot_volcano(result)` take a result only,
  and `plot_patterns(result)` is new. The volcano maps area fractions against
  area q-values; the patterns plot maps each subset's share against its
  fold. matplotlib is the default backend; pass `backend="plotly"` for
  plotly. `Theme` drops `sequential`, `diverging`, `scale()` and `box()`.
- `WadeResult` gains `subset(gene)`, `rare_subsets(n)`, `sample_load()`,
  `bounded`, and the `plot_gene`, `plot_volcano` and `plot_patterns`
  methods. `result.expression`, 0.1.0's `result.tpm`, holds the jittered
  normalized counts.
- `gene_detail(gene)` returns the exact quantile intervals of
  `result.expression`, without strata. Their signed area is the difference in
  means of those jittered values, so it differs from `area` by the jitter.
- `as_counts` drops `exclude=`: name the sample columns with
  `sample_columns=`. `Counts.meta`, `Counts.normalizer()` and
  `WadeResult.gene_meta` are gone.
- `write_results` writes integer gene IDs, which 0.1.0 refused, and NumPy
  scalar labels.

**Removed names.** `ALTERNATIVES`, `DEFAULT_MAX_PROBS`, `DEFAULT_NPERMS`,
`affected_fraction`, `detectability_floor`, the `direction` function,
`draw_jitter`, `draw_perms`, `library_sizes`, `null_statistics`,
`perm_pvalues`, `plot_drivers`, `plot_linked`, `plot_stages`,
`probability_grid`, `subset_drivers`, `subset_log2_fc`, `tpm_like`,
`wade_gene` and `wade_stats`. From `wade.plotting`: `BACKENDS`,
`DEFAULT_BACKEND`, `DEFAULT_THEME`, `QUADRANTS`, `available_backends`,
`GenePanel`, `VolcanoData`, `StagesData`, `DriverPanel`, `gene_panels`,
`volcano_data`, `stages_data` and `driver_panel`.

**New names.** `DEFAULT_MAX_PERMUTATIONS`, `Normalization`, `Subset`,
`fit_normalization`, `filter_counts`, `cpm`, `tpm` and `plot_patterns`.
`cpm` and `tpm` convert counts for reporting; they are not test inputs.
`permutation_space` stays.

## Behaviour to note

- Genes with no reads in the samples the tests use have `status`
  `all_zero`: no test and no subset (`subset_side` `"none"`). Like filtered
  genes, they stay in the table outside the BH families.
- Strata holding only one group are left out of both tests, with a warning.
  A gene whose reads all lie in left-out samples is then `all_zero`.
- Missing values in strata raise an error, whether strata come as one
  vector, a list of factors or `condition(..., strata=)`. 0.1.0 grouped them
  silently.
- The manifest records the hypothesis families, the p-value settings, the
  strata, the depth kept, the groups' median size factors and every stopping
  record.

The evidence for the new tests is in [validation](validation.md).
