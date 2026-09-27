# WADE — Wasserstein Area Differential Expression

WADE tests whether a gene's expression distribution differs between two
cohorts. It detects distribution changes that can include a bulk shift,
a recurrent subset, or both, then describes those patterns separately.
It is intended to complement conventional differential expression in large
RNA-seq cohorts. Small studies remain a natural setting for DESeq2.

**0.2.0 is under development.** The public API below is the working point-based
baseline. The release targets **directional biomarker discovery**; unsigned-only
distribution detection is not the agreed release solution. The
[research comparison](research/README.md#replicated-point-comparison) evaluates
direction and scale without changing public inference.
Measurement kernels are deferred until after 0.2.0. The [release plan](PLAN.md) governs that work;
[current documentation](docs/index.md) separates usage from archived studies.
Rare-event probabilities across the transcriptome are a subsequent research
phase; a large descriptive tail is not a certified rare-event call.

```python
import wade

# Raw counts: genes × samples. 1 = case, 0 = control.
result = wade.wade(counts, condition)
result.to_frame().sort("padj")  # optional Polars dependency
result.plot_gene("ETV1")       # gene IDs come from a frame or gene_names=
```

The standard workflow uses RLE size factors, `log2(count / size_factor + 1)`,
exact empirical W1, label permutations and one BH adjustment over all tested
genes. There is no added noise, fitted-fold null, or second significance test.

Fit normalization explicitly when using validated internal controls or
sharing factors across test-gene selections:

```python
normalization = wade.fit_normalization(counts, reference_genes=control_gene_ids)
keep = wade.filter_counts(counts, min_count=3, min_samples=2)
result = wade.wade(counts, condition, normalization=normalization, test_genes=keep)
wade.write_results(result, "results.tsv")  # also writes a complete manifest
```

A test filter never refits normalization. Candidate and usable reference IDs,
excluded genes, actual permutation counts and sample order are recorded.
Choose reference genes and technical exclusions before examining significance.
RLE needs a defensible stable reference population; more genes alone do not
repair a mostly shifted reference. Sparse assays can supply externally fitted
size factors. `cpm` and `tpm` provide independent per-sample abundance conversions.

`statistic`, `pvalue` and `padj` belong to the whole-gene distribution test.
`bulk_shift` is a difference in log-expression medians. `upper_tail` and
`lower_tail` describe decile differences relative to that bulk shift. They
have no separate p-values, and they do not estimate subtype prevalence.

The default cap is 499,999 shuffles; obvious nulls stop at their 20th
exceedance. Runtime depends strongly on how many genes differ. A smaller
`max_permutations` can speed exploration but raises the p-value floor.
The [manual](docs/manual.md) explains resolution, normalization, filtering,
strata and interpretation. The [method](docs/method.md) gives exact definitions.

## Install from this checkout

```sh
python -m pip install -e ".[io,plot]"
```

Building requires Rust. The core depends on NumPy; Polars and plotting
libraries are optional. Published wheels include the native kernel. A readable
NumPy implementation provides a correctness reference and slower fallback.

[Demo](notebooks/demo.qmd) · [Benchmarks](notebooks/reference-benchmarks.qmd) ·
[Prostate](notebooks/examples.qmd) · [Breast](notebooks/examples-breast.qmd) ·
[Release plan](PLAN.md) · [Contributing](CONTRIBUTING.md)
