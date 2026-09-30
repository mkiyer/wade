# WADE 0.2.0 manual

For every gene WADE reports two things and tests each: the **mean shift**
between cases and controls, and the **subset** of cases that departs from it.
The [method](method.md) gives exact definitions; [validation](validation.md)
gives the measured performance.

## Before running

Fix the contrast, the biological population and the independent unit before
looking at results. Keep one observation per unit (one library per patient
and sample class), or use strata for a justified restricted design. Cells or
spots from one donor are not independent patients. Decide technical sample
exclusions in advance; never remove cases because they weaken a result.
Expression-derived labels (PAM50, RNA-called fusions) can select on the
outcome, and unknown annotations are not negatives.

`wade.library_qc(counts)` reports depth, detected genes and the share of reads
held by the top genes. The run record also reports the median size factor in
each group: a large difference means the groups differ in depth or
composition, which normalization cannot make exchangeable.

## Input

```python
import wade

# NumPy arrays have genes in rows and samples in columns.
result = wade.wade(counts, [1, 1, 1, 0, 0, 0], gene_names=gene_ids, sample_names=sample_ids)

# Frames and sample sheets are aligned by ID, strictly.
data = wade.as_counts(frame, id_column="gene_id", sample_columns=sample_ids)
cond = wade.condition(sample_sheet, key="sample_id", column="group", case="tumour", control="normal")
result = wade.wade(data, cond)
```

Raw integer gene counts are the validated input; negative or missing values
are errors. At least two samples per group are required, which is an input
guard, not a claim of adequate power. Sparse matrices are accepted through
`toarray()` and materialized.

## Normalize once, filter separately

The default fits RLE size factors (DESeq2's median-of-ratios) on all genes
positive in every sample, centered to geometric mean one. Fit once, then choose
the tested genes without refitting:

```python
norm = wade.fit_normalization(data)                                 # ordinary bulk RLE
norm = wade.fit_normalization(data, reference_genes=validated_ids)  # validated internal controls
keep = wade.filter_counts(data, min_count=3, min_samples=2)
result = wade.wade(data, cond, normalization=norm, test_genes=keep)
```

| Situation | Use |
| --- | --- |
| Typical bulk gene counts | default RLE |
| A protocol supplies validated internal controls | `fit_normalization(..., reference_genes=ids)` |
| Exposures fitted elsewhere | `Normalization.from_size_factors(data, factors, source=...)`, complete positive exposures including depth |
| Total-count exposure is the intended reference | `normalization="library"` |
| Per-sample abundance for reporting | `wade.cpm(counts)`, `wade.tpm(counts, lengths)`; not inputs to the test |

RLE assumes most genes do not change. When many genes shift together, the
reference moves and can create false discoveries; supply validated reference
genes or external factors in that case. A filter must be label-independent:
never choose thresholds by group differences or q-values. Genes outside the
filter keep their descriptions but have no test (`status == "filtered"`).
`wade_contrast(data, case_ids, control_ids, normalization=norm)` selects
groups by sample ID and keeps an existing fit.

## What WADE reports

**The mean shift** is the signed area between the case and control quantile
functions, which is exactly the difference in group means. `mean_log2fc`
states it as a log2 fold; `mean_pvalue` and `mean_padj` test it.

**The subset** is found from the mean shift itself. If every case carried the
mean shift alike (the same fold on count + 1, or on the log2 scale the same
difference), each case quantile would sit at a predicted level. Walking up the
cases, WADE tracks how far they have fallen below or risen above those
predictions in total; the walk's lowest point splits the cases into those that
carry less than their share and those that carry more. The smaller side, which
departs further per case, is the subset: above the mean shift
(`subset_log2fc > 0`) or below it (`subset_log2fc < 0`). `subset_fraction` is
its share of the cases. `subset_pvalue` and `subset_padj` test the null that
every case carries the mean shift.

Any combination can occur: a mean shift up with a subset further up (an
amplified subgroup among raised tumours), a mean shift down with a subset
raised above it (a fusion-driven subgroup in tumours that otherwise lose the
gene), or either with a lowered subset. No subset size is chosen in advance,
and a walk over noise also has a lowest point, so read the subset with its
q-value. One gene reports one subset; a raised and a lowered subset in the same
gene share one split.

## Direction and scale

```python
wade.wade(data, cond)                    # both tests two-sided
wade.wade(data, cond, direction="up")    # is the gene higher in cases, as a whole or in a subset?
wade.wade(data, cond, scale="log2")      # tests log2(count / size_factor + 1)
```

**`direction`** sets the alternative of both tests. `"both"` (default) tests
the mean shift two-sided and counts subsets on either side. `"up"` asks whether
cases are higher: a mean shift up, and a subset above the mean shift. `"down"`
asks the reverse. A one-sided subset test compares the observed subset only
with chance subsets on its own side. Declare the direction before looking; do
not run `"both"` and report one side as if it had been declared.

**`scale="linear"`** (default) tests `count / size_factor`. A subset far above
normal dominates the linear mean and the linear walk, which is what makes the
linear scale sensitive to raised subsets. **`scale="log2"`** tests
`log2(count / size_factor + 1)`, where a halving weighs as much as a doubling;
it is the better scale for lowered subsets and whole-cohort decreases. The two
scales can disagree about the same gene: a strong raised subset can pull the
linear mean up while most cases fall, which the log2 mean reports as a
decrease. Choose the scale before running.

## P-values

Every p-value is estimated to about `relative_error` (default 0.2).

- **The subset test is shuffled.** Label shuffles preserve group sizes and any
  strata. Each gene stops at its ⌈1/relative_error²⌉-th exceedance (25 at the
  default), whose count estimates p to that relative standard error, or at
  `max_permutations` shuffles.
- **The mean test is computed** with `pvalues="auto"` (default): the exact
  permutation distribution of the cases' total has a double-saddlepoint tail,
  with no shuffles and no floor above the chance of the single most extreme
  labelling. Its error is estimated from the saddlepoint's own next term and
  from how coarse the exact distribution is there: a smooth approximation
  cannot be much closer than one labelling among those in the tail. A gene
  whose estimate exceeds `relative_error` is shuffled instead, as are all genes
  when `strata` are given. In a small design this shuffles the genes far enough
  in the tail that only a handful of labellings reach them.
- **`pvalues="permutation"`** shuffles every p-value, including the mean test's.

A smaller `relative_error` buys precision with shuffles: each shuffled gene
that stops early needs about 1/(relative_error² · p) of them. The default
budget, `max_permutations=999_999`, lets a shuffled p-value reach 10⁻⁶, so an
isolated BH discovery at q ≤ 0.05 stays possible among up to 50,000 genes. The
manifest's `minimum_discovery_rank` says how many genes must sit at the
shuffled floor before any can pass. Genes with strong subsets use the whole
budget, so they dominate runtime; a smaller budget is fine for exploration.

Shuffled p-values near a threshold vary with the seed. Fix the seed, budget and
precision before looking, and never rerun borderline genes until they pass.

```python
result = wade.wade(data, cond, strata=batch_ids, seed=1)
```

`strata` restricts shuffles to exchangeable blocks, preserving each block's
case count. A block with only cases or only controls contributes no freedom;
a design with no possible exchange is rejected. `gene_chunk` changes memory,
never the answer.

## Jitter

Each count receives independent Uniform(0, 1) jitter before normalization,
drawn from a stream seeded by the run's seed and the gene, so no two values
tie. Counts are discrete, and many samples share one, above all zero; the
subset test's null moves every case by the mean shift and would pull such ties
apart, which the observed data never do. Without jitter the subset test is
anti-conservative for low-count genes in unbalanced designs
([validation](validation.md#the-subset-test-and-jitter)). The jitter moves a
count by less than one, so it barely changes well-expressed genes; for a gene
full of zeros, which zeros fall inside a subset depends on the seed.

## Read, plot and export

| Column | Meaning |
| --- | --- |
| `mean_log2fc` | The mean shift: log2((case mean + 1)/(control mean + 1)) on the linear scale, the difference in mean log2 expression on the log2 scale |
| `mean_pvalue`, `mean_padj` | Its p-value and BH q-value over the tested genes |
| `subset_fraction` | The subset's share of the cases (0 when every case carries the mean shift) |
| `subset_log2fc` | The subset's fold over the mean shift's prediction for the same cases: positive above it, negative below |
| `subset_pvalue`, `subset_padj` | The test that every case carries the mean shift, and its q-value |
| `status` | `tested`, `filtered` or `all_zero` |

`result.mean_permutations` and `result.subset_permutations` record the
shuffles each test used (0 for a computed mean test); the exceedances sit
beside them.

```python
result.subset("ETV4")             # the subset's cases, its side, share, fold, p-value and q-value
result.gene_detail("ETV4")        # exact quantile intervals; .area is the signed area
result.plot_gene("ETV4")          # ECDFs and quantile differences on log2
result.plot_volcano(label=["ETV1", "ETV4"])
result.plot_patterns()            # each gene's subset share against its fold
wade.write_results(result, "results.tsv")   # also writes results.manifest.json
```

Plots use matplotlib by default or `backend="plotly"`, with themes `light`,
`dark` and `high-contrast`; `import wade` loads no plotting library. Exports
support TSV/CSV, Parquet and Arrow through Polars. The manifest records the
normalization, references, test family, sample order, seed, strata, group
size-factor medians and every stopping record.

## Limits

- Samples must be exchangeable under the null. Batch or depth confounded with
  the groups violates this; normalization and strata help only where the
  design allows.
- A small `subset_padj` means the cases depart from the mean shift more than
  relabelling explains. It does not say which patients are individually
  outliers, that a subset is clinically separable, or how prevalent it is.
- The subset test plugs in the estimated mean shift, which makes it
  approximate; its calibration is measured, not guaranteed.
- In large, heterogeneous cohorts nearly every gene's cases depart from a
  uniform shift (tumour purity alone does it), so the subset test rejects
  almost everywhere. There the subset's fold and share, not its q-value, rank
  the genes.
- Very low counts carry little information on either scale; filter them.
