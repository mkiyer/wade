# WADE 0.2.0 manual

WADE tests, gene by gene, whether case expression is raised above or lowered
below a control cohort, in the whole cohort or in a subset of patients. The
[method](method.md) gives exact definitions; [validation](validation.md) gives
the measured performance.

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

## Direction and scale

```python
wade.wade(data, cond)                   # discovery: both directions, one FDR family
wade.wade(data, cond, direction="up")   # a declared upward hypothesis
wade.wade(data, cond, scale="log")      # areas on log2(count / size_factor + 1)
```

**`direction="both"`** tests both areas in one pass with shared permutations,
and BH covers every tested gene-direction pair (2 × genes hypotheses); a gene
can be called up, down, or both. **`direction="up"`** or **`"down"`** tests one
direction; BH covers the tested genes and the other direction is not computed. Declaring one direction
halves the family and saves the permutations the other direction would use.
Do not run "both" and then report one direction's q-values as if it had been
declared.

**`scale="linear"`** integrates normalized counts. A subset 100× above normal
dominates the area, which is what makes linear areas sensitive to subsets.
**`scale="log"`** integrates log2 expression, which compresses large subsets
and weights relative changes. Descriptions and plots use log2 on either scale.

The linear scale is asymmetric. An increase is unbounded, but a decrease can
remove at most the control level, and a large raised subset widens the
permutation spread of both areas. So the lowered bulk beside a raised subset
is rarely called down on the linear scale: the confirmation called 2 of 960
such genes down, and in the development comparison linear areas called 7 of 80 such
genes down where log areas called 56. Whole-cohort decreases are detected on
either scale. When decreases, or the bulk beside a subset, are the question,
declare `scale="log"`.

## Permutations and resolution

Each gene and direction stops at its 20th permutation at least as extreme as
the observed area, giving p = 20 / draws, or finishes the cap B and gives
p = (exceedances + 1) / (B + 1). The default B = 999,999 resolves p down to
10⁻⁶. BH's first discovery among m hypotheses needs p ≤ 0.05 / m, so the
default can make an isolated discovery among up to 50,000 hypotheses (25,000
genes with `direction="both"`). The manifest's `minimum_discovery_rank` says
how many genes must sit at the floor before any can pass.

Monte Carlo p-values near a threshold vary with the seed. Fix the seed and cap
before looking, and never rerun borderline genes until they pass. Genes with
strong differences use the whole budget, so runtime grows with the number of
changed genes; a smaller `max_permutations` is fine for exploration.

```python
result = wade.wade(data, cond, strata=batch_ids, max_permutations=999_999, seed=1)
```

`strata` restricts shuffles to exchangeable blocks, preserving each block's
case count. A block with only cases or only controls contributes no freedom;
a design with no possible exchange is rejected. `gene_chunk` and `backend`
change memory and speed, never the answer.

## How many controls

Within one gene, the controls limit the evidence for a raised subset. A
random relabelling puts k given samples all among the cases with probability

    C(n_case, k) / C(n_case + n_control, k)

When k cases lie above every control and the other cases are no higher than
the controls (a pure subset), the upward p-value stays near or above this
value, however large the fold change; other cases that are also raised add
evidence beyond it. Cases a pure subset needs above every control to reach a
given p-value:

| Cases : controls | p ≤ 10⁻² | p ≤ 10⁻⁴ | p ≤ 2.5 × 10⁻⁶ |
| --- | ---: | ---: | ---: |
| 250 : 250 | 7 | 14 | 19 |
| 50 : 500 | 2 | 4 | 6 |
| 445 : 52 (TCGA prostate) | 40 | 77 | 104 |
| 971 : 112 (TCGA breast) | 42 | 82 | 112 |
| 445 : 302 (prostate + 250 normals) | 9 | 18 | 25 |

This is information, not method: 52 normals with none high cannot rule out a
few percent of high normals. In tumour/normal comparisons where thousands of
genes change, BH thresholds are lenient and ranking by the descriptions
matters more than calling. For recurrent subsets, add controls: a larger
normal reference processed compatibly, or strata that make external normals
exchangeable within blocks.

## Read, plot and export

| Column | Meaning |
| --- | --- |
| `area_up`, `area_down` | Exact areas on the tested scale; `area_up − area_down` is the mean difference |
| `pvalue_up`, `pvalue_down` | Permutation p-values; NaN for an untested direction or gene |
| `padj_up`, `padj_down` | BH q-values in the declared family |
| `bulk_shift` | Case minus control median of log2(count / size factor + 1) |
| `subset_up_fraction` | Share of cases in the learned upward subset |
| `subset_up_log2fc` | That subset's median minus the control median (log2) |
| `subset_down_fraction`, `subset_down_log2fc` | The same for the learned downward subset |
| `status` | `tested`, `filtered` or `all_zero` |

Areas are in the units of the tested scale and are not comparable across genes;
use the log2 descriptions to compare and rank. A negative `bulk_shift` with a
small `subset_up_fraction`, a large `subset_up_log2fc` and a small `padj_up` is
a raised subset above a lowered bulk. Descriptions have no p-values.

### Learned subsets

No subset size is chosen in advance. Each gene's samples are ranked, cases and
controls together, and a walk steps down the ranking. At each step it takes the
fraction of cases above minus the fraction of controls above, divided by the
standard deviation that difference would have if the labels were random. The
cases at or above the walk's peak form the upward subset; the cases at or below
its lowest point form the downward subset. The walk uses only ranks, so the
scale does not change it; ties never straddle a boundary. A subset must lie on
its own side of the controls: an upward subset whose median is not above the
control median is reported as empty, and likewise downward.

- **Read subsets with the q-value.** A walk over noise also has a peak: on
  unchanged genes the upward subset averaged a third of the cases, and was
  empty for only about one gene in six.
- **A raised subset over a lowered or unchanged bulk** is where the walk is
  precise. With 250 cases and 250 controls it recovered planted 10% subsets
  with precision 0.88–1.00 (see [validation](validation.md#learned-subsets)).
- **With few controls** the boundary is uncertain, and a subset over an
  unchanged bulk is over-called (precision about 0.5 with 50 controls).
- **A whole-cohort shift** gives the most-shifted part of the cohort (31–89% of
  cases in the confirmation designs), not every case; `bulk_shift` describes
  the cohort-wide change.
- **A high mode inside a raised majority** gets no boundary of its own: the
  walk compares cases with controls, not cases with each other. In TCGA breast
  tumours ERBB2's upward subset is the 588 of 971 raised above most normals; it
  holds 106 of the 107 amplified tumours and 465 that are not amplified (see
  [validation](validation.md#on-tcga-tumours)).
- The subset is a description of where cases concentrate, not a set of
  individually significant patients or a prevalence estimate.

```python
result.subset("ETV4", "up")              # learned members, log2 boundary, share and fold
result.subset("ERBB2", "up", cutoff=16)  # the cases at or above log2 16, a level read off the plot
result.gene_detail("ETV4")        # exact quantile intervals; .area_up and .area_down
result.plot_gene("ETV4")          # ECDFs and quantile differences on log2
result.plot_volcano(direction="up", label=["ETV1", "ETV4"])
result.plot_patterns(direction="down")
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
- A small `padj_up` means the upward area is larger than label exchange
  explains. It does not say which patients are outliers, that a subset is
  clinically separable, or how prevalent it is.
- Very low counts carry little information on either scale; filter them.
- A pure subset cannot beat the within-gene bound above without borrowing
  information across genes, which this release does not do.
