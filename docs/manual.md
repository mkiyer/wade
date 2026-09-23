# WADE 0.2.0 manual

WADE has one inferential contract: test whether a gene's expression distribution
differs, then describe its bulk shift and tails. A gene's q-value does not
separately establish the direction of its bulk shift or significance of one tail.

## Before running

Fix the contrast, biological population and independent sampling unit. Retain
one independent observation per unit, or use scientifically justified restricted
permutations. Cells from one donor are not independent patients. Prespecify
technical sample exclusions; do not remove cases because they weaken a result.
Expression-derived subtypes and RNA-derived fusion labels can induce selection;
unknown annotations are not negative labels. A target-gene-selected subgroup
is exploratory. See the [operating procedure](../archive/development-20260923/docs/operating-procedure-0.2.0.md).

Inspect `wade.library_qc(counts)` for depth, detection and concentration.
Normalization does not make batch-confounded groups exchangeable or remove
depth-dependent variance. Sparse assays need their own design and calibration.

## Input and a first analysis

```python
import wade

# NumPy arrays have genes in rows and samples in columns.
result = wade.wade(counts, [1, 1, 1, 0, 0, 0],
                   gene_names=gene_ids, sample_names=sample_ids)
```

At least two samples per group are required; that minimum is an input guard,
not a claim of adequate power. Labels, count values and sample alignment are
checked. Negative or missing counts are errors. Fractional nonnegative estimates
are accepted; raw gene counts are the validated bulk input.

Pandas and Polars frames can supply labels. Declare sample columns if the frame
also contains numeric gene metadata. Sparse matrices are accepted through their
`toarray()` method and materialized; this is not a sparse-memory inference engine.

```python
data = wade.as_counts(frame, id_column="gene_id", sample_columns=sample_ids)
cond = wade.condition(sample_sheet, key="sample_id", column="group",
                      case="tumour", control="normal")
result = wade.wade(data, cond)
```

Alignment is strict: missing or unmatched sample IDs raise. `drop_other=True`
on `condition` records other biological levels, but does not silently remove
columns from a subsequent matrix.

## Normalization once, filtering separately

The default is all-positive-reference RLE. It estimates one exposure factor per
sample using median log ratios; factors have geometric mean one on the fit
cohort. A broad, mostly stable reference population is the identifying assumption.
Tiny panels or a mostly shifted transcriptome do not establish that assumption.

```python
norm = wade.fit_normalization(data)  # ordinary bulk RLE
norm = wade.fit_normalization(data, reference_genes=validated_control_ids)
```

Internal controls may also be tested. `norm.reference_genes` records nominated
IDs, and `norm.used_reference_genes` records those actually used. RLE discards
any candidate with a zero in a fit sample; no usable reference produces an error.
WADE does not automatically nominate or certify low-variance housekeeping genes.

Initial sample/gene QC and reference selection can precede the fit. Subsequent
test filters use the existing factors:

```python
keep = wade.filter_counts(data, min_count=3, min_samples=2)
keep &= custom_prespecified_mask
result = wade.wade(data, cond, normalization=norm, test_genes=keep)
```

These example thresholds are not universal recommendations. `min_total` adds
a pooled total-count threshold. Custom Boolean masks or explicit gene IDs are
also accepted by `test_genes`. A valid pooled filter is unchanged by the allowed
label shuffles. Group-specific filtering or selecting thresholds by q-values
requires separate justification. The default only removes all-zero genes.
Excluded genes retain descriptors but have NaN test statistics/p-values/q-values
and zero permutation counts. Status distinguishes `all_zero`, `filtered`, `tested`.

Externally fitted size factors, including justified deconvolution factors, can
be imported explicitly:

```python
norm = wade.Normalization.from_size_factors(data, factors_by_sample,
                                           source="external fitted exposure")
result = wade.wade(data, cond, normalization=norm)
```

Factors must be positive and include depth. For edgeR composition corrections,
multiply by library totals first. Import centers once; selecting samples does
not refit or recenter. External sparse normalization is supported as an input,
not certified as a general sparse-assay operating procedure.

`fit_normalization(data, method="library")` uses total-count exposures when
comparable composition is defensible. `wade.cpm(data)` and
`wade.tpm(data, gene_lengths)` independently convert each sample to abundance
units; zeros remain zeros. These functions are for abundance reporting and
QC. Pass raw counts, not already transformed CPM/TPM, to the inference driver.
`norm.transform(data)` returns linear `count / size_factor`, not CPM. WADE
uses `log2(count / size_factor + 1)` with an offset of one normalized count.

## Permutations and computational settings

```python
result = wade.wade(data, cond, normalization=norm,
                   strata=batch_ids, max_permutations=499_999, seed=1)
```

Strata must describe legitimate exchangeability blocks. Each shuffle preserves
the case count within every block. Pure-case or pure-control blocks provide no
swapping freedom; a design with no allowed exchange is rejected.

Every gene uses the same shuffle stream. Null-like genes stop at exactly their
20th inclusive exceedance and receive `20 / draws_used`. Others finish the cap
and receive `(exceedances + 1) / (cap + 1)`. For caps below 20 the stopping count
is the cap. The implementation does not stop at a provisional BH decision.
Interrupted unfinished tests have no reportable result.

The default cap of 499,999 permits p-values down to 2e-6. With 20,000 tests,
an isolated q=0.05 discovery needs p<=2.5e-6. A cap of 4,999 cannot resolve it,
although a large set of small p-values may still produce many BH discoveries.
Resolution does not imply precision: Monte Carlo results near a threshold can
change with seed. Decide budgets before reviewing results, and do not rerun
individual borderline genes until they pass.

`gene_chunk` bounds sorting memory and `backend="numpy"` selects the readable
reference. The native kernel is selected automatically when present. These
settings do not change the statistical procedure. Memory includes the input
and retained log-expression matrix, plus bounded sorting/shuffle scratch;
there is no genes-by-permutations null bank. Strong alternatives can use the
entire budget, so runtime grows with the number of altered genes.

`wade_contrast(data, case_sample_ids, control_sample_ids, normalization=norm)`
selects groups by ID and retains the original normalization fit. A method name
instead fits on the selected cohort. Strata must cover the full input matrix.

## Read, plot and export

| Column | Meaning |
| --- | --- |
| `statistic` | Exact unsigned empirical W1, in log-expression units |
| `pvalue`, `padj` | One distribution-test Monte Carlo p-value and family-wide BH q-value |
| `bulk_shift` | Case minus control median log expression |
| `upper_tail` | Upper-decile mean difference, minus the bulk shift |
| `lower_tail` | Lower-decile mean difference, minus the bulk shift |
| `n_permutations`, `n_exceedances` | Actual completed stopping records |
| `status` | Testing eligibility and exclusion reason |

Tail means integrate the empirical quantile function over the top/bottom 10%,
including fractional boundary observations. Descriptions are signed: a positive
upper tail means the upper decile differs more positively than the median.
The same gene may have negative `bulk_shift` and positive `upper_tail`.
This is not an estimated affected fraction or a fusion call.

```python
result.plot_gene("ETV4")
result.plot_patterns(label=["ETV1", "ETV4"])
result.plot_volcano()
result.plot_gene("ETV4", backend="plotly", theme="dark")
frame = result.to_frame()
wade.write_results(result, "results.tsv")
```

Matplotlib returns standard figures; Plotly returns interactive figures.
Themes are `light`, `dark`, and `high-contrast`. No plotting dependency is
imported by a normal `import wade`. Exports support TSV/CSV, Parquet and Arrow
with Polars. The adjacent manifest records preprocessing, references, sample
and gene order, exclusions, seed, strata and all stopping records.

## Scope of evidence

The frozen bulk workflow has whole-transcriptome simulation evidence, including
reference/test overlap, filtering, correlation and imbalance. See the
[full assessment](../archive/development-20260923/docs/transcriptome-workflow-validation.md) for uncertainty and
missed patterns. Limited controls can seriously limit tail sensitivity even
with known exposures. The new method changes the inferential target; old
mean/subset q-values are not aliases or comparable gene-by-gene results.
No superiority to DESeq2 or OUTRIDER follows merely from the choice of W1.
