# WADE — Wasserstein Area Differential Expression

WADE finds genes whose expression in cases is raised above, or lowered below,
a control cohort, across the whole cohort or in a subset of patients. It is
built for biomarker discovery in large RNA-seq cohorts: an ETV1-like gene,
strongly overexpressed in 10% of tumours while the rest sit below normal, is
an upward discovery, not a downregulated gene.

For each gene WADE compares the case and control quantile functions and
measures two areas:

- **`area_up`** integrates the case-minus-control quantile differences where
  cases are higher;
- **`area_down`** integrates them where cases are lower.

Their difference is the mean difference. A mean (or rank) test lets a lowered
bulk cancel a raised subset; the two areas keep them apart. Each area is tested
by label permutation, so no count distribution is assumed.

```python
import wade

# Raw counts: genes × samples. 1 = case, 0 = control.
result = wade.wade(counts, condition)          # direction="both", scale="linear"
table = result.to_frame().sort("padj_up")      # optional Polars dependency
result.drivers("ETV1", "up")                   # cases above every control
result.plot_gene("ETV1")
```

## Choose the hypothesis before you look

| `direction=` | Question | False discovery family |
| --- | --- | --- |
| `"both"` (default) | Discovery: which genes go up, which go down? Both areas in one pass. | One BH family over every tested gene-direction pair |
| `"up"` or `"down"` | A declared one-directional hypothesis | BH over the tested genes; the other direction is not tested |

`scale="linear"` (default) tests `count / size_factor`, which keeps large
subsets large; `scale="log"` tests `log2(count / size_factor + 1)`. In our
benchmarks linear areas matched log areas for whole-cohort shifts and were far
more sensitive to raised subsets; a decrease beside a raised subset needs the
log scale ([manual](docs/manual.md#direction-and-scale)). Choose both settings
before running.

## Results

| Column | Meaning |
| --- | --- |
| `area_up`, `area_down` | Exact areas on the tested scale (normalized counts by default) |
| `pvalue_up`, `pvalue_down` | Monte Carlo permutation p-values, one per tested direction |
| `padj_up`, `padj_down` | BH q-values in the family set by `direction` |
| `bulk_shift` | Case minus control median of log2 expression |
| `upper_tail`, `lower_tail` | Top/bottom decile mean differences minus `bulk_shift` (log2) |
| `status` | `tested`, `filtered` or `all_zero` |

A raised subset above a lowered bulk reads as a small `padj_up`, a negative
`bulk_shift` and a large positive `upper_tail`. `result.drivers(gene, "up")`
lists the cases above every control. Descriptions carry no p-values.

## What limits discovery: the controls

A random relabelling puts k given samples all among the cases with
probability C(n_case, k) / C(n_case + n_control, k). So when k cases lie above
every control and the other cases are no higher than the controls (a pure
subset, like ETV1 or ETV4 in prostate cancer), the upward p-value stays near or
above that value, whatever the fold change:

| Cases : controls | 18 cases above all controls | 36 cases above all controls |
| --- | ---: | ---: |
| 445 : 52 (TCGA prostate) | 0.13 | 0.016 |
| 250 : 250 | 2.8 × 10⁻⁶ | 3.7 × 10⁻¹² |

Other cases that are also raised add evidence beyond it. Subset discovery
needs enough controls; the [manual](docs/manual.md#how-many-controls) gives the
arithmetic for your design.

## Evidence

On 120 fresh simulated transcriptomes (16,384 genes each), the default run kept
the mean false discovery proportion at 2.5–5.4% for q ≤ 0.05. With 250 cases
and 250 controls it found 95–100% of genes (baseline mean ≥ 5 counts) where
10% of cases were raised 32–1,024× above a halved bulk; a one-sided Wilcoxon
test found none of them upregulated and called most of them down. Details,
limits and how to reproduce them are in [validation](docs/validation.md).

## Install

```sh
python -m pip install -e ".[io,plot]"
```

Building from source requires Rust. The core needs only NumPy; Polars and the
plotting libraries are optional. Published wheels include the native kernel,
and a readable NumPy implementation is the correctness reference and fallback.

[Manual](docs/manual.md) · [Method](docs/method.md) · [Validation](docs/validation.md) ·
[Changes](docs/changes-0.2.0.md) · [Demo](notebooks/demo.qmd) · [Prostate](notebooks/examples.qmd) ·
[Breast](notebooks/examples-breast.qmd) · [Contributing](CONTRIBUTING.md)
