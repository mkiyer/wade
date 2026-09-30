# WADE — Wasserstein Area Differential Expression

WADE finds genes that differ between cases and controls, across the whole
cohort or in a subset of cases. It is built for biomarker discovery in RNA-seq
cohorts, where the gene that matters may be switched on in a tenth of tumours
while the rest do something else entirely. For every gene it reports two
things and tests each:

- **The mean shift**: the signed area between the case and control quantile
  functions, which is exactly the difference in means.
- **The subset**: the cases that depart from that mean shift, above it or below
  it. If every case carried the mean shift alike, each case quantile would sit at
  a predicted level; WADE walks up the cases and splits them where their running
  departure from those predictions is lowest. The smaller side is the subset.

A gene can shift one way as a whole and have a subset that departs the other
way: a gene overexpressed in 10% of tumours while the rest fall below normal
reports a lowered mean and a raised subset. Both tests use label permutation,
so no count distribution is assumed.

```python
import wade

# Raw counts: genes × samples. 1 = case, 0 = control.
result = wade.wade(counts, condition)
table = result.to_frame().sort("subset_padj")   # optional Polars dependency
result.subset("ETV1")                            # the cases departing from its mean shift
result.plot_gene("ETV1")
```

## Choose before you look

| Setting | Default | Meaning |
| --- | --- | --- |
| `direction` | `"both"` | Both tests two-sided. `"up"` asks whether cases are higher, as a whole or in a subset above the mean shift; `"down"` the reverse |
| `scale` | `"linear"` | Tests `count / size_factor`, sensitive to raised subsets. `"log2"` tests `log2(count / size_factor + 1)`, where a halving weighs as much as a doubling |
| `relative_error` | `0.2` | The precision of every p-value |
| `pvalues` | `"auto"` | Computes the mean test exactly by a saddlepoint, and shuffles where its error estimate exceeds `relative_error`. `"permutation"` shuffles every p-value |
| `max_permutations` | `999_999` | The shuffle budget: a shuffled p-value can reach 10⁻⁶, so an isolated discovery stays possible among 50,000 genes |

## Results

| Column | Meaning |
| --- | --- |
| `mean_log2fc` | The mean shift as a log2 fold of the group means (count + 1 on the linear scale) |
| `mean_pvalue`, `mean_padj` | Its permutation p-value and BH q-value |
| `subset_fraction` | The subset's share of the cases |
| `subset_log2fc` | Its fold over the mean shift's prediction: positive above, negative below |
| `subset_pvalue`, `subset_padj` | The test that every case carries the mean shift, and its q-value |
| `status` | `tested`, `filtered` or `all_zero` |

Each test has its own BH family over the tested genes. A subset is a
description of where cases depart from the mean shift, not a set of
individually significant patients; a walk over noise also has a lowest point,
so read it with its q-value. The [manual](docs/manual.md) explains every
setting and column.

## Evidence

On 120 fresh simulated transcriptomes (16,384 genes each), the default run kept
the mean false discovery proportion at 3.3–5.1% for mean shifts and 3.6–7.0%
for subsets at q ≤ 0.05 (a two-sided Wilcoxon test: 4.5–5.1%). With 250 cases
and 250 controls, the subset test found every gene (baseline mean ≥ 5 counts)
where a tenth of the cases rose 8–1,024× above a halved majority, genes that
Wilcoxon called down. Across designs from 3 v 52 to 445 v 10, the computed
mean-test p-value was within 1.5× of the exact permutation p-value for 99.7% of
the genes it kept. On TCGA prostate tumours, the subsets found for ERG, ETV1,
ETV4 and FLI1 held no fusion-negative tumour and 89–100% of the fusion-positive
ones, annotations WADE never saw. Details, limits and how to reproduce them are
in [validation](docs/validation.md).

## Install

```sh
python -m pip install -e ".[io,plot]"
```

Building from source requires Rust. The core needs only NumPy; Polars and the
plotting libraries are optional. Published wheels include the native subset walk.

[Manual](docs/manual.md) · [Method](docs/method.md) · [Validation](docs/validation.md) ·
[Changes](docs/changes-0.2.0.md) · [Demo](notebooks/demo.qmd) · [Prostate](notebooks/examples.qmd) ·
[Breast](notebooks/examples-breast.qmd) · [Contributing](CONTRIBUTING.md)
