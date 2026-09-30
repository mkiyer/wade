# WADE — Wasserstein Area Differential Expression

WADE finds genes that differ between cases and controls in RNA-seq cohorts,
and is built for finding and ranking biomarkers: the gene that matters may be
raised across the tumours, or switched on in a tenth of them while the rest
match normal. For every gene it reports two things and tests each:

- **The area**: the signed area between the case and control quantile
  functions of normalized counts. It is exactly the difference in group means,
  in reads per sample, so it grows with expression. Its share of the total
  area under both curves, the **area fraction**, lies between −1 and 1 and
  reads the same at every expression level.
- **The subset**: the cases that depart from the one fold every case would
  share. WADE asks whether the cases are more unequal than the controls,
  finds where the cases' Lorenz curve falls furthest below the controls', and
  reports the smaller side of that split, above or below the rest of the
  cases.

The area test runs on counts thinned to the shallowest sample's sequencing
depth. A difference in depth between the groups therefore cannot pass for a
difference in expression, and genes with too few reads to trust are washed
out rather than called. Both tests use label permutation, so no count
distribution is assumed.

```python
import wade

# Raw counts: genes × samples. 1 = case, 0 = control.
result = wade.wade(counts, condition)
table = result.to_frame().sort("area_padj")     # optional Polars dependency
result.subset("ETV4")                            # the cases departing from a common fold
result.plot_gene("ETV4")
```

## Choose before you look

| Setting | Default | Meaning |
| --- | --- | --- |
| `direction` | `"both"` | Both tests two-sided. `"up"` asks whether cases are higher: a larger area, and a subset above the rest of the cases. `"down"` asks the reverse |
| `relative_error` | `0.2` | The precision of every p-value |
| `pvalues` | `"auto"` | Computes the area test exactly by a saddlepoint, and shuffles where its error estimate exceeds `relative_error`. `"permutation"` shuffles every p-value |
| `max_permutations` | `999_999` | The shuffle budget: a shuffled p-value can reach 10⁻⁶, so an isolated discovery stays possible among 50,000 genes |
| `seed` | `1` | Seeds the jitter, the thinning and the shuffles |

## Results

| Column | Meaning |
| --- | --- |
| `case_mean`, `control_mean` | Mean normalized count per group: reads per sample at the cohort's typical depth |
| `area` | `case_mean − control_mean` |
| `area_fraction` | `area / (case_mean + control_mean)`: 1/3 is a doubling, −1/3 a halving |
| `area_pvalue`, `area_padj` | The area's permutation p-value, on depth-equalized counts, and its BH q-value |
| `subset_side` | `"up"` (above the rest of the cases), `"down"` or `"none"` |
| `subset_fraction` | The subset's share of the cases |
| `subset_log2fc` | Its fold over what the common fold predicts for it |
| `subset_pvalue`, `subset_padj` | The test that every case carries one common fold, and its q-value |
| `alignment_log2fc`, `lorenz_gap` | The common fold, and the subset test's statistic |
| `status` | `tested`, `filtered` or `all_zero` |

Each test has its own BH family over the tested genes. A subset describes
where cases depart from a common fold. It is not a set of individually
significant patients, and noise also has a largest gap, so read it with its
q-value. The [manual](docs/manual.md) explains every setting and column.

## Limits

- **The subset test is validated only for well-expressed genes:** where the
  sparser group averages at least 128 reads per sample. Below that its
  p-values can be anti-conservative, and `wade()` warns how many tested genes
  fall there. The area test holds at every expression level.
- **Every sample is thinned to the shallowest one**, so remove failed or
  unusually shallow libraries first. `result.params["depth_equalized"]` is the
  share of its reads the median sample kept.
- **RLE size factors absorb part of any change shared by many genes.** When a
  tenth of the genes double in one group, unchanged genes look changed, in
  WADE as in any DE tool that relies on size factors. Balanced changes are
  harmless.
- **The area test's null is one distribution for both groups.** Genes whose
  groups differ only in spread can be called too; their area stays near zero.
- **Samples must be independent.** Strata and paired designs are not
  supported yet.

## Evidence

The evidence comes from an exact count-model reference built from 445 TCGA
prostate tumours. The tumours are split at random into two groups and thinned
binomially to known depths and folds.

- **The area test held its level in all 50 null cells.**
  - The cells span four designs (222:223, 40:405, 405:40 and 20:20), depths
    down to 1/64 of the original libraries, and depth gaps between the
    groups of up to 2×.
  - At p ≤ 0.01 it rejected 0.0–1.6% of null genes in every expression bin.
  - The same test without depth equalization rejected up to 96% of the
    lowest-count null genes when one group was sequenced half as deep.
- **Equalization costs power only at low counts.** Against the unthinned test
  at a 1.5-fold difference, the median loss was:
  - 0.1 points at 128 or more reads per sample (at most 0.8);
  - 0.8 points at 32–128 reads;
  - more below 32 reads, by design.
- **On TCGA prostate with its own labels** it called 14,266 genes at
  q ≤ 0.05, against 14,770 without equalization. Two thinning seeds agreed on
  97% of the calls at 128 or more reads per sample.
- **The subset test** held its level at or above 128 reads per sample across
  folds from 0.5 to 4 and every depth arm measured (worst 1.4% at p ≤ 0.01).
  In simulation it recovered a tenth of the cases raised 8× or more, on the
  right side, with precision 0.96–1.00.
- **On TCGA prostate** the subsets found for ERG, ETV1, ETV4 and FLI1 held no
  fusion-negative tumour and 89–100% of the fusion-positive ones, annotations
  WADE never saw. With 52 controls, the small ETV1 and ETV4 subsets are called
  near the threshold (q = 0.032 and 0.044).

Details, the measured limits and how to reproduce them are in
[validation](docs/validation.md).

## Install

```sh
python -m pip install -e ".[io,plot]"
```

Building from source requires Rust. The core needs only NumPy; Polars and the
plotting libraries are optional. Published wheels include the native subset walk.

[Manual](docs/manual.md) · [Method](docs/method.md) · [Validation](docs/validation.md) ·
[Changes](docs/changes-0.2.0.md) · [Research ideas](docs/research-ideas.md) ·
[Demo](notebooks/demo.qmd) · [Prostate](notebooks/examples.qmd) ·
[Breast](notebooks/examples-breast.qmd) · [Contributing](CONTRIBUTING.md)
