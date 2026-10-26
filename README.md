# WADE — Wasserstein Area Differential Expression

WADE finds genes that differ between cases and controls in RNA-seq cohorts,
and is built for finding and ranking biomarkers: the gene that matters may be
raised across the tumours, or switched on in a tenth of them while the rest
match normal. For every gene it reports two things and tests each:

- **The area**: the signed area between the case and control quantile
  functions of normalized counts. It is exactly the difference in group means,
  in reads per sample, so it grows with expression.
  - Its share of the total area under both curves, the **area fraction**,
    lies between −1 and 1 and reads the same at every expression level.
  - The ratio of the same two means is the **fold**, `log2fc`.
- **The subset**: the cases that depart from the one fold every case would
  share. WADE asks whether the cases are more unequal than the controls,
  finds where the cases' Lorenz curve falls furthest below the controls', and
  reports the smaller side of that split, above or below the rest of the
  cases.

Every normalization ends as one size factor per sample. That holds for RLE
on all genes, RLE on internal control genes, or factors fitted elsewhere. WADE
divides counts by the size factors for the effects. Before both tests it
thins every sample binomially until all size factors equal the smallest.

- A difference in depth between the groups therefore cannot pass for a
  difference in expression.
- Genes with too few reads to trust are washed out rather than called.
- Both tests use label permutation, so no count distribution is assumed.
- With strata (study, site, batch), labels are exchanged only within each
  stratum, and each stratum is thinned to its own smallest size factor.

```python
import wade

# Raw counts: genes × samples. 1 = case, 0 = control.
norm = wade.fit_normalization(counts)                            # one size factor per sample (RLE)
# norm = wade.fit_normalization(counts, reference_genes=controls)   # or RLE on internal control genes
keep = wade.filter_counts(counts, min_count=3, min_samples=2)
result = wade.wade(counts, condition, normalization=norm, test_genes=keep)
# result = wade.wade(counts, condition, normalization=norm, strata=study)   # merged studies
table = result.to_frame().sort("area_padj")     # optional Polars dependency
result.subset("ETV4")                            # the cases departing from a common fold
result.rare_subsets(n=25)                        # the rarest subsets, with the genes that share their members
result.sample_load()                             # which cases carry the rarest subsets
result.plot_gene("ETV4")
```

The [manual](docs/manual.md#running-wade-on-a-dataset) walks through the
workflow, and the [demo](notebooks/demo.qmd) runs it.

## Choose before you look

| Setting | Default | Meaning |
| --- | --- | --- |
| `direction` | `"both"` | Both tests two-sided. `"up"` asks whether cases are higher: a larger area, and a subset above the rest of the cases. `"down"` asks the reverse |
| `relative_error` | `0.2` | The precision of every p-value |
| `pvalues` | `"auto"` | Computes the area test exactly by a saddlepoint, and shuffles where its error estimate exceeds `relative_error`. `"permutation"` shuffles every p-value |
| `fdr` | `0.05` | The false discovery rate at which BH decides. Shuffles stop once each gene's decision is settled; the decisions are exactly those of finished runs. Rejected genes report a p-value bound. `None` finishes every run |
| `max_permutations` | `999_999` | The shuffle budget: a shuffled p-value can reach 10⁻⁶, so an isolated discovery stays possible among 50,000 genes |
| `seed` | `1` | Seeds the jitter, the thinning and the shuffles |
| `strata` | `None` | One label per sample (study, site, batch), or a list of factors. Labels are exchanged only within strata; strata with one group are left out, with a warning |

## Results

| Column | Meaning |
| --- | --- |
| `case_mean`, `control_mean` | Mean normalized count per group: reads per sample at the cohort's typical depth (with strata, Mantel–Haenszel standardized) |
| `area` | `case_mean − control_mean` |
| `area_fraction` | `area / (case_mean + control_mean)`: 1/3 is a doubling, −1/3 a halving |
| `log2fc` | `log2(case_mean / control_mean)`: the fold of the group means, ±inf where a group has no reads |
| `area_pvalue`, `area_padj` | The area's permutation p-value, on depth-equalized counts, and its BH q-value |
| `subset_side` | `"up"` (above the rest of the cases), `"down"` or `"none"` |
| `subset_fraction` | The subset's share of the cases |
| `subset_log2fc` | Its fold over what the common fold predicts for it |
| `subset_pvalue`, `subset_padj` | The test that every case carries one common fold, and its q-value |
| `subset_rarity` | The share of genes whose subset is at least as rare: a size-matched rank that puts rare, strong subsets first |
| `alignment_log2fc`, `lorenz_gap` | The common fold, and the subset statistic on normalized counts (the test walks it on thinned counts) |
| `status` | `tested`, `filtered` or `all_zero` |

Each test has its own BH family over the tested genes. A subset describes
where cases depart from a common fold. It is not a set of individually
significant patients, and noise also has a largest gap, so read it with its
q-value. The [manual](docs/manual.md) explains every setting and column.

## Limits

- **The subset test is validated where the sparser group averages at least 32
  reads per sample.** Below that its p-values can be anti-conservative when
  every case carries a fold, and `wade()` warns how many tested genes fall
  there. The area test holds at every expression level.
- **The subset test's calls near the threshold depend on its thinning.**
  Thinning keeps the test valid at low counts, but which reads it keeps is
  random, so the test averages four draws. On TCGA prostate two seeds agreed
  on 81% of the genes either called (90% at 128 reads and above, against 81%
  with one draw). The subset's side, share, fold and rarity do not move.
- **Every sample is thinned until its size factor equals the smallest**, so
  remove failed or unusually shallow libraries first.
  `result.params["depth_equalized"]` is the share of its reads the median
  sample kept.
- **RLE size factors absorb part of any change shared by many genes.** When a
  tenth of the genes double in one group, unchanged genes look changed, in
  WADE as in any DE tool that relies on size factors. Balanced changes are
  harmless.
- **The area test's null is one distribution for both groups.** Genes whose
  groups differ only in spread can be called too; their area stays near zero.
- **Unbalanced designs cost power.** With ten cases per control, in
  simulation:
  - the area test called 9% of genes raised 1.5× in every case, where a
    Wilcoxon test called 48%;
  - a subset of a tenth of the cases was rarely called (1–2%). With 50
    controls, 0.67% of random labellings also leave every one of those cases
    out of the controls, so their p-values sit near that chance, too high
    for a transcriptome-wide cutoff. Such subsets are still found and
    described.
- **Samples must be exchangeable,** across the cohort or within the strata
  you pass. Strata cannot separate a study effect from a condition that
  coincides with it, or bring in external controls. Paired designs are not
  validated yet.

## Evidence

The evidence comes from an exact count-model reference built from 445 TCGA
prostate tumours. The tumours are split at random into two groups and thinned
binomially to known depths and folds. A confirmation, predeclared and run on
fresh seeds, then checked the finished build.

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
- **Speed.** Each gene's shuffles stop once its BH decision is settled. The
  decisions are exactly those of finished runs, and the TCGA breast
  transcriptome (25,328 genes × 1,083 samples) ran in 158 seconds on a
  shared laptop; finished runs of the earlier test took 89 minutes.
- **The subset test** held its level at or above 32 reads per sample across
  folds from 0.5 to 4, depth gaps of 2× either way and three designs: its
  worst share at p ≤ 0.01 was 1.8%, which an exact test also reaches by
  chance. Genes that read zero in half the tumours stay near their level
  (at most 1.25% at p ≤ 0.01). In
  simulation it recovered a tenth of the cases raised 8× or more, on the
  right side, with precision 0.96–1.00, and 98–100% of them reached
  p ≤ 0.05.
- **Strata** hold the level where study effects confound unstratified tests.
  TCGA and GTEx prostate normals, relabelled with case shares that differ by
  study, were called at p ≤ 0.01 in 42% (area) and 28% (subset) of genes
  unstratified, against 0.78% for each stratified.
- **On TCGA prostate** the subsets found for ERG, ETV1, ETV4 and FLI1 held no
  fusion-negative tumour and 89–100% of the fusion-positive ones, annotations
  WADE never saw. With 52 controls the small subsets sit near the threshold:
  ETV1's is called, and ETV4's is called in some runs and not others
  (q = 0.031–0.054).
- **The confirmation passed all 27 of its checks** (protocol version 7,
  frozen before any of its data existed).
  - Over 400,000 null genes per test (count-model, TCGA genes under random
    labels, and stratified), the tests rejected 4.93–5.05% at p ≤ 0.05 and
    0.079–0.115% at 0.001.
  - In 480 simulated transcriptomes with no change, the area test made any
    discovery at q ≤ 0.05 in 5.6% and the subset test in 1.5%.
  - In 60 with planted changes, their false discovery proportions were 4.8%
    and 2.2%.

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
