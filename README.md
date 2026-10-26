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

Every normalization ends as one size factor per sample: RLE on all genes, RLE
on internal control genes, or factors fitted elsewhere. WADE divides counts by
the size factors for the effects. Before both tests it thins every sample
binomially until all size factors equal the smallest.

- A difference in depth between the groups therefore cannot pass for a
  difference in expression, or for a subset.
- Genes with too few reads to trust are washed out rather than called.
- Both are permutation tests. The thinning is exact when each sample's
  reads are Poisson-distributed around its expression level; expression
  itself may follow any distribution.
- With strata (study, site, batch), labels are exchanged only within each
  stratum, and each stratum is thinned to its own smallest size factor.

## Install

```sh
pip install "wade-rnaseq[io,plot]"
```

The distribution is `wade-rnaseq`; the import is `wade`. The core needs only
NumPy. The `io` extra adds Polars, for tables and written results; `plot` adds
matplotlib and plotly. Published wheels include the native subset walk.

From source, which requires Rust:

```sh
git clone https://github.com/mkiyer/wade
cd wade
python -m pip install -e ".[io,plot]"
```

## Quick start

```python
import wade

# Raw counts: genes × samples. 1 = case, 0 = control.
norm = wade.fit_normalization(counts)                            # one size factor per sample (RLE)
# norm = wade.fit_normalization(counts, reference_genes=controls)   # or RLE on internal control genes
keep = wade.filter_counts(counts, min_count=3, min_samples=2)
result = wade.wade(counts, condition, normalization=norm, test_genes=keep)
# result = wade.wade(counts, condition, normalization=norm, strata=study)   # merged studies
table = result.to_frame().sort("area_padj")     # needs Polars
result.subset("ETV4")                            # the cases departing from a common fold
result.rare_subsets(n=25)                        # the rarest subsets, with the genes that share their members
result.sample_load()                             # which cases carry the rarest subsets
result.plot_gene("ETV4")
```

The [manual](https://github.com/mkiyer/wade/blob/main/docs/manual.md) walks
through the workflow and explains every setting and column. The
[demo notebook](https://github.com/mkiyer/wade/blob/main/notebooks/demo.qmd)
(a Quarto source) runs it on simulated counts.

| Columns | Meaning |
| --- | --- |
| `area`, `area_fraction`, `log2fc` | The difference in means, in reads per sample; its share of the total area; the fold of the means |
| `area_pvalue`, `area_padj` | The area test, on depth-equalized counts, and its BH q-value |
| `subset_side`, `subset_fraction`, `subset_log2fc` | `"up"` (above the rest of the cases), `"down"` or `"none"`; the subset's share of the cases; its fold over what the common fold predicts |
| `subset_pvalue`, `subset_padj` | The test that every case carries one common fold, and its q-value |
| `subset_rarity` | The share of genes whose subset is at least as rare: a size-matched rank that puts rare, strong subsets first |

Each test has its own BH family over the tested genes. A subset describes
where cases depart from a common fold. It is not a set of individually
significant patients, and noise also has a largest gap, so read it with its
q-value.

## Choose before you look

Set these before seeing any result.

| Setting | Default | Meaning |
| --- | --- | --- |
| `direction` | `"both"` | Both tests two-sided. `"up"` asks whether cases are higher: a larger area, and a subset above the rest of the cases. `"down"` asks the reverse |
| `relative_error` | `0.2` | The precision of every p-value |
| `pvalues` | `"auto"` | The area test's p-value is a saddlepoint approximation to its exact permutation distribution, shuffled where the approximation's error estimate exceeds `relative_error`. `"permutation"` shuffles every p-value. The subset test is always shuffled |
| `fdr` | `0.05` | The false discovery rate at which BH decides. Shuffles stop once each gene's decision is settled; the decisions are exactly those of finished runs. Rejected genes report a p-value bound. `None` finishes every run |
| `max_permutations` | `999_999` | The shuffle budget: a shuffled p-value can reach 10⁻⁶, so an isolated discovery stays possible among 50,000 genes |
| `seed` | `1` | Seeds the jitter, the thinning and the shuffles. Subset calls near the threshold move with the thinning, which the test averages over four draws ([validation](https://github.com/mkiyer/wade/blob/main/docs/validation.md#the-subset-test-under-the-count-model)) |
| `strata` | `None` | One label per sample (study, site, batch), or a list of factors. Labels are exchanged only within strata; strata with one group are left out, with a warning |

## Limits

- **The subset test is validated where the lower group averages at least 32
  reads per sample.** Below that its p-values can be anti-conservative when
  every case carries a fold, and `wade()` warns how many tested genes fall
  there. The area test holds at every expression level
  ([validation](https://github.com/mkiyer/wade/blob/main/docs/validation.md#the-subset-test-under-the-count-model)).
- **Every sample is thinned until its size factor equals the smallest**, so
  remove failed or unusually shallow libraries first.
  `result.params["depth_equalized"]` is the share of its reads the median
  sample kept: 29.5% on TCGA prostate
  ([validation](https://github.com/mkiyer/wade/blob/main/docs/validation.md#the-area-test)).
- **Unbalanced designs cost power.** In simulation at 500 cases v 50
  controls, the area test called 9% of genes raised 1.5× in every case,
  against 48% for a Wilcoxon test. Raised subsets of a tenth of the cases
  were called in 1–2%: 0.67% of labellings leave all of those cases out of
  the controls, so their p-values sit near that chance, too high for a
  transcriptome-wide cutoff. Such subsets are still found, described and
  ranked by rarity
  ([validation](https://github.com/mkiyer/wade/blob/main/docs/validation.md#confirmation-protocol-version-7)).
- **Samples must be exchangeable,** across the cohort or within the strata
  you pass. Strata cannot separate a study effect from a condition that
  coincides with it, or bring in external controls. Paired designs are not
  validated ([validation](https://github.com/mkiyer/wade/blob/main/docs/validation.md#strata)).

The [manual](https://github.com/mkiyer/wade/blob/main/docs/manual.md#limits)
lists every limit.

## Evidence

The evidence comes from an exact count-model reference built from 445 TCGA
prostate tumours, split at random into two groups and thinned binomially to
known depths and folds, and from a confirmation of the finished build,
predeclared and run on fresh seeds.

- **The area test held its level in all 50 null cells**
  ([area test](https://github.com/mkiyer/wade/blob/main/docs/validation.md#the-area-test)).
  - The cells span four designs (222:223, 40:405, 405:40 and 20:20), depths
    down to 1/64 of the original libraries, and depth gaps between the
    groups of up to 2×. At p ≤ 0.01 it rejected 0.0–1.6% of null genes in
    every expression bin.
  - Without depth equalization the same test rejected up to 95.6% of the
    lowest-count null genes when one group was sequenced half as deep.
  - Equalization costs power at low counts: at a 1.5-fold difference, a
    median 0.1 points at 128 or more reads per sample, 0.8 at 32–128, more
    below.
- **The subset test held its level at 32 reads per sample and above**
  across folds from 0.5 to 4, depth gaps of 2× either way and three designs
  ([subset test](https://github.com/mkiyer/wade/blob/main/docs/validation.md#the-subset-test-under-the-count-model)).
  - Its worst share at p ≤ 0.01 was 1.78%, where an exact test's worst of
    the 48 cells is about 1.8%.
  - On TCGA prostate, 445 tumours v 52 normals, the subsets found for ERG,
    ETV1, ETV4 and FLI1 held no fusion-negative tumour and 89–100% of the
    fusion-positive ones, annotations WADE never saw
    ([TCGA](https://github.com/mkiyer/wade/blob/main/docs/validation.md#on-tcga-tumours)).
    With 52 controls the small subsets sit near the threshold: ETV1's is
    called, and ETV4's in some runs and not others (q = 0.031–0.054).
- **Strata hold the level where study effects confound unstratified tests**
  ([strata](https://github.com/mkiyer/wade/blob/main/docs/validation.md#strata)).
  TCGA and GTEx prostate normals, relabelled with case shares that differ by
  study, were called at p ≤ 0.01 by the area test in 42.5% of genes
  unstratified and 0.78% stratified. The subset test, at 32 reads and above,
  called 29.3–30.9% unstratified and 0.74–0.76% stratified.
- **The confirmation passed all 27 of its checks** (protocol version 7,
  frozen before any of its data existed;
  [confirmation](https://github.com/mkiyer/wade/blob/main/docs/validation.md#confirmation-protocol-version-7)).
  - Over its per-gene null arms the tests rejected 4.93–5.05% at p ≤ 0.05
    and 0.079–0.115% at 0.001.
  - In 480 simulated transcriptomes with no change, the area test made any
    discovery at q ≤ 0.05 in 5.6% and the subset test in 1.5%. In 60 with
    planted changes, their false discovery proportions were 4.8% and 2.2%.

Stopping at the decision keeps whole transcriptomes fast: the TCGA breast
contrast, 971 tumours v 112 normals over 25,328 genes, ran in 158 seconds on a
shared laptop
([stopping](https://github.com/mkiyer/wade/blob/main/docs/validation.md#decision-based-stopping)).
[Validation](https://github.com/mkiyer/wade/blob/main/docs/validation.md)
gives every number with its source.

[Manual](https://github.com/mkiyer/wade/blob/main/docs/manual.md) ·
[Method](https://github.com/mkiyer/wade/blob/main/docs/method.md) ·
[Validation](https://github.com/mkiyer/wade/blob/main/docs/validation.md) ·
[Changes](https://github.com/mkiyer/wade/blob/main/docs/changes-0.2.0.md) ·
[Research ideas](https://github.com/mkiyer/wade/blob/main/docs/research-ideas.md) ·
[Contributing](https://github.com/mkiyer/wade/blob/main/CONTRIBUTING.md) ·
notebooks as Quarto sources:
[demo](https://github.com/mkiyer/wade/blob/main/notebooks/demo.qmd),
[prostate](https://github.com/mkiyer/wade/blob/main/notebooks/examples.qmd),
[breast](https://github.com/mkiyer/wade/blob/main/notebooks/examples-breast.qmd)
