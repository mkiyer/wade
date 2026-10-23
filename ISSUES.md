# Open issues

## Before releasing 0.2.0

1. **Confirm the 30 September 2026 build (protocol version 7).**
   - *The build:*
     - the area test on counts thinned until every sample's size factor
       equals the smallest;
     - the area and the area fraction as effects;
     - the subset test with its validated range of 128 reads per sample
       and a warning below it;
     - no log2 scale and no `subsets="either"`.
   - *Measured so far* under the count-model reference, on TCGA prostate with
     its own labels, and on the recovery simulations
     ([docs/validation.md](docs/validation.md)).
   - *Pending:* the fresh-seed confirmation, protocol version 7
     ([plan](docs/subset-null-plan.md), section 4, step 1):
     - freeze first;
     - predeclare quantities, tolerances and simulation sizes;
     - run complete-null and partial-null matrices;
     - validate at the thresholds users meet.
     `research/confirm.py` runs the build. Bump the protocol file's version
     and seed root before running it.
2. **Review.** The author approves:
   - the rendered demo, prostate and breast notebooks. The demo and prostate
     notebooks were re-rendered for the build on 30 September 2026; the breast
     notebook still needs rerunning (it took 89 minutes before);
   - the documents' figures, tables and claims.

   Record the TCGA count-pipeline provenance: workflow, genome and
   annotation versions, and row order.
3. **Publish.** Tag `v0.2.0` on `main`; CI tests every platform and publishes.

## Method research

The ideas and their evidence are in
[docs/research-ideas.md](docs/research-ideas.md).

- **The subset test below 128 reads per sample.**
  - A valid test there must include the fold's estimation in its null. The
    principled route is a mixture fit with a bootstrap, candidate A (ideas,
    section 4).
  - The count-exact pairwise scores are a valid DE statistic at every level,
    but blind to magnitude (ideas, section 2.1).
- **The subset test's small-sample residual** on bimodal nulls (ideas,
  section 4.9).
- **Normalization under composition change.**
  - When a tenth of the genes double in one group, RLE moves about 5% and the
    area test calls 20% of unchanged genes at p ≤ 0.05.
  - Candidates: a diagnostic of how asymmetric the RLE ratios are between
    the groups; trimmed or reference-gene normalization (ideas, section 6).
- **The thinning draw.** The area test's calls below about 32 reads per
  sample move with the seed: on TCGA prostate, 80–86% of the genes one seed
  calls there are called by another, against 97% at 128 reads or more.
  Whether to offer a stability check is open.
- **Transcriptome-wide rarity.** A `subset_rarity` column: the share of genes
  whose k most extreme cases depart at least as far from their own common
  fold, k the subset's size (a conformal rank; no shuffles).
  - A prototype on prostate, before the Lorenz statistic, ranked ETV4 10th,
    ETV1 36th and ETV5 46th of 24,246 genes, and the most extreme singleton
    (MC4R) 2nd.
  - Add a per-sample count of rare subsets to flag contaminated or artefact
    libraries: one prostate tumour holds six of the forty most extreme
    singletons.
  - Conditioning on abundance helps known biology, but needs a model, not a
    window.
- **A dataset-wide model of "expected".** A model of how a gene's
  distribution depends on its mean (mean–variance and shape, fitted across
  genes). It would give rarity a principled reference, and supply tails that
  a few controls cannot show.
- **Performance at 20,000–80,000 samples.**
  - The area test is computed by saddlepoint for nearly every gene, and the
    thinning is one binomial draw per count.
  - At saturation the subset test runs the full shuffle budget for every
    strong gene. Stop each gene when its BH decision is settled (Gandy and
    Hahn) rather than when its p-value is precise, vectorize the walk, and
    stream gene chunks from Arrow.
- **Covariates, strata and pairing.** Passing `strata` is an error until a
  construction is validated. `notebooks/rna100k.qmd`, which blocks the
  RNA100K compendium by study, waits for it.

## After 0.2.0

- **Subset p-values below the shuffle floor.** Multilevel splitting was
  prototyped on the native walk (2026-09-28, outside the repository) and is not
  safe as built: on outlier genes such as ETV4 its local swap moves cannot cross
  between the tail's modes, so some runs collapsed by many decades, and larger
  populations shrank but did not remove a low bias, at a cost near brute force.
  Ties were not the cause, so jitter would not help. It would only rank the
  strongest subsets, never change a call at the 10⁻⁶ floor.
- **A stratified saddlepoint,** once strata are supported. The double
  saddlepoint extends to restricted exchange with one count constraint per
  stratum; worth doing if stratified designs become common.
- **Two subsets in one gene.** One walk makes one split, so a raised and a
  lowered subset in the same gene share it. A second split of each side
  (binary segmentation) would report both.
- **More controls.** Methods for pooling external normals (e.g. GTEx) with
  batch strata, and guidance on when that is exchangeable.
- **Native comparators.** Rerun DESeq2/edgeR and a classic outlier statistic
  (COPA or outlier sum) on the confirmation matrices for a publication.
- **Sparse assays.** Single-cell, spatial and cell-free RNA need their own
  exchangeability design and validation.
- **Measurement uncertainty.** Per-sample count kernels were explored and
  deferred; the derivations are in history at `f07d5f9`.
