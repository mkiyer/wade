# Open issues

## Before releasing 0.2.0

1. **Notebooks.** Update the demo, prostate and breast notebooks to the
   directional API and render them. Record the TCGA count-pipeline provenance
   (workflow, genome and annotation versions, row order).
2. **Review.** The author approves the rendered figures, tables and claims.
3. **Publish.** Tag `v0.2.0` on `main`; CI tests every platform and publishes.

## After 0.2.0

- **Moderate subsets above a lowered bulk (4–16×).** A calibrated tail
  statistic (the largest standardized excess of cases among the top k samples,
  with the cutoff search inside every permutation) found more of these than
  the linear area, but fewer whole-cohort shifts; see [validation](docs/validation.md).
  Worth revisiting if that range matters for a real application.
- **Beating the control floor.** Judging how unusual an amplitude is across
  the transcriptome needs a calibrated cross-gene model (the "rarity" route).
- **More controls.** Methods for pooling external normals (e.g. GTEx) with
  batch strata, and guidance on when that is exchangeable.
- **Native comparators.** Rerun DESeq2/edgeR and a classic outlier statistic
  (COPA or outlier sum) on the confirmation matrices for a publication.
- **Sparse assays.** Single-cell, spatial and cell-free RNA need their own
  exchangeability design and validation.
- **Measurement uncertainty.** Per-sample count kernels were explored and
  deferred; the derivations are in history at `f07d5f9`.
