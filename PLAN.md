# WADE 0.2.0 release queue

2026-09-23 — reset after review of the kernel depth-stress study.
The [canonical design and validation plan](docs/kernel-validation-roadmap.md)
is the single source for future scientific work. This file is the release queue.
Older proposed experiments are evidence and options, not parallel active plans.
Uncertainty modeling remains an intended 0.2.0 requirement; no kernel candidate
has yet met the scientific release gates. No production behavior changed in
this reset.

## Workspace cleanup

- [x] Review and separate the public package, kernel research, operational tools
  and completed studies. Preserve a verified source snapshot and move historical
  reports/prototypes/tests to the [development archive](archive/development-20260923/README.md).
- [x] Consolidate the kernel evaluator in [research/](research/README.md), with
  explicit normalization/prior inputs and numerical replay/independent checks.
- [x] Reduce active documentation to eight pages; retain one scientific plan
  and a concise issue register. The agreed research sequence below is unchanged.

## Next: measure the right quantities

- [x] Audit normalization and kernel terminology. All 184 depth-stress states
  record RLE; ten full-matrix refits reproduce saved factors exactly. Both
  Beta candidates' full CDFs are transformed to the same scale as the points.
  See the [audit](.release/0.2.0/design-reset-20260923/normalization-audit.json).
- [x] Consolidate the goals, count-model/kernel distinction, evidence limits,
  effect/subset scorecard and ordered gates in the canonical plan.
- [x] Agree on the sequence and specification, including what “fold change” and “affected
  subset” mean. Prior choice, technical dispersion and null calibration remain
  open; neither another fixed prior nor a fitted population reference is selected.
- [ ] Build the small truth-recovery and failure-replay benchmark on saved data:
  abundance curves/areas, explicit fold contrasts, subset fraction, and patient
  membership, alongside calibration and detection. Check noiseless examples
  first and distinguish identifiable descriptions from latent component estimates.
- [ ] Reproduce relevant historical false positives with the current matched
  point baseline. Separate normalization, raw-noise offsets, depth-associated
  sampling and the retired fitted-bulk subset test. Preserve negative results.

## Then: select and validate one complete procedure

- [ ] Derive one candidate at a time with an explicit measurement model,
  abundance target, inference/prior rule, normalization and null. Check
  zero/one/many-read uncertainty, location, predictive behavior and precise spikes.
  Compare full kernels with their own means and matched normalized points.
- [ ] Freeze the usable candidate, metrics, acceptance margins, replication
  target and fresh seeds before confirmation. Validate realistic large-cohort
  bulk settings first; report operating boundaries separately.
- [ ] Validate full-transcriptome FDR and effect/subset recovery, including RLE
  and designated controls, reference/test overlap, gene filtering, sample
  eligibility, dependence and all exposed directions/scales. Supply sufficient
  permutation resolution; audit numerical bounds and sequential stopping.
- [ ] Compare with frozen production WADE and native DESeq2/edgeR for appropriate
  gene-level targets, and OUTRIDER/related methods for separate patient-level
  outlier targets. Use established biological inputs and known-truth matrices.
- [ ] Demonstrate a reproducible uncertainty benefit with useful sensitivity
  and adequate error control. Neither fewer calls nor correct integration alone
  establishes scientific value. Resolve the retained weak-subset regression,
  singleton collapse and ETV4 interpretation without tuning to those genes.

## Finish the package and examples after scientific selection

- [ ] Migrate the selected procedure into a compact API, without an internal
  prior/bandwidth/model menu or legacy compatibility requirements.
- [ ] Optimize measured bottlenecks after selection; add Rust only if useful,
  with replay against a checked Python reference.
- [ ] Recompute TCGA prostate fusion-oriented and breast subtype analyses using
  full gene counts only. Finalize count-pipeline/version and genome/annotation
  provenance; distinguish expression-derived subtypes from independent evidence.
- [ ] Reexecute all six notebooks: demo, benchmark, reference benchmarks,
  prostate, breast and private RNA100k. Recheck descriptions, probes, themes,
  results, plots and the before-running case-exclusion recipe against the final API.
- [ ] Inspect final HTML/PDF/interactive artifacts and obtain user approval of
  the exact results, figures and claims. Earlier artifacts do not approve revisions.
- [ ] Pass source, installed-wheel, fallback, distribution and cross-platform CI
  checks; update release notes and citation metadata. Publish/tag only with final
  release authorization.

## Evidence retained; scope of prior completion

The [production migration](archive/development-20260923/docs/production-migration-0.2.0.md) completed the
single unsigned empirical log-W1 core, normalization/filtering API, source/wheel
checks and transcriptome replay. The six notebooks were rebuilt and inspected
for that baseline. Those checks remain evidence for the recorded source and
artifacts; they do not validate a changed kernel procedure.

The [scientific pilot](archive/development-20260923/docs/kernel-scientific-pilot.md),
[paired downsampling](archive/development-20260923/docs/kernel-downsampling.md) and
[minority-depth study](archive/development-20260923/docs/kernel-depth-stress.md) are completed research,
with no production kernel selected. Use the [documentation index](docs/index.md)
for the earlier mechanism studies and benchmark inventories.

The long pre-reset queue is preserved [verbatim](.release/0.2.0/design-reset-20260923/previous/PLAN.md).
The earlier two-stage migration archive remains in [archive](archive/README.md).
No completed experiment or negative result was discarded.

After a reviewed production candidate: transcriptome rarity and predictive
surprise. Sparse single-cell, spatial and cell-free RNA recommendations require
their own validation and are not certified by the bulk release work.
