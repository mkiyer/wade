# WADE 0.2.0 change notes

Release candidate; publication and all result/figure approvals remain pending.

These notes describe the implemented unsigned log-W1 baseline. The 2026-09-25
[release plan](../PLAN.md) now targets a validated point-based release and defers
measurement kernels. Directional/scale API changes are being evaluated in
research and will be documented here after selection and production migration.
The user requires directional biomarker discovery; the unsigned-only baseline
described below does not satisfy that release objective.

## One inferential contract

WADE now asks whether a gene's expression distribution differs between two
cohorts. Exact unsigned empirical W1 on `log2(count / size_factor + 1)` replaces
the former two-stage procedure. This can detect broad and recurrent subset
changes, including opposing changes that would cancel in a signed area.
The resulting q-value belongs to the whole distribution comparison.

There is one p-value and one BH-adjusted q-value per tested gene. The
Besag–Clifford sequential label test stops at the twentieth inclusive exceedance
or a prespecified cap (499,999 by default), with exact stopping counts exported.
Rust accelerates the same arithmetic as the readable NumPy path. No fitted
bulk correction, thinning null, GPD extrapolation or saddlepoint remains.

## A smaller API

- `wade(counts, condition)` is the default bulk-RLE workflow.
- `fit_normalization(..., reference_genes=...)` supports declared internal controls;
  fitted objects record candidate/used references and retain their scale across
  test filters and sample subsets.
- `Normalization.from_size_factors(...)` accepts externally justified complete
  exposures. `normalization="library"` provides explicit total-count exposures.
- `filter_counts` and custom `test_genes` masks give pooled filtering flexibility.
  All-zero/filtered genes remain in exports with an explicit status and no test.
- `cpm` and `tpm` calculate independent per-sample abundance summaries.
- `bulk_shift`, `upper_tail` and `lower_tail` are fixed signed descriptions;
  they have no separate significance, affected fraction or membership estimate.
- Gene, volcano and pattern plots share the current result schema and three themes.

This is an intentional breaking change. Old stage-specific fields, fitted-tail
options, noise controls and compatibility aliases are absent. Use the
[manual](manual.md) and current notebooks, or replay historical analyses with
their [archived sources](../archive/README.md).

## Evidence and scope

The [migration assessment](../archive/development-20260923/docs/production-migration-0.2.0.md) records mathematical,
backend, transcriptome, comparator and biological checks with their exact
artifacts. All six notebooks use the current API, including full-gene-count
TCGA prostate and breast examples. Reference method fits retain their native
settings and inferential units.

Normalization reference bias can cause false biological discoveries, especially
at large sample sizes. The high-alteration benchmark exposes this directly;
it is not hidden behind higher sensitivity. Sparse/depth-confounded workflow
certification and predictive rare-event significance remain future research.
No result here establishes universal FDR control or replacement of DESeq2 or
native sample-gene outlier methods.
