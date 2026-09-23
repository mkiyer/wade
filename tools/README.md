# Operational tools

This directory contains package, notebook, data-input and release helpers.
Kernel development lives in [research/](../research/README.md); completed study
runners and superseded benchmarks are in the [development archive](../archive/development-20260923/README.md).
No current helper imports an archived prototype.

| Task | Tools |
| --- | --- |
| Notebook execution and review | `render_release.py`, `review_notebook.py`, `notebook_support.py`, `audit_notebook_exports.py` |
| TCGA inputs and examples | `fetch_tcga_annotations.py`, `tcga_examples.py` |
| Reference inputs and verified comparisons | `fetch_reference_counts.py`, `refresh_comparisons.py`, `comparison_support.py`, `score_outlier_reference.py`, `seal_reference_results.py` |
| Independent comparison checks | `audit_comparisons.py`, `audit_outlier_comparison.py`, `check_production_reference.py` |
| Production replay and release review | `review_production.py`, `assemble_release_review.py`, `smoke_install.py` |
| Workspace boundaries and documentation links | `check_workspace.py` |

Use the [notebook guide](../docs/notebook-builds.md) for actual build commands
and required data/configuration. Comparison refreshes reuse native outputs only
when their inputs and settings match. Production replay targets the frozen
point-method evidence; it does not validate a future kernel procedure.

Notebook builds use isolated snapshots and verified computation caches. An
inference change requires fresh results; revised figures or prose require review.
The private workbench and local data remain excluded from distribution. Review
hubs link the dated baseline assessment and the current release plan separately.

The old scaling benchmark, legacy comparator cache importer and stage-specific
p-value tools were archived because they still refer to retired APIs. They are
not supported entrypoints against the current package. Reproduce historical
studies with their sealed source or the verified pre-cleanup snapshot.
