# Rebuilding and reviewing the 0.2.0 notebooks

All six sources use the migrated single-test API. The private RNA100k workbench
remains local and excluded from release archives. All current examples use gene
counts; no splice-junction analysis is included.

Use the intended Python environment with NumPy, Polars, pandas, SciPy,
Matplotlib, Plotly, Jupyter, Excel-reading support and Quarto/Typst available.
These are notebook/release dependencies, not additions to the NumPy-only
inference runtime.

```sh
python tools/render_release.py demo --root .release/0.2.0/migrated-core
python tools/render_release.py examples --root .release/0.2.0/migrated-core --config /path/to/tcga-config.json
python tools/render_release.py examples-breast --root .release/0.2.0/migrated-core --config /path/to/tcga-config.json
python tools/render_release.py rna100k --root .release/0.2.0/migrated-core --config /path/to/tcga-config.json
```

The TCGA inputs follow [the configuration contract](tcga-examples.md). No
external sample data are bundled. The private workbench also needs its local
cohort paths; it is not a public sparse-assay recipe.

The two comparison notebooks read the **current sealed comparison refresh**:

```sh
WADE_COMPARISON_RESULTS=/absolute/path/to/production-comparisons \
python tools/render_release.py benchmark --root .release/0.2.0/migrated-core
WADE_COMPARISON_RESULTS=/absolute/path/to/production-comparisons \
python tools/render_release.py reference-benchmarks --root .release/0.2.0/migrated-core
```

That directory contains `large`, `reference`, `outlier` and
`reference-diagnostic`, each with a checksum registry. The current local
candidate is `.release/0.2.0/production-comparisons-20260919`. The notebooks
verify every sealed file before reading it and retain numerical exports and
provenance. `tools/refresh_comparisons.py` recomputes WADE using the public API
and preserves frozen native DESeq2/edgeR/OUTRIDER-family tables byte for byte.
`tools/audit_comparisons.py` independently checks stopping records, BH and
comparison metrics. Native R fits are reused; rendering does not invoke R.
The known-exposure diagnostic is a post-refresh analysis on existing data,
not a new estimator or independent confirmation. Its source script and all
result hashes are retained alongside the comparison refresh.

The render tool creates a new `builds/<notebook>-<UTC-time>/` under the chosen
root. It snapshots package, native kernel, tools and notebook source, writes
an environment record and interpreter-specific kernelspec, and renders HTML
and Typst PDF. Each build records its complete source fingerprint, execution
log, return code and artifact hashes. Expensive TCGA inference is cached by
inference code, dependency versions, input identity and numerical settings.
Cache contents are checksummed; figure changes rerun the presentation cells.
The demo executes the public API directly.

The runner sets writable notebook caches. Quarto also needs its platform
application-data log directory and local Jupyter sockets; restricted
execution environments must allow those. A cache load is not a measured
inference runtime. Preserve the original uncached timing record when quoting
performance.

## Review each exact build

```sh
python tools/review_notebook.py /path/to/completed-build
```

The gallery includes complete HTML/PDF, every emitted figure, numerical
exports, captions, source cells and artifact hashes. Missing declared figures
or error cells fail registration. Inspect all figures and every PDF page;
compare displayed tables with full-precision exports. Record completed
technical checks in `review/qa.json`, without marking user approval.

```sh
python tools/assemble_release_review.py \
  /path/to/demo-build /path/to/reference-build /path/to/benchmark-build \
  /path/to/prostate-build /path/to/breast-build /path/to/private-build \
  --output .release/0.2.0/review
```

The assembler verifies artifact hashes and preserves prior selection manifests.
Revised figures/results need new approval; successful execution or technical
QA never grants it. The local review hub and private data are not a published
website. The [review register](../archive/development-20260923/docs/release-0.2.0-review.md) identifies the selected
builds and remaining scientific/release gates.
