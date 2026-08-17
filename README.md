# WADE — Wasserstein Area Differential Expression

A quantile-area differential-distribution test built for **subset detection**: it
compares a heterogeneous case group against a control group without assuming the
two differ by a location shift, and it is designed to surface genes altered in
only a fraction of cases.

For each gene, WADE evaluates both groups' empirical quantile functions on a
shared grid of `min(n_case, n_ctrl)` probabilities. The signed area between them
is the bulk axis; the mean difference over the upper-tail window is the **subset
axis**, where a rare high-expressing subpopulation concentrates. Inference on
both axes is by label permutation, with a Generalized Pareto fit refining
p-values whose empirical resolution has run out.

The method joins two literatures: quantile/Wasserstein differential-distribution
testing (scDD, waddR) supplies the mechanism, and cancer-outlier profile analysis
(COPA, OS, ORT, MOST, LSOSS) supplies the subset-detection goal.

## Status

WADE was prototyped in R inside a cell-free RNA analysis project. The laboratory
is moving from R to Python, so WADE is being rebuilt as a Python package with a
Rust-backed core for the permutation loop. The R in `reference/` is the source of
truth to port *from* and to validate *against*; it is not shipped and will not be
maintained. See [`docs/design-decisions.md`](docs/design-decisions.md).

| | |
|---|---|
| Algorithm specification | complete |
| R reference implementation | staged, runnable, byte-identical to origin |
| Golden parity fixtures | 20 fixtures, generated from the R through a deterministic seam |
| Python package | implemented |
| Parity suite | layers 0–9, passing |
| Rust kernel | implemented (PyO3 + rayon), 18–22× faster than the NumPy path |

**Parity, measured:** worst-case relative deviation against the R reference is
**9.2e-15** across 610 comparisons. The normalized matrix, the quantile grids,
the permutation null matrices and the rank scores are **bit-for-bit identical**
to R. Full breakdown, and the two places a correct port must *disagree* with the
R, in [`docs/port-status.md`](docs/port-status.md).

## Layout

```
src/wade/                 the Python package
rust/src/lib.rs           the permutation kernel
tests/                    the parity suite (layers 0–9) and the golden fixtures
tools/r/                  the deterministic seam and the fixture generator
docs/                     the specifications — written for the port, read these first
reference/R/              the original R, runnable in a pinned renv sandbox
reference/docs/           the source documents the specifications were written from
reference/PROVENANCE.md   what was copied from where, with checksums
```

## Install and use

The Rust kernel is optional: without a toolchain the package installs and runs
on the NumPy path, which is the correctness baseline the kernel is validated
against.

```bash
conda create -n wade -c conda-forge python=3.12 numpy scipy pytest maturin rust
conda activate wade
pip install -e . --no-build-isolation
pytest                    # the full parity suite, ~2 s
```

```python
import numpy as np
from wade import wade

res = wade(counts, normalizer, cond, nperms=2000)   # raw counts, genes x samples
res.nprobs, res.k                                   # what the design actually resolves
res.diff_mean, res.tail_mean                        # the bulk and subset axes
res.padj_tail                                       # BH-adjusted, per axis
```

The entry point takes **raw counts**, not a normalized matrix. The continuity
jitter that breaks ties in sparse data is applied at count precision *before*
division, so a pre-normalized matrix cannot reproduce it; `wade_from_matrix()`
accepts one anyway and documents what that costs.

## Reading order

Start with the theory, then the implementation detail:

1. [`docs/algorithm.md`](docs/algorithm.md) — the mathematical specification.
   Implementable without reading any R; this is the contract.
2. [`docs/rationale.md`](docs/rationale.md) — why this statistic exists and why
   each design choice was made. Section 7, on what the permutation p-value is
   actually protecting against, is the one to read if you only read one.
3. [`docs/limits.md`](docs/limits.md) — what WADE does not do, and the two hard
   constraints that decide whether it can answer your question at all.
4. [`docs/r-implementation.md`](docs/r-implementation.md) — the ten R functions,
   line by line, with the conventions a port must reproduce.
5. [`docs/porting-hazards.md`](docs/porting-hazards.md) — eleven places two
   implementations will silently disagree, each with a test, plus a layered
   parity-suite design.
6. [`docs/design-decisions.md`](docs/design-decisions.md) — what is settled and
   what is deliberately open.
7. [`docs/port-status.md`](docs/port-status.md) — what the port measures against
   the reference, and the decisions taken while building it.
8. [`ROADMAP.md`](ROADMAP.md) — the ordered work queue.

[`docs/NEXT_SESSION.md`](docs/NEXT_SESSION.md) is the orientation page for a
session starting fresh in this repository.

## Running the R reference

The sandbox is pinned and offline. Three environment details are load-bearing and
documented in [`reference/R/README.md`](reference/R/README.md):

```bash
export PATH="/usr/local/bin:$PATH"
cd reference/R
RENV_CONFIG_SANDBOX_ENABLED=FALSE \
RENV_PATHS_CACHE="$HOME/Library/Caches/org.R-project.R/R/renv/cache" \
  Rscript validation_sims_v7.R
```

`reference/R/wade.R` must remain byte-identical to its origin; verify with
`shasum -a 256 -c sha256sums.txt` from that directory.

## When not to use WADE

Stated up front because each item is a live limitation rather than a
hypothetical. Full treatment in [`docs/limits.md`](docs/limits.md).

- **No covariate or batch adjustment.** The test takes no design matrix. A
  confound can be detected but not removed.
- **No repeated-measures handling.** Samples are assumed exchangeable; libraries
  from the same subject are not.
- **One-sided upward.** The test and the rank scores reward up-in-case genes.
  Genes lost in cases are visible in the bulk statistic but are not what the
  ranking surfaces.
- **The subset axis needs order statistics to exist.** The quantile grid has
  `min(n_case, n_ctrl)` points and the tail window is the top 10% of them, so
  below roughly 20 in the smaller group the "tail mean" is one or two order
  statistics wearing the name of an average.
- **A combinatorial floor limits subset detection**, and it binds harder than the
  original write-up suggested. If `k` samples carry a signal, label shuffling
  places all of them in the case group with probability
  `C(n_case, k) / C(n_case + n_ctrl, k)`; when that exceeds your alpha, no effect
  size and no permutation count can separate the signal. The floor is driven by
  group *imbalance*, so evaluate it for your own design.

## Provenance

Extracted from the MCTP cfRNA analysis repository at commit `828f2f1c`. The
consumer layer that called WADE (contrast registry, feasibility screen, figures)
stays in that repository and is copied here only as evidence of what a caller
needs — see [`reference/R/downstream/README.md`](reference/R/downstream/README.md).
No patient-derived data is included; every fixture and simulation here is
synthetic. Details in [`reference/PROVENANCE.md`](reference/PROVENANCE.md).

## License

GPL-3. See [`LICENSE`](LICENSE).
