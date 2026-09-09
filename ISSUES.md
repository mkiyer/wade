# Issues and development items

The live work queue. What WADE does is in [`README.md`](README.md) and
[`docs/manual.md`](docs/manual.md); how it works and why is in
[`docs/method.md`](docs/method.md). Nothing is re-derived here. Items are
numbered for reference and never renumbered; closed items move to the end.

## Open

### 1. Stage 2's p-value tail — the open statistical problem

**Status: research, not implementation.** Stage 2's p-values come from 2,000
permutations plus a generalized-Pareto tail fit. Measured against 1e8
brute-force permutations, that fit is 4–80× conservative in one regime and
anti-conservative to 0.019 in another, and which regime a gene is in is set by
a latent property of the data. Ordering survives; calling with an FDR loses
real subset genes. Stage 1 was solved with an exact double saddlepoint
(`stage1="saddlepoint"`); every route that worked there is closed to stage 2,
whose statistic is a maximum over widths and a subset sum at no geometry.

[`docs/pvalue-review.md`](docs/pvalue-review.md) is the full write-up, written
for a statistician without code access, and its §8 is the ordered list of
questions. The concrete first step is mechanical: freeze `μ_k` and `σ_k` from a
separate uniform permutation sample, so the observed statistic is a fixed
function of the labels (it moves 3% between samples at B = 2,000 today).
Multilevel splitting on the frozen statistic then applies, at ~1–3 s per gene
at the floor. Do not start with more GPD variants or a bound on the statistic:
seven such suggestions were tested and refuted (pvalue-review Appendix B).

### 2. The saddlepoint on atom-dominated genes

`stage1="saddlepoint"` is a continuous approximation to a lattice distribution
when a gene's nonzero support is smaller than a group: most relabellings tie
on a handful of sums. Exact ties at the extreme are now counted exactly, but
inside the range the approximation is off by several-fold in either direction
(measured 0.39 against 0.09 brute force for 3 nonzero values among 40 with
the default jitter). The docstring states the limitation. A gate that falls
back to the permutation p-value for genes with support below some `k`, or a
lattice correction, would close this. Not a concern for expressed genes.

### 3. Publish to PyPI

CI builds wheels for Linux x86-64 and aarch64, macOS arm64 and x86-64, and
Windows x64, plus an sdist, and the `publish` job is a manual
`workflow_dispatch`. Before pressing it:

- The name `wade` on PyPI belongs to an unrelated 2018 project, so the
  distribution is `wade-rnaseq` (import name `wade`). A PEP 541 transfer request
  for `wade` is possible but takes months; if granted, only `pyproject.toml`
  and the install lines change.
- Create the PyPI trusted publisher for `mkiyer/wade` and a GitHub
  environment named `pypi`.
- Verify the `ubuntu-24.04-arm` and `macos-15-intel` runner labels are still
  current; the wheel matrix was changed to native runners on 2026-09-08 and
  has not yet run.
- Free-threaded CPython builds cannot use the abi3 wheels and fall back to
  the sdist, which needs a Rust toolchain. Stated in the manual.

### 4. The benchmark on real cohorts

`notebooks/benchmark.qmd` compares nine methods on simulated counts through
one permutation null, one refinement and one BH. The same comparison on real
cohorts with known subset structure is what the rank-recovery work on the
real cohort needs, and has not been done.

### 5. `notebooks/rna100k.qmd` is a private analysis

It reads the 31k × 83k cfRNA cohort from a local Dropbox path and is
manuscript material, not a demo. Decide whether it ships in the public repo
at all, and relabel it if it does.

### 6. Composition: detect and warn?

Library-size normalization couples genes, so a matrix in which a large share
of genes carry strong signal distorts `affected_fraction` and `direction` of
every null gene (the subset *test* is immune; its statistic is invariant to a
global offset). No method on normalized data escapes this. Open question:
whether WADE should measure it (the case/control library-size ratio is a
start) and warn.

### 7. Stage 2's memory peak

Peak memory is about three times the count matrix and lives in stage 2's
working space: it barely depends on `nperms` or `gene_chunk`, and
`subset=False` drops it by a third. The remaining cut is that transient, not
the result object (already 1.1× the input) and not float32 storage.

### 8. GPD moment fit → maximum likelihood

Moments are poorly behaved for `ξ > 0.5`, the heavy-tailed regime the
refinement exists for. Any upgrade must keep the floor `1/(B · n_tail)` and
the `ξ ≤ 0` exponential branch. Subsumed by item 1 if stage 2 gets a
different estimator.

### 9. Whether `w1` earns its place

The 1-Wasserstein distance is reported and tested; nothing consumes it.

### 10. Single cell and spatial

Sparsity is an opportunity (a gene's sorted values are a run of zeros then
the sorted non-zeros, so per-permutation work is `O(nnz)`), but the trap is
statistical: cells are not exchangeable, donors are. Either pseudobulk, or
permute donor labels with cells kept together; `strata=` is already the
machinery for the second. Not started.

### 11. Re-measure one cell of `method.md` §9.3

The (μ = 2, 1000 v 1000) thinned false-subset rate (0.10) was measured on 60
genes. Re-measure with a few hundred before quoting it outside that document.

## Standing decisions

Recorded so they stop resurfacing. Reasoning is in the document named.

| decision | where |
|---|---|
| Raw counts in; no entry point for a normalized matrix; no division fallback | `method.md` §7, §9.3 |
| Two stages, reported separately; no combined p-value, no categorical label | `method.md` §5 |
| The subset test's null is the **fitted global shift**, built by binomial thinning | `method.md` §3, §9.3 |
| Stage 1 linear, stage 2 log; no transform knob | `method.md` §9.1–9.2 |
| `affected_fraction` uses the fourth moment on the log curve | `method.md` §4 |
| The grid is capped (`max_probs`) and the realized `m` recorded | `method.md` §1 |
| Chunking is a memory layout, never a numerical choice: bit-identical | `wade.api._run`, `tests/test_chunking.py` |
| Paths that change numbers are opt-in and named: `stage1`, `fit_backend`; `backend="auto"` never changes an answer | `method.md` §6 |
| Rank past the p-value floor with magnitude and the permutation z, never a normal CDF | `method.md` §6 |
| The combinatorial floor is a scale, not a bound; never clamp a p-value to it | `method.md` §11 |
| Bootstrap intervals are within-group resampling with replacement | `method.md` §9.5 |
| **WADE reads no files**; arrays, frames and sparse in; your reader reads | `wade/io.py` |
| Gene metadata is carried and displayed, never read by a statistic | `wade/io.py` |
| Plotting is an extension: `import wade` imports no plotting library | `CONTRIBUTING.md` |
| Volcano x axis is `log2_fc`, never `mean_shift`; cutoff lines are the BH raw-p cutoff | `wade/plotting/data.py` |

## Not doing, and why

- **A pre-normalized entry point** — the jitter and the thinning need counts.
- **`cpm` / `rle` as functions** — CPM is `normalizer=1.0`; other size
  factors go in through `lib_sizes=`.
- **featureCounts / MatrixMarket readers** — polars can, and WADE accepts
  what it produces.
- **Gene metadata in the statistics** — carried and displayed, never read.
- **`adjustText` as a dependency** — the in-house label placement did the
  job.
- **Publication typesetting and arbitrary figure layouts** — export the
  table.
- **Making the saddlepoint the default** — not until stage 2 is solved: the
  permutation loop runs anyway for stage 2, so the saddlepoint is pure added
  cost (~60× the stage-1 loop) today.
- **A fixed-endpoint tail for stage 2** — closed 2026-09-04; stage 2's
  maximum is not approached by the permutation null and its tail shape is
  genuinely positive on some genes (pvalue-review §8, question 2).
- **Poisson or hybrid bootstrap schemes** — measured miscalibrated against
  the within-group bootstrap.

## Closed

- **Pre-publication review (2026-09-08).** Fixed: `gene_detail` crashed with
  a matrix normalizer; an all-zero matrix ran instead of being refused; the
  saddlepoint's one-labelling floor was tie-blind (ten orders anti-conservative
  on an exact-ties gene) and its bracket widened at one end only; the Rust fit
  panicked on a non-positive bisection range; `subset_drivers` sized the driver
  set by the grid instead of the case group; colouring a figure by an
  arbitrary column crashed both renderers; `label="MYC"` labelled nothing;
  the manifest materialized the full pseudocount matrix; a quadratic loop in
  `Condition.vector`; the pseudocount check was unreachable for scalars;
  `refined_*` was set even where the GPD bailed. Removed: `thin=False` and the
  division correction, `report()`, `to_frame()` as a function, the
  `manifest=` flag, unread `r_test` / `b` matrices, duplicate mean-difference
  functions, and the porting-era narrative throughout. Renamed the
  distribution to `wade-rnaseq`.
- **Stage 1's p-value floor (2026-09-03).** `stage1="saddlepoint"`.
- **The 30,000 × 80,000 target (2026-08-20).** Runs in 30 min at a 56 GB
  peak with 2,000 permutations.
- **Parity fixtures after the R reference was deleted (2026-09-04).**
  `tests/test_independent_reference.py` re-derives every fixture value from
  NumPy, SciPy and the closed forms, and is forbidden from importing WADE.
