# Roadmap

**A work queue.** What has been built is described in [`README.md`](README.md)
and specified in [`docs/method.md`](docs/method.md); *why* anything was decided
lives in the document that owns the subject. Nothing is re-derived here.

**Scope: WADE is for discrete count data.** Raw counts in, and only raw counts
— the continuity jitter is applied at count precision before division and the
subset stage's null is built by thinning reads, so a pre-normalized matrix can
reproduce neither and there is no entry point for one. Continuous input is not
a target: it is neither tested nor tuned for, and no item here exists to
support it.

The goal that governs the order: **a scientist should be able to load their
data, run WADE, understand the answer, and show it to someone.**

### Standing decisions

| decision | reasoning in |
|---|---|
| Two stages, reported separately; no combined p-value, no categorical label | `method.md` §5 |
| The subset test's null is the **fitted global shift**, not no-difference | `method.md` §3 |
| Stage 1 linear, stage 2 log; no transform knob | `method.md` §10.1–10.2 |
| Raw counts only; `thin=False` keeps the division as a *demonstration*, not an option | `method.md` §10.3 |
| The quantile grid is capped (`max_probs`) and the realized `m` recorded | `method.md` §1 |
| Chunking is a memory layout, never a numerical choice — bit-identical | `scaling.md` §2.2 |
| Fast paths that change numbers are opt-in and named (`stage1`, `fit_backend`) | `scaling.md` §3.1, §3.3 |
| Rank past the p-value floor with magnitude and the permutation z — never a normal CDF | `method.md` §4, §6 |
| **WADE reads no files** — arrays, frames and sparse in; your reader reads | `wade/io.py` |
| One gene id is all the core needs; other per-gene columns are display only | `docs/plotting.md` |
| Bootstrap CIs are within-group resampling; Poisson/hybrid measured miscalibrated | `scaling.md` §6 |

### Landed 2026-08-20/21 (see `git log`)

The 80,000-sample performance queue — grid cap, gene chunking, GEMM stage 1, a
gene-major mean-shift kernel, a 34× fold-change fit, threaded bootstrap — so
30,000 × 80,000 now runs in **29.7 min at 55.6 GB**, measured rather than
extrapolated. The ranking columns (`subset_log2_fc`, `z_mean_shift`,
`z_subset`) and bootstrap intervals for all five descriptors. Restricted
permutation (`strata=`) with `permutation_space()`. A package audit's nine
release blockers and ten doc/code mismatches. And a simplification pass:
**~1,300 lines deleted**, five redundant paths collapsed to one,
`wade_from_matrix` removed.

---

## 1. Now

1. **The remaining core items** — six small ones, specified in
   [`docs/plan.md`](docs/plan.md) §5. Led by the three that are pure plumbing:
   stage 2 computes its exceedance count and refinement flag and then throws
   them away, which is why `limits.md` §5's "is this gene *at* the floor?"
   reading rule can only be applied to stage 1.
2. **Plotting becomes an extension** — a `wade/plotting/` subpackage and
   [`docs/plotting.md`](docs/plotting.md). Reorganization only: no public name
   changes, no statistic touched.

## 2. Soon

### Visualization — an extension, not the core

Stance and detail in [`docs/plotting.md`](docs/plotting.md). Most DE tools
provide little of this: the results table is the interface, and exporting it to
ggplot is a first-class path rather than a fallback. These are conveniences,
sequenced after the core is finished.

- **Theme and palette tokens** — one token set, light/dark/high-contrast built
  in, `theme=` on every plot function. Dark mode is then a palette, not a
  feature.
- **Draw the bootstrap intervals.** All five descriptors carry one; the figures
  show none, so a subset resting on four affected samples looks exactly as firm
  as one resting on four hundred.
- **A driver figure** — the figure form of `subset_drivers` + `library_qc`:
  which samples are in the affected region, with their complexity and depth
  beside them. This is the check that reversed a tempting reading of the real
  data (`notebooks/rna100k.qmd`); it should not be text-only.
- **Linked views (plotly)** — click a volcano point, see that gene's panel.
  Needs a live kernel, so it does not survive into exported HTML.
- **A stage-1 figure.** `GeneDetail.cumulative_area` is stage 1's statistic and
  is drawn nowhere.
- **Optional per-gene metadata in figures and tables** — gene symbol above all.
  The core needs one gene id and nothing more; anything else the user supplies
  is carried and *optionally displayed*, never read by a statistic.
- **Label de-collision**, worth doing only after the above: the smear is worst
  exactly where p-values tie, which is where real cohorts sit.

### Inference

- **P-value resolution at scale** — `scaling.md` §4, with the deciding
  experiment specified there. Observed on real data: 18.8% of genes at the
  mean-shift floor. Three candidates (GPD today, fgsea-style multilevel
  splitting, a saddlepoint for stage 1 — which `stage1="gemm"` has already made
  exactly linear). Deliberately demoted: magnitude and the z-scores already
  supply the *ordering* this would buy.
- **GPD moment fit → maximum likelihood.** Moments are poorly behaved for
  `xi > 0.5`, the heavy-tailed regime the refinement exists for. Two
  constraints on any upgrade: the floor `1/(B · n_tail)` must survive it (it
  states what `B` permutations can support and is not a numerical guard), and
  the `xi <= 0` exponential branch must survive.
- **The (μ = 2, 1000 v 1000) cell of `method.md` §10.3** was measured on 60
  genes. Re-measure with a few hundred before quoting it outside that doc; the
  harness was `prototypes/scale_and_counts.py`, recoverable at `fddf2c1`.
- **Permutation and p-value construction** more broadly — the user has
  references to bring to this.

### The real dataset

`notebooks/rna100k.qmd` runs on the 31k × 83k cohort. What it measured and what
that demands is in `scaling.md` §7: group-associated depth, the saturation of
significance, the artefact libraries, 47% sparsity. The **rank-recovery
benchmark on real background** is the priority there, and it wants the
benchmark notebook below.

### Release

CI exists — three OSes × two Pythons, the declared numpy floor, a kernel-less
run, wheels and an sdist. What remains: publishing wheels, an API
documentation build, and a decision on whether the validation simulations ship
as tests, as documentation, or both.

## 3. Notebooks

Three deliverables, sequenced after the implementation above.

1. **`demo`** — small synthetic data with planted ground truth. The showcase
   and gallery: a README in executable form, every claim checkable.
2. **`benchmark`** — the head-to-head, on simulated counts *and* a few real
   datasets from the literature: COPA / OS / ORT / MOST / LSOSS as the
   subset-detection competitors, t-test and Wilcoxon as floors, waddR as the
   nearest Wasserstein relative. This is where WADE's claim is tested against
   alternatives rather than against itself.
3. **`rna100k`** (exists) — the real cohort. Manuscript material rather than a
   demo, and labelled as such.

## 4. Not doing, and why

Recorded so they stop resurfacing.

- **A second chunked driver** for pre-normalized input — there is no
  pre-normalized entry point.
- **`cpm` / `rle` as functions** — unreachable from `wade()`, and the
  capability is already there: CPM is `normalizer=1.0`, other size factors go
  in through `lib_sizes=`.
- **featureCounts / MatrixMarket readers** — when the data is in hand and the
  shape is known, not before. The default answer to "can WADE read X?" stays
  "polars can, and then WADE accepts it".
- **Gene metadata in the statistics.** Carried and displayed, never read.
- **`adjustText` as a dependency** — only if label de-collision's own approach
  fails.
- **Publication typesetting and arbitrary figure layouts** — export the table.

## 5. Open questions

- **Single cell and spatial** — `scaling.md` §5. Sparsity is an *opportunity*:
  a gene's sorted values are a run of zeros then the sorted non-zeros, so the
  per-permutation work becomes `O(nnz)`. The trap is statistical, not
  computational: **cells are not exchangeable, donors are.** A million cells
  from twenty donors is twenty samples. Either pseudobulk, or permute donor
  labels with cells kept together — the second is a genuine research direction,
  and `strata=` is already the machinery for it.
- **What happens to the parity fixtures long-term.** They pin the machinery
  underneath the statistic — normalization, the grids, the null, the GPD, BH —
  rather than the statistic itself, which has been replaced. Still worth
  pinning. Whether `reference/R/` and the renv sandbox stay once nothing new
  will be generated from them is a separate call; deleting them would leave
  `tests/fixtures/` unfalsifiable.
- **Whether `w1` earns its place.** Reported and tested; nothing consumes it.
- **What to do about composition.** Library-size normalization couples genes,
  so a signal-saturated matrix distorts `affected_fraction` and `direction`
  (the subset *test* is immune — its statistic is invariant to a global
  offset). No method on normalized data escapes this; the question is whether
  WADE should detect and warn.
