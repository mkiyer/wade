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

Then the last six core items, and **the whole visualization queue below it**:
`plotting.py` became the `wade/plotting/` subpackage (a verified relocation —
all 54 definitions byte-identical in their new homes), and every bullet that
stood under "Visualization" is now built. `import wade` is still NumPy-only,
and the parity ledger never moved off **9.155e-15** over 436 comparisons.

| item | what landed |
|---|---|
| theme and palette tokens | one frozen `Theme`; `"light"` / `"dark"` / `"high-contrast"`; `theme=` everywhere; **no colour literal outside `theme.py`**, pinned by a test |
| bootstrap intervals drawn | error bars on any axis holding a descriptor with an interval, on the labelled points; a band in `plot_gene` for the affected region's edge; nothing at `n_boot=0` |
| a driver figure | `plot_drivers` — four columns per driver, each against the cohort's median and IQR |
| per-gene metadata | `meta=` names points by symbol and joins the hover and every `table()`; still read by no statistic |
| linked views | `plot_linked`, a plotly `FigureWidget`; needs `anywidget` and a live kernel |
| a stage-1 figure | `plot_gene(cumulative_area=True)` — stage 1's statistic accumulating, beside stage 2's curve |
| label de-collision | a greedy slot assignment; overlapping pairs 24 → 1, 28 → 0, 24 → 0 (`plotting.md`) |
| `GenePanel.table()` | the third dataclass finally has the twin the other two had |

`docs/plotting.md` owns all of it — the stance, the module map, the measured
numbers, and the two limits worth knowing (a linked view does not survive
export; twelve labels on near-coincident points is past what placement can fix).

---

## 1. Now

**Stage 2's p-value tail** (§2 below). Stage 1 was solved on 2026-09-03 —
`stage1="saddlepoint"` gives it an exact permutation p-value with no sampling
and no floor — and that leaves stage 2 as the one part of the method whose
inference is known wrong: 4–80× off where it is safe, and anti-conservative to
0.019 where it is accurate. It is also the distinctive stage and 80% of the
runtime. Every route that worked for stage 1 is closed to it, so this is a
research item, not an implementation item; `docs/pvalue-review.md` is the
question written out for external review, and its §8 is the ordered list.

The **first step is mechanical, not statistical**: freeze `μ_k` and `σ_k` from
a separate uniform permutation sample. Until that happens the observed `T` is
not a fixed function of the labels (it moves 3% between samples at B = 2,000),
which blocks multilevel splitting and every other conditioned-sampling method.

v0.1.0 was tagged 2026-09-02 and CI is green on all 12 jobs. What remains of
the release is distribution — PyPI and a GitHub Release with the wheels CI
already builds — and it is paused deliberately, not forgotten: see §2.

Release was the previous "Now" because on 2026-09-02 it turned out never to
have been tried: `main` was 37 commits ahead of `origin/main`, so the CI added in one of
those commits had never run, and **11 of its 12 jobs would have failed** —
`--no-build-isolation` with no maturin in the runner, `--no-index` blocking
numpy as well as PyPI, and a smoke test naming `wade.permutation` for a
function that lives in `wade.plotting`. All three are fixed. The fourth
failure was real and is the interesting one: see §2's Release entry.

## 2. Soon

### Inference

- **P-value resolution at scale** — **promoted 2026-09-02**, because
  `scaling.md` §4.5's deciding experiment was run and the answer was worse
  than the assumption behind demoting it. Against 1e9 brute-force
  permutations, the GPD refinement is **75x to 2000x conservative** over
  1e-5 to 1e-7 — the range BH actually decides in on a 20,000-gene cohort —
  and never reaches its own nominal 2e-6 floor, because the fitted shape is
  negative on every deep-tail gene and the `xi <= 0` exponential branch is a
  far heavier tail than a bounded permutation null. Ordering survives;
  calling does not, so this is lost power rather than lost cosmetics.
  Both alternatives were prototyped in `tools/pvalue_study.py` and both work:
  the saddlepoint is within 1% to 1e-6 with no permutations, multilevel
  splitting within 10% throughout at 8 ms/gene.
  **Stage 2 measured too** (§4.6), and it inverts the priority: the GPD is
  only 4-80x off there against stage 1's 75-2000x, because stage 2's fitted
  shape is negative for 71% of deep-tail genes against stage 1's 100%, so the
  `xi <= 0` exponential substitution fires far less. **Stage 1 has the worse
  problem**, which is the opposite of the stages' relative importance.
  Multilevel works on stage 2 at 0.94x median — but only after `mu` and
  `sigma` are **frozen** from a uniform sample, because stage 2's statistic is
  standardized by moments estimated from the same permutations the tail is
  read from and is therefore not a fixed function of a label assignment.
  **Stage 1 is decided** (`scaling.md` §3.1, §4.9): the exact signed area
  `x1bar − x0bar` at every geometry, p-value by double saddlepoint, no
  permutations. Validated against 1e8-permutation truth on four geometries,
  on NB counts through `wade()`'s own normalization, on sparse counts, and at
  300 v 300 — 0.91–1.15 median, worst 0.61–0.89, where the shipped path is
  3–4,906× conservative. **Implemented 2026-09-03** as
  `src/wade/saddlepoint.py` and `stage1="saddlepoint"`, opt-in with 18 tests;
  it is not the default and should not become one until stage 2 is solved,
  because stage 2 keeps the permutation loop running and the saddlepoint is
  then pure added cost (~60× the stage-1 loop) — `scaling.md` §4.9.
  One finding worth carrying: the grid quadrature is the **better statistic**
  off balance, by 2–4× when n1 > n0, because interpolating the larger group
  trims it and expression is right-skewed (§4.10). It still loses end to end,
  because its floor at 2e-6 costs more than the trimming buys. An exact or
  relative-error tail for an **L-statistic** would recover that 2–4×, and is
  question 4 in `pvalue-review.md` §8.
  **Stage 2 stays open** and is now §1: freeze the moments, then
  multilevel with the move in the kernel for genes at the empirical floor;
  no analytic endpoint exists for it, so none of the fixed-endpoint routes
  reach it. A computable upper bound on stage 2's statistic — even a loose
  one — would make the fixed-endpoint GPD available and is worth more than
  either route.
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
benchmark notebook below. §7.2's artefact check is no longer text-only —
`plot_drivers` is the figure for it.

### Release

CI exists — three OSes × two Pythons, the declared numpy floor, a kernel-less
run, wheels and an sdist — and as of 2026-09-02 it is *correct*, which is a
different claim: every job was reproduced in a clean venv and the four
failures fixed.

**The fourth was not a workflow bug.** `tests/test_stage1_gemm.py` asserted a
per-element `rtol=1e-12` on `mean_shift`, a difference of two large nearly
equal group means — so a gene whose groups almost cancel was being held to a
relative tolerance it cannot carry. Which BLAS numpy links decides the
summation order, and it is not ours to choose: conda-forge gives OpenBLAS,
PyPI gives Accelerate. On OpenBLAS chunked gemm comes out exactly bitwise; on
Accelerate it lands at 2.7e-12 of that one element, and the suite fails. In
the units that mean something — the statistic's own scale — both read a steady
1.2e-14 to 1.8e-14. The tests now assert that, and `scaling.md` §3.1 carries
the measurement. **A dev machine with one BLAS cannot see this**; CI is what
sees it, which is the argument for running it.

**v0.1.0 was tagged and pushed on 2026-09-02** and all 12 jobs are green.

What remains: publishing wheels to PyPI and a GitHub Release, an API
documentation build, and a decision on whether the validation simulations ship
as tests, as documentation, or both. Distribution is **paused pending the
user's call** — the first public version would carry stage 2's known
conservatism, which `limits.md` now states plainly, and whether to ship on
that basis is theirs to decide, not a default.

## 3. Notebooks

Three deliverables, in this order, not three phases of one. The first has
landed; **`benchmark` is the current work** (§1).

1. ~~**`demo`**~~ — **done 2026-08-22**, `notebooks/demo.qmd`. Small synthetic
   data with planted ground truth, renders to HTML and PDF, and is the master
   source for `docs/figures/*.png` and the numbers the README quotes.
2. ~~**`benchmark`**~~ — **done 2026-09-02**, `notebooks/benchmark.qmd`, on
   simulated counts: COPA / OS / ORT / MOST / LSOSS as the subset-detection
   competitors, `t`-test and Wilcoxon as floors, waddR as the nearest
   Wasserstein relative and the one run rather than reimplemented. Every
   method goes through the same permutation null, the same GPD refinement and
   the same BH, from `wade.pvalues`, so a difference between two rows is a
   difference between two statistics. The statistics are in
   `tools/competitors.py`, each verified against its exact null expectation.
   **Still open**: the same head-to-head on real cohorts from the literature,
   which is what the rank-recovery work in §2 wants.
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
- **`adjustText` as a dependency** — settled: the in-house slot assignment did
  the job (`plotting.md`), and it would not have fixed the one case that still
  overlaps either.
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
