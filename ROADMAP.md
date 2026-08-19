# Roadmap

What is planned, in dependency order, with the reason for the order stated where
it is load-bearing. This is a work queue, not a changelog: what has been built is
described in [`README.md`](README.md) and specified in
[`docs/method.md`](docs/method.md).

The goal that governs everything below: **a scientist should be able to load
their data, run WADE, understand the answer, and show it to someone.** The
statistic is done; the path from a file to a figure is not.

---

## 1. Plotting

Blocks the notebook, and the single-gene panel is how the method is explained,
so it comes first.

- **`plot_gene()`** — the log-ratio curve `R(p)` against quantile, which *is*
  the cliff-versus-straight-line reading. A flat line is a global fold change; a
  curve that sits at zero and then climbs is a subset. Show the two quantile
  functions beneath it (log y) so the reader can see the raw distributions.
  `wade_gene()` already returns `p`, `y1`, `y0`, `cum` and `r`.
- **`plot_volcano()`** — effect size against significance, **coloured by
  `affected_fraction`**, with `alternative` respected. Two variants, one per
  stage. Note that `mean_shift` is in TPM-like units and spans thousands, so it
  collapses to a vertical line if plotted like a fold change: use `log2_fc` on
  the x axis.
- **`plot_stages()`** — `p_mean_shift` against `p_subset`, which is the figure
  that shows the four quadrants of the README's table directly.

Design constraint: matplotlib should be an *optional* dependency, imported
inside the plotting functions, so the statistic keeps its one-dependency
surface.

## 2. I/O with Polars

Goal-5 work, and quick, because Polars already handles the formats.

- **Readers** for a counts matrix — CSV/TSV/Parquet/IPC — returning the matrix
  plus gene and sample names, and a loader for a sample sheet giving the
  condition vector by sample name.
- **Writers** for the result frame.
- **Name-awareness throughout.** `wade_contrast()` currently takes integer
  column indices, which no user has; it should accept sample names. This is the
  single biggest ergonomic gap in the API.

## 3. Demo notebook

The deliverable that ties 1 and 2 together, and the integration test for both.

- **Synthetic data, no download** (decided). Generated in-notebook with known
  ground truth, so every claim it makes is checkable.
- It must show the thing that motivates the method: a planted subset and a
  planted global shift that have **the same fold change**, told apart by
  `p_subset` and `affected_fraction`.
- **Method comparison.** t-test and Wilcoxon as floors, and COPA / OS / ORT /
  MOST as the honest competitors — they share the subset-detection goal and are
  a few lines each. This is where WADE's claim gets tested against alternatives
  rather than against itself.
- It should also show the failure modes on purpose: a design below the
  combinatorial floor, and a composition-distorted characterization.

## 4. A Rust kernel for the subset test

**Now the dominant cost.** Measured at 20,000 genes, 100 v 100, B = 2000: the
mean-shift null takes 5.4 s on the existing kernel, the subset test takes
**302 s** on NumPy — 98% of the runtime.

- Two passes over the permutations on the shift-corrected matrix: one for the
  bridge's null moments at every width, one for the standardized maximum.
- **Parallelize over genes, not permutations.** Each gene's two passes then
  share cache, and the moment accumulators need no cross-thread reduction — the
  alternative would need one `(genes × widths)` accumulator per thread.
- Validate against the NumPy path elementwise, the same way the existing kernel
  was: the baseline is already correct, so a disagreement has one cause.

Independent of 1–3; can be done at any point.

## 5. Scale and inference refinements

- **Memory.** The null is always materialized — `keep_null` controls retention,
  not construction. A streaming path needs two passes regardless, because the
  GPD refinement needs each refined gene's full null vector.
- **Thread cap** for the kernel, and progress reporting for multi-minute runs.
- **GPD moment fit → maximum likelihood.** Moments are poorly behaved for
  `xi > 0.5`, which is the heavy-tailed regime the refinement exists for. Two
  constraints on any upgrade: the resolution floor `1/(B · n_tail)` must survive
  it, because it is a statement about what `B` permutations can support rather
  than a numerical guard; and the `xi <= 0` exponential branch must survive,
  because a negative shape gives the GPD a hard upper bound past which a strong
  statistic collapses to machine epsilon.
- **Permutation and p-value construction** more broadly — the user has
  references and ideas to bring to this.

## 6. Packaging and release

Wheels across platforms, CI, an API documentation build, and a decision on
whether the validation simulations ship as tests, as documentation, or both.

---

## Open questions

- **What happens to the parity fixtures long-term.** They now pin the machinery
  underneath the statistic — normalization, the quantile grids, the permutation
  null, the GPD, BH — rather than the statistic itself, which has been replaced.
  That is still worth pinning. Whether `reference/R/` and the renv sandbox stay
  once nothing new will be generated from them is a separate call; deleting them
  would leave `tests/fixtures/` unfalsifiable.
- **Whether `w1` earns its place.** It is reported and tested but nothing
  consumes it.
- **What to do about composition.** Library-size normalization couples genes, so
  a signal-saturated matrix distorts `affected_fraction` and `direction` (the
  subset *test* is immune — its statistic is invariant to a global offset). No
  method on normalized data escapes this; the question is whether WADE should
  detect and warn.
