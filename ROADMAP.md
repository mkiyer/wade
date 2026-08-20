# Roadmap

What is planned, in dependency order, with the reason for the order stated where
it is load-bearing. This is a work queue, not a changelog: what has been built is
described in [`README.md`](README.md) and specified in
[`docs/method.md`](docs/method.md).

**Scope, which decides what is in and out of this list: WADE is for discrete
count data.** Continuous input is not a target; it is neither tested nor tuned
for, and no item here exists to support it. And **behaviour before
performance** — the speed items are grouped in §3 and deliberately come after
anything that changes an answer.

The goal that governs everything below: **a scientist should be able to load
their data, run WADE, understand the answer, and show it to someone.** The
statistic is done and so are the figures; the path from a file to a result is
not.

Done since the last revision, so no longer here: the data-in/results-out
boundary — `as_counts`, `condition`, name-aware `wade_contrast`, `to_frame`,
`write_results` and the JSON manifest (`wade.io`; **WADE reads no files**, and
that is a decision, not a gap); the plotting layer
(`plot_gene`, `plot_volcano`, `plot_stages`, two backends on one data layer —
see the README); the Rust kernel for the subset test (bitwise against the
NumPy path, about 80× faster; `docs/implementation-notes.md` §4); and the
count-native subset stage — binomial thinning, the one-count pseudocount and
bootstrap intervals (`docs/method.md` §10, `wade.thinning`).

---

## 1. Demo notebook

The deliverable that ties plotting and the data boundary together, and the
integration test for both.

- **Synthetic data, no download** (decided). Generated in-notebook with known
  ground truth, so every claim it makes is checkable.
- It must show the thing that motivates the method: a planted subset and a
  planted global shift that have **the same fold change**, told apart by
  `p_subset` and `affected_fraction`.
- **Method comparison.** t-test and Wilcoxon as floors, and COPA / OS / ORT /
  MOST / LSOSS as the honest competitors — they share the subset-detection
  goal and are a few lines each — plus waddR's location/size/shape
  decomposition, the nearest Wasserstein relative. This is where WADE's claim
  gets tested against alternatives rather than against itself. Include a
  **global shift at low counts** scenario: it is the one where, today, WADE
  loses to its own claim (item 1).
- It should also show the failure modes on purpose: a design below the
  combinatorial floor, and a composition-distorted characterization.
- `tools/make_readme_figures.py` is a ready-made starting point: its design
  (300 v 300; 600 null, 15 global 2×, 15 subsets of 15% at 8×, 15 subsets of
  5% at 8×) already shows the same-fold-change/different-shape pair and the
  5% subsets sitting at the floor.

## 2. Performance and memory — the 80,000-sample target

**The next session's work.** Measurements, hypotheses, risks and the reasoning
behind every item here are in [`docs/scaling.md`](docs/scaling.md); this is
only the queue. The target is a real dataset: **~30,000 genes x ~80,000
samples**, which today needs **~3.7 hours and ~163 GB** and therefore does not
run at all.

The wall is not speed. `m = min(n_case, n_ctrl)` becomes 40,000, so every
`(genes x m)` array is the size of the data matrix.

1. **Cap the quantile grid** (`max_probs`, default 2,000). Divides every
   `(genes x m)` array by 40x. Measured cost: faithful to a 0.1% subset at
   m = 1,000; the rule is `m ~ 2.5 / smallest fraction of interest`. Changes a
   documented property of the design — `method.md` §1 — so it must be stated
   there and the realized `m` recorded in the result and the manifest.
2. **Chunk over genes.** Peak memory must not depend on G. No numerical
   consequence; the jitter must be indexed per chunk, never redrawn.
3. **Stage 1 as one GEMM on balanced designs.** `mean_shift` is *exactly* the
   difference of group means when balanced (2.1e-13), so the whole null is
   `X @ W`: measured **139–185x**, ~40 s at the target. Agrees to 1e-9, not
   bitwise, so it is opt-in beside the parity-pinned kernel.
4. **One sort per gene in the mean-shift kernel** — the subset kernel's trick,
   ~10x, and it covers the unbalanced designs the GEMM path does not.
5. **The fold-change fit**, which is the largest single term (15.6 s of 20.3 s
   at n = 6,000): bracket from the interquartile-mean ratio, fit in count
   space, or move it into the kernel. Must stay unbiased.
6. **Parallelize `n_boot`**, and consider float32 *storage*.

Target after all of it: **single-digit minutes**.

## 3. Inference refinements

- **Stage 1's scale is settled** (`method.md` §10.1): linear, measured best or
  tied for every alternative tried. Do not add a transform knob to stage 1.
- **The (μ = 2, 1000 v 1000) cell of §10.3** was measured at 0.10 on 60 genes.
  Re-measure with a few hundred before it is quoted anywhere outside the
  method doc.
- **P-value resolution at scale** — [`docs/scaling.md`](docs/scaling.md) §4.
  At 30,000 genes BH needs ~1e-6, which is today's floor, so p-values will pile
  up and stop ranking anything. Three candidates, one experiment to choose
  between them: GPD (today), fgsea-style **adaptive multilevel splitting**
  (unbounded resolution with an explicit error bar, but per-gene chains, so it
  gives up "one shuffle serves all genes"), and a **saddlepoint approximation**
  for stage 1 (unbounded resolution, `O(n)` per gene, no permutations — but
  only for a linear statistic). The deciding experiment, and what to measure,
  is specified there.
- **GPD moment fit → maximum likelihood.** Moments are poorly behaved for
  `xi > 0.5`, which is the heavy-tailed regime the refinement exists for. Two
  constraints on any upgrade: the resolution floor `1/(B · n_tail)` must survive
  it, because it is a statement about what `B` permutations can support rather
  than a numerical guard; and the `xi <= 0` exponential branch must survive,
  because a negative shape gives the GPD a hard upper bound past which a strong
  statistic collapses to machine epsilon.
- **Permutation and p-value construction** more broadly — the user has
  references and ideas to bring to this.

## 4. Format helpers, and other small things

Small, and none blocks the notebook.

- **featureCounts and MatrixMarket helpers** — wanted (the user asked for
  them), low priority, and each a thin wrapper: featureCounts is
  `as_counts(pl.read_csv(path, separator="\t", comment_prefix="#"),
  sample_columns=[...])` with the five annotation columns known in advance;
  MatrixMarket is `scipy.io.mmread` plus the barcode/feature label files.
  Anything beyond these two needs a reason — the default answer to "can WADE
  read X?" is "polars can, and then WADE accepts it".
- **`pandas` is untested.** The frame support is duck-typed on `.columns` and
  `.to_numpy()`, which pandas satisfies, but pandas is not in the environment
  so nothing exercises it. Add it as a test-only dependency and parametrize
  `tests/test_io.py`'s frame fixtures over both.

- **Draw the bootstrap intervals**, not just print them in the subtitle: an
  error bar on the volcano's colour, or a band in `plot_gene`.
- **Linked views** in the plotly backend: click a point on the volcano or the
  two-stage plot and see its `plot_gene` panel. `go.FigureWidget` (anywidget)
  in a notebook makes this a callback; the data layer already has everything.
- **A dark theme.** Both renderers share one light palette; a dark one is a
  second token set, not a redesign.
- **Label de-collision** is deliberately minimal (alternate above/below along
  x). If the static figures need more, `adjustText` would be a third optional
  dependency — decide whether it earns that.

## 5. Packaging and release

Wheels across platforms, CI, an API documentation build, and a decision on
whether the validation simulations ship as tests, as documentation, or both.

---

## Open questions

- **Single cell and spatial** — [`docs/scaling.md`](docs/scaling.md) §5. Sparsity
  is an *opportunity*: a gene's sorted values are a run of zeros then the sorted
  non-zeros, so the per-permutation work becomes `O(nnz)` and the grid cap of §2
  becomes essential rather than merely helpful. The trap is statistical, not
  computational: **cells are not exchangeable, donors are.** A million cells
  from twenty donors is twenty samples. Either pseudobulk, or permute donor
  labels with cells kept together — the second is a genuine research direction,
  because "is this gene altered in a subset of *cells* within the affected
  donors?" is a question WADE's shape stage is already built to ask.

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
