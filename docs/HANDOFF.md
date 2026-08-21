# Session handoff

Written to let a new session resume without re-deriving anything. Read
[`../README.md`](../README.md) and [`method.md`](method.md) first; this file
covers only what is *not* obvious from them.

---

## 1. Where things stand

The statistic is finished and tested, both permutation loops have Rust
kernels, the subset stage is count-native, and as of 2026-08-20 **the
performance queue is done**: the 30,000 × 80,000 target that needed ~163 GB
and did not run completes, measured, in **29.7 minutes at a 55.6 GB peak**
(B = 2,000, all fast paths; `scaling.md` §1). Since then the package has been
audited, corrected and **cut down**: ~1,300 lines deleted, five redundant
paths collapsed to one, and the documents rewritten to match the code.

As of 2026-08-21 the **implementation queue is empty**: the last six core items
landed, `plotting.py` became the `wade/plotting/` subpackage, and the whole
visualization queue is built — themes, drawn bootstrap intervals, a driver
figure, per-gene metadata in figures, linked views, a stage-1 row, and label
de-collision. **738 tests pass in about 12 seconds**, parity unchanged at
**9.155e-15** over 436 comparisons.

**The immediate work is the three notebooks** — `ROADMAP.md` §1 and §3. Each is
a separate deliverable: the `demo` first (the showcase, the README in executable
form), then `benchmark` (which the real cohort's rank-recovery work is waiting
on), then `rna100k` relabelled as manuscript material. `plan.md` was the recipe
for phases B and C and was deleted when C landed, as it said to; its outcome is
in `ROADMAP.md`, `docs/plotting.md` and the code.

```bash
export PATH="/usr/local/bin:$PATH"          # only if you need R
conda activate wade
pytest -q                                   # everything
pytest -q -m "not validation"               # skip the simulations (8 s vs 12 s)
pytest -q -m "not kernel"                   # if the Rust kernel is not built
```

Speed, measured on the 16-core M3 Max: a full `wade()` at 1,000 genes,
6,000 v 6,000, B = 200 went **19.8 s → 2.27 s** over the day (grid cap +
new mean-shift kernel + affine fit + opt-in `stage1="gemm"` and
`fit_backend="rust"`); 5,000 genes with `gene_chunk=1000` went 104 → 11.7 s
at 3.7 GB peak instead of 7.2. `n_boot=200` at 20,000 × (100 v 100) went
19.7 → 4.0 s (threaded, bit-identical). Every measurement is in
`docs/scaling.md` beside the hypothesis it tested; `tools/bench_scaling.py`
reproduces them.

The new `wade()` surface, all documented in the docstring: `max_probs`
(grid cap, default 2,000 — **the one deliberate answer change**, stated in
`method.md` §1, realized `m` in the result and manifest), `gene_chunk`
(block size for the one counts driver; `None` is a single block), `stage1="gemm"`
(balanced designs, the exact mean difference as one matrix product, opt-in),
`fit_backend="rust"` (the fold-change fit's bisection in the kernel, opt-in
because it is *not* bitwise against NumPy — see §3 below).

The package is `src/wade/`. Ten modules, the `plotting/` subpackage, and the
kernel:

| module | what it owns |
|---|---|
| `quantiles.py` | the probability grid, type-7 quantiles, `capped_nprobs` |
| `stats.py` | `wade_stats()` — the grids, `mean_shift`, `w1`, `fc` |
| `subset.py` | the bridge, the subset test, `affected_fraction`, `direction`, `subset_log2_fc` (the subset's magnitude), the bootstrap (threaded, chunkable) |
| `thinning.py` | the count-native shift correction: `fit_fold_change` (one affine `alpha=` path, NumPy or kernel backend), `midmean`, `thin_counts`, `one_count`, `gene_chunks` (§10.3–10.4) |
| `permutation.py` | the nulls; dispatches to the Rust kernel; `mean_diff_stat` / `mean_diff_null` (the GEMM stage 1); `draw_perms(strata=)`, `permutation_space()` (restricted permutation) and `detectability_floor()` (`limits.md` §1's pre-flight check) |
| `pvalues.py` | empirical p, GPD refinement, BH, `alternative` |
| `normalize.py` | `tpm_like` (ported) and the jitter. **One normalizer**: CPM is `normalizer=1.0`, other size factors go in via `lib_sizes=` — `cpm`/`rle` were deleted as unreachable from `wade()` |
| `diagnostics.py` | `wade_gene()` — the per-gene curves; `library_qc()` — per-library depth/complexity/concentration; `subset_drivers()` — which samples drive a gene's subset |
| `api.py` | `wade()` and `wade_contrast()` — **one driver**; `WadeResult` now carries `cond` and has `gene_index()` / `gene_detail()` |
| `io.py` | the data boundary: `as_counts` (arrays, frames, sparse), `condition`, `to_frame`, `write_results` + manifest. **No file readers, by decision.** Per-gene columns are carried in `Counts.meta` for display and read by no statistic |
| `plotting/` | **an extension, not the core** (`plotting.md`): five figures — `plot_gene()`, `plot_volcano()`, `plot_stages()`, `plot_drivers()`, `plot_linked()` — over a pure-NumPy data layer (`gene_panels`, `volcano_data`, `stages_data`, `driver_panel`, each with `.table()`). `theme.py` holds every colour and label and imports nothing; `data.py` imports no backend; `_plotly.py` / `_matplotlib.py` import their library **inside** their functions; `__init__.py` resolves the backend and the theme. Was one 1,124-line module until 2026-08-21 |
| `rust/src/lib.rs` | three kernels: `null_statistics` (mean shift, gene-major since 2026-08-20), `subset_null` (the subset test), `fit_bisect` (the fold-change fit, **opt-in and not bitwise** — see §3) |

## 2. What is NOT built, in order

The queue is [`../ROADMAP.md`](../ROADMAP.md). In short:

1. **Three notebooks**, and this is the current work: a small-synthetic `demo`
   (the showcase), a `benchmark` head-to-head against COPA / OS / ORT / MOST /
   LSOSS, t-test, Wilcoxon and waddR on simulated *and* literature data, and
   `rna100k` relabelled as manuscript material rather than a demo.
2. **The real dataset's own agenda** — `scaling.md` §7, which is where
   everything `notebooks/rna100k.qmd` measured now lives: the saturation of
   significance and the two controls that explain it, the artefact libraries
   and the diagnostic that was measured and *refuted*, group-associated depth,
   the observed p-value pile-up, 47% sparsity. The rank-recovery benchmark on
   real background is the priority there, and it wants the `benchmark`
   notebook first.
3. **The resident-matrix question** — `scaling.md` §2.3. The driver's peak is
   four full `genes × samples` matrices: the caller's `counts` plus the three a
   `WadeResult` carries (`tpm`, `jitter`, `pseudocount`). The jitter is a seeded
   stream and the pseudocount a rank-1 product, so neither *has* to be
   materialized. A contract question about what a result carries, not a kernel
   question — and note the answer interacts with `plot_drivers`, which needs the
   raw counts precisely **because** the result does not carry them.

Four standing decisions from the user, recorded in `ROADMAP.md`'s preamble,
`method.md`'s scope note and `wade/io.py`'s module docstring:

* **WADE does not own I/O.** No file readers, ever, beyond thin helpers for
  genuinely established count formats. It accepts arrays, DataFrames and
  sparse matrices; your reader reads. The user's words: "I just don't want the
  wade method to have to own the I/O process."
* **WADE is for discrete count data.** Continuous input is not a target and is
  neither tested nor tuned for. `wade_from_matrix` was deleted 2026-08-21 for
  offering a stage-2 test known broken at low counts; `thin=False` remains
  only as the comparison that shows why thinning exists.
* **Behaviour before performance** — behaviour settled 2026-08-19, the
  performance queue landed 2026-08-20. The rule that governed that work
  stands for any future performance change:
* **Nothing in the performance work may change an answer**, with one stated
  exception (the grid cap, `method.md` §1). `pytest -q` is the check. The
  two opt-in fast paths (`stage1="gemm"`, `fit_backend="rust"`) change
  numbers only when explicitly chosen, and say what they change.

### Where WADE is going, so today's choices are not local

The user's horizon is **single cell and spatial transcriptomics** — thousands
to millions of cells, sparse and zero-inflated. Not to be built yet, and
`scaling.md` §5 has the detail, but two things are worth carrying while doing
the performance work:

* **Sparsity is an opportunity, not just a storage format.** A gene's sorted
  values are a run of zeros followed by the sorted non-zeros, so quantiles and
  the per-permutation partition become `O(nnz)`. The grid cap and the gene
  chunking are exactly the two changes that make that regime reachable — so do
  them in a way that generalizes, not one that hard-codes dense arrays.
* **Cells are not exchangeable, donors are.** A million cells from twenty
  donors is twenty samples. Permuting cell labels tests a hypothesis nobody
  holds. This is `limits.md` §2.2 at extreme multiplicity, and no amount of
  engineering removes it.

### The scale question, and what came of it

The user asked whether WADE should run on the linear or the log scale, having
found subsets "much more pronounced" in linear space before. The derivation and
the measurements are `method.md` §10. Outcome:

- **Stage 1 is linear and stays linear** — measured best or tied for every
  alternative tried (5% subset at 8×, μ = 2: power 0.73 linear, 0.07 log), and
  the transform cannot change its level because permutation is exact for any
  statistic. No knob was added.
- **Stage 2 must be log** (the only scale on which a multiplicative shift is
  flat) — but a fold change in NB counts is not a multiplicative shift at low
  expression, so the division correction of §3 was firing on genuine global
  shifts: 0.95 at 2 counts, 0.72 at 20 counts with 1000 v 1000, **growing with
  n**. The earlier validation used continuous lognormal data and never
  exercised counts.
- **Fixed** (`wade.thinning`, `method.md` §10.3–10.5): binomial thinning of the
  raw counts under a thinning-matched fold change, applied to the observed
  statistic *and* the null; a one-count pseudocount on the log-ratio curve; and
  opt-in bootstrap intervals. False-subset rate now 0.02–0.05 everywhere.

### The plotting layer, as built

Two backends were evaluated against the field (plotly 6.9, matplotlib 3.11,
altair 6.2, bokeh 3.9, plotnine 0.15 on conda-forge as of August 2026) and
two were kept:

- **plotly** is the default when installed: hover carries the gene name and
  every statistic, `Scattergl` is used past 2,000 points, and `FigureWidget`
  is what `plot_linked` is built on — which needs **`anywidget`** (installed
  here, `pip install 'wade[linked]'` elsewhere) and a live kernel. Static
  export needs `python-kaleido` *and* a Chrome/Chromium on the machine — it
  works on this laptop, it is friction on a headless box.
- **matplotlib** is the static/publication backend: PDF/SVG with nothing but
  the library. `notebooks/demo.qmd` renders the committed PNGs with it, so they
  are reproducible.
- **altair** was the other serious candidate (declarative, `vl-convert`
  exports without a browser) and lost on the 5,000-row default cap and
  SVG rendering of 20,000-point clouds; **plotnine** would have been natural
  for an R-origin user but pulls pandas. Neither is wired in; the data layer
  makes adding one a ~100-line renderer.

Both renderers read the same dataclasses, so a figure says the same thing in
either — and since 2026-08-21 that includes **where the labels go**, which is
computed once in the data layer in normalized axis units precisely so the two
backends cannot drift. All four dataclasses now have a `table()`: the arrays,
for any other tool, and the table-view twin of the chart.
`tests/test_plotting.py` pins the contract that `import wade` imports neither
library, and a second test greps the whole package for a hex or `rgb()` string
outside `theme.py`.

The stance is [`plotting.md`](plotting.md): this layer is an extension, the
results table is the interface, and exporting to ggplot is a first-class path
rather than a fallback. It also carries the two limits worth knowing before
promising anything — a linked view does not survive export to HTML, and a dozen
labels on near-coincident points is past what any placement rule can fix.

### The documentation map

| file | what it is for |
|---|---|
| [`../README.md`](../README.md) | the user-facing contract |
| [`method.md`](method.md) | what WADE computes and why — §10 is the scale/counts derivation |
| [`limits.md`](limits.md) | what it cannot do; read before running |
| [`scaling.md`](scaling.md) | **the research agenda for large data** — measurements, hypotheses, risks, single cell, and what the real cohort measured |
| [`plotting.md`](plotting.md) | the plotting **extension** — the data layer, plotting elsewhere, and what it is not for |
| [`implementation-notes.md`](implementation-notes.md) | R parity, cross-language traps, the kernels |
| [`../ROADMAP.md`](../ROADMAP.md) | the work queue |
| this file | what is not obvious from any of the above |

## 3. Things that will bite you, learned the hard way

Each of these cost real time to discover. They are not in the code comments
because they are about *reasoning*, not implementation.

### NumPy reductions are layout-stable only conditionally

The chunking contract (`gene_chunk` bit-identical to one-pass) survives on
three measured facts, and breaking any of them breaks it silently:

- **Column fancy-indexing returns F-ordered arrays.** `xs[:, lo_i]` in
  `type7_quantiles` makes `Q1`, `Q0` and `D` F-ordered, and a row reduction
  over an F-ordered array chooses its strategy by the row count: a `(1, m)`
  array sums differently (last-ulp) than the same row inside a bigger
  matrix. Measured on `D.sum(axis=1)` at m = 40. Hence **no chunk is ever a
  single row** — `gene_chunks` refuses `gene_chunk=1` and folds a one-row
  remainder into the previous chunk. Fresh **C-contiguous** reductions are
  row-count-independent (verified k = 1..24, two shapes, pinned by
  `tests/test_chunking.py`), which is why `_middle_mean` forces contiguity.
- **Sequential chunked `Generator` calls reproduce a full-matrix call's
  stream exactly** (uniform and binomial both) — that is what makes the
  chunked jitter, fit and thinning bitwise. But the *layout* of consumption
  is part of the answer: the fit re-seeds two streams per bisection step
  (case and control), and `thin_counts` draws all case blocks then all
  control blocks, so their chunked versions must consume in exactly that order
  (iteration-outer/chunk-inner; two phases). Reordering either changes every
  fitted fold change.
- **A column sum's association depends on blocking**, so library sizes are
  always one full-matrix pass, never chunked.

### The fit kernel is deliberately not bitwise

`fit_bisect` (opt-in `fit_backend="rust"`) cannot reproduce NumPy's binomial
draws — no two samplers consume randomness alike — so for a given seed the
two backends' `f̂` differ by up to a bisection cell (measured max |log ratio|
0.035). That is why it is **never auto-dispatched**: `backend="auto"` must
never let the machine pick the answer. Its guarantees are different ones:
deterministic given the seed, chunk- and thread-invariant (each gene's
stream is a function of `(seed, global gene index)` alone), and statistically
held to the NumPy path. The same logic gives `stage1="gemm"` its caveat: BLAS
blocking depends on matrix shape, so gemm mode is allclose (1e-12) rather
than bitwise between chunked and unchunked.

### Composition couples every gene

Library-size normalization divides each gene by a column total every gene
contributes to. **Any simulation with a large fraction of strongly differential
genes will produce nonsense null genes.** Measured: five strongly-up genes among
205 moved null genes' `direction` from ~0 to near −1, because the case libraries
inflated 8.4% and every other gene's log-ratio shifted by log2(1.084).

I wasted a cycle on this twice — once diagnosing a "bug" that was the effect,
once writing a test whose premise it violated. **When simulating, keep the
signal fraction realistic (under ~10%), or pass `lib_sizes=np.ones(n)` to bypass
normalization entirely.** The subset *test* is immune (its statistic is
invariant to a global offset); the characterization is not.

### Claims about the statistic must be measured, not reasoned

Three assertions I wrote from theory turned out to be wrong when run:

- The scan's argmax "obeys the arcsine law and is unstable" — false for the
  *bridge*, which is pinned at both ends. It is stable but not shape-robust.
- "π̂ works on the log scale" — true for global changes, false for small subsets
  until the fourth moment replaced the second.
- "Permutation gives the right null for the subset test" — false, and it cost
  14–19% false positives on genuine fold changes.

The pattern: build a planted-ground-truth simulation and check, before writing
it down.

### The R sandbox is fiddly

Only needed to regenerate `tests/fixtures/`. Three load-bearing details in
[`../reference/R/README.md`](../reference/R/README.md): R 4.6.1 is not on the
default `PATH`, you must run from `reference/R/`, and `renv::restore()` succeeds
and *then* errors on a socket — run it twice. Fixtures are committed, so the
suite runs with no R present.

### plotly has quiet failure modes

Three that cost time this session; none raise.

- **`add_hline` / `add_vline` / `add_vrect` with `row=`/`col=` skip subplots
  that do not yet contain data** (`exclude_empty_subplots=True` is the
  default). Add shapes *after* the traces, or the shading for the untested
  half of a one-sided volcano silently never appears.
- **An explicit axis range on one axis breaks autorange on axes `matches`-ed
  to the other.** Setting `range=[-0.02, 1.02]` on the shared x of
  `plot_gene` made every matched log-ratio y axis render as `[-1, 4]`,
  clipping curves. Leave `p` to autorange; it already spans `[0, 1]`.
- **`shared_yaxes="rows"` in `make_subplots` shares every row.** For "share
  the top row only", use `fig.update_yaxes(matches="y", row=1, col=j)`.
- **`make_subplots` silently drops an empty `subplot_titles` entry**, so the
  annotation indices are not the subplot indices and there is no spare slot to
  write into later. `plot_linked` wanted a title *and* a subtitle for its panel
  and got one annotation for both — put the two lines in one string, as
  `_gene_plotly` does.

And one for matplotlib: a colour bar is an `Axes`, so `len(fig.axes)` is one
more than the number of panels.

### The subset stage and the number of permutations

`B = 500` is enough for the mean-shift stage and not for a figure of the
subset stage: the empirical floor is `1/(B+1)` and the GPD only refines past
it, so with a few hundred genes no subset gene clears BH at 0.05 and
`plot_stages` has an empty upper-right quadrant. Use `B = 2000` — it costs
two seconds now. Separately, a planted 5% subset in 200 or 300 cases sits
near the combinatorial floor (about `1e-3` for 10 of 200) and will be found
at raw `p` but not at FDR across a few hundred genes; that is the floor doing
its job, and the figure script plants a 15% subset alongside so the "both
stages significant" quadrant has something in it. Note also that a 15%
subset at 8× has the *same* `log2_fc` as a global 2× — which is exactly the
pair the README's figures show.

### Traps in the count-native subset stage

All measured, all cost time:

- **Thinning must be applied to the observed statistic as well as the null.**
  The division could correct only the null, because the bridge is exactly
  invariant to division. It is *not* invariant to thinning — that is the
  point — so both sides are computed on the thinned matrix. `SubsetResult`
  keeps both curves: `r` (observed, for the characterization and the plot)
  and `r_test` (what the statistic was computed on).
- **A robust fold-change estimator is not automatically an unbiased one.** The
  interquartile-mean ratio reads 2.08 for a true NB 2× — the middle of a
  skewed distribution does not scale with its mean — and at 1000 v 1000 that
  bias alone is a 0.07–0.15 false-subset rate. An *inaccurate* `f̂` is
  **anti-conservative** (30% off → 0.26–0.28), so unbiasedness is the
  requirement; contamination by a large subset is not, because such a gene is
  a mixture and is correctly called one.
- **Fit on the jitter-free scale.** With the jitter included, an all-zero gene
  has a nonzero "higher" group and the bisection thins noise for sixteen
  steps, returning `f̂ = 1024`. `wade()` passes the fit a jitter-free
  normalizer for this reason.
- **Inverse-variance node weights are label-free but not signal-free** and
  were rejected: the subset's own values inflate the null variance of the top
  nodes, so weighting erases the signal region (weighted `affected_fraction`
  read 0.00 for genuine subsets).
- **The validation suite generates counts now.** `tests/test_subset.py` used
  to generate continuous lognormal values — a pure multiplicative model in
  which the division correction is exact — which is exactly why the low-count
  failure went unnoticed for so long. It now generates NB counts at 50 counts
  per sample with unit library sizes (so a normalized value *is* a count and
  the pseudocount is exactly +1), and runs the shipped default. Every property
  holds, several more tightly than before: planted fractions recover to two
  decimals (0.022, 0.049, 0.099, 0.252, 0.514), `f̂` reads 1.50 / 2.00 / 8.01
  for planted 1.5 / 2 / 8, and the level is 0.03–0.06 at every fold change.
- **The count analogue of "a variance change" is not raising the dispersion.**
  Raising NB dispersion at a fixed mean also skews the distribution — the
  median falls while the mean is held — so it reads `direction` −0.4 to −0.65
  rather than 0. The symmetric-spread scenario `direction` is meant for is a
  half-up/half-down mixture, which is what `_mk(..., "var")` now generates.

### Composition, again — and how big a demo matrix has to be

The rule from earlier sessions was "keep the signal fraction under ~10% of
**genes**". The load-bearing quantity is actually the fraction of the **library
mass**, and in a small matrix those differ. The first NB version of
the README's figure dataset had 45 signal genes among 600 nulls: 7% of
genes, 6.6% of mass, every null gene acquired a −0.09 log2 fold change and
**186 of 600 nulls cleared BH** on the mean-shift stage at 300 v 300. Raising
the null count to 3,000 put the signal at 1.5% of mass, the null median fold
change at −0.02, and the figure back to showing the method. Real matrices have
20,000 genes; toy ones have to be told.


The first draft of the figure dataset had 40 signal genes in 340 (12%) and 31
null genes passed the mean-shift test at FDR 0.05 — the case libraries had
inflated and every null gene read as slightly down. At 45 in 645 (7%) it is 2.
The `< 10%` rule from the previous session holds; it is a cliff, not a slope.

### Test-suite conventions

- Parity runs pin `alternative="greater"`, because the R oracle is one-sided and
  the port defaults to two-sided. Comparing the default against R would be
  comparing two different tests.
- `portrun.py` pins `backend="numpy"` so layers 0–7 validate the readable path;
  `test_kernel.py` holds the kernel to the same fixtures separately.
- Every comparison goes through `assert_close`, which records into a ledger
  printed at the end of the run. **Report the worst-case deviation, not "the
  assertions passed"** — that number is what distinguishes an exact port from a
  close one.

## 4. Decisions already made — do not relitigate

Reasoning is in `method.md`; this is the index.

| decision | where |
|---|---|
| Raw counts in, not a normalized matrix (the jitter needs count precision) | `method.md` §8 |
| Two stages, reported separately; no combined p-value and no categorical label | `method.md` §5 |
| The subset test's null is the **fitted global shift**, not no-difference | `method.md` §3 |
| `affected_fraction` uses the **fourth** moment on the **log** curve | `method.md` §4 |
| Two-sided by default, `alternative` for one-sided | `method.md` §9 |
| Ranking past the p-floor: `z_mean_shift`/`z_subset` (permutation z, the NES analogue — never a calibrated p) and `subset_log2_fc` (subset magnitude, quantile-matched) | `method.md` §4/§6, `scaling.md` §4.0 |
| The tail window, `tail_conc`, `F` and the rank scores are retired | `method.md` §7 |
| Higher Criticism and max-Z lost the detector comparison | commit `dcba920` |
| Names: `mean_shift`, `p_subset`, `affected_fraction`, `direction`; no aliases | commit `ff22cbb` |
| Volcano x axis is `log2_fc`, never `mean_shift` (TPM-like units, spans thousands, collapses the cloud to a line) | `plotting.py` `volcano_data` |
| Volcano/stages cutoff lines are the BH raw-p cutoff at `alpha`; when nothing passes, `alpha / G` | `plotting.py` `_bh_cutoff` |
| Plotting backends are imported inside the functions; `import wade` stays NumPy-only | `tests/test_plotting.py` |
| **WADE does not own I/O**: no file readers. It accepts arrays, DataFrames and `.toarray()` sparse matrices; polars/pandas do the reading | `ROADMAP.md` §1 |
| Labels optional (positional fallback), orientation explicit (`genes="rows"`), normalizer may name a column of the counts frame | `ROADMAP.md` §1.1–1.2 |
| Condition comes from sample metadata via `wade.condition()`; alignment is strict and names the offenders; no reloadable bundle | `ROADMAP.md` §1.3–1.4 |
| Stage 1 on the linear scale, stage 2 on the log scale; no transform knob | `method.md` §10.1–10.2 |
| The quantile grid is capped: `m = min(n1, n0, max_probs)`, default 2,000; realized `m` recorded, rule `m ≳ 2.5/π_min` | `method.md` §1, `scaling.md` §2.1 |
| Chunking is a memory layout, never a numerical choice: `gene_chunk` is bit-identical, asserted with `assert_array_equal` | `scaling.md` §2.2, `tests/test_chunking.py` |
| Fast paths that change numbers are opt-in and named: `stage1="gemm"`, `fit_backend="rust"`; `backend="auto"` never changes an answer | `scaling.md` §3.1/§3.3 |
| The fit bracket was rejected: a missed bracket is anti-conservative | `scaling.md` §3.3 |
| Bootstrap CIs stay within-group resampling with replacement; Poisson and hybrid schemes measured miscalibrated, m-of-n under-covers | `scaling.md` §6 |
| Stage 2's shift correction is binomial thinning of raw counts (observed and null), `f̂` by middle-half matching after thinning | `method.md` §10.3 |
| The `R` curve carries a one-count pseudocount — `log(x+1)`, the user's choice over a half-count floor, measured equal-or-better | `method.md` §10.4 |
| **Raw counts only** — no pre-normalized entry point; `thin=False` keeps the division as a demonstration, not an option | `method.md` §10.3 |

**Prototypes are deleted once their findings are documented**, and recovered
from git if a design choice ever needs re-litigating with the original
evidence: the detector and shape-test prototypes at commits `dcba920` and
`664cfd5`, and `prototypes/scale_and_counts.py` — the thinning / f-hat /
pseudocount / bootstrap study behind `method.md` §10, including the rejected
inverse-variance weights, half-count floor and `p_profile` variants — at
`fddf2c1`. The reproduction recipes that are still *live* stay in `tools/`
(`bench_scaling.py`, `bootstrap_scheme_study.py`), because `scaling.md`
cites them.

## 5. Suggested first move

**The `demo` notebook**, `ROADMAP.md` §3. It is the cleanest of the three: small
synthetic data with planted ground truth, every claim checkable in the output,
and nothing left to build first — the figures it needs all exist, including the
two that did not a day ago (`plot_drivers` for "should I believe this?" and
`plot_gene(cumulative_area=True)` for stage 1). It is **written**: `demo.qmd`
generates the dataset, runs it, and writes `docs/figures/*.png` — it is the
master source for the README's figures and numbers, which is why
`tools/make_readme_figures.py` was deleted. `benchmark` is the next one.

```bash
conda activate wade
pytest -q                                   # 738 passing, ~12 s — the baseline
quarto render notebooks/demo.qmd            # the demo, and the README's figures
```

Three rules govern everything here, and they have earned their place:

* **Nothing changes a reported number.** `pytest -q` is the check and the
  parity ledger printed at the end of the run is the sharper one (currently
  9.155e-15 over 436 comparisons). Fast paths that *do* change numbers are
  opt-in and named — `stage1="gemm"`, `fit_backend="rust"`.
* **Measure before and after**, and put the numbers in the document that owns
  the subject (`scaling.md` for performance), never in a commit message.
  `tools/bench_scaling.py` reproduces every performance claim.
* **Prefer deleting to adding.** A four-hunter audit removed ~1,300 lines on
  2026-08-21 and the package got better. Before adding a parameter or a code
  path, ask whether an existing one already covers the case — twice recently
  the answer was yes (CPM is `normalizer=1.0`; the chunked driver with one
  chunk *is* the one-pass driver).

To see the current state in one command:

```bash
python tools/bench_scaling.py --genes 1000 --n 6000 --nperms 200 --full \
    --stage1 gemm --fit-backend rust     # ~2.3 s; drop the flags for ~9 s
```
