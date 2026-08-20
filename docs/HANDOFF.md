# Session handoff

Written to let a new session resume without re-deriving anything. Read
[`../README.md`](../README.md) and [`method.md`](method.md) first; this file
covers only what is *not* obvious from them.

---

## 1. Where things stand

The statistic is finished and tested, both permutation loops have Rust
kernels, the subset stage is count-native, and the plotting layer exists.
**617 tests pass in about 12 seconds.**

```bash
export PATH="/usr/local/bin:$PATH"          # only if you need R
conda activate wade
pytest -q                                   # everything
pytest -q -m "not slow"                     # skip the validation simulations
pytest -q -m "not kernel"                   # if the Rust kernel is not built
```

Speed, measured at 20,000 genes, 100 v 100, B = 2000 on a 16-core M3 Max:
a full `wade()` call is about **17 s** (mean-shift null 5.0 s, subset test
2.9 s, the thinning fit ~7 s, the rest under a second); `thin=False` is 9.7 s
and `n_boot=200` adds ~24 s. Before the subset kernel the subset test alone
was about 300 s.

The package is `src/wade/`. Twelve modules, all small:

| module | what it owns |
|---|---|
| `quantiles.py` | the probability grid and type-7 quantiles |
| `stats.py` | `wade_stats()` — the grids, `mean_shift`, `w1`, `fc` |
| `subset.py` | the bridge, the subset test, `affected_fraction`, `direction`, the bootstrap |
| `thinning.py` | the count-native shift correction: `fit_fold_change`, `thin_counts`, `one_count` (§10.3–10.4) |
| `permutation.py` | the nulls; dispatches to the Rust kernel |
| `pvalues.py` | empirical p, GPD refinement, BH, `alternative` |
| `normalize.py` | `tpm_like` (ported), `cpm`, `rle` (new) |
| `diagnostics.py` | `wade_gene()` — the per-gene curves for plotting |
| `api.py` | `wade()`, `wade_from_matrix()`, `wade_contrast()`; `WadeResult` now carries `cond` and has `gene_index()` / `gene_detail()` |
| `io.py` | the data boundary: `as_counts` (arrays, frames, sparse), `condition`, `to_frame`, `write_results` + manifest. **No file readers, by decision** |
| `plotting.py` | `plot_gene()`, `plot_volcano()`, `plot_stages()`; a pure-NumPy data layer (`gene_panels`, `volcano_data`, `stages_data`) and two thin renderers, plotly and matplotlib, imported inside the functions |
| `rust/src/lib.rs` | both permutation kernels: `null_statistics` (mean shift) and `subset_null` (the subset test) |

## 2. What is NOT built, in priority order

See [`../ROADMAP.md`](../ROADMAP.md) for the queue and
[`scaling.md`](scaling.md) for the reasoning behind item 1.

1. **Performance and memory — this is the next session's work.** The user has a
   real dataset at **~30,000 genes x ~80,000 samples**. WADE cannot run it:
   measured, it needs **~3.7 hours and ~163 GB** against 128 GB of RAM. The
   wall is not speed — `m = min(n_case, n_ctrl)` becomes 40,000, so every
   `(genes x m)` array is the size of the data matrix. ROADMAP §2 has the
   six-item queue; `scaling.md` has every measurement, hypothesis and risk.
2. **Demo notebook** — nothing exists. `tools/make_readme_figures.py` already
   builds the dataset and the three figures it should open with. Deferred
   deliberately: the user wants the large dataset unblocked first, so the
   notebook can be a real analysis rather than a synthetic one.
3. **Format helpers** (featureCounts, MatrixMarket) and **pandas coverage** —
   ROADMAP §4, both small.

Four standing decisions from the user, recorded in `ROADMAP.md`'s preamble,
`method.md`'s scope note and `wade/io.py`'s module docstring:

* **WADE does not own I/O.** No file readers, ever, beyond thin helpers for
  genuinely established count formats. It accepts arrays, DataFrames and
  sparse matrices; your reader reads. The user's words: "I just don't want the
  wade method to have to own the I/O process."
* **WADE is for discrete count data.** Continuous input is not a target and is
  neither tested nor tuned for; `wade_from_matrix` and `thin=False` run on it
  and carry caveats. Do not add features for it.
* **Behaviour before performance** — and as of 2026-08-19 behaviour *is*
  settled, so performance is now the priority and the large dataset is the
  substrate for it.
* **Nothing in the performance work may change an answer**, with one stated
  exception (the grid cap). `pytest -q` is the check.

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
  every statistic, `Scattergl` is used past 2,000 points, `FigureWidget` makes
  linked views possible later. Static export needs `python-kaleido` *and* a
  Chrome/Chromium on the machine — it works on this laptop, it is friction on
  a headless box.
- **matplotlib** is the static/publication backend: PDF/SVG with nothing but
  the library. `tools/make_readme_figures.py` uses it so the committed PNGs
  are reproducible.
- **altair** was the other serious candidate (declarative, `vl-convert`
  exports without a browser) and lost on the 5,000-row default cap and
  SVG rendering of 20,000-point clouds; **plotnine** would have been natural
  for an R-origin user but pulls pandas. Neither is wired in; the data layer
  makes adding one a ~100-line renderer.

Both renderers read the same dataclasses, so a figure says the same thing in
either — and every dataclass has a `table()` (the arrays, for any other tool,
and the table-view twin of the chart). `tests/test_plotting.py` pins the
contract that `import wade` imports neither library.

### The documentation map

| file | what it is for |
|---|---|
| [`../README.md`](../README.md) | the user-facing contract |
| [`method.md`](method.md) | what WADE computes and why — §10 is the scale/counts derivation |
| [`limits.md`](limits.md) | what it cannot do; read before running |
| [`scaling.md`](scaling.md) | **the research agenda for large data** — measurements, hypotheses, risks, single cell |
| [`implementation-notes.md`](implementation-notes.md) | R parity, cross-language traps, the kernels |
| [`../ROADMAP.md`](../ROADMAP.md) | the work queue |
| this file | what is not obvious from any of the above |

## 3. Things that will bite you, learned the hard way

Each of these cost real time to discover. They are not in the code comments
because they are about *reasoning*, not implementation.

### Composition couples every gene

Library-size normalization divides each gene by a column total every gene
contributes to. **Any simulation with a large fraction of strongly differential
genes will produce nonsense null genes.** Measured: five strongly-up genes among
205 moved null genes' `direction` from ~0 to near −1, because the case libraries
inflated 8.4% and every other gene's log-ratio shifted by log2(1.084).

I wasted a cycle on this twice — once diagnosing a "bug" that was the effect,
once writing a test whose premise it violated. **When simulating, keep the
signal fraction realistic (under ~10%), or use `wade_from_matrix` to bypass
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
`tools/make_readme_figures.py` had 45 signal genes among 600 nulls: 7% of
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
| Stage 2's shift correction is binomial thinning of raw counts (observed and null), `f̂` by middle-half matching after thinning | `method.md` §10.3 |
| The `R` curve carries a one-count pseudocount — `log(x+1)`, the user's choice over a half-count floor, measured equal-or-better | `method.md` §10.4 |
| `wade_from_matrix` keeps the division and says so; `thin=False` exists for continuous data | `method.md` §10.3 |

The detector and shape-test prototypes were deleted once their findings were
documented; they are recoverable at commits `dcba920` and `664cfd5` if a design
choice ever needs re-litigating with the original evidence.

## 5. Suggested first move

**Performance and memory, against the real dataset.** Read
[`scaling.md`](scaling.md) end to end first — it is the research agenda, and
every number in it was measured on 2026-08-19 — then work ROADMAP §2 in order:
the grid cap and gene chunking unblock the dataset at all; the GEMM path and
the one-sort-per-gene kernel make it fast; the fold-change fit is the largest
single remaining term.

Two rules for that work, both from the user:

* **Measure before and after, on the real dataset.** Every number in
  `scaling.md` came from a script; add yours the same way, and put the
  measurement in the doc rather than in a commit message.
* **Nothing there may change an answer** — except the grid cap, which changes
  `affected_fraction`'s resolution and the `mean_shift` quadrature. That one
  needs its own entry in `method.md` §1 and the realized `m` recorded in the
  result and the manifest.

To watch the wall before touching anything:

```bash
conda activate wade
python - <<'PY'
import numpy as np, time, wade
rng = np.random.default_rng(0)
G, n, B = 1000, 6000, 200          # 20 s, 1 GB — then scale n and watch it break
cond = np.r_[np.ones(n, int), np.zeros(n, int)]
counts = rng.poisson(rng.gamma(10, 5, (G, 1)), size=(G, 2 * n)).astype(float)
t = time.perf_counter(); res = wade.wade(counts, np.ones(G), cond, nperms=B)
print(f"{time.perf_counter() - t:.1f}s, m = {res.nprobs}")
PY
```
