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
NumPy path, about 80× faster; `docs/implementation-notes.md` §4); the
count-native subset stage — binomial thinning, the one-count pseudocount and
bootstrap intervals (`docs/method.md` §10, `wade.thinning`); and **the
80,000-sample performance queue** (2026-08-20, all measurements in
`docs/scaling.md`): the quantile-grid cap (`max_probs`, `method.md` §1), gene
chunking (`gene_chunk`, bit-identical), the balanced-design GEMM stage 1
(`stage1="gemm"`, opt-in), the one-sort-per-gene mean-shift kernel (bitwise,
6–7.5×), the fold-change fit 34× (affine path + opt-in `fit_backend="rust"`),
and the threaded bootstrap. A full `wade()` at (1,000 genes, 6,000 v 6,000,
B = 200) went 19.8 → 2.27 s; the 30,000 × 80,000 target went from
does-not-run (~3.7 h projected, ~163 GB needed) to **measured 29.7 minutes
and 55.6 GB** at B = 2,000 with every fast path on.

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

## 2. What the real data demands

The performance queue landed (preamble; `docs/scaling.md` §§2–3 for every
measurement) and `notebooks/rna100k.qmd` runs real contrasts. Two findings
from first contact set this queue, and both are measured rather than argued:

**Significance saturates.** The plasma contrast (1,343 malignant v 1,662
non-malignant, 126 s) called most of the transcriptome subset-significant.
The notebook's falsification section shows this is **not a bug** (permuted
labels on the same matrix: 0 mean-shift / 53 subset BH hits, raw
`p_subset < 0.05` at 3.9% — nominal) and **not only cross-study structure**
(a single-study 283 v 117 contrast still calls ~70%). At these `n` the
stage-2 point null rejects on every real deviation, so **rankings and
descriptors are the operative outputs** — which `subset_log2_fc` and the
permutation z-scores now provide.

**A few anomalous libraries drive many genes' subsets.** Chasing the
magnitude ranking's top found FLI1, EWSR1, TRGC2 and LYVE1 sharing 23–24 of
their top 25 driver libraries (FLI1 and the *unrelated* TRGC2 share 24 —
more than the FLI1/EWSR1 "fusion pair" — which is what falsified a tempting
Ewing-sarcoma reading). Those libraries detect 5,563 genes against 11,114
elsewhere and hold 36% of their mass in ten genes (cohort median 15%). At
cohort scale the skew is severe: **some libraries sit in the top 25 of a
third of all genes** (10,229 of 30,976; 258 expected if uniform). Meanwhile
**ETV4** passes every control — zero overlap with those libraries, ordinary
complexity, *below*-median depth, median 0 TPM across all 3,005 plasma
libraries, and its top 25 are 25 distinct patients, all PDAC, replicated
across two independent studies.

In order:

1. **Per-library QC, and driver attribution built on it — the top priority.**
   Measured 2026-08-20, and the measurement **refuted the obvious statistic**:
   "how often do this gene's drivers drive everything else?" does *not* flag
   the artefact. FLI1's and TRGC2's drivers appear in 375 of 6,769 up-gene
   subsets against a cohort median of 1,769 — *far below* average — because a
   low-complexity library is zero in ~25,000 genes and therefore ranks at the
   **bottom** of most genes' orderings. (Cross-gene recurrence is real — one
   library sits in 3,583 of 6,769 subsets — but in a different set of
   libraries, and it is not the artefact mechanism.) What separates cleanly is
   the drivers' **own library complexity**: ~5,600 genes detected for
   FLI1/EWSR1/TRGC2 against 10,545 for ETV4 and a cohort median of 11,096. So:
   * **per-library QC as first-class** — genes detected, top-10 mass share,
     depth: the same status as the combinatorial floor, properties of the
     design to look at *before* trusting a run. A small `wade` helper is
     defensible (it is about whether the null can be trusted, not about I/O).
   * **driver attribution as the substrate** — which samples sit in a gene's
     affected region — so a gene can inherit and report its drivers' QC
     profile (median driver complexity, or the gene's share of its drivers'
     library mass: FLI1 takes 12.6% of an artefact library, ETV4 ~1.5%).
   * **the reported score is still an open design question** and the user's
     call, since it is a new number in the method: driver-complexity summary,
     mass-share, or a normalized composite — to be chosen against item 3's
     benchmark rather than by argument.
2. **Rank-recovery benchmark**, now properly specified: plant known subset
   and global signals into real non-malignant plasma background **including
   the artefact libraries**, and require the ranking to put planted genes
   above artefact-driven ones. The direct test of "the best genes are the
   best genes", and the way to validate item 1's statistic.
3. **Restricted (within-stratum) permutation.** Permute labels within study
   or protocol. The defensible contrasts on this cohort need it, ETV4's
   cross-study replication is the positive control, and it is the same
   machinery single cell's donor permutation needs (`docs/scaling.md` §5.3).
4. **The depth experiment.** Group-associated depth appears in both
   contrasts (1.7× one way, 1.3× the other; cohort spans 0.8M–89M), but
   ETV4's drivers sit *below* median depth, so depth is not a simple
   monotone confound. Planted depth imbalance, false-positive rates per
   stage, guards led by depth-equalizing binomial thinning (the count-exact
   form of the user's Poisson-noise instinct).
5. **P-value resolution** (§3, `docs/scaling.md` §4) — 18.8% of genes at the
   mean-shift floor, observed. Deliberately **demoted**: it buys ordering
   that magnitude and the z-scores now supply for free, and the user's own
   framing is that magnitude matters more than p. Worth doing for
   defensible small-cohort inference, not for ranking this cohort.
6. **47% zeros**: the §5.1 sparse walk buys ~2× here, not 10× — worth
   having, not urgent.

### Performance leftovers

* **The resident-matrix question** (`docs/scaling.md` §2.3, sharpened): the
  chunked driver's peak is four full `genes x samples` float64 residents —
  counts, jitter, `tpm`, pseudocount, all carried by `WadeResult` — plus the
  int32 thinned counts. Jitter is a seeded stream and the pseudocount a
  rank-1 outer product; neither has to be materialized. A contract question
  about what a result carries, not a kernel question.
* **The subset kernel is the wall-clock wall** at full scale: B × O(n) per
  gene, intrinsic for dense data (~40 min of a ~50-min total at B = 2,000).
  The real answer is §5.1's sparse zero-run walk (the single-cell path
  anyway); the cheap dense shortcut is a smaller `B` for stage 2 than for
  stage 1.

Not taken, deliberately: the fit bracket (a missed bracket is
anti-conservative, `docs/scaling.md` §3.3) and a Poisson or hybrid bootstrap
scheme (measured miscalibrated in plausible regimes, §6).

## 3. Completing the package — the audited queue

A five-dimension audit of the package (2026-08-20, findings verified
individually against the code) returned **40 real gaps**; the nine that
blocked a release are fixed — `wade_contrast` silently misapplying every
per-sample argument to its reordered columns, `write_results` collapsing two
runs' manifests onto one path, an all-zero library normalizing every gene to
a constant, contradictory GPL/MIT metadata, the false "installs without a
Rust toolchain" claim, no CI at all, and pandas / `cpm` / `rle` /
`alternative="less"` all promised but exercised by nothing. What the audit
found and this section does not repeat is in the workflow result; what
remains is below, grouped and ordered within each group by effort.

### API surface

- **Bootstrap intervals: confidence level not settable from wade(), and no interval for subset_log2_fc** *(small)* — A user cannot ask for a 90% or 99% interval without dropping to the private call and re-deriving the pseudocount and grid cap themselves. And the one number the docs tell them to rank by carries no uncertainty at all, in exactly the low-count regime where….
- **Most public entry-point parameters are undocumented, including the ones that set the p-value floor** *(small)* — `n_tail` and `n_exc_min` set the GPD extrapolation floor `1/(B*n_tail)` that docs/limits.md §5 devotes an entire section to — a user told 'the only remedy is more permutations' is never told these knobs exist. `keep_null` is the only way to obtain the null….
- **No helper for the combinatorial detectability floor the docs require users to compute before running** *(small)* — The single check the documentation says decides whether WADE can answer the question at all — and the reason 'a study with 18 controls cannot find a 5% subtype' — must be hand-coded from a doc snippet, and the closest available function returns a similarly….
- **Stage 2 has no exceedance or GPD-refinement diagnostics** *(small)* — docs/limits.md §5 makes a reading rule out of these: 'Treat the floor as a censoring point... A gene sitting *at* the floor has not been measured at 2e-6.' A user can apply that rule to `p_mean_shift` and cannot apply it to `p_subset` — the stage the whole….
- **wade_from_matrix's default configuration fails on any matrix containing a zero** *(small)* — The documented entry point for an already-normalized matrix hard-crashes on first use for sparse data. The error explains what happened but not what to do; the user must find `pseudocount` (which is undocumented in this function — see the docstring gap) or….
- **The characterization statistics are unreachable without running the permutation test, and their bootstrap intervals are emitted without them** *(medium)* — A user who wants only the descriptors — 'what fraction of my cases is this gene altered in?' — must pay for a full two-stage permutation run, or get nothing. And anyone who sets n_boot without the subset stage gets a written results table containing….
- **wade_from_matrix cannot chunk genes** *(medium)* — A user whose data is already normalized (a public TPM/CPM matrix, a Seurat/scanpy export, someone else's pipeline output) cannot run WADE on a large cohort at all — the only documented remedy for the memory wall is unavailable on their entry point, and….

### Data boundary

- **Missing counts are rejected without saying where, and with no handling policy** *(small)* — A single blank cell in a large counts file stops the run with a message that does not locate it and offers no remedy, leaving the user to find the row themselves with no guidance from WADE on what it should be replaced by.
- **The manifest cannot reproduce the run it documents** *(small)* — A user cannot tell from the manifest whether a result came from length normalization, CPM-like scaling or a scalar normalizer, what jitter scale produced the tie-breaking, or which samples were cases — so the JSON that exists to say "what produced this….
- **Whether a p-value was GPD-extrapolated is never exported** *(small)* — A reader of results.tsv cannot tell which small p-values were counted from permutations and which were extrapolated by the tail fit — precisely the distinction the README's inference paragraph rests on, and the one that matters most for the genes at the….
- **featureCounts and MatrixMarket helpers are absent** *(small)* — The two count formats most users actually have on disk take extra per-file knowledge to load correctly, and getting `sample_columns` wrong is the one mistake that would silently analyse an annotation column as a sample.
- **pandas support is claimed but unexercised, and the pandas-idiomatic layout silently loses labels** *(small)* — A pandas user following the README gets a run whose result table is labelled gene0, gene1, ... instead of their gene ids, with no error to explain it — and no test would catch a regression in the pandas path.
- **Per-gene annotation cannot be carried from input to the written results** *(medium)* — Every user must re-join their gene annotation onto the result themselves, and the annotated table they actually share has no provenance file because the only writer that emits a manifest cannot include the annotation columns.

### Plotting layer

- **Data layer is second-class: not re-exported, and no table() for the gene panel** *(small)* — A user writing their own renderer (the documented path, README.md:226) must reach into a submodule that the top-level API otherwise hides, and gets no exported table for `plot_gene` — the numbers behind the flagship figure can only be pulled out by hand….
- **GeneDetail.cum (stage 1's statistic) is never drawn** *(small)* — There is no figure anywhere for stage 1: a user can see what `p_subset` prices (the curve against the dashed fitted shift) but never the area that produced `mean_shift` and `p_mean_shift`. Two docstrings and a notebook caption promise a panel that does not….
- **Label de-collision overlaps in the regime the data actually hits** *(small)* — `label=n` on a real result — where genes pile up at the p-value floor and therefore share a y — produces a smear of overlapping gene names, which is the one thing direct labelling exists to avoid.
- **No dark theme and no public palette knob** *(small)* — Every figure renders as a white slab in a dark notebook, dark docs site or dark slide deck, and there is no supported way to change it short of monkeypatching private module globals (which still leaves plotly's hard-coded `plotly_white` template).
- **No p-value or permutation-null calibration figure** *(small)* — The calibration check the notebook itself says must precede any interpretation is a print statement. A user cannot see the p-value distribution of their own run, or where a gene's statistic sits in its null — the first diagnostic figure of every….
- **README misdescribes plot_gene's dashed reference line** *(small)* — A reader following the README mis-attributes the curve's offset from the dashed line: they believe it is a deviation from the curve's own median when it is a deviation from the independently fitted global shift. It is the sentence that teaches people how….
- **plotly's two-stage volcano does not actually share the x axis** *(small)* — In the default backend, zooming or panning one stage does not move the other, so the documented comparison — the same gene low on the mean-shift panel and high on the subset panel — drifts out of alignment the moment the user interacts with it, and the….
- **Bootstrap intervals are computed but never drawn** *(medium)* — `n_boot=300` pays for a full bootstrap and buys three intervals the user can only read as a text line on one figure. On the volcano and two-stage plots every point is a bare point estimate, so a subset resting on 4 affected samples looks exactly as firm as….
- **No linked views in the plotly backend** *(medium)* — Exploring a 31,000-gene result means reading a hover tooltip, copying the gene name by hand, and calling `plot_gene` in a new cell for every gene — the loop the interactive backend exists to remove.
- **No per-sample or driver figure for a gene** *(medium)* — A user who finds a gene at `affected_fraction` 0.017 cannot see which samples those are, or that they are three anomalous libraries rather than a coherent patient group — the exact check that reversed the notebook's Ewing-sarcoma reading is text-only.….

### Test coverage

- **The `cum[-1] == mean_shift` invariant is documented as tested and is asserted nowhere** *(small)* — The one identity linking the plotted diagnostic curve to the reported statistic is unguarded, and the docstring tells the next developer it is guarded. A reversal-order or grid-orientation regression in `wade_gene` would make every `plot_gene` panel….
- **The `slow` and `validation` markers do not select what pyproject says they do** *(small)* — A developer or CI job that trusts the markers to trade coverage for speed gets the opposite of what the labels promise: `-m "not slow"` still pays for the slowest test, and `-m "not validation"` silently drops the count-native layer's unit assertions….
- **`allow_extra` is the documented remedy for a metadata mismatch and is unreachable from `wade()`** *(small)* — A user with a sample sheet that legitimately spans more runs than the current matrix — the ordinary case for a cohort sheet — is told to pass a flag that the entry point does not accept, and has to discover `Condition.vector` from the source to proceed.
- **`alternative="less"` is documented and tested nowhere** *(small)* — One third of the documented `alternative` surface has no test. A sign inversion between `greater` and `less` in `_orient` is exactly the kind of error that produces plausible-looking output — a full page of p=1.0 for the genes the user cares about, or the….

### Packaging and docs

- **The documented verification step (`pytest`) fails on a machine without the compiled kernel** *(small)* — A kernel-less user's first interaction with the package is 73 red tests, with nothing in the README to tell them that is expected or how to deselect them.
- **mamba_env.yaml has drifted from the environment the work actually requires** *(small)* — A user who follows README's Install block gets an environment in which the README's own data-in example (pl.read_excel) raises, and in which the notebook deliverable cannot be rendered at all.
- **No API documentation build — only hand-written markdown** *(medium)* — There is no per-function reference: a user who wants the signature and semantics of as_counts, condition, wade_contrast, thin_counts, characterization_ci, etc. has to read the source, and nothing checks that documented examples still run.

## 4. Inference refinements

- **Stage 1's scale is settled** (`method.md` §10.1): linear, measured best or
  tied for every alternative tried. Do not add a transform knob to stage 1.
- **The (μ = 2, 1000 v 1000) cell of §10.3** was measured at 0.10 on 60 genes.
  The harness was `prototypes/scale_and_counts.py`, deleted at `fddf2c1` —
  recover it from git and raise `reps`.
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

## 5. Format helpers, and other small things

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

## 6. Packaging and release

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
