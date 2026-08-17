# Design decisions: what is settled and what is open

This document exists to stop the next session relitigating ground that has already
been decided. It records five decisions made in this repository's planning sessions
— the reasoning, not only the outcome — and five questions deliberately left open,
each with the considerations that bear on it and no invented answer.

Two of the settled decisions **reverse recommendations** in
[`reference/docs/WADE_REPO_SCOPE.md`](../reference/docs/WADE_REPO_SCOPE.md), which is
staged reference material and was written before them. Where that happens this
document says what the source recommends, what was decided instead, and why. A
reader who finds the two in conflict should follow this document; the scope analysis
is preserved as-is because it is the record of how the extraction was reasoned
about, not because every conclusion in it survived.

The mechanics referenced below — how the R normalization works, where the cost is,
what breaks across languages — are in [`r-implementation.md`](r-implementation.md)
and [`porting-hazards.md`](porting-hazards.md).

---

## Settled

### S1. WADE is a Python package with a Rust-backed core

**Decided by the user, as an institutional direction.** The lab is moving from R to
Python. WADE will be developed as a Python package with a Rust extension module for
the numerical core (PyO3/maturin). The R in `reference/` is reference material: it
will not be shipped, packaged, or maintained.

**What this forecloses.** The dual-language repository shape proposed in
`WADE_REPO_SCOPE.md` — which sketches parallel `python/wade/` and `R/` trees with a
`DESCRIPTION` and `NAMESPACE`, and an R-side `wade::wade()` entry point — is not
being built. There is no R package to maintain, no CRAN submission, and no
obligation to keep an R API in step with the Python one. The R's role is narrower
and more useful: it is the **oracle**. It runs, it is pinned, and when the port and
the reference disagree, the reference can be re-run to determine which is wrong.
That is why [`reference/R/README.md`](../reference/R/README.md) documents the
sandbox as a cross-checking tool rather than a package, and why `wade.R` must stay
byte-identical to the cfRNA original.

**What it does not foreclose.** The cfRNA project continues to `source()` its own
copy of `wade.R`; nothing in this repository changes that, and no migration of the
existing cfRNA analyses is implied by this decision.

**A consequence for the development session.** The reference implementation is only
an oracle for as long as it runs. The sandbox has real constraints — R 4.6.1 is not
on the default `PATH`, `renv` must be activated from `reference/R/` specifically, and
packages can only be linked from a pre-existing read-only cache because there are no
R 4.6 arm64 binaries on CRAN and no install path is available. All of that is
documented in `reference/R/README.md`, including the `renv::restore()` socket-server
error that must be run through twice. Treat that file as operational documentation,
not background.

### S2. Normalization stays in the package, and the entry point takes raw counts

**Decided by the user.** `wade()` keeps its current input contract: a raw count
matrix, a normalizer, and library sizes. Normalization is part of the package.

**This reverses `WADE_REPO_SCOPE.md`.** That document's *Proposed cut* is explicit:
"the new repo's primary entry point takes a *matrix that is already on a comparable
scale* plus the condition vector", with `wade_normalize()` and `wade_lib_size()`
demoted to "an optional convenience layer (`wade.normalize.tpm_like()`), clearly
documented as one choice among many". Its interface sketch shows
`wade.wade(X, cond, ...)` — matrix in, no counts. That is not what is being built.

**The user's reasoning.** Every differential-expression method must either provide a
normalization or recommend one; there is no coherent third option, because the
statistic's behaviour depends on the scale of its input. Providing one keeps WADE
freestanding and keeps the dependency surface small — a user does not need a second
package, or a house pipeline, to run the test.

**The technical constraint that decides it, and it is not a preference.** The
continuity jitter is applied **at count precision, before division**. In
`wade_normalize()` the jitter is added to the counts and then a matching per-cell
correction appears in the denominator; the arithmetic is worked through in
[`r-implementation.md`](r-implementation.md), and the short version is that
`denom[g,j] = lib_sizes[j] + nz[g,j]/normalizer[g,j]` — the library size in the
counterfactual where gene `g` alone received jitter. **A pre-normalized matrix cannot
reproduce that.** Once counts have been divided by a normalizer and a library size,
the information needed to construct the correct denominator is gone: you no longer
know what one count was worth in that cell, so a jitter applied afterwards is not
the same perturbation and does not produce the same numbers.

This is what makes the scope document's cut untenable rather than merely different.
Its recommendation would have kept `normalize_tpm_like()` available as a convenience
function, which sounds like it preserves the behaviour — but a user who called the
core entry point with their own normalized matrix would silently get a *different
statistic*, one whose ties were never broken and whose zeros are still exactly
equal. The jitter is not a preprocessing nicety; it is part of how the test behaves
on sparse counts, and the scope document says as much in its own assessment
("the jitter in particular is genuinely part of how the test behaves on sparse
counts"). Having said that, it then proposed a boundary that discards it.

**Note the second-order effect on library sizes.** `wade_run()` computes
`lib_sizes` from the *subset* matrix, after selecting contrast columns and the
cohort's genes. So a library's size factor depends on which genes are in the matrix,
and changing the gene set changes every normalized value. The cfRNA cache key
includes the gene count for exactly this reason
(`reference/R/downstream/wade_contrasts.R` lines 404-415). Design assertion: a port
should either compute library sizes on the matrix it is handed (matching the R) or
require them as an explicit argument, and should document which — this is a real
reproducibility surface, not a detail.

**What remains true from the scope document's critique.** Its two objections are not
dismissed by this decision, and both should reach the port's documentation. First,
WADE's internal normalization is a second normalization decision living inside a
test — measured on the cfRNA cohort, WADE's internal TPM and the house `ic_spl_sum`
normalizer correct depth comparably (residual Spearman of per-library mean abundance
against sequencing depth: ρ = 0.41 for WADE's TPM, ρ = 0.36 for `ic_spl_sum`, both
against ρ = 0.79 uncorrected on the all-panels cohort; 0.30 / 0.28 / 0.70 on the
v3-omitted cohort), so this is redundancy rather than a defect, but it is still a
choice the method is making on the user's behalf. Second, a user with an existing
pipeline has a normalized matrix and no obvious way in. Design assertion: the honest
resolution is to document the jitter constraint prominently — a user who insists on
pre-normalized input should be told explicitly that they are running a variant
without tie-breaking, and what that costs on sparse data.

### S3. Normalizers are separate functions from the core statistic

**Decided by the user, as the shape of S2.** Normalization is in the package but not
*in* `wade()`'s statistical core: the normalizers are separate, named, individually
callable functions, and three are wanted.

| normalizer | form | note |
|---|---|---|
| TPM-like | the existing `count / normalizer` pair, scaled to library size | what `wade_normalize()` implements; the (count, normalizer) pair generalizes — splice-junction counts over intron count gives "sjTPM", total counts over effective length gives standard TPM |
| CPM | counts per million | not present in the R |
| RLE | median-of-ratios | not present in the R |

The first is a port; the second and third are new work with no reference
implementation in this repository, and therefore nothing to test against in
`reference/`. Design assertion: RLE in particular has a well-known behaviour on
zero-heavy data (the geometric-mean reference is undefined for genes with any zero,
so implementations restrict to fully-detected genes), and cfRNA data is sparse — the
interaction between that restriction and the continuity jitter is unexamined here
and should be decided deliberately rather than inherited from another package's
defaults.

**What this settles about the API's shape.** Two layers, not one: normalizers
produce a matrix, the statistic consumes one, and the raw-count entry point composes
them. That keeps S2's contract (`wade()` takes counts) while making the
normalization choice explicit and swappable, and it means a new normalizer can be
added without touching the statistic.

### S4. No golden fixtures were generated

**Decided by the user.** No committed test fixtures were produced in the planning
sessions. The position is that the R sandbox is the ground truth and it is runnable,
so fixtures can be generated when there is a port to test.

**What this means for the development session.** Generating them is the **first
task**, before any statistical code is written, because the fixture generator
determines what is testable. In particular it is where the RNG barrier gets resolved
in practice: the fixtures must carry the jitter matrix and the permutation matrix,
since those cannot be reproduced across languages from a seed. The full inventory —
what to emit, at what shapes, and the layer-by-layer assertion order — is in
[`porting-hazards.md`](porting-hazards.md); the two constraints worth repeating here
because they are cheap to get wrong and expensive to discover late are that fixtures
must be **non-square** in genes versus samples and in genes versus permutations (a
square fixture makes a wrong-axis broadcast invisible), and that they must include
**ties and zeros**, which is where quantile conventions and the GPD's strict
exceedance inequality diverge.

**What it forecloses.** Nothing, but note the cost of the ordering: until fixtures
exist, "the port agrees with R" cannot be asserted by anyone other than someone with
the R sandbox running. The sandbox's operational constraints (S1) make that a real
dependency rather than a formality.

### S5. `reference/` is frozen

Everything under `reference/` is staged, checksummed reference material and is not to
be modified. `wade.R` is byte-identical to the cfRNA original at git SHA
`828f2f1c6a10afd182fee0e119ba3595d98b6a72`; identity is verifiable with
`shasum -a 256 -c sha256sums.txt` from `reference/R/`, and provenance for every file
is recorded in [`reference/PROVENANCE.md`](../reference/PROVENANCE.md).

**Why this is a decision and not just tidiness.** The reference is only an oracle if
it is unmodified. A "small fix" to `wade.R` — the `tail.conc` guard is the obvious
temptation, and the `nprobs == 1` reshape is the other — destroys the only
independent check the port has. Fixes belong in the port, with the divergence from
the reference documented as intentional. Where a test needs a patched R (one option
for testing the `tail.conc` fix), the patch belongs in the test harness, not in
`reference/`.

---

## Open

Five questions the development session should decide. Each is stated with what is
known, what the considerations are, and what a decision would foreclose. None has a
recommended answer here.

### O1. The `tail.conc` guard

**The defect is measured and the form of the fix is settled; the parameter is not.**
`wade()` guards the ratio at `abs(diff.mean * nprobs) < 1e-8`, which reconstructs
the denominator and tests it against a fixed absolute threshold — at the cfRNA
primary contrast's `nprobs` = 22, that is `|diff.mean| < 5 × 10⁻¹⁰`, roughly nine
orders of magnitude tighter than the cases that occur. Measured on the cfRNA primary
contrast: 120 of 2,219 genes (5.4%) return `|tail.conc| > 2`, the largest 149.

The recommended form, developed in [`porting-hazards.md`](porting-hazards.md), is to
guard on the ratio's conditioning rather than the denominator's absolute size:
report `tail.conc` only when `sum(|D|) / |sum(D)| <= F`. That form has the property
that the guard bounds the output — `|tail.conc| <= F` — so the threshold is a
documentable promise rather than a tuning knob. A clamp to [0, 1] is ruled out: values
slightly above 1 are legitimate, and on a synthetic reproduction 618 of 4,000 genes
fell in `1 < |tail.conc| <= 1.5`.

**What is open:** the value of `F`, and what to return when the guard fires. The
factor sweep in `porting-hazards.md` (measured on one synthetic construction, not on
real data) shows the trade — at `F` = 3, all 354 pathological genes are caught at the
cost of withholding 4.0% of well-conditioned ones; at `F` = 10, 49% are caught at a
cost of 0.3%. Whether to return a missing value, or the value plus a
`tail_conc_ok` flag, is also open; a flag preserves information and forces the
consumer to look, which is arguably the point.

**What a decision forecloses:** the value of `F` becomes a documented output
guarantee, so raising it later widens what users have already been told to expect.
Pick it as a promise, not as a default.

### O2. Method-of-moments → maximum likelihood for the GPD

`.gpd_tail_p()` fits the exceedance distribution by method of moments —
`xi = ½(1 − m²/v)`, `sigma = ½m(1 + m²/v)` — and the file's own comment calls
MLE-based GPD "the production upgrade" (line 172). `WADE_REPO_SCOPE.md` agrees and
places the upgrade in the new repository rather than in cfRNA.

**Considerations.** The moment estimators are closed-form and dependency-free, which
is why they were chosen; MLE needs an optimizer and can fail to converge, which
introduces a failure mode the current code does not have and which must return
*something* on every gene. Moment estimators for the GPD are also known to be
poorly behaved for `xi` above about 0.5 (the variance of the exceedance distribution
does not exist for `xi >= 0.5`), which is precisely the heavy-tailed regime the
refinement is for — so the upgrade is not cosmetic.

**Two constraints on any upgrade.** First, **the floor must survive it.** The floor
`1/(B · n_tail)` is a statement about what `B` permutations can resolve, not a
numerical guard, and a better tail fit does not buy more resolution. A port that
reaches smaller p-values via MLE has not improved the method; it has removed an
honesty constraint. Second, the branch structure must survive: the `xi <= 0`
exponential fallback exists because a negative shape gives the GPD a hard upper
bound at `−sigma/xi`, beyond which a strong observed statistic collapses to machine
epsilon. Measured, that branch is taken routinely rather than exceptionally — an
exponential null (true `xi` = 0) produced a moment estimate of −0.133, and a uniform
null produced −0.896.

**What is open:** whether to do it at all in the first version, whether to keep
moments as a fallback when the optimizer fails, and whether the two estimators should
be selectable so the port can reproduce the R exactly *and* offer the better fit.
Note that if the estimator is selectable, the parity fixtures must pin the moment
path.

### O3. Whether the rank scores belong in the package

`wade_score()` computes `score` and `tail.score` — signed rank products of empirical
CDFs — and `rank`/`tail.rank` from them. The tension is real and both sides are
documented.

**For inclusion.** They are method-shaped: they read only columns `wade()` produced,
they need no cohort knowledge, and they port directly. And they are what nomination
actually used, because on the cfRNA cohort no gene survived BH on either axis —
`reference/R/downstream/README.md` records that with roughly 2,200 genes and 22
controls the permutation p-values cannot survive correction, so nomination was by
rank score with the p-values read as effect-strength annotation. A package that
omits them omits the thing the historical analysis ran on.

**Against.** They are **not part of the test**. `wade.R`'s own header calls `score`
"the ORIGINAL panel-selection score, preserved for continuity" — a lab heuristic,
carried forward for comparability rather than derived. `WADE_REPO_SCOPE.md` states
the risk plainly: a package that offers `score` next to `padj.diff` without comment
invites a user to treat a heuristic rank as inference. Its recommendation is to move
them in a separate `wade.rank` submodule, documented as nomination-not-inference.

**A consideration neither source raises.** The two scores are not the same
construction with one term swapped, despite the header describing them that way.
`score` takes its magnitude from the ECDF of an **absolute** value (`|log2 fc|`) and
its sign separately from `diff.frac`; `tail.score` takes both from the ECDF of a
**signed** value (`tail.mean`), so a gene strongly *down* in the case tail gets a
small negative score rather than a large one. If the scores ship, that asymmetry
should be documented or reconciled — it is currently a latent inconsistency in a
heuristic, which is the kind of thing that becomes an issue report.

**What is open:** ship or omit; if ship, whether in a submodule, under what name,
and with what framing in the docstring. Also open, and worth deciding at the same
time: whether the package should say anything about the **directionality** limit —
the scores reward up-in-case genes, so genes *lost* in cases are visible in
`diff.mean` but are not what the ranking surfaces (stated in
`reference/docs/supplement_wade.qmd`, S7.6). That is a property of the scores, not of
the test, and it disappears from view if the scores ship without comment.

### O4. Where the Rust/Python boundary falls

**The one hard fact, and it is not in dispute:** the serial permutation loop in
`wade()` is the entire cost of the method. `nperms` iterations, each doing two
`rowQuantiles` calls over a genes × samples slice plus a subtraction and three
reductions; everything else in `wade.R` is a single pass. `WADE_REPO_SCOPE.md`
identifies the same target. Any boundary that puts the loop on the Rust side captures
essentially all of the available speedup; any boundary that does not, captures
almost none.

**A design constraint carried over from the RNG barrier
([`porting-hazards.md`](porting-hazards.md), hazard 2):** the kernel should accept a
**caller-supplied permutation matrix**, so the fixture path and the production path
exercise the same code. If the fixtures go through a separate entry point, the parity
suite validates a code path users never run. The same argument applies to the jitter,
though the jitter is in the normalizer rather than the kernel.

**Considerations that bear on where exactly to cut.**

- **Memory.** The R materializes two `g × nperms` double matrices up front:
  `2 · g · nperms · 8` bytes, about 71 MB at the cfRNA scale (2,219 genes,
  `nperms` = 2000), but 3.2 GB at 20,000 genes and 10,000 permutations. A kernel
  that returns the full null matrix reproduces the R's memory profile; one that
  accumulates p-values internally does not. But note `.gpd_tail_p()` needs each
  refined gene's **full null vector**, so an accumulating kernel must either retain
  rows for candidate genes or make two passes.
- **What crosses the boundary matters more than what is behind it.** Returning the
  full null matrix is what makes layer 5 of the parity suite possible — asserting the
  `g × nperms` nulls elementwise against the R's serial loop is the only place a
  kernel bug is cleanly separable from a p-value bug. A kernel that returns only
  p-values is harder to test, whatever its speed.
- **Where the quantile computation lives.** The two `rowQuantiles` calls per
  permutation are the inner cost. Type-7 quantiles are simple to implement, and
  implementing them in Rust removes a dependency on NumPy's behaviour inside the hot
  loop — but it also means the port has two quantile implementations (one in the
  observed path if that stays in Python, one in the kernel) that must agree exactly.
  Using one implementation for both paths is worth more than it sounds.
- **Parallelism is available and unexploited.** The permutations are independent.
  The R's loop is serial and its serialism is incidental, not required.

**What is open:** everything except the fact that the loop is the target. Whether the
kernel takes the normalized matrix or the counts, whether it returns nulls or
p-values, whether quantiles are computed in Rust or NumPy, and whether it
parallelizes.

### O5. Whether `weight` and `log2_scale` are worth carrying

`wade_stats()` takes two optional transforms: `weight` (multiplies control columns
by a constant before quantiles, `> 1` meaning stringency) and `log2_scale`
(`log2(x + 1)` after weighting).

**Measured: neither is used anywhere in the staged reference material.** A search
across every `.R`, `.qmd` and `.md` file under `reference/` for `weight =` or
`log2_scale` outside `wade.R`'s own definitions returns nothing. Every call site
uses the defaults — the three validation-simulation `wade()` calls, the notebook's
three `wade_run()` calls, and the cfRNA contrast layer.

**They have a real cost, and it is structural rather than notional.** Their presence
is why `wade()` has two permutation code paths. The driver computes
`lean <- (weight == 1 && !log2_scale)` and calls `.wade_null_stats()` — the fast
path, existing solely to avoid the derived means, fold change and list construction
— only when both are at their defaults. Set either one and every permutation runs
full `wade_stats()`. So two of the ten functions in the file, and the file's only
code-path branch, exist to support options that no reference call site uses. A
kernel-backed port inherits that: either the kernel supports both transforms, or
setting them silently drops the user off the fast path, which is the R's behaviour
and is worse when the fast path is a compiled kernel and the slow path is Python.

**Considerations on the other side.** `log2_scale` is not exotic — a quantile-area
statistic on a log scale is a different and defensible statistic, and a user may
reasonably want it. `weight` is harder to justify: it is a one-sided rescaling of
one group whose effect on the null is not documented anywhere in the staged material,
and note that it interacts with the label permutation in a way that deserves thought
before it is carried — the weighting is applied by `cond`, so under permutation the
*weighted* group changes membership each iteration, which is either the intended
conditional-inference behaviour or an artefact, and nothing in the reference says
which.

**What is open:** carry both, carry `log2_scale` only, or drop both. Design
assertion: if either is carried, the port owes it a test — there is currently no
evidence in this repository that either option has ever been exercised, so "it
ports directly" is an assumption rather than an observation.

---

## Decisions this document does not cover

Three things a reader might expect to find here and will not.

**The API's exact signatures.** Settled here is the input *contract* (raw counts, S2)
and the layering (normalizers separate, S3), not the names, argument order, or
return type. `WADE_REPO_SCOPE.md` sketches a Python interface, but that sketch is
built on the matrix-in cut that S2 reverses, so it should be read as an illustration
of shape rather than a specification.

**The statistical questions.** Whether the tail statistic is the right subset
detector, how it relates to the COPA/OS/ORT/MOST/LSOSS outlier literature and to the
scDD/waddR quantile-distance literature, and what the method cannot do are covered in
[`rationale.md`](rationale.md) and [`limits.md`](limits.md). One correction belongs
with them rather than here: the "below roughly 5% of cases" detection floor claimed
in `reference/docs/supplement_wade.qmd` (line 161) and repeated in the v7 and v8
notebook sections **understates the constraint** — the floor formula
`C(n1,k)/C(n1+n0,k)` is exact, but evaluated at the v7 design (77 cases, 18 controls)
it gives 0.43 at 5% of cases, and the unsupportable region extends to roughly 18%
against a bare α = 0.05. See [`limits.md`](limits.md) for the full evaluation.

**Anything about the cfRNA consumer layer.** The contrast registry, the confounding
screen, the control taxonomy and the cache discipline stay in cfRNA.
`reference/R/downstream/README.md` is explicit that none of it is a porting target.
Two things in it are really statements about WADE and should reach the port's
documentation — that `min(n0, n1)` *is* the quantile-grid resolution and
`ceiling(tail_q · min(n0, n1))` is the tail window, so a caller needs that arithmetic
before deciding whether the subset axis means anything; and that group definition
belongs outside the statistic, enforced by a closed registry rather than a substring
match. Both are covered in [`r-implementation.md`](r-implementation.md) under what
the consumer layer implies about the API.
