# Orientation for a session starting in this repository

This repository was staged in a single session whose only job was to prepare for
yours. Nothing here is Python; nothing here is optimized. The point was to write
down what WADE is, how the original R does it, and what will break in a port,
before any of it is reimplemented.

## What is here, and what is not

**Here:** the R implementation, runnable; the documents specifying the algorithm
and the port; a record of where everything came from.

**Not here:** any Python or Rust source, any build configuration, any golden
fixtures, and any patient-derived data. The first three are your work. The fourth
is deliberate and permanent — this repository is intended for public release, and
every number in it is either synthetic, an aggregate statistic, or a design
constant.

The Rust toolchain was **not verified**. No install was attempted. Check `cargo`
and `rustc` before you plan around them.

## Read in this order

[`../README.md`](../README.md) gives the one-paragraph version. Then:

1. [`algorithm.md`](algorithm.md) — the mathematical contract. If you implement
   from one document, it is this one.
2. [`rationale.md`](rationale.md) — why the statistic is shaped this way. Section
   7 explains what the permutation p-value is actually for, which is the least
   obvious and most important thing about the method.
3. [`limits.md`](limits.md) — the two hard constraints, and the checklist of
   when to reach for something else.
4. [`r-implementation.md`](r-implementation.md) — the ten functions, line by line.
   Read section 2 carefully; the normalization denominator is the subtlest
   arithmetic in the file.
5. [`porting-hazards.md`](porting-hazards.md) — eleven hazards, each with a test,
   and a layered parity-suite design. The suite design at the end is written to be
   read as a specification for your first task.
6. [`design-decisions.md`](design-decisions.md) — five settled decisions, five
   open questions.
7. [`../ROADMAP.md`](../ROADMAP.md) — the ordered work queue.

## The ground truth, and how to run it

`reference/R/wade.R` is a byte-identical copy of the original, and the parity
target for everything you write. It must stay byte-identical: verify with
`shasum -a 256 -c sha256sums.txt` from `reference/R/`. If you need to change
behaviour to test something, wrap it — do not edit it.

The sandbox runs offline from a pre-existing read-only package cache. Three
details are load-bearing and all three are easy to get wrong; they are documented
in [`../reference/R/README.md`](../reference/R/README.md), including a known
`renv::restore()` wrinkle where the restore succeeds and *then* errors on a socket
it cannot bind. Run it twice.

`reference/R/validation_sims_v7.R` runs the three method-validation simulations in
about 95 seconds and is the fastest way to confirm the sandbox works.

## Load-bearing decisions already made

Recorded so you do not relitigate them. Full reasoning in
[`design-decisions.md`](design-decisions.md).

- **The entry point takes raw counts.** Not a normalized matrix. The continuity
  jitter that breaks ties in sparse data is applied at *count precision* before
  division, so a pre-normalized matrix cannot reproduce it. This reverses the
  recommendation in `reference/docs/WADE_REPO_SCOPE.md`, which proposed
  matrix-in with an optional convenience layer; the jitter is what decides it.
- **Normalization ships inside the package**, as functions separate from the core
  statistic. The user's reasoning: a differential-expression method must either
  provide or recommend a normalization, and providing it keeps the package
  freestanding with fewer dependencies.
- **Exact parity requires injected randomness.** R's Mersenne-Twister and NumPy's
  PCG64 cannot be made to agree on a shared seed, so the permutation and jitter
  matrices must be suppliable as inputs — and on the same argument path production
  uses, or the parity suite validates code nobody runs.
- **No fixtures were generated here** (a deliberate scoping decision). The R
  sandbox is the ground truth and it runs; generating fixtures from it is task 1.
- **`reference/` is frozen.** It is evidence. Write new material elsewhere.

## Which claims in these documents are measured, and which are asserted

This distinction is maintained throughout the documents, and it matters when you
decide what to trust.

**Measured in the R sandbox during the staging session** — reproducible by
re-running it: the null-calibration KS p-values (0.6554 bulk, 0.2327 subset) and
realized type-I error (0.0512, 0.0488); the subset-versus-bulk discrimination
ratios; the power curve and its step between 20% and 35%; the normalization
denominator algebra, reproduced to 7.3e-12; type-7 quantile agreement between R
and NumPy; the `nprobs == 1` reshape defect; the `gene_names = NULL` column-drop
defect; and the `diff.mean` identity failing on unequal group sizes.

**Measured on the cfRNA cohort and quoted from the source documents** — not
reproducible here, since the data are absent: the `tail.conc` defect rate (120 of
2,219 genes, largest 149); the normalizer comparison correlations; that exactly
one gene across 18 contrast runs reached FDR < 0.10.

**Design assertions** — reasoned positions, not measurements: the parity-suite
layer ordering, the recommended `tail.conc` guard form (though its bound is
proved), the API shape, and every "open" item in `design-decisions.md`.

**Unquotable** — flagged where it occurs: the specifics of the GPD-floor artefact
callout (gene name, exact p-values, patient count) are unevaluated inline
expressions in the source documents and the cohort data are not here.
[`limits.md`](limits.md) §5 argues that case from the constants that *are*
verifiable instead.

## Two errors in the original write-up, already corrected

Do not reintroduce these from the reference documents, which still contain them:

- `WADE_REPO_SCOPE.md` says nine functions; there are **ten**. The missing one is
  `wade_run()`, the entry point the notebook actually called — so a port built
  from that document's tables omits the historical contract.
- The `diff.mean = mean(case) - mean(ctrl)` identity is stated unqualified in
  `wade.R`'s header comment and in three source documents. It holds **only when
  the groups are equal-sized**; on unequal groups `diff.mean` is a grid quadrature
  biased toward the larger group's skew. Inference is unaffected — the permutation
  null inherits the same bias — but a port that reimplements `diff.mean` as a
  difference of sample means will not reproduce the R.

A third correction concerns interpretation rather than fact: the source documents
put the subset-detection floor at "below roughly 5% of cases". The formula is
exact, but that gloss understates it badly — at 5% of cases in the cfRNA design
the floor is 0.43, and the unsupportable region extends to roughly 18% of cases
against a bare alpha. [`limits.md`](limits.md) §4 carries the corrected treatment
and the reason it is driven by group imbalance.

## What deliberately stayed behind

The cfRNA repository keeps the layer that *consumed* WADE: the contrast registry,
the confounding screen that asks whether a contrast is confounded with the capture
panel it was run on, the cohort plumbing and cache-key discipline, and the control
taxonomy. Copies live in `reference/R/downstream/` as evidence of what a real
caller needs from the API and what the statistic looks like plotted — they are
**not porting targets**, and they do not run standalone here. See that directory's
README for a per-file account of what is cfRNA-specific and what generalizes.

One item flagged for the user and not resolved: those files encode the cohort's
clinical vocabulary — stratum names, disease categories, contrast definitions.
The staging session's assessment was that this is publishable study design, but it
does disclose the shape of an unpublished cohort. If that turns out to matter, the
remedy is narrow (abstract the stratum names) and the screen's logic survives it.
