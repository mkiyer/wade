# Roadmap

The work queue for developing WADE as a Python package with a Rust-backed core.
Ordered, with the reason for the ordering stated where it is load-bearing.

Nothing in this file is a schedule. It is a dependency order.

## The ordering constraint that governs everything below

**Establish parity on readable code before optimizing.** If the NumPy
implementation and the Rust kernel are written together, a numerical disagreement
with the R has two candidate causes — a port error or an optimization error — and
no way to distinguish them. Written in order, the NumPy implementation is
validated against the R, and the kernel is then validated against a NumPy
implementation already known to be correct.

The corollary is that the fixtures come first. They are the only artifact that
makes any later claim of correctness checkable.

## 1. Golden fixtures from the R

The first task, and the one everything else depends on.

- Build the **deterministic seam**: a wrapper that sources
  `reference/R/wade.R` unmodified and accepts a supplied jitter matrix and
  permutation index matrix in place of R's RNG draws. Exact cross-language parity
  is impossible without this — see [`docs/porting-hazards.md`](docs/porting-hazards.md)
  hazard 2. Keep it a wrapper: `reference/R/wade.R` must stay byte-identical.
- Emit fixtures at full precision (17 significant digits) covering the cases in
  the parity-suite design at the end of `porting-hazards.md`: non-square in genes
  versus samples *and* genes versus permutations, ties and zeros in the counts,
  both even and odd group sizes, both normalizer forms, a group small enough that
  the tail window collapses to one point, and a hand-checkable tiny case.
- Emit **intermediates**, not just endpoints: the `Q1`, `Q0` and `D` grids, and
  the full `g × nperms` null matrices. This is what lets a failure be localized
  rather than merely detected.
- Cover `.gpd_tail_p` separately with hand-constructed nulls hitting every
  branch: positive shape, the `xi <= 0` exponential branch, the floor binding,
  fewer than ten exceedances, an observed value at or below the threshold, and a
  degenerate zero-variance tail. It is pure scalar math, so parity here is exact
  and it is the cheapest place to catch an error.

## 2. NumPy reference implementation

Written for clarity, not speed. This is the correctness baseline.

- Implement against [`docs/algorithm.md`](docs/algorithm.md), using
  [`docs/r-implementation.md`](docs/r-implementation.md) for the conventions.
- Accept the supplied permutation and jitter matrices **on the same argument path
  production uses**. A fixture path that bypasses production code validates code
  users never run.
- The normalizers — TPM-like, CPM, RLE — as separate functions from the core
  statistic, per [`docs/design-decisions.md`](docs/design-decisions.md) S3.

## 3. Parity suite

Build it inside-out, layer 0 through layer 6, per the design at the end of
`porting-hazards.md`. Stop at the first failing layer; a suite that reports six
failures from one root cause has told you less than one that reports the innermost.

Report the **worst-case relative deviation** observed, not merely that assertions
passed. That number is what distinguishes an exact port from a close one.

Two places the port must **disagree** with the R, so the suite must not enforce
agreement there:

- `tail.conc`, where the R's guard is defective (hazard 5). Assert the numerator
  and denominator separately, and test the port's corrected guard on its own terms.
- A one-sample group, where the R's reshape recycles a scalar across every gene
  (hazard 11).

## 4. Method validation, ported

Translate `reference/R/validation_sims_v7.R` into the test suite: null calibration
by KS test, subset-versus-bulk discrimination, and power against the combinatorial
floor. These validate the *method* rather than the port, which is why they belong
in the repository at all rather than only in a notebook.

Targets measured in the R sandbox this session are in
[`reference/R/README.md`](reference/R/README.md); the power curve's step between
20% and 35% is a property of the design, explained in
[`docs/limits.md`](docs/limits.md) §4, and a port that reproduces the method
should reproduce the step.

## 5. Rust kernel

Only after 1–4 pass.

- The target is unambiguous: the serial permutation loop is the entire cost of the
  method. Everything else in `wade.R` is a single pass.
- Validate against the layer-5 fixtures — the full `g × nperms` null matrices —
  which is the only place a kernel bug is cleanly separable from a p-value bug.
- The boundary itself is open. See `design-decisions.md` O4 for the considerations:
  memory profile at scale (about 71 MB of null matrices at the cfRNA scale, 3.2 GB
  at 20,000 genes and 10,000 permutations), whether the kernel returns nulls or
  p-values, whether type-7 quantiles are implemented in Rust or delegated to
  NumPy, and parallelism across independent permutations.
- **The Rust toolchain is unverified in this environment.** Nothing was installed
  this session. Confirm `cargo` and `rustc` before planning against them.

## 6. Open design questions

Each is stated with its considerations in
[`docs/design-decisions.md`](docs/design-decisions.md); none has an answer this
session was entitled to pick.

- **O1 — the `tail.conc` guard.** The one place the port should deliberately
  deviate from the R. Guard on the ratio's magnitude rather than on `diff.mean`;
  a clamp to `[0, 1]` is wrong because values slightly above 1 are legitimate.
  A guard on `sum|D| / |sum D| <= F` bounds the output at `|tail.conc| <= F`,
  which makes the threshold and the guarantee the same number.
- **O2 — method-of-moments to maximum likelihood for the GPD fit.** The
  resolution floor must survive the upgrade; a port that returns smaller p-values
  by "improving" the floor is a regression, not an enhancement.
- **O3 — whether the rank scores belong in the package at all**, and under what
  framing. They are method-shaped but are not part of the test, and offering
  `score` beside `padj.diff` without comment invites a reader to treat a
  heuristic rank as inference.
- **O5 — whether `weight` and `log2_scale` are worth carrying.** Neither is used
  at any call site in the staged reference material, and their presence is the
  sole reason `wade()` has two permutation code paths. Note that `weight`
  interacts with label permutation in a way the reference material never
  resolves: the weighting is applied by condition, so under permutation the
  weighted group changes membership each iteration.

## 7. Packaging and release

Wheel building across platforms, benchmarks against the R at realistic gene counts
and permutation counts, API documentation, and a decision on whether the
validation simulations ship as tests, as documentation, or both.

## Documentation debt to clear on the way

Two errors in the original write-up are corrected in `docs/` and should not be
reintroduced from the reference documents:

- `WADE_REPO_SCOPE.md` says `wade.R` has nine functions; it has ten. The omitted
  one, `wade_run()`, is the entry point the notebook actually called.
- The `diff.mean = mean(case) - mean(ctrl)` identity, stated unqualified in
  `wade.R`'s header and in three source documents, holds exactly **only when the
  groups are equal-sized**. Verified: exact at 5-versus-5 and 20-versus-20, off by
  up to 22.4 at 7-versus-3. Inference is unaffected because the permutation null
  inherits the same quadrature bias — so this is a documentation fix, not an
  arithmetic one, but a port that reimplements `diff.mean` as a difference of
  means will not reproduce the R.
