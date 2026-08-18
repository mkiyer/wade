# Roadmap

The work queue. Ordered, with the reason for the ordering stated where it is
load-bearing. Nothing here is a schedule; it is a dependency order.

## Done — the port

The Python package with a Rust-backed core, ported from `reference/R/wade.R`.

| | |
|---|---|
| Golden fixtures from the R, through a deterministic seam | 20 fixtures, committed, R-free to consume |
| NumPy reference implementation | complete |
| Parity suite, layers 0–9 | 690 tests, worst relative deviation **9.155e-15** over 610 comparisons |
| Method validation simulations, ported | all three; power table identical to the R |
| Rust kernel (PyO3 + rayon) | 18–22× the NumPy path, null matrices bitwise identical to the R |

Four layers — the normalized matrix, the quantile grids, the permutation null
matrices and the rank scores — are **bit-for-bit** identical to the reference.
The details are in `docs/port-status.md`.

That established the *machinery* is correct. It did not establish that the
statistic on top of it is the right one, which is what the rest of this file is
about.

---

## Now — replacing the tail window

**The problem.** The subset axis is built on `k = max(1, ceil(tail_q · m))` with
`tail_q = 0.10`: three stacked heuristics — a fraction, a rounding rule and a
floor — plus a fourth (`F`) to guard the ratio they produce. It forces the user
to declare what they are looking for, which is exactly the failure that made
COPA require re-running at every percentile cutoff. And `k` is an integer window
on a grid whose size is set by the design, so the realized tail fraction
sawtooths between 0.100 and 0.182 as group size varies and jumps discontinuously
— `m = 20 → 21` moves the window from 10.0% to 14.3% of the grid on one extra
sample. `tail_mean` is therefore not comparable across contrasts.

**The shape of the replacement.** Two steps, neither taking a threshold:

1. **Detect** a difference between the groups, with significance, sensitive both
   to a global shift and to a difference confined to a small subset of one group.
2. **Characterize** it — report *where on the spectrum* the gene sits, from a
   global mean shift to a cliff affecting a few percent of cases.

### Step 2 is solved: the effective affected fraction

On the log-ratio curve `R(p) = log2 Q_case(p) − log2 Q_ctrl(p)`, the
participation ratio

    pi_hat = (sum_p R(p)^2)^2 / (m * sum_p R(p)^4)

equals the affected fraction exactly for a step, and 1 exactly for a curve flat
at `log2(FC)`. No percentile, no window, no `k`, no `F`.

Measured, 500 v 500, planted fractions against the estimate:

| true | 1% | 2% | 5% | 10% | 25% | 50% | 80% | fc=2 | fc=8 |
|---|---|---|---|---|---|---|---|---|---|
| `pi_hat` | 0.018 | 0.025 | 0.058 | 0.109 | 0.278 | 0.556 | 0.850 | 0.979 | 0.998 |

Two details that are not free parameters but *derivations*, and must survive any
reimplementation. The **log scale** is what anchors a global fold change at 1.0
independently of its magnitude; on the raw scale a 2× shift reads 0.70 and the
reference point drifts with each gene's own dispersion. The **fourth moment** is
what removes the noise floor: a floor of height ε against signal h contributes
ε/h to the second-moment form and (ε/h)² to this one, which is why the
second-moment version reads a 2% subset as 0.23.

**Resolution limit, stated rather than hidden.** The grid has `m = min(n0, n1)`
points, so no fraction finer than `1/m` is resolvable. Measured: quantitative
above m ≈ 100, degrading through m ≈ 50, and below that only qualitative —
global (0.89–0.94) still separates from concentrated (0.12–0.27), but 2% and 5%
become indistinguishable. This is the same grid resolution `docs/limits.md`
already documents, surfacing honestly instead of being absorbed by a
`max(1, ceil(·))`.

### Step 1 is open: the detector — PROTOTYPE NEXT

Removing the window removes `tail_mean`, which was the subset-sensitive test.
Detecting a signal confined to a few percent of the curve is a sparse-signal
detection problem, where L2-type omnibus statistics lose power badly because the
signal is diluted across the region that did not move.

**Prototype three detectors and measure power**, at planted affected fractions
of 1, 2, 5, 10, 25, 50 and 100%, across the geometries that matter (77 v 18,
100 v 100, 500 v 500):

1. **Scan statistic** — max over `k` of the standardized partial sum from the
   top of the grid, calibrated by permutation. Threshold-free by *maximization*
   rather than by choosing, with the permutation null absorbing the multiplicity;
   this is the direct answer to COPA's "run it at every cutoff". Its `argmax`
   yields the affected fraction as a by-product, so detection and
   characterization would come from one computation and could not disagree.
2. **Higher Criticism** (Donoho & Jin) — the reference detector for sparse
   alternatives, also parameter-free.
3. **A plain mean test** — the floor. WADE's claim is *additional* sensitivity
   to subset signals, not better detection of global shifts, and that claim is
   only meaningful measured against an ordinary test.

Report power curves, not verdicts. Then agree a plan before implementing.

### Sequencing after the prototype

1. Agree the detector from the power curves.
2. **Document the agreed plan** — the new method, in one place.
3. **Consolidate the documentation** and retire what the plan supersedes.
   Current: ten markdown files, most of them porting-process artefacts. Target
   four — `README.md`, `docs/method.md` (algorithm and rationale, rewritten
   around the new statistic), `docs/limits.md`, `docs/implementation-notes.md`
   (the cross-language hazards that still bite — quantile convention, broadcast
   axis, summation order, R's non-round-tripping parser — plus the parity
   result). `r-implementation.md`, `porting-hazards.md`, `design-decisions.md`
   and `port-status.md` fold into those and are deleted.

---

## Then — making it usable (goal 5)

Blocked on the redesign only where noted; the tail definition decides what the
plots and the demo have to show.

- **I/O with Polars.** Readers for count matrices across the formats Polars
  already handles, and result writers. `WadeResult.to_polars()`. Sample-name and
  gene-name aware throughout — the current API takes integer column indices,
  which no user has.
- **Plotting.**
  - Single-gene panel: the quantile pair with the difference curve, descended
    from HITLIB's `areadiff_plot`. This is where the cliff-versus-straight-line
    reading lives, so it is the figure the method is explained with.
  - Volcano: effect size against significance, coloured by the affected
    fraction. *Waits on the redesign* — the colour axis is the new statistic.
  - The two-axis scatter, bulk against subset, which is what the discrimination
    simulation actually shows.
- **Demo notebook.** **Synthetic data, no download** (decided). It doubles as a
  method comparison: t-test and Wilcoxon as floors, and COPA / OS / ORT / MOST
  as the honest competitors, since they share the subset-detection goal and are
  a few lines each.

## Later

- **Memory at scale.** The null matrices are always materialized: peak RSS
  tracks `2 · g · B · 8` bytes plus about 40% (measured 2.22 GB at 20,000 genes
  × 5,000 permutations). `keep_null` controls retention, not construction. A
  chunked or streaming path needs two passes, because the GPD refinement needs
  each refined gene's full null vector.
- Thread cap for the kernel, and progress reporting for long runs.
- **GPD moment fit → maximum likelihood.** Moments are poorly behaved for
  `xi > 0.5`, which is the heavy-tailed regime the refinement exists for. The
  resolution floor `1/(B · n_tail)` must survive any upgrade: it is a statement
  about what B permutations can support, and a better tail fit does not buy more
  resolution.
- Permutation and p-value construction more broadly.
- Wheels across platforms, CI, and an API documentation build.
