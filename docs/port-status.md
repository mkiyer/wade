# Port status: what was built, and what it measures

This document records the state of the Python/Rust implementation and,
more importantly, **the numbers that justify calling it correct**. It
follows the convention the rest of `docs/` uses: *measured* means a number
produced by running code in this repository; *design assertion* means a
reasoned position.

Everything below was produced by running the R sandbox described in
[`../reference/R/README.md`](../reference/R/README.md) and the Python test
suite in the conda environment `wade`.

---

## 1. Environment, as found and as left

| | at session start | now |
|---|---|---|
| R 4.6.1 sandbox | present, offline `renv` closure restored | unchanged |
| `reference/` byte-identity | 8 of 8 files `OK` | 8 of 8 files `OK`, now asserted by the test suite |
| conda env `wade` | **empty** — no Python | Python 3.12.13, NumPy 2.5.2, SciPy 1.18.0, pytest, hatchling, maturin |
| `cargo` / `rustc` | **absent** — no `rustup`, no `~/.cargo` | cargo 1.97.1, rustc 1.97.1, maturin 1.14.1, installed **via conda into `wade`** |

The Rust toolchain was genuinely missing, as `ROADMAP.md` §5 warned it
might be. It was installed from conda-forge into the `wade` environment
rather than via `rustup`, so it lives inside the environment like every
other dependency and does not touch the base environment or the user's
home directory. `crates.io` is reachable from this environment; `pyo3`
0.27, `numpy` 0.27 and `rayon` 1.12 resolved and built.

## 2. What exists

```
src/wade/            the Python package
  quantiles.py         type-7 quantiles, the probability grid, the tail window
  normalize.py         tpm_like (ported), cpm and rle (new work)
  stats.py             the statistic and the tail_conc guard
  permutation.py       the null, dispatching to NumPy or the Rust kernel
  pvalues.py           empirical p, the GPD refinement, BH-FDR
  scores.py            the rank scores (nomination, not inference)
  diagnostics.py       the single-gene cumulative-area curve
  api.py               wade(), wade_from_matrix(), wade_contrast()
rust/src/lib.rs      the permutation kernel (PyO3 + rayon)
tools/r/
  wade_seam.R          the deterministic seam
  json_emit.R          the fixture writer
  generate_fixtures.R  the fixture generator
tests/               the parity suite, layers 0-9, plus the kernel and divergence tests
tests/fixtures/      20 golden fixtures, ~4.7 MB
```

## 3. The deterministic seam

`tools/r/wade_seam.R` sources `reference/R/wade.R` **unmodified** and makes
its two sources of randomness injectable.

The mechanism is worth recording because the obvious approach does not
work. `wade()`'s `sample(cond)` is unqualified and so resolves lexically —
shadowing it is easy. But the jitter comes from `stats::runif`, and `::`
bypasses lexical scoping. The usual escape is `assignInNamespace`, which
mutates the `stats` namespace for the whole session.

It is not needed. **Base R's `` `::` `` is an ordinary closure**, so it can
itself be shadowed in the enclosing environment chain. `wade.R` is sourced
into an environment whose parent binds both `sample` and `` `::` ``; the
`::` shim returns the jitter stub for `stats::runif` when armed and
delegates everything else — `matrixStats::rowQuantiles`, `tibble::tibble`,
`dplyr::dense_rank`, `stats::var`, `stats::p.adjust`, `stats::ecdf` — to
`getExportedValue()` unchanged. Nothing is written to any namespace and
nothing global is touched.

**Measured:** `seam_selftest()` runs a native `wade()` call and a seam call
injected with R's own realised draws and compares the two frames with
`identical()`. They agree **bitwise**, with the runif shim firing exactly
once and the sample shim exactly `nperms` times. Repeated five times to
rule out the byte compiler re-resolving `` `::` ``.

## 4. Fixture format: why every number is written twice

The roadmap asks for 17 significant digits. That is emitted — and it is not
sufficient on its own, for a reason that only shows up when you try to
verify it.

**Measured:** R's `sprintf("%.17g", x)` produces correctly rounded digits,
but R's own `as.numeric()` does **not** read them back exactly. On 42
`runif(0, 0.01)` draws, 8 round-tripped one ULP off — and they still did at
`%.18g`, `%.19g` and `%.20g`. So the defect is R's parser, not the digits.

Each double is therefore written in two channels: `"dec"`, the 17-digit
decimal, and `"hex"`, a C99 `%a` hex float, which is a lossless base-2
rendering. R verifies the hex channel round-trips (it does, exactly), and
the Python loader asserts `float(dec) == float.fromhex(hex)` for **every
value it reads**. Hex is authoritative; a disagreement fails the suite
rather than passing quietly.

Non-finite values are spelled out, and `NA` is kept distinct from `NaN` —
`wade()` produces `NA` from its `tail_conc` guard and from `nperms == 0`,
so collapsing them would erase a distinction the port is being tested on.
Every 2-D array carries `"shape": [nrow, ncol]` and is nested as row
arrays, which is hazard 8's fix: a flat vector plus dimensions is exactly
what lets two languages disagree about fill order.

## 5. The fixtures

20 files. Every scenario is non-square in genes vs samples **and** genes vs
permutations, so a wrong-axis broadcast raises instead of lying.

| fixture | geometry | what it is for |
|---|---|---|
| `tiny` | 6×7, 3v4, m=3, k=1, B=11 | hand-checkable; vector normalizer |
| `main` | 40×27, 15v12, m=12, k=2, B=37 | the workhorse; all-zero, zero-run, heavy-ties and constant genes |
| `even_larger` | 13×19, 7v12, m=7, B=23 | the interpolated group has **even** length |
| `vecnorm` | 17×15, 8v7, B=29 | the per-gene vector normalizer, zero-heavy |
| `m2` | 9×8, 6v2, m=2 | the degenerate two-point grid |
| `onesample` | 5×4, 1v3, m=1 | **records the R's defect**; not a parity target |
| `nperms0` | 8×11, B=0 | effect sizes only, all p-values missing |
| `gate_closed` | 12×27, B=300 | the refinement gate's `B >= 500` condition |
| `refine` | 25×27, B=501 | GPD refinement fires; carries the fit internals |
| `weighted`, `log2scaled` | 10×15, 8v7 | the non-lean pre-transform paths |
| `zerolib` | 6×8, one all-zero sample | the exactly-1e6 artefact |
| `nonames` | 7×9, no rownames | R drops the gene column |
| `tailconc` | 200×22, 11v11 | deliberate bulk/tail cancellation |
| `tiesheavy` | 15×13, 7v6 | counts of 0-3, whole zero rows |
| `gpd_branches` | 12 constructed nulls | every branch and bail-out of `.gpd_tail_p` |
| `bh_padjust` | 5 vectors | including the `NA` and tie behaviour |
| `ecdf_denserank` | 4 vectors | the ECDF and `dense_rank` conventions |
| `worked_example` | 2×9 | `algorithm.md` §2.7, reproduced from `wade_stats()` |
| `quantile_type7` | 2 vectors × 6 grids | odd and even lengths with ties |

Each scenario carries the whole computation, not just the endpoints: the
jitter and permutation matrices, `nprobs`/`k`/`n1`/`n0`/`q`, `lib_sizes`
and the normalized matrix, the full `Q1`/`Q0`/`D` grids, every per-gene
reduction, `tail_conc`'s **numerator and denominator separately**, the full
`g × nperms` null matrices for both axes, the exceedance counts, which
genes were refined and each refined gene's GPD internals, the result frame,
the rank scores, and the `wade_gene()` diagnostic.

**Integrity, enforced at generation time.** The intermediates are produced
by replicating `wade()`'s body step by step; that replication is then
checked against a real `wade()` call under the same injection, column by
column, with `identical()`. If they ever disagreed the generator would
abort rather than emit a fixture whose intermediates and endpoints describe
two different computations. Likewise the instrumented `.gpd_tail_p` mirror
asserts its returned p against the real function on every call.

## 6. Parity: the measured result

**Worst-case relative deviation across 610 comparisons: 9.155e-15.**

| layer | comparisons | worst relative deviation |
|---|---|---|
| 1 — grid scalars and `q` | 14 | **0** (bitwise) |
| 2 — `lib_sizes`, normalized matrix | 28 | **0** (bitwise) |
| 3 — `Q1`, `Q0`, `D` grids | 54 | **0** (bitwise) |
| 4 — per-gene reductions | 178 | 2.6e-16 |
| 5 — the `g × nperms` null matrices | 26 | **0** (bitwise) |
| 6 — p-values and the GPD | 110 | 9.2e-15 |
| 7 — BH and the frame | 33 | 9.8e-16 |
| 8 — rank scores | 44 | **0** (bitwise) |
| divergences — `tail_conc` components | 29 | **0** (bitwise) |
| Rust kernel | 94 | 2.2e-15 |

For scale, `porting-hazards.md` sets the interpretive threshold at 1e-12
relative for a real bug and 1e-14 for summation order. The worst number
here is 9.2e-15, and it is in the GPD's `xi` — an `exp`/`pow` difference
between two C libraries, not an algorithmic one.

The bitwise results are the load-bearing ones. **The normalized matrix, the
quantile grids, the null matrices and the rank scores are bit-for-bit
identical to R**, which means hazards 1, 8, 9 and 11 are excluded exactly
rather than to a tolerance.

Two implementation choices are what bought that, and both are documented at
the code:

- **Type-7 is implemented directly, mirroring R's arithmetic**, rather than
  delegated to `numpy.quantile`. R evaluates `(1-h)*x[lo] + h*x[hi]` while
  NumPy uses a two-sided lerp that switches formula at `t >= 0.5`; both are
  correct type 7 and they differ in the last bits. R also **skips the
  interpolation** where `x[hi] == x[lo]`, which matters because
  `(1-h)*a + h*a` is not guaranteed to be exactly `a` — and WADE's target
  regime is zero-heavy count data, which is nothing but ties.
- **The score product is written out left to right.** Factoring the two
  shared ECDF terms into a subexpression changes `((s*A)*B)*C` into
  `(s*A)*(B*C)`. The scores still agreed to 1e-16, but `dense_rank` breaks
  ties on **exact** equality, so a one-ulp difference merged a tie that R
  did not have and shifted six ranks. Found by the suite; fixed by matching
  the association.

### One correction to `algorithm.md`

§1.1 states that the grid nodes "coincide exactly with the order statistics
of the smaller group". That is exact in real arithmetic and only to
floating point in practice: `1 + (m-1)*q` does not always land on an
integer. **Measured at m = 8** the indices come out as
`8, 7, 6, 5, 4, 3.0000000000000009, 2.0000000000000004, 1`, so two nodes
interpolate. **R does the same thing** — `matrixStats::rowQuantiles` and
`stats::quantile(type = 7)` both fail an `identical()` against the sorted
sample at m = 8, 13 and 21 while agreeing bitwise with each other — so this
is a property of the method's arithmetic, not a divergence. Asserted at
1e-15 rather than bitwise.

## 7. The two deliberate divergences

Both are asserted **positively**, stating what R does and what the port
does instead. An untested divergence is indistinguishable from an
oversight.

**`tail_conc`.** R guards at `abs(diff.mean * nprobs) < 1e-8`, which
reconstructs the ratio's own denominator and tests it against a fixed
absolute threshold. **Measured on the `tailconc` fixture**, that guard
catches **zero** of the pathological genes, and the smallest `|diff.mean|`
among them is above 1e-4 — many orders of magnitude from the threshold. The
port guards on conditioning instead: report only when
`sum(|D|) / |sum(D)| <= F`. Since `|sum(D over tail)| <= sum(|D|)`, that
bounds the output at `|tail_conc| <= F`, so the threshold and the guarantee
are the same number. Genes failing it get `nan` **and** a `tail_conc_ok`
flag, and the components stay available on `WadeStats`.

**A one-sample group.** R's reshape guard assumes a dropped dimension means
a single gene; at `nprobs == 1` it means a single probability, so a
length-`g` vector becomes `1 × g`, genes become probabilities, and the
frame reports one recycled value for every gene. The port refuses by
default with a message naming the condition, and computes the
mathematically correct per-gene answer under
`allow_single_sample_group=True` — which is still not R's.

## 8. Method validation, ported

The three simulations from `validation_sims_v7.R`, at the v7 geometry
(77 cases vs 18 controls, 1000 permutations). These cannot match R to the
digit — different generators, so different synthetic data and different
nulls — and they reproduce its behaviour closely:

| | port | R sandbox |
|---|---|---|
| (a) KS vs Uniform, bulk / subset | 0.4553 / 0.1673 | 0.6554 / 0.2327 |
| (a) type-I error at 0.05 | 0.0500 / 0.0612 | 0.0512 / 0.0488 |
| (b) subset genes, median `tail.mean` / `diff.mean` | 27,943 / 3,060 (ratio 9.01) | 27,537 / 3,046 (ratio 8.9) |
| (b) bulk shifts | 4,730 / 1,546 (ratio 3.18) | 4,064 / 1,348 (ratio 3.3) |
| (b) null background | 551 / −41 | 535 / −46 |
| (c) power at 3/5/8/12/20/35/50 % | 0,0,0,0,0,1,1 | 0,0,0,0,0,1,1 |

**The power table is identical**, including the combinatorial floor column
to three significant figures. The 0→1 step between 20% and 35% is asserted
against its cause rather than its location: BH at q = 0.10 over 80 true
positives among 680 genes can only declare p ≤ 0.0118, and the floor
crosses that threshold exactly there. The test asserts power is 0 at every
sweep point where the floor exceeds the BH threshold, so it tests the
mechanism, not a memorized curve.

## 9. The Rust kernel

Built last, on purpose. `ROADMAP.md`'s ordering constraint exists so that a
disagreement with R has one candidate cause instead of two; with the NumPy
path already validated, the kernel had a baseline known to be correct.

The boundary (open question O4) was cut as follows:

- **The kernel takes the normalized matrix and the permutation label
  matrix, and returns the full `g × nperms` null matrices** — not p-values.
  Returning the nulls is what makes layer 5 possible at all, and it is the
  only place a kernel bug is separable from a p-value bug. It is also
  required by the GPD refinement, which needs each refined gene's full null
  vector.
- **Type-7 is reimplemented in Rust**, removing a dependency on NumPy's
  behaviour inside the hot loop at the cost of two definitions that must
  agree. A dedicated test holds them to **bitwise** equality across group
  sizes 1–34, three data regimes and every tail window.
- **Summation is sequential**, matching R's `rowSums`. NumPy's pairwise
  summation is more accurate and therefore guaranteed to differ from R on
  some inputs; since R is the oracle, matching its association is what
  keeps the nulls bitwise.
- **`weight` and `log2_scale` are supported in the kernel**, unlike the
  reference, which falls back to its slow path when either is set. Silently
  moving a user from compiled code to a Python loop is worse than the R's
  behaviour, not better.
- **Parallel across permutations** with rayon, which changes nothing
  numerically: each permutation's arithmetic is self-contained.

**Measured.** The kernel's null matrices are **bitwise identical to R on
all 13 permutation scenarios**. Against the NumPy path it agrees to 1e-15
on the fixtures, on five randomized sweeps of group sizes and data regimes,
and on the pre-transform paths.

Speed, 16 rayon threads on this machine:

| scale | NumPy | Rust | speedup |
|---|---|---|---|
| 800 genes, 77v18, B=1000 | 0.91 s | 0.05 s | 18.6× |
| 2,219 genes, 60v22, B=2000 (cfRNA production) | 4.10 s | 0.20 s | 20.4× |
| 5,000 genes, 50v50, B=2000 | 13.90 s | 0.63 s | 22.2× |
| 20,000 genes, 50v50, B=1000 | 30.26 s | 1.35 s | 22.4× |

The three validation simulations run in **0.93 s** against the R sandbox's
~95 s, and produce the same table.

## 10. Decisions taken, and what is still open

Taken, with the reasoning at the code:

- **`nperms` defaults to 2000**, the source project's production constant.
  The reference has two different defaults and they are not
  interchangeable; one had to be picked and stated.
- **`gene_names=None` synthesizes positional identifiers** rather than
  dropping the column.
- **Invalid inputs raise**: a wrong-axis normalizer, a condition vector with
  labels outside {0, 1}, a malformed permutation matrix, a flat jitter
  vector. Each of these is silent in the R and each produces plausible
  output.
- **The rank scores are off by default** and live in their own module (O3,
  partially). They ship, because they are what the historical nomination
  ran on, but a caller has to ask.
- **`weight` and `log2_scale` are carried** (O5), in both backends, and are
  now tested — the reference material contains no evidence either was ever
  exercised.
- **CPM and RLE apply the jitter at count precision** but use consistent
  library sizes, deliberately not reproducing `tpm_like`'s per-cell
  denominator quirk. RLE's geometric-mean reference is restricted to genes
  with strictly positive **raw** counts, so a jitter draw of order 0.01
  cannot promote an undetected gene into the reference.

Still open, and deliberately not decided here:

- **O1 — the value of `F` for the `tail_conc` guard.** The *form* is
  settled and implemented; `DEFAULT_TAIL_CONC_MAX_FACTOR = 3.0` is a
  starting position, not a measurement. It is an output guarantee, so it
  should be picked as a promise. This wants a look at real data.
- **O2 — method-of-moments → MLE for the GPD.** Not attempted. The moment
  path is what the fixtures pin, so if the estimator becomes selectable the
  fixtures must continue to pin moments.
- **Permutation and p-value construction** more broadly — the user has
  references and ideas to bring to this later.
- **Packaging** (roadmap §7): wheels across platforms, and whether the
  validation simulations ship as tests, documentation, or both. They are
  currently tests.
