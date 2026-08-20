# Implementation notes

What a second implementation needs to know, and what the current one is verified
against. This consolidates the porting-process documents; the method itself is
in [`method.md`](method.md) and what it cannot do is in [`limits.md`](limits.md).

---

## 1. Parity with the R reference

WADE was ported from `reference/R/wade.R`, which remains in the repository as
the oracle the golden fixtures were generated from. The port is verified
independently of it: the fixtures are committed, so the test suite runs with no
R present.

**Worst-case relative deviation: 9.155e-15 across 325 comparisons.**

| layer | comparisons | worst relative deviation |
|---|---|---|
| 1 — grid scalars and `q` | 12 | **0** (bitwise) |
| 2 — library sizes, normalized matrix | 24 | **0** (bitwise) |
| 3 — `Q1`, `Q0`, `D` grids | 48 | **0** (bitwise) |
| 4 — per-gene reductions | 80 | 2.2e-16 |
| 5 — the `g × nperms` null matrix | 11 | **0** (bitwise) |
| 6 — p-values and the GPD | 95 | 9.2e-15 |
| 7 — BH and the frame | 17 | 9.8e-16 |
| Rust kernel | 38 | 8.9e-16 |

**What the fixtures are for, now that the statistic they validated has been
replaced.** The tail window is gone (`method.md` §7), so parity on `tail.mean`
and `tail.conc` no longer tests anything that runs. What they still pin is the
machinery *underneath* every statistic — normalization, the type-7 quantile
grids, the permutation null, the GPD refinement and BH — which the current
method uses unchanged, and which is exactly where a silent cross-language
disagreement would do the most damage. Parity runs pin
``alternative="greater"``, since the R oracle is one-sided and the port now
defaults to two-sided.

Interpretive threshold: **1e-12 relative is a real bug; 1e-14 is summation
order.** The worst figure here is in the GPD's `xi` and is an `exp`/`pow`
difference between two C libraries, not an algorithmic one.

The suite is built **inside-out** and should be read that way: intermediate
quantities localize a disagreement, endpoint quantities only detect one. If
`padj` differs, the cause could be the quantile type, the tail window, the grid
orientation, the exceedance broadcast, the GPD branch, the floor, or BH — seven
candidates and no information. If layers 1–3 have already passed, six are
eliminated before the endpoint is compared. **Stop at the first failing layer.**

---

## 2. Where two implementations silently disagree

Every item here returns plausible numbers of the right shape and sign. None of
them raises. They are ordered by how much damage they do.

### 2.1 Quantile convention — every number changes

R offers nine quantile types. WADE requires **type 7**, and it must be pinned
explicitly rather than inherited from a library default. On an even-length
vector with ties the seven common conventions give five different answers at
p = 0.75. Since a tail statistic at realistic `m` averages one or two of those
positions, a convention mismatch does not perturb the statistic — it replaces it.

**Test on both an even- and an odd-length vector.** The odd-length case cannot
distinguish types 1, 2 and 7 at all, so a suite built only on odd lengths passes
with type 1 substituted.

Two further details, both of which the current implementation reproduces
deliberately rather than delegating to `numpy.quantile`:

- R evaluates the interpolation as `(1-h)·x[lo] + h·x[hi]`; NumPy's `linear`
  method uses a two-sided lerp that switches formula at `t >= 0.5`. Both are
  correct type 7 and they differ in the last bits.
- **R skips the interpolation where it cannot matter** — only where
  `index > lo` *and* `x[hi] != x[lo]`. On a run of equal values
  `(1-h)·a + h·a` is not guaranteed to be exactly `a`, and WADE's target regime
  is zero-heavy count data, which is nothing but ties.

Reproducing both is what makes the quantile grids bitwise identical rather than
merely close.

### 2.2 The RNG barrier — exact parity is impossible without design for it

R's Mersenne-Twister and NumPy's PCG64 are different algorithms; a shared
integer seed produces different numbers. Since every WADE output is a function
of the jitter and the permutation set, two implementations given the same counts
and seed produce two different answers, both correct.

**The only fix is to make both suppliable as inputs, on the same argument path
production uses.** A fixture path that bypasses production code validates a code
path nobody runs. `wade()` therefore accepts `jitter=` and `perms=` directly.

Distributional agreement — uniform p-values under the null, matching power
curves — should also be checked, but it cannot localize a disagreement. It
passes when the quantile type is wrong in a place where both are unbiased, when
the tail window is off by one, when the jitter fill order is transposed, and
when the exceedance count broadcasts along the wrong axis in a square fixture.
Use both; do not use only the second.

### 2.3 Broadcast axis in the exceedance count

R's `rowSums(perm >= obs)` recycles `obs` **down** each column, so gene *i*'s
observed value meets gene *i*'s null draws. NumPy broadcasts trailing dimensions
first, so a bare `null >= obs` aligns `obs` against the **permutation** axis —
and it only raises when the two lengths differ. In a square fixture it silently
compares gene *i*'s draw *j* against gene *j*'s observed value.

Correct form: `(null >= obs[:, None]).sum(axis=1)`. **Make every fixture
non-square** in genes vs samples *and* genes vs permutations; a 6 × 6 fixture is
worse than useless, it is a test that passes on broken code.

### 2.4 Sample versus population variance

The GPD fit uses R's `var`, the **sample** variance with denominator `n − 1`;
NumPy defaults to `ddof=0`. Both moment estimators are functions of `m²/v`, so
an understated variance pushes `xi` down and `sigma` up — and the branch test is
`xi <= 0`, so the wrong divisor can flip the branch. Measured `xi` values from
ordinary nulls include +0.031 and −0.133, so the boundary sits at the operating
point and is crossed by noise.

### 2.5 The GPD's branches and its floor

Six independently reachable outcomes, all of which the suite hits: the `xi > 0`
GPD form; the `xi <= 0` exponential limit; the floor binding; fewer than ten
exceedances; an observation at or below the threshold; and a degenerate variance.
A seventh, non-positive `sigma`, is **unreachable** — with strictly positive
exceedances and `v > 0`, `sigma` is always positive — and the reference's check
for it is defensive rather than live.

Three details that change the answer:

- **The exceedance inequality is strict** (`s > thr`), so ties at the threshold
  are excluded. `len(exc)` is normally exactly `n_tail` but is smaller whenever
  the null ties there, and permutation nulls of discrete-ish statistics do tie.
- **The rescaling uses the nominal `n_tail / B`**, not `len(exc) / B`. With ties
  those differ.
- **The `xi <= 0` branch takes the exponential limit**, not the GPD form,
  because a negative shape gives the GPD a hard upper bound at `−sigma/xi`
  beyond which a strong statistic would collapse to a machine-epsilon p-value.

Reaching the `gpd` branch in a test takes care: pushing the observed value far
into the tail makes the **floor** bind, which is a different branch. Test with
the observation at a moderate order statistic of the null as well as far out.

### 2.6 Summation order

Floating-point addition is not associative. `rowSums`, `numpy.sum` (pairwise
summation) and a sequential loop legitimately differ in the last bits — and
NumPy's pairwise form is *more* accurate, so it is guaranteed to differ from R
on some inputs. Merely reversing a summation order changes about two thirds of
genes at a relative magnitude of 1e-14.

Consequences: the Rust kernel sums **sequentially**, matching R's association,
which is what keeps the null matrices bitwise identical. And the cumulative
curve's endpoint identity is a **tolerance** assertion, not an exact one — it
sums the same terms in the opposite order.

**Assert bitwise equality only on integers, ranks and branch flags.**

### 2.7 Fill order in a supplied jitter

R fills `matrix(v, g, n)` column-major; NumPy's `reshape` is row-major. This is
irrelevant to production runs but becomes load-bearing the moment a fixture
supplies a jitter matrix: if the two languages disagree about how a flat vector
maps into a shape, the "same" jitter is a transposed scramble, and since the
jitter perturbs values by about one part in 10⁶ the result is small, structured,
and easily mistaken for accumulated floating-point error.

**Serialize 2-D arrays with their own shape, never as a flat vector plus
dimensions.**

### 2.8 BH-FDR

Not a parity risk at the level that matters — available implementations agree
with R to at most one unit in the last place. Two behaviours that are *not*
about the arithmetic do matter, because `wade()` produces all-missing p-value
columns when `nperms = 0`:

- **Missing values are dropped, not propagated**, the adjustment uses the
  reduced count, and the missing value is put back in position. R achieves this
  through lazy evaluation of its `n = length(p)` default, forced only after `p`
  has been subset to the non-missing entries.
- Ties receive equal adjusted values, and the monotonicity enforcement (the
  cumulative minimum from the largest p downward) is applied.

### 2.9 The normalization denominator

The subtlest arithmetic in the method, and the most dangerous shape of error.
Substituting the naive `lib_sizes[j]` for the per-cell denominator changes every
value by about **1.7e-6** relative and leaves within-row rank ordering **exactly
unchanged** — so the quantile grid, which reads only within-row order, produces
qualitatively identical output. It passes every plausibility check and fails
only an exact test. That is the argument for having one, and the reason the
normalization layer carries the tightest tolerance in the suite.

### 2.10 Where a correct port must DISAGREE with R

A suite that enforces agreement here enforces a bug, so the divergence is
asserted positively — stating what R does and what the port does instead —
because an untested divergence is indistinguishable from an oversight.

**A one-sample group.** `rowQuantiles` drops the dimension attribute when the
result has a single row *or* a single column, and R's guard assumes the former.
At `nprobs == 1` it means the latter, so a length-`g` vector is reshaped to
`1 × g`, genes become probabilities, and the frame reports one recycled value
for every gene — no error, right shape, wrong answer. The port refuses
`min(n0, n1) == 1` by default.

(`tail.conc` used to be the other one. The statistic it guarded has been
retired, so there is no longer a ratio to disagree about.)

---

## 3. Generating fixtures from the R

### The deterministic seam

`tools/r/wade_seam.R` sources `wade.R` **unmodified** and makes its two sources
of randomness injectable, without touching any namespace.

`sample(cond)` is unqualified, so it resolves lexically and is easy to shadow.
The jitter comes from `stats::runif`, and `::` bypasses lexical scoping — the
usual escape being `assignInNamespace`, which mutates the `stats` namespace for
the whole session. That is not needed: **base R's `` `::` `` is an ordinary
closure**, so it can itself be shadowed in the enclosing environment chain.
`wade.R` is sourced into an environment whose parent binds both `sample` and
`` `::` ``; the shim returns the jitter stub for `stats::runif` when armed and
delegates everything else to `getExportedValue()` unchanged.

`seam_selftest()` proves transparency: a native run and a seam run injected with
R's own realised draws agree **bitwise**.

### Why every number is written twice

Fixtures carry each double as a 17-significant-digit decimal *and* as a C99 hex
float. Both are needed because **R cannot verify its own decimal output**:
measured, `sprintf("%.17g")` produces correctly rounded digits but R's
`as.numeric` is not a correctly-rounded parser and reads 8 of 42 uniform draws
back one ULP off — and still does at `%.20g`. So R checks the hex channel
round-trips, and the Python loader asserts `float(dec) == float.fromhex(hex)` on
every value it reads. Hex is authoritative.

`NA` is kept distinct from `NaN`: `wade()` produces `NA` from its guard and from
`nperms = 0`, so collapsing them would erase a distinction the port is tested on.

### Regenerating

```bash
export PATH="/usr/local/bin:$PATH"
cd reference/R
RENV_CONFIG_SANDBOX_ENABLED=FALSE \
RENV_PATHS_CACHE="$HOME/Library/Caches/org.R-project.R/R/renv/cache" \
  Rscript ../../tools/r/generate_fixtures.R
```

The sandbox restores offline from a read-only cache; `renv::restore()` succeeds
and then fails on a socket it cannot bind, so run it twice. Details in
[`../reference/R/README.md`](../reference/R/README.md).

---

## 4. The Rust kernel

The serial permutation loop is the **entire cost** of the method; everything
else is a single pass. The kernel was written **last**, on purpose: with the
NumPy path already validated against R, a disagreement has one candidate cause
instead of two.

Boundary decisions:

- **It returns the full `g × nperms` null matrices, not p-values.** Asserting
  those elementwise against R's serial loop is the only place a kernel bug is
  cleanly separable from a p-value bug. It is also required by the GPD, which
  needs each refined gene's full null vector.
- **Type 7 is reimplemented in Rust**, removing a dependency on NumPy inside the
  hot loop at the cost of two definitions that must agree — held to bitwise
  equality by a dedicated test across group sizes 1–34 and three data regimes.
- **Summation is sequential**, matching R (§2.6).
- **Parallel across permutations** with rayon, which changes nothing
  numerically: each permutation's arithmetic is self-contained.

Measured, 16 threads: 18–22× the NumPy path, and null matrices **bitwise
identical to R** on every permutation scenario.

| scale | NumPy | Rust |
|---|---|---|
| 2,219 genes, 60v22, B=2000 | 4.10 s | 0.20 s |
| 20,000 genes, 50v50, B=1000 | 30.26 s | 1.35 s |

**The subset test has its own kernel** (`subset_null`), validated elementwise
against `wade.permutation._subset_null_numpy`, which remains the contract.
Before it, the subset test was 98% of the runtime at scale. Three design
choices, each stated in the kernel's own docs:

- **Parallel over genes, not permutations.** The two passes over the
  permutations (null moments of the bridge, then the standardized maximum) run
  for one gene on one thread, so the moment accumulators need no cross-thread
  reduction and the order of accumulation over permutations is the NumPy
  path's. Pass 2 re-reads the bridges pass 1 stored (per thread, `B · m`
  doubles, capped at 64 MB) and recomputes them above the cap.
- **One sort per gene.** A permutation only re-partitions the same values, so
  the sorted case and control groups are read off the gene's one sorted row by
  an O(n) walk.
- **`log2` only where type 7 interpolates.** Elsewhere the quantile is an order
  statistic whose log was taken once per gene; where the guard fires, `log2` is
  called on the interpolated value exactly as NumPy does. The inputs to `log2`
  are therefore the same, and so are the bits.

Measured, 16 threads — and every comparison **bitwise** (statistic, null,
`mu`, `sd`, `argmax_k`; worst relative deviation 0 over 111 comparisons):

| scale | NumPy | Rust | mean-shift kernel, same run |
|---|---|---|---|
| 2,000 genes, 100v100, B=500 | 8.4–9.3 s | 0.08–0.09 s | 0.13 s |
| 2,219 genes, 60v22, B=2000 | 10.1 s | 0.13 s | 0.24 s |
| 20,000 genes, 100v100, B=2000 | 307 s | 3.9 s | 5.8 s |

Single-threaded (`RAYON_NUM_THREADS=1`) the small case takes 0.82 s, so the
kernel is about 11× the NumPy path per thread before parallelism.

**Memory is the unsolved half.** The null is always materialized: peak RSS
tracks `g · B · 8` bytes plus overhead. `keep_null` controls retention, not
construction. A streaming path needs two passes anyway, because the GPD needs
full null vectors for refined genes.

---

## 5. Conventions

- **Genes are rows, samples are columns.** Nothing transposes and nothing checks.
- **The probability grid runs high to low.** Position 0 is the group maximum.
- **Group membership is by exact equality to 1 and 0.** R silently drops any
  other label from *both* groups, which changes `m`, which changes every number;
  the port raises instead.
- **The normalizer may be a per-gene vector or a genes × samples matrix.** R
  relies on recycling and has no length check, so a per-*sample* vector recycles
  cleanly and returns wrong numbers with no diagnostic — the one input error in
  the reference that produces plausible output. The port validates the shape.
- **`gene_names=None` synthesizes positional identifiers.** R defaults to
  `rownames(counts)`, and `tibble()` *drops* a `NULL` column, so the frame
  silently loses its identifier while every other column is correct.
