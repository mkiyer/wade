# The R implementation, function by function

> **Note on citations.** This document cites `reference/docs/` and
> `reference/R/downstream/`, which were pruned once the port was verified.
> They are quoted here as the provenance of specific claims; the full record,
> including digests, is in [`../reference/PROVENANCE.md`](../reference/PROVENANCE.md).

This document describes what [`reference/R/wade.R`](../reference/R/wade.R) actually
does, line by line, and which of its behaviours a port must reproduce. It is a
description of an implementation, not a specification of the method: the
language-agnostic mathematics is in [`algorithm.md`](algorithm.md) and the
reasoning behind the statistic is in [`rationale.md`](rationale.md). Where the two
differ — where the R does something the mathematics does not require, or does it
in a way that is observable in the output — this document is the authority, because
the R is the reference implementation and the numbers it produces are what a port
will be tested against.

Three conventions are used throughout.

**Measured** means a number produced by running code in this repository's pinned R
sandbox during the session that wrote this document, using the invocation in
[`reference/R/README.md`](../reference/R/README.md). Such numbers are reproducible
by re-running the sandbox; they are not committed fixtures, because no fixtures
were generated (see [`design-decisions.md`](design-decisions.md)).

**Measured on the cfRNA cohort** means a number quoted from a source document in
`reference/docs/` or `reference/R/`, produced on data this repository does not
contain. A port's users have a different cohort and should expect different values.

**Design assertion** means a statement about intent or consequence that is not
itself a measurement.

## The ten functions

`wade.R` is 317 lines and defines ten top-level functions. Line numbers below are
verified against the file as staged.

| # | Function | Line | Role |
|---|---|---|---|
| 1 | `wade_lib_size()` | 59 | per-sample size factor from the (count, normalizer) pair |
| 2 | `wade_normalize()` | 63 | raw counts → TPM-like matrix, with the continuity jitter |
| 3 | `wade_stats()` | 83 | the statistic: quantile grids and all per-gene quantities |
| 4 | `.wade_null_stats()` | 120 | permutation-loop fast path; two quantities only |
| 5 | `.gpd_tail_p()` | 143 | Generalized Pareto tail refinement for one gene |
| 6 | `wade_perm_pvalues()` | 174 | empirical p across genes, then selective refinement |
| 7 | `wade()` | 193 | driver: normalize → observe → permute → BH → tidy frame |
| 8 | `wade_score()` | 257 | the two rank scores (nomination, not inference) |
| 9 | `wade_run()` | 283 | convenience wrapper from matrices plus two library-id vectors |
| 10 | `wade_gene()` | 304 | single-gene quantile and cumulative-area detail |

### Correction: the count of nine is wrong

`reference/docs/WADE_REPO_SCOPE.md` opens by
describing `wade.R` as "16 kB, 9 exported functions", and its two tables account
for nine: seven under *What moves: the method*, plus `wade_lib_size` and
`wade_normalize` under *What moves but should be reconsidered*. The file defines
**ten**. The omitted function is **`wade_run()`** (line 283), which appears in
neither table and is mentioned only in passing, as the signature the proposed
interface is "close to".

This matters beyond arithmetic: `wade_run()` is the entry point the historical
usage actually goes through. The v7 notebook calls it directly
(`reference/docs/notebook_sections/v7_section9_wade_discovery.md`, lines 131-133,
three contrasts), and the cfRNA contrast layer calls it as its single point of
contact with the method (`reference/R/downstream/wade_contrasts.R`, line 451). A
port built from the scope document's tables alone will omit the function that
defines what a caller was expected to hand WADE. The discrepancy is already
recorded in [`reference/PROVENANCE.md`](../reference/PROVENANCE.md) under
*Discrepancies found during staging*; it is repeated here because this is the
document a porter reads for the function inventory.

One further point of vocabulary: "exported" is aspirational. `wade.R` is
`source()`d, not installed, so nothing is exported in R's sense. The distinction
the file draws is by naming convention — two functions are dot-prefixed
(`.wade_null_stats`, `.gpd_tail_p`) to mark them internal.

## Conventions that cut across every function

Establish these once; they are assumed silently in every function below.

**Genes are rows, samples are columns.** Every matrix argument is genes × samples,
every returned per-gene quantity is a vector of length `nrow`, and every quantile
grid is genes × probabilities. Nothing in the file transposes, and nothing checks.

**Matrices are column-major.** R fills `matrix(v, g, n)` down the first column
before starting the second. This is observable in the jitter draw and is a
concrete porting hazard; see [`porting-hazards.md`](porting-hazards.md).

**There is no argument validation anywhere in the file.** No dimension checks, no
class checks, no range checks on `tail_q`, no check that `cond` is binary, no check
that `length(normalizer)` matches `nrow(counts)`. The single guard is
`stopifnot(n1 > 0, n0 > 0)` in `wade_stats()` (line 87). Several of the resulting
failure modes are silent rather than loud, and they are documented at the function
that owns them.

**The test is one-sided, upper-tail.** Both p-values count permutation statistics
at or above the observed value (`wade_perm_pvalues()`, line 176). A gene strongly
*reduced* in cases is not significant on either axis. Measured: on a synthetic
40-gene matrix at 12 versus 12 with one gene planted strongly down in cases and one
planted strongly up, the down gene returned `diff.mean` −18,545 with `p.diff` =
1.0000 and `tail.mean` −41,645 with `p.tail` = 1.0000, while the up gene returned
`diff.mean` +108,355 with `p.diff` = 0.0173 and `tail.mean` +127,412 with `p.tail`
= 0.0276. The effect sizes carry the direction; the p-values do not.

**`stats::` and package-qualified calls.** The file `library()`s only
`matrixStats` (line 31). Everything else is called with an explicit namespace:
`stats::runif`, `stats::var`, `stats::p.adjust`, `stats::ecdf`,
`matrixStats::rowQuantiles`, `tibble::tibble`, `dplyr::dense_rank`, `dplyr::desc`.

---

## 1. `wade_lib_size(counts, normalizer)` — line 59

```r
colSums(as.matrix(counts) / normalizer)
```

One line. Divides the count matrix by the normalizer, then sums down each column,
returning a vector of length `ncol(counts)`. This is the per-sample size factor
that `wade_normalize()` divides by; it is *not* a count of reads, and it is not on
any interpretable scale — it is the sum of normalizer-adjusted counts, whose units
depend entirely on what the normalizer is.

**The two normalizer forms.** The division is R's arithmetic, so the argument may be
either shape and the file relies on that:

- **A vector of length `nrow(counts)`** — one value per gene. R recycles it down
  the flattened column-major array, which for a length-`nrow` vector means it
  aligns with rows and repeats identically across every column. The cfRNA use is
  intron count per gene (splice-junction counts per intron, giving "sjTPM").
- **A genes × samples matrix** — element-wise division, one normalizer per
  gene per sample. The cfRNA use is effective length, which varies per library with
  the fragment-length distribution; `counts / efflen` scaled to library size
  reproduces standard TPM exactly. This is the form the production contrast layer
  passes (`reference/R/downstream/wade_contrasts.R` lines 447-448 take
  `counts[gi, ]` and `normalizer[gi, ]`).

**Failure mode: the vector form has no length check, and misalignment is silent.**
R's recycling rule warns only when the shorter length does not divide the longer.
`nrow × ncol` is frequently divisible by `ncol`, so passing a per-*sample* vector
where a per-*gene* vector belongs recycles cleanly and returns wrong numbers with
no diagnostic. Measured: on a 6 × 4 matrix of all-ones counts with the correct
length-6 normalizer `c(1,2,4,8,16,32)`, `wade_lib_size()` returns 1.96875 for all
four samples; with a length-4 normalizer `c(1,2,4,8)` — the wrong axis — it returns
3.37500, 2.25000, 3.37500, 2.25000, with no error and no warning. A length-5
normalizer does emit R's "longer object length is not a multiple of shorter object
length" warning and still returns a value. Design assertion: a port should
validate the normalizer's shape against `counts` and raise, because this is the
one input error in the file that produces plausible output.

## 2. `wade_normalize(counts, normalizer, lib_sizes, noise = 0.01, norm_factor = 1e6, seed = 1L)` — line 63

Four lines of body, and the subtlest arithmetic in the file:

```r
counts <- as.matrix(counts)
g <- nrow(counts); n <- ncol(counts)
if (!is.null(seed)) set.seed(seed)
nz    <- matrix(stats::runif(g * n, 0, noise), g, n)
y     <- (counts + nz) / normalizer
denom <- sweep(nz / normalizer, 2, lib_sizes, "+")
norm_factor * y / denom
```

### The jitter

`nz` is a genes × samples matrix of Uniform(0, `noise`) draws, `noise` defaulting to
0.01 — a hundredth of one count. It is added to the **counts**, before any
division, and it is drawn **once**: `wade()` calls `wade_normalize()` a single time
and then permutes labels on the fixed result, so the permutation null is
*conditional* on this one noise draw. That is a deliberate choice — it keeps the
noise out of the null — and it is stated in the file's header comment (lines 51-54)
and in `wade()`'s (lines 189-191).

`set.seed(seed)` runs only when `seed` is non-`NULL`. With the default `seed = 1L`
the jitter is fixed. With `seed = NULL` the function draws from the ambient RNG
state and consumes `g × n` uniforms from it; measured, two consecutive
`seed = NULL` calls return different matrices, while two calls each preceded by the
same `set.seed()` return identical ones.

The jitter serves three purposes, only the first of which the file names:

1. **Tie-breaking.** In sparse, zero-heavy count data many samples share a count of
   zero, so the quantile grid degenerates into flat runs. The jitter separates them.
2. **Strict positivity.** After jitter every entry is positive, so `fc = s1/s0` and
   `log2(fc)` in `wade_score()` are finite even for genes with all-zero counts in
   one group. Measured: on a 6 × 8 matrix with one row set entirely to zero and
   another zeroed in the control half, every `fc` and every `diff.frac` returned
   finite, and `wade_score()` ran without warning.
3. **A defined value at zero library size.** See the fixed point below.

### The denominator, worked through

This is the part a port will misread. The naive reading is that the denominator
should be `lib_sizes[j]`, giving the ordinary TPM form `1e6 · (C/N) / L`. It is
not; it is `nz[g,j]/normalizer[g,j] + lib_sizes[j]`, per cell.

The algebra. `lib_sizes` was computed by `wade_lib_size()` from the **unjittered**
counts:

```
L[j] = sum over genes h of  C[h,j] / N[h,j]
```

The numerator, however, uses jittered counts. Dividing a jittered numerator by an
unjittered library size is inconsistent: the row sums of the numerator no longer
sum to `L[j]`. The consistent library size for the jittered matrix would be

```
L'[j] = sum over h of (C[h,j] + nz[h,j]) / N[h,j]  =  L[j] + sum over h of nz[h,j]/N[h,j]
```

— that is, `L[j]` plus the whole column's jitter contribution. What
`wade_normalize()` adds instead is only **this gene's own** jitter contribution:

```
denom[g,j] = L[j] + nz[g,j] / N[g,j]
```

So the per-cell denominator is the library size in the counterfactual where gene
`g` alone received jitter and every other gene did not. Measured, and this is the
check that pins the interpretation: constructing that counterfactual explicitly —
for each `(g, j)`, jitter only gene `g`, recompute `sum(C[,j]/N[,j])` from the
singly-jittered column, and divide — reproduces `wade_normalize()`'s output to a
maximum absolute difference of 7.3 × 10⁻¹² on a 50 × 4 example (agreement to
floating point, not bitwise, because the two expressions sum in different orders).

`sweep(nz / normalizer, 2, lib_sizes, "+")` is how that is written: form the
matrix of per-cell jitter contributions `nz/N`, then add `lib_sizes[j]` to every
entry of column `j` (`MARGIN = 2`).

**Three observable consequences, all measured.**

*Column sums are not exactly `norm_factor`.* On an 800 × 6 Poisson(30) matrix with
per-gene normalizers drawn from 1..12, the unjittered TPM form gives column sums of
exactly 1,000,000.00 in all six columns, while `wade_normalize()` gives
1,000,161.99, 1,000,159.48, 1,000,164.98, 1,000,163.94, 1,000,166.29,
1,000,167.22. The excess is the difference between the per-gene and per-column
jitter corrections. WADE's output is therefore TPM-*like*, not TPM: it does not sum
to a fixed constant. Nothing downstream depends on it summing to one million, but a
port that "fixes" this to restore the constant has changed every number.

*The naive denominator is wrong by a small, structured amount.* Same matrix at 20
samples: substituting `lib_sizes[j]` for the per-cell denominator changes values by
at most 1.68 × 10⁻⁶ relative, median 1.31 × 10⁻⁷. That is bounded above by
`noise / (min(normalizer) · min(lib_size))`, which evaluates to 1.699 × 10⁻⁶ on
this example — consistent with the observed maximum. Within-row rank ordering was
unchanged in 0 of 800 genes. **This is the dangerous shape of error**: the quantile
grid reads within-row order, the order does not move, so the statistic's
qualitative behaviour is preserved and every number is wrong in roughly the
seventh significant figure. It will pass every plausibility check and fail an exact
parity test, which is the argument for having one.

*A gene that is the entire library normalizes to exactly `norm_factor`.* If gene
`g` is the only nonzero count in column `j`, then `L[j] = C[g,j]/N[g,j]` and
`denom[g,j] = (C[g,j] + nz[g,j])/N[g,j]`, which is exactly the numerator, so the
ratio is exactly 1 and the value is exactly `norm_factor`. Measured: 1000000.0000000000.
The same identity is what gives a defined answer when a whole column is zero —
there `L[j] = 0` and `denom[g,j] = nz[g,j]/N[g,j]`, so the numerator and
denominator are again equal and **every gene in an all-zero sample returns exactly
1e6**. Measured on a 3 × 4 matrix whose first column is entirely zero: all three
genes returned 1e+06 in that column. Design assertion: this is not a sensible
value for an empty library, it is an artefact of the algebra, and a port should
either refuse zero-library columns or document the behaviour. Nothing in `wade.R`
detects the case.

## 3. `wade_stats(tpm, cond, tail_q = 0.10, log2_scale = FALSE, weight = 1)` — line 83

The statistic. Takes an already-normalized matrix and a condition vector; returns a
list of 14 elements.

### Group assignment

```r
cond <- as.integer(cond)
i1 <- which(cond == 1L); i0 <- which(cond == 0L)
n1 <- length(i1); n0 <- length(i0)
stopifnot(n1 > 0, n0 > 0)
```

Membership is by **exact equality to 1 and to 0**, not by truthiness or by
complement. A sample whose `cond` is anything else — 2, `NA`, −1 — is silently
dropped from *both* groups. Measured: `cond = c(1L, 0L, 2L, 0L)` gives `n1` = 1,
`n0` = 2, `nprobs` = 1, and column 3 contributes to nothing. The only check is
that neither group is empty. Design assertion: this is a reasonable convention but
a port should validate that the labels partition the columns, because a dropped
sample changes `nprobs`, which changes the quantile grid, which changes every
number — with no diagnostic.

### The two optional transforms

```r
X <- tpm
if (weight != 1) X <- sweep(X, 2, ifelse(cond == 0L, weight, 1), "*")
if (log2_scale)  X <- log2(X + 1)
```

`weight` multiplies the **control** columns (those with `cond == 0L`) by `weight`,
before quantiles are taken. The header comment (line 80) describes `> 1` as
upweighting controls "(stringency)": inflating the control distribution makes the
case-minus-control difference smaller and the test harder to pass. Note the
`ifelse` keys on `cond == 0L`, so a sample with an out-of-range label receives
weight 1 — it is excluded from the groups but would still be scaled as a case if
`weight` were applied by complement. It is not.

`log2_scale` applies `log2(x + 1)` *after* weighting, so the two interact
multiplicatively-then-logarithmically rather than commuting.

Measured: **neither option is used anywhere in the staged reference material.** A
search across every `.R`, `.qmd` and `.md` file under `reference/` for `weight =`
or `log2_scale` outside `wade.R`'s own definitions returns nothing. Every call
site — the three validation-simulation calls, the notebook's three `wade_run()`
calls, the contrast layer — uses the defaults. The consequence for the driver is
recorded at `wade()` below and the question of whether to carry them at all is
open in [`design-decisions.md`](design-decisions.md).

### The quantile grid

```r
nprobs <- min(n0, n1)
q  <- seq(1, 0, length.out = nprobs)                 # high -> low
```

Two things are load-bearing.

**The grid has `min(n0, n1)` points.** The number of probabilities is set by the
*smaller* group, not by a parameter. This is the sample-size constraint that the
cfRNA screen exists to report and that [`limits.md`](limits.md) covers: `nprobs` is
resolution, and it cannot be raised by any argument.

**The grid runs high to low.** `seq(1, 0, length.out = nprobs)` starts at
probability 1 and descends to 0, so **column 1 is the maximum and the first `k`
columns are the upper tail**. Every tail quantity in the file indexes `1:k` and
depends on this ordering. A port that builds an ascending grid and keeps `1:k` will
compute a *lower*-tail statistic and call it `tail.mean` — a sign-flipped,
wrong-end number that will not error. Measured small cases: `nprobs = 1` gives the
single probability `1`; `nprobs = 2` gives `1, 0`; `nprobs = 3` gives `1, 0.5, 0`.

```r
Q1 <- matrixStats::rowQuantiles(X[, i1, drop = FALSE], probs = q, useNames = FALSE)
Q0 <- matrixStats::rowQuantiles(X[, i0, drop = FALSE], probs = q, useNames = FALSE)
if (is.null(dim(Q1))) { Q1 <- matrix(Q1, nrow = 1); Q0 <- matrix(Q0, nrow = 1) }
D  <- Q1 - Q0
k  <- max(1L, ceiling(tail_q * nprobs))              # upper-tail window
```

Quantiles are R's **type 7** (the default for both `stats::quantile` and
`matrixStats::rowQuantiles`), verified identical between the two in the sandbox.
Type 7 is linear interpolation between order statistics with the plotting position
`(i−1)/(n−1)`. The convention is a hard compatibility point and is treated at
length in [`porting-hazards.md`](porting-hazards.md).

The result is two genes × `nprobs` matrices, and `D = Q1 − Q0` is the signed
quantile difference. `k` is the tail window: `ceiling(tail_q · nprobs)`, floored at
1, so it is never zero but is frequently 1 or 2 at realistic group sizes.

### The reshape guard, and a defect

`matrixStats::rowQuantiles` drops the dimension attribute when its result has a
single row **or** a single column, and the guard restores it as `matrix(., nrow = 1)`.
That is correct for the single-*gene* case, which is how `wade_gene()` calls this
function: one row, `nprobs` probabilities, reshaped back to 1 × `nprobs`.

It is wrong for the other collapse case. When `nprobs == 1` — which happens
whenever the smaller group has exactly one sample — the result is one value per
gene, a vector of length `nrow`, and the guard reshapes it to a **1 × nrow** matrix.
Genes become columns. Every downstream `rowSums`/`rowMeans` then reduces across
*genes* instead of across probabilities, and the function returns length-1
statistics for a many-gene input.

Measured, on a 5-gene matrix with `cond = c(1L, 0L, 0L, 0L)`: `nprobs` = 1, `k` =
1, `dim(Q1)` = 1 × 5, `length(diff.mean)` = 1 for a 5-row input. Carried into the
driver, `wade()` returns a 5-row tibble in which `diff.mean` is the *same number*
in all five rows (−67368.6748), because a length-1 column recycles. No error, no
warning, and the output has the right shape. Design assertion: this is a silent
wrong answer, not merely an edge case, and a port must distinguish the two collapse
conditions — reshape on `nrow == 1`, and either reshape correctly or refuse when
`nprobs == 1`. A one-sample group has no quantile function worth comparing, so
refusing is defensible.

### What is returned

```r
list(
  q = q, nprobs = nprobs, k = k, n1 = n1, n0 = n0, Q1 = Q1, Q0 = Q0, D = D,
  diff.mean = sD / nprobs,                          # = mu_case - mu_ctrl
  w1        = rowMeans(abs(D)),                     # 1-Wasserstein distance
  tail.mean = rowMeans(D[, 1:k, drop = FALSE]),     # SUBSET axis
  tail.conc = rowSums(D[, 1:k, drop = FALSE]) / sD, # share of signed area in tail
  fc         = s1 / s0,
  cond1.mean = s1 / nprobs,
  cond0.mean = s0 / nprobs,
  tot.mean   = (s1 + s0) / nprobs
)
```
with `s1 = rowSums(Q1)`, `s0 = rowSums(Q0)`, `sD = rowSums(D)`.

The eight grids and scalars (`q`, `nprobs`, `k`, `n1`, `n0`, `Q1`, `Q0`, `D`) are
returned for plotting and for `wade_gene()`; `wade()` discards them. A port should
keep them reachable — they are what localizes a parity failure, and they are what
`wade_gene()`'s diagnostic panel is built from.

**`diff.mean` and `cond1.mean`/`cond0.mean` are grid quantities, not sample
means.** This is the most easily missed convention in the file. `cond0.mean` is the
mean of the *interpolated quantile grid*, `rowSums(Q0)/nprobs`. For the group whose
size equals `nprobs` — the smaller group — the type-7 grid evaluated at
`nprobs` equally spaced probabilities from 1 to 0 hits every order statistic
exactly, so the grid mean recovers the sample mean. Measured on a 4 × 28 matrix at
20 cases versus 8 controls (so `nprobs` = `n0` = 8): the control grid mean matches
the control sample mean to a maximum absolute difference of 7.1 × 10⁻¹⁵ — equal to
floating point, not bitwise, because the two sum in different orders.

For the **larger** group it does not. The 20 case samples are read at only 8
probabilities, so `cond1.mean` is the mean of an 8-point subsample of the case
quantile function. Measured on the same example, `cond1.mean` differs from the case
sample mean by up to 14.96% relative. Consequently `diff.mean` is the difference of
two grid means and *not* the difference of two sample means, despite the file's
header comment writing it as `mu_case − mu_ctrl` (lines 15-16). The identity holds
exactly on the smaller group and approximately on the larger; the discrepancy grows
with group imbalance. Design assertion: this is a property of the estimator, not an
error — comparing two quantile functions on a shared grid is the method — but a
port that "corrects" `cond1.mean` to the sample mean will change `fc`,
`diff.frac`, `score` and `rank` while leaving `diff.mean` inconsistent with them.

Two exact identities follow from the definitions and were confirmed numerically:
`tot.mean == cond1.mean + cond0.mean`, and `fc == cond1.mean / cond0.mean`. Note
what the second means: `tot.mean` is a **sum**, not an average of the two groups,
so `diff.frac = diff.mean / tot.mean` lies in [−1, 1] and is a normalized
contrast rather than a fraction of a mean. Also, since
`diff.mean = (s1 − s0)/nprobs = cond1.mean − cond0.mean` exactly,
`sign(diff.frac)` and `sign(log2(fc))` always agree — which is why `wade_score()`
can take its sign from one and its magnitude from the other.

**`tail.conc` is a ratio with a vanishing denominator.** `rowSums(D[, 1:k]) / sD` —
signed tail area over signed total area. When the lower quantiles' differences
cancel the upper ones, `sD → 0` and the ratio diverges. It is guarded, badly, in
`wade()`; the quantification and the fix are in
[`porting-hazards.md`](porting-hazards.md). Note also that a constant gene gives
`0/0`. Measured: a row set to a single repeated value returns `diff.mean` 0, `w1`
0, `tail.mean` 0, `fc` 1, and `tail.conc` `NaN` — the one pathology `wade()`'s
existing guard does catch.

## 4. `.wade_null_stats(tpm, cond, nprobs, q, k)` — line 120

The permutation loop's fast path. It recomputes group membership from the permuted
`cond`, takes the same two `rowQuantiles` calls, and returns only the two
quantities the null needs:

```r
list(diff.mean = rowSums(D) / nprobs,
     tail.mean = rowMeans(D[, 1:k, drop = FALSE]))
```

Three properties matter for a port.

**It is numerically identical to `wade_stats()` on the defaults, by construction
rather than by coincidence.** The expressions are the same expressions:
`wade_stats()` computes `sD/nprobs` where `sD = rowSums(D)`, and this computes
`rowSums(D)/nprobs`; both compute `tail.mean` as `rowMeans(D[, 1:k])`. Same
quantile calls, same subtraction, same reductions, same order. So the observed and
null statistics are commensurable in the strict sense — a port that vectorizes the
null must preserve not just the formulas but the reduction order if it wants exact
agreement with the observed values (see the summation-order note in
[`porting-hazards.md`](porting-hazards.md)).

**It exists only for speed, and it takes `nprobs`, `q` and `k` as arguments**
precisely because those depend only on group sizes and `tail_q`, which do not
change under permutation. Hoisting them out of the loop is the optimization. It
does *not* support `weight` or `log2_scale`; the driver falls back to full
`wade_stats()` when either is non-default.

**It carries the same reshape defect** as `wade_stats()` — the identical
`is.null(dim(Q1))` guard on line 124 — so the `nprobs == 1` pathology is present in
the null as well.

## 5. `.gpd_tail_p(o, null, n_tail = 250)` — line 143

Scalar math on one gene's null vector. The permutation null's upper tail is fit by a
Generalized Pareto distribution and the p-value read from its CDF, following
Knijnenburg et al. 2009. Every step is arithmetically explicit, so a port can be
tested branch by branch.

```r
B <- length(null)
n_tail <- min(n_tail, floor(B / 2))
s   <- sort(null, decreasing = TRUE)
thr <- s[n_tail + 1]
exc <- s[s > thr] - thr
emp <- (1 + sum(null >= o)) / (B + 1)
if (length(exc) < 10 || o <= thr) return(emp)
m <- mean(exc); v <- stats::var(exc)
if (!is.finite(v) || v <= 0) return(emp)
y <- o - thr
p_floor <- 1 / (B * n_tail)
xi    <- 0.5 * (1 - m^2 / v)
sigma <- 0.5 * m * (1 + m^2 / v)
if (!is.finite(sigma) || sigma <= 0) return(emp)
if (xi <= 0) {
  tail_prob <- exp(-y / sigma)
} else {
  tail_prob <- (1 + xi * y / sigma)^(-1 / xi)
}
max((n_tail / B) * tail_prob, p_floor)
```

**The threshold and the exceedances.** `n_tail` is capped at `floor(B/2)` — at most
half the null can be called "tail". The null is sorted **descending**, and the
threshold is the `(n_tail + 1)`-th value, i.e. the first value *below* the tail
window. Exceedances are `s[s > thr] - thr`, using a **strict** inequality, so ties
at the threshold are excluded. That has a consequence: `length(exc)` is normally
exactly `n_tail`, but is smaller whenever values tie at `thr`, and can be zero if
the top `n_tail + 1` values are all equal. Measured: a null of `rep(5, 400)`
followed by `rexp(1600, 5)` gives `thr` = 5 and `length(exc)` = 0, and the function
returns the empirical p exactly (0.00049975, matching `(1 + sum(null >= 9))/(B + 1)`).

**The two bail conditions on the first `return`.** Fewer than ten exceedances, or an
observed value at or below the threshold. The second is the common one: refinement
is only meaningful for a statistic beyond the fitted region. Measured: asking for
the p-value of the 300th largest null value (below `thr`) returns the empirical
0.150425.

**`stats::var` is the sample variance, denominator `n − 1`.** This is a live
cross-language hazard, since NumPy's default is `ddof = 0`. Measured on
`c(1, 2, 3, 4, 10)`: R's `var` = 12.5, population variance = 10.0, ratio 1.25. The
moment estimators are functions of `m²/v`, so the wrong divisor perturbs both `xi`
and `sigma` and can flip the branch.

**The method-of-moments estimators.** For a GPD with shape `xi` and scale `sigma`,
matching the first two moments of the exceedances gives
`xi = ½(1 − m²/v)` and `sigma = ½m(1 + m²/v)`. Both are closed-form and
dependency-free, which is why the file uses them; the documented upgrade to
maximum likelihood is open (see [`design-decisions.md`](design-decisions.md)).

**Two branches.** For `xi > 0` the GPD survival function
`(1 + xi·y/sigma)^(−1/xi)` is used directly. For `xi <= 0` the file switches to the
`xi → 0` **exponential** limit `exp(−y/sigma)` rather than the GPD form, and the
comment (lines 165-167) says why: a negative shape gives the GPD a hard upper
bound at `−sigma/xi`, beyond which the survival function is zero, so a strong
observed statistic would collapse to a machine-epsilon p-value. The exponential
limit is defined for all `y ≥ 0` and is conservative by comparison. Note that this
means the fitted `xi` is used *only* to choose the branch when it is non-positive;
its magnitude is discarded there, and only `sigma` survives.

**The scaling and the floor.** `tail_prob` is a conditional probability — given the
statistic exceeds `thr` — so it is multiplied by `n_tail / B`, the empirical
probability of exceeding `thr` (0.125 at the defaults). The result is then floored
at `1/(B · n_tail)`. The comment (lines 155-160) frames the floor as an honesty
constraint rather than a numerical convenience: with `B` permutations and `n_tail`
tail points, that is the smallest defensible tail probability, and returning
machine-epsilon values would create false ties and mis-order strong hits.

**Measured branch values.** Constructed nulls of length 2000, `n_tail` = 250,
observed set to a multiple of the null maximum. These are exact reference values a
port can target:

| null | observed | `thr` | `n_exc` | `m` | `v` | `xi` | `sigma` | p | branch |
|---|---|---|---|---|---|---|---|---|---|
| `rt(2000, 2.5)`, seed 1 | 20.509099 | 1.572381 | 250 | 1.133161 | 2.525647 | +0.24580 | 0.854633 | 6.372558e-05 | GPD |
| `rexp(2000, 1)`, seed 2 | 10.088564 | 2.089624 | 250 | 0.957363 | 0.723512 | −0.13340 | 1.085074 | 7.859260e-05 | exponential |
| `rnorm(2000)`, seed 3 | 6.334738 | 1.138272 | 250 | 0.496683 | 0.167755 | −0.23528 | 0.613544 | 2.621901e-05 | exponential |
| `runif(2000)`, seed 4 | 0.999536 | 0.870413 | 250 | 0.061106 | 0.001337 | −0.89642 | 0.115882 | 4.101967e-02 | exponential |
| `rexp(2000, 1)`, seed 5 | 167.333277 | 2.009647 | 250 | 0.989502 | 1.043189 | +0.03071 | 0.959113 | 2.000000e-06 | floor |

The floor is `1/(B · n_tail)` = 2.0 × 10⁻⁶ at `B` = 2000 and 4.0 × 10⁻⁶ at
`B` = 1000. Note the second row: an exponential null — whose true shape is
`xi = 0` — produced a moment estimate of −0.133 and took the exponential branch.
The branch boundary is therefore crossed routinely by sampling noise on
light-tailed nulls, and both branches are exercised in ordinary use, not just at
extremes. The fourth row shows the bounded case the fallback protects: a uniform
null gives `xi` = −0.896, and the GPD form would place a hard bound just above the
observed maximum.

## 6. `wade_perm_pvalues(obs, perm, n_exc_min = 10, n_tail = 250)` — line 174

Vectorized across genes, with a scalar refinement loop:

```r
B <- ncol(perm)
nexc <- rowSums(perm >= obs)                          # obs recycled per row
p <- (1 + nexc) / (B + 1)
refine <- which(nexc < n_exc_min & B >= 2 * n_tail)
for (i in refine) p[i] <- .gpd_tail_p(obs[i], perm[i, ], n_tail = n_tail)
p
```

**The recycling is the whole trick, and it is easy to get backwards.** `perm` is
genes × permutations and `obs` is length `nrow`. R compares column-major, so `obs`
is recycled *down* each column — gene `i`'s observed value against gene `i`'s null
draws, which is what is wanted. Measured on a 3 × 3 null with rows (10, 11, 12),
(20, 21, 22), (30, 31, 32) and `obs` = (11.5, 21.5, 100): `rowSums(perm >= obs)`
returns 1, 1, 0, matching the per-gene answer. In NumPy the equivalent is
`(perm >= obs[:, None]).sum(axis=1)`; a bare `perm >= obs` broadcasts along the
wrong axis and, when the number of genes happens to equal the number of
permutations, does so without error.

**The empirical p-value is `(1 + nexc) / (B + 1)`**, the add-one form. It cannot be
zero, its minimum is `1/(B + 1)`, and it is the value returned for every gene that
is not refined.

**Which genes get refined.** Those with fewer than `n_exc_min` = 10 exceedances —
i.e. only the genes whose empirical p-value is already at the resolution limit —
**and** only when `B >= 2 · n_tail`. That second condition is a property of the run,
not the gene: with the default `n_tail` = 250 it requires `B >= 500`. Measured:
`B` = 250 and `B` = 499 do not qualify, `B` = 500, 1000 and 2000 do. So at
`nperms < 500` **no refinement ever happens** and the smallest achievable p-value
is `1/(nperms + 1)`. Note that `.gpd_tail_p()` separately caps `n_tail` at
`floor(B/2)`, so the two guards are consistent but not identical — the gate here
uses the *requested* `n_tail`, not the capped one.

**One-sided, and no two-sided option exists.** `perm >= obs` counts only
exceedances above.

## 7. `wade(counts, normalizer, lib_sizes, cond, nperms = 1000, tail_q = 0.10, noise = 0.01, log2_scale = FALSE, weight = 1, seed = 1L, gene_names = rownames(counts), verbose = TRUE)` — line 193

The driver. Order of operations, which a port must preserve because the RNG state is
threaded through it:

1. **Normalize once.** `wade_normalize(counts, normalizer, lib_sizes, noise = noise, seed = seed)`. Note `norm_factor` is not forwarded — the driver cannot change it from 1e6.
2. **Observed statistics.** `wade_stats(tpm, cond, tail_q, log2_scale, weight)`.
3. **Allocate two `g × nperms` matrices** of `NA_real_`, one per axis.
4. **Reseed with `seed + 1L`.** A second, offset stream, so the permutations do not continue the jitter's stream. With `seed = NULL` this step is skipped and permutations draw from ambient state.
5. **Choose the path.** `lean <- (weight == 1 && !log2_scale)`.
6. **Loop `b` in `1:nperms`, serially.** Each iteration calls `sample(cond)` — a full permutation of the label *values* — and then either `.wade_null_stats(tpm, sample(cond), obs$nprobs, obs$q, obs$k)` or full `wade_stats()`. Results are written into column `b` of each matrix. Progress prints every 500 iterations when `verbose`.
7. **P-values on both axes** via `wade_perm_pvalues()`, at its defaults (`n_exc_min` = 10, `n_tail` = 250) — the driver exposes neither.
8. **BH on both axes** via `stats::p.adjust(., "BH")`.
9. **Return a tibble.**

When `nperms == 0` the loop is skipped and both p-value vectors are all `NA_real_`;
`p.adjust` of an all-`NA` vector is all-`NA`, so the four p-value columns are `NA`
and the frame is effect sizes only. That is a supported mode, not an error.

### The returned frame

Fourteen columns: `gene`, `diff.mean`, `diff.frac`, `w1`, `tail.mean`, `tail.conc`,
`fc`, `cond1.mean`, `cond0.mean`, `tot.mean`, `p.diff`, `p.tail`, `padj.diff`,
`padj.tail`.

`diff.frac = diff.mean / tot.mean` is computed here rather than in `wade_stats()`.

**A shape trap in the default argument.** `gene_names` defaults to
`rownames(counts)`, which is `NULL` for an unnamed matrix, and `tibble::tibble()`
**drops** a `NULL` column rather than erroring. Measured: with row names the frame
has 14 columns; without, it has 13 and there is no `gene` column at all — the
identifier silently disappears while every other column is present and correct.
Anything downstream that joins on `gene` breaks at that point rather than here.
Design assertion: a port should synthesize positional identifiers rather than drop
the column.

### The `tail.conc` guard

```r
tail.conc = ifelse(abs(obs$diff.mean * obs$nprobs) < 1e-8, NA_real_, obs$tail.conc)
```

Note that `diff.mean * nprobs` reconstructs `sD`, the ratio's denominator, so the
guard is on the denominator's absolute magnitude at a fixed threshold of 1e-8. At
`nprobs` = 22 that is `|diff.mean| < 5 × 10⁻¹⁰`. The pathology occurs at
denominators many orders of magnitude larger, so the guard almost never fires
where it is needed; it does catch the exact-zero case (a constant gene's `0/0`).
Quantification, structural cause, and the recommended replacement are in
[`porting-hazards.md`](porting-hazards.md). This is the one place where a port
*should* deviate from the R.

### Cost and memory, as arithmetic

The permutation loop is a serial `for` and is **the entire cost of the method**:
`nperms` iterations, each doing two `rowQuantiles` calls over a genes × samples
slice plus a subtraction and three reductions. Everything else in `wade.R` is one
pass. Design assertion: this is the only part of the file whose performance
matters, and it is the obvious target for a native kernel.

Memory is two `g × nperms` double matrices, allocated up front:
`2 · g · nperms · 8` bytes. At the cfRNA production scale — 2,219 genes,
`nperms` = 2000 — that is about 71 MB, which is why the R version holds them in
memory without comment. At 20,000 genes and 10,000 permutations it would be 3.2 GB,
which is not. A port that streams the null through the p-value accumulation instead
of materializing it changes this, but note that `.gpd_tail_p()` needs each refined
gene's *full* null vector, so a streaming implementation must either retain rows
for candidate genes or make two passes.

Measured runtime anchor, from [`reference/R/README.md`](../reference/R/README.md):
[`reference/R/validation_sims_v7.R`](../reference/R/validation_sims_v7.R) runs in
about 95 s on one core. That is nine `wade()` calls at `nperms` = 1000 on matrices of
800, 650 and 680 genes (one call each for the calibration and discrimination
simulations, seven for the power sweep), all at 95 samples.

## 8. `wade_score(df, log2fc_exp = 1, case_exp = 1, ctrl_exp = 1, tail_exp = 1)` — line 257

Takes a `wade()` frame and adds five columns. Not part of the test — the file's own
header (lines 240-255) calls `score` "the ORIGINAL panel-selection score, preserved
for continuity", and the framing question is open in
[`design-decisions.md`](design-decisions.md).

Four empirical CDFs are built **across genes**, from the frame itself:

```r
Ffc   <- stats::ecdf(abs(log2(df$fc)))
Fcase <- stats::ecdf(df$cond1.mean)
Fctrl <- stats::ecdf(df$cond0.mean)
Ftail <- stats::ecdf(df$tail.mean)
```

R's `ecdf` is `F(t) = #{x ≤ t} / n`, right-continuous, with ties counted together.
Measured on `c(3, 1, 4, 1, 5)`: the values at the sorted points are 0.4, 0.4, 0.6,
0.8, 1.0 — the two tied 1s both evaluate to 2/5. So `F` reaches 1 at the maximum and
is never 0 at an observed point, which means `(1 − Fctrl)` *can* be exactly 0 (for
the gene with the highest control mean) and every term is in [0, 1]. A port must use
`≤`, not `<`.

The three terms and the two scores:

```r
df$log2fc     <- log2(df$fc)
df$score      <- sign(df$diff.frac) * Ffc(abs(df$log2fc))^log2fc_exp *
                 Fcase(df$cond1.mean)^case_exp * (1 - Fctrl(df$cond0.mean))^ctrl_exp
df$tail.score <- sign(df$tail.mean) * Ftail(df$tail.mean)^tail_exp *
                 Fcase(df$cond1.mean)^case_exp * (1 - Fctrl(df$cond0.mean))^ctrl_exp
```

`score` rewards genes that are high in cases, **low in controls**, and large in
absolute fold change. `tail.score` replaces the fold-change term with the tail
statistic's own rank, keeping the case-high/control-low terms. Both are products of
three quantities in [0, 1] times a sign, so both lie in [−1, 1]. The four `_exp`
arguments tune term emphasis and are all 1 by default; no reference call site
changes them.

Note the asymmetry between the two: `Ffc` is the CDF of an **absolute** value, so
its term is a magnitude and the sign is supplied separately from `diff.frac`;
`Ftail` is the CDF of a **signed** value, so a gene with a strongly negative
`tail.mean` gets `Ftail` near 0 *and* a negative sign, giving a small negative
score. The two scores are therefore not the same construction with one term
swapped, despite the header describing them that way.

```r
df$rank      <- dplyr::dense_rank(dplyr::desc(df$score))
df$tail.rank <- dplyr::dense_rank(dplyr::desc(df$tail.score))
```

`dense_rank` gives ties the same rank with no gaps afterwards, and propagates `NA`.
Measured: `dense_rank(desc(c(0.5, 0.5, 0.2, NA, 0.9)))` returns 2, 2, 3, `NA`, 1.
Rank 1 is the best score. The scored frame has 19 columns.

## 9. `wade_run(counts, normalizer, case_libs, ctrl_libs, gene_names = NULL, nperms = 2000, tail_q = 0.10, seed = 1L, verbose = FALSE, ...)` — line 283

The convenience driver `WADE_REPO_SCOPE.md` omits, and the function the historical
call sites use.

```r
libs <- c(case_libs, ctrl_libs)
cond <- c(rep(1L, length(case_libs)), rep(0L, length(ctrl_libs)))
Xc   <- counts[, libs, drop = FALSE]
Xn   <- normalizer[, libs, drop = FALSE]
gn   <- if (is.null(gene_names)) rownames(counts) else gene_names
ls   <- wade_lib_size(Xc, Xn)
res  <- wade(Xc, Xn, ls, cond, nperms = nperms, tail_q = tail_q,
             seed = seed, gene_names = gn, verbose = verbose, ...) |> wade_score()
attr(res, "n_case") <- length(case_libs)
attr(res, "n_ctrl") <- length(ctrl_libs)
res
```

Five things it does that a caller would otherwise have to get right:

1. **Column subsetting by identifier.** Both matrices are subset to `libs` in
   case-then-control order, so `cond` is a block vector rather than a lookup. The
   library ids are used as column indices, so they must be column *names* (or
   positions).
2. **Condition coding**, 1 for case and 0 for control, in the same order.
3. **Library sizing on the subset**, not on the full matrix. This is the important
   one: `lib_sizes` is computed from `Xc`/`Xn` *after* subsetting, so a library's
   size factor depends only on the genes in the matrix handed in — change the gene
   set and every normalized value changes. The cfRNA cache key includes the gene
   count for exactly this reason (`wade_contrasts.R` lines 404-415).
4. **Scoring**, unconditionally piped through `wade_score()`.
5. **Group sizes as attributes** `n_case` and `n_ctrl`. Note it records the
   *requested* group sizes; if a library id appeared in both vectors or the labels
   were malformed, these would not match what `wade_stats()` used.

Two divergences from `wade()` worth carrying into an API:

**The default `nperms` is 2000 here and 1000 in `wade()`.** The cfRNA layer sets
`CFRNA_WADE_NPERMS = 2000L` and passes it explicitly, and the validation script
passes 1000; but a caller relying on defaults gets different resolution depending
on which function they call, and — because the refinement gate needs `B >= 500` —
different floor behaviour too.

**`wade_run()` supports only the matrix normalizer form.** `normalizer[, libs]`
requires two dimensions, so a per-gene vector fails. Measured: passing a length-`g`
vector raises `subscript out of bounds`. So `wade()` and `wade_normalize()` accept
both forms while `wade_run()` accepts one — an asymmetry a port should resolve
deliberately rather than inherit.

It does not record `nperms`, `tail_q` or `seed` as attributes; the cfRNA wrapper
attaches those itself (`wade_contrasts.R` lines 456-462), along with contrast id,
cohort, gene count and store fingerprint. Design assertion: the settings that
determine the numbers should travel with the result, and the R leaves that to the
caller.

## 10. `wade_gene(tpm_row, cond, tail_q = 0.10, log2_scale = FALSE, weight = 1)` — line 304

Per-gene detail for the diagnostic panel. Takes **one already-normalized row**, not
counts, and not the matrix.

```r
st <- wade_stats(matrix(tpm_row, nrow = 1), cond, tail_q, log2_scale, weight)
y1 <- rev(as.numeric(st$Q1)); y0 <- rev(as.numeric(st$Q0))
data.frame(
  p    = rev(st$q),                        # ascending 0 -> 1
  y1   = y1, y0 = y0,
  cum  = cumsum(y1 - y0) / st$nprobs       # endpoint = diff.mean
)
```

It reshapes the row to a 1 × n matrix and calls `wade_stats()` — the path the
single-row reshape guard exists for — then reverses everything to ascending
probability and accumulates.

### Why reverse *after* computing rather than before

Two distinct reasons, and only the second is about floating point.

**The statistics must be computed on the descending grid.** `wade_stats()` defines
the tail as the first `k` columns, which is the upper tail only because `q` runs
1 → 0. If a caller reversed the row's quantile grid before computing, `1:k` would
select the *lowest* `k` probabilities and `tail.mean` would measure the bottom of
the distribution while still being called `tail.mean`. The reversal must therefore
happen after the statistic, purely for display. This is the load-bearing part: it
is not an ordering preference, it is what keeps the plotted curve consistent with
the number the test used.

**Reversing changes the summation order, and the endpoint identity survives only to
floating point.** The curve is built to satisfy `cum[nprobs] == diff.mean`, so the
plot's endpoint is the statistic. In exact arithmetic that is trivially true. In
floating point, `diff.mean` is `rowSums(D)/nprobs` while the endpoint is
`cumsum(rev(D))[nprobs]/nprobs`, and these sum the same terms in opposite orders.

Measured, over 2,000 random rows at 16 cases versus 15 controls, comparing against
`wade_stats()`'s `rowSums`-based `diff.mean`:

| expression | rows differing bitwise (of 2000) |
|---|---|
| `sum(D) / nprobs` | 0 |
| `sum(rev(D)) / nprobs` | 1284 |
| `cumsum(rev(D))[nprobs] / nprobs` | 1284 |

and `sum(rev(D))` versus `cumsum(rev(D))`'s endpoint differ in 0 of 2000 — the
disagreement is entirely attributable to the reversal, not to accumulating. The
worst relative difference observed was 3.5 × 10⁻¹⁴.

[`reference/R/README.md`](../reference/R/README.md) reports the smoke test's
endpoint as equalling `diff.mean` "to floating-point equality", which is consistent
with this; the measurement above refines it usefully for a port. The identity is a
tolerance assertion, not an exact one, and a parity test that demands bitwise
equality here will fail on about two thirds of genes for reasons that have nothing
to do with correctness.

The returned frame has `nprobs` rows and four columns. `y1`/`y0` are the two
quantile functions for the panel's upper plot; `cum` is the cumulative signed area
for the lower one. The reading of that panel is recorded in
`reference/R/downstream/README.md`: a broad
shift rises steadily, a rare high subset stays flat then climbs inside the tail
window, and a single outlier stays flat then spikes.

---

## Dependency surface

Four packages, with very different degrees of entanglement.

| Package | Used for | Load-bearing? |
|---|---|---|
| **matrixStats** | `rowQuantiles` — four call sites, in `wade_stats()` and `.wade_null_stats()` | **Yes.** It is the statistic and it is the hot path. Its type-7 default is a compatibility point. |
| **stats** (base R) | `runif`, `var`, `p.adjust`, `ecdf` | Yes, but all four are ordinary numerical primitives with well-defined equivalents. |
| **tibble** | `tibble()` — one call, `wade()`'s return | **Incidental.** Any data frame would do. Its `NULL`-column-dropping behaviour is the only semantics that leaks (see `wade()` above). |
| **dplyr** | `dense_rank`, `desc` — two call sites, both in `wade_score()` | **Incidental.** Two ranking helpers in the non-inferential layer. |

So the method proper needs one real dependency and base numerics. The file's own
header (lines 27-29) states this as the design intent: "Dependencies: matrixStats
(fast row quantiles), and tibble for the result frame" — which under-reports
`dplyr`, used in `wade_score()`.

## What the consumer layer implies about the API

`reference/R/downstream/README.md` is
explicit that nothing in `reference/R/downstream/` is a porting target, and that
the cfRNA layers are there as evidence of what a caller needs. Four points from it
bear on the API rather than on cfRNA.

**The caller needs `nprobs` and `k` without running the test.** The cfRNA screen
computes `nprobs = min(n1, n0)` and `k_tail = max(1L, ceiling(tail_q * nprobs))`
itself, before deciding whether a contrast is worth running
(`wade_contrasts.R` lines 286-288), because the answer determines whether the
subset axis means anything. Design assertion: a port that exposes `tail_q` should
expose the arithmetic that turns it into a count of order statistics, rather than
making every caller re-derive it.

**Group definition belongs outside the statistic.** `control_strata.R` exists
because two group-assignment defects — a substring match that silently returned zero
controls, and an unregistered label defaulting into the wrong arm — were expensive.
The discipline that fixed them (a closed registry of reference groups, complement as
the case class, unregistered labels an error, and an audit function) is described in
the downstream README as a correctness pattern any two-group test needs. `wade.R`
takes an integer vector and asks no questions; that is the right boundary, and it
puts the obligation on the caller.

**The parameter constants are cohort decisions, not method properties.** From
`wade_contrasts.R` lines 112-130, with the file's own stated rationale:
`CFRNA_WADE_MIN_N = 10` is the floor to run at all, chosen because
`ceiling(0.10 × 10) = 1` — still a single point; `CFRNA_WADE_TAIL_MIN_N = 20` is
the threshold for *believing* the subset axis, because 20 gives a two-point window
and 30 gives three; `CFRNA_WADE_PANEL_V_MAX = 0.50` is the Cramér's V above which a
design-confounded contrast is refused, 0.5 being a conventional strong-association
mark; `CFRNA_WADE_NPERMS = 2000` and `CFRNA_WADE_TAIL_Q = 0.10` preserve v7's
settings. A port should not hardcode any of them, but the *reasoning* — that the
tail window must contain enough order statistics to be an average — generalizes and
belongs in the documentation.

**`wade_gene()` is required, not optional.** The downstream README states that the
per-gene diagnostic is what makes the statistic legible, and that a port unable to
reproduce that panel is missing `wade_gene()`'s output. It also records a related
plotting lesson worth carrying into any documentation: WADE's signed-area
quantities are not fold changes and do not plot like them — an early gallery figure
put `diff.mean` on the x axis, where it spans roughly ±9,000 in cfRNA units, and
collapsed every point onto a vertical line.

## See also

- [`porting-hazards.md`](porting-hazards.md) — where two implementations will
  silently disagree, with the measured quantile, RNG, BH and `tail.conc` findings.
- [`design-decisions.md`](design-decisions.md) — what is settled (raw counts in,
  normalization inside the package) and what is open.
- [`algorithm.md`](algorithm.md) — the language-agnostic mathematics.
- [`limits.md`](limits.md) — the combinatorial floor and the sample-size
  constraint, including the correction to the "below roughly 5%" claim.
- [`reference/R/wade.R`](../reference/R/wade.R) — the file itself. It is 317 lines
  and heavily commented; read it alongside this document rather than instead of it.
