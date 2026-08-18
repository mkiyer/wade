# Cross-language porting hazards

> **Note on citations.** This document cites `reference/docs/` and
> `reference/R/downstream/`, which were pruned once the port was verified.
> They are quoted here as the provenance of specific claims; the full record,
> including digests, is in [`../reference/PROVENANCE.md`](../reference/PROVENANCE.md).

The places where two correct-looking implementations of WADE will disagree. Each
entry states the hazard, why it bites, and how to test for it. The organizing
principle is that **none of these produces an error**: every one of them returns
plausible numbers of the right shape and sign, so the only thing standing between a
port and a silent wrong answer is a test that was written on purpose.

This document is written against [`reference/R/wade.R`](../reference/R/wade.R) as
the reference implementation, and it extends the parity section of
`reference/docs/WADE_REPO_SCOPE.md` with
measurements taken in this repository's pinned R sandbox and in a local Python
environment during the session that wrote this document. The function-by-function
conventions are in [`r-implementation.md`](r-implementation.md); this document
assumes them.

**Measured** below means a number produced by running code in that session. Package
versions for the Python side: NumPy 2.4.6, SciPy 1.17.1, statsmodels 0.14.6. R side:
R 4.6.1 with the pinned `renv` closure described in
[`reference/R/README.md`](../reference/R/README.md). **Design assertion** means a
statement about consequence or intent rather than a measurement.

## Hazard summary

| # | Hazard | Severity | Detectable by |
|---|---|---|---|
| 1 | Quantile type convention | every number changes | exact fixture comparison |
| 2 | RNG: Mersenne-Twister vs PCG64 | exact parity impossible | nothing — must be designed around |
| 3 | BH-FDR implementation | agrees today; pin it | exact fixture comparison |
| 4 | GPD moment fit: branches and floor | wrong p-values in the tail | per-branch unit tests |
| 5 | `tail.conc` guard | 5.4% of genes wrong in the R | comparison against a *fixed* R, not the R |
| 6 | Sample vs population variance | flips the GPD branch | unit test on `var` |
| 7 | Broadcast axis in the exceedance count | wrong p-values, no error | non-square fixture |
| 8 | Column-major vs row-major jitter fill | different jitter, plausible output | fixed-jitter fixture |
| 9 | Grid orientation (high→low) | measures the wrong tail | sign test on a planted subset |
| 10 | Summation order | last 2-3 significant figures | tolerance, not bitwise, assertions |
| 11 | `nprobs == 1` reshape | recycled scalar, right shape | one-sample-group test |

Hazards 1-5 are the ones `WADE_REPO_SCOPE.md` identifies; 6-11 came from reading the
file and running it.

---

## 1. Quantile convention

**The hazard.** Every WADE statistic is a function of the two quantile grids `Q1`
and `Q0`. R's `quantile()` offers nine types; `matrixStats::rowQuantiles` and
`stats::quantile` both default to **type 7**. NumPy's `quantile` defaults to
`method="linear"`. If a port inherits a different convention — because a library's
default differs, or because someone matches a published method that uses type 1 or
2 — **every number changes and nothing errors**.

**Why it bites.** The literature this method sits next to is not consistent. Type 7
is R's default and NumPy's, but quantile-based differential-expression work often
uses the inverse-empirical-CDF conventions (types 1 and 2), which do not
interpolate. Since WADE's grid is `min(n0, n1)` points spanning probability 1 to 0,
the interior probabilities frequently fall between order statistics of the *larger*
group, which is exactly where interpolating and non-interpolating conventions part
company.

**Verification: R type 7 and NumPy `linear` agree exactly.** Two vectors chosen to
stress the cases where conventions diverge — an odd length with ties, and an even
length with ties — evaluated on WADE's own grid shape, `seq(1, 0, length.out = 5)`
= (1, 0.75, 0.5, 0.25, 0).

`v_odd = c(0, 0, 0, 1, 2, 2, 5, 13, 100)`, n = 9:

| convention | values at p = 1, 0.75, 0.5, 0.25, 0 |
|---|---|
| R type 7 | 100, 5, 2, 0, 0 |
| R `matrixStats::rowQuantiles` | 100, 5, 2, 0, 0 |
| NumPy `linear` | 100, 5, 2, 0, 0 |
| R type 1 / NumPy `inverted_cdf` | 100, 5, 2, 0, 0 |
| R type 2 / NumPy `averaged_inverted_cdf` | 100, 5, 2, 0, 0 |
| R type 4 / NumPy `interpolated_inverted_cdf` | 100, **4.25**, **1.5**, 0, 0 |
| R type 6 / NumPy `weibull` | 100, **9**, 2, 0, 0 |

`v_even = c(0, 0, 3, 3, 3, 7, 11, 40)`, n = 8:

| convention | values at p = 1, 0.75, 0.5, 0.25, 0 |
|---|---|
| R type 7 | 40, 8, 3, 2.25, 0 |
| R `matrixStats::rowQuantiles` | 40, 8, 3, 2.25, 0 |
| NumPy `linear` | 40, 8, 3, 2.25, 0 |
| R type 1 / NumPy `inverted_cdf` | 40, **7**, 3, **0**, 0 |
| R type 2 / NumPy `averaged_inverted_cdf` | 40, **9**, 3, **1.5**, 0 |
| R type 4 / NumPy `interpolated_inverted_cdf` | 40, **7**, 3, **0**, 0 |
| R type 6 / NumPy `weibull` | 40, **10**, 3, **0.75**, 0 |

Compared elementwise, NumPy `linear` equals R type 7 **exactly** on both vectors —
maximum absolute difference 0.0, and `np.array_equal` returns `True`. So the
defaults do line up, which confirms what `WADE_REPO_SCOPE.md` asserts. Note also
that `matrixStats::rowQuantiles` and `stats::quantile(type = 7)` returned
bit-identical values, so the choice between them is not itself a hazard.

But look at the spread. On the even-length vector the seven conventions give five
different answers at p = 0.75 (7, 8, 9, 10, and 8 again) and four at p = 0.25.
Since `tail.mean` at a realistic `nprobs` is the mean of one or two of these
columns, a convention mismatch does not perturb the statistic — it replaces it.

**How to test.** Pin the convention explicitly in the port's source
(`method="linear"` written out, not defaulted) and assert it, rather than inheriting
it. The assertion should be a fixture comparison on both an even- and an
odd-length group with ties, because the odd-length vector above cannot distinguish
types 1, 2 and 7 at all — it agrees across all three. A test built only on odd
lengths would pass with type 1 substituted.

## 2. The RNG barrier

**The hazard.** `wade.R` consumes randomness in two places: the continuity jitter in
`wade_normalize()` (line 67, `stats::runif(g * n, 0, noise)` after
`set.seed(seed)`) and the label permutations in `wade()` (line 208, `sample(cond)`
after `set.seed(seed + 1L)`). R's generator is Mersenne-Twister with R's own
seeding and its own `sample()` algorithm; NumPy's `default_rng` is PCG64. **A
shared integer seed produces different numbers.** This is not a bug to work around
and not a version-pinning problem — the generators are different algorithms, and
`sample()`'s permutation algorithm differs from NumPy's `permutation` besides.

**Why it bites harder than it looks.** It removes the possibility of the obvious
parity test. Every WADE output is a function of the jitter (which perturbs the
normalized matrix) and of the permutation set (which defines the null), so two
implementations given the same counts and the same seed produce two different
answers, both correct. A porter who does not plan for this discovers it after
writing the port, when the natural test is impossible.

**This is a design requirement, not a caveat.** Exact cross-language parity is only
achievable if the **permutation index matrix** and the **jitter matrix** can be
supplied as inputs. `WADE_REPO_SCOPE.md` reaches the same conclusion and calls
option (a) — supply them — "worth designing for". This document states it more
strongly: it is the only option that yields an exact test, and the alternative is
strictly weaker.

Design assertion, with three specifics a port should honour:

1. **Both objects, not just permutations.** Supplying permutations alone leaves the
   jitter unmatched, and the jitter perturbs every normalized value. A port needs
   an optional `jitter` argument (a genes × samples array used in place of the
   internal draw) *and* an optional permutation matrix (`nperms` label vectors or
   index vectors).
2. **On the same code path production uses.** If the fixture path is a separate
   function, or a branch that skips the normal preprocessing, then the test
   validates the test harness rather than the library. The supplied objects should
   substitute for the internal draw at the point of the draw and change nothing
   else. This is also the constraint that decides where the native-kernel boundary
   falls; see [`design-decisions.md`](design-decisions.md).
3. **Generate the fixtures in R, consume them in both.** The R sandbox is the ground
   truth (it runs — see [`reference/R/README.md`](../reference/R/README.md)), so the
   fixture generator should emit the jitter matrix, the permutation matrix, the
   inputs, and every intermediate, and both implementations should read the same
   files.

**The weaker alternative, and why it is weaker.** Comparing *distributions* rather
than values — checking that the port's p-values are uniform under the null, that its
power curve matches, that its statistics agree in distribution — tests real
properties and should also be done. But it cannot localize a disagreement. A
distributional test passes when the port has the wrong quantile type in a place
where both types are unbiased, when the tail window is off by one, when the jitter
is drawn with the wrong fill order, and when the exceedance count broadcasts along
the wrong axis in a square fixture. Every one of those is a bug that a
fixed-permutation exact test catches immediately and a distributional test does not.
Use both; do not use only the second.

**Note on what the seed does and does not control.** `set.seed(seed + 1L)` for
permutations is an offset stream, so a port replicating the *structure* should
mirror that (jitter and permutations must not share a stream), even though the
values will differ. And `seed = NULL` is a supported mode in which
`wade_normalize()` draws from ambient state — measured, two consecutive
`seed = NULL` calls give different matrices, while two calls each preceded by the
same `set.seed()` give identical ones. A port needs an equivalent
"don't touch the global state" path.

## 3. BH-FDR

**The hazard.** `wade()` calls `stats::p.adjust(p, "BH")` twice (lines 234-235). The
Python ecosystem offers at least two implementations with different signatures and
defaults: `statsmodels.stats.multitest.multipletests(p, method="fdr_bh")` returns a
tuple whose second element is the adjusted p-values, and
`scipy.stats.false_discovery_control(p, method="bh")` returns the array directly.
Picking one without checking is how a port ends up with an adjusted-p column that is
subtly not BH.

**Verification: all three agree to floating point.** On the 15-value vector
`c(0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216, 0.222, 0.45, 0.6, 0.74, 0.99)`:

| implementation | first five adjusted values |
|---|---|
| R `p.adjust(., "BH")` | 0.014999999999999999, 0.059999999999999998, 0.126, 0.126, 0.126 |
| statsmodels `fdr_bh` | 0.015000000000000001, 0.060000000000000005, 0.12600000000000003, 0.12600000000000003, 0.12600000000000003 |
| SciPy `false_discovery_control` | 0.014999999999999999, 0.059999999999999998, 0.126, 0.126, 0.126 |

Compared to R position by position: **SciPy matched bitwise at all 15 of 15
positions** (maximum difference 0 ULP). **statsmodels matched at 9 of 15**, differing
at positions 1, 2, 3, 4, 5 and 13 by **exactly one unit in the last place** each —
maximum relative difference 2.2 × 10⁻¹⁶. Design assertion: **BH is not a parity risk
at the level that matters** — a one-ULP disagreement cannot change a thresholding
decision at any realistic q. Pin one implementation, assert against R fixtures with a
relative tolerance rather than bitwise, and move on. If a bitwise-exact match to R
happens to be wanted on this vector, SciPy provided it and statsmodels did not; that
is a single observation on 15 values, not a general claim about the two libraries.

**Two behaviours worth pinning that are *not* about the arithmetic.** Measured on
`c(0.9, 0.01, NA, 0.5, 0.01, 1.0)`, R's `p.adjust` returns
`1, 0.025, NA, 0.8333333333, 0.025, 1` — it **drops the `NA`, adjusts using n = 5,
and puts the `NA` back in position**. That matters because `wade()` produces all-`NA`
p-value columns when `nperms == 0`, and can produce `NA` p-values in no other case.
A port whose BH implementation propagates `NA` across the whole vector, or counts it
in `n`, will disagree on any frame containing one. Also note both ties received the
same adjusted value and the monotonicity enforcement (the cumulative-minimum step
from the largest p downwards) was applied — assert both.

## 4. The GPD moment fit

**The hazard.** `.gpd_tail_p()` ([`reference/R/wade.R`](../reference/R/wade.R) line
143) has two distribution branches, three bail-outs, a conditional-probability
rescaling and a floor. A port that implements only the `xi > 0` branch, or only the
happy path, will be correct on most genes and wrong on exactly the genes the
refinement exists for — the strongest hits.

**The floor is a deliberate honesty constraint.** `p_floor = 1 / (B · n_tail)`, and
the file's comment (lines 155-160) is explicit: with `B` permutations and `n_tail`
tail points that is the smallest defensible tail probability, and returning
machine-epsilon p-values would create false ties and mis-order strong hits.
`WADE_REPO_SCOPE.md` makes the same point in stronger terms, and it is worth
restating without ambiguity: **a port that "improves" the floor by returning smaller
p-values is a regression, not an upgrade.** The floor is not a numerical
convenience protecting against overflow; it is a statement about what 2,000
permutations can support. If the port later replaces method-of-moments with maximum
likelihood (see [`design-decisions.md`](design-decisions.md)), the floor must
survive that change.

**Branches a test suite must hit.** Six, each independently reachable:

1. **Positive shape** (`xi > 0`) — the GPD form `(1 + xi·y/sigma)^(−1/xi)`.
2. **Non-positive shape** (`xi <= 0`) — the `xi → 0` exponential limit
   `exp(−y/sigma)`, used instead of the GPD's hard upper bound at `−sigma/xi`.
3. **Floor binding** — the computed tail probability falls below `1/(B · n_tail)`.
4. **Fewer than ten exceedances** — `length(exc) < 10`, return the empirical p.
5. **Observed at or below the threshold** — `o <= thr`, return the empirical p.
6. **Degenerate tail** — `!is.finite(v) || v <= 0`, or `sigma` non-finite or
   non-positive, return the empirical p.

Measured reference values, from constructed nulls of length 2000 with `n_tail` = 250.
These are exact targets for a port's unit tests; the observed value in each case is
a stated multiple of the null maximum:

| null (seed) | observed | `thr` | `n_exc` | `m` | `v` | `xi` | `sigma` | p | branch |
|---|---|---|---|---|---|---|---|---|---|
| `rt(2000, 2.5)` (1) | 20.509099 | 1.572381 | 250 | 1.133161 | 2.525647 | +0.24580 | 0.854633 | 6.372558e-05 | 1 (GPD) |
| `rexp(2000, 1)` (2) | 10.088564 | 2.089624 | 250 | 0.957363 | 0.723512 | −0.13340 | 1.085074 | 7.859260e-05 | 2 (exp) |
| `rnorm(2000)` (3) | 6.334738 | 1.138272 | 250 | 0.496683 | 0.167755 | −0.23528 | 0.613544 | 2.621901e-05 | 2 (exp) |
| `runif(2000)` (4) | 0.999536 | 0.870413 | 250 | 0.061106 | 0.001337 | −0.89642 | 0.115882 | 4.101967e-02 | 2 (exp) |
| `rexp(2000, 1)` (5) | 167.333277 | 2.009647 | 250 | 0.989502 | 1.043189 | +0.03071 | 0.959113 | 2.000000e-06 | 3 (floor) |

Floor values: 2.0 × 10⁻⁶ at `B` = 2000, 4.0 × 10⁻⁶ at `B` = 1000.

Additional measured bail-outs: asking for the p-value of a null value *below* the
threshold returned the empirical 0.150425 (branch 5); a null of `rep(5, 400)`
followed by `rexp(1600, 5)` gives `thr` = 5 with **zero** exceedances (all top
values tied) and returned the empirical 0.00049975, exactly matching
`(1 + sum(null >= o))/(B + 1)` (branch 4 via the tie mechanism); and at `B` = 100
`n_tail` is capped to `floor(B/2)` = 50.

**Two easily-missed details in the arithmetic.**

*The exceedance inequality is strict.* `exc <- s[s > thr] - thr`, so ties at the
threshold are excluded and `length(exc)` is normally exactly `n_tail` but is smaller
whenever the null has ties there — and permutation nulls of discrete-ish statistics
do have ties. A port using `>=` gets a different `m`, a different `v`, and possibly
a different branch.

*The rescaling is by `n_tail / B`, not by `length(exc) / B`.* The GPD models the
conditional distribution given exceedance, so the unconditional tail probability
requires multiplying by P(exceed `thr`), and the file uses the *nominal* `n_tail/B`
= 0.125 at the defaults rather than the realized exceedance fraction. When ties
reduce `length(exc)` below `n_tail`, those two differ. Reproduce the file's choice.

**The refinement gate is a separate hazard.** `wade_perm_pvalues()` refines only
genes with `nexc < n_exc_min` (10) **and** only when `B >= 2 * n_tail`. Measured:
`B` = 250 and `B` = 499 do not qualify; `B` = 500, 1000 and 2000 do. So **at
`nperms < 500` no refinement ever happens** and the minimum p-value is
`1/(nperms + 1)`. A port that omits the `B` condition will refine at `nperms` = 200
and produce p-values the R cannot produce, on a code path no fixture at
`nperms` = 1000 or 2000 will ever exercise. Test it at `nperms` = 300 explicitly.

## 5. The `tail.conc` guard — the one place the port should deviate

**The hazard, and it is a defect in the R rather than a parity concern.** `wade()`
guards the ratio at

```r
tail.conc = ifelse(abs(obs$diff.mean * obs$nprobs) < 1e-8, NA_real_, obs$tail.conc)
```

Since `diff.mean * nprobs` reconstructs `sD = rowSums(D)` — the ratio's own
denominator — the guard is on the denominator's magnitude at a fixed absolute
threshold of 1e-8. At the cfRNA primary contrast's `nprobs` = 22 that means
`|diff.mean| < 5 × 10⁻¹⁰`.

**Measured on the cfRNA primary contrast** (quoted from
`reference/docs/WADE_REPO_SCOPE.md`, whose
figures `reference/R/downstream/README.md`
also carries): **120 of 2,219 genes (5.4%) return `|tail.conc| > 2`, the largest
being 149.** A "share of the signed area" of 149 is not a share of anything.
`WADE_REPO_SCOPE.md` additionally records that no *nominated* gene in any of the 18
cfRNA runs was affected — but describes that as luck rather than construction, since
any threshold on `tail.conc` sits directly on top of it.

**The cause is structural, not numerical.** `tail.conc = sum(D[1:k]) / sum(D)`. The
denominator is the total signed area, which vanishes whenever the lower quantiles'
differences cancel the upper ones — a gene up in part of the case distribution and
down in another part. That happens at `diff.mean` values that are small relative to
the gene's own variation but nowhere near 1e-10. The guard is roughly nine orders of
magnitude tighter than the cases that occur, so it fires almost never; what it does
catch is the exact-zero case, a constant gene's `0/0` (measured: a row set to a
single repeated value returns `tail.conc` = `NaN`, which the guard converts to `NA`).

**Independently reproduced on synthetic data**, to confirm the mechanism rather than
the cohort. 4,000 genes at 11 cases versus 11 controls (`nprobs` = 11, `k` = 2),
lognormal, with partial cancellation constructed deliberately — in each gene, three
random case samples multiplied up by a factor in [1.4, 3.0] and the remaining case
samples multiplied down by a factor in [0.30, 0.85]:

- 354 of 4,000 genes (8.85%) returned `|tail.conc| > 2`; largest 2,869.6.
- Of those 354, the 1e-8 guard NA'd out **zero**.
- Among the affected genes, `|diff.mean|` had minimum 0.0163 and median 11.21 —
  against a guard threshold of 9.1 × 10⁻¹⁰ at this `nprobs`.
- The ratio `sum|D| / |sum D|` among them had median 9.7 and maximum 6,661.5.

**A clamp to [0, 1] is the wrong fix**, and both source documents say so: values
slightly above 1 are legitimate, because the tail can carry more than the total
signed area when the bulk partially cancels. Measured on the same synthetic matrix,
**618 genes had `1 < |tail.conc| <= 1.5`** — a clamp would silently rewrite all of
them, destroying real information to hide a pathology elsewhere.

**The recommended fix: guard on the ratio's conditioning, not on the denominator's
absolute size.** Report `tail.conc` only when the total signed area is not
dominated by cancellation — that is, when

```
sum(|D|) / |sum(D)|  <=  F
```

for a stated factor `F`, and return a missing value otherwise. This has a property
the current guard lacks: because `|sum(D[1:k])| <= sum(|D|)`, the condition bounds
the reported statistic, `|tail.conc| <= F`. The guard and the guarantee are the same
number, which makes `F` a documentable promise rather than a tuning knob.

Measured factor sweep on the 4,000-gene synthetic matrix above:

| `F` | genes flagged | of the 354 pathological (abs value > 2) | of the 2,855 well-behaved (abs value <= 1) |
|---|---|---|---|
| 2 | 982 | 354 (100%) | 282 (9.9%) |
| 3 | 613 | 354 (100%) | 113 (4.0%) |
| 5 | 352 | 286 (81%) | 29 (1.0%) |
| 10 | 183 | 173 (49%) | 8 (0.3%) |
| 20 | 101 | 97 (27%) | 2 (0.1%)  |
| 50 | 33 | 33 (9%) | 0 (0.0%) |

At `F` = 3 the largest surviving `|tail.conc|` was exactly 2.00, consistent with the
bound. Design assertion: the choice of `F` is a documentation decision, not a
numerical one — it trades how many well-conditioned genes are withheld against the
ceiling on what gets reported — and this table is one synthetic construction, not a
recommendation of a value. What should not be left open is the *form*: guard on
conditioning, bound the output, and state the factor in the docs.

**The parity consequence, which is easy to get backwards.** If the port fixes this
and the R does not, then the port and the R **must** disagree on `tail.conc` for the
affected genes — that disagreement is the fix working. So a parity suite cannot
assert equality on `tail.conc` against unmodified `wade.R`. Three defensible
options: compare `tail.conc` only on genes where both implementations report a value
and the R's guard is not implicated; compare against a patched local copy of the R
guard used solely for testing; or assert the *ratio's components* (`sum(D[1:k])` and
`sum(D)` separately, which are exactly comparable) and test the guard's behaviour
against its own specification rather than against R. The third is cleanest, and it
generalizes: comparing the numerator and denominator separately localizes any
disagreement in a ratio.

For context, cfRNA guards at the display layer instead —
`wade_tail_conc_display(tail_conc, ceiling = 1.5)` in
`reference/R/downstream/wade_figures.R` line 104 returns `NA` when the value is
missing or `|value| > ceiling` and passes everything else through. That was a
deliberate choice to avoid invalidating every cached result for a descriptive
column, and it is not the right choice for a new package.

## 6. Sample versus population variance

**The hazard.** `.gpd_tail_p()` calls `stats::var(exc)`, which is the **sample**
variance with denominator `n − 1`. NumPy's `np.var` defaults to `ddof = 0`, the
population variance. Measured on `c(1, 2, 3, 4, 10)`: R's `var` = 12.5, population
variance = 10.0, ratio 1.25.

**Why it bites.** Both moment estimators are functions of `m²/v`:

```
xi    = 0.5 * (1 - m^2/v)
sigma = 0.5 * m * (1 + m^2/v)
```

An understated `v` inflates `m²/v`, which pushes `xi` **down** and `sigma` **up**.
Since the branch test is `xi <= 0`, a variance off by a factor of `n/(n−1)` can flip
a gene from the GPD branch to the exponential branch. At `n_tail` = 250 the factor
is 250/249 ≈ 1.004, so the shift is small — but the measured table in hazard 4 shows
`xi` values of +0.0307 and −0.1334 arising from ordinary nulls, so the branch
boundary is genuinely near the operating point and is crossed by noise. A port using
`ddof = 0` will disagree on branch assignment for some genes and will produce
different p-values for those.

**How to test.** Unit-test the two estimators directly against R on a fixed
exceedance vector, asserting `xi` and `sigma` to full precision rather than only the
returned p. `np.var(exc, ddof=1)`.

## 7. Broadcast axis in the exceedance count

**The hazard.** `wade_perm_pvalues()` computes `rowSums(perm >= obs)` where `perm`
is genes × permutations and `obs` is length `nrow`. R compares column-major, so
`obs` recycles **down** each column and gene `i`'s observed value meets gene `i`'s
null draws. Measured on a 3 × 3 null with rows (10, 11, 12), (20, 21, 22),
(30, 31, 32) and `obs` = (11.5, 21.5, 100): `rowSums(perm >= obs)` = 1, 1, 0, which
is the correct per-gene answer.

**Why it bites.** NumPy broadcasts trailing dimensions first. `perm >= obs` with
`perm` shaped `(g, B)` and `obs` shaped `(g,)` aligns `obs` against the
**permutation** axis, which is wrong — and it *only raises* when `g != B`. In a
square fixture it silently compares gene `i`'s null draw `j` against gene `j`'s
observed value. The correct form is `(perm >= obs[:, None]).sum(axis=1)`.

**How to test.** Make every parity fixture non-square in genes versus permutations,
so the wrong axis raises rather than lying. A fixture at 6 genes × 6 permutations is
worse than useless here — it is a test that passes on broken code.

## 8. Column-major versus row-major jitter fill

**The hazard.** `wade_normalize()` builds the jitter as
`matrix(stats::runif(g * n, 0, noise), g, n)`. R fills column-major: the first `g`
draws become column 1. NumPy's `reshape` is row-major by default: the first `n`
draws become row 1.

Measured, on the stream R produces from `set.seed(1L)` with `runif(6, 0, 0.01)` —
0.002655, 0.003721, 0.005729, 0.009082, 0.002017, 0.008984 — reshaped to 3 × 2:

| fill | row 1 |
|---|---|
| R `matrix(v, 3, 2)` | 0.002655, 0.009082 |
| NumPy `reshape(3, 2)` (C order) | 0.002655, 0.003721 |
| NumPy `reshape(3, 2, order='F')` | 0.002655, 0.009082 |

**Why it bites.** Hazard 2 already means the port cannot reproduce R's stream, so
this looks irrelevant — and for production runs it is. It becomes load-bearing the
moment a fixture supplies a jitter matrix (hazard 2's design requirement): the
fixture is a genes × samples array, and if the two implementations disagree about
how a flat vector maps into that shape, the "same" jitter is a transposed
scramble. Since the jitter perturbs values by about 1 part in 10⁻⁶ (see the
normalization measurements in [`r-implementation.md`](r-implementation.md)), the
resulting disagreement is small, structured, and easily mistaken for accumulated
floating-point error.

**How to test.** Serialize the fixture jitter as a 2-D array in a
format that carries its own shape and orientation, never as a flat vector plus
dimensions. Then assert on the array's `[0, 1]` and `[1, 0]` entries explicitly in
both languages before using it.

## 9. Grid orientation

**The hazard.** `q <- seq(1, 0, length.out = nprobs)` runs **high to low**, so
column 1 is the maximum and the first `k` columns are the upper tail. Every tail
quantity in the file indexes `1:k`. A port that builds an ascending grid — which is
the more natural default in most quantile APIs — and keeps the `1:k` slice computes
a **lower**-tail statistic under the name `tail.mean`.

**Why it bites.** It does not error, the shapes are identical, and on a null gene the
values are comparable in magnitude. It shows up as a sign inversion on exactly the
genes the method exists to find: a rare high-expressing subset would give a large
negative `tail.mean` instead of a large positive one, and — because the test is
one-sided upper-tail (hazard: see `wade_perm_pvalues()`) — those genes would return
`p.tail` near 1.

**How to test.** A directional fixture, not a numerical one: plant a gene elevated
in a small fraction of cases, and assert `tail.mean > diff.mean > 0` and
`p.tail` small. Then plant a gene *reduced* in a fraction of cases and assert
`tail.mean < 0` with `p.tail` near 1. Measured in the R on a 40-gene synthetic
matrix at 12 versus 12, one gene planted strongly down and one strongly up in cases:
the down gene gave `diff.mean` −18,545 / `p.diff` 1.0000 / `tail.mean` −41,645 /
`p.tail` 1.0000; the up gene gave `diff.mean` +108,355 / `p.diff` 0.0173 /
`tail.mean` +127,412 / `p.tail` 0.0276. Those two rows pin the orientation and the
one-sidedness together.

## 10. Summation order

**The hazard.** Floating-point addition is not associative, so `rowSums`,
`numpy.sum` (pairwise summation), a hand-written loop, and a Rust `iter().sum()` can
give different last bits on the same values. `numpy.sum` in particular uses pairwise
summation, which is *more* accurate than sequential addition and therefore
**guaranteed to differ** from R's `rowSums` on some inputs.

Is it material? Measured, and the honest answer is: **not for the statistic, but yes
for how the test suite is written.** Over 2,000 random rows at 16 cases versus 15
controls, comparing expressions for `diff.mean` against `wade_stats()`'s
`rowSums(D)/nprobs`:

| expression | rows differing bitwise (of 2,000) | worst relative difference |
|---|---|---|
| `sum(D) / nprobs` | 0 | — |
| `sum(rev(D)) / nprobs` | 1,284 | 3.5 × 10⁻¹⁴ |
| `cumsum(rev(D))[nprobs] / nprobs` | 1,284 | 3.5 × 10⁻¹⁴ |

So merely *reversing the summation order* of the same `nprobs` values changes the
result in about two thirds of genes, at a relative magnitude of 10⁻¹⁴. That is
irrelevant to any scientific conclusion and fatal to a bitwise assertion.

Two concrete consequences:

*`wade_gene()`'s endpoint identity is a tolerance assertion.* The cumulative curve
is built so that `cum[nprobs] == diff.mean`, and that is the identity the function
exists to satisfy. It holds to floating point, not bitwise, because the curve
accumulates the reversed grid. [`reference/R/README.md`](../reference/R/README.md)
reports the smoke test's endpoint as equalling `diff.mean` "to floating-point
equality", consistent with this. A port asserting exact equality will fail on the
majority of genes for no reason.

*Parity tolerances should be relative and stated.* Design assertion: assert relative
agreement at something like 1e-12 for grid quantities and derived means, and reserve
exact equality for integer-valued things — `nprobs`, `k`, exceedance counts, ranks,
and the boolean of which branch `.gpd_tail_p()` took. A disagreement that exceeds
1e-12 relative is a real bug; one at 1e-14 is summation order.

## 11. The `nprobs == 1` reshape

**The hazard.** `wade_stats()` restores the dimension attribute that
`matrixStats::rowQuantiles` drops:

```r
if (is.null(dim(Q1))) { Q1 <- matrix(Q1, nrow = 1); Q0 <- matrix(Q0, nrow = 1) }
```

`rowQuantiles` drops dimensions when the result has a single row *or* a single
column, and the guard assumes the former. When the smaller group has exactly one
sample, `nprobs` = 1, the result is one value per gene, and the guard reshapes a
length-`nrow` vector into a **1 × nrow** matrix — transposing genes into
probabilities.

Measured, on a 5-gene matrix with `cond = c(1L, 0L, 0L, 0L)`: `nprobs` = 1, `k` = 1,
`dim(Q1)` = 1 × 5, and `length(diff.mean)` = 1 for a 5-row input. Carried into the
driver, `wade()` returns a **5-row tibble in which `diff.mean` is the same number in
all five rows** (−67368.6748), because R recycles a length-1 column to the frame's
height. No error, no warning, correct shape, wrong answer for four of five genes.
The identical guard is present in `.wade_null_stats()` (line 124), so the null is
affected too.

**Why it bites in a port.** A port will not reproduce this bug — NumPy's
`np.quantile(X, q, axis=1)` returns a well-shaped array and there is nothing to
guard — so the port and the R will disagree, and the port will be right. That makes
it a *parity-suite* hazard rather than a code hazard: a fixture with a one-sample
group will show a difference that must not be "fixed" toward the R.

**How to test.** Assert that the port either refuses `min(n0, n1) == 1` or returns
per-gene values of length `nrow`; and add a regression note that the R disagrees
here by construction. Design assertion: refusing is defensible — a one-sample group
has no quantile function worth comparing — and refusing loudly is better than the
R's silence.

---

## What a parity suite should assert, and in what order

The organizing principle: **intermediate quantities localize a disagreement; endpoint
quantities only detect one.** If a port's `padj.tail` differs from R's, the cause
could be the quantile type, the tail window, the grid orientation, the exceedance
broadcast, the GPD branch, the floor, or BH — seven candidates and no information
about which. If instead the suite has already asserted that `Q1`, `Q0` and `D`
agree, six of those candidates are eliminated before the endpoint is ever compared.
So build the suite inside-out, and stop at the first failing layer.

**Layer 0 — fixtures, generated in R.** From the R sandbox, emit: the count matrix,
the normalizer (in both supported forms — vector and matrix), the condition vector,
the **jitter matrix**, the **permutation matrix**, and every intermediate below.
Make the fixture **non-square in genes versus samples and in genes versus
permutations** (hazard 7), include **ties and zeros** in the counts (hazards 1, 4),
and include both an **even and an odd** group size (hazard 1). Include a
tiny fully-hand-checkable case: a 6-gene × 7-sample fixture at 3 cases versus 4
controls gives `nprobs` = 3, `q` = (1, 0.5, 0) and `k` = 1, which is small enough
to verify with a calculator.

**Layer 1 — the deterministic scalars.** `nprobs`, `k`, `n1`, `n0`, and the grid `q`
itself. Assert exactly; these are integer or exactly-representable. This layer
catches the grid-size and tail-window arithmetic, and asserting `q[0] == 1` and
`q[-1] == 0` catches orientation (hazard 9) before any statistic is computed.

**Layer 2 — normalization, against a supplied jitter.** Assert the normalized matrix
elementwise at relative tolerance. This is where the per-cell denominator gets
tested, and it is worth asserting the two structural facts from
[`r-implementation.md`](r-implementation.md) as *properties* rather than only against
fixtures, because they are cheap and they pin the algebra: column sums are **not**
exactly 1e6 (measured: 1,000,161.99 on an 800 × 6 Poisson example where the
unjittered form gives exactly 1,000,000.00), and a gene that constitutes the entire
library normalizes to exactly 1e6. A port that reproduces the naive
`lib_sizes`-only denominator passes a rank-order check and fails this layer by about
1.7 × 10⁻⁶ relative — which is why the tolerance here should be tight (1e-12), not
generous.

**Layer 3 — the quantile grids `Q1`, `Q0`, `D`.** Elementwise, tight tolerance. This
is the highest-value layer in the suite: every statistic downstream is a reduction
of `D`, so if `D` agrees, hazards 1, 8, 9 and 11 are all excluded at once, and any
later disagreement is in the reductions or the p-values. Assert on the **full
grids**, not on summaries of them.

**Layer 4 — the per-gene reductions.** `diff.mean`, `w1`, `tail.mean`, `fc`,
`cond1.mean`, `cond0.mean`, `tot.mean`, `diff.frac`. Relative tolerance ~1e-12
(hazard 10). Assert the two exact identities as well —
`tot.mean == cond1.mean + cond0.mean` and `fc == cond1.mean / cond0.mean` — since
they are cheap and they catch a whole class of transcription error. Handle
`tail.conc` separately per hazard 5: assert its numerator and denominator, not the
ratio.

**Layer 5 — the null, against a supplied permutation matrix.** Assert the full
`g × nperms` null matrices for both axes elementwise. This is the layer that
validates a vectorized or native permutation kernel against the R's serial loop, and
it is the only place a kernel bug is cleanly separable from a p-value bug. If the
port's kernel is Rust, this is the layer that tests it.

**Layer 6 — the p-values.** Empirical first: assert the exceedance **counts**
(integers, exact) before the p-values, since that separates the broadcast question
(hazard 7) from the `(1 + nexc)/(B + 1)` arithmetic. Then the GPD refinement, as
unit tests on `.gpd_tail_p()`'s six branches with the measured table in hazard 4 as
targets, asserting `xi`, `sigma` and the branch taken alongside the returned p.
Then which genes were selected for refinement (`nexc < 10 & B >= 500`), including a
`nperms` = 300 case where the answer must be "none".

**Layer 7 — BH and the frame.** Adjusted p-values at loose tolerance (hazard 3),
plus the `NA` handling. Then the frame's shape and column names, plus the
`gene_names = NULL` behaviour — R drops the column silently, and a port that
synthesizes identifiers instead should assert that difference deliberately rather
than discover it.

**Layer 8 — the rank scores, if they ship.** `score`, `tail.score`, `rank`,
`tail.rank`. Assert the empirical-CDF convention directly (`F(t) = #{x ≤ t}/n`,
ties sharing a value, `F(max) == 1`) before the scores, and `dense_rank`'s tie and
`NA` behaviour (measured: `dense_rank(desc(c(0.5, 0.5, 0.2, NA, 0.9)))` = 2, 2, 3,
`NA`, 1) before the ranks. Whether these ship at all is open — see
[`design-decisions.md`](design-decisions.md).

**Layer 9 — distributional and behavioural checks, which are not parity.** The three
simulations in
[`reference/R/validation_sims_v7.R`](../reference/R/validation_sims_v7.R) are the
natural content here, and their measured outputs in
[`reference/R/README.md`](../reference/R/README.md) are the diff targets: null
calibration KS *p* = 0.6554 on the bulk axis and 0.2327 on the subset axis, realized
type-I error at nominal 0.05 of 0.0512 and 0.0488, discrimination medians of
`tail.mean` 27,537 against `diff.mean` 3,046 for planted subset genes (ratio 8.9)
versus 4,064 against 1,348 for bulk shifts (ratio 3.3), and the power sweep
transitioning 0 → 1 between 20% and 35% of cases. These are measurements of the
R sandbox at 77 cases versus 18 controls with 1,000 permutations; a port will not
match them to the digit (different RNG — hazard 2) and should match them
qualitatively. Note the KS test warns about ties because permutation p-values are
discrete; that warning is expected in any language.

**Two things the suite should refuse to do.** Do not assert bitwise equality
anywhere except on integers, ranks and branch flags (hazard 10). And do not use the
R as the oracle for `tail.conc` (hazard 5) or for a one-sample group (hazard 11) —
in both cases a correct port disagrees with `wade.R`, and a suite that enforces
agreement enforces the bug.
