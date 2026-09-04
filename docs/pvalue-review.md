# WADE: the method, and the open problem in its inference

**A request for statistical review.** Current as of 2026-09-04.

WADE is a two-stage differential-distribution test for RNA-seq counts. Its
statistics are fine; its **p-values are the problem**. Both stages need
resolution near 1e-6 from 2,000 permutations, and the generalized-Pareto
refinement that is supposed to provide it is wrong by three orders of
magnitude in one regime and anti-conservative by two in another — with which
one you get set by a property of your data rather than by anything we choose.

Stage 1 now has a good answer. **Stage 2 does not**, and that is what we would
most like help with.

> **Provenance.** `docs/scaling.md` §4.5–§4.10 is the canonical record inside
> the project; this file is the self-contained version for a reader without
> the code. Where they disagree, `scaling.md` is right and this is stale.
> Every number here is measured against brute-force permutation — never
> simulated from theory — and each table states its permutation count.

---

## Contents

1. [What WADE computes](#1-what-wade-computes)
2. [How its p-values are made](#2-how-its-p-values-are-made)
3. [Why the tail matters](#3-why-the-tail-matters)
4. [The candidate estimators](#4-the-candidate-estimators)
5. [What we measured](#5-what-we-measured)
6. [Diagnosis](#6-diagnosis)
7. [Where things stand](#7-where-things-stand)
8. [The open questions](#8-the-open-questions)
9. [Appendix A — the design floor](#appendix-a--the-design-floor)
10. [Appendix B — suggestions already tested](#appendix-b--suggestions-already-tested)
11. [Appendix C — reproducing any of this](#appendix-c--reproducing-any-of-this)

---

## 1. What WADE computes

WADE separates a shift affecting every sample from a change confined to a
subset of one group — two situations that produce the same mean difference and
that a first-moment test cannot tell apart. The motivating case is a gene
elevated in 15% of tumours, which an ordinary DE method sees as a weak average
effect.

### Setup

Raw counts in `n1` cases and `n0` controls, normalised by a per-gene factor
and a per-sample library size, with a small continuity jitter applied at count
precision to break the ties that dominate sparse data. Both groups' quantile
functions are read on a shared grid of `m = min(n1, n0, 2000)` probabilities
(type-7); call them `Q1(p)`, `Q0(p)`.

### Stage 1 — is there a difference?

```
mean_shift  =  (1/m) · Σ[j=1..m] ( Q1(p_j) − Q0(p_j) )
```

The signed area between the two quantile functions. Because
`∫₀¹ Q(p) dp = E[X]` is an identity, that area **is** the difference of group
means; the grid is a quadrature of it, exact when `n1 = n0` and an
interpolation of the larger group otherwise. Which of those two things you
compute turns out to matter (§5.4).

### Stage 2 — what kind of difference is it?

With `R(p) = log2 Q1(p) − log2 Q0(p)` (plus a one-count pseudocount), a pure
global fold change makes `R` flat, so its running total grows proportionally.
Stage 2 measures departure from proportionality with a discrete Brownian
bridge:

```
S_k  =  Σ[i≤k] R_i
B_k  =  S_k − (k/m)·S_m          # identically 0 under a global shift
T    =  max[k<m] ( B_k − μ_k ) / σ_k
```

The scan runs over every window width rather than asking the user to choose
one, and the permutation null prices in the multiplicity of having looked.
Two properties are load-bearing later:

* **The null is the fitted global shift, not no-difference.** For counts this
  is realised by binomial thinning of the raw reads under a fitted fold
  change, applied to the observed statistic and the null alike.
* **`μ_k` and `σ_k` are estimated from the same permutations the tail is read
  from.** So `T` is *not* a fixed function of a label assignment: the observed
  value moves 3% between permutation samples at `B = 2,000`, and 0.09% at
  500,000. Anything that samples a *conditioned* distribution has to freeze
  them first, or it estimates the tail of a different statistic.

---

## 2. How its p-values are made

Draw `B = 2,000` permutations of the case/control label vector — **one shuffle
serves every gene**, preserving the joint null's gene–gene correlation
structure. Count exceedances, take the empirical p-value, refine into the tail
where the empirical value has run out, then Benjamini–Hochberg within each
stage separately. No combined p-value, no categorical label.

The refinement follows Knijnenburg et al. (2009): take the top
`n_tail = 250` null values, subtract the threshold to get exceedances `y`, fit
a generalized Pareto distribution by method of moments, read the p-value off
the fit.

```
r  = mean(y)² / var(y)
ξ̂  = (1 − r)/2
σ̂  = mean(y)·(1 + r)/2

tail = (1 + ξ̂·y/σ̂)^(−1/ξ̂)      if ξ̂ > 0
tail = exp(−y/σ̂)               if ξ̂ ≤ 0    ← the substitution

p    = max( (n_tail/B)·tail , 1/(B·n_tail) )
```

**That `ξ̂ ≤ 0` branch is the whole subject of this document.** A GPD with
negative shape has a hard upper bound at `−σ̂/ξ̂`, past which the survival
function is zero, and an observation beyond it would collapse a strong
statistic to a machine-epsilon p-value. The exponential limit was substituted
to avoid that.

---

## 3. Why the tail matters

A typical run is 20,000–30,000 genes, so BH at α = 0.05 decides the top genes
near `0.05/20000 ≈ 2.5e-6`. The empirical p-value from 2,000 permutations
floors at `1/(B+1) = 5e-4` — more than two orders of magnitude short. On one
real 31k × 83k cohort, **18.8% of genes sat at that floor**.

Refinement is therefore not a nicety. Genes whose true p-value lies between
1e-5 and 1e-7 are simultaneously *the genes BH is ruling on* and the genes no
permutation count can resolve. Whatever the refinement does to them is what
the method's power is.

Raising `B` is not a way out: the floor moves as `1/B`, so an order of
magnitude of resolution costs an order of magnitude of compute, and the
permutation loop already dominates runtime.

---

## 4. The candidate estimators

| | what it is | cost | needs |
|---|---|---|---|
| **Empirical** | `(1 + #{null ≥ obs})/(B+1)` | free | nothing |
| **GPD moments** | the shipped refinement above | free | a tail model |
| **GPD ML** | identical but `(ξ̂, σ̂)` by maximum likelihood, negative-shape survival used as-is | free | a tail model |
| **Double saddlepoint** | the exact permutation tail of a *subset sum*, computed | `O(n)` per gene, **no permutations** | the statistic to be a monotone function of a subset sum |
| **Multilevel splitting** | fgsea's adaptive scheme, adapted from gene-set sampling to label permutation | ~1 s per gene (stage 2) | per-gene sampling; frozen moments for stage 2 |

**The saddlepoint.** For stage 1, with `A` the case-subset sum and `V` the
total, `T = A(1/n1 + 1/n0) − V/n0`, so `P(T ≥ t) = P(A ≥ a)`. Sampling `n1` of
`n` without replacement is handled by conditioning: with independent Bernoulli
selectors, `K(s,t) = Σ log(1 + exp(s·v_i + t))`, and Skovgaard's
double-saddlepoint formula gives `P(A ≥ a | ΣZ = n1)`. **Group balance is not
a requirement** — that identity holds for any `n1, n0`. What is required is
that the statistic *be* a subset sum, which the mean difference is and the
grid quadrature is not (it is an L-statistic). Stage 2 is not one at any
geometry.

**Multilevel splitting.** Hold ~1,000 label vectors; each round discard the
lower half by statistic and duplicate the upper half, recording the discarded
median as the level, then move every survivor by swapping one case label with
one control label, accepting only while the statistic stays above the level.
`P ≈ (1/2)^k` after k rounds, so 1e-100 is ~332 rounds rather than 1e100
draws. The estimator is not `2^−k`: each level is a *sample* median whose
exceedance probability is a Beta order statistic, so rounds accumulate
`E[log U] = ψ(a) − ψ(a+b)`, which is what makes `log p` approximately
unbiased. Two costs — it is **per gene**, abandoning the shared permutation
ensemble, and for stage 2 it cannot be applied until `μ_k, σ_k` are frozen.

---

## 5. What we measured

Ground truth is brute force: **1e8 streamed permutations per design**, four
geometries with `n = 80` held fixed (40v40, 60v20, 70v10, 20v60) so a
difference between them is a difference in *balance* rather than in cost. 64
genes per design in 8 effect rungs; a gene counts as resolved at ≥ 10
exceedances. Reported throughout: the ratio `p̂ / p_true`. Above 1 is
conservative, below 1 is anti-conservative.

### 5.1 Stage 1, 40 v 40

| true p | n | empirical | GPD moments | GPD ML | multilevel | saddlepoint |
|---|---|---|---|---|---|---|
| 1e-3 – 1e-2 | 4 | 1.33 | 3.22 | 1.03 | **0.98** | **1.00** |
| 1e-4 – 1e-3 | 9 | 3.29 | 12.4 | 1.03 | **0.99** | **0.99** |
| 1e-5 – 1e-4 | 10 | 21.5 | **76.4** | **0.09** | **1.01** | **0.99** |
| 1e-6 – 1e-5 | 7 | 114 | **304** | 0.83 | **1.02** | **1.02** |
| 1e-7 – 1e-6 | 8 | 2028 | **2137** | 8.12 | **0.86** | **0.91** |

### 5.2 Safety — the worst single gene anywhere

| design | empirical | GPD moments | GPD ML | multilevel | saddlepoint |
|---|---|---|---|---|---|
| 40 v 40 | 0.83 | 1.000 | **0.022** | 0.54 | 0.61 |
| 60 v 20 | 0.63 | 1.000 | **0.007** | 0.65 | — |
| 70 v 10 | 0.57 | 1.000 | **0.018** | 0.77 | — |
| 20 v 60 | 0.61 | 1.000 | **0.031** | 0.69 | — |
| **40 v 40, stage 2** | 0.93 | **0.019** | **0.019** | — | n/a |

**GPD-ML has the best median accuracy of any free method and is the only one
that is unsafe** — p-values up to 140× too small. For a method feeding FDR
control that is disqualifying.

### 5.3 The finding that explains all of it

The shipped estimator's behaviour is governed by one latent quantity: **how
often the fitted shape comes out negative, and so how often the exponential
substitution fires.** All eight design × stage cells, sorted by that rate:

| design | stage | ξ̂ < 0 | substitution fires | worst ratio | median range |
|---|---|---|---|---|---|
| 40 v 40 | 1 | 100% | 100% | 1.000 safe | 3.2 – 2137 |
| 60 v 20 | 1 | 100% | 100% | 1.000 safe | 3.9 – 4906 |
| 70 v 10 | 1 | 100% | 100% | 1.000 safe | 3.5 – 2947 |
| 20 v 60 | 1 | 100% | 100% | 1.000 safe | 3.6 – 1909 |
| 20 v 60 | 2 | 100% | 100% | 1.000 safe | 3.0 – 220 |
| 40 v 40 | 2 | 67% | 50% | **0.019 unsafe** | 0.80 – 14.4 |
| 60 v 20 | 2 | 20% | 20% | **0.345 unsafe** | 1.03 – 299 |
| 70 v 10 | 2 | — | — | 0.891 | too few deep genes |

Monotone in both directions at once. **Where the substitution fires on
everything, the estimator is never anti-conservative and is conservative by
three orders of magnitude. Where it fires rarely, it becomes accurate and
becomes unsafe.** We are not choosing between a safe estimator and an accurate
one — we ship one estimator whose position on that trade-off is set by a
latent property of each dataset.

Stage 2's statistic is a maximum over widths, which has a heavier null tail
than a mean difference, so the GPD form actually fits there and the
substitution fires less. That is why stage 2 is *more accurate* and *less
safe*.

### 5.4 Stage 1 has an answer: the saddlepoint

Against 1e8-permutation brute force, on the exact mean difference:

| data | design | median, 1e-2 → 1e-7 | worst |
|---|---|---|---|
| log-normal | 40 v 40 | 1.00 · 0.99 · 0.99 · 1.02 · 0.91 | 0.61 |
| log-normal | 60 v 20 | 1.00 · 1.00 · 1.00 · 0.95 · 1.10 | 0.83 |
| log-normal | 70 v 10 | 1.01 · 1.00 · 1.01 · 0.99 · 1.01 | 0.63 |
| log-normal | 20 v 60 | 0.99 · 0.99 · 1.00 · 1.02 · 1.15 | 0.75 |
| **NB counts**, through the real normalization | 40 v 40 | 1.00 · 1.00 · 1.02 · 0.99 · 0.84 | 0.84 |
| **NB counts, 20% zeros, median count 3** | 40 v 40 | 1.01 · 1.06 · 0.94 · 0.93 · 0.76 | 0.67 |
| **NB counts** | **300 v 300** | 1.00 · 1.00 · — · 1.03 · 1.14 | 0.89 |

Geometry-blind, better at larger `n` as an asymptotic method should be, and it
holds on sparse counts. Null p-values are uniform to Monte Carlo error at
**52% zeros**. There is no floor above `1/C(n, n1)` — 9.3e-24 at 40 v 40,
against the empirical floor of 5.0e-4 it replaces.

**Cost.** For 20,000 genes: 0.7 min at 40 v 40, 4.6 at 300 v 300, 17.6 at
1,000 v 1,000, 49.8 at 3,000 v 3,000 — about **60× the stage-1 permutation
loop it replaces**, at every size, since both are `O(n)` per gene. Fine below
about a thousand per group; the wrong tool at tens of thousands, where the
combinatorial floor is astronomically small and refinement matters least
anyway.

### 5.5 A twist worth reporting: the quadrature is the better *statistic*

We expected the exact mean difference to dominate, since the quadrature is
biased off balance. With *both* p-values brute-forced — no floor, no
approximation in either arm — that is wrong:

| design | log-normal, 1.6× | NB counts, 2.0× | winner |
|---|---|---|---|
| 100 v 10 | 3.0e-2 vs 7.3e-2 | 4.0e-6 vs 1.7e-5 | **quadrature, 2–4×** |
| 70 v 10 | 2.5e-2 vs 6.9e-2 | — | **quadrature, 2.8×** |
| 55 v 55 | identical | identical | tied, as the identity requires |
| 20 v 60 | 6.6e-3 vs 3.4e-3 | 1.6e-4 vs 1.2e-4 | **mean difference, 1.3–1.9×** |

The mechanism is that the grid reads *both* groups at `m = min(n1, n0)`
points, so the larger group is interpolated. Interpolating the **case** group
is effectively trimming, and on right-skewed expression a trimmed summary
beats a mean that a few large values dominate. Interpolating the **control**
group throws away precision in the reference. The quadrature's bias is also a
robustification.

**End to end the saddlepoint still wins**, because the p-value dominates the
statistic. 2,000 genes, BH 0.05:

| planted | 100 v 10: grid+GPD | saddlepoint | 55 v 55: grid+GPD | saddlepoint |
|---|---|---|---|---|
| null | 0.1% | 0.4% | 0.1% | 0.6% |
| 1.3× | 4.0% | **12.0%** | 50.0% | **78.0%** |
| 1.5× | 34.0% | **42.0%** | 100% | 100% |
| 1.8× | 72.0% | **80.0%** | 100% | 100% |
| 2.2× | **96.0%** | 94.0% | 100% | 100% |

A 2–4× better statistic loses because its p-value floors at 2e-6: the grid's
smallest reported p was 1.2e-5 where the saddlepoint reached 2.6e-12. **The
refinement, not the statistic, is what limits stage 1.**

---

## 6. Diagnosis

The permutation distribution of these statistics has a **finite right
endpoint**. There are only `C(n, n1)` label assignments and the statistic is
maximised by the one placing the largest values in the case group. A bounded
distribution sits in the reversed-Weibull domain of attraction, so the correct
limiting tail has negative shape — and that is what we fit, for 100% of
deep-tail stage-1 genes across all four geometries, median ξ̂ −0.20 to −0.29.

**The negative shape is the right answer. Both implementations mishandle it,
in opposite directions.**

| | what it assumes | how it fails |
|---|---|---|
| Substituting `exp(−y/σ̂)` | the endpoint is at infinity | assigns mass far beyond the true endpoint; every observation there is inflated — 76× at 1e-5, 2137× at 1e-7. Safe, and it costs real discoveries |
| Keeping the negative-shape GPD | a point estimate of the endpoint is exact | support ends at `−σ̂/ξ̂`; past it survival is 0 and the p-value falls to the floor. The endpoint is estimated from 250 points and is routinely too small — ratios of 0.007 |

**The whole problem is estimating the right endpoint of a bounded distribution
from its 250 largest order statistics, and then behaving sensibly when the
observation lands near or past it.** The two shipped behaviours are the two
degenerate answers.

For stage 1 that problem is now bypassed rather than solved — the saddlepoint
does not model a tail at all. **For stage 2 it is unsolved**, and stage 2 has
no analytic endpoint to exploit either (see question 1).

---

## 7. Where things stand

| | stage 1 | stage 2 |
|---|---|---|
| statistic | difference of group means (a subset sum at every geometry) | max over widths of a studentised bridge |
| shipped p-value | permutation + GPD, 3–4,906× conservative | permutation + GPD, 4–80× off *and* anti-conservative to 0.019 |
| available answer | **double saddlepoint** — 0.99–1.00 median, no floor, no sampling | **none** |
| status | implemented as `stage1="saddlepoint"`, opt-in, validated | **open** |

Stage 2 is the harder and more important half: it is WADE's distinctive stage,
it is 80% of the runtime, and every route that works for stage 1 is closed to
it. Multilevel splitting works (0.94 median, worst 2.5×, over the 12 genes
below 1e-4) but costs ~1 s per gene and requires freezing the moments first.

---

## 8. The open questions

Ordered by what would help most.

### 1. A tail estimator for a bounded null that is never anti-conservative

The central question. We need `P(T ≥ t)` near 1e-6 to 1e-8 for a statistic
whose permutation null has a finite endpoint, from `B = 2,000` draws, at
`O(B)` cost — and we need it to **never** return a p-value that is too small,
because it feeds FDR control. Being conservative by a known, bounded factor is
an acceptable answer; being conservative by three orders of magnitude is not.

Specifically: is there a standard treatment for tail probabilities near an
*estimated* finite endpoint — one that degrades gracefully as the observation
approaches and passes it, rather than saturating at zero?

### 2. Stage 2's endpoint

For stage 1 the maximum of the statistic is available in closed form (the
largest `n1` values as cases), and a GPD with the endpoint **fixed** there is
never anti-conservative and only 1.5–35× conservative (Appendix B). Stage 2's
statistic is `max_k (B_k − μ_k)/σ_k` over a cumulative bridge, and we do not
know its maximum. **Is there a computable upper bound — even a loose one?** A
greedy or relaxation bound would make the fixed-endpoint approach available
for stage 2 and might close this entirely.

### 3. Should stage 2's studentisation be frozen?

`μ_k, σ_k` are currently estimated from the same permutations the tail is read
from, which makes `T` not a fixed function of the labels. We believe they
should be frozen from a separate uniform sample — bulk quantities converging
as `1/√B`, kept apart from a rare-event quantity — which would also make
multilevel splitting applicable. Is that the right instinct, and does it cost
anything we have not thought of?

### 4. A relative-error tail for an L-statistic

The best stage 1 available would be the *quadrature's* statistic with an
*exact* p-value: §5.5 shows it is 2–4× more powerful than the mean difference
when `n1 > n0`, and the saddlepoint cannot touch it because it is a linear
combination of order statistics rather than a subset sum. Is there a
saddlepoint-like approximation for the permutation distribution of an
L-statistic?

### 5. Threshold selection

We use the top 250 of 2,000 (12.5%), inherited from the original
implementation, with no threshold-stability analysis. Smaller thresholds make
things *worse*, not better (Appendix B), which surprised us. What diagnostic
would you want to see?

### 6. Multiplicity under per-gene sampling

Multilevel splitting is per-gene, so it abandons the shared permutation
ensemble. We have been told BH needs only marginal validity and PRDS, so this
is fine. Is that right in this setting, where genes are strongly correlated
through library-size normalisation?

---

## Appendix A — the design floor

Worth stating because we had it wrong in our own documentation until recently,
and because it bounds what any estimator can do.

If exactly `k` case samples carry the signal, the probability that a
relabelling puts all of them in the case group is `C(n1,k)/C(n1+n0,k)`. We
described this as a hard lower bound on the achievable p-value. **It is not.**

300 v 300, `B = 40,000` so every p-value is a raw permutation count:

| planted k | floor | p/floor q10 | median | q90 | fell below |
|---|---|---|---|---|---|
| 3 of 300 | 1.24e-1 | 0.48 | 1.96 | 3.23 | **21%** |
| 6 of 300 | 1.52e-2 | 0.26 | 1.10 | 1.71 | **42%** |

It is exact only in the noise-free limit, where every relabelling placing all
`k` affected samples in cases *ties* the observed value. With noise those
relabellings scatter above and below it, so roughly half fall below — and in
count data `k` is not even well defined, since dispersion makes some
unaffected samples look extreme. It survives as a good estimate of the *order
of magnitude* a design can support. It cannot be used to clamp a p-value,
which was the first fix we tried.

**Why it matters practically**, at a geometry we are actually asked about —
100 cases against 10 controls:

| affected cases | floor |
|---|---|
| 5 of 100 | 0.62 |
| 10 of 100 | 0.37 |
| 25 of 100 | 0.067 |
| 50 of 100 | 1.6e-3 |
| all 100 (a global shift) | 2.1e-14 |

Stage 2 cannot work at that geometry unless roughly half the cases share the
change. Stage 1 is perfectly usable. No estimator changes this.

---

## Appendix B — suggestions already tested

An earlier round of review produced eight recommendations. Each checkable one
was checked against the existing ground truth; recording the outcomes so the
same ground is not covered twice.

**Confirmed.**

* *Redefine stage 1 as the plain mean difference so the saddlepoint applies
  universally.* Right, and the most valuable suggestion received — it is §5.4.
* *A GPD with the endpoint fixed at the analytic maximum.* Right: never
  anti-conservative at 40v40 and 60v20 (worst 1.000), conservative by 1.5–35×
  against the exponential substitution's 3–2137×, and a one-parameter
  closed-form fit. Dominated by the saddlepoint where both apply; **out of
  reach for stage 2, which has no analytic endpoint** — hence question 2.
* *BH needs marginal validity and PRDS, not a shared ensemble.* Accepted,
  though question 6 revisits it under strong correlation.

**Refuted by measurement.**

* *`n_tail = 250` drags bulk data into the fit and biases ξ̂ negative; use
  20–60.* Backwards. The shape is negative at every threshold from 1.2% to
  12.5% and **most** negative at the smallest:

  | n_tail | 25 | 50 | 100 | 150 | 250 |
  |---|---|---|---|---|---|
  | median ξ̂ (ML) | −0.38 | −0.11 | −0.14 | −0.19 | −0.19 |
  | ML worst ratio | 0.125 | 0.062 | 0.031 | — | 0.022 |
  | ML median at 1e-7 | 81× | 48× | 20× | — | 8× |

  A smaller threshold makes the ML fit worse on both axes at once. The bounded
  support is real, not a bulk artefact.

* *A profile-likelihood upper bound on survival gives 1.5–2× controlled
  conservatism.* At 95% it is 1.9–67× conservative **and** anti-conservative
  to 0.31; at 99%, 2.2–255× and 0.46. Neither safe nor tight. The χ²
  asymptotics look doubtful near a boundary parameter, which the endpoint is.

---

## Appendix C — reproducing any of this

The harness is `tools/pvalue_study.py` (both stages, five estimators, a
`stress` driver over every geometry × stage), `tools/pvalue_meandiff.py` and
`tools/pvalue_saddlepoint_counts.py`. Four implementation details that had to
be right, recorded because none was obvious:

* **Memory, not speed, stops brute force.** The permutation matrix is `(B, n)`
  and the nulls `(genes, B)`: at `B = 1e8` that is 64 GB and 51 GB.
  Permutations are streamed in blocks with only exceedance counts retained.
* **Stage 2's observed statistic moves with the permutation sample** (§1), so
  brute force streams blocks of 500,000 each carrying its own observed value.
* **Stage 2's effect ladder is `k`, not effect size.** At a fixed `k = 8` of
  40, subsets of 2× through 25× all returned p between 2.5e-4 and 5e-4: the
  gene had reached its combinatorial floor and no effect size buys depth a
  design has not got.
* **Multilevel must score the statistic the method actually computes.** An
  early version scored a subset sum, which off balance is not `mean_shift`,
  and produced medians of 0.00–0.13 at 70 v 10 with the sign of the error
  following which group was larger. It looked exactly like multilevel failing
  on unbalanced designs; it was the harness.

The saddlepoint implementation (`src/wade/saddlepoint.py`) required four
numerical fixes, each of which returned a *plausible wrong p-value* rather
than an error: a bracket found by geometric search that ballooned past what
bisection could close; a warm start that could sit where Newton made no
progress, making the outer function non-monotone; an outer bracket scaled by
the range of the sum rather than of the values; and an `atanh`
reparameterisation that capped the saddlepoint an order of magnitude short
once normalisation put values near 1e8. All four were caught only because the
final answer is checked against both saddlepoint equations before it is
returned. We mention it because it is the kind of thing that silently
contaminates a methods comparison.
