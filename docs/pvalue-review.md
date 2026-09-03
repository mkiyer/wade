# Estimating permutation tails when the support is bounded

**A request for statistical review.** Prepared 2026-09-03 for external
collaboration.

WADE needs p-values near 1e-6 from 2,000 permutations. Every estimator we have
tried is either three orders of magnitude too conservative, or — in the cases
that are accurate — anti-conservative by two. This document sets out the method,
the five candidates, what each does against brute-force ground truth, and where
we think the difficulty actually lies.

> **Provenance.** `docs/scaling.md` §4.5–§4.6 is the canonical record of these
> measurements inside the project; this file is the self-contained snapshot
> written for a reader without access to the code. Where the two ever disagree,
> `scaling.md` is right and this file is stale.

| | |
|---|---|
| Method | WADE — two-stage differential-distribution test for RNA-seq counts |
| Ground truth | 1e8 permutations per design, streamed |
| Designs | 40v40, 60v20, 70v10, 20v60 (n = 80 throughout) |
| Status | **unresolved** |

---

## 0. What we are asking

The permutation null of our test statistics has **bounded support** — the
statistic attains a maximum when the most extreme samples all land in one group.
Extreme-value theory therefore predicts a reversed-Weibull tail with negative
shape parameter, and that is exactly what we fit: `ξ < 0` for **100% of
deep-tail stage-1 genes in every design tested**.

Both standard ways of using that fit fail, in opposite directions:

* Substituting the `ξ → 0` **exponential limit** (what we ship) is conservative
  by 3–4 orders of magnitude.
* Keeping the **negative-shape GPD**, whose support ends at `−σ/ξ`, returns
  survival zero past the estimated endpoint and is anti-conservative by 2.

And the safety of the first is an accident of the second: our shipped estimator
is never anti-conservative on stage 1 **only because the substitution fires on
every gene there**. On stage 2 it fires half the time, and where it does not the
same estimator returns p-values 50× too small.

We would like help finding a tail estimator for a bounded permutation null that
is accurate near 1e-6 to 1e-8 and *never* anti-conservative, at `O(B)` cost with
B = 2,000. §8 lists the specific questions; §1–§7 are the evidence.

---

## 1. The method

WADE is a two-stage differential-distribution test for RNA-seq count data. It
exists to separate a shift affecting every sample from a change confined to a
subset of one group — two situations that produce the same mean difference.

### Setup

For each gene we have raw counts in `n1` cases and `n0` controls. Counts are
normalised (a per-gene length factor and a per-sample library size), with a
small continuity jitter applied at count precision to break the ties that
dominate sparse data. Both groups' quantile functions are read on a shared grid
of `m = min(n1, n0, 2000)` probabilities using the type-7 definition; call them
`Q1(p)` and `Q0(p)`.

### Stage 1 — is there a difference?

```
mean_shift  =  (1/m) · Σ[j=1..m] ( Q1(p_j) − Q0(p_j) )
```

The signed area between the two quantile functions. **When `n1 = n0` this is
exactly the difference of sample means**, and therefore a linear statistic — a
fact that matters a great deal in §3. When the groups are unbalanced the larger
group is read at only `m` probabilities, so the statistic becomes a weighted sum
of that group's order statistics: an **L-statistic**, with no reduction to a
subset sum.

### Stage 2 — what kind of difference is it?

Let `R(p) = log2 Q1(p) − log2 Q0(p)` (with a one-count pseudocount). Under a
pure global fold change `R` is flat, so its running total grows proportionally.
Stage 2 measures departure from that proportionality with a discrete Brownian
bridge:

```
S_k  =  Σ[i≤k] R_i
B_k  =  S_k − (k/m)·S_m          # identically 0 under a global shift
T    =  max[k<m] ( B_k − μ_k ) / σ_k
```

The scan runs over every window width rather than requiring the user to pick
one, and the permutation null prices in the multiplicity of having looked. Two
properties are load-bearing later:

* **The null hypothesis is the fitted global shift, not no-difference.** For
  counts this is realised by binomial thinning of the raw reads under a fitted
  fold change, applied to the observed statistic and to the null alike.
* **`μ_k` and `σ_k` are estimated from the same permutations the tail is read
  from.** So `T` is *not* a fixed function of a label assignment — the observed
  value moves 3% between permutation samples at B = 2,000, and 0.09% at
  B = 500,000.

### Inference

Both stages use the same machinery. Draw `B = 2,000` permutations of the
case/control label vector — **one shuffle serves every gene**, which preserves
the joint null's gene–gene correlation structure. Count exceedances, take the
empirical p-value, refine into the tail where the empirical value has run out,
then Benjamini–Hochberg within each stage separately. There is no combined
p-value and no categorical label.

---

## 2. Why the tail matters

A typical run is 20,000–30,000 genes. BH at α = 0.05 then decides the top genes
at a threshold around `0.05/20000 ≈ 2.5e-6`. The empirical p-value from
B = 2,000 permutations floors at `1/(B+1) = 5e-4` — more than two orders of
magnitude short. On one real 31k × 83k cohort, **18.8% of genes sat at that
floor**.

So refinement is not a nicety. Genes whose true p-value lies between 1e-5 and
1e-7 are simultaneously *the genes BH is ruling on* and the genes no permutation
count can resolve. Whatever the refinement does to them is what the method's
power is.

Raising B is not a way out: the floor moves as `1/B`, so an order of magnitude
of resolution costs an order of magnitude of compute, and the permutation loop
already dominates runtime.

---

## 3. The five candidates

Each takes the same observed statistic and the same 2,000 permutations, except
the saddlepoint, which takes no permutations at all.

### 3.1 Empirical

```
p  =  ( 1 + #{ null ≥ obs } ) / ( B + 1 )
```

Exact and assumption-free; floors at 5e-4. Included as the reference point
rather than as a candidate.

### 3.2 Generalised Pareto, fitted by moments — *what we ship*

Following Knijnenburg et al. (2009). Take the top `n_tail = 250` null values,
subtract the threshold to give exceedances `y`, and fit a GPD by method of
moments:

```
r  = mean(y)² / var(y)
ξ̂  = (1 − r)/2
σ̂  = mean(y)·(1 + r)/2

tail = (1 + ξ̂·y/σ̂)^(−1/ξ̂)      if ξ̂ > 0
tail = exp(−y/σ̂)               if ξ̂ ≤ 0    ← the substitution

p    = max( (n_tail/B)·tail , 1/(B·n_tail) )
```

The `ξ̂ ≤ 0` branch is deliberate: a GPD with negative shape has a hard upper
bound at `−σ̂/ξ̂`, beyond which the survival function is zero, and an observation
past it would collapse a strong statistic to a machine-epsilon p-value. The
exponential limit was chosen to avoid that. **It is the single decision this
whole document turns on.**

### 3.3 Generalised Pareto, fitted by maximum likelihood

Identical in every other respect — same threshold, same rescaling, same floor —
with `(ξ̂, σ̂)` from ML instead of moments, and the negative-shape survival
function used as-is rather than substituted. Included because moments are known
to behave poorly, and because it isolates which of the two failures is smaller.

### 3.4 Double saddlepoint

Where the statistic is a monotone function of a **subset sum**, the tail
probability has a classical saddlepoint approximation with relative-error
accuracy and no sampling at all. For stage 1, with `A` the case-subset sum and
`V` the total:

```
T = A·(1/n1 + 1/n0) − V/n0        so   P(T ≥ t) = P(A ≥ a)
```

Sampling `n1` of `n` without replacement is handled by conditioning: with
independent Bernoulli selectors, `K(s,t) = Σ log(1 + exp(s·v_i + t))`, and
Skovgaard's double-saddlepoint formula gives `P(A ≥ a | ΣZ = n1)`.

**Group balance is not a requirement of the saddlepoint** — that identity holds
for any `n1, n0`. What is required is that `mean_shift` *be* that quantity,
which is true only when `n1 = n0` (§1). Unbalanced, stage 1 is an L-statistic
and this approach does not apply. Stage 2 has no such reduction at any geometry.

### 3.5 Adaptive multilevel splitting

The scheme in `fgsea` (Korotkevich et al.), adapted from gene-set sampling to
label permutation. Hold a population of ~1,000 label vectors. Each round:
discard the lower half by statistic and duplicate the upper half, recording the
discarded median as the level; then move every survivor by swapping one case
label with one control label, accepting only while the statistic stays above the
level. The level ratchets upward, so `P ≈ (1/2)^k` after k rounds — 1e-100 is
~332 rounds rather than 1e100 draws.

The estimator is **not** `2^−k`. Each level is a *sample* median whose
exceedance probability is a Beta order statistic, so rounds accumulate
`E[log U] = ψ(a) − ψ(a+b)` rather than `log E[U]`, which is what makes `log p`
approximately unbiased instead of systematically optimistic:

```
log p̂  =  k·[ ψ(N/2) − ψ(N+1) ]  +  ψ(rem + 1) − ψ(N+1)
```

Two costs. It is **per gene**, so it abandons the one-shuffle-serves-all-genes
design and with it the joint null's correlation structure — marginal p-values
stay valid, the joint does not. And for stage 2 it cannot be applied at all
until `μ_k, σ_k` are frozen from a separate uniform sample, because it samples a
*conditioned* distribution whose moments are not the null's.

---

## 4. Experimental design

Ground truth is brute force. Everything below is measured against it, not
against theory.

* **Geometry.** `n = 80` held fixed, split four ways: 40v40, 60v20, 70v10,
  20v60. Holding `n` fixed makes a difference between geometries a difference in
  *balance* rather than in permutation cost.
* **Ground truth.** 1e8 permutations per design, streamed in blocks of 500,000
  with only exceedance counts retained — the full permutation matrix at that
  size would be 64 GB. A gene counts as resolved at ≥ 10 exceedances; 44–62 of
  64 genes resolve per stage-1 design.
* **Genes.** 64 per design, in 8 rungs of 8. Stage 1 plants a global
  multiplicative shift per rung; stage 2 plants `k` affected cases at 8×.
  Scatter *within* a rung is wanted — it fills the p-value range continuously
  rather than leaving 8 discrete points.
* **Data.** Stage 1 uses log-normal expression: continuous and tie-free, the
  cleanest setting in which to measure a tail. Stage 2 uses negative-binomial
  counts, because its null is built by thinning reads.
* **Reported.** For each estimator, the ratio `p̂ / p_true`, binned by decade of
  true p. A ratio above 1 is conservative; below 1, anti-conservative.

---

## 5. Results

### 5.1 Stage 1, by decade of true p (40 v 40)

Median ratio `p̂/p_true`. Brute force 1e8 permutations, 44 of 64 genes resolved.

| true p | n | empirical | GPD moments | GPD mle | multilevel | saddlepoint |
|---|---|---|---|---|---|---|
| 1e-3 – 1e-2 | 4 | 1.33 | 3.22 | 1.03 | **0.98** | **1.00** |
| 1e-4 – 1e-3 | 9 | 3.29 | 12.4 | 1.03 | **0.99** | **0.99** |
| 1e-5 – 1e-4 | 10 | 21.5 | **76.4** | **0.09** | **1.01** | **0.99** |
| 1e-6 – 1e-5 | 7 | 114 | **304** | 0.83 | **1.02** | **1.02** |
| 1e-7 – 1e-6 | 8 | 2028 | **2137** | 8.12 | **0.86** | **0.91** |

### 5.2 The same statistic, four geometries

Range of bin medians across the five decades.

| design | empirical | GPD moments | GPD mle | multilevel | saddlepoint |
|---|---|---|---|---|---|
| 40 v 40 | 1.3 – 2028 | 3.2 – 2137 | 0.09 – 8.1 | **0.86 – 1.02** | **0.91 – 1.02** |
| 60 v 20 | 1.1 – 1660 | 3.9 – 4906 | 0.09 – 6.6 | **0.92 – 1.01** | n/a |
| 70 v 10 | 1.1 – 1204 | 3.5 – 2947 | 0.13 – 4.8 | **0.89 – 1.13** | n/a |
| 20 v 60 | 1.2 – 1351 | 3.6 – 1909 | 0.09 – 5.4 | **0.86 – 1.04** | n/a |

The saddlepoint is inapplicable off balance because `mean_shift` is then an
L-statistic rather than a subset sum.

### 5.3 Worst anti-conservative case, and the fitted shape

Smallest `p̂/p_true` observed on any single gene. The last two columns are over
genes with true p < 1e-5.

| design | empirical | GPD moments | GPD mle | multilevel | saddlepoint | ξ̂ < 0 | median ξ̂ |
|---|---|---|---|---|---|---|---|
| 40 v 40 | 0.83 | **1.000** | **0.022** | 0.54 | 0.61 | 100% | −0.23 |
| 60 v 20 | 0.63 | **1.000** | **0.007** | 0.65 | — | 100% | −0.29 |
| 70 v 10 | 0.57 | **1.000** | **0.018** | 0.77 | — | 100% | −0.25 |
| 20 v 60 | 0.61 | **1.000** | **0.031** | 0.69 | — | 100% | −0.20 |

On stage 1, GPD-moments is never anti-conservative. §5.4 shows why that is not
the reassurance it looks like.

### 5.4 Stage 2, and the result that changes the picture

Stage 2's statistic is a maximum over widths, which has a heavier null tail than
a mean difference. The fitted shape is negative for only **67%** of deep-tail
genes (median −0.05) against stage 1's 100%, and the exponential substitution
fires on **50%** against 100%.

40 v 40, brute force 1e8, 28 of 64 resolved:

| true p | n | empirical | GPD moments | GPD mle |
|---|---|---|---|---|
| 1e-3 – 1e-2 | 10 | 1.47 | **1.17** | 0.86 |
| 1e-4 – 1e-3 | 7 | 2.84 | **0.80** | 0.85 |
| 1e-5 – 1e-4 | 5 | 28.0 | 4.10 | 1.69 |
| 1e-6 – 1e-5 | 3 | 180 | **1.43** | 0.72 |
| 1e-7 – 1e-6 | 3 | 2630 | 14.4 | 10.5 |

Worst anti-conservative ratio:

| | empirical | GPD moments | GPD mle |
|---|---|---|---|
| 40 v 40, stage 2 | 0.93 | **0.019** | **0.019** |

**This is the important row.** GPD-moments is far more *accurate* on stage 2
than on stage 1 — and it is also, here, **anti-conservative by 50×**. Its
perfect safety record on stage 1 was not a property of the estimator. It was a
property of the exponential substitution firing on every single gene. Where the
substitution does not fire, the shipped estimator behaves like the ML one,
because it *is* the same GPD form.

So the honest summary is: **no GPD variant we have tested is safe.** One of them
merely never gets the chance to be unsafe on one of the two stages.

A separate observation, worth recording because it inverts an expectation: the
p-value problem is **worse for stage 1 than for stage 2**, which is the opposite
of the stages' relative importance to the method.

> The three unbalanced geometries for stage 2 were still computing when this was
> written. They are the main gap in the evidence below.

---

## 6. Diagnosis

The permutation distribution of these statistics has a **finite right
endpoint**. There are only `C(n, n1)` label assignments, and the statistic is
maximised by the one placing the largest values in the case group. A
distribution with bounded support sits in the reversed-Weibull domain of
attraction, so the correct limiting tail has negative shape. Our fits agree:
`ξ̂ < 0` for 100% of deep-tail stage-1 genes across all four geometries, median
−0.20 to −0.29.

**So the negative shape is not a pathology to be worked around. It is the right
answer, and both of our implementations mishandle it.**

| | what it does | why it fails |
|---|---|---|
| **Substituting `exp(−y/σ̂)`** | assumes the endpoint is at infinity | an exponential tail assigns mass far beyond the true endpoint, so every observation in that region is inflated — 76× at 1e-5, 2137× at 1e-7. Safe, and it costs real discoveries |
| **Keeping the negative-shape GPD** | trusts a point estimate of the endpoint exactly | support ends at `−σ̂/ξ̂`; an observation past that gets survival exactly zero and falls to the floor. The endpoint is estimated from 250 points and is routinely too small — hence ratios of 0.007 |

Put another way: **the whole problem is estimating the right endpoint of a
bounded distribution from its 250 largest order statistics**, and then behaving
sensibly when the observation lands near or past it. The two shipped behaviours
are the two degenerate answers.

---

## 7. The dilemma

| candidate | status | verdict |
|---|---|---|
| **GPD by moments** | shipped | Conservative by 3–4,900× on stage 1. Anti-conservative by 50× on stage 2. Its stage-1 safety is an artefact of a substitution that fires there and not elsewhere |
| **GPD by ML** | rejected | Best median accuracy of any `O(B)` method, and returns p-values up to 140× too small. For a method whose output feeds FDR control, disqualifying |
| **Multilevel splitting** | works | 0.86–1.13 median in every geometry, worst case 0.54. But per-gene: abandons the shared-permutation joint null, needs frozen moments for stage 2, and costs ~1 s/gene there against ~0 for a GPD fit |
| **Saddlepoint** | limited | The most accurate thing we have, with no sampling at all — and it needs the statistic to be a subset sum, which stage 1 only is on a balanced design and stage 2 never is |

The awkward part is that **the ranking by median accuracy is nearly the reverse
of the ranking by safety.** The estimator we would pick on accuracy is the one
we must reject; the one we ship is the least accurate and, on the stage where it
looks safest, only looks safe. The two methods that are both accurate and safe
each carry a structural cost — one needs a statistic we only sometimes have, the
other needs us to give up a design property (one shuffle for all genes) that we
chose deliberately.

---

## 8. Questions

1. **Endpoint estimation.** Is there a standard treatment for tail probabilities
   near the finite endpoint of a bounded distribution, when the endpoint must
   itself be estimated? We are effectively asking for a survival estimate that
   degrades gracefully as the observation approaches and passes the estimated
   endpoint, rather than saturating at zero.

2. **A controlled conservative bound.** Would a profile-likelihood *lower
   confidence bound* on the endpoint give a p-value that is conservative by a
   known, small factor instead of by three orders of magnitude? That would be an
   acceptable answer — we do not need unbiasedness, we need safety at a bounded
   price.

3. **Threshold choice.** We use the top 250 of 2,000 (12.5%), inherited from the
   original implementation, with no threshold-stability analysis. Does the
   negative shape persist across thresholds, or is some of it a threshold
   artefact? What diagnostic would you want to see?

4. **Family choice.** Is a GPD the right family here at all, or would a
   distribution parameterised directly on its endpoint behave better under
   estimation?

5. **Multiplicity.** Multilevel splitting is per-gene, so its p-values are
   marginally valid but do not share a permutation ensemble. How much does BH's
   FDR control actually depend on that shared joint structure in this setting?

6. **L-statistics.** For the unbalanced case, is there a relative-error tail
   approximation for the permutation distribution of a linear combination of
   order statistics, analogous to the saddlepoint for a linear statistic?

7. **A design question.** We could redefine stage 1 as the plain difference of
   means for all geometries, making it linear everywhere and the saddlepoint
   universal. The cost is breaking byte-level agreement with the R
   implementation we validated against, on unbalanced fixtures. Is the
   quadrature worth defending on statistical grounds, or is it just history?

8. **Studentisation.** Stage 2 standardises by moments estimated from the same
   permutations the tail is read from. We believe those should be frozen from a
   separate uniform sample — bulk quantities converging as `1/√B`, kept apart
   from a rare-event quantity. Is that the right instinct?

---

## Appendix A — the design floor

Related, and worth flagging because we had it wrong in our own documentation
until this week. If exactly `k` case samples carry the signal, the probability
that a relabelling puts all of them in the case group is
`C(n1,k)/C(n1+n0,k)`. We had described this as a hard lower bound on the
achievable p-value. **It is not.**

300 v 300, B = 40,000 so every p-value is a raw permutation count and no
refinement fires:

| planted k | floor | p/floor q10 | median | q90 | fell below |
|---|---|---|---|---|---|
| 3 of 300 | 1.24e-1 | 0.48 | 1.96 | 3.23 | **21%** |
| 6 of 300 | 1.52e-2 | 0.26 | 1.10 | 1.71 | **42%** |

If the floor were a bound, every ratio would be ≥ 1.

It is exact only in the noise-free limit, where every relabelling placing all `k`
affected samples in cases *ties* the observed value. With noise those
relabellings scatter above and below it, so roughly half fall below — and in
count data `k` is not even well defined, since dispersion makes some unaffected
samples look extreme. The quantity survives as a good estimate of the *order of
magnitude* a design can support, which is what it is used for when advising on
study design. It is not a statement about any individual gene, and it cannot be
used to clamp a p-value — which was the first fix we tried.

---

## Appendix B — reproducibility

All figures are measured against brute-force permutation, not simulated from
theory; every table states its ground-truth permutation count and how many genes
resolved. Ratios are `p̂/p_true`.

The harness is `tools/pvalue_study.py` in the WADE repository:

```
python tools/pvalue_study.py calibrate 40 40 1     # find a p ladder
python tools/pvalue_study.py truth     40 40 1 1e8 # the ground-truth column
python tools/pvalue_study.py compare   40 40 1     # the table
python tools/pvalue_study.py stress                # every geometry x stage
```

Four implementation details that had to be right, recorded because none was
obvious in advance:

* **Memory, not speed, stops brute force.** The permutation matrix is `(B, n)`
  and the nulls are `(genes, B)`: at B = 1e8 that is 64 GB and 51 GB.
  Permutations are streamed in blocks with only exceedance counts retained, so B
  is unbounded and the footprint constant.
* **Stage 2's statistic is not a fixed function of a label assignment** (§1), so
  brute force streams blocks of 500,000 each carrying its own observed value, and
  multilevel splitting must freeze the moments first.
* **Stage 2's effect ladder is `k`, not the effect size.** At a fixed `k = 8` of
  40, subsets of 2× through 25× all returned p between 2.5e-4 and 5e-4: the gene
  had reached the combinatorial floor for that `k`, and no effect size buys depth
  a design has not got.
* **Multilevel must score the statistic the method actually computes.** An early
  version scored a subset sum, which off balance is not `mean_shift`, and
  produced medians of 0.00–0.13 at 70v10. It looked exactly like multilevel
  failing on unbalanced designs; it was the harness. The corrected version scores
  every population member with the same kernel brute force uses.
