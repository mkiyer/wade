# WADE 0.2.0 method

For each gene WADE measures two things and tests each: the **area**, the
signed area between the case and control quantile functions, and the
**subset**, the cases that depart from the one fold every case would share.
This page gives the exact definitions; the [manual](manual.md) says how to use
them and [validation](validation.md) what they were measured to do.

## Expression

Label-independent size factors s_j are fitted on a declared normalization
universe (RLE by default: genes positive in every fit sample, their geometric
means, column medians of log ratios, centered to geometric mean one). The test
gene mask is fixed after fitting and never refits the factors. A normalized
count c_gj/s_j is reads per sample at the cohort's typical depth.

Counts are discrete, and many samples share a count, above all zero. Each count
c_gj therefore receives independent jitter u_gj ~ Uniform(0, 1), drawn from a
stream seeded by the run's seed and the gene's row, so that it depends neither
on chunking nor on which other genes are tested. The subset is found and tested
on

\[x_{gj}=(c_{gj}+u_{gj})/s_j .\]

## The area

Let Q₁ and Q₀ be the empirical quantile functions of the n₁ cases and n₀
controls. The signed area between them is exactly the difference in means,

\[\int_0^1 \{Q_1(u)-Q_0(u)\}\,du=\bar m_1-\bar m_0,\]

for any group sizes, because an empirical quantile function is a step function
whose integral is the group mean. Where cases are higher the area adds; where
they are lower it subtracts. WADE reports the group means of the normalized
counts c/s, with no jitter (`case_mean`, `control_mean`), their difference
(`area`, in reads per sample), and the **area fraction**

\[\frac{\bar m_1-\bar m_0}{\bar m_1+\bar m_0}\in[-1,1],\]

the area over the total area under both curves. It needs no pseudocount, and
for a fold f between the means it is (f − 1)/(f + 1): 1/3 for a doubling,
−1/3 for a halving, and ±1 when one group has no reads. These are the
difference and fraction of the author's HITLIB method.

## The area test

**Depth equalization.** Read counts carry Poisson noise, which shrinks, relative
to the mean, as depth grows. Two samples with the same expression but different
depths are therefore not exchangeable at low counts, even after dividing by size
factors: the shallower one's normalized count is noisier. Under the count model

\[C_j\mid\Lambda_j\sim\text{Poisson}(s_j\Lambda_j),\]

with Λ_j sample j's latent expression, keeping each read of sample j
independently with chance π_j = min_k s_k / s_j gives
C′_j | Λ_j ~ Poisson(min_k s_k · Λ_j), whatever Λ_j: every sample at the
shallowest depth. When the cases' and the controls' latent levels are drawn
from one distribution, the null, the thinned counts are exchangeable and a
permutation test of any statistic of them is exact, at every expression level.
Genes whose counts are too low to trust are thinned to mostly zeros and washed
out, not called; well-expressed genes lose little. The test's values are

\[z_{gj}=(C'_{gj}+u_{gj})/\min_k s_k,\qquad C'_{gj}\sim\text{Binomial}(c_{gj},\pi_j),\]

with the thinning drawn from a stream seeded by the run's seed and the gene's
row (counts rounded to whole reads), and u_gj the gene's jitter.
`params["depth_equalized"]` records min_k s_k over the median size factor: the
share of its reads the median sample keeps. The effects above use every read;
only the test runs on the thinned counts.

**The statistic.** The difference in means of z is an increasing function of
the cases' total A, the sum of n₁ of the gene's n values drawn without
replacement, so its permutation p-value is a tail probability of A.
`direction="both"` counts labellings whose total lies at least as far from its
average n₁z̄ as the observed a, `"up"` those with A ≥ a, `"down"` those with
A ≤ a.

**Computed.** Independent Bernoulli selectors Z_i, conditioned on
Σ Z_i = n₁, draw n₁ values without replacement, so (A, Σ Z_i) has the cumulant
generating function K(s, t) = Σ_i log(1 + e^{s z_i + t}) up to a constant. With
(ŝ, t̂) solving ∂K/∂s = a and ∂K/∂t = n₁ (Newton with backtracking on the
convex φ(s, t) = K − s a − t n₁, to machine precision), Skovgaard's double
saddlepoint gives

\[P(A\ge a)\approx 1-\Phi(w)-\phi(w)\Big(\frac1w-\frac1u\Big),\quad
w=\operatorname{sign}(\hat s)\sqrt{2\{\varphi(0,t_0)-\varphi(\hat s,\hat t)\}},\quad
u=\hat s\sqrt{\det K''(\hat s,\hat t)\,/\,K_{tt}(0,t_0)},\]

with t₀ = log(n₁/n₀) and K_tt(0, t₀) = n₁n₀/n. At the centre, 1/w − 1/u is
0/0; its limit, the count-conditioned skewness κ₃/(6κ₂^{3/2}), is used once
|ŝ| falls below ε^{1/3} (ε the machine precision), where the rounding of the
difference meets the limit's own error. A total at the top of the reachable
range has its exact chance, the labellings that take the n₁ largest values
(ties included) over C(n, n₁); nothing is smaller. Two-sided p-values add the
two tails.

The p-value's error is estimated as the larger of two terms. One is the
saddlepoint's own next term: the second-order correction of the joint density
of (A, Σ Z) at the saddlepoint, ρ₄/8 − (2ρ₂₃² + 3ρ₁₃²)/24 in McCullagh's
invariants of the cumulants of K, less the count's own λ₄/8 − 5λ₃²/24 at t₀.
The other is the granularity of the exact distribution the saddlepoint
smooths: with L = p·C(n, n₁) labellings at least as extreme, a smooth
approximation can be off by about one of them, 1/L. A gene whose estimate
exceeds `relative_error` is shuffled instead, so a small design shuffles the
genes whose tail holds fewer than 1/`relative_error` labellings.

**Shuffled.** Label shuffles preserve the group sizes; every gene sees the same
stream, so results do not depend on chunking. A shuffle counts as an exceedance
when its statistic reaches the observed one within a floating-point tolerance.
Each gene stops at its h-th exceedance, h = ⌈1/relative_error²⌉, drawn at
shuffle T, with p = h/T, whose relative standard error is about 1/√h; or it
finishes the budget B = `max_permutations` with K exceedances and
p = (K + 1)/(B + 1). Under exchangeability this sequential rule is valid:
P(p ≤ α) ≤ α at every attainable α
([Besag, §2.2.2](https://imsarchives.nus.edu.sg/oldwww/Programs/mcmc/files/besag_tl.pdf)).
`pvalues="permutation"` shuffles every gene.

## The subset

Let z = x + 1 for the cases and w = x + 1 for the controls, and let a group's
Lorenz curve L(t) be the share of its total held by its lowest fraction t. At
the case boundaries t = k/n₁, k = 1, …, n₁ − 1, the walk

\[R(k)=\bar z\,\{L_z(k/n_1)-L_w(k/n_1)\}\]

starts and ends at zero. It is also the running departure of the case
quantiles from f{Q₀(u) + 1} − 1, f = z̄/w̄, the prediction if every case
carried the fold of the means, so each R(k) is a prefix sum and a walk costs
O(n). The statistic, `lorenz_gap`, is

\[G=\max_k\{L_w(k/n_1)-L_z(k/n_1)\}_+=-\min_k R(k)/\bar z:\]

the largest amount by which the cases' Lorenz curve falls below the controls',
or how much more concentrated expression is among the cases. Any common fold on
count + 1 applied to every case leaves it unchanged.

The cut is the highest boundary among the walk's minima, within the rounding of
its sums; boundaries fall only between distinct case values (equal within 100
ulps). A gene whose gap is within rounding of zero has no subset
(`subset_side` "none"): its cases are no more unequal than its controls. The
subset is the smaller side of the cut, which departs further per case: the
cases above it (`"up"`) or below it (`"down"`); equal sides count as up.
`subset_fraction` is its share of the cases.

**The registration.** `alignment_log2fc` is log₂ ρ, the fold on count + 1 that
best aligns the groups' quantile functions of log(count + 1): the weighted
median of log{(Q₁(u) + 1)/(Q₀(u) + 1)} over their merged breakpoints, which
minimizes the area between the two (their Wasserstein-1 distance). Where the
median is an interval its midpoint is taken, so swapping the groups inverts ρ
exactly. Under an alternative it is an alignment, not necessarily any
subgroup's fold. `subset_log2fc` is log₂ of the subset's mean of count + 1 over
ρ times the controls' mean of count + 1 over the same quantiles. In genes whose
extreme values dominate both groups the two can point opposite ways; read the
side from `subset_side`.

## The subset test

The null is that every case carries one common fold of the controls'
distribution on count + 1: z has the distribution of ρw. The test pools the
controls with the cases carried back by the registration, (x + 1)/ρ − 1,
shuffles labels over the pool and walks each shuffle; the gap needs no new
registration, because no fold moves it. `direction="both"` counts shuffles
whose gap reaches the observed gap. `"up"` scores each walk's gap signed by its
subset's side (positive above the rest of the cases) and counts those at least
the observed signed gap; `"down"` the reverse. Stopping and p-values follow the
shuffled rule above; a gene without a subset has p = 1.

The registration is estimated from the same data and then held fixed through
the shuffles, so the test is approximate. Under the count model it is
calibrated where the sparser group expects at least 128 reads per sample
(`params["subset_validated_reads"]`); below that it can be anti-conservative, most of all
when the cases are the lower or the shallower group, whose extra Poisson noise
reads as concentration ([validation](validation.md#the-subset-test-under-the-count-model)).
`wade()` warns how many tested genes have a group mean of normalized counts
below 128, and `params["subset_validated_genes"]` counts the rest.

## Hypothesis families

The area test and the subset test each have one BH adjustment over the tested
genes. Excluded genes have no p-values or q-values. BH is not guaranteed under
arbitrary dependence.

## Implementation

The walk and the registration are native code (`rust/src/lib.rs`), one
implementation for the observed split and for every shuffle, run over genes in
parallel. The thinning, the saddlepoint and the shuffled area test are NumPy,
vectorized over genes. Genes are processed in bounded chunks with no genes ×
shuffles matrix, and the jitter, the thinning, the shuffle stream and the
saddlepoint depend only on the gene, the design and the seed, so neither
chunking nor threads change an answer.

## Limits

- **Exchangeability is assumed.** Batch confounded with the groups breaks it,
  and neither normalization nor permutation repairs it. Depth is handled by the
  thinning, but only through size factors: when many genes change one way, the
  size factors absorb part of that change and unchanged genes look changed
  ([validation](validation.md#normalization-when-many-genes-change)).
- **The area test's null is one distribution for both groups.** It is built to
  detect a difference in means, but it can also reject when the groups differ
  only in spread or shape: with equal means, a smaller group that is also the
  wider one was called at about 33% at p ≤ 0.05 in simulation
  ([research ideas](research-ideas.md), section 2.4). Rank such genes by their
  area, which stays near zero.
- **The subset test plugs in an estimated registration** and is validated only
  where the sparser group expects at least 128 reads per sample.
- **A subset is a description** of where cases depart from a common fold, not a
  set of individually significant patients or a prevalence estimate.
