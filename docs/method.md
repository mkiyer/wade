# WADE 0.2.0 method

For each gene WADE measures two things and tests each: the **area**, the
signed area between the case and control quantile functions, and the
**subset**, the cases that depart from the one fold every case would share.
This page gives the exact definitions; the [manual](manual.md) says how to use
them and [validation](validation.md) what they were measured to do.

## Expression

Every normalization ends as one label-independent size factor s_j per sample:
the sample's depth as the normalizer measures it.

- **RLE by default:** on the genes with reads in every fit sample, the column
  medians of the log ratios to the genes' geometric means, centered to
  geometric mean one.
- **Internal control genes:** the same RLE restricted to the controls.
- **Supplied factors:** from any other estimator.
- **Library totals:** only when chosen.

The test gene mask is fixed after fitting and never refits the factors. A
normalized count c_gj/s_j is reads per sample at the cohort's typical depth.
The size factors serve twice: as the denominators of the effects, and as the
depths the area test's thinning equalizes.

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

the area over the total area under both curves. It needs no pseudocount.
The **fold** is the ratio of the same two means,

\[f=\bar m_1/\bar m_0,\qquad \texttt{log2fc}=\log_2 f,\]

with no pseudocount either: +∞ when the controls have no reads, −∞ when the
cases have none. The area fraction is (f − 1)/(f + 1): 1/3 for a doubling,
−1/3 for a halving, and ±1 when one group has no reads. These are the
difference, fraction and fold of the author's HITLIB method (`diff.mean`,
`diff.frac` and `fc`).

## The area test

**Depth equalization.** Read counts carry Poisson noise, which shrinks, relative
to the mean, as depth grows. Two samples with the same expression but different
depths are therefore not exchangeable at low counts, even after dividing by size
factors: the shallower one's normalized count is noisier. Under the count model

\[C_j\mid\Lambda_j\sim\text{Poisson}(s_j\Lambda_j),\]

with Λ_j sample j's latent expression, keeping each read of sample j
independently with chance π_j = min_k s_k / s_j gives
C′_j | Λ_j ~ Poisson(min_k s_k · Λ_j), whatever Λ_j: every sample at the
smallest size factor. When the cases' and the controls' latent levels are drawn
from one distribution, the null, the thinned counts are exchangeable and a
permutation test of any statistic of them is exact, at every expression level.
Genes whose counts are too low to trust are thinned to mostly zeros and washed
out, not called; well-expressed genes lose little. The test's values are

\[z_{gj}=(C'_{gj}+u_{gj})/\min_k s_k,\qquad C'_{gj}\sim\text{Binomial}(c_{gj},\pi_j),\]

with the thinning drawn from a stream seeded by the run's seed and the gene's
row (counts rounded to whole reads), and u_gj the gene's jitter.
`params["depth_equalized"]` records min_k s_k over the median size factor: the
share of its reads the median sample keeps. The effects above use every read;
only the test runs on the thinned counts. The exactness needs the size factors
to be the true exposures up to one common constant: RLE on all genes, when
most genes do not change, or on internal control genes that do not.

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

**Shuffled.** Label shuffles preserve the group sizes, and every gene, in both
tests, sees one stream of them.

- **The stream.** It comes in blocks of 1,024 shuffles, each drawn by its own
  generator seeded by the run's seed and the block's index. Any shuffle can be
  regenerated from its position alone, so results depend on neither chunking
  nor the order in which genes run.
- **An exceedance** is a shuffle whose statistic reaches the observed one,
  within a floating-point tolerance.
- **The stopping rule.** Each gene stops at its h-th exceedance,
  h = ⌈1/relative_error²⌉, drawn at shuffle T, with p = h/T, whose relative
  standard error is about 1/√h. Otherwise it finishes the budget
  B = `max_permutations` with K exceedances, and p = (K + 1)/(B + 1).
- **Validity.** Under exchangeability this sequential rule is valid:
  P(p ≤ α) ≤ α at every attainable α
  ([Besag, §2.2.2](https://imsarchives.nus.edu.sg/oldwww/Programs/mcmc/files/besag_tl.pdf)).
- `pvalues="permutation"` shuffles every gene.

**Stopping at the decision.** After T shuffles with K < h exceedances, the run
cannot finish with a p-value above

\[U=\frac{h}{T+h-K}\ \text{ if } T+h-K\le B,\qquad
U=\frac{K+B-T+1}{B+1}\ \text{ otherwise}:\]

its p-value if every remaining shuffle were an exceedance. U never rises as
shuffles are added, and it equals p once the run stops.

With `fdr` = α, BH's threshold is found top-down:

1. Start with r = m, the number of tested genes in the family.
2. Run every shuffled gene until U ≤ t = rα/m, or until its run finishes.
3. Count the genes with U ≤ t, or with a computed p ≤ t. If the count is r,
   t is BH's threshold. Otherwise set r to the count and repeat.

Every gene then either has U ≤ t, so its finished p is at most t, or has
finished. Each count is therefore what the finished p-values give, and the
iteration converges to BH's own threshold. BH on the reported values rejects
exactly the genes BH on finished runs rejects, on every run
([Fischer, Barry and Ramdas, 2026](https://doi.org/10.1214/26-EJS2514),
lemma 4.1).

- The rejected genes report U, at most the threshold, as their p-value.
- Every other gene's p-value is its finished one.
- A rejected gene needs at most about h/t shuffles, against up to B.
- The false discovery rate is that of BH on finished runs. Like that, it
  rests on positive dependence among the genes' p-values, here computed on
  shared shuffles.
- `fdr=None` finishes every run.

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
distribution: under the count model, the cases' latent levels are the controls'
times one fold. At low counts a fold changes Poisson noise and zeros as well as
levels, so rescaling the cases' counts cannot reproduce the null, and the extra
noise of the lower group reads as concentration. Thinning can reproduce it: a
thinned mixed-Poisson count is mixed Poisson at the lower rate, whatever the
latent level. The test therefore runs on counts made exchangeable by thinning.

1. **Depth.** Every sample is thinned to the smallest size factor, the area
   test's own draw (above), so that no difference in depth reads as a subset.
2. **The fold.** ρ is the Wasserstein-1 fold of those equalized, jittered
   counts: the weighted median of the log ratio of the groups' quantile
   functions of log(count + 1), the midpoint of any median interval.
3. **Thinning by the fold.** The higher group is thinned further by ρ. Each
   case keeps each read with chance 1/ρ when ρ > 1; each control keeps each
   read with chance ρ when ρ < 1. The draw comes from a stream seeded by the
   run's seed and the gene's row (key 3). With the true fold, the groups'
   thinned counts are exchangeable under the null.
4. **The shuffles.** Labels are shuffled over the jittered thinned counts and
   every shuffle is walked, with no further registration.

`direction="both"` counts shuffles whose gap reaches the observed gap. `"up"`
scores each walk's gap signed by its subset's side (positive above the rest of
the cases) and counts those at least the observed signed gap; `"down"` the
reverse. Stopping and p-values follow the shuffled rule above. A gene without a
subset on its normalized counts, the subset that is reported, has p = 1.

The subset's description (its side, share, fold and gap) is read on normalized
counts, as the area's effects are; only the test runs on thinned counts.

The fold is estimated from the same data and held fixed through the shuffles,
so the test is approximate. Under the count model it is calibrated where the
sparser group expects at least 32 reads per sample
(`params["subset_validated_reads"]`). Its worst share there at p ≤ 0.01 is
1.9%, where an exact test with the true fold reaches 1.8% by chance
([validation](validation.md#the-subset-test-under-the-count-model)). Below 32
reads it can be anti-conservative when every case carries a fold. `wade()` warns
how many tested genes have a group mean of normalized counts below 32, and
`params["subset_validated_genes"]` counts the rest.

The thinning is one seeded draw, and near BH's threshold it decides calls: on
TCGA prostate two seeds agreed on 70% of the genes either called, against 93%
when only the shuffles change
([validation](validation.md#the-subset-test-under-the-count-model)).

## Rarity

Let gene g's subset lie at cut k_g, with k_g cases below it, and let
d_h(k) = L_w,h(k/n₁) − L_z,h(k/n₁) be gene h's gap at boundary k. The walk
computes d_h(k) at every k, and gives −∞ where equal cases forbid a boundary.

- **Why size-matched.** Lorenz curves lie on or below the diagonal, so
  d_h(k) ≤ k/n₁, the share of the cases below the cut. A lowered subset's
  gap is capped by its own share, a raised subset's by the share of the
  rest. For a fraction π raised φ-fold over an otherwise even group, the gap
  is π(1 − π)(φ − 1)/(1 − π + πφ): 0.79 for 5% at 100×, 0.45 for 50% at
  20×. So gaps of different cuts do not compare.
- **The size-matched share.** Within g's group G of tested genes,

  \[s_g=\frac{\#\{h\in G: d_h(k_g)\ge d_g(k_g)\}}{|G|},\]

  the share of genes whose gap at g's own cut, a subset of the same size and
  side, is at least g's. A gene without a subset has s = 1.
- **The rank.** Each gene was scored at its own best cut, so the shares are
  ranked:

  \[\texttt{subset\_rarity}_g=\frac{\#\{h\in G: s_h\le s_g\}}{|G|}.\]

  If the genes of G are exchangeable, this is a conformal rank, uniform over
  G ([Bates, Candès, Lei, Romano and Sesia, 2023](https://doi.org/10.1214/22-AOS2244)).
  Ranking within a group keeps that property within each group
  ([Vovk, 2013](https://doi.org/10.1007/s10994-013-5355-6)).
- **The two groups.** The tested genes whose lower group mean is at least
  32 reads per sample, the subset test's validated range, and the rest.

The cost is about |G|² comparisons of gaps, in chunks of genes. Rarity is
relative to the cohort's genes and is not a test. Selecting genes by it before
BH would break the BH family.

**Sample load.** Case j's load is

\[\ell_j=\sum_g \mathbf 1[j\in S_g]\,\frac{w_g}{|S_g|}\Big/\frac{\sum_g w_g}{n_1},
\qquad w_g=-\log \texttt{subset\_rarity}_g,\]

over the genes with a subset S_g. If each subset's members were drawn at
random from the cases, every case would expect a load of 1, and the loads
average exactly 1. Co-regulated genes share members, so the loads are not
independent sums, and no tail probability is attached.

## Strata

**The null.** Strata are the groups that one or more factors define: each
combination of their levels is a stratum. Within each stratum the labels are
exchangeable; strata may differ in anything.
- Every shuffle exchanges labels only within strata. The stream's block
  generator permutes each stratum's labels in turn, in the order of the
  strata, so the stream stays position-indexed.
- Both tests are then exact under the stratified null, up to the subset test's
  registration.
- A stratum with only one group cannot move a label, so its samples are left
  out of both tests.
- The effective size is N_eff = Σ_s n₁ₛn₀ₛ/nₛ.
- With one stratum holding both groups, the tests are the unstratified ones on
  its samples.

**The area test.**
- Each stratum is thinned to its own smallest size factor,
  z_gj = (C′_gj + u_gj)/min_{k∈s(j)} s_k. Exchangeability within a stratum
  needs equal depths only there, and the size factors need be right only up
  to one constant per stratum.
- The values are centred within their stratum. The statistic is the cases'
  total, the Cochran–Mantel–Haenszel sum
  T = Σ_s (n₁ₛn₀ₛ/nₛ)(z̄₁ₛ − z̄₀ₛ), centred at zero under the stratified null,
  as the two-sided test needs.

**Computed, with strata.** The selectors are conditioned on one count per
stratum:

\[K(s,\mathbf t)=\sum_j\log\bigl(1+e^{s z_j+t_{s(j)}}\bigr).\]

This is an (S + 1)-dimensional double saddlepoint.
- **The denominator** has a closed form: t̃_s = log(n₁ₛ/n₀ₛ), with
  determinant Π_s n₁ₛn₀ₛ/nₛ.
- **The numerator's Hessian is an arrowhead.** s meets every t_s, and no t_s
  meets another. Each Newton step is O(n), through the Schur complement
  c = K_ss − Σ_s K_{st_s}²/K_{t_st_s}.
- **w** is the unstratified one, from per-selector Bregman divergences about
  each stratum's centre.
- **u** is ŝ (c Π_s K_{t_st_s} / Π_s n₁ₛn₀ₛ/nₛ)^{1/2}.
- **The second-order error term** is a sum over samples. With
  e_j = (z_j, the unit vector of j's stratum), every cumulant tensor is
  Σ_j k_j⁽ᵐ⁾ e_j^{⊗m}, and the inverse Hessian M gives
  e_j′Me_k = z̃_j z̃_k/c + [same stratum]/K_{t_st_s}, with
  z̃ = z − K_{st_s}/K_{t_st_s}. From these, McCullagh's ρ₄, ρ₁₃ and ρ₂₃ follow
  without forming a tensor, less the strata counts' own
  Σ_s (λ₄ₛ/8 − 5λ₃ₛ²/24).
- **The granularity term** uses L = p Π_s C(nₛ, n₁ₛ). The top of the range is
  exact.
- With one stratum it is the unstratified saddlepoint.

**The subset test, with strata.**
1. Each stratum is thinned to its own smallest size factor.
2. One fold is taken for every case: the weighted median of the strata's own
   Wasserstein-1 folds of the equalized counts, with weights n₁ₛn₀ₛ/nₛ. Where
   the weights below a value make exactly half, it is the midpoint.
3. The higher group is thinned by that fold in every stratum.
4. Each stratum is moved, by a fold on count + 1, to the cohort's median
   log(count + 1). Within a stratum every sample moves alike, so no label is
   favoured.
5. One walk compares the pooled cases with the pooled controls, with labels
   shuffled within strata.

The null is that within each stratum every case carries one fold of the
stratum's controls, the same fold in every stratum, while strata may differ in
level. The subset is described on normalized counts with each stratum moved to
the cohort's level and the cases carried back by the strata's fold. Four other
constructions were measured and set aside ([plan](strata-plan.md), section 4).

**The effects.**
- case_mean and control_mean are the Mantel–Haenszel standardized means over
  the strata with both groups: Σ_s w_s mean₁ₛ / Σ_s w_s, likewise for the
  controls, with w_s = n₁ₛn₀ₛ/nₛ.
- The area, the area fraction and log2fc follow from them, so log2fc is the
  Mantel–Haenszel fold.

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

- **Exchangeability is assumed,** across the cohort or, with strata, within
  each stratum. Batch confounded with the groups breaks it. Strata repair a
  batch that both groups share; nothing separates a batch that coincides with
  the condition. Depth is handled by the
  thinning, but only through size factors: when many genes change one way, the
  size factors absorb part of that change and unchanged genes look changed
  ([validation](validation.md#normalization-when-many-genes-change)).
- **The area test's null is one distribution for both groups.** It is built to
  detect a difference in means, but it can also reject when the groups differ
  only in spread or shape: with equal means, a smaller group that is also the
  wider one was called at about 33% at p ≤ 0.05 in simulation
  ([research ideas](research-ideas.md), section 2.4). Rank such genes by their
  area, which stays near zero.
- **The subset test plugs in an estimated fold** and is validated only where
  the sparser group expects at least 32 reads per sample. Below that, a
  valid test would need the fold's estimation inside its null.
- **A subset is a description** of where cases depart from a common fold, not a
  set of individually significant patients or a prevalence estimate.
