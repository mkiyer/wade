# WADE 0.2.0 method

For each gene WADE measures two things and tests each: the **mean shift**, the
signed area between the case and control quantile functions, and the **subset**,
the cases that depart from that shift. This page gives the exact definitions;
the [manual](manual.md) says how to use them and [validation](validation.md)
what they were measured to do.

## Expression

Label-independent size factors s_j are fitted on a declared normalization
universe (RLE by default: genes positive in every fit sample, their geometric
means, column medians of log ratios, centered to geometric mean one). The test
gene mask is fixed after fitting and never refits the factors.

Counts are discrete, and many samples share a count, above all zero. Each count
c_gj therefore receives independent jitter u_gj ~ Uniform(0, 1), drawn from a
stream seeded by the run's seed and the gene's row, so that it does not depend
on chunking or on which other genes are tested. The tested expression is

\[x_{gj}=(c_{gj}+u_{gj})/s_j \quad(\texttt{scale="linear"})
\qquad\text{or}\qquad x_{gj}=\log_2\!\big((c_{gj}+u_{gj})/s_j+1\big)\quad(\texttt{scale="log2"}).\]

The jitter spreads each count over its own unit, [c, c + 1). The subset test
below moves every case by the mean shift; without jitter it would pull apart
ties, zeros above all, that the observed data keep together.

## The mean shift

Let Q₁ and Q₀ be the empirical quantile functions of the n₁ cases and n₀
controls. The signed area between them is exactly the difference in means,

\[\int_0^1 \{Q_1(u)-Q_0(u)\}\,du=\bar x_1-\bar x_0,\]

for any group sizes, because an empirical quantile function is a step function
whose integral is the group mean. Where cases are higher the area adds; where
they are lower it subtracts. `mean_log2fc` reports it as a fold:
log₂((x̄₁ + 1)/(x̄₀ + 1)) on the linear scale, x̄₁ − x̄₀ on the log2 scale.

## The mean test

Under the null the labels are exchangeable. The difference in means is an
increasing function of the cases' total A, the sum of n₁ of the gene's n values
drawn without replacement, so its permutation p-value is a tail probability of
A. `direction="both"` counts labellings whose total lies at least as far from
its average n₁x̄ as the observed a, `"up"` those with A ≥ a, `"down"` those with
A ≤ a.

**Computed.** Independent Bernoulli selectors Z_i, conditioned on
Σ Z_i = n₁, draw n₁ values without replacement, so (A, Σ Z_i) has the cumulant
generating function K(s, t) = Σ_i log(1 + e^{s x_i + t}) up to a constant. With
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
genes whose tail holds fewer than 1/`relative_error` labellings. All genes are
shuffled when `strata` restrict the shuffles, since the representation above
assumes free exchange.

**Shuffled.** Label shuffles preserve the group sizes and, with strata, each
stratum's case count; every gene sees the same stream, so results do not
depend on chunking. A shuffle counts as an exceedance when its statistic
reaches the observed one within a floating-point tolerance. Each gene stops at
its h-th exceedance, h = ⌈1/relative_error²⌉, drawn at shuffle T, with
p = h/T, whose relative standard error is about 1/√h; or it finishes the budget
B = `max_permutations` with K exceedances and p = (K + 1)/(B + 1). Under
exchangeability this sequential rule is valid: P(p ≤ α) ≤ α at every attainable
α ([Besag, §2.2.2](https://imsarchives.nus.edu.sg/oldwww/Programs/mcmc/files/besag_tl.pdf)).
`pvalues="permutation"` shuffles every gene.

## The subset

The mean shift, carried by every case alike, predicts the case quantiles: the
same fold on count + 1 on the linear scale, the same difference on the log2
scale,

\[\hat Q_1(u)=f\{Q_0(u)+1\}-1,\ f=\frac{\bar x_1+1}{\bar x_0+1}
\qquad\text{or}\qquad \hat Q_1(u)=Q_0(u)+(\bar x_1-\bar x_0),\]

and in both cases ∫Q̂₁ = x̄₁. Walking up the case quantiles, the running
departure from that prediction at the boundary after the k-th case,

\[R(k)=\int_0^{k/n_1}\{Q_1(u)-\hat Q_1(u)\}\,du,\qquad k=1,\dots,n_1-1,\]

starts and ends at zero, falls through cases that carry less than their share
and rises through cases that carry more. Each R(k) is a prefix sum of the cases
less the integral of the prediction, so a walk costs O(n). Boundaries fall only
between distinct case values (equal within 100 ulps). The walk's lowest point,
the highest boundary among the minima within rounding, splits the cases in two;
if R never falls below the rounding of its sums, the gene has no subset. The
depth −min R is the departure the split carries, a deficit below the cut
matched by an excess above it.

The subset is the smaller side of the split, which departs further per case:
the cases above the cut (`"up"`, above the mean shift) or below it (`"down"`);
equal sides count as up. `subset_fraction` is its share of the cases, and
`subset_log2fc` its mean against the prediction's mean over the same quantiles:
log₂ of their ratio on count + 1 on the linear scale, their difference on the
log2 scale. One walk gives one split, so one gene reports one subset; a raised
and a lowered subset in the same gene would need a second split.

## The subset test

The null is that every case carries the mean shift. Each shuffle removes the
observed shift from the cases ((x + 1)/f − 1, or x − (x̄₁ − x̄₀)), pools them with
the controls, draws new labels, applies the shift to the new cases and walks
again. `direction="both"` counts shuffles whose depth reaches the observed
depth; `"up"` scores each walk's depth signed by its subset's side (positive
above the mean shift) and counts those at least the observed signed depth;
`"down"` the reverse. Stopping and p-values follow the shuffled rule above; a
gene without a subset has p = 1. The shift is estimated from the data, so the
test is approximate; the jitter above is what keeps it calibrated at low counts
([validation](validation.md#the-subset-test-and-jitter)).

## Hypothesis families

The mean test and the subset test each have one BH adjustment over the tested
genes. Excluded genes have no p-values or q-values. BH is not guaranteed under
arbitrary dependence; the validation includes correlated gene modules.

## Implementation

The walk is native code (`rust/src/lib.rs`), one implementation for the
observed split and for every shuffle, run over genes in parallel. The
saddlepoint is NumPy, vectorized over genes. Genes are processed in bounded
chunks with no genes × shuffles matrix, and the jitter, the shuffle stream and
the saddlepoint depend only on the gene, the design and the seed, so neither
chunking nor threads change an answer.

## Limits

Exchangeability is assumed: batch or depth confounded with the groups breaks
it, and neither normalization nor permutation repairs it. The subset test
plugs in the estimated mean shift. A subset is a description of where cases
depart from the mean shift, not a set of individually significant patients or
a prevalence estimate.
