# WADE 0.2.0 method

## Expression

Fit label-independent size factors s_j on a declared normalization universe
(RLE by default: genes positive in every fit sample, their geometric means,
column medians of log ratios, centered to geometric mean one). The tested
expression of gene g in sample j is

\[x_{gj}=c_{gj}/s_j \quad(\texttt{scale="linear"})
\qquad\text{or}\qquad x_{gj}=\log_2(c_{gj}/s_j+1)\quad(\texttt{scale="log"}).\]

The test gene mask is fixed after fitting and never refits the factors.

## Areas

Let Q₁ and Q₀ be the empirical quantile functions of cases and controls, and
F₁, F₀ their CDFs. The two areas are

\[A_\uparrow=\int_0^1 \max\{Q_1(u)-Q_0(u),0\}\,du=\int \max\{F_0(x)-F_1(x),0\}\,dx,\]
\[A_\downarrow=\int_0^1 \max\{Q_0(u)-Q_1(u),0\}\,du=\int \max\{F_1(x)-F_0(x),0\}\,dx.\]

Both integrals describe the same region between the two curves, so each form
is exact. A↑ + A↓ is the Wasserstein-1 distance and A↑ − A↓ is the difference
in means. Quantiles where cases are lower add nothing to A↑ rather than
subtracting from it, which is why a raised subset survives a lowered bulk: for
controls [10, 10, 10, 10] and cases [2, 2, 2, 30], A↑ = 5, A↓ = 6, while the
mean difference is −1.

Sort the N pooled values; let d_k be the gap between sorted values k and k+1,
and a_k the number of cases among the first k. With n₁ cases and n₀ controls,
the scaled CDF difference over gap k is m_k = n₁k − N a_k, and

\[A_\uparrow=\frac{1}{n_1n_0}\sum_{k=1}^{N-1}\max(m_k,0)\,d_k,\qquad
A_\downarrow=\frac{1}{n_1n_0}\sum_{k=1}^{N-1}\max(-m_k,0)\,d_k.\]

Each group keeps its own empirical mass; there is no common quantile grid,
smoothing, trimming or interpolation. Ties have zero gaps.

## Permutation test

Label shuffles preserve group sizes and, with strata, each stratum's case
count. Every gene uses the same stream of shuffles, so results do not depend
on chunking. The pooled values are sorted once per gene, and each shuffle
yields both areas from one pass over the sorted values.

A shuffle counts as an exceedance when its area is at least the observed area
times 1 − 100ε (a conservative tolerance for floating-point ties). Each
direction has its own Besag–Clifford record: it stops at the h = 20th
exceedance, drawn at shuffle T, with p = h/T, or it finishes the cap B with K
exceedances and p = (K + 1)/(B + 1). The default cap is B = 999,999. Under
exchangeability the rule is valid: the probability that p ≤ α is at most α
at every attainable α ([Besag, §2.2.2](https://imsarchives.nus.edu.sg/oldwww/Programs/mcmc/files/besag_tl.pdf)).
A direction that is not tested draws nothing and has no p-value.

## Hypothesis families

`direction="both"` is a discovery hypothesis. Both directions are tested from
the same shuffles, and one BH adjustment covers every tested gene-direction
pair (2G hypotheses for G tested genes). `direction="up"` or `"down"` is a
declared one-directional hypothesis with one BH adjustment over the G tested
genes. Excluded genes have no p-values or q-values in either case.

The null is exchangeability of the labels, equivalently equality of the two
expression distributions. A pure downward shift leaves A↑ near zero, so it
rarely produces an upward call; the confirmation measured no wrong-direction
calls on whole-cohort shifts. BH is not guaranteed under arbitrary dependence;
the validation includes correlated gene modules.

## Descriptions

The bulk shift is median(case) − median(control) of log2(count/s + 1), with
the ordinary average at even group sizes. The upper and lower tails integrate
each empirical quantile function over [0.9, 1] or [0, 0.1], take case minus
control, and subtract the bulk shift; fractional boundaries are integrated
exactly. These descriptions use log2 whatever the tested scale, so they are
comparable across genes. They are fixed summaries, not tests or prevalence
estimates.

## Implementation

The native Rust kernel and the readable NumPy path perform the same ordered
sums and give identical stopping records; tests check both against an exact
rational-arithmetic oracle that integrates the quantile functions instead.
Genes are processed in bounded chunks, with no genes × permutations matrix.
Sorting is O(G N log N); after that, work is proportional to the shuffles each
gene and direction actually uses, so strongly changed genes dominate runtime.

## Limits

Within one gene, k cases above every control carry evidence worth
C(n₁, k) / C(n₁ + n₀, k), the chance that a random relabelling places all of
them among the cases; when the other cases are no higher than the controls, the
upward p-value stays near or above it, whatever their amplitude. Borrowing information
across genes to judge how unusual an amplitude is would need a model of
transcriptome-wide variation; this release does not attempt it. Batch or depth
confounded with the groups breaks exchangeability, which neither normalization
nor permutation repairs.
