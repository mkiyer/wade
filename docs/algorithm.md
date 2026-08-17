# WADE — algorithm specification

WADE (Wasserstein Area Differential Expression) is a two-group
differential-*distribution* test for per-gene abundance data, built to detect
genes altered in only a fraction of the case group. This document is a
self-contained mathematical specification: it defines every quantity the method
computes, in notation, without reference to any implementation. An
implementation written against this document alone should reproduce the
reference behaviour up to the conventions flagged in
[`r-implementation.md`](r-implementation.md).

Scope. This document specifies **what** is computed. It does not specify
in-memory layout, iteration order, or language-level conventions; those are in
[`r-implementation.md`](r-implementation.md), and the places where two languages
can silently disagree are in [`porting-hazards.md`](porting-hazards.md). The
reasoning behind the design is in [`rationale.md`](rationale.md); what the
method cannot do is in [`limits.md`](limits.md).

Reference source. Every definition below was read from
[`../reference/R/wade.R`](../reference/R/wade.R) and, where a claim needed
checking, verified by running that file in the pinned R sandbox described in
[`../reference/R/README.md`](../reference/R/README.md). Measurements attributed
to "this session" were produced that way and are labelled where they appear.

---

## 1. Setup and notation

The input is a matrix of non-negative abundance values

$$X \in \mathbb{R}_{\ge 0}^{G \times N},$$

with $G$ genes (rows) and $N$ samples (columns), together with a binary
condition vector

$$c \in \{0, 1\}^N,$$

where $c_j = 1$ marks sample $j$ as a **case** and $c_j = 0$ as a **control**.
Write the index sets and their sizes

$$I_1 = \{\,j : c_j = 1\,\}, \quad n_1 = |I_1|, \qquad
  I_0 = \{\,j : c_j = 0\,\}, \quad n_0 = |I_0|,$$

so $n_1 + n_0 = N$. Both groups must be non-empty. The genes are processed
independently and identically; the notation below drops the gene index except
where two genes must be distinguished, and $x_{1} = (X_{g,j})_{j \in I_1}$,
$x_{0} = (X_{g,j})_{j \in I_0}$ denote gene $g$'s case and control value
vectors.

WADE is **one-sided upward** throughout: every statistic is signed
case-minus-control, and every p-value is an upper-tail probability. Genes
depleted in cases carry negative statistics and p-values near 1. This is a
design property, not an oversight — see [`limits.md`](limits.md) §2.4.

### 1.1 Quantile grid resolution

Let

$$m \;=\; \min(n_0, n_1)$$

be the **grid size** (`nprobs` in the reference implementation). The two groups
are compared at $m$ probabilities

$$p_1 > p_2 > \cdots > p_m, \qquad
  p_i = \frac{m-i}{m-1} \quad (m \ge 2), \qquad
  p_1 = 1 \quad (m = 1),$$

that is, the $m$ equally spaced points spanning the closed unit interval,
$\{0, \tfrac{1}{m-1}, \tfrac{2}{m-1}, \ldots, 1\}$, including both endpoints.
The endpoints are included deliberately: $p = 1$ is the group maximum, which is
where a rare high-expressing subset lives.

**Why the smaller group sets the resolution.** With $m = \min(n_0, n_1)$ the
grid nodes coincide exactly with the order statistics of the *smaller* group —
verified in the sandbox: on a sample of size $m$, the type-7 quantiles at these
$m$ probabilities are precisely that sample's sorted values. The smaller group
is therefore read without interpolation, and only the larger group is
interpolated onto the grid. Choosing $m > \min(n_0, n_1)$ would interpolate both
groups and let interpolation, rather than data, carry part of the comparison;
choosing $m < \min(n_0, n_1)$ would discard order statistics the smaller group
supplies. $m = \min(n_0, n_1)$ is the largest grid on which at least one group
is exact.

This makes $m$ a **property of the design, not a tunable parameter**. Its
consequences for what the method can resolve are the subject of
[`limits.md`](limits.md) §3.

### 1.2 Empirical quantile functions

For a value vector $x$ of length $n$ with order statistics
$x_{(1)} \le x_{(2)} \le \cdots \le x_{(n)}$, the empirical quantile function is
the **type-7** (linear-interpolation) definition: for $p \in [0,1]$ set
$h = (n-1)p + 1$ and

$$Q_x(p) \;=\; x_{(\lfloor h \rfloor)} \;+\; \big(h - \lfloor h \rfloor\big)\,
             \Big( x_{(\lceil h \rceil)} - x_{(\lfloor h \rfloor)} \Big).$$

This is the default of both R's `quantile()` and NumPy's `quantile()`, and it is
**part of the specification**: every WADE statistic is a function of these
quantiles, so a different quantile type changes every number the method returns
without raising an error. An implementation must pin type 7 explicitly rather
than inherit it — see [`porting-hazards.md`](porting-hazards.md).

Define the two per-gene quantile vectors and their difference on the grid:

$$Q_1[i] = Q_{x_1}(p_i), \qquad
  Q_0[i] = Q_{x_0}(p_i), \qquad
  D[i] = Q_1[i] - Q_0[i], \qquad i = 1, \ldots, m.$$

$D$ is the **quantile difference profile**, and every statistic in §2 is a
functional of it.

---

## 2. The statistics

### 2.1 `diff.mean` — the bulk axis

The signed area between the two quantile functions,

$$\texttt{diff.mean} \;=\; \int_0^1 \big[Q_{\text{case}}(p) - Q_{\text{ctrl}}(p)\big]\,dp,$$

estimated as the equal-weight (rectangle-rule) average over the grid:

$$\boxed{\;\texttt{diff.mean} \;=\; \frac{1}{m}\sum_{i=1}^{m} D[i]\;}$$

This is the **bulk** axis: a whole-group location shift moves it, and it is the
quantity a conventional mean-difference test also targets.

#### 2.1.1 The identity with $\mu_{\text{case}} - \mu_{\text{ctrl}}$ — exact only when $n_1 = n_0$

**Source discrepancy, corrected here.** Every prose source in this repository
states the identity without qualification.
[`supplement_wade.qmd`](../reference/docs/supplement_wade.qmd) line 30, the v7
narrative
([`../reference/docs/notebook_sections/v7_section9_wade_discovery.md`](../reference/docs/notebook_sections/v7_section9_wade_discovery.md)
line 45), the v8 supplement
([`../reference/docs/notebook_sections/v8_S7_wade_method_supplement.md`](../reference/docs/notebook_sections/v8_S7_wade_method_supplement.md)
line 46) and the header comment of
[`../reference/R/wade.R`](../reference/R/wade.R) (line 14) all write

$$\texttt{diff.mean} \;=\; \int_0^1 \big[Q_{\text{case}}(p) - Q_{\text{ctrl}}(p)\big]\,dp \;=\; \mu_{\text{case}} - \mu_{\text{ctrl}},$$

and `wade.R` line 101 carries the inline comment `# = mu_case - mu_ctrl`. **The
right-hand identity holds exactly when $n_1 = n_0$ and is an approximation
otherwise.** The left-hand equality — signed area of the quantile difference —
is exact in all cases and is the definition; it is the second equals sign that
is conditional.

The reason is quadrature, not the integral. The population statement
$\int_0^1 Q_x(p)\,dp = \mathbb{E}[x]$ is true, but WADE evaluates it as an
$m$-node equal-weight average of a piecewise-linear interpolant. For a group of
size $n$ read on an $m$-node grid:

- **When $m = n$** the nodes coincide with the order statistics, the average is
  $\frac{1}{n}\sum_i x_{(i)}$, and it equals the sample mean exactly.
- **When $m < n$** the grid mean is still a weighted average of the order
  statistics — the interpolation makes it a linear functional of them, with
  weights summing to 1 — but the weights are not uniform. Verified numerically:
  the two **extreme** order statistics each carry weight exactly $1/m$, against
  the $1/n$ the sample mean gives them, and the interior weight is spread over
  the remaining nodes. (For the $n = 5$, $m = 4$ case of §2.7 the weight vector
  is $(\tfrac14, \tfrac16, \tfrac16, \tfrac16, \tfrac14)$ against a uniform
  $\tfrac15$.) The grid mean therefore equals the sample mean only when the
  order statistics are linear in their rank — confirmed in the sandbox to
  $1.4\times10^{-14}$ across 50 random arithmetic-sequence groups at random $n$
  and $m$. Otherwise the over-weighted extremes decide the sign of the gap:
  upward for a right-skewed group, downward for a left-skewed one.

Since $m = \min(n_0, n_1)$, the smaller group is always exact and the larger
group is exact only under the rank-linearity condition above. Hence: **exact when
$n_1 = n_0$; for skewed data, biased in whichever group is larger, with the sign
set by the direction of skew.**

Measured in the sandbox this session, on synthetic
$\mathrm{lognormal}(3, 0.6)$ data at the v7 cohort's geometry (77 cases,
18 controls, $m = 18$), over 400 replicates with no group signal planted:
`diff.mean` exceeded $\mu_{\text{case}} - \mu_{\text{ctrl}}$ by a mean of
$+1.42$ (SD $0.92$) in every one of the 400 replicates. Reversing the roles
(18 cases, 77 controls) reversed the sign to $-1.37$; negating the data, which
makes the groups left-skewed, also reversed the sign. The gap shrinks with
balance — at 77 cases the mean gap was $+1.38$ against 18 controls, $+0.52$
against 33, $+0.22$ against 50, and exactly $0$ against 77.

**This does not invalidate the inference, and the reason matters.** The
permutation null is computed with the same estimator on the same grid, so it
inherits the same bias: measured on 200 null genes at 77-vs-18, the permutation
null of `diff.mean` centred at $+252$ TPM-like units rather than at zero, while
the observed statistics carried a median bias of $+243$ — leaving a median
standardized residual of $+0.016$ null SDs, and a realised type-I error of
exactly $0.050$ at the nominal $0.05$ level. The bias is common to observation
and null and cancels in the p-value. What it does affect is the **effect size**:
`diff.mean` should be read as "signed quantile area on the $m$-node grid", which
is what it is, and not as a drop-in estimate of the mean difference when the
groups are unbalanced. See [`limits.md`](limits.md) §2.5.

An implementation should reproduce the estimator as specified — the grid average
of $D$ — and **not** "fix" it by substituting $\bar{x}_1 - \bar{x}_0$. That
would change the statistic, invalidate parity against the reference, and (since
the null would have to change with it) alter the p-values.

### 2.2 `w1` — the 1-Wasserstein distance

$$\boxed{\;\texttt{w1} \;=\; \frac{1}{m}\sum_{i=1}^{m} \big|D[i]\big|\;}$$

the grid estimate of $\int_0^1 |Q_{\text{case}}(p) - Q_{\text{ctrl}}(p)|\,dp$,
which is the 1-Wasserstein (earth-mover) distance $W_1$ between the two
empirical distributions. It is unsigned and satisfies
$\texttt{w1} \ge |\texttt{diff.mean}|$ by the triangle inequality, with equality
exactly when $D$ does not change sign. The same quadrature caveat as §2.1.1
applies: when $n_1 = n_0$ this reproduces the exact empirical $W_1$
(verified in the sandbox to 6 decimal places at $25$-vs-$25$); when the groups
are unbalanced it is a coarse-grid estimate of it (measured $9.174$ against an
exact empirical $7.685$ at $77$-vs-$18$ on one synthetic gene).

`w1` is reported for description. **No permutation p-value is computed for it**
— inference runs on `diff.mean` and `tail.mean` only.

### 2.3 The tail window

The subset axis reads the upper end of the grid. The window size is

$$\boxed{\;k \;=\; \max\!\big(1,\ \lceil\, q_{\text{tail}} \cdot m \,\rceil\big)\;}$$

with default $q_{\text{tail}} = 0.10$, and the window is the $k$ grid nodes with
the **largest** probabilities, $\{p_1, \ldots, p_k\}$ — i.e. the set
$\mathcal{T} = \{ i : p_i \text{ among the } k \text{ largest} \}$. Because
$p_1 = 1$ is the group maximum, $\mathcal{T}$ always contains the maximum.

The ceiling and the $\max(1, \cdot)$ floor together mean $k \ge 1$ always, so
the statistic is always computable — including in regimes where it should not be
believed. Worked values at $q_{\text{tail}} = 0.10$, computed in the sandbox:

| $m$ | 5 | 10 | 18 | 20 | 22 | 33 | 34 | 60 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| $k$ | 1 | 1 | 2 | 2 | 3 | 4 | 4 | 6 | 10 |

At $m \le 10$ the "tail mean" is a single order statistic. That the formula
still returns a number there is the subject of [`limits.md`](limits.md) §3.

**Ordering is an implementation convention, not mathematics.** The definition
above is in terms of the largest probabilities. The reference implementation
happens to store the grid in *descending* probability order, which makes
$\mathcal{T}$ the first $k$ positions; an implementation storing ascending
probabilities must take the *last* $k$. The mathematical content is "upper
tail"; the indexing that realises it is documented in
[`r-implementation.md`](r-implementation.md), and getting it backwards is a
silent error that produces a lower-tail statistic with no warning.

### 2.4 `tail.mean` — the subset axis

$$\boxed{\;\texttt{tail.mean} \;=\; \frac{1}{k}\sum_{i \in \mathcal{T}} D[i]\;}$$

the mean quantile difference over the tail window. This is the **subset** axis.
A gene elevated in a minority of cases has a small `diff.mean` (the difference
is confined to a few quantiles out of $m$) but a large `tail.mean` (those are
exactly the quantiles the window averages).

Note the normalization: `tail.mean` divides by $k$, whereas `diff.mean` divides
by $m$. `tail.mean` is a mean over the window, not a windowed contribution to
the total.

### 2.5 `tail.conc` — tail concentration

$$\boxed{\;\texttt{tail.conc} \;=\; \frac{\sum_{i \in \mathcal{T}} D[i]}{\sum_{i=1}^{m} D[i]}
   \;=\; \frac{k \cdot \texttt{tail.mean}}{m \cdot \texttt{diff.mean}}\;}$$

the share of the total signed area that falls in the tail window (the second
form verified numerically in the sandbox). A value near 1 marks a gene whose
entire case-control difference is carried by its highest-expressing cases.

Three properties an implementer must handle deliberately:

1. **It is a ratio of signed quantities, so it is not confined to $[0,1]$.**
   Values slightly above 1 are **legitimate**: when the lower quantiles' 
   differences are negative they cancel part of the tail's contribution, the
   denominator shrinks below the numerator, and the tail genuinely carries more
   than the net total. Constructed in the sandbox: a gene with cases below
   controls across the bulk and above them in the top 2 nodes gives
   $\sum_{\mathcal{T}} D = 674$, $\sum D = 512$,
   $\texttt{tail.conc} = 1.316$. **A clamp to $[0,1]$ is therefore the wrong
   fix.**
2. **The denominator can vanish, and the statistic diverges.** Since
   $\sum_i D[i] = m \cdot \texttt{diff.mean}$, `tail.conc` has a pole wherever
   the lower quantiles exactly cancel the upper ones. Bisected in the sandbox on
   a constructed gene: at the root the value is $\pm\infty$; a perturbation
   giving $\texttt{diff.mean} = 5\times10^{-5}$ returns
   $\texttt{tail.conc} = 1.62\times10^{5}$.
3. **The reference guard is far too tight to catch this.** `wade()` returns
   `NA` when $|\texttt{diff.mean} \cdot m| < 10^{-8}$, i.e. when
   $|\texttt{diff.mean}| < 10^{-8}/m$ — about $4.5\times10^{-10}$ at $m = 22$.
   Divergence occurs at `diff.mean` values many orders of magnitude larger.
   Measured on the cfRNA primary contrast, 120 of 2,219 genes (5.4%) return
   $|\texttt{tail.conc}| > 2$, the largest 149. This is a known defect with a
   quantified magnitude; the port's obligation is set out in
   [`porting-hazards.md`](porting-hazards.md) and the failure mode is analysed
   in [`limits.md`](limits.md) §6.

### 2.6 Auxiliary quantities

Let $s_1 = \sum_i Q_1[i]$ and $s_0 = \sum_i Q_0[i]$ be the grid sums. The
reference returns, alongside the four statistics above:

$$\texttt{cond1.mean} = \frac{s_1}{m}, \qquad
  \texttt{cond0.mean} = \frac{s_0}{m}, \qquad
  \texttt{fc} = \frac{s_1}{s_0} = \frac{\texttt{cond1.mean}}{\texttt{cond0.mean}},$$

$$\texttt{tot.mean} = \frac{s_1 + s_0}{m}, \qquad
  \texttt{diff.frac} = \frac{\texttt{diff.mean}}{\texttt{tot.mean}}.$$

Two naming traps, both verified in the sandbox:

- **`tot.mean` is the *sum* of the two group grid-means, not the pooled mean of
  all samples.** $\texttt{tot.mean} = \texttt{cond1.mean} + \texttt{cond0.mean}$
  identically. It is not $\frac{1}{N}\sum_j X_{g,j}$, and it is not the average
  of the two group means. Consequently `diff.frac` is the normalized contrast
  $(\mu_1 - \mu_0)/(\mu_1 + \mu_0)$, which lies in $[-1, 1]$ for non-negative
  data — not a fraction of overall abundance.
- **`fc` is a ratio of grid means, not a fold change of medians or of totals**,
  and it is undefined ($\pm\infty$ or `NaN`) when $s_0 = 0$. Downstream code
  takes $\log_2$ of it (§6), so a zero-abundance control group propagates.

$m$, $k$, $n_1$, $n_0$, and the grids $Q_1$, $Q_0$, $D$ are also returned; the
single-gene diagnostic (§7) and any plotting layer need them.

### 2.7 Worked example

A deterministic fixture, computed in the sandbox, that an implementation can
check against directly. Two genes, 5 cases and 4 controls, so $m = 4$,
$k = \max(1, \lceil 0.4 \rceil) = 1$, and the grid is
$p = (1, \tfrac{2}{3}, \tfrac{1}{3}, 0)$. No jitter or normalization is applied
— these are $Q$/$D$ and the statistics computed directly from the values.

| gene | case values | control values |
|---|---|---|
| `gA` | 10, 12, 14, 16, **60** | 10, 12, 14, 16 |
| `gB` | 20, 22, 24, 26, 28 | 10, 12, 14, 16 |

| | $Q_1$ | $Q_0$ | $D$ |
|---|---|---|---|
| `gA` | 60, 15.3333, 12.6667, 10 | 16, 14, 12, 10 | 44, 1.3333, 0.6667, 0 |
| `gB` | 28, 25.3333, 22.6667, 20 | 16, 14, 12, 10 | 12, 11.3333, 10.6667, 10 |

| statistic | `gA` | `gB` |
|---|---|---|
| `diff.mean` | 11.500000 | 11.000000 |
| `w1` | 11.500000 | 11.000000 |
| `tail.mean` | 44.000000 | 12.000000 |
| `tail.conc` | 0.956522 | 0.272727 |
| `fc` | 1.884615 | 1.846154 |
| `cond1.mean` | 24.500000 | 24.000000 |
| `cond0.mean` | 13.000000 | 13.000000 |
| `tot.mean` | 37.500000 | 37.000000 |
| `diff.frac` | 0.306667 | 0.297297 |
| $\mu_{\text{case}} - \mu_{\text{ctrl}}$ | **9.400000** | **11.000000** |

The fixture exercises three things at once. The two genes are nearly
indistinguishable on the bulk axis ($11.5$ against $11.0$) and separate by a
factor of $3.7$ on the subset axis ($44$ against $12$) — which is the entire
motivation for the shape statistics. And the identity of §2.1.1 fails for `gA`
($11.5 \ne 9.4$) while holding exactly for `gB` ($11.0 = 11.0$): `gB`'s case
values are an arithmetic sequence, hence linear in rank, which is the condition
under which the grid mean of the larger group equals its sample mean.

---

## 3. Optional pre-transformations

Two transformations may be applied to $X$ before §2, both off by default:

- **Control weighting.** For $w \ne 1$, multiply every control column by $w$:
  $X_{\cdot j} \leftarrow w X_{\cdot j}$ for $j \in I_0$. Values $w > 1$ inflate
  controls and so raise the bar a case group must clear (stringency). This
  rescales controls only and is not a symmetric reweighting.
- **Log scaling.** For $\log_2$ mode, $X \leftarrow \log_2(X + 1)$, applied
  after weighting.

Both change every statistic in §2. Neither is used in the reference
configuration, and the reference's fast permutation path does not support them
(§4.3).

---

## 4. Permutation inference

### 4.1 The null

Let $T_g(c)$ denote a statistic for gene $g$ under condition vector $c$.
Inference is by exchangeability of the labels: for $b = 1, \ldots, B$ draw
$c^{(b)}$ as a uniform random permutation of the entries of $c$ (so $n_1$ and
$n_0$ are preserved), and recompute the statistic for every gene.

Two properties of this construction are load-bearing:

- **Both axes are tested independently, on the same shuffles.** `diff.mean` and
  `tail.mean` each get their own null distribution and their own p-value. They
  are not combined into a single statistic, and neither gates the other.
- **One shuffle serves all genes.** Within permutation $b$, the *same*
  $c^{(b)}$ is applied to every gene. The null therefore preserves the
  gene-gene correlation structure of the data, which matters for the
  multiple-testing step in §5. An implementation that drew an independent
  shuffle per gene would be computing a different — and, across correlated
  genes, anti-conservative — null.

### 4.2 The empirical p-value

For observed $t_g = T_g(c)$ and nulls $t_g^{(b)} = T_g(c^{(b)})$:

$$\boxed{\;p_g^{\text{emp}} \;=\; \frac{1 + \#\{\, b : t_g^{(b)} \ge t_g \,\}}{B + 1}\;}$$

The $+1$ in both numerator and denominator counts the observed labelling among
the exchangeable outcomes; it makes the p-value valid (never zero) and bounds it
below by $1/(B+1)$. The comparison is $\ge$, one-sided upward, consistent with
§1.

With $B = 2000$ the floor is $p \ge 1/2001 \approx 5.00\times10^{-4}$; with
$B = 1000$, $\approx 9.99\times10^{-4}$.

### 4.3 What is recomputed per permutation

Only `diff.mean` and `tail.mean` are needed for §4.2, so a permutation pass need
compute only $Q_1$, $Q_0$, $D$ and those two reductions — not `w1`,
`tail.conc`, `fc`, or the auxiliary means. The grid $\{p_i\}$, $m$ and $k$ are
fixed by $n_1$ and $n_0$, which permutation preserves, so they are computed once
and reused. This is a performance specialisation with no effect on the numbers.

The reference restricts this fast path to the default configuration
($w = 1$, no $\log_2$) and falls back to the full computation otherwise; that is
an implementation detail, documented in
[`r-implementation.md`](r-implementation.md).

### 4.4 Conditional inference: the noise draw is outside the null

When the pipeline includes the continuity jitter of §8, the jitter is drawn
**once**, before any permutation, and the labels are then shuffled on that fixed
matrix. Inference is therefore *conditional* on the realised jitter rather than
marginal over it. This is what makes a given seed reproduce a given result
exactly, and the reasoning is in [`rationale.md`](rationale.md) §5.

An implementation must not re-draw the jitter inside the permutation loop.
Doing so would integrate over the noise, change the null, and destroy
reproducibility.

---

## 5. Tail refinement and multiple testing

### 5.1 When refinement triggers

The empirical p-value cannot resolve below $1/(B+1)$, and a gene whose observed
statistic exceeds every null draw is known only to be "below that". For such
genes the upper tail of the null is modelled parametrically
(Knijnenburg et al. 2009, as cited by the sources — see
[`rationale.md`](rationale.md) §6).

Refinement is applied to gene $g$ when **both** hold:

$$\#\{\, b : t_g^{(b)} \ge t_g \,\} \;<\; n_{\text{exc,min}}
  \qquad\text{and}\qquad B \;\ge\; 2\,n_{\text{tail}},$$

with defaults $n_{\text{exc,min}} = 10$ and $n_{\text{tail}} = 250$ (so the
second condition is $B \ge 500$). Genes with $\ge 10$ exceedances keep their
empirical p-value: a well-resolved p-value is never replaced by an extrapolated
one.

### 5.2 The refinement, step by step

Given the observed value $o = t_g$ and the null sample
$\{t_g^{(b)}\}_{b=1}^{B}$:

1. **Tail size.** $n_{\text{tail}} \leftarrow \min(n_{\text{tail}}, \lfloor B/2 \rfloor)$.
2. **Threshold.** Sort the null in decreasing order as
   $s_1 \ge s_2 \ge \cdots \ge s_B$ and set the threshold
   $u = s_{\,n_{\text{tail}}+1}$ — the $(n_{\text{tail}}+1)$-th largest value,
   so that $n_{\text{tail}}$ null draws lie at or above it.
3. **Exceedances.** $e = \{\, s - u : s \in \text{null},\ s > u \,\}$, the strict
   excesses over the threshold. (With ties at $u$, $|e|$ may be smaller than
   $n_{\text{tail}}$.)
4. **Fallbacks to the empirical p-value.** Return $p^{\text{emp}}$ if any of:
   $|e| < 10$; $o \le u$ (the observation is not in the modelled tail);
   $\widehat{v} = \operatorname{Var}(e)$ is not finite or $\le 0$;
   or $\widehat{\sigma} \le 0$ / non-finite from step 6.
5. **Excess of the observation.** $y = o - u$.
6. **Method-of-moments GPD fit.** With $\widehat{\mu} = \operatorname{mean}(e)$
   and $\widehat{v} = \operatorname{Var}(e)$ (the unbiased, $n-1$ denominator
   variance):

   $$\widehat{\xi} = \frac{1}{2}\left(1 - \frac{\widehat{\mu}^2}{\widehat{v}}\right),
     \qquad
     \widehat{\sigma} = \frac{1}{2}\,\widehat{\mu}\left(1 + \frac{\widehat{\mu}^2}{\widehat{v}}\right).$$

   These are closed-form and dependency-free, which is why they were chosen over
   maximum likelihood.
7. **Conditional exceedance probability, with the $\widehat{\xi} \le 0$ branch.**

   $$\Pr\big(Y > y\big) \;=\;
     \begin{cases}
       \exp\!\left(-\dfrac{y}{\widehat{\sigma}}\right), & \widehat{\xi} \le 0,\\[2ex]
       \left(1 + \dfrac{\widehat{\xi}\, y}{\widehat{\sigma}}\right)^{-1/\widehat{\xi}}, & \widehat{\xi} > 0.
     \end{cases}$$

   The upper branch is the standard generalized-Pareto survival function. The
   lower branch is the $\widehat{\xi} \to 0$ **exponential limit**, used instead
   of the GPD form because a GPD with $\widehat{\xi} < 0$ has a hard upper
   bound at $-\widehat{\sigma}/\widehat{\xi}$: an observation beyond it would
   return probability zero, collapsing a strong statistic to a
   machine-epsilon p-value. What this prevents is discussed in
   [`rationale.md`](rationale.md) §6.

8. **Rescale to an unconditional p-value, and floor it.**

   $$\boxed{\;p_g \;=\; \max\!\left(\frac{n_{\text{tail}}}{B}\cdot\Pr(Y > y),
     \;\; \frac{1}{B \cdot n_{\text{tail}}}\right)\;}$$

   The factor $n_{\text{tail}}/B$ is the estimated probability of exceeding the
   threshold $u$, converting a conditional tail probability into an
   unconditional one. Note it uses $n_{\text{tail}}$, not $|e|$; with ties these
   differ, and the reference's choice is $n_{\text{tail}}$.

   The floor $1/(B \cdot n_{\text{tail}})$ is the **extrapolation resolution
   limit** — $2.0\times10^{-6}$ at $B = 2000$, $n_{\text{tail}} = 250$;
   $4.0\times10^{-6}$ at $B = 1000$. It is a deliberate honesty constraint, not
   a numerical convenience: a port that "improves" it by returning smaller
   p-values is a regression. How to read a p-value sitting *at* this floor is
   the worked example in [`limits.md`](limits.md) §5.

A sanity check on the fit, run in the sandbox: on $B = 2000$ draws from a
standard exponential null (true $\xi = 0$), step 2 gives $u = 2.129$ with
exactly 250 exceedances, and the moment estimators return
$\widehat{\xi} = -0.011$, $\widehat{\sigma} = 1.019$ — correctly landing on the
$\widehat{\xi} \le 0$ exponential branch with a scale near the truth.

### 5.3 BH-FDR

Benjamini–Hochberg adjustment is applied **separately to each axis**, across all
$G$ genes: `padj.diff` from the $G$ values of `p.diff`, `padj.tail` from the $G$
values of `p.tail`. The two axes are not pooled into one family of $2G$ tests.

For sorted p-values $p_{(1)} \le \cdots \le \cdots \le p_{(G)}$, the adjusted
values are the standard step-up

$$\tilde{p}_{(i)} = \min_{j \ge i} \left\{ \min\!\left(1, \frac{G}{j}\, p_{(j)}\right) \right\},$$

i.e. the cumulative minimum from the largest p-value downward, restored to the
original gene order. This is what R's `p.adjust(method = "BH")` computes; the
equivalent call in other stacks must be pinned rather than assumed — see
[`porting-hazards.md`](porting-hazards.md).

---

## 6. Rank scores

Two per-gene scores are computed from the result frame, for **nomination, not
inference**. Let $F_{\text{fc}}$, $F_{\text{case}}$, $F_{\text{ctrl}}$,
$F_{\text{tail}}$ denote the empirical CDFs, taken across the $G$ genes, of
$|\log_2 \texttt{fc}|$, `cond1.mean`, `cond0.mean` and `tail.mean`
respectively. With exponents $a, b, d, e$ (all default 1):

$$\texttt{score} \;=\; \operatorname{sign}(\texttt{diff.frac}) \cdot
   F_{\text{fc}}\big(|\log_2 \texttt{fc}|\big)^{a} \cdot
   F_{\text{case}}(\texttt{cond1.mean})^{b} \cdot
   \big(1 - F_{\text{ctrl}}(\texttt{cond0.mean})\big)^{d},$$

$$\texttt{tail.score} \;=\; \operatorname{sign}(\texttt{tail.mean}) \cdot
   F_{\text{tail}}(\texttt{tail.mean})^{e} \cdot
   F_{\text{case}}(\texttt{cond1.mean})^{b} \cdot
   \big(1 - F_{\text{ctrl}}(\texttt{cond0.mean})\big)^{d}.$$

Each rewards a gene that is high in cases and low in controls, `score` on the
bulk axis via fold change and `tail.score` on the subset axis via the tail
statistic. An empirical CDF returns values in $(0, 1]$, so both scores lie in
$[-1, 1]$. Ranks are then dense ranks of each score in descending order (ties
share a rank, and the next rank is consecutive).

These are **heuristics**, and the framing must travel with them: a rank of 1 is
not a claim of significance. Why nomination fell back to rank in the source
project, and whether these scores belong in a general-purpose package at all, is
discussed in [`rationale.md`](rationale.md) §8 and left open in
[`design-decisions.md`](design-decisions.md).

---

## 7. Single-gene diagnostic

For one gene, the diagnostic output is the grid in **ascending** probability
order together with the cumulative signed area. Let $\pi(i)$ reindex so that
$p_{\pi(1)} < p_{\pi(2)} < \cdots < p_{\pi(m)}$ (i.e. reverse the grid of §1.1),
and write $\tilde{Q}_1$, $\tilde{Q}_0$ for the correspondingly reordered
quantile vectors. Then for $r = 1, \ldots, m$:

$$\boxed{\;\texttt{cum}[r] \;=\; \frac{1}{m}\sum_{i=1}^{r}
   \big(\tilde{Q}_1[i] - \tilde{Q}_0[i]\big)\;}$$

The reversal must happen **before** the accumulation, and the division is by $m$
(the full grid size) at every $r$, not by $r$. Two consequences define the plot:

- The curve reads left to right in increasing quantile.
- Its endpoint satisfies the **exact identity**
  $\texttt{cum}[m] = \texttt{diff.mean}$. Verified in the sandbox to
  floating-point equality; the reference's own smoke test checks it. This is the
  cheapest correctness test an implementation has, and it should be asserted.

The reading: a broad location shift accumulates steadily across all quantiles; a
rare high-expressing subset stays flat and then climbs inside the tail window; a
single-outlier gene stays flat and then spikes at the last node. The second and
third are not distinguishable from this curve alone — separating them is the
permutation p-value's job ([`rationale.md`](rationale.md) §7).

---

## 8. Optional normalization layer

The reference ships a normalization step ahead of §2. It is specified here for
completeness; whether it belongs inside a method package is an open question
covered in [`design-decisions.md`](design-decisions.md), and its status as a
second normalization decision living inside a test is a stated limitation
([`limits.md`](limits.md) §2.3).

Inputs are a raw count matrix $C \in \mathbb{R}_{\ge 0}^{G \times N}$ and a
**normalizer** $L$, which is either a per-gene vector of length $G$ (broadcast
across samples — e.g. intron count) or a full $G \times N$ matrix (e.g.
per-library effective length). Then:

1. **Library sizes.** $\displaystyle \ell_j = \sum_{g=1}^{G} \frac{C_{gj}}{L_{gj}}$.
2. **Jitter.** Draw $\eta_{gj} \sim \mathrm{Uniform}(0, \nu)$ independently, once,
   under a fixed seed, with default $\nu = 0.01$. This is drawn at **count**
   precision — before division by $L$ — so it breaks ties in zero-heavy data
   without materially moving non-zero values.
3. **Scaled values.**

   $$X_{gj} \;=\; \kappa \cdot
     \frac{\big(C_{gj} + \eta_{gj}\big)\big/L_{gj}}
          {\ \ell_j + \eta_{gj}/L_{gj}\ },$$

   with $\kappa = 10^6$. The jitter appears in the denominator as well as the
   numerator; without the normalizer and jitter this reduces to
   $\kappa \, (C_{gj}/L_{gj}) / \ell_j$, i.e. standard TPM when $L$ is effective
   length.

The seeded, once-only draw is what §4.4 depends on.

---

## 9. Default parameters

| parameter | default | role |
|---|---|---|
| $q_{\text{tail}}$ | `0.10` | upper-tail fraction defining the subset window |
| $B$ (permutations) | `1000` in the core driver, `2000` in the convenience wrapper | null resolution |
| $\nu$ (jitter) | `0.01` | continuity jitter width, at count precision |
| $\kappa$ | `1e6` | TPM-like scale factor |
| seed | `1` | jitter draw; permutations use a distinct offset stream |
| $n_{\text{exc,min}}$ | `10` | exceedance count below which refinement triggers |
| $n_{\text{tail}}$ | `250` | null draws entering the GPD fit |
| $w$ (control weight) | `1` (off) | control-column upweighting |
| $\log_2$ | off | pre-transformation |

**The two permutation defaults differ, and this is in the source.** The core
driver defaults to $B = 1000$ while the convenience wrapper defaults to
$B = 2000$; the source project's production constant was 2000, and its
validation simulations ran at 1000. A port should pick one default and state it,
because $B$ sets both the empirical floor $1/(B+1)$ and the extrapolation floor
$1/(B \cdot n_{\text{tail}})$ — it changes every small p-value, not just the
runtime.

### 9.1 Consumer-layer thresholds

These are **not** method parameters — they are the source project's decisions
about when to believe the method's output, taken from
[`../reference/R/downstream/wade_contrasts.R`](../reference/R/downstream/wade_contrasts.R)
(lines 113–130, constants defined at 116, 121, 126, 129 and 130). They are
recorded here because their *rationale* is arithmetic
on WADE's own parameters and generalizes even though the numbers do not.

| constant | value | rationale as stated in the source |
|---|---|---|
| minimum smaller-group size to run at all | 10 | $\lceil 0.10 \times 10 \rceil = 1$ — a single tail point, so 10 is the floor for running, not for believing |
| minimum smaller-group size to believe the subset axis | 20 | 20 gives a 2-point window; 30 gives 3 |
| maximum Cramér's V (group × design version) | 0.50 | conventional "strong association" mark; above it the contrast is refused as design-confounded rather than corrected |
| permutations | 2000 | production setting |
| $q_{\text{tail}}$ | 0.10 | production setting |

The first two encode the fact that $m$ is a design property: any implementation
exposing $q_{\text{tail}}$ should be able to tell a caller how many order
statistics that buys. See [`limits.md`](limits.md) §3.

---

## 10. Reference algorithm, end to end

Given $C$, $L$, $c$, and parameters:

1. Compute $\ell$ and the normalized matrix $X$ (§8), drawing the jitter once
   under the seed. If the caller supplies an already-comparable matrix, skip.
2. Compute $m$, the grid $\{p_i\}$, and $k$ from $n_1$, $n_0$, $q_{\text{tail}}$
   (§1.1, §2.3).
3. Compute observed $Q_1$, $Q_0$, $D$ and the statistics of §2, for all genes.
4. For $b = 1, \ldots, B$: shuffle $c$, recompute `diff.mean` and `tail.mean`
   for all genes on the fixed $X$ (§4.3), and store both null rows.
5. Per axis and per gene: empirical p (§4.2), then GPD refinement where the
   gate of §5.1 fires.
6. Per axis: BH adjustment across genes (§5.3).
7. Optionally: rank scores (§6).

Returned per gene: `diff.mean`, `diff.frac`, `w1`, `tail.mean`, `tail.conc`
(`NA` under the guard of §2.5), `fc`, `cond1.mean`, `cond0.mean`, `tot.mean`,
`p.diff`, `p.tail`, `padj.diff`, `padj.tail`; plus `log2fc`, `score`,
`tail.score`, `rank`, `tail.rank` if §6 is applied.

When $B = 0$ the statistics are returned with p-values absent.

---

## 11. Validation targets

The reference behaviour is pinned by three simulations in
[`../reference/R/validation_sims_v7.R`](../reference/R/validation_sims_v7.R),
which run standalone. An implementation should reproduce them qualitatively;
exact numerical agreement is **not** achievable across languages because the
random number generators differ ([`porting-hazards.md`](porting-hazards.md)).

1. **Null calibration.** Counts with no group signal; both axes' p-values should
   be $\mathrm{Uniform}(0,1)$ and the realised type-I error should sit at
   nominal.
2. **Subset-vs-bulk discrimination.** Planted whole-group shifts and planted
   minority-subset genes should separate on `tail.mean` even where they do not
   on `diff.mean`.
3. **Power against the combinatorial floor.** Detection power as the altered
   fraction of cases is swept, against the exact floor
   $\binom{n_1}{k}\big/\binom{n_1+n_0}{k}$.

The measured values from the reference sandbox, and what they do and do not
establish, are reported in [`rationale.md`](rationale.md) §9 and
[`limits.md`](limits.md) §4.

Three cheap invariants are worth asserting in any implementation, all verified
in the sandbox this session:

- $\texttt{cum}[m] = \texttt{diff.mean}$ to floating-point equality (§7).
- $\texttt{w1} \ge |\texttt{diff.mean}|$ always (§2.2).
- $\texttt{tot.mean} = \texttt{cond1.mean} + \texttt{cond0.mean}$ identically
  (§2.6).

And two degenerate cases an implementation must handle deliberately:

- **$m = 2$** — the grid is $\{1, 0\}$, so `diff.mean` is the difference of the
  two mid-ranges and $k = 1$. Confirmed against the reference for both
  single-gene and multi-gene input.
- **$m = 1$** — the grid is the single point $p = 1$, so mathematically
  `diff.mean` = `tail.mean` = `w1` = the difference of group maxima, and
  `tail.conc` $\equiv 1$. **The reference computes this correctly for
  single-gene input only.** Verified in the sandbox this session: with one gene
  it returns the difference of maxima as specified, but with five genes and
  $m = 1$ it returns a length-1 statistic for a five-row input, which the driver
  then recycles so that all five genes report the same value — no error, correct
  output shape, wrong answer. The cause is a matrix-reshape guard, diagnosed in
  [`r-implementation.md`](r-implementation.md) and treated as a parity hazard in
  [`porting-hazards.md`](porting-hazards.md).

  A port will not reproduce that defect and should not be "fixed" toward it. The
  specification above is what $m = 1$ means; refusing $m = 1$ outright is also
  defensible, since a one-sample group has no quantile function worth comparing
  ([`limits.md`](limits.md) §3).
