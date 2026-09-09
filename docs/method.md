# The WADE method

WADE compares two groups' whole distributions, gene by gene, and answers two
questions that ordinary differential expression collapses into one:

1. **Is there a difference?**
2. **What kind of difference is it?** — a shift affecting every sample, or a
   pronounced change confined to a subset of one group.

The second question is the point. A gene altered in 5% of cases and a gene
shifted 2× in all of them can produce the same mean difference, and a
first-moment test cannot tell them apart. Nothing here requires the user to
declare in advance which they are looking for.

> **Scope.** WADE is for **discrete count data**. The entry point takes raw
> counts, the continuity jitter is applied at count precision (§7), the subset
> stage's null is built by thinning reads (§9.3) and the log-ratio curve
> carries a one-count pseudocount (§9.4) — none of which means anything for a
> continuous measurement. Continuous input is neither tested nor tuned for.

Sections 1–8 define the statistic and its inference; §9 is the derivation
of why stage 2 works on counts the way it does; §10 is what the method cannot
do. Every number here was measured, not reasoned.

---

## 1. The quantile grid

For a gene with case values $x_1$ and control values $x_0$, let

$$m = \min(n_0, n_1)$$

and compare the two groups at $m$ probabilities running **high to low**,
$p = (1, \ldots, 0)$. Quantiles are **type 7** (linear interpolation), pinned
explicitly rather than inherited — nine conventions exist and a different one
changes every number without raising anything.

Two curves follow, and everything else is a functional of one of them:

$$D(p) = Q_1(p) - Q_0(p) \qquad\text{(absolute difference)}$$
$$R(p) = \log_2 Q_1(p) - \log_2 Q_0(p) \qquad\text{(log-ratio)}$$

**Why $m = \min(n_0, n_1)$.** At this size the grid nodes coincide with the
order statistics of the *smaller* group, so it is read essentially without
interpolation and only the larger group is interpolated. It is the largest grid
on which at least one group is exact. This makes $m$ a **property of the design,
not a parameter** — no argument raises it, and it is the resolution limit on
everything below.

**The cap: `max_probs`.** At large cohorts the design's grid becomes a
liability rather than a resolution — at 40,000 per group every per-gene curve
is the size of the data matrix, and nothing needs an affected fraction
resolved to 1/40,000. The realized grid is therefore

$$m = \min(n_0,\; n_1,\; \texttt{max\_probs})$$

with `max_probs = 2000` by default, so **every design at or under 2,000 per
group — including everything this document's measurements were made on — is
untouched.** Above it, the grid is capped and two things change, both
measured:

* `affected_fraction`'s resolution becomes $1/\texttt{max\_probs}$. The
  fidelity rule is $m \gtrsim 2.5 / \pi_{\min}$ for the smallest fraction of
  interest $\pi_{\min}$: at $m = 1000$ the estimate is faithful to a 0.1%
  subset (measured drift $\le 0.0003$ down to that fraction), and a global
  shift survives $m = 100$.
* `mean_shift`'s quadrature coarsens, and the error is **one-sided**: a
  uniform-in-$p$ grid gives the extreme node weight $1/m$ while its value is
  large, so the number inflates for concentrated signals (17.5 → 21.2 at
  $m = 100$ for a 5% subset at 8×). **Inference is unaffected** — the
  permutation null inherits the same quadrature and the inflation cancels in
  the p-value — but on a capped grid `mean_shift` should be read as signed
  quantile area, not a mean-difference estimate, and the §2 balanced-design
  identity holds only approximately. `stage1="saddlepoint"` (§6) is exempt: it
  computes the group means directly and never reads the grid.

The cap is never applied silently: the realized $m$ is `result.nprobs` and is
recorded with `max_probs` in `result.params` and the manifest. `max_probs=None`
removes it.

**Why the descending order is load-bearing.** Position 0 is the group maximum.
An implementation that builds an ascending grid and keeps the same indexing
computes a lower-tail statistic under an upper-tail name, silently.

**Why two curves.** $D$ is the natural scale for *detection*: the area between
the quantile functions is the 1-Wasserstein distance, and it is what the
permutation test is powerful on. $R$ is the natural scale for
*characterization*: a global fold change is a **flat** $R$ regardless of its
magnitude, whereas on the absolute scale a 2× shift produces a $D$ that is
itself concentrated at high quantiles and is indistinguishable from a subset
effect. Measured: the same characterization statistic reads a pure 2× shift as
0.70 on the $D$ scale and 0.98 on the $R$ scale.

---

## 2. Stage 1 — detection

$$\texttt{mean\_shift} = \frac{1}{m}\sum_p D(p)$$

The signed area between the quantile functions. Because
$\int_0^1 Q(p)\,dp = E[X]$ is an identity, that area *is* the difference of
the two sample means, and the grid is a quadrature of it: exact when
$n_1 = n_0$, and on unequal groups an interpolation that over-weights the
extremes of the larger group. Inference is unaffected, because the permutation
null inherits the same bias and it cancels in the p-value.

`stage1="saddlepoint"` computes the identity instead of the quadrature, which
buys an exact p-value with no permutations and no floor (§6). The two are not
interchangeable: off balance the quadrature is a *trimmed* summary and is the
more powerful statistic — measured at 2–4× when $n_1 > n_0$ — while the
saddlepoint still wins end to end because the floor costs more than the
trimming buys (`pvalue-review.md` §5.5). Under `stage1="saddlepoint"` this
paragraph's quadrature is not computed at all.

**This is the ordinary test**, and naming it plainly matters: it is what any
conventional DE method already computes, and WADE's claim is not to improve on
it. Measured, it is the *best* single detector for weak diffuse effects (10% of
cases at 2×: power 0.700, against 0.295 for the shape test). WADE's claim is
that it adds sensitivity the mean test lacks for strong concentrated effects
(2% of cases at 8×: 0.700 against 0.935).

---

## 3. Stage 2 — the shape test

If the difference were a pure global shift, the running total of $R$ would grow
*proportionally* — the top 20% of quantiles would carry 20% of it. Measure the
departure from proportionality:

$$S_k = \sum_{i \le k} R_i, \qquad
  \boxed{\;B_k = S_k - \frac{k}{m}S_m\;}$$

$B$ is identically zero under a pure global fold change and positive when the
difference is front-loaded. It is a discrete Brownian bridge, orthogonal to the
total by construction, so it is not re-testing stage 1 — measured correlation
with the log fold change under the null is **+0.02**.

The statistic scans every window width and takes the most surprising one:

$$T = \max_k \frac{B_k - \mu_k}{\sigma_k}$$

with $\mu_k, \sigma_k$ the null moments of $B_k$. ($B_m \equiv 0$ carries no
information, so the scan runs over $k < m$.) The **argmax** is exposed as a
diagnostic but is *not* a fraction estimate: it tracks the affected fraction
when the subset is shifted multiplicatively and underestimates it several-fold
when the subset's values are replaced outright, because the log-ratio curve then
declines steeply across the affected region and the cumulative departure peaks
before it ends. Measured total absolute error against planted fractions across
both signal shapes: 0.315 for the argmax against 0.084 for `affected_fraction`. **Threshold-free by
maximization rather than by choosing.** This is the direct answer to COPA's
defect: COPA required a percentile cutoff, so it had to be run at several and
left the user holding several answers. The scan runs at all of them and the
permutation null prices in the multiplicity of having looked.

### The null is a global shift, not "no difference"

This is the part that makes stage 2 work, and getting it wrong is not subtle.

Permutation generates the null of *no difference at all*. Stage 2's null is
*a pure global shift* — the mean shift is the hypothesis being argued against.
Permuting data that contains a real shift produces groups that are mixtures of
shifted and unshifted samples, whose spread does not match the shift model, so
the standardization's denominator comes out too small.

Measured, that error is large: scanning $B$ against an ordinary permutation null
fires on **14–19%** of genuine global fold changes.

The fix is to make the two groups exchangeable **under the fitted global
shift** and permute *that*. For continuous data the operation would be a
division: $B$ is exactly invariant to a global fold change ($R \to R - c$
sends $S_k \to S_k - kc$ and leaves $B_k$ unchanged), so the observed
statistic needs no adjustment and only the null does. For counts a division
does not make the groups exchangeable at low expression, and §9.3 replaces
it with binomial thinning of the raw counts, which is what `wade()` does; the
observed statistic is then read off the thinned matrix too. The reasoning is
the same either way, and it is the whole point: stage 2's null is a fitted
global shift, not no-difference.

Measured effect of the correction on continuous data (lognormal, 500 v 500),
with the shift estimated by the median of $R$:

| | ordinary permutation null | shift-corrected null |
|---|---|---|
| global fc = 1.5 | 0.145 | **0.040** |
| global fc = 2 | 0.190 | **0.045** |
| global fc = 8 | 0.105 | **0.050** |
| subset 2% (power) | 0.815 | **0.830** |

Exact nominal level at every fold-change magnitude, at no cost in power.

### What the shape test actually claims

**"A global shift does not explain this."** That is broader than "a subset is
higher", and the difference is documented rather than hidden:

- A **pure variance increase** — same median, wider spread — fires it at 1.000.
  That is not a false positive; it genuinely is not a global shift, and
  heterogeneity in one group is a real finding. §4 separates it.
- A **downward subset** fires it at 1.000, where the mean test sees nothing at
  all (0.000). WADE was previously one-sided upward and blind to genes *lost*
  in a subset of cases; the shape test sees them for free, and §4 reports the
  direction so this is never ambiguous.

---

## 4. Characterization

Three parameter-free numbers, all read off $R$: how much of the group
differs, which way, and by how much.

### `affected_fraction` — how much of the group differs

$$\boxed{\;\hat{\pi} = \frac{\left(\sum_p R(p)^2\right)^2}{m \sum_p R(p)^4}\;}$$

A participation ratio. For a **step** — fraction $\pi$ of cases shifted, the
rest untouched — this equals $\pi$ exactly. For a curve **flat** at
$\log_2(\mathrm{FC})$ it equals 1. It answers the question a biologist actually
asks: *what fraction of cases is this gene altered in?*

**Why the fourth moment.** A broad low-level noise floor of height $\varepsilon$
against signal $h$ contributes $\varepsilon/h$ to the second-moment form and
$(\varepsilon/h)^2$ to this one. The log scale is what anchors a global change
at 1.0, but its unaffected quantiles carry enough sampling noise to swamp a 2%
signal; the quartic form suppresses that floor and keeps the anchor. Measured,
the second-moment version reads a 2% subset as 0.23 and this one as 0.025.

### `direction` — which way

$$\boxed{\;\texttt{up\_share} = \frac{\sum_p \max(R(p), 0)}{\sum_p |R(p)|}\;}$$

Bounded $[-1, +1]$: $+1$ every part moved up, $-1$ every part moved down, $0$ upward and downward movement exactly balancing.

Together the pair is a complete description. Measured at 500 v 500:

| gene | `affected_fraction` | `direction` | reads as |
|---|---|---|---|
| global fc = 2 | 0.978 | +1.000 | everything, up |
| global fc = 0.5 | 0.979 | −1.000 | everything, down |
| variance × 1.6 | 0.316 | −0.010 | two-sided spread |
| subset 5% up | 0.053 | +0.916 | 5% of cases, up |
| subset 5% down | 0.053 | −0.928 | 5% of cases, down |
| subset 5% up + 5% down | 0.095 | −0.006 | 10% total, split |

A symmetric variance change and a one-sided 30% subset both give
$\hat{\pi} \approx 0.3$ and are separated by `direction` (0.50 against ~1.0).

### `subset_log2_fc` — by how much

$$\texttt{subset\_log2\_fc} = \frac{1}{k}\sum_{\text{top }k} R(p),
\qquad k = \lceil \hat{\pi}\, m \rceil$$

— the mean of the log-ratio curve over the region `affected_fraction` names
(the *bottom* $k$ nodes when `direction` is negative). This is the magnitude
the subset test itself deliberately lacks: `p_subset` says a global shift is
inadequate, $\hat{\pi}$ says how much of the group departs, and this says **by
how many folds** — a 10% subset at a modest 3× and one at 100× can share a
saturated p-value and differ here by ~3.5 log2 units. No threshold enters:
the region width is the data's own $\hat{\pi}$.

Read it knowing what it compares: the affected cases against the **controls'
own quantiles at those extremes**, not against the control mean. A subset
planted at 8× the mean of an NB(50) reads ~2 rather than $\log_2 8 = 3$,
because the controls' top decile already sits well above their mean; a
subset that barely clears the controls' natural tail reads near 0 however
far above the control *mean* it is. That discounting of the control spread
is what makes the number a measure of how *distinctive* the subset is.
Boundary behaviour is coherent: a global change has $\hat{\pi} \approx 1$, so
this becomes the gene's overall log2 fold change, and the number includes
any global shift the gene also carries (the fitted shift is reported
separately, so the subset's excess over it is one subtraction away). For a
balanced up-and-down mixture (`direction` $\approx 0$) one signed number is
the wrong shape, as it is for `direction` itself.

### Composition moves both, and that is not a defect

Library-size normalization couples genes. A matrix in which a substantial
fraction of genes are strongly up in cases inflates the case libraries, which
pushes every *other* gene down — and because the offset is systematic while the
sampling noise is not, at large $n$ it takes very little signal to dominate.
Measured at 400 v 400, five strongly-up genes among 205 were enough to move the
null genes' `direction` from ~0.5 to near 0.

Consequences to keep in view when reading a characterization: null genes
acquire a small consistent fold change, so their `affected_fraction` drifts toward 1
(they genuinely *are* globally shifted, relative to the library) and their
`direction` collapses toward 0 or 1. **The shape test itself is unaffected** —
the bridge is invariant to exactly this kind of global offset — but the two
descriptive statistics are not. This is a property of normalized data that
affects any differential method, not something WADE introduces.

### Resolution limit, stated rather than hidden

The grid has $m$ points, so **no fraction finer than $1/m$ is resolvable.**
Measured: quantitative above $m \approx 100$, degrading through $m \approx 50$,
and below that only qualitative — global (0.89–0.94) still separates from
concentrated (0.12–0.27), but 2% and 5% become indistinguishable.

| geometry | $m$ | 2% | 5% | 10% | 25% | global |
|---|---|---|---|---|---|---|
| 2000 v 2000 | 2000 | 0.021 | 0.052 | 0.108 | 0.280 | 0.997 |
| 500 v 500 | 500 | 0.028 | 0.054 | 0.106 | 0.274 | 0.992 |
| 100 v 100 | 100 | 0.080 | 0.080 | 0.121 | 0.279 | 0.957 |
| 77 v 18 | 18 | 0.131 | 0.117 | 0.146 | 0.269 | 0.890 |

This is the resolution limit §10 states, surfacing honestly instead of
being absorbed by a rounding rule.

---

## 5. Reading the output

There is **no categorical label**. These are continuous statistics and turning
them into classes would need thresholds, which is exactly what the redesign
removed. The table is how to read them, not a function the package provides.

| `p_mean_shift` | `p_subset` | interpretation |
|---|---|---|
| significant | — | global shift; an ordinary DE method finds this too |
| significant | significant | concentrated in a subset strong enough to move the mean — `affected_fraction` says how much, `direction` which way |
| — | significant | distributional change with no net mean shift: a balanced subset, or a variance change |
| — | — | not differential |

---

## 6. Inference

**Permutation of the labels**, with two properties that are load-bearing:

- **One shuffle serves all genes.** Within a permutation the same label vector
  applies to every gene, so the null preserves the gene–gene correlation
  structure. Drawing an independent shuffle per gene would give a different and,
  across correlated genes, anti-conservative null.
- **The continuity jitter is drawn once, before any permutation** (§7), so
  inference is *conditional* on one realised noise draw. That is what makes a
  seed reproduce a result exactly. It must never be re-drawn inside the loop.

**Restricted permutation** (`strata=`) narrows the exchange to within strata,
which is the right null when the cohort is assembled from studies, batches or
protocols: shuffling labels across them tests exchangeability the design does
not have. The cost is arithmetic and must be looked at, not assumed away —
unrestricted, the space is $\binom{n}{n_1}$; restricted, it is
$\prod_s \binom{n_s}{k_s}$, and any stratum holding a single class
contributes a factor of one. `wade.permutation_space()` returns the realized
space and the p-value floor it implies, and the manifest records both
(§10.2).

**The empirical p-value** is $(1 + \#\{t^{(b)} \ge t\})/(B+1)$ — the add-one
form, which counts the observed labelling among the exchangeable outcomes,
cannot be zero, and is bounded below by $1/(B+1)$.

**Generalized Pareto refinement.** Where a gene exceeds nearly every null draw,
the empirical p-value has run out of resolution. The upper tail of the null is
then fitted by a GPD (method of moments, closed-form and dependency-free) and
the p-value read from it. Refinement fires only when a gene has fewer than 10
exceedances **and** $B \ge 2 n_{\text{tail}}$ — with the default
$n_{\text{tail}} = 250$ that means $B \ge 500$, so below 500 permutations no
refinement ever happens and the floor is $1/(B+1)$.

The refined p-value is floored at $1/(B \cdot n_{\text{tail}})$. **This is an
honesty constraint, not a numerical guard**: it is what $B$ permutations can
support. A port that returns smaller p-values by "improving" it is a regression.

**Exact stage-1 inference** (`stage1="saddlepoint"`, opt-in). The mean
difference is a monotone function of the case-group subset sum, so its
permutation distribution is the distribution of a subset sum drawn without
replacement — which Skovgaard's double saddlepoint evaluates directly, with no
permutations and no tail model. With $A$ the case subset sum and $V$ the total,
$t = A(1/n_1 + 1/n_0) - V/n_0$; sampling without replacement is handled by
conditioning independent Bernoulli selectors on $\sum Z = n_1$, giving
$K(s,t) = \sum_i \log(1 + e^{s v_i + t})$ and a two-equation saddlepoint.
**Balance is not required** — only that the statistic be a subset sum, which
the exact mean difference is at every geometry and the grid quadrature is not.

Validated against $10^8$ brute-force permutations on four geometries, on
negative-binomial counts through the real normalization, at 52% zeros, and at
300 v 300: median $\hat p/p$ between 0.91 and 1.15 down to $10^{-7}$, against
3–4,906× for the GPD path on the same genes. The floor drops from $1/(B
\cdot n_{\text{tail}})$ to $1/\binom{n}{n_1}$, which is 9.3e-24 at 40 v 40.
It costs about 60× the stage-1 permutation loop it replaces — 0.7 min at
40 v 40 and 50 min at 3,000 v 3,000 for 20,000 genes — and needs SciPy, which
is otherwise optional. Stage 2 still permutes, so the loop runs either way.
**Stage 2 has no equivalent**: its statistic is a maximum over widths and is a
subset sum at no geometry. See `docs/pvalue-review.md`.

**BH-FDR** is applied across genes, separately per axis.

**Permutation z-scores** (`z_mean_shift`, `z_subset`): the observed statistic
standardized against its own gene's null,
$(t - \bar{t}^{(\cdot)})/\mathrm{sd}(t^{(\cdot)})$ — the analogue of GSEA's
normalized enrichment score, in z form because stage 1's statistic is signed.
They exist for **ranking**: on large cohorts thousands of genes tie at the
p-value floor while their separations from the null differ by orders of
magnitude, and the z keeps ordering them at no extra cost, since the null
matrix is already in hand. They are *not* calibrated tail probabilities and
must not be pushed through a normal CDF — the null of a maximum statistic is
not normal, and honesty about the tail is the floor's job. Calibrated
resolution beyond the floor is the open problem of `pvalue-review.md` and
`ISSUES.md` item 1.

### The combinatorial floor

If $k$ samples carry a signal, label shuffling places all of them in the case
group with probability $\binom{n_1}{k}/\binom{n_1+n_0}{k}$. **That is the
scale of the smallest p-value a label-permutation test can resolve**, whatever
the effect size, the detector, or the permutation count — exact in the
noise-free limit, and within a factor of about two of the median gene in
measured count data (§10.4). At 77 cases vs 18 controls it is
0.173 at 10% affected — so nothing below roughly 25% is detectable at
$\alpha = 0.05$, by any method in this family. This dominates detector choice at
small $n$ and is the single most important thing to check before running.

---

## 7. Normalization

The entry point takes **raw counts**, because the continuity jitter is applied
at count precision *before* division and a pre-normalized matrix cannot
reproduce it.

$$X_{gj} = \kappa \cdot
  \frac{(C_{gj} + \eta_{gj}) / L_{gj}}{\ell_j + \eta_{gj}/L_{gj}},
  \qquad \eta \sim \mathrm{Uniform}(0, \nu)$$

The jitter breaks ties in zero-heavy data, where many samples share a count of
zero and the quantile grid would degenerate into flat runs; it also makes every
value strictly positive, so fold changes stay finite.

**The per-cell denominator** adds only *this gene's* jitter contribution to
the library size, not the whole column's, so it is the library size in a
counterfactual where gene $g$ alone received jitter. Consequences: column
sums are not exactly $\kappa$ (the output is TPM-*like*), and any gene that
constitutes an entire library normalizes to exactly $\kappa$ — including
every gene in an all-zero sample, which is why `wade()` refuses a sample with
a zero library size rather than analysing that constant.

`tpm_like` is the only normalizer the package ships, and it is the definition
the golden fixtures pin. A caller who wants a consistent library size supplies
one through `lib_sizes=`; CPM is `normalizer=1.0`; any other size factors go
in the same way.

---

## 8. Defaults

| parameter | default | role |
|---|---|---|
| $B$ (`nperms`) | 2000 | null resolution; sets both p-value floors, $1/(B+1)$ and $1/(B\,n_{\text{tail}})$ |
| `alternative` | `two-sided` | detect differences in either direction; `greater` / `less` restrict it |
| $\nu$ (`noise`) | 0.01 | continuity jitter width, in counts |
| $\kappa$ (`norm_factor`) | 1e6 | TPM-like scale factor |
| `pseudocount` | 1 | one count, in each sample's normalized units, before the log (§9.4) |
| `max_probs` | 2000 | cap on the quantile grid (§1) |
| `seed` | 1 | jitter, permutations, thinning and bootstrap on four streams |
| $n_{\text{exc,min}}$ (`n_exc_min`) | 10 | exceedance count below which GPD refinement fires |
| $n_{\text{tail}}$ (`n_tail`) | 250 | null draws entering the GPD fit |

**No parameter in the shape test or the characterization asks the user what
shape of difference to look for.**

---

## 9. Scale — and what counts at low expression do to stage 2

Everything above is a functional of two quantile functions, and quantiles
commute with any monotone transform: $Q_{g(X)}(p) = g(Q_X(p))$. A transform
therefore enters WADE only through the difference curve
$D_g(p) = g(Q_1(p)) - g(Q_0(p))$; the grid, the permutation, the GPD and BH are
the same for every $g$. The one-parameter families people reach for —
Box–Cox $g_\lambda(x) = (x^\lambda - 1)/\lambda$, or $\log(x + c)$ with $c$
running from $0$ to $\infty$ — are a knob between the linear scale
($\lambda = 1$) and the log scale ($\lambda \to 0$). WADE sits at two
points on it on purpose: **stage 1 is $\lambda = 1$** (the area under $D$ is
the difference of means) and **stage 2 and the characterization are
$\lambda = 0$** (the curve $R$). This section says why, with numbers, and
what breaks at low counts.

### 9.1 Stage 1: the transform is a power choice, never a level choice

Under label permutation the test is exact for *any* statistic, so $g$ cannot
change the false-positive rate. Measured, null rejection at $\alpha = 0.05$
is 0.03–0.07 for every transform at every expression level tried. What $g$
changes is power, and the mechanism is one ratio. For a body at level $x_b$
with noise sd $\sigma_b$ and a fraction $\pi$ of cases at $x_s$, the
permutation z-score of the area statistic on scale $g$ is

$$z_g \;\approx\; \frac{\pi\,\rho_g\sqrt{n/2}}{\sqrt{1 + \pi_{\text{pool}}(1-\pi_{\text{pool}})\,\rho_g^2}},
\qquad
\rho_g = \frac{g(x_s) - g(x_b)}{g'(x_b)\,\sigma_b}$$

— the subset's contrast in units of the body's noise *on that scale* (delta
method). Two regimes. When $\rho_g$ is small, $z \propto \rho_g$ and the
transform with the most tail weight wins: for a body at 2 TPM and a subset at
1000, linear has $\rho \approx 1000$ and log $\rho \approx \ln 500 /
\mathrm{CV} \approx 20$ — and at low counts log also inflates the
denominator, because $\mathrm{Var}[\log X] \approx 1/\mu + \phi$ is
dominated by the one-count-is-one-log floor below $\mu \approx 1/\phi$.
When $\rho_g$ is large, $z$ saturates at
$\sqrt{n/2}\,\pi/\sqrt{\pi_{\text{pool}}(1-\pi_{\text{pool}})}$: the
p-value is then only "how many of the big values landed in the case group",
the combinatorial floor of §6, and *no* transform can move it. Past that
point more tail weight ($\lambda > 1$) only adds variance.

Measured, NB counts with dispersion $\phi = 0.1$, 200 v 200, $B = 500$,
two-sided, power at $\alpha = 0.05$:

| gene | linear | sqrt | $\log_2(x{+}10)$ | $\log_2(x{+}1)$ | $\log_2$ |
|---|---|---|---|---|---|
| $\mu = 2$, 5% of cases at 8× | **0.73** | 0.37 | 0.53 | 0.29 | 0.07 |
| $\mu = 20$, 5% at 8× | **0.91** | 0.79 | 0.67 | 0.46 | 0.45 |
| $\mu = 500$, 5% at 8× | **0.97** | 0.88 | 0.60 | 0.57 | 0.57 |
| $\mu = 20$, $\phi = 0.5$, global 1.3× | **0.88** | — | 0.84 | — | 0.68 |
| square ($\lambda = 2$), $\mu = 20$, 5% at 8× | 0.81 against 0.92 linear | | | | |

Linear is best or tied in every row, including the weak global shift with heavy
tails that theory gives log its only edge on (it is tiny at realistic fold
changes); square is worse than linear everywhere. **Stage 1 stays on the
linear scale.** The pseudocount family is strictly intermediate, and the
"subsets are much more pronounced in linear space" experience that motivated
this section is the small-$\rho$ regime above, measured.

### 9.2 Stage 2 needs the log scale — and why that is not enough for counts

$g(fx) - g(x)$ is constant in $x$ only for $g = a\log x + b$. That uniqueness
is why §3 reads the bridge off $R$: it is the one scale on which a global fold
change is flat, so that "departure from flat" means "not a global fold change".
The knob is not available here. Measured, the bridge on $\log_2(x + 10)$
fires on genuine 2× shifts at 0.99 below 2 counts and 0.21 at 20 counts, and
on the linear scale at 1.00 everywhere (which is the "0.70 against 0.98" of
§1, seen from the other side).

But the premise — *a global fold change is a multiplicative shift of the whole
distribution* — is true of continuous data and **false of counts at low
expression**, for two reasons. The continuity jitter turns a control count of
0 into $U(0, 0.01)$, so a case count of 1 against it is $R \approx +7.6$, a
spike at the low quantiles; and at $\mu \lesssim 1$ the median of $R$ sits
inside the zero floor, so the shift correction of §3 estimates $f \approx 1$
and removes nothing. Separately, a fold change in an NB *mean* is not a
multiplicative shift of the NB *distribution* — $\mathrm{Var} = \mu +
\phi\mu^2$, so the shifted group is relatively tighter at the bottom — and
$R$ slopes from about 1.2 at low $p$ to 0.95 at high $p$ for a 2× shift at 20
counts with no zeros at all. A large enough sample resolves that slope as "not
a global shift".

Measured on the current implementation, the rate at which the subset stage
fires on a **genuine 2× NB mean shift**, two-sided, $\alpha = 0.05$:

| $\mu$ (counts) | 200 v 200 | 1000 v 1000 |
|---|---|---|
| 0.5 | 0.98 | — |
| 2 | 0.95–0.99 | 1.00 |
| 20 | 0.17–0.19 | 0.72–0.83 |
| 100 | — | 0.12–0.17 |
| 500 | 0.04 | 0.00 |

The rate **grows with $n$**, so it bites hardest in the large-cohort regime
WADE is for. The artifact is bottom-heavy: `alternative="greater"`, the
cancer-outlier direction, is clean at $\mu \ge 2$ (0.03–0.04 at 200 v 200)
but not at the extreme floor. The characterization has the same blind spot —
median `affected_fraction` for a global 2× is 0.99 at 500 counts, 0.95 at 20,
**0.16 at 2 and 0.24 at 0.5** — so the "a global change reads 1.0" anchor of §4
holds only above a few tens of counts.

`tests/test_subset.py` validated stage 2 on continuous lognormal data with no
zeros and pure multiplicative noise, which is exactly the regime where the
premise holds; this one was not exercised. The fix is not a transform. It is a
noise-model choice, and it was prototyped and measured before being written
here.

### 9.3 The correction for counts: binomial thinning

The division trick of §3 worked because the bridge is *exactly invariant* to
division, so only the null needed correcting. The analogue for counts is
**binomial thinning**: NB is closed under thinning with the same dispersion
(thin a Poisson–Gamma by $q$ and it is NB$(q\mu, \phi)$), so thinning the
higher group's *raw counts* by $1/f$ produces a matrix that is exactly
exchangeable under "a fold change in the count model", zeros included, and
the Poisson component along with it. WADE takes raw counts precisely so that
this kind of operation is possible.

This is what `wade()` does by default (`thin=True`, `wade.thinning`). Three
things the prototype established, in order of how much they cost to learn:

1. **Thin the observed statistic as well as the null.** The bridge is not
   invariant to thinning — that is the point — so the observed curve must be
   read off the thinned matrix with the original labels, and the null off the
   same matrix permuted. Thinning only the null, as first specified, did
   nothing at 20 counts and inflated the null at the floor. Drawn once,
   before any permutation, like the jitter (§6).
2. **$f$ must be unbiased on a true global shift; its robustness matters
   less than it looks.** Two separate questions, measured separately.
   *Accuracy*: forcing $\hat f$ 30% too large or too small on a genuine 2×
   gives false-subset rates of 0.26 and 0.28 at 200 v 200 — a residual NB-mean
   shift between thinned cases and controls is itself a non-multiplicative
   shape, so an inaccurate $\hat f$ is **anti-conservative**, and more so at
   larger $n$. So the estimator must be unbiased. The ratio of interquartile
   means is not (2.08 for a true 2: the middle of a skewed distribution does
   not scale with its mean) and at 1000 v 1000 that alone is 0.07–0.15 false
   subsets; the ratio of means is unbiased but a 5% subset at 8× moves it to
   1.35. The estimator that is unbiased *and* out of reach of a subset below
   25% is **the thinning factor that makes the two groups' interquartile means
   agree** — bisection on $\log f$ with thinning inside the loop and common
   random numbers; it asks "after thinning, do the middles agree?", which
   assumes nothing about how the middle scales. It reads 2.00.
   *Contamination*: a subset larger than 25% does move it (33% at 2×: 1.24;
   60%: 1.55; 80% at 2× with 20% unchanged: 1.79) — and that costs nothing,
   because such a gene *is* a mixture, over-thinning leaves a two-level
   structure the bridge still sees, and the thinned test calls it non-global
   (power 0.97, 0.97 and 0.64 respectively at 200 v 200 and 20 counts; the
   80/20 gene is 0.28 today). The window is not data being thrown away — every
   sample enters the test and the characterization — it is the range over
   which "one global shift" and "a mixture" can still be told apart.
   An estimator-free alternative, $p = \max_f p_f$ over candidate fold
   changes (reject only if no $f$ explains the data), is valid by construction
   and measured too conservative to be the default (0.24 against 0.64 on the
   80/20 gene, 0.24 against 0.76 on a 5% subset at 2 counts), so it was
   measured and set aside rather than kept.
3. **Inverse-variance node weights do not substitute for this.** Weights from
   the permutation variance of $R(p)$ are label-free but not signal-free — a
   subset's own values inflate the null variance of the top nodes, so the
   signal region is down-weighted and the weighted `affected_fraction` reads
   ~0 for genuine subsets. Delta-method weights $1/(1/\mu + \phi)$ are far
   too generous at 1–3 counts. Both measured, both set aside.

Measured, with the matched estimator, the rate at which the subset stage fires
on a genuine 2× NB mean shift — the same cells as the table above:

| $\mu$ | 200 v 200, division | 200 v 200, thinned | 1000 v 1000, division | 1000 v 1000, thinned |
|---|---|---|---|---|
| 0.5 | 0.98 | **0.05** | — | 0.03 |
| 2 | 0.95 | **0.05** | 1.00 | 0.10† |
| 20 | 0.17 | **0.05** | 0.72 | 0.02 |
| 100 | — | — | 0.17 | 0.05 |

† 60 genes; within Monte Carlo error of 0.05, and the one cell worth
re-measuring with more replicates before it is quoted anywhere else.

The null stays at 0.02–0.07. Power is preserved above 2 counts (5% at 8×,
$\mu = 2$: 0.69 thinned against 0.64; 15%: 0.98 against 0.91) and is lower at
the extreme floor (5% at 8×, $\mu = 0.5$: 0.03 against 0.13), where the old
number was not power — it came with the 0.98 false-subset rate on global
shifts.

**The cost.** The fit is sixteen thinning-and-renormalizing passes, so at
20,000 genes, 100 v 100 and $B = 2000$ the run goes from 9.7 s to 16.6 s.

**Where it does not apply.** Genuinely continuous data, which is out of
WADE's scope: there the division is exact under a multiplicative model and
thinning would add Poisson noise the data does not have. There is no switch
for it; the package takes counts.

Two implementation points worth stating, both pinned by tests. The fold-change
fit runs on the **jitter-free** normalized scale — a hundredth of a count has
no business in a fold-change estimate, and on an all-zero gene it would
otherwise decide which group is "higher" and keep deciding it at every
bisection step. And non-integer counts (salmon, kallisto) are **rounded for
the thinning only**; the observed matrix and the characterization use the
values as given.

### 9.4 One zero is enough: the pseudocount for the log

The characterization's failure at low counts has a sharper cause than "the
floor", and it reaches far above low counts. At 20,000 v 20,000 and a mean of
20, NB$(20, 0.1)$ produces at least one zero in the control group about 30% of
the time; under the jitter that minimum is $\approx 0.005$, the case minimum is
$\approx 5$, and the bottom node reads $R = \log_2(5/0.005) \approx 10$.
The fourth moment raises that to $10^4$ — half the weight of the other
20,000 nodes combined — and a global 2× reads **0.39, 0.59, 0.70** instead of
0.97 on three of six seeds. **The tie-breaking jitter is being read as a
measurement.** The same mechanism, several nodes wide, is the 0.16 at 2
counts.

A count of zero means "less than one"; its logarithm should be bounded, not
$-7$. So the $R$ curve is computed with a **pseudocount of one count**:

$$R(p) = \log_2\big(Q_1(p) + 1\big) - \log_2\big(Q_0(p) + 1\big)$$

with the "1" meaning *one count, converted per sample into the normalized
units* — not 1 TPM; the normalization knows the conversion
(`wade.thinning.one_count`, exposed on the result as `pseudocount` and used by
`wade_gene` so a plotted curve is the one the statistics were read from). It
is `pseudocount=1.0` on `wade()`, in counts, and `0` disables it. A sample
with a zero library size has no count scale and gets no pseudocount. Two candidates
were measured side by side: this, and a floor at half a count
($\max(x, \tfrac12)$ before the log, which leaves every nonzero value on the
pure log scale). They behave the same within Monte Carlo error everywhere, the
pseudocount is slightly better where they differ (a global 2× at 5 counts
reads 0.86 against 0.74), it has no negative logarithms, and it is the
convention. Measured, 200 v 200:

| | raw log | log$_2$(x+1) |
|---|---|---|
| global 2× at 2 counts, `affected_fraction` | 0.17 | **0.70** |
| global 2× at 5 counts | 0.07 | **0.86** |
| global 2× at 20 counts, 20,000 v 20,000, worst of six seeds | 0.39 | **0.97** |
| 5% at 8× at 2 counts | 0.04 | 0.10 |
| 5% at 8× at 20 counts | 0.05 | 0.05 |
| thinned stage-2 power, 15% at 8×, mean 0.5 | 0.32 | **0.84** |
| thinned stage-2 power, 5% at 8×, mean 2 | 0.79 | 0.87 |
| thinned stage-2 level, global 2×, all means | 0.01–0.05 | 0.01–0.06 |

Above about 20 counts every number is identical: the pseudocount only ever
touches the bottom of the distribution. Under thinning the test stays exact —
a deterministic transform applied to observed and permuted thinned data alike
cannot move the level — and its power at the extreme floor rises a lot,
because those spikes were noise in the bridge too.

Two consequences for §7. The jitter is **irrelevant to stage 2 once there is
a pseudocount** (`log₂(0.005 + 1) ≈ 0.007`; measured, the $R$ curve with and
without jitter gives the same numbers), so it stays where it is needed — the
continuity of stage 1's null for the GPD tail fit — and stops being read as a
measurement. And adding Poisson(1) noise in its place, which was considered,
is measured to cost power at the floor (15% at 8× at a mean of 0.5: 0.84 →
0.49) while removing neither zeros (37% survive) nor ties; it is not used.

What remains at $\lesssim 5$ counts is the limit the section opened with: on
the log scale a Poisson count of 2 carries noise of about $\pm 0.7$ per node,
so the characterization is **noise-dominated** there — a 5% subset at 8× reads
0.10 rather than 0.05, the null reads 0.08, and a subset has to be roughly 16×
to stand clear. That is a resolution limit, to be reported with its
confidence interval (§9.5) rather than hidden, not an artifact to be fixed.
The "global reads 1.0" anchor of §4 is therefore *approximate below ~5 counts
and exact above*, and the measured table at the end of §4 — all at
lognormal(3, 0.6), i.e. around 20 units with no zeros — should be read with
that qualifier.

### 9.5 Confidence intervals for the descriptors

`affected_fraction`, `direction` and `log2_fc` are estimates and should carry
intervals.

**What is resampled, and how much of it.** The ordinary nonparametric
bootstrap: each replicate draws $n_1$ case columns from the $n_1$ cases **with
replacement** and $n_0$ control columns from the $n_0$ controls, so the
resampled groups are the *same size* as the real ones. That is the point — the
interval should describe the sampling variability of an estimate made at the
sample size you actually have, and drawing fewer would describe a smaller
study. It is not a hold-out: about 63% of the distinct samples appear in any
one replicate, and the rest are absent, but that is a consequence of sampling
with replacement rather than a chosen fraction. Groups are resampled
independently so the group sizes — and therefore the grid $m$ — never move.

Two variants exist for cases this one is known to fail — the $m$-out-of-$n$
bootstrap and subsampling without replacement — and neither is used here;
Poisson and hybrid resampling schemes were measured miscalibrated against
it, and $m$-out-of-$n$ under-covers. `wade(..., n_boot=300)` computes the
intervals (opt-in: 200 replicates cost about as much as the whole rest of a
20,000-gene run) and reports them as
`ci_affected_fraction`, `ci_direction`, `ci_log2_fc`, and as `*_lo` / `*_hi`
columns. The bootstrap — resample samples within each group with
replacement, recompute the curve, read the three numbers; percentile
intervals — is cheap (the curve is two quantile grids) and behaves. Measured at
a mean of 20, 300 bootstrap replicates, median 95% interval:

| gene | 200 v 200 | 1000 v 1000 | covers the estimand |
|---|---|---|---|
| 5% at 8× | [0.03, 0.09] | [0.04, 0.07] | yes (≈1.00) |
| 33% at 8× | [0.28, 0.42] | [0.31, 0.37] | yes |
| 33% at 2× | [0.25, 0.69] | [0.37, 0.62] | yes — but the estimand is ≈ 0.5, not 0.33 |

Two caveats the last row carries, both properties of the estimator rather than
of the interval. A 2× step against 30% noise is not a step but a ramp, and the
participation ratio of a ramp is larger than the planted fraction (0.5 for
33%; 0.93 for 80% at 2×): the interval covers *the estimand*, and the
estimand is biased upward for small fold changes. And the participation ratio
summarizes a **single-level** departure by construction: "global 2× plus 5%
at 8×" reads 0.43, which is neither number — it is the effective fraction of
the curve's energy. Both are true today and belong beside §4's table. The
permutation p-values need no interval beyond their Monte Carlo error,
$\sqrt{p(1-p)/B}$, and the resolution floors of §6.

---

## 10. Limits and failure modes

What WADE cannot do, and the two constraints that decide whether it can
answer a question at all. The first two are arithmetic on the **design**, not
properties of the implementation, and no amount of data or computation moves
them.

### 10.1 The two hard constraints

**Resolution.** The quantile grid has $m = \min(n_1, n_0, \texttt{max\_probs})$
points. Nothing finer than $1/m$ is estimable, so `affected_fraction` is a
quantitative estimate only above roughly 100 samples in the smaller group,
degrades through about 50, and below that distinguishes "global" from
"concentrated" without quantifying which fraction (the table in §4). **The
smaller group governs.** Adding cases to a study with 18 controls buys nothing
here.

**Detectability.** If $k$ samples carry a signal, label shuffling puts all of
them in one group with probability $\binom{n_1}{k}/\binom{n_1+n_0}{k}$, and
that sets the scale of the smallest p-value the design can support, whatever
the effect size, the statistic or the permutation count. It is driven by
**imbalance**, not by $k$ alone: the floor is roughly $(n_1/n)^k$.

```python
>>> wade.detectability_floor(77, 18, 15)      # 15 affected of 77 cases
0.031963032251420324
>>> wade.detectability_floor(48, 47, 15)      # the same 95 samples, balanced
9.904929906140583e-06
```

Balancing the groups buys far more than adding cases. Measured at 60 v 60
with the subset stage, planted effect 8× in every row:

| affected | $k$ | floor | power |
|---|---|---|---|
| 2% | 1 | 0.5 | 0.033 |
| 5% | 3 | 0.122 | 0.050 |
| 10% | 6 | 0.0137 | 0.467 |
| 25% | 15 | 1.1e-05 | 1.000 |

Power tracks the floor, not the effect size.

**A worked case: 100 cases against 10 controls.** The grid has 10 points, so
`affected_fraction` is qualitative at best. Five affected cases of 100 floor
at 0.62, 10 at 0.37, 25 at 0.067, and only 50 reach 1.6e-3: stage 2 cannot
work at that geometry unless roughly half the cases share the change. A
*global* shift is different — all 100 affected floors at 2.1e-14 — so stage 1
is usable; run it with `stage1="saddlepoint"` and read stage 2 as a screen.
Ten more controls would buy more than a hundred more cases.

### 10.2 What WADE does not do

**No covariate or batch adjustment.** The test takes a binary condition
vector and nothing else. A confound that tracks the case/control split is
indistinguishable from signal, and this is not an oversight to be patched:
the permutation null is built by exchanging labels, and a covariate makes
labels non-exchangeable. For a *categorical* confounder `strata=` shuffles
labels only within each stratum, holding study, batch or protocol fixed
instead of testing it as biology. It is not covariate adjustment and it is
not free: the permutation space becomes $\prod_s \binom{n_s}{k_s}$, a
stratum containing only one class contributes no freedom, and
`wade.permutation_space()` reports what is left. Otherwise detect the
confound and refuse, or run within each level and combine afterwards.

**No repeated-measures handling.** Two libraries from the same subject are
not exchangeable; feeding them in inflates the effective sample size and
biases every p-value anti-conservatively, silently. Aggregate to one value
per subject first. `strata=` does not solve this: it exchanges labels *within*
a stratum, whereas repeated measures need whole subjects exchanged *between*
groups, and a per-subject stratum has only one class in it.
`permutation_space()` will say so.

**Below about five counts per sample the characterization is
noise-dominated.** On the log scale a Poisson count of 2 carries roughly
±0.7 per node, so `affected_fraction` reads a global 2× as 0.70 rather than
1.0, a 5% subset at 8× as 0.10 rather than 0.05, and a null gene as 0.08
(§9.4). The ordering survives; the numbers are not quantitative. The *test*
is unaffected, because thinning is exact at every expression level. Run with
`n_boot=300` and read the interval, which is wide exactly there.

**Normalization lives inside the test, and couples genes** (§4). One gene's
normalized values depend on every other gene through the library size, so a
matrix in which a large share of genes carry strong signal distorts the
characterization of every null gene. The subset test is immune; the
descriptors and `mean_shift` are not. This affects every differential method
on normalized data.

**`mean_shift` is a grid quadrature, not a difference of means, on unequal
groups** (§2). Inference is unaffected; read it as "signed quantile area"
when the groups are unbalanced, or use `stage1="saddlepoint"`, which computes
the mean difference exactly.

### 10.3 A p-value at the extrapolation floor is not a small number

The GPD refinement floors its output at $1/(B \cdot n_{\text{tail}})$, 2.0e-6
at the defaults. A gene sitting *at* the floor has not been measured there; it
has been measured as "beyond what this many permutations can resolve", and the
floor is the most extreme claim the evidence supports. Two genes both at the
floor are not tied in strength; they are both unresolved.

**The reading rule.** Treat the floor as a censoring point. Break ties among
genes at the floor with `subset_log2_fc`, `log2_fc` and the permutation
z-scores, never by pretending the p-values differ. On a large cohort those
are the operative outputs rather than a fallback: a real 83,000-sample
contrast put 18.8% of genes at the floor. Resolving *below* it costs
permutations linearly. `refined_mean_shift` and `refined_subset` say which
p-values were read off the tail fit rather than counted, and refinement can
move a p-value in either direction.

Measured on 40 planted genes at 300 v 300, the shipped refinement at
$B = 2{,}000$ against the empirical p-value at $B = 50{,}000$: median ratio
1.19, worst under-statement 0.53, worst over-statement 17.2. Over the range
$p \approx 10^{-2}$ to $10^{-3}$ it errs high. Its behaviour further out, at
$10^{-6}$ and below, is the open problem of `pvalue-review.md`: 3–4,906×
conservative on stage 1 (solved by the saddlepoint) and 4–80× conservative
or anti-conservative to 0.019 on stage 2 (open).

### 10.4 The combinatorial floor is a scale, not a bound

The floor of §9.1 is exactly the p-value in the noise-free limit. Real data
is not noise-free: the unaffected samples are random too, so relabellings
that put all $k$ affected samples in the case group scatter above and below
the observed rather than tying it, and $k$ itself is not well defined in
count data. Measured on planted NB counts at 300 v 300, $B = 40{,}000$ so
that no refinement fires:

| planted $k$ | floor | $p$ / floor, q10 – median – q90 | fell **below** the floor |
|---|---|---|---|
| 3 | 1.24e-01 | 0.48 – 1.96 – 3.23 | 21% |
| 6 | 1.52e-02 | 0.26 – 1.10 – 1.71 | 42% |

The floor predicts the **order of magnitude** of the smallest p-value the
design can support; the median lands within a factor of two of it and a
substantial minority of genes come in under it. Use it to decide whether a
design can find a subtype at all. Never clamp a gene's p-value to it.
