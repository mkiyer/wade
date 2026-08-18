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

> **Status.** All of this is implemented. Sections 1–2 and 6–8 are additionally
> verified against the R reference; sections 3–5 are new work with no R
> counterpart, validated against planted ground truth in `tests/test_shape.py`.
> Section 7 records the superseded machinery, still present because the parity
> fixtures pin it. `ROADMAP.md` tracks what remains.

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

The signed area between the quantile functions. When $n_1 = n_0$ this is
*exactly* the difference of the two sample means; on unequal groups it is a
grid quadrature that over-weights the extremes of the larger group, so it should
be read as "signed quantile area" rather than as a drop-in mean estimate.
Inference is unaffected, because the permutation null inherits the same bias and
it cancels in the p-value.

**This is the ordinary test**, and naming it plainly matters: it is what any
conventional DE method already computes, and WADE's claim is not to improve on
it. Measured, it is the *best* single detector for weak diffuse effects (10% of
cases at 2×: power 0.700, against 0.295 for the shape test). WADE's claim is
that it adds sensitivity the mean test lacks for strong concentrated effects
(2% of cases at 8×: 0.700 against 0.935).

> Formerly `diff.mean`, described as the "bulk axis". Renamed: "bulk" told a
> reader nothing.

---

## 3. Stage 2 — the shape test

Replaces the fixed tail window described in §7.

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
both signal shapes: 0.315 for the argmax against 0.084 for $\hat\pi$. **Threshold-free by
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

The fix uses the fact that $B$ is **exactly invariant** to a global fold change
— $R \to R - c$ sends $S_k \to S_k - kc$ and leaves $B_k$ unchanged. So the
observed statistic needs no adjustment; only the null does. Divide the case
columns by the estimated fold change, which makes the two groups exchangeable
under $H_0$, and permute *that*.

The shift is estimated by the **median** of $R$, not the mean: a subset signal
moves the top quantiles and leaves the median alone, so estimating the shift
this way does not quietly remove the signal being tested for.

Measured effect of the correction, 500 v 500:

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

Two bounded, parameter-free numbers, both read off $R$.

### The effective affected fraction

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

### The direction

$$\boxed{\;\texttt{up\_share} = \frac{\sum_p \max(R(p), 0)}{\sum_p |R(p)|}\;}$$

The share of total distributional movement that is upward. Bounded $[0,1]$.

Together the pair is a complete description. Measured at 500 v 500:

| gene | $\hat{\pi}$ | `up_share` | reads as |
|---|---|---|---|
| global fc = 2 | 0.978 | 1.000 | everything, up |
| global fc = 0.5 | 0.979 | 0.000 | everything, down |
| variance × 1.6 | 0.316 | 0.495 | two-sided spread |
| subset 5% up | 0.053 | 0.958 | 5% of cases, up |
| subset 5% down | 0.053 | 0.031 | 5% of cases, down |
| subset 5% up + 5% down | 0.095 | 0.495 | 10% total, split |

A symmetric variance change and a one-sided 30% subset both give
$\hat{\pi} \approx 0.3$ and are separated by `up_share` (0.50 against ~1.0).

### Composition moves both, and that is not a defect

Library-size normalization couples genes. A matrix in which a substantial
fraction of genes are strongly up in cases inflates the case libraries, which
pushes every *other* gene down — and because the offset is systematic while the
sampling noise is not, at large $n$ it takes very little signal to dominate.
Measured at 400 v 400, five strongly-up genes among 205 were enough to move the
null genes' `up_share` from ~0.5 to near 0.

Consequences to keep in view when reading a characterization: null genes
acquire a small consistent fold change, so their $\hat\pi$ drifts toward 1
(they genuinely *are* globally shifted, relative to the library) and their
`up_share` collapses toward 0 or 1. **The shape test itself is unaffected** —
the bridge is invariant to exactly this kind of global offset — but the two
descriptive statistics are not. This is a property of normalized data that
affects any differential method, not something WADE introduces, and it is
asserted in `tests/test_shape.py` so it cannot be forgotten.

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

This is the same grid resolution `limits.md` documents, surfacing honestly
instead of being absorbed by a rounding rule.

---

## 5. Reading the output

| `p_shift` | `p_shape` | interpretation |
|---|---|---|
| significant | — | global shift; an ordinary DE method finds this too |
| significant | significant | concentrated in a subset strong enough to move the mean — $\hat\pi$ says how much, `up_share` which way |
| — | significant | distributional change with no net mean shift: a balanced subset, or a variance change |
| — | — | not differential |

---

## 6. Inference

**Permutation of the labels**, with two properties that are load-bearing:

- **One shuffle serves all genes.** Within a permutation the same label vector
  applies to every gene, so the null preserves the gene–gene correlation
  structure. Drawing an independent shuffle per gene would give a different and,
  across correlated genes, anti-conservative null.
- **The continuity jitter is drawn once, before any permutation** (§8), so
  inference is *conditional* on one realised noise draw. That is what makes a
  seed reproduce a result exactly. It must never be re-drawn inside the loop.

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

**BH-FDR** is applied across genes, separately per axis.

### The combinatorial floor

If $k$ samples carry a signal, label shuffling places all of them in the case
group with probability $\binom{n_1}{k}/\binom{n_1+n_0}{k}$. **No
label-permutation test can return a p-value below that**, whatever the effect
size, the detector, or the permutation count. At 77 cases vs 18 controls it is
0.173 at 10% affected — so nothing below roughly 25% is detectable at
$\alpha = 0.05$, by any method in this family. This dominates detector choice at
small $n$ and is the single most important thing to check before running.

---

## 7. What the subset machinery is today

**IMPLEMENTED, and superseded by §3–4.** Recorded because it is what the code
currently does and what the parity fixtures pin.

The tail window is $k = \max(1, \lceil q_{\text{tail}} \cdot m \rceil)$ with
$q_{\text{tail}} = 0.10$, giving `tail.mean` (the mean of $D$ over the top $k$
nodes) and `tail.conc` (its share of the total signed area).

Why it is being replaced: three stacked heuristics — a fraction, a rounding
rule, and a floor — plus a fourth to guard the ratio they produce. It forces the
user to declare what they are looking for, and $k$ is an integer window on a
grid whose size is set by the design, so the realized tail fraction sawtooths
between 0.100 and 0.182 as group size varies and jumps discontinuously
($m = 20 \to 21$ moves it from 10.0% to 14.3% on one extra sample). `tail.mean`
is therefore not comparable across contrasts of different sizes.

Measured, the replacement matches it in power and beats it wherever the signal
is not concentrated near 10% — at 77 v 18 with a weak global change, 0.975
against 0.830.

`tail.conc` additionally has a **pole**: its denominator is a signed sum, so it
diverges whenever the bulk cancels the tail. Measured on a realistic simulation,
the largest $|$`tail.conc`$|$ among genes with *no signal at all* was 3,483. The
R reference guards it at $|$`diff.mean`$\cdot m| < 10^{-8}$, roughly nine orders
of magnitude tighter than the cases that occur, which catches none of them. §4's
$\hat\pi$ has no pole and needs no guard.

---

## 8. Normalization

The entry point takes **raw counts**, because the continuity jitter is applied
at count precision *before* division and a pre-normalized matrix cannot
reproduce it.

$$X_{gj} = \kappa \cdot
  \frac{(C_{gj} + \eta_{gj}) / L_{gj}}{\ell_j + \eta_{gj}/L_{gj}},
  \qquad \eta \sim \mathrm{Uniform}(0, \nu)$$

The jitter breaks ties in zero-heavy data, where many samples share a count of
zero and the quantile grid would degenerate into flat runs; it also makes every
value strictly positive, so fold changes stay finite.

**The per-cell denominator is a quirk, reproduced for parity rather than
endorsed.** It adds only *this gene's* jitter contribution to the library size,
not the whole column's, so it is the library size in a counterfactual where gene
$g$ alone received jitter. The origin is historical: WADE's ancestor normalized
one gene at a time, where `lib_sizes + noise/length` is the natural expression,
and matrixizing it preserved a per-gene correction. Consequences: column sums
are not exactly $\kappa$ (the output is TPM-*like*), and any gene that
constitutes an entire library normalizes to exactly $\kappa$ — including every
gene in an all-zero sample, which is an artefact rather than a sensible value.

`cpm` and `rle` ship alongside and deliberately do **not** reproduce the quirk:
they apply the jitter at count precision, which is the load-bearing part, and
use consistent library sizes.

---

## 9. Defaults

| parameter | default | role |
|---|---|---|
| $B$ (permutations) | 2000 | null resolution; sets both p-value floors |
| $\nu$ (jitter) | 0.01 | continuity jitter width, at count precision |
| $\kappa$ | 1e6 | TPM-like scale factor |
| seed | 1 | jitter and permutations on separate streams |
| $n_{\text{exc,min}}$ | 10 | exceedance count below which GPD refinement fires |
| $n_{\text{tail}}$ | 250 | null draws entering the GPD fit |

The R reference had two different permutation defaults (1000 in the driver,
2000 in the wrapper). One is picked here and stated, because $B$ sets both the
empirical floor $1/(B+1)$ and the extrapolation floor $1/(B n_{\text{tail}})$ —
it changes every small p-value, not just the runtime.

$q_{\text{tail}}$ and the `tail.conc` guard factor appear nowhere in §3–4.
**No parameter in the shape test or the characterization asks the user what
shape of difference to look for.** They survive only in the superseded
machinery of §7, which the parity fixtures pin.
