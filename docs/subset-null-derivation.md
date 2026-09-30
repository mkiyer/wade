# The subset test's null hypothesis: a derivation for review

*Draft for external statistical review, 29 September 2026.*

> **Status after three reviews (29 September 2026).** The proposed
> construction stays experimental. The second review found residual
> anti-conservatism within the formal model: on a bimodal null with 20 cases
> and 20 controls, 0.4–0.6% at the 0.1% level. The third found that it is
> still measurable at 250 v 250 over 200,000 genes. It also found that the
> null defined here, a common fold of observed count + 1, is not the RNA-seq
> count-model null.
>
> The plan that followed the reviews, with its section 8.3 of corrections, is
> `docs/subset-null-plan.md` at tag `analysis-2026-09-30`. The present
> [plan](subset-null-plan.md) records the decisions since, and
> [method](method.md) the test as built on 30 September 2026. Its
> registration is this page's W1 fold, and its claim is limited to genes whose
> sparser group expects at least 128 reads per sample. Several statements
> below are too strong and will be rewritten with the redesign:
>
> - **The alternative.** The depth over the case mean + 1 equals the largest
>   gap between the controls' and the cases' Lorenz curves of count + 1. The
>   test therefore targets excess concentration of expression among cases, a
>   one-sided departure, not "any departure from a common fold". A contraction
>   (cases more homogeneous than controls) gives D = 0.
> - **R1 and R2** are design choices that suffice for the constructions
>   compared here, not universal requirements for composite-null Monte Carlo
>   tests.
> - **Independent error in the registration** did no harm in these
>   experiments; that is not a general result.
> - **The normalizer.** A subset below half the cases does move the median
>   case (a separated upper share π puts it near the 0.5/(1 − π) quantile of
>   the majority); it cannot move it without bound.
> - **Breakdown.** Replacing observations changes ranks and quantile
>   pairings, so the uncontaminated a(u) values are not a fixed set. Any
>   breakdown statement must say which group is contaminated.
> - **The fold-form regression** supports a multiplicative scale only as an
>   approximation across genes. The attenuation argument does not hold, since
>   median and interquartile-range errors share observations.
> - **Few controls** limit the evidence a missing control tail provides; that
>   is not a universal p-value floor. The released test's power at 500 v 50
>   came with invalidity on sparse-tail genes, but whether part of it is
>   attainable by a valid test needs a comparison at matched size.
> - **Folds on observed expression.** The whole proposed p-value, not just
>   the statistic, is unchanged by a fold applied to the cases' observed
>   values. Simulated folds of the latent level therefore test the count
>   model, not the registration.

This note is self-contained. It defines every quantity it uses and does not
assume the reader has seen WADE's code. Section 4 describes the construction
in the current release candidate (wade 0.2.0, not yet released) and its
measured failure. Section 5 derives a replacement. Section 7 lists what we
cannot yet justify and where we would most value criticism. Every number comes
from `research/subset_null_study.py` (in git at tag `analysis-2026-09-30`); its
output is kept in `.release/0.2.0/subset-null-study-20260929.txt`.

## 1. Summary

WADE compares RNA-seq expression between cases (for example tumours) and
controls (for example normal tissue), one gene at a time. For each gene it
reports the **mean shift**, the difference in group means, and a **subset**:
the cases that depart from what the mean shift predicts if every case carried
it alike (section 2). The **subset test** asks whether that departure is
larger than chance under

> H₀: every case carries the same change; the case distribution is the
> control distribution scaled by one common fold.

The fold is unknown, so H₀ is composite. The test's null distribution is built
by permutation after removing an estimated fold from the cases, and in the
release candidate that estimate is the mean shift's fold. On real TCGA data
with randomly assigned labels, that test rejects 6–12 times as often as exact
permutation at p ≤ 0.001 among genes where a few samples dominate the mean,
which is common in RNA-seq (section 4).

We derive a construction from two requirements (section 5): the test statistic
must be invariant to the unknown fold, and the pooled null sample must be
registered by a fold that a few samples cannot move. The result is:

1. register the cases onto the controls by the fold that minimizes the
   Wasserstein-1 distance between their log-scale quantile functions (a
   weighted median, over quantiles, of the log fold), then
2. run an ordinary label-permutation test of the walk's depth measured in
   units of the median case.

On real data under random labels it matches exact permutation in three
designs. In simulations with sparse extreme tails and uniform folds it holds
its level. On power (section 6):

- in balanced designs it keeps the released test's power;
- when controls are few, the power it gives up was never real;
- when cases are few it loses real power (at 50 cases v 500 controls, a raised
  tenth at 8× over a halved majority is called in 17% of genes against 78%).

The derivation leaves open questions, above all a proof of validity and the
low-count regime (section 7).

## 2. Definitions

### 2.1 Data

For one gene there are n samples with counts $c_j$, size factors $s_j > 0$
(fitted on all genes without using the labels), and labels $L_j \in \{0, 1\}$:
$n_1$ cases ($L_j = 1$) and $n_0 = n - n_1$ controls. Each count receives
independent jitter $u_j \sim \text{Uniform}(0, 1)$ so that no two values tie.
Expression on the default **linear scale** is

$$v_j = (c_j + u_j) / s_j ,$$

and on the optional **log2 scale** it is $v_j = \log_2\{(c_j + u_j)/s_j + 1\}$.
Write the sorted case values $x_{(1)} \le \dots \le x_{(n_1)}$, the sorted
control values $y_{(1)} \le \dots \le y_{(n_0)}$, and their empirical quantile
functions

$$Q_1(u) = x_{(\lceil u n_1 \rceil)}, \qquad Q_0(u) = y_{(\lceil u n_0 \rceil)}, \qquad u \in (0, 1].$$

Samples are independent units (one per patient). Genes are tested separately
and adjusted together by Benjamini–Hochberg; this note concerns one gene.

### 2.2 The mean shift

$$\Delta = \bar x - \bar y = \int_0^1 \{Q_1(u) - Q_0(u)\}\, du ,$$

the signed area between the two quantile functions. On the linear scale the
change is also expressed as a fold on count + 1,

$$\hat f = \frac{\bar x + 1}{\bar y + 1} ,$$

where the 1 keeps the fold finite for genes with zero expression. WADE tests
Δ separately with an exact permutation test; that test is not at issue here.

### 2.3 The prediction, the walk, the depth and the subset

If every case carried the mean shift alike, the case quantiles would be the
control quantiles changed by the fold:

$$\hat Q_1(u) = \hat f\,\{Q_0(u) + 1\} - 1 \ \ \text{(linear)}, \qquad \hat Q_1(u) = Q_0(u) + \Delta \ \ \text{(log2)} .$$

In both cases $\int_0^1 \hat Q_1(u)\,du = \bar x$: the prediction carries
exactly the observed mean shift. The **walk** is the cases' running departure
from this prediction,

$$R(k) = \int_0^{k/n_1} \{Q_1(u) - \hat Q_1(u)\}\, du , \qquad k = 0, 1, \dots, n_1 ,$$

with $R(0) = R(n_1) = 0$. It falls through cases below the prediction and
rises through cases above it. (A boundary k is allowed only between distinct
case values; with jitter every value is distinct.) Its **depth** is

$$D = -\min_{0 < k < n_1} R(k) ,$$

set to 0 when R never falls below floating-point rounding. The **cut** $k^*$
is the largest k at which the minimum is attained. The **subset** is the
smaller side of the cut:

- the $n_1 - k^*$ cases above the cut if $n_1 - k^* \le k^*$ (a raised subset);
- otherwise the $k^*$ cases below it (a lowered subset).

WADE reports the subset's share of the cases and its fold over the
prediction. D is the departure the split carries: the deficit of the cases on
one side of the cut below the prediction, which equals the excess of the
cases on the other side above it.

### 2.4 The subset test

The two-sided subset test uses D as its statistic; the one-sided versions sign
D by the subset's side. The null distribution is built by Monte Carlo
(sections 4 and 5), and p-values use Besag–Clifford sequential stopping. The
test shuffles until the h-th shuffle whose statistic reaches the observed one
(h = 25, about 20% relative error) and reports p = h/T at shuffle T. If it
reaches B shuffles first, it reports (K + 1)/(B + 1), where K is the number of
exceedances. B = 999,999 in the software; the studies below use 9,999 or
99,999.

## 3. The null hypothesis

**H₀.** There is a fold f > 0 such that

$$x \overset{d}{=} T_f(y), \qquad T_f(v) = f\,(v + 1) - 1$$

on the linear scale (on the log2 scale, $T_\delta(v) = v + \delta$). Every case
carries the same change, and that change is a fold on count + 1.

The alternative is any departure from a common fold: a subset of cases above
or below it, or a change of shape such as a wider spread at the same mean. The
walk's cut summarizes where the departure sits.

**How f relates to the mean shift.** Under H₀,
$\mathbb E[x] + 1 = f\,\{\mathbb E[y] + 1\}$, so $\hat f$ of section 2.2
estimates f. H₀ says the mean shift applies to all cases, and the walk's
prediction is H₀ with f estimated by $\hat f$.

**Why a fold rather than a difference.** An additive null,
$x \overset{d}{=} y + \Delta$, would keep the spread of expression fixed
while its level changes; a fold scales both. To see which describes real
differences between groups, we took every gene whose median is at least 50 in
both groups of two TCGA cohorts. Across these genes we regressed the log₂
ratio of interquartile ranges (tumour over normal) on the log₂ ratio of
medians. A fold predicts slope 1 and a difference predicts slope 0.

| Cohort | Genes | Slope | Intercept |
| --- | ---: | ---: | ---: |
| Prostate, 445 tumours v 52 normal | 14,646 | 0.95 | +0.30 |
| Breast, 971 tumours v 112 normal | 14,669 | 0.99 | +0.76 |
| Prostate, two random halves of the tumours | 15,126 | 0.60 | −0.01 |
| Breast, two random halves of the tumours | 15,027 | 0.39 | +0.00 |

The binned medians agree. In prostate, genes whose median falls by a factor
of 2–4 have their interquartile range fall by a median of 0.98 log₂ units.
Genes whose median rises by 2–4× have it rise by 1.54 log₂ units.

- Noise in the median ratio attenuates a regression slope, so the true slope
  is, if anything, closer to 1.
- The positive intercepts say tumours are more heterogeneous than normal
  tissue overall. That is itself the kind of departure from H₀ the subset test
  exists to detect.
- The random halves have almost no spread in their median ratios, so their
  slopes carry no information; their intercepts (−0.01 and 0.00) confirm
  that exchangeable groups have matching spreads.

**What H₀ does not cover.** $T_f$ acts on observed expression, and its 1 (one
count at a typical size factor) is part of the model. At high counts that is
the multiplicative model of latent expression. At low counts Poisson
counting noise does not scale with the fold: a gene at two counts whose cases
all double does not have $T_f(y)$ as its case distribution. At low counts,
therefore, a uniform biological change violates H₀ as stated (section 7, item 4).

**What H₀ gives a permutation test.** Under H₀ with independent samples, the
registered cases $T_f^{-1}(x_i) = (x_i + 1)/f - 1$ and the controls are n
independent draws from the control distribution G. If f were known, permuting
labels over the registered sample and recomputing any statistic would give an
exact test. f is unknown, and how it is handled is the whole question.

## 4. The released construction and why it fails

### 4.1 Construction

The release candidate registers by the mean shift's fold (on the log2 scale,
by Δ). It pools
$P = \{T_{\hat f}^{-1}(x_i)\} \cup \{y_j\}$. Each shuffle draws $n_1$ of the n
pooled values as new cases and applies $T_{\hat f}$ to them, keeping the rest
as new controls. It then recomputes the walk (with the shuffled data's own
mean shift) and compares the shuffled depth $D^*$ with D.

**Adjusting controls or cases makes no difference.** Scaling the controls up
instead of the cases down, that is pooling $T_{\hat f}(P)$ in case units,
produces exactly the same shuffled data sets. The new cases are the same
values, and the new controls are $T_{\hat f}^{-1}$ of the same values in case
units, which are again the same values. The question is therefore not which
group is adjusted, but by what.

### 4.2 Evidence of failure

**Real genes under random labels.** In TCGA prostate (24,246 genes, 497
samples) we reassigned the labels at random at three sizes, eight labellings
each. Random labels make every gene null with f = 1, so plain label
permutation ("exact") is exactly valid and serves as the benchmark. Its rates
differ from 5%, 1% and 0.1% only because eight labellings of correlated genes
are a small sample. "Heavy" genes are the 406 whose largest sample is at least
4× the next on count + 1.

Share of genes at p ≤ 0.05 / 0.01 / 0.001, pooled over eight labellings,
linear scale, up to 9,999 shuffles:

| Design (cases v controls) | Exact, all genes | Released, all genes | Exact, heavy genes | Released, heavy genes |
| --- | --- | --- | --- | --- |
| 445 v 52 | 3.53 / 0.60 / 0.056% | 3.72 / 0.73 / 0.103% | 4.25 / 0.89 / 0.092% | 7.17 / 2.89 / 1.078% |
| 248 v 249 | 4.26 / 0.83 / 0.077% | 4.31 / 0.90 / 0.112% | 4.96 / 0.86 / 0.123% | 9.21 / 3.73 / 1.355% |
| 52 v 445 | 6.74 / 1.46 / 0.144% | 6.12 / 1.18 / 0.130% | 5.82 / 1.45 / 0.185% | 10.07 / 4.00 / 1.170% |

Across all genes the released construction is close to exact at p ≤ 0.05 and
0.01, and up to 1.8 times as often at p ≤ 0.001. Among the heavy genes it
rejects 6–12 times as often as exact at p ≤ 0.001.

**Single genes.** When one gene's own values are relabelled at random 200
times (445 v 52), the released test gives p ≤ 0.001 in 27% of labellings for
MC4R, 52% for DRGX, 49% for AMBN, 14% for IRS4 and 8% for ETV4 (section 6.2).
Each of these genes has one or a few samples far above the rest.

**Simulations.** The simulations add sparse extreme tails: 1% of samples at
100×, or 0.5% at 500×, in both groups. With at least as many cases as
controls, the released construction rejects 12–41% of null genes at p ≤ 0.05
(section 6.3).

### 4.3 Two mechanisms

**Registration.** The mean shift's fold has unbounded influence: one case can
move it arbitrarily far. Take ETV4 in TCGA prostate (445 tumours v 52 normal):

- 22 tumours exceed every normal sample, reaching 425,000 against a normal
  median of 159;
- the typical tumour sits at about half the normal level (tumour median 75);
- yet the mean shift's fold is 15.4.

Dividing every tumour by 15.4 places 423 of the 445 below the lowest normal
sample. The pool is then a mixture of two shapes, not one sample from G, and
its shuffles do not reproduce how D behaves under H₀.

*A toy example to check by hand.* Take ten controls at 1, nine cases at 1 and
one case at 20.

- The means are 2.9 and 1, so $\hat f = 3.9/2 = 1.95$ and the prediction is
  2.9 for every case quantile.
- The walk falls by 0.19 per case through the nine cases at 1, to R(9) = −1.71,
  and returns to 0 at the tenth. So D = 1.71, $k^* = 9$, and the subset is the
  single case at 20.
- The mean-shift registration divides count + 1 by 1.95. The nine ordinary
  cases become 0.03, below every control, and the extreme one becomes 9.8.
- The registration of section 5.2 has a(u) = 0 over nine tenths of the
  quantiles, so it is 1 and leaves the data unchanged.

**Scale.** The depth is measured in expression units, so it grows with the
fold.

> **Lemma 1.** For every c > 0, $D(T_c(x), y) = c \cdot D(x, y)$, and the cut
> is unchanged. (Proof in appendix A.)

Under H₀ with fold f, therefore, $D = f \cdot D(x^0, y)$ with
$x^0 = T_f^{-1}(x)$ a sample from G, and the null distribution of D depends
on f. In the construction of section 4.1 the shuffled depths are on the scale
of the registration fold, while the observed depth is on the scale of f.
Registering by the right fold is therefore not enough: the registration fold
must also scale the null exactly as f scales the observed depth.

The mean shift's fold nearly does this on well-behaved genes, by coincidence
rather than design. A case sample that by chance has a heavier upper tail has
both a larger mean, hence a larger $\hat f$ and a larger null scale, and a
deeper walk. Section 6.4 separates the two mechanisms. Registering by a
robust fold while keeping D fixes the heavy-tailed genes but runs high when
controls outnumber cases. Registering by the true fold with independent
random error of the same size does not.

## 5. Derivation of the proposed construction

A Monte Carlo null for a composite hypothesis needs two things:

- **R1:** a statistic whose null distribution does not depend on the nuisance
  parameter;
- **R2:** a way to draw from the part of the null that the statistic does
  depend on.

### 5.1 R1: an invariant statistic

Common folds form a group acting on the cases, $x \mapsto T_c(x)$, and H₀ is
invariant under it: if $x \overset{d}{=} T_f(y)$ then
$T_c(x) \overset{d}{=} T_{cf}(y)$. Suppose a statistic S satisfies
$S(T_c(x), y) = S(x, y)$ for every c. Then under H₀

$$S(x, y) = S\big(T_f^{-1}(x), y\big) = S(x^0, y), \qquad x^0 \sim G ,$$

so the null distribution of S depends only on G and the design, never on f.
This is the standard reduction by invariance (Lehmann and Romano, *Testing
Statistical Hypotheses*, ch. 6). By Lemma 1, dividing D by any statistic of
the cases that scales as D does makes it invariant:

> **Lemma 2.** If m satisfies $m(T_c(x)) + 1 = c\,\{m(x) + 1\}$, which every
> quantile of the cases does, then $S = D / \{m(x) + 1\}$ satisfies
> $S(T_c(x), y) = S(x, y)$.

We take m to be the median case. It lies on the larger side of the split, or
straddles the cut when the two sides are equal, so a subset cannot inflate it.
S is then the depth in units of the typical case's expression (count + 1).

On the log2 scale D is already invariant: adding δ to every case adds δ to Δ
and to the prediction and leaves R unchanged. There S = D.

### 5.2 R2: registration

Under H₀, $S(x, y)$ is distributed as $S(x^0, y)$ with $x^0$ and y independent
samples from G, and permuting labels over one sample from G reproduces exactly
that distribution. The registered sample $\{T_g^{-1}(x_i)\} \cup \{y_j\}$ is
such a sample when g = f, and approximately one when g is an estimate
$\hat g$. Because S is invariant, $\hat g$ no longer affects the null's scale,
only how well the registered cases superimpose on the controls. What we need
from $\hat g$ is therefore consistency under H₀ and robustness to a few
extreme values.

WADE's own geometry offers a natural choice: the fold that best superimposes
the two distributions in area on the log scale,

$$\hat g = \arg\min_{g > 0} \int_0^1 \big| \log\{Q_1(u) + 1\} - \log g - \log\{Q_0(u) + 1\} \big| \, du .$$

Since $\log\{T_g^{-1}(v) + 1\} = \log(v + 1) - \log g$, the integral is the
Wasserstein-1 distance between the log-scale distributions of the registered
cases and the controls. Its minimizer is the weighted median over
u ∈ (0, 1) of

$$a(u) = \log \frac{Q_1(u) + 1}{Q_0(u) + 1} .$$

This is computed exactly: a is piecewise constant on the merged breakpoints
$\{k/n_1\} \cup \{j/n_0\}$, and each piece is weighted by its length (we take
the lower weighted median). Its properties:

- **Consistency under H₀.** In the population
  $Q_1(u) + 1 = f\,\{Q_0(u) + 1\}$ for every u, so $a(u) \equiv \log f$ and
  the minimizer is f.
- **Robustness.** Like any median it has breakdown point ½. Changing a share
  π < ½ of the case values arbitrarily moves $\hat g$ only within the spread
  of the log folds of the other quantiles. The mean shift's fold has breakdown
  point 0. For ETV4, $\hat g$ = 0.51 against a mean-shift fold of 15.4.
- **Relation to the mean shift.** The mean shift's fold is the fold at which
  the *signed* area between $T_g(Q_0)$ and $Q_1$ vanishes on the linear scale,
  which is the requirement that the adjusted mean difference be zero.
  $\hat g$ is the fold at which the *unsigned* area between the log quantile
  functions is smallest. Under H₀ both estimate f. They differ when a subset
  exists, which is exactly what the test looks for.
- **The Wasserstein-2 analogue** is the geometric-mean fold,
  $\exp\{\overline{\log(x + 1)} - \overline{\log(y + 1)}\}$. It is consistent
  too, but one extreme case still moves it by $\log(\text{value})/n_1$, which
  matters when cases are few (sections 6.1 and 6.3).

On the log2 scale the registration is the weighted median of
$Q_1(u) - Q_0(u)$.

### 5.3 The proposed procedure

1. Compute the walk on the data and $S_{\text{obs}} = D / \{\operatorname{median}(x) + 1\}$.
2. Compute $\hat g$ and the registered cases $\tilde x_i = (x_i + 1)/\hat g - 1$.
3. Pool $\{\tilde x_i\} \cup \{y_j\}$. For each shuffle, draw $n_1$ pooled
   values as new cases and recompute the walk and $S^*$.
4. Report the Besag–Clifford p-value of $S_{\text{obs}}$ against the $S^*$.

By Lemma 2, $S_{\text{obs}} = S(\tilde x, y)$, and re-applying $\hat g$ to
shuffled cases would not change $S^*$. A numerical check gave identical
exceedance counts in all 132 genes tested. The procedure is thus exactly
"register, then run an ordinary label-permutation test of S". With the true
fold in place of $\hat g$ it would be an exact permutation test. Its only
approximation is using $\hat g$, which depends on the labels.

Unchanged are the mean shift and its test, the walk, the cut, and the
subset's share and fold. The zero-mean-difference constraint still defines the
walk's prediction; it no longer builds the null.

### 5.4 Alternatives considered

- **No registration** (plain label permutation) is exact when f = 1. Under H₀
  with f ≠ 1, though, the pool is a mixture of G and a scaled copy of G, and
  the test fails: up to 13% at p ≤ 0.05 with light tails and 19% with sparse
  tails (section 6.3).
- **Robust registration with the raw depth** fails R1: it fixes the
  heavy-tailed genes but runs high when controls outnumber cases
  (section 6.4).
- **The mean shift's fold with the scaled depth** fails R2 badly (sections 6.1
  and 6.3).
- **Registering by $\hat g$ while multiplying shuffled cases by the mean
  shift's fold** keeps the released scale but, under a subset, inflates every
  shuffled depth. In exploratory runs (not in the study script) it removed
  nearly all power.
- **Other normalizers.** Any scale-equivariant statistic of the cases
  satisfies R1: the geometric mean or interquartile mean of count + 1, the
  registration fold times the median control, or the mean of the larger side
  of the split. In exploratory runs the first three behaved like the median;
  the last was less well calibrated in mixed simulated matrices.
- **p-values maximized over a confidence set for f** (Berger and Boos, 1994)
  would give validity without relying on the accuracy of $\hat g$, at the
  cost of running the test at many folds per gene. We have not pursued this
  (section 7, item 1).
- **A null built from the controls alone**, for example by resampling them,
  avoids registration. It cannot represent tail events that the controls
  happen to miss but the more numerous cases reveal. A model of the control
  distribution fitted across genes might supply them (section 7, item 5).

## 6. Evidence

### 6.1 Real genes under random labels

The same experiment as section 4.2, with every construction: the share of
genes at p ≤ 0.05 / 0.01 / 0.001, pooled over eight labellings per design.
Compare each row with **exact**, which is exactly valid here. Its departures
from 5 / 1 / 0.1% show the sampling noise of eight labellings of correlated
genes. With 406 heavy genes × 8 labellings, 0.03% is one rejection.

Heavy genes (406), linear scale:

| Construction | 445 v 52 | 248 v 249 | 52 v 445 |
| --- | --- | --- | --- |
| exact | 4.25 / 0.89 / 0.09% | 4.96 / 0.86 / 0.12% | 5.82 / 1.45 / 0.19% |
| released (mean-shift fold, depth) | 7.17 / 2.89 / 1.08% | 9.21 / 3.73 / 1.36% | 10.07 / 4.00 / 1.17% |
| W1 fold, depth | 4.31 / 0.89 / 0.09% | 4.99 / 0.89 / 0.12% | 5.94 / 1.02 / 0.15% |
| mean-shift fold, scaled depth | 16.32 / 10.41 / 6.80% | 26.51 / 19.24 / 14.13% | 11.33 / 11.24 / 10.47% |
| log-mean (W2) fold, scaled depth | 4.50 / 1.08 / 0.25% | 5.54 / 1.05 / 0.12% | 9.48 / 2.34 / 0.37% |
| **proposed (W1 fold, scaled depth)** | 3.76 / 0.62 / 0.06% | 4.16 / 0.55 / 0.03% | 6.83 / 0.37 / 0.03% |

All 24,246 genes, linear scale:

| Construction | 445 v 52 | 248 v 249 | 52 v 445 |
| --- | --- | --- | --- |
| exact | 3.53 / 0.60 / 0.056% | 4.26 / 0.83 / 0.077% | 6.74 / 1.46 / 0.144% |
| released | 3.72 / 0.73 / 0.103% | 4.31 / 0.90 / 0.112% | 6.12 / 1.18 / 0.130% |
| W1 fold, depth | 3.54 / 0.62 / 0.057% | 4.33 / 0.85 / 0.082% | 6.96 / 1.61 / 0.178% |
| mean-shift fold, scaled depth | 4.27 / 1.09 / 0.319% | 5.25 / 1.57 / 0.525% | 7.78 / 2.46 / 0.672% |
| log-mean (W2) fold, scaled depth | 3.59 / 0.63 / 0.058% | 4.33 / 0.86 / 0.084% | 7.03 / 1.56 / 0.163% |
| **proposed** | 3.50 / 0.58 / 0.045% | 4.21 / 0.78 / 0.065% | 6.76 / 1.36 / 0.101% |

Log2 scale, all genes (heavy genes in brackets); the depth is invariant there,
so only the registration differs:

| Construction | 445 v 52 | 248 v 249 | 52 v 445 |
| --- | --- | --- | --- |
| exact | 3.62 / 0.64 / 0.060% (3.73 / 0.77 / 0.06%) | 4.16 / 0.78 / 0.077% (4.83 / 1.14 / 0.06%) | 7.82 / 1.90 / 0.250% (7.51 / 1.75 / 0.09%) |
| released (mean shift) | 3.64 / 0.65 / 0.061% (3.73 / 0.80 / 0.06%) | 4.20 / 0.81 / 0.085% (4.93 / 1.23 / 0.09%) | 7.90 / 1.94 / 0.272% (7.60 / 1.82 / 0.15%) |
| W1 registration | 3.62 / 0.64 / 0.060% (3.76 / 0.80 / 0.06%) | 4.19 / 0.79 / 0.080% (4.90 / 1.14 / 0.06%) | 7.84 / 1.89 / 0.259% (7.51 / 1.82 / 0.15%) |

Reading the linear-scale tables against the requirements of section 5:

- **Both requirements met.** Only the proposed construction matches exact
  throughout.
- **Registration wrong (R2).** Registering by the mean shift's fold with the
  invariant statistic is the worst case. The pool's distorted shape is no
  longer partly offset by the released construction's scale coincidence
  (section 4.3).
- **Invariance missing (R1).** The robust fold with the raw depth is close to
  exact here but runs slightly high among all genes at 52 v 445 (up to 1.2
  times exact). Section 6.4 isolates this.
- **The Wasserstein-2 fold** leaks among the heavy genes when cases are few
  (52 v 445).

On the log2 scale all three registrations match exact: the depth is invariant
there, and a mean of log values is hard for a few samples to move.

### 6.2 Single genes

One gene's own values, relabelled at random 200 times at 445 v 52 (up to
99,999 shuffles). The first columns give the p-value with the real labels. The
others give the share of random labellings with p ≤ 0.05 / 0.01 / 0.001,
where 0.5% is one labelling.

| Gene | Largest tumours; normal maximum | Real-label p: exact / released / proposed | Exact | Released | Proposed |
| --- | --- | --- | --- | --- | --- |
| MC4R | 5,274, 25, 17, 11; 5 | 0.048 / 6.0e-5 / 0.18 | 4.5 / 1.0 / 0% | 53 / 45 / 27% | 0.5 / 0 / 0% |
| DRGX | 17,862, 26, 15, 11; 15 | 0.78 / 0.033 / 0.76 | 3.0 / 0 / 0% | 75 / 62 / 52% | 0 / 0 / 0% |
| AMBN | 4,229, 54, 36, 24; 2 | 0.11 / 7.0e-5 / 0.12 | 7.0 / 2.0 / 0.5% | 56 / 53 / 49% | 4.5 / 1.0 / 0% |
| IRS4 | 39,619, 28,021, 21,352, 6,425; 82 | 6.7e-3 / 2.0e-5 / 0.063 | 6.0 / 0.5 / 0% | 41 / 29 / 14% | 1.5 / 0 / 0% |
| ETV4 | 424,961, 217,808, 190,548, 137,820; 583 | 5.9e-3 / 1.0e-5 / 5.9e-3 | 3.5 / 1.0 / 0.5% | 16 / 10 / 8% | 4.0 / 1.0 / 0% |
| ETV1 | 172,764, 158,477, 112,162, 66,836; 2,126 | 5.2e-3 / 2.0e-5 / 6.1e-3 | 4.0 / 0 / 0% | 4.0 / 1.0 / 0% | 4.0 / 0 / 0% |
| ERG | 40,082, 38,873, 36,259, 35,691; 17,121 | 7.9e-3 / 0.011 / 0.010 | 6.5 / 0.5 / 0.5% | 6.5 / 2.5 / 0.5% | 6.5 / 1.5 / 0.5% |

- The proposed test is conservative for genes with a single extreme sample
  (MC4R, DRGX). With the real labels, its p-values for the fusion-driven genes
  ETV4, ETV1 and ERG are close to plain permutation's.
- ETV4's own values give the released test p ≤ 0.001 in 8% of random
  labellings, and its real-label p of 1.0e-5 is of that kind. The proposed
  test's 5.9e-3 is what 52 controls can support.

### 6.3 Simulated nulls

Six designs (cases v controls) × four distributions × three uniform folds on
the cases (×1, ×2, ×½), at a baseline of 50 counts. Latent levels are
Gamma-distributed (CV² 0.5) or lognormal (CV² 1). Two further distributions
add sparse tails to the Gamma: 1% of samples ×100, or 0.5% ×500, in either
group. Counts are Poisson with jitter, 1,500 genes per cell, up to 9,999
shuffles.

A cell's rejection rate has a standard error of about 0.56 points at 5% and
0.26 at 1%, so 7.5% and 1.5% lie about 4 and 2 standard errors above nominal.
The full grid is in the study output. The table gives, for each construction,
the largest rejection rate over the 36 cells of each kind at p ≤ 0.05 / 0.01,
and how many cells exceed 7.5% / 1.5%:

| Construction | Light tails: largest | Light tails: cells above | Sparse tails: largest | Sparse tails: cells above |
| --- | --- | --- | --- | --- |
| exact | 13.3 / 2.73% | 2 / 3 | 18.8 / 13.67% | 7 / 21 |
| released | 6.4 / 2.27% | 0 / 4 | 40.9 / 23.93% | 24 / 32 |
| W1 fold, depth | 7.5 / 1.73% | 1 / 4 | 8.5 / 1.73% | 3 / 1 |
| mean-shift fold, scaled depth | 8.2 / 3.47% | 2 / 11 | 69.8 / 67.20% | 24 / 36 |
| log-mean (W2) fold, scaled depth | 6.3 / 1.73% | 0 / 1 | 8.1 / 3.80% | 2 / 10 |
| **proposed** | 5.9 / 1.33% | 0 / 0 | 7.1 / 1.20% | 0 / 0 |

Plain permutation ("exact") fails only when there is a fold. Its largest rates
are 6.1 / 1.5% with no fold, 13.3 / 5.9% at ×2 and 18.8 / 13.7% at ×½: the
pool is then a mixture of G and a scaled copy of G.

The released and proposed constructions in the sparse-tail cells, by design
(range over the two tail distributions and three folds):

| Design | Released, p ≤ 0.05 | Released, p ≤ 0.01 | Proposed, p ≤ 0.05 | Proposed, p ≤ 0.01 |
| --- | --- | --- | --- | --- |
| 10 v 100 | 4.7–6.6% | 2.9–4.1% | 5.9–7.1% | 0.2–0.7% |
| 50 v 500 | 0.8–2.3% | 0.3–2.1% | 4.1–5.9% | 0.5–1.2% |
| 20 v 20 | 12.3–16.1% | 9.1–12.5% | 2.9–4.1% | 0.2–0.7% |
| 100 v 100 | 19.3–26.4% | 15.9–23.9% | 2.2–4.3% | 0.2–0.5% |
| 200 v 20 | 11.5–22.3% | 4.0–8.3% | 1.9–2.4% | 0.2–0.4% |
| 500 v 50 | 15.3–40.9% | 4.9–21.0% | 1.0–2.5% | 0.1–0.4% |

With sparse tails and at least as many cases as controls, the proposed test is
conservative (1–4% at p ≤ 0.05). The registered pool contains the tail values,
and shuffles that move them among the controls raise the null depths; that is
a power cost. With only ten cases it runs slightly high at p ≤ 0.05
(5.9–7.1%) but not at p ≤ 0.01 (0.2–0.7%).

### 6.4 Mechanism: correlated error, not error

Section 4.3 claims that the raw depth fails when the error of the
registration fold is correlated with the statistic, not merely because the
fold is estimated. Here the registration is fixed to the true fold, to the
true fold times independent lognormal noise with the same spread as the W1
estimate's error, and to the W1 estimate itself. Both statistics are run.
There are 3,000 genes per cell and no sparse tails; entries are
p ≤ 0.05 / 0.01.

| Design, distribution, fold | Depth, true fold | Depth, true fold × noise | Depth, W1 estimate | Scaled, W1 estimate (proposed) | Released |
| --- | --- | --- | --- | --- | --- |
| 20 v 200, Gamma, ×1 | 5.33 / 1.33% | 5.03 / 1.13% | 6.73 / 1.90% | 5.33 / 0.83% | 5.43 / 1.20% |
| 20 v 200, Gamma, ×2 | 4.70 / 1.00% | 4.40 / 1.03% | 5.87 / 1.50% | 4.53 / 0.53% | 4.77 / 0.83% |
| 20 v 200, lognormal, ×1 | 4.33 / 1.00% | 4.10 / 0.90% | 4.80 / 1.10% | 4.60 / 0.50% | 4.27 / 0.50% |
| 20 v 200, lognormal, ×2 | 4.57 / 1.27% | 4.20 / 1.20% | 5.17 / 1.37% | 4.83 / 0.50% | 3.93 / 0.83% |
| 50 v 500, Gamma, ×1 | 4.77 / 0.77% | 4.77 / 0.73% | 5.43 / 1.00% | 4.40 / 0.70% | 4.60 / 0.67% |
| 50 v 500, Gamma, ×2 | 4.77 / 1.10% | 5.10 / 1.13% | 6.43 / 1.70% | 5.37 / 0.90% | 5.37 / 1.10% |
| 50 v 500, lognormal, ×1 | 5.83 / 1.10% | 5.63 / 1.03% | 5.80 / 1.03% | 5.30 / 0.87% | 4.93 / 0.67% |
| 50 v 500, lognormal, ×2 | 5.87 / 1.30% | 5.73 / 1.30% | 6.27 / 1.50% | 5.73 / 1.17% | 5.10 / 1.10% |
| 200 v 20, Gamma, ×1 | 5.17 / 1.03% | 5.07 / 1.07% | 5.27 / 1.17% | 4.93 / 1.03% | 5.23 / 1.17% |
| 200 v 20, Gamma, ×2 | 5.33 / 1.00% | 5.17 / 0.90% | 5.30 / 0.93% | 5.27 / 0.90% | 5.33 / 1.00% |
| 200 v 20, lognormal, ×1 | 4.90 / 0.80% | 5.00 / 0.83% | 5.17 / 0.70% | 5.03 / 0.73% | 5.20 / 0.90% |
| 200 v 20, lognormal, ×2 | 4.83 / 0.93% | 5.07 / 0.93% | 4.93 / 0.90% | 4.83 / 0.77% | 5.27 / 1.03% |

- **Noise alone does no harm.** Independent error in the registration fold
  changes nothing.
- **The estimate's own error does.** With the raw depth, the W1 estimate runs
  high when cases are few, most with the right-skewed Gamma: up to 1.90%
  against 1.33% at p ≤ 0.01.
- **Scaling removes it.** The scaled statistic removes the excess with every
  registration. With the scaled statistic, the true fold and the noisy true
  fold (not shown) gave 4.0–6.0% / 0.7–1.4% across these cells.
- **The released construction is close to nominal here**, because these
  light-tailed genes are where the mean-shift fold's coincidence holds
  (section 4.3).

### 6.5 Power

The generator WADE uses for its confirmation study plants known patterns among
16,384 simulated genes. The baseline expression of the planted genes is 0.5,
5, 50 or 500 counts. The biology alternates between Gamma–Poisson and
correlated lognormal–Poisson, and size factors vary. Here it runs on fresh
seeds: six matrices per design, each with planted patterns ("mixed") and six
without ("complete nulls"). Each cell below is the share of planted genes with
baseline ≥ 5 counts (18 per cell) called at BH q ≤ 0.05, shown as
released / W1 fold with raw depth / proposed.

| Planted pattern | 250 v 250 | 500 v 50 | 50 v 500 |
| --- | --- | --- | --- |
| Every case ×½ and a tenth ×1,024 instead | 100 / 100 / 100% | 89 / 0 / 0% | 100 / 100 / 100% |
| … a tenth ×128 | 100 / 100 / 100% | 94 / 0 / 0% | 100 / 100 / 78% |
| … a tenth ×32 | 100 / 100 / 100% | 56 / 0 / 0% | 94 / 94 / 56% |
| … a tenth ×8 | 100 / 100 / 94% | 22 / 0 / 0% | 78 / 72 / 17% |
| A tenth ×16 | 100 / 100 / 100% | 22 / 0 / 0% | 83 / 67 / 44% |
| A fifth ×4 | 50 / 56 / 39% | 11 / 6 / 6% | 22 / 28 / 17% |
| A tenth ×4 | 6 / 6 / 0% | 0 / 0 / 0% | 0 / 0 / 0% |
| A tenth ×4 with the mean held fixed | 78 / 72 / 61% | 0 / 0 / 0% | 39 / 22 / 11% |
| A fifth ×4 and another fifth ×¼ | 89 / 89 / 83% | 22 / 6 / 0% | 44 / 44 / 22% |
| The same mean with a wider spread | 83 / 89 / 83% | 44 / 22 / 11% | 61 / 61 / 17% |
| A twentieth ×4; a tenth ×¼; one case ×16 | 0% throughout | 0% throughout | 0% throughout |
| Every case ×1.5 or ×⅔ (no subset; should not be called) | 0% throughout | 0% throughout | 0% throughout |
| False discovery proportion in mixed matrices (mean calls) | 6.3% (30.2) / 6.2% (30.3) / 3.3% (27.8) | 6.1% (13.2) / 4.2% (1.2) / 0% (0.5) | 5.3% (23.8) / 14.2% (25.2) / 3.7% (14.5) |
| Complete-null matrices with any call | 0 / 0 / 0 of 6 | 0 / 0 / 0 of 6 | 0 / 0 / 0 of 6 |

- **Balanced designs.** The proposed test keeps the released test's power on
  the patterns it was built for, and makes about half as many false
  discoveries.
- **Few controls (500 v 50).** The released test's power was its invalidity
  (sections 4.2 and 6.1). With 50 controls, no subset of a tenth of the cases
  reaches genome-wide significance within one gene.
- **Few cases (50 v 500).** The proposed test loses power on moderate
  patterns: the ×8 pattern falls from 78% to 17%, and a tenth ×16 from 83% to
  44%. The released test is valid on these simulated light-tailed genes (false
  discovery proportion 5.3%), though not on heavy-tailed real genes
  (section 6.1). The W1 fold with the raw depth, which fails R1, reaches a
  14.2% false discovery proportion. Keeping validity and recovering this
  power is open question 3.
- The simulations have no sparse extreme tails, so they do not show the
  released test's main failure.

### 6.6 TCGA with the real labels

Both cohorts with their real labels (up to 99,999 shuffles, so the smallest
possible p is 1.0e-5), p-values with BH q in brackets:

| Gene | Released | Proposed |
| --- | --- | --- |
| *Prostate, 445 v 52: subset calls at q ≤ 0.05* | 6,093 | 5,907 |
| ERG | 1.1e-2 (0.046) | 1.0e-2 (0.045) |
| ETV1 | 2.0e-5 (6.5e-4) | 6.1e-3 (0.031) |
| ETV4 | 1.0e-5 (4.1e-4) | 5.9e-3 (0.031) |
| ETV5 | 4.5e-4 (5.3e-3) | 1.5e-2 (0.057) |
| FLI1 | 0.52 (0.66) | 0.54 (0.68) |
| SPINK1 | 0.050 (0.13) | 0.040 (0.11) |
| MC4R | 6.0e-5 (1.3e-3) | 0.18 (0.33) |
| AMBN | 7.0e-5 (1.5e-3) | 0.12 (0.24) |
| IRS4 | 2.0e-5 (6.5e-4) | 0.063 (0.16) |
| DEFA5 | 1.0e-5 (4.1e-4) | 8.6e-3 (0.040) |
| *Breast, 971 v 112: subset calls at q ≤ 0.05* | 20,347 | 20,267 |
| ERBB2 | 5.0e-5 (9.6e-5) | 1.2e-4 (2.2e-4) |
| GRB7 | 2.0e-5 (4.1e-5) | 5.0e-5 (9.9e-5) |
| STARD3, PGAP3, MIEN1, FGFR1, CCND1 | 1.0e-5 (2.2e-5) | 1.0e-5 (2.3e-5) |

- **The ETS fusion genes** (ERG, ETV1, ETV4, ETV5; each switched on by a gene
  fusion in a share of prostate tumours) get p-values of 0.006–0.015 from the
  proposed test. That is what 52 normal samples can support. ETV1, ETV4 and
  ERG are still called only because thousands of genes are.
- **Genes with one or a few extreme tumours** (MC4R, AMBN, IRS4) are no longer
  called. DEFA5 is, on several tumours at 700–1,700 against a normal maximum
  of 8.
- **In breast**, with 112 normal samples, the ERBB2 amplicon genes stay highly
  significant and the number of calls barely moves.
- **Saturation.** In both cohorts most genes are called by either
  construction: tumour purity and subtype make almost every gene's tumours
  depart from a uniform fold. Ranking genes then needs more than this test;
  we set that aside here.

### 6.7 Low counts

H₀ applies the fold to observed expression (count + 1). A uniform change of
the latent, biological level is a different thing once Poisson noise matters.
Two experiments measure the gap. Every construction shares it.

**Real counts.** TCGA prostate tumours were split at random, 222 v 223 and
45 v 400. In a random half of the genes, the second group's counts were
thinned binomially by ½. For any Poisson mixture this makes every case carry
exactly the same 2× change of its latent level. Genes are binned by their mean
over all tumours before thinning, a level that depends on neither the split
nor the thinning. Binning instead by the thinned group's own mean selects genes
whose controls happened to come out low; an early exploratory analysis did
that and overstated the effect about threefold. Each cell gives the share at
p ≤ 0.01 as exact / released / proposed.

| Mean count before thinning (genes with the change / unchanged) | 222 v 223: 2× change | 222 v 223: unchanged | 45 v 400: 2× change | 45 v 400: unchanged |
| --- | --- | --- | --- | --- |
| below 2 (5 / 10 and 9 / 6 genes) | 0 / 0 / 0% | 0 / 0 / 0% | 11 / 11 / 11% (one gene) | 0 / 0 / 0% |
| 2–5 (133 / 141 and 139 / 135) | 6.8 / 6.8 / 6.0% | 1.4 / 1.4 / 1.4% | 7.2 / 1.4 / 4.3% | 1.5 / 2.2 / 1.5% |
| 5–20 (2,486 / 2,599 and 2,537 / 2,548) | 4.5 / 4.2 / 3.8% | 1.2 / 1.1 / 1.2% | 3.5 / 1.5 / 1.9% | 1.0 / 1.0 / 0.7% |
| 20–100 (2,231 / 2,264 and 2,185 / 2,310) | 2.1 / 1.8 / 1.2% | 1.1 / 1.3 / 1.1% | 1.6 / 0.9 / 0.8% | 1.0 / 1.1 / 0.7% |
| 100 and above (7,279 / 7,098 and 7,143 / 7,234) | 1.1 / 1.2 / 1.1% | 1.2 / 1.2 / 1.2% | 0.6 / 0.7 / 0.8% | 1.0 / 0.8 / 0.9% |

**Simulated counts.** Gamma–Poisson genes (CV² 0.5), every case carrying the
same fold of its latent level, 3,000 genes per cell. Each cell gives the share
at p ≤ 0.05 / 0.01 with every case ×2, as exact / released / proposed. With
no change (×1), all three constructions gave 4.0–5.6% / 0.6–1.2% in every
design and baseline.

| Baseline count | 100 v 100 | 250 v 250 | 50 v 500 | 200 v 20 |
| --- | --- | --- | --- | --- |
| 1 | 35.8/13.3 · 33.3/13.4 · 31.9/11.6 | 67.2/35.5 · 62.2/31.4 · 60.8/29.8 | 43.1/19.5 · 30.9/10.5 · 29.1/10.6 | 16.0/4.4 · 16.3/4.7 · 16.1/4.3 |
| 2 | 20.5/5.8 · 18.8/6.0 · 18.1/4.4 | 42.1/17.4 · 36.9/13.6 · 35.5/12.3 | 31.9/11.2 · 18.7/4.9 · 17.5/4.7 | 10.7/2.7 · 11.0/2.8 · 10.4/2.4 |
| 5 | 10.1/2.3 · 8.9/2.4 · 8.3/1.9 | 18.3/5.3 · 14.5/3.6 · 14.1/3.2 | 19.6/6.2 · 10.1/2.6 · 9.5/2.3 | 8.1/2.0 · 8.2/2.1 · 8.0/2.0 |
| 20 | 7.2/1.7 · 5.9/1.4 · 5.6/1.1 | 9.8/2.4 · 7.6/1.6 · 7.5/1.3 | 13.4/3.1 · 5.7/1.3 · 5.8/1.1 | 6.1/1.3 · 6.2/1.4 · 5.9/1.2 |

At low counts a uniform biological fold is read as a departure from H₀. The
effect grows with sample size and fades with depth. This limits what the
subset test can claim for weakly expressed genes, whatever the registration;
it is open question 4.

## 7. Open questions and concerns

We would value the reviewer's view on each of these.

1. **Validity is measured, not proven.** The registered pool is exchangeable
   only if $\hat g = f$, and $\hat g$ depends on the labels. We have no
   finite-sample bound on the size distortion and no asymptotic proof; we rely
   on the measurements in section 6. Is there a result covering a permutation
   test after robust alignment, with a statistic invariant under the alignment
   group, that would make it asymptotically exact? Aligned rank tests (Hodges
   and Lehmann, 1962) and studentized permutation tests (Chung and Romano,
   2013) are the nearest we know. What conditions on $\hat g$, for example
   $n^{1/2}$-consistency under H₀, would be needed? Alternatively, a
   Berger–Boos construction would give validity outright at extra
   computational cost.
2. **Is Wasserstein-1 on log(count + 1) the right registration?** Under H₀
   any consistent registration will do; the choice matters through robustness
   and through the pool's shape when H₀ is false (power). In exploratory runs
   the Hodges–Lehmann estimator (median of all pairwise log ratios, costing
   $O(n_1 n_0)$) behaved like the Wasserstein-1 registration. The
   Wasserstein-2 registration leaks when cases are few and tails are sparse.
   Is there an argument for Wasserstein-1 beyond robustness and consistency
   with WADE's use of area?
3. **The normalizer, and power when cases are few.** S divides by the median
   case. With 50 cases, the median's sampling error adds noise to S and costs
   power (section 6.5). Is there an invariant normalizer with smaller variance
   that a subset still cannot inflate? Would studentizing D be better? Or is
   the power loss the honest price of not knowing f?
4. **Low counts.** H₀ applies the fold to observed values, which is not the
   biological uniform change once Poisson noise matters (section 6.7). Two
   candidates:
   - a count-consistent null by binomial thinning: under H₀, thinning the
     higher group's counts by the fold reproduces the lower group's
     distribution exactly for any Poisson mixture;
   - a variance-stabilizing scale derived from a mean–variance relation fitted
     across genes, on which a uniform change is approximately a shift.

   Neither is settled, and both interact with registration.
5. **Few controls.** When controls are few relative to cases, a test within
   one gene cannot tell a tail that only cases have from a shared tail that the
   controls happened to miss. For example, 52 of 497 exchangeable samples miss
   all 12 members of a tail with probability 0.26. The proposed test
   respects this limit; the released one appeared to beat it. Can information
   across genes (a dataset-wide model of tails) be borrowed without giving up
   validity?
6. **What real data can test.** Random labels test only f = 1. Our real-data
   check with f ≠ 1 is a single binomial-thinning experiment (f = 2,
   section 6.7); simulations cover other folds with parametric shapes. What
   would a convincing real-data validation at f ≠ 1 look like?
7. **The fold model rests on between-gene evidence** (section 3). Within a
   gene, a uniform change could also alter the shape (saturation, bursty
   transcription).
8. **The description still uses the mean shift.** The walk's prediction
   carries the observed mean shift, so for a heavy-tailed gene the subset can
   be driven by a few samples. The test is now valid for such genes, but is a
   subset defined relative to the mean shift the most useful description?
   This lies outside the null, but a reviewer may have views.
9. **Multiple testing.** The p-values are approximately valid gene by gene.
   Benjamini–Hochberg across correlated genes is unchanged from the released
   method.

## 8. Reproduction

The script and the package code it needs are in git at tag
`analysis-2026-09-30`; run it from a checkout of that tag.

```sh
WADE_TCGA_CONFIG=.release/0.2.0/tcga-config.json PYTHONPATH=src:research \
    python research/subset_null_study.py > .release/0.2.0/subset-null-study-20260929.txt
```

The script implements every construction compared here; its sections can be
run separately (`checks`, `foldform`, `realnull`, `genes`, `simnull`,
`mechanism`, `power`, `truelabels`, `thinning`, `lowcount`). The TCGA sections read the
lab's rna100k cohort bundles (GENCODE v44 gene counts; see
[the TCGA examples](tcga-examples.md)). The simulations use WADE's
confirmation generator (`research/wade_research/point_simulation.py`) with the
patterns of confirmation protocol version 6 and a fresh seed. The full run
takes two to three hours on a 16-core laptop.

## Appendix A. Proofs

**Lemma 1.** For c > 0, $D(T_c(x), y) = c\,D(x, y)$, and the cut is
unchanged.

*Proof.* Write $x' = T_c(x)$, so $x'_i + 1 = c\,(x_i + 1)$. $T_c$ is
increasing, so the order of the cases is preserved and
$Q'_1(u) + 1 = c\,\{Q_1(u) + 1\}$. Then $\bar x' + 1 = c\,(\bar x + 1)$ and
$\hat f' = c\hat f$, so the prediction satisfies
$\hat Q'_1(u) + 1 = \hat f'\{Q_0(u) + 1\} = c\,\{\hat Q_1(u) + 1\}$. Hence

$$R'(k) = \int_0^{k/n_1} \big[\{Q'_1(u) + 1\} - \{\hat Q'_1(u) + 1\}\big] du = c\,R(k) .$$

Distinct case values stay distinct, so the allowed boundaries are the same.
The minimum scales by c and is attained at the same k. ∎ (The software's
rounding threshold scales only approximately; the study's `checks` section
finds agreement to a relative error of $3 \times 10^{-13}$.)

**Lemma 2.** If $m(T_c(x)) + 1 = c\,\{m(x) + 1\}$, then
$S = D/\{m(x) + 1\}$ is invariant. Every sample quantile satisfies the
condition, because $T_c$ maps each order statistic $x_{(i)}$ to
$c(x_{(i)} + 1) - 1$; so does the average of two order statistics.

*Proof.* Divide Lemma 1 by the condition. ∎

**The median case lies on the larger side of the split.** Let the cut be
$k^*$.

- For a raised subset, $n_1 - k^* \le k^*$, so $k^* \ge n_1/2$ and the median
  order statistic, or for even $n_1$ the lower of the two middle ones, is
  among the $k^*$ cases below the cut.
- For a lowered subset, $k^* < n_1/2$ and the median is among the cases above
  the cut.

When the sides are equal, the two middle cases straddle the cut. ∎

**The Wasserstein-1 registration is a weighted median.** For one-dimensional
distributions with quantile functions $F^{-1}$ and $G^{-1}$,
$W_1(F, G) = \int_0^1 |F^{-1}(u) - G^{-1}(u)|\,du$. Shifting one distribution
by t shifts its quantile function by t, so the shift minimizing $W_1$
minimizes $\int_0^1 |a(u) - t|\,du$ with $a = F^{-1} - G^{-1}$. That is
minimized at a median of a(u) under u ~ Uniform(0, 1). For empirical step
quantile functions, a is constant on each segment between consecutive points
of $\{k/n_1\} \cup \{j/n_0\}$, so any weighted median of the segment values,
weighted by segment length, is a minimizer. Here $F^{-1} = \log(Q_1 + 1)$,
$G^{-1} = \log(Q_0 + 1)$ and $t = \log g$. ∎

**Adjusting controls up or cases down gives the same shuffles.** Let
$P = \{T_g^{-1}(x_i)\} \cup \{y_j\}$ and $P' = T_g(P)$. For any labelling,
the first construction's new cases are $T_g$ applied to the selected members
of P, which are the selected members of P'. Its new controls are the other
members of P, which are $T_g^{-1}$ of the other members of P'. These are
exactly the data sets built from P' by keeping the new cases and mapping the
new controls back. ∎

## References

- Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery rate:
  a practical and powerful approach to multiple testing. *Journal of the Royal
  Statistical Society B* 57, 289–300.
- Berger, R. L. and Boos, D. D. (1994). P values maximized over a confidence
  set for the nuisance parameter. *Journal of the American Statistical
  Association* 89, 1012–1016.
- Besag, J. and Clifford, P. (1991). Sequential Monte Carlo p-values.
  *Biometrika* 78, 301–304.
- Chung, E. and Romano, J. P. (2013). Exact and asymptotically robust
  permutation tests. *Annals of Statistics* 41, 484–507.
- Hodges, J. L. and Lehmann, E. L. (1962). Rank methods for combination of
  independent experiments in analysis of variance. *Annals of Mathematical
  Statistics* 33, 482–497.
- Lehmann, E. L. and Romano, J. P. (2005). *Testing Statistical Hypotheses*,
  3rd edition. Springer.
