# Limits and failure modes

What WADE cannot do, and the two constraints that decide whether it can answer
your question at all. Read this **before** running it, not after: the first two
sections are arithmetic on your design, not properties of the implementation,
and no amount of data or computation moves them.

The method itself is in [`method.md`](method.md).

---

## 1. The two hard constraints, stated first

**Resolution.** The quantile grid has `m = min(n_case, n_ctrl, max_probs)`
points — the smaller group, or the cap of 2,000 on a very large cohort,
whichever is less ([`method.md`](method.md) §1). Nothing finer than `1/m` is
estimable, so `affected_fraction` is a quantitative estimate only above roughly
100 samples in the smaller group, degrades through about 50, and below that
distinguishes "global" from "concentrated" without quantifying which fraction.

**Detectability.** If `k` samples carry a signal, label shuffling puts all of
them in one group with probability `C(n1,k) / C(n1+n0,k)`, and **that sets the
scale of the smallest p-value the design can support** — whatever the effect
size, the statistic or the permutation count. It is not a hard bound on any
individual gene (§4.1 measures how far under it a gene can land), but where it
exceeds your alpha the signal is undetectable by this family of methods. Check
it for your design before you run anything.

`wade.detectability_floor(n_case, n_ctrl, k)` computes it, so the check is one
call rather than a formula to transcribe:

```python
>>> wade.detectability_floor(77, 18, 15)      # 15 affected of 77 cases
0.031963032251420324
>>> wade.detectability_floor(48, 47, 15)      # the same 95 samples, balanced
9.904929906140583e-06
```

Both are properties of the *design*. They are the reason a study with 18
controls cannot find a 5% subtype, no matter how dramatic the subtype is — and
the second call above is why **balancing the groups buys more than adding
cases**.

---

## 2. What WADE does not do

### 2.1 No covariate or batch adjustment

The test takes a binary condition vector and nothing else. There is no design
matrix, no offsets, no random effects. A confound that tracks the case/control
split is indistinguishable from signal.

This is not an oversight to be patched: the permutation null is built by
exchanging labels, and a covariate makes labels non-exchangeable. Adjusting
would require a different null — a stratified or restricted permutation scheme.

**One such scheme now exists**, for a *categorical* confounder:
`wade(..., strata=labels)` shuffles labels only within each stratum, so study,
batch or protocol is held fixed instead of being tested as though it were
biology. It is not covariate adjustment — there is still no design matrix, no
continuous covariate and no random effect — and it is not free: within-stratum
exchangeability shrinks the permutation space to a product of much smaller
numbers, and a stratum containing only one class contributes no freedom at
all. `wade.permutation_space()` reports what is left, and the manifest records
it beside the results. A design stratified into many small studies can have a
space too small to support the p-values asked of it.

**What to do otherwise.** Detect the confound and refuse, or run within each
level of the confounder and combine afterwards. Refusing a confounded contrast
is more honest than reporting one.

### 2.2 No repeated-measures handling

Samples are assumed exchangeable under the null. Two libraries from the same
subject are not. Feeding them in inflates the effective sample size and
anticonservatively biases every p-value, silently.

If your design has repeated measures, aggregate to one value per subject before
running. `strata=` does **not** solve this case: it exchanges labels *within* a
stratum, whereas repeated measures need whole subjects exchanged *between*
groups — the subject is the unit, and a per-subject stratum has only one class
in it, so the permutation space collapses to nothing. `permutation_space()`
will say so (`uninformative_strata` equal to the number of subjects), which is
the honest failure rather than a silent one.

### 2.3 The subset stage at low expression: what it does and does not fix

The subset stage tests "is a global fold change an adequate explanation?" by
building a null in which the two groups are exchangeable under that
hypothesis. For raw counts it does this by **binomial thinning**
(`method.md` §10.3), which is exact for counts where the division it replaced
was not; before that fix a genuine 2× shift was called "not a global shift"
0.95 of the time at 2 counts and 0.72 at 20 counts with 1000 v 1000. Measured
after: 0.02–0.05 at every expression level and sample size tried.

Two things this does **not** fix, both properties of the data rather than the
method:

* **`thin=False` cannot thin.** It has no counts to draw from, so it keeps the
  division and inherits the old behaviour at low expression. If your matrix
  is count-derived and sparse, go through `wade()` with the counts.
* **Below about five counts per sample the characterization is
  noise-dominated.** On the log scale a Poisson count of 2 carries roughly
  ±0.7 per node, so `affected_fraction` reads a global 2× as 0.70 rather than
  1.0, a 5% subset at 8× as 0.10 rather than 0.05, and a null gene as 0.08.
  The ordering survives; the numbers are not quantitative. Run with
  `n_boot=300` and read the interval, which is wide exactly there.

### 2.4 Normalization lives inside the test, and couples genes

WADE ships its own normalization (`method.md` §8) because the continuity jitter
has to be applied at count precision. That means a normalization decision is
being made on your behalf inside a statistical test.

**The consequence that surprises people**: library-size normalization divides
every gene by a column total that every gene contributes to, so **one gene's
normalized values depend on all the others**. Measured — the same gene, with
byte-identical raw counts, in two matrices: alone with 200 null genes its
`direction` read 0.276; adding five strongly-up genes inflated the case
libraries by 8.4% and it read 0.014. The gene did not change; its denominator
did.

Consequences for reading the output:

* Null genes acquire a small consistent fold change, so their
  `affected_fraction` drifts toward 1 — relative to the library they genuinely
  *are* globally shifted — and their `direction` collapses toward ±1.
* **The subset test is immune.** Its statistic is a bridge that is exactly
  invariant to a global offset, which is precisely what composition produces.
* The characterization statistics are not immune, and neither is `mean_shift`.

This affects every differential method on normalized data. WADE does not
introduce it and cannot remove it. What you can do is be suspicious of a
characterization read off a matrix in which a large fraction of genes carry
strong signal.

### 2.5 `mean_shift` is a grid quadrature, not a difference of means, on unequal groups

`mean_shift` is the signed area between the two quantile functions, evaluated as
an equal-weight average over `m` nodes. When `n_case = n_ctrl` this is *exactly*
the difference of the two sample means. When the groups are unbalanced it is
not: the larger group is read at only `m` probabilities, and the two extreme
order statistics each carry weight `1/m` against the sample mean's `1/n`, so the
over-weighted extremes bias it in the direction of that group's skew.

**Inference is unaffected** — the permutation null is computed with the same
estimator on the same grid, so the bias is common to observation and null and
cancels in the p-value. What it affects is the **effect size**: read
`mean_shift` as "signed quantile area", not as a drop-in estimate of the mean
difference, when your groups are unbalanced.

---

## 3. The grid is a sample-size constraint, not a parameter

`m = min(n_case, n_ctrl)` is a ceiling you cannot raise. It is the largest grid
on which at least one group is read without interpolation, and it sets
everything downstream. `max_probs` is the one knob here and it only ever
*lowers* it — a memory cap for very large cohorts, not a resolution dial
([`scaling.md`](scaling.md) §2.1). The rows below all sit under the default cap
of 2,000. Measured, planted fractions against the estimate:

| geometry | m | 2% | 5% | 10% | 25% | global |
|---|---|---|---|---|---|---|
| 2000 v 2000 | 2000 | 0.021 | 0.052 | 0.108 | 0.280 | 0.997 |
| 500 v 500 | 500 | 0.028 | 0.054 | 0.106 | 0.274 | 0.992 |
| 100 v 100 | 100 | 0.080 | 0.080 | 0.121 | 0.279 | 0.957 |
| 77 v 18 | 18 | 0.131 | 0.117 | 0.146 | 0.269 | 0.890 |

At 77 v 18 the 2% and 5% rows are indistinguishable. The number is still
*computed* — it is always computable — but it is not a fraction estimate at that
geometry, and reporting it as one would be false precision.

**The smaller group governs.** Adding cases to a study with 18 controls buys
nothing here.

---

## 4. The combinatorial floor

### 4.1 The formula

If exactly `k` of the `n1` case samples carry a signal, the probability that a
uniform relabelling assigns all `k` of them to the case group is

```
C(n1, k) / C(n1 + n0, k)
```

**It is a scale, not a bound** — and this file said otherwise until
2026-09-02, when it was measured. The floor is exactly the p-value in the
noise-free limit: if the statistic depended *only* on how many affected samples
land in the case group, every relabelling that puts all `k` in cases would tie
the observed, and `p` would equal that probability exactly.

Real data is not noise-free. The unaffected samples are random too, so those
relabellings scatter above and below the observed rather than tying it, and
roughly half fall below. Worse, **`k` itself is not well defined in count
data**: dispersion makes some unaffected samples look extreme, so the number of
samples carrying the signal is a random variable, not the planted constant.

Measured on planted NB counts at 300 v 300, with `B = 40,000` so that every
p-value is a raw permutation count and no refinement fires:

| planted `k` | floor | `p` / floor, q10 – median – q90 | fell **below** the floor |
|---|---|---|---|
| 3 | 1.24e-01 | 0.48 – 1.96 – 3.23 | 21% |
| 6 | 1.52e-02 | 0.26 – 1.10 – 1.71 | 42% |

So the floor predicts the **order of magnitude** of the smallest p-value the
design can support — the median lands within a factor of two of it — and a
substantial minority of genes come in under it. Use it to decide whether a
design can find a subtype at all, which is what §4.3 does with it. Do not use
it as an assertion about any individual gene's p-value, and do not clamp a
p-value to it.

### 4.2 It is driven by imbalance, not by subset size alone

This is the part that is easy to get wrong. The floor is roughly
`(n1/n)^k` — it depends on the *proportion* of samples in the case group, not
just on `k`. A balanced design drives it down fast; an unbalanced one does not.

At 77 cases vs 18 controls, `n1/n = 0.81`, so each additional signal-carrying
sample multiplies the floor by only about 0.81. Fifteen of them still leave it
at 0.032. At 60 v 60 the multiplier is 0.5 and six samples reach 0.014.

Measured at 60 v 60 with the subset stage, alpha = 0.05:

| affected | k | floor | power |
|---|---|---|---|
| 2% | 1 | 0.5 | 0.033 |
| 5% | 3 | 0.122 | 0.050 |
| 10% | 6 | 0.0137 | 0.467 |
| 25% | 15 | 1.1e-05 | 1.000 |

Power tracks the floor, not the effect size — the planted effect was 8x in every
row.

### 4.3 The practical upshot

Compute the floor for the smallest subset you care about, before running
anything:

```python
from math import lgamma
import numpy as np

def floor(n1, n0, k):
    lc = lambda n, r: lgamma(n+1) - lgamma(r+1) - lgamma(n-r+1)
    return np.exp(lc(n1, k) - lc(n1 + n0, k))
```

If it exceeds your alpha — or your BH threshold, which is stricter — the study
cannot find that subtype and a different design is the only remedy. **Balance
the groups if you can.** It helps more than adding cases.

---

## 5. A p-value at the extrapolation floor is not a small number

The GPD refinement floors its output at `1/(B * n_tail)` — 2.0e-6 at the
defaults. That is a statement about what `B` permutations can support, not a
numerical guard.

A gene sitting *at* the floor has not been measured at 2e-6. It has been
measured as "beyond what this many permutations can resolve", and the floor is
the most extreme claim the evidence supports. Two genes both at the floor are
not tied in strength; they are both unresolved.

**The reading rule.** Treat the floor as a censoring point. If ranking genes by
p-value puts several at the floor, break the tie with the effect size and the
characterization, not by pretending the p-values differ. `subset_log2_fc`,
`log2_fc` and the permutation z-scores (`z_mean_shift`, `z_subset`) exist for
exactly this, and on a large cohort they are the operative outputs rather than
a fallback — see `scaling.md` §7.1, where a real contrast put 18.8% of genes at
the floor. If you need to resolve *below* it, the only remedy is more
permutations — and the floor moves as `1/B`, so an order of magnitude costs an
order of magnitude.

**The refinement was checked against brute force, and is slightly
conservative.** The worry was that the GPD extrapolates below what the
permutations can support. Measured on 40 planted genes at 300 v 300, comparing
the shipped refinement at `B = 2,000` against the empirical p-value at
`B = 50,000` — where no refinement fires and the count is the answer:

| | refined / true |
|---|---|
| median | **1.19** |
| worst under-statement | 0.53 |
| worst over-statement | 17.2 |

The refinement errs high, not low: the median refined p-value is 19% *larger*
than the truth, and the one badly wrong gene was wrong by being 17× too
conservative. Nothing here is evidence that the GPD tail is optimistic over
the range it was measured on, `p` around 1e-2 to 1e-3. Its accuracy further out
— 1e-6 and below, where it matters on a large cohort — is **not** established
by this and is the experiment in `scaling.md` §4.5.

**Which p-values were extrapolated is reported.** `refined_mean_shift` and
`refined_subset` (both in the written table) say whether a stage's p-value was
counted from permutations or read off the GPD tail fit. That distinction
matters most for exactly the genes this section is about, and it is worth
knowing that refinement can move a p-value in **either** direction: it replaces
a coarse count with a model estimate, so a refined p-value is not necessarily
smaller than an unrefined one.

A port that "improves" the floor by returning smaller values has removed an
honesty constraint, not added resolution.

---

## 6. Checklist: when not to use WADE

Work down this list before running. Any "yes" is a reason to stop.

1. **Is the smaller group under 10?** The grid has fewer than 10 points. Stop.
2. **Is the combinatorial floor above your threshold** for the smallest subset
   you care about? Nothing in this family can find it. Stop, or rebalance.
3. **Do you have a covariate or batch effect** that tracks the contrast? Detect
   and refuse, or stratify.
4. **Do you have repeated measures?** Aggregate first.
5. **Do you need a fraction estimate** and have fewer than ~50 per group?
   `affected_fraction` will separate global from concentrated but will not tell
   you 2% from 5%.
6. **Is a large share of your genes strongly differential?** The
   characterization will be distorted by composition. The subset test's
   significance is not.
7. **Do you actually have a location shift?** If the question is "is the mean
   different", an ordinary DE method answers it and is better understood.
   WADE's value is the second question, not the first.
