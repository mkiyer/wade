# WADE — limits and failure modes

> **Note on citations.** This document cites `reference/docs/` and
> `reference/R/downstream/`, which were pruned once the port was verified.
> They are quoted here as the provenance of specific claims; the full record,
> including digests, is in [`../reference/PROVENANCE.md`](../reference/PROVENANCE.md).

What WADE does not do, and where it breaks down. This document exists to protect
a port's future users from overreading its output. It is the counterweight to
[`algorithm.md`](algorithm.md), which says what the method computes, and
[`rationale.md`](rationale.md), which says why.

Two constraints in here are **hard**: they are properties of the experimental
design, and no implementation choice, permutation count, or model improvement
moves them. The rest are stated limitations of the method as written, one live
numerical defect, and one worked example of how to misread a p-value.

An honest summary before the detail. WADE is a two-group, one-sided,
non-covariate-adjusted test whose resolution is set by the size of the *smaller*
group, and whose ability to detect a subset signal at all is bounded by a
combinatorial floor determined by group sizes before any data are collected.
Within those bounds it is well calibrated ([`rationale.md`](rationale.md) §9).
Outside them it returns numbers that look like results.

---

## 1. The two hard constraints, stated first

| constraint | what sets it | what does not fix it |
|---|---|---|
| **Quantile grid resolution.** The comparison happens at $m = \min(n_0, n_1)$ points, and the subset window is $k = \max(1, \lceil q_{\text{tail}} m \rceil)$ of them. | The smaller group's size. | Lowering $q_{\text{tail}}$ (the $\max(1,\cdot)$ floor still returns a single point); adding samples to the *larger* group; any amount of sequencing depth. |
| **Combinatorial permutation floor.** If $k^{*}$ samples carry a signal, no label-permutation test can return a p-value below $\binom{n_1}{k^{*}}\big/\binom{n_1+n_0}{k^{*}}$. | Group sizes and their balance. | More permutations; a better tail fit; a larger effect size; a different test statistic, as long as inference is by label permutation. |

Both are consequences of the design, evaluable **before** the experiment runs.
Both are developed below (§3 and §4).

---

## 2. What WADE does not do

Four stated limitations, taken from
`../reference/docs/supplement_wade.qmd`
§S7.6, plus a fifth established while checking the algorithm this session. Each
is a live constraint, not a hypothetical.

### 2.1 No covariate adjustment

WADE is a two-group test. There is no covariate term, no batch term, no depth
term, no patient random effect, and no facility for adding one — the statistic is
a functional of two empirical quantile functions, and there is nowhere in that
construction for a nuisance parameter to enter.

The consequence is sharper than "adjust beforehand": **a confound can be
detected but not removed.** If a batch variable is associated with the group
assignment, every quantile of the affected group shifts and the method
manufactures signed area. There is no residualization step that fixes this
inside the test, because the test does not accept a design matrix.

The source project's response is the pattern worth carrying, and it is a
*workflow* answer rather than a statistical one:

- **Screen before running**, measuring the association between group and the
  suspected confound (in cfRNA: Cramér's V on group × probe-design version,
  and a depth comparison).
- **Refuse the contrast rather than correct it** when the association is strong
  — above Cramér's V of 0.50 in cfRNA, described in the source as a conventional
  "strong association" mark. The verdict vocabulary distinguishes a contrast that
  is *underpowered* (still interpretable; report effect sizes, exclude from the
  nomination product) from one that is *design-confounded* (not interpretable at
  any sample size). The second cannot be rescued by collecting more samples,
  which is why refusing is the correct response.
- **Match by restricting libraries** where a subset of the data is unconfounded
  — in cfRNA, restricting to one probe-design version — accepting the loss of
  sample size as the price.

A port inherits this limitation and should say so. If a package's users are going
to need covariate adjustment, that is an argument for a different method, not for
a wrapper that pretends the test accepts one.

### 2.2 No repeated-measures handling

Multiple samples from the same subject are treated as independent, **and they are
not.** This bites the permutation test at its foundation: the null is
exchangeability of labels across samples, and samples from one subject are not
exchangeable with samples from different subjects. Correlated replicates counted
as independent observations inflate the effective sample size and make the null
too narrow, which is anti-conservative.

The source project measured the correlation and reported every statistic at two
units — all libraries, and one library per patient — treating the patient-level
unit as the one that decides feasibility. The measured within-patient versus
same-disease-different-patient correlation values are inline-computed in
`../reference/docs/section4_wade_setup.qmd`
and are **not resolvable from this repository** (the `r ...` expressions are
unevaluated and the cohort data are not present), so no number is quoted here.
The qualitative finding recorded in the narrative is that within-patient library
pairs correlate substantially more than same-disease different-patient pairs, and
that a library-level test therefore counts correlated replicates as independent.

Two notes for a port:

- The practical mitigation is to **select one sample per subject** before
  calling the test. That is a caller's responsibility and belongs in the
  documentation, not in the method.
- The validation simulations of [`rationale.md`](rationale.md) §9 use fully
  independent samples. They therefore establish calibration under an assumption
  the source application violates, and must not be cited as evidence that
  p-values are calibrated on multi-sample-per-subject data.

### 2.3 Its own internal normalization

The reference `wade()` applies a (count, normalizer) TPM-like scaling internally,
with a seeded continuity jitter, rather than consuming an already-normalized
matrix ([`algorithm.md`](algorithm.md) §8). Two distinct problems, which the
source separates carefully:

**It is a second normalization decision living inside a test.** The source
project had settled on a different house normalizer after a 16-candidate
evaluation, and every other analysis used it; `wade()` alone normalizes its own
way. Measured on the source cohort, the two correct sequencing-depth dependence
comparably — the residual Spearman correlation of per-library mean abundance
against on-target yield was reported as materially better than uncorrected for
both, and similar between them (the specific values are inline-computed in
`../reference/docs/section4_wade_setup.qmd`
and unevaluated there; the prior analysis in
`../reference/docs/WADE_REPO_SCOPE.md`
records $\rho = 0.41$ for WADE's internal TPM against $\rho = 0.36$ for the
house normalizer and $\rho = 0.79$ uncorrected, on the all-panels cohort). So on
that cohort this is a **redundancy rather than a defect** — but it is a
redundancy whose behaviour on any other dataset is unmeasured.

**A method that normalizes for you cannot be handed an already-normalized
matrix**, which is the common case for anyone with an existing pipeline. That is
an interface problem rather than a statistical one, and the prior analysis
recommends the entry point take a matrix already on a comparable scale with the
normalization shipped as an optional layer. The decision is
[`design-decisions.md`](design-decisions.md)'s.

What a user must know either way: **the jitter is not optional in effect.** If a
caller supplies an already-normalized matrix and skips the jitter, tied values in
zero-heavy data are no longer broken, and the tail statistic — which reads as few
as one order statistic — is the part of the method most exposed to ties. A port
that makes normalization optional must decide, and document, what happens to the
jitter.

### 2.4 Directionality: the method and the scores are one-sided upward

Two separate one-sidednesses compound, and it is worth separating them because
they have different remedies.

**The p-values are one-sided upward.** Every permutation p-value is
$\Pr(\text{null} \ge \text{observed})$ ([`algorithm.md`](algorithm.md) §4.2). A
gene *depleted* in cases gets a negative statistic and a p-value near 1 — not a
small p-value with a negative effect size. Confirmed in the sandbox this session:
a gene constructed to be strongly down in cases returned `p.diff` = 1.000 and
`p.tail` = 1.000 while its mirror-image up-in-case gene returned $3.1\times
10^{-4}$ and $8\times10^{-6}$. A user scanning for significant genes will simply
not see loss-of-expression signal. `w1`, being unsigned, is large for both
(measured: 461,308 for the down gene, essentially $|$`diff.mean`$|$) — which is
why `w1` is the only column in which a depleted gene is visible as "different",
and it carries no p-value.

**The rank scores also reward up-in-case genes**, independently of the above:
they multiply by $F(\texttt{cond1.mean})$ and $1 - F(\texttt{cond0.mean})$, so a
gene lost in cases cannot score highly. Genes lost in cases are visible in
`diff.mean` as a negative number but are not what the ranking surfaces.

If down-in-case genes matter for a use case, the workable approach is to run the
test a second time with the labels swapped, and treat it as a second family of
tests for multiple-testing purposes. That doubles the testing burden, which is
itself a reason the choice was made — but a port should state the limitation
rather than let a user infer symmetry that is not there.

### 2.5 `diff.mean` is a grid quadrature, not the mean difference, when groups are unbalanced

Established and measured this session; developed in full in
[`algorithm.md`](algorithm.md) §2.1.1. Summarised here because it is a limit on
*interpretation*.

Every prose source in this repository states
$\texttt{diff.mean} = \mu_{\text{case}} - \mu_{\text{ctrl}}$ without
qualification, including the `wade.R` header and an inline code comment. The
identity is **exact only when $n_1 = n_0$.** Otherwise the larger group is read
on a grid coarser than its own sample size, its extreme order statistics are
over-weighted (each carrying $1/m$ instead of $1/n$), and for right-skewed data
the grid mean exceeds the sample mean.

Measured at the source cohort's geometry (77 cases, 18 controls, $m = 18$) on
synthetic lognormal data with no group signal: `diff.mean` exceeded
$\mu_{\text{case}} - \mu_{\text{ctrl}}$ in **400 of 400 replicates**, by a mean
of $+1.42$ units. The gap shrinks monotonically with balance ($+1.38$ at 18
controls, $+0.52$ at 33, $+0.22$ at 50, exactly $0$ at 77) and reverses sign when
the control group is the larger one or the data are left-skewed.

**This does not invalidate the p-values.** The permutation null uses the same
estimator on the same grid and inherits the same bias; measured on 200 null genes
at 77-vs-18, the null distribution of `diff.mean` centred at $+252$ rather than
zero, leaving a standardized residual of $+0.016$ null SDs and a realised type-I
error of exactly 0.0500. What it affects is the **effect size**: read
`diff.mean` as "signed quantile area on the $m$-node grid" — which is what it is
— and not as a drop-in estimate of the mean difference when groups are
unbalanced. The same caveat applies to `w1` as an estimate of $W_1$, and to
`fc`, `cond1.mean`, `cond0.mean` and `tot.mean`, all of which are grid means.

The correct response in a port is to reproduce the estimator and fix the
*documentation*, not the arithmetic. Substituting $\bar{x}_1 - \bar{x}_0$ would
change the statistic and the null together.

---

## 3. The quantile grid is a sample-size constraint, not a tunable parameter

The grid has $m = \min(n_0, n_1)$ points and the tail window is
$k = \max(1, \lceil q_{\text{tail}} m \rceil)$ of them
([`algorithm.md`](algorithm.md) §2.3). The subset axis therefore needs enough
order statistics to exist. Reproduced from
`../reference/docs/supplement_wade.qmd`
§S7.5, with the $k$ values re-derived in the sandbox:

| smaller group | grid points | tail window | subset axis |
|---:|---:|---:|:---|
| 5 | 5 | 1 | a single order statistic — not a subset detector |
| 10 | 10 | 1 | still one point |
| 20 | 20 | 2 | minimum for a tail *mean* |
| 34 | 34 | 4 | usable |

**The critical sentence, and the one to carry into a port's documentation: a
one-point "tail mean" is an order statistic wearing the name of an average.** It
is the group maximum minus the other group's maximum. It has no averaging, no
variance reduction, and no robustness to a single aberrant sample — and it is
reported in a column called `tail.mean`, which invites exactly the wrong reading.

Three consequences:

1. **$q_{\text{tail}}$ is not a knob that buys resolution.** Because of the
   $\max(1, \cdot)$ floor, lowering it cannot produce a window smaller than one
   point, and raising it broadens the window toward the bulk — which defeats the
   purpose. At $m \le 10$, $q_{\text{tail}} = 0.10$ and
   $q_{\text{tail}} = 0.01$ give the identical statistic.
2. **The method computes it anyway.** There is no error, no warning, and no `NA`
   at $m = 1$. The statistic is always defined, which means the *caller* is
   responsible for knowing whether to believe it. The source project's response
   was to report the window size per contrast and carry a flag for whether the
   subset axis is usable — falling back to bulk-axis-only nomination below a
   smaller-group size of 20. Those thresholds
   ([`algorithm.md`](algorithm.md) §9.1) are cohort decisions, but their
   *rationale* is arithmetic on WADE's own parameters and generalizes: 10 gives a
   1-point window and is the floor for running at all; 20 gives 2 points and is
   the threshold for believing the subset axis; 30 gives 3.
3. **Any port exposing $q_{\text{tail}}$ should tell the caller how many order
   statistics that buys.** This is the cheapest possible guard-rail: $m$ and $k$
   are known from $n_0$, $n_1$ and $q_{\text{tail}}$ before any data are read.

A related trap at the very bottom of the range, confirmed against the reference:
at $m = 1$ the grid is the single point $p = 1$, so `diff.mean`, `tail.mean` and
`w1` all collapse to the difference of the two group maxima, and `tail.conc` is
identically 1. At $m = 2$ the grid is $\{1, 0\}$ and `diff.mean` is the
difference of the two mid-ranges. Neither case errors; both produce numbers that
mean something quite different from what their column names suggest.

At $m = 1$ the reference is additionally **wrong** on multi-gene input — it
returns one value that the driver recycles across every gene, silently and with
the correct output shape (verified in the sandbox this session; diagnosed in
[`r-implementation.md`](r-implementation.md) and
[`porting-hazards.md`](porting-hazards.md)). Since a one-sample group has no
quantile function worth comparing, refusing $m = 1$ is the better behaviour for a
port, and refusing loudly is better than the reference's silence.

Descriptive use of a small-$m$ contrast is legitimate — the effect sizes are real
measurements of the samples in hand. Reporting it as *subset detection* is the
analysis overstating its own resolution.

---

## 4. The combinatorial floor

**This is the most consequential correction in this repository, and the source
documents' gloss on it understates the constraint.**

### 4.1 The formula, which is exact

If $k^{*}$ samples carry a signal, the probability that a random relabelling
assigns **all** of them to the case group is

$$\boxed{\;P_{\text{floor}}(k^{*}) \;=\; \frac{\dbinom{n_1}{k^{*}}}{\dbinom{n_1+n_0}{k^{*}}}\;}$$

No label-permutation test can return a p-value below this, for any statistic, at
any effect size, with any number of permutations. The reasoning is immediate: in
a fraction $P_{\text{floor}}$ of all relabellings the null group assignment
recovers the true signal-carrying set, so the null statistic is at least as
extreme as the observed one in at least that fraction of draws. A better tail fit
does not help — the floor is not a resolution limit, it is a property of the
null's support. It is a **property of the design**, computable before any data
exist.

The floor also has a favourable reading, which is the mechanism of
[`rationale.md`](rationale.md) §7: at $k^{*} = 1$ it equals $n_1/N$, which is why
a single-outlier gene cannot post a small p-value.

### 4.2 What the source documents claim, and why it is not tight

`../reference/docs/supplement_wade.qmd`
line 161 states — as do
`../reference/docs/notebook_sections/v8_S7_wade_method_supplement.md`
line 178 and the v7 narrative and figure caption
(`../reference/docs/notebook_sections/v7_section9_wade_discovery.md`
lines 592 and 640):

> Below roughly 5% of cases, a shuffled null reproduces the signal and **no
> permutation test can separate it**, whatever the effect size.

**The formula is exact; the "roughly 5%" gloss is far from tight, and it errs in
the dangerous direction.** Evaluating $P_{\text{floor}}$ for the v7 simulation
design ($n_1 = 77$ cases, $n_0 = 18$ controls, on that script's own sweep grid) —
computed independently twice this session:

| fraction of cases | $k^{*}$ | floor | clears $\alpha = 0.05$? | clears BH threshold 0.0118? |
|---:|---:|---:|:---:|:---:|
| 3% | 2 | 0.6553 | no | no |
| 5% | 4 | **0.4251** | no | no |
| 8% | 6 | 0.2728 | no | no |
| 12% | 9 | 0.1373 | no | no |
| 20% | 15 | 0.03196 | yes | no |
| 35% | 27 | 0.001146 | yes | yes |
| 50% | 38 | 2.792e-05 | yes | yes |

At 5% of cases the floor is **0.43** — a shuffled null reproduces the signal 43%
of the time. The claim is not false (5% is indeed unsupportable) but it is
nowhere near the boundary: against a bare $\alpha = 0.05$ the unsupportable
region extends to roughly **18% of cases**, and further against a
multiple-testing threshold.

This matters because of how a reader uses the number. "Below roughly 5%" invites
the inference that 10% is fine. At 10% of cases in this design ($k^{*} = 8$) the
floor is **0.173** — not merely insufficient for FDR control, but not
significant at any conventional threshold. **A reader who trusted the "~5%"
figure would design an underpowered study and read null results as biology.**

The final column is what actually governed the simulation. With 80 true positives
among 680 genes, BH at $q = 0.10$ can only declare p-values at or below
$0.10 \times 80/680 = 0.0118$, and the floor crosses that between the 20% and 35%
rows.

### 4.3 The independently re-run simulation is consistent with the floor, not in conflict

[`../reference/R/validation_sims_v7.R`](../reference/R/validation_sims_v7.R)
measured power **0 at every swept fraction through 20%, and 1 from 35% on**
([`rationale.md`](rationale.md) §9.3). Compare that against the last column of
the table above: the transition sits exactly where the floor crosses the BH
threshold. The power curve and the floor arithmetic agree; they are two views of
the same constraint.

This is worth stating explicitly because the script's own header frames the 35%
figure as a *discrepancy* with the notebook's ~5%, and it is easy to read that as
two measurements disagreeing. They are not. The notebook's ~5% is a loose gloss
on the floor formula; the 35% is where power crosses 0.5 under one specific
multiple-testing configuration (80 true positives among 680 genes at $q = 0.10$).
Both are downstream of the same exact formula, and neither is the general answer:
the 18% figure above is where the floor clears a bare $\alpha = 0.05$, and it is
different again. **Three different numbers, one formula, three different
questions.** Quote the formula.

### 4.4 The floor is driven by imbalance, not by subset size alone

This is the part most likely to be mis-transferred, because the source project
reports numbers from one heavily imbalanced cohort. Holding the number of
signal-carriers fixed at $k^{*} = 4$ and cases at 77, varying only the control
count:

| controls | floor at $k^{*} = 4$ |
|---:|---:|
| 18 | 0.4251 |
| 33 | 0.2344 |
| 77 | 0.06006 |
| 200 | 0.005638 |

Same subset, same case count, and the floor moves by a factor of 75. And the
smallest subset fraction that clears a bare $\alpha = 0.05$:

| design | smallest $k^{*}$ clearing 0.05 | as a fraction of cases | floor there |
|---|---:|---:|---:|
| 77 cases / 18 controls | 14 | **18.2%** | 0.0411 |
| 77 cases / 33 controls | 9 | **11.7%** | 0.0347 |
| 50 / 50 | 5 | **10.0%** | 0.0281 |
| 30 / 30 | 5 | **16.7%** | 0.0261 |
| 20 / 60 | 3 | **15.0%** | 0.0139 |

Note that the balanced 50/50 design (100 samples) resolves a smaller *fraction*
of cases than the imbalanced 77/18 design (95 samples) despite having fewer
cases — and that 20 cases against 60 controls resolves 15% while 30 against 30
resolves only 16.7%. **Adding controls buys resolution; the ratio matters more
than the totals.**

### 4.5 The mechanism, in one paragraph

With 77 cases among 95 samples, **81% of all samples are cases**. A random
shuffle assigns each sample to the case group with that probability, so putting
all $k^{*}$ signal-carriers in the case group is not a rare event: at $k^{*} = 2$
it is $(77/95)(76/94) = 0.655$. The permutation null is not being asked to
produce an unlikely configuration — it is being asked to produce the *typical*
one. When one group dominates the sample, label shuffling barely changes the
group composition, and a test built on label shuffling has correspondingly little
to work with.

### 4.6 The practical upshot

**A user must evaluate the formula for their own design.** No number from the
source cohort transfers. The computation is one line and needs no data — only
$n_1$, $n_0$, and the smallest subset size worth detecting. A port should
consider exposing it as a function and inviting its use before a study is run,
because it answers the only question that matters at the design stage: *given
these group sizes, is the subset I care about detectable at all?*

The honest framing: this floor, not the FDR threshold and not the effect size, is
the binding limit on subset discovery in small or imbalanced designs.

---

## 5. A p-value at the extrapolation floor is not a small number

The clearest worked example in the source material of how to misread WADE's
output. It is also an argument **for** the floor doing its job, not against it.

### 5.1 The case

Across all contrast runs in the source project (both cohorts), **exactly one gene
reached FDR < 0.10 on either axis**: a gene on the subset axis of the
pancreatic-precursor contrast. It should not be read as a discovery, for three
reasons that compound. From
`../reference/docs/supplement_wade.qmd`
§S7.4:

1. **Its p-value sat exactly at the GPD extrapolation floor**
   $1/(B \cdot n_{\text{tail}})$ — not at a measured exceedance rate. The whole
   result was produced by the tail fit. The counterfactual the source computes:
   from the *empirical* floor $1/(B+1)$ alone, BH across that contrast's gene
   count gives an adjusted p-value nowhere near significance.
2. **Its contrast's tail window was a single order statistic** — the smaller
   group was small enough that $k = 1$, so the "subset axis" was a difference of
   maxima (§3). The source project's own screen had already marked that contrast
   `descriptive` for this reason.
3. **The signal was carried by fewer patients than the combinatorial floor
   permits** (§4).

**A note on what is and is not resolvable here.** The supplement quotes the gene
name, the p-value, the adjusted p-value, the gene count, the smaller-group size,
and the patient count as inline `r ...` expressions computed from the live cohort
in
`../reference/docs/section4_wade_setup.qmd`
(lines 293–337). **Those expressions are unevaluated in this repository and the
cohort data are not present, so none of those specific values can be quoted.**
What is verifiable here is the structure of the argument and the constants: at
the source project's production setting of $B = 2000$ with
$n_{\text{tail}} = 250$, the extrapolation floor is $2.0\times10^{-6}$ and the
empirical floor $1/(B+1)$ is $5.00\times10^{-4}$ — a factor of 250 apart, which
is the entire distance between "not significant" and "the one significant
result". Those two numbers are computed from the defaults in
[`../reference/R/wade.R`](../reference/R/wade.R) and were confirmed in the
sandbox. The identification of *which* gene, in which contrast, at what adjusted
p-value, requires the cohort.

### 5.2 The reading rule

> Read any p-value sitting **at** the extrapolation floor as "beyond this
> cohort's resolution", not as a small number.

A floored p-value is a **censored value reported at its detection limit**. It
carries the information "smaller than the permutation sample can resolve" and no
more. Feeding it to BH treats it as a measured probability, and BH will duly
return an adjusted value that looks like significance — which is exactly what
happened.

Diagnostically this is easy to check and a port should make it easy: any p-value
below $1/(B+1)$ is necessarily a fitted value, and any p-value equal to
$1/(B \cdot n_{\text{tail}})$ is at the floor. Both conditions are computable
from $B$ and $n_{\text{tail}}$ without inspecting anything else. A port would do
its users a service by flagging floored p-values in its output rather than
leaving them to be recognised — a boolean column costs nothing and prevents this
exact misreading.

### 5.3 This is an argument for the floor, not against it

The tail refinement did what it was designed to do: extrapolate where the
empirical resolution ran out, and floor the answer honestly rather than returning
machine epsilon ([`rationale.md`](rationale.md) §6.2). Without the floor the same
gene would have received something like $10^{-15}$, would have survived BH by a
much wider margin, and would have been *harder* to diagnose as an artefact —
because nothing about the number would have signalled that it was extrapolated.
The floor is what made the artefact visible.

**The refinement extends resolution; it does not manufacture power.** The three
compounding problems above are all power and design problems, and no p-value
transformation addresses any of them.

---

## 6. `tail.conc` is numerically unsound as written

A live defect with a measured magnitude, not a hypothetical. The statistic's
definition and the exact mechanism are in [`algorithm.md`](algorithm.md) §2.5;
the port's obligation is in
[`porting-hazards.md`](porting-hazards.md). What belongs here is why it is a
failure mode and why the obvious fix is wrong.

### 6.1 The defect

`tail.conc` is a share: signed tail area over total signed area,
$\sum_{\mathcal{T}} D \big/ \sum D$. Since $\sum_i D[i] = m \cdot
\texttt{diff.mean}$, the denominator vanishes whenever the lower quantiles'
differences cancel the upper ones — which happens at `diff.mean` values that are
small but entirely ordinary, nowhere near machine precision. The ratio then
diverges.

`wade()` does guard it, returning `NA` when
$|\texttt{diff.mean} \cdot m| < 10^{-8}$ — about
$|\texttt{diff.mean}| < 4.5\times10^{-10}$ at $m = 22$. That is roughly **nine
orders of magnitude tighter than the cases that actually occur.**

Measured on the source cohort's primary contrast: **120 of 2,219 genes (5.4%)
return $|\texttt{tail.conc}| > 2$, the largest 149.** A "share of the signed
area" of 149 is not a share of anything. Reproduced in the sandbox this session
on a constructed gene: bisecting to the pole gives $\pm\infty$ at the root, and a
perturbation to $\texttt{diff.mean} = 5\times10^{-5}$ returns
$\texttt{tail.conc} = 1.62\times10^{5}$ — with the reference guard not firing.

The source project checked and found that no *nominated* gene in any of its runs
was affected — but records that this is "luck, not construction". Any threshold
on `tail.conc` sits directly on top of the defect, and the volcano plot's
subset-versus-bulk colour split was one such threshold.

### 6.2 Why a clamp to $[0, 1]$ is the wrong fix

**Values slightly above 1 are legitimate.** When the lower quantiles' differences
are negative they cancel part of the tail's contribution, the denominator shrinks
below the numerator, and the tail genuinely carries more than the *net* total.
Constructed in the sandbox: a gene with cases below controls across the bulk and
above them in the top two nodes gives $\sum_{\mathcal{T}} D = 674$,
$\sum D = 512$, and $\texttt{tail.conc} = 1.316$ — a correct description of that
gene. Clamping would silently convert a meaningful 1.32 into a fictitious 1.00
and destroy the distinction between "tail carries everything and the bulk
partially opposes it" and "tail carries exactly everything".

The failure is not that the value exceeds 1. It is that the value is a ratio
whose denominator can approach zero, so its magnitude becomes uninformative long
before it becomes infinite. The distinction between a legitimate 1.3 and a
meaningless 149 is not the numerator — it is whether the denominator is large
enough for the ratio to mean anything.

The source's recommended direction is therefore to make the guard a statement
about the denominator's adequacy rather than about the ratio's range: report
`tail.conc` only when $\big|\sum D\big|$ is large enough relative to
$\sum |D|$ for the ratio to be meaningful, and return `NA` otherwise. The
specific criterion is a port decision and belongs in
[`porting-hazards.md`](porting-hazards.md) and
[`design-decisions.md`](design-decisions.md).

### 6.3 Why the source did not fix it in place

Recorded because the reasoning is legitimate and a port should understand it
rather than read the unfixed guard as carelessness. The source project guarded at
the **display layer** — a helper with a ceiling that returns `NA` above it —
rather than changing `wade()`, because editing the statistic would have
invalidated every cached result for a column that is descriptive and gates
nothing. That is a defensible choice for a running analysis with a cache
keyed on the statistic, and an indefensible one for a new package. **A port
inherits no cache and no such constraint, and should fix it at the source.**

---

## 7. Checklist: when not to use WADE

Drawn from the above. Each item is a "look elsewhere" signal rather than a
threshold to tune.

- **You need covariate or batch adjustment.** The test does not accept a design
  matrix, and a confound can be detected but not removed (§2.1). Screen and
  refuse, or use a model that adjusts.
- **Your samples are not exchangeable** — repeated measures, family structure,
  paired designs — and you cannot reduce to one sample per subject (§2.2).
- **Your smaller group is below about 20** and the subset axis is what you came
  for (§3). Below 10 the tail window is a single order statistic; the numbers
  will still compute.
- **The subset fraction you care about is below your design's combinatorial
  floor** (§4). Evaluate $\binom{n_1}{k^{*}}/\binom{n_1+n_0}{k^{*}}$ first; if it
  exceeds your alpha, no amount of data quality or permutation count will help.
- **You are looking for genes lost in cases.** The test and the scores are
  one-sided upward (§2.4). Swap the labels and pay the multiple-testing cost, or
  use a two-sided method.
- **You need a calibrated effect size on unbalanced groups.** `diff.mean` is a
  grid quadrature, biased toward the larger group's skew (§2.5). The p-values are
  unaffected; the effect size is not the mean difference.
- **You want to threshold on `tail.conc`** without first fixing the guard (§6).

And the converse, so the document is not only negative: WADE is the right tool
when you have two reasonably sized, reasonably balanced, exchangeable groups; you
suspect a signal present in a minority of one of them; you cannot specify the
number of subgroups or the mixing fraction in advance; and the combinatorial
floor for the subset size you care about clears your significance threshold.
Within that envelope it is calibrated at nominal on both axes and it separates
minority-subset signal from whole-group shifts
([`rationale.md`](rationale.md) §9).
