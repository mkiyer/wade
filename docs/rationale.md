# WADE — statistical rationale

This document carries the reasoning that must survive the port. The formulas are
in [`algorithm.md`](algorithm.md); what the method cannot do is in
[`limits.md`](limits.md). Here: why this statistic exists, why each design choice
was made, and which of those choices are load-bearing rather than incidental.

A port that reproduces the arithmetic but discards the reasoning will be
maintained into incorrectness — the shape statistics will be read as inference,
the p-value floor will be "improved", the rank scores will be printed next to
adjusted p-values without comment. Each of those is a specific, foreseeable
error, and each has a paragraph below.

Sources are cited by relative path. Numbers are attributed at the point of use:
measurements from the reference sandbox are labelled as such, numbers from the
source cohort are labelled as cohort-specific, and design assertions are
labelled as assertions.

---

## 1. The hypothesis a location-shift test cannot state

Standard differential-expression tests — the two-sample $t$-test, Wilcoxon
rank-sum, and the negative-binomial GLM tests of DESeq2 and edgeR — compare
group means or mean ranks. Each is built on a location-shift alternative: the
two groups are drawn from distributions that differ by a shift in centre (or, in
the GLM case, in a fitted mean parameter with a specified variance
relationship).

That alternative is wrong for the question WADE was built to answer, and wrong
in a way that no increase in sample size repairs. A cancer case group is
**heterogeneous by construction**. If a subtype marker is elevated in 15 of 77
cases and silent in the other 62, then:

- its mean difference is diluted by roughly the reciprocal of the affected
  fraction, so it is modest;
- its rank-sum is close to null, because 62 of 77 cases are interchangeable with
  controls;
- and a per-gene dispersion estimate, if the model fits one, absorbs the
  bimodality as extra variance and *widens the interval*, which pushes the
  p-value further from significance.

The example is the source project's own framing:
[`../reference/docs/supplement_wade.qmd`](../reference/docs/supplement_wade.qmd)
puts it as "a gene elevated in 15 of 60 cases and silent in the other 45", and
the v7 narrative uses 15 of 77
([`../reference/docs/notebook_sections/v7_section9_wade_discovery.md`](../reference/docs/notebook_sections/v7_section9_wade_discovery.md)
line 37). The two differ in denominator because they were written against
different cohort versions; the argument is the same and does not depend on
either number.

The point is stronger than low power, and the distinction matters because it
determines what the fix has to be. A test that is *underpowered* against a
signal returns a large p-value and would return a small one given more samples.
A test that is testing a **different hypothesis** does not converge on the right
answer as $n$ grows: a heterogeneous alternative in which the case distribution
gains an upper mode is not a location shift, and a location-shift test's null is
not the null one wants rejected. As the supplement puts it, the test "is not
merely underpowered against that signal — it is testing a different hypothesis."

WADE's response is to compare the two groups' whole **distributions** and then
read specifically the part of that comparison where a rare high-expressing
subset would appear. That is the entire design in one sentence, and everything
below is a consequence of it.

**One caveat on the framing, so the argument is not overclaimed.** Location-shift
tests are not the only alternative; the outlier-detection literature of §2
exists precisely because this problem was recognised, and a mixture model or a
two-part test could also target a heterogeneous alternative. WADE's claim is not
that it is the only method that can see subset signal — it is that a quantile-area
comparison sees it without requiring the number of subgroups, the mixing
proportion, or a parametric form for the elevated component to be specified in
advance.

---

## 2. The two literatures WADE joins

The source documents place WADE at the intersection of two method families, and
state the division of labour explicitly: the quantile/Wasserstein
differential-distribution test supplies the **mechanism**, and cancer-outlier
profile analysis supplies the **goal**. That framing appears verbatim in the
`wade.R` header
([`../reference/R/wade.R`](../reference/R/wade.R) lines 21–24), in
[`../reference/docs/supplement_wade.qmd`](../reference/docs/supplement_wade.qmd)
§S7.2, and in the v7 narrative.

**Differential-distribution testing** — cited by the sources as scDD and waddR —
contributes the idea that two groups should be compared as distributions rather
than as summary statistics, and that a distance between empirical distributions
(in waddR's case an explicitly Wasserstein one) is a usable test statistic.
WADE takes from this family its comparison object: the pair of empirical
quantile functions, and the integrated difference between them. What WADE does
*not* take is the decomposition of that distance into interpretable components,
or any of these methods' distributional machinery.

**Cancer-outlier profile analysis** — cited as COPA, OS (outlier sum), ORT
(outlier robust $t$-statistic), MOST, and LSOSS — contributes the goal.
This family was developed for exactly the regime of §1: find genes
over-expressed in a *subset* of tumours, where the affected fraction is unknown
and small. Its characteristic move is to score a gene by the behaviour of its
most extreme case samples rather than by a group average. WADE takes this
objective and this emphasis on the upper end of the case distribution.

WADE's contribution is the combination: it reads the *shape* of a
quantile-function difference in a specified upper window, which gives a
subset-detection statistic that is a functional of the same quantile comparison
that produces the bulk effect size. Both axes come from one object, which is why
they can be read together (§4).

**A note on citation depth, deliberately observed.** The source documents cite
these eight methods by name and by the one contribution each family makes; they
do not compare WADE's operating characteristics against any of them, and this
repository contains no such comparison. Neither does this document. There is no
bibliography anywhere in the reference material — the citations are bare names
in prose, and the only one carrying a year is Knijnenburg et al. 2009 for the
tail refinement (§6). Anyone writing a package README, a paper, or a
benchmarking section will need to resolve these to full references and, if a
comparative claim is wanted, generate the comparison. **No benchmark of WADE
against COPA, OS, ORT, MOST, LSOSS, scDD or waddR exists in this repository**,
and the absence should be stated rather than papered over.

---

## 3. Why `diff.mean` alone is insufficient

`diff.mean` is the signed area between the quantile functions. It is a
first-moment quantity, and a first-moment quantity cannot distinguish *how* a
difference is distributed across the case group. Two genes with the same
`diff.mean` may have:

- a modest elevation present in every case, or
- a large elevation present in a small minority of cases.

These are different biological claims — a pathway shifted across a disease, or a
marker of a subtype — and they are the two hypotheses the analysis most needs to
tell apart. `diff.mean` collapses them to the same number. This is the sentence
the sources repeat in every version, and it is the entire motivation for the
shape statistics.

The worked fixture in [`algorithm.md`](algorithm.md) §2.7 makes the collapse
concrete and is worth restating because it is the cheapest illustration of the
method's premise. Two genes, five cases against four controls: one with a single
case at 60 against a baseline of 10–16, the other shifted as a whole group to
20–28. Their bulk statistics are nearly identical — `diff.mean` of 11.5 against
11.0, a 4.5% difference. Their subset statistics differ by a factor of 3.7 —
`tail.mean` of 44 against 12. A bulk-only readout would rank these two genes
side by side; the subset axis separates them.

`tail.conc` then answers the follow-up question — *what share of this gene's
total signed difference is carried by its top quantiles* — which is what
distinguishes a gene that is elevated in a subset **and nothing else** from one
that has both a broad shift and a tail. In the fixture, 0.96 against 0.27.

This is also why `w1` (the 1-Wasserstein distance) is reported but not tested.
$W_1$ is an unsigned magnitude: it tells you the two distributions differ and by
how much, but it discards direction and gives no information about *where* in the
distribution the difference sits. As a description of total distributional
distance it is useful; as a subset detector it would have exactly the deficiency
`diff.mean` has, plus the loss of sign.

---

## 4. Reading the two axes together

The two axes are computed from the same $D$ profile and are reported side by
side, uncombined: neither gates the other and there is no composite statistic.
That is deliberate — combining them would reintroduce the collapse of §3 in a
new form.

The intended way to read them is the per-gene diagnostic
([`algorithm.md`](algorithm.md) §7): the cumulative signed area, accumulated in
ascending quantile order, whose endpoint is exactly `diff.mean`. Its three
readings, as the source project's figure module records them
([`../reference/R/downstream/README.md`](../reference/R/downstream/README.md)):

- a broad shift **rises steadily** across all quantiles;
- a rare high-expressing subset **stays flat then climbs** inside the tail
  window;
- a single outlier **stays flat then spikes** at the final node.

The identity that the curve's endpoint equals `diff.mean` is what makes the plot
honest: the bulk effect size is not a separate number annotated onto the figure,
it is the figure's endpoint. Any implementation that cannot reproduce this panel
is missing the per-gene detail output, and the identity is the cheapest assertion
available for checking a port ([`algorithm.md`](algorithm.md) §11).

The second and third readings above are the important pair, and they are the
subject of §7. **They are not distinguishable from the shape statistics.**

A related view from the same source: in the volcano plot of bulk against subset
effect, every gene sits above the $y = x$ diagonal, because the tail is by
construction the high end of the distribution. The diagonal is therefore drawn
rather than fitted, and the **distance above it** is the subset signal. This is
a useful sanity check on any port's output — points below the diagonal in the
upper-right quadrant would indicate an indexing error in the tail window.

---

## 5. Why inference is by permutation, and why the noise draw is held out

### 5.1 Permutation

There is no tractable null distribution for a quantile-area statistic on
zero-inflated capture data. Each obstacle is separately sufficient:

- The statistic is a **weighted sum of order statistics** of two samples, whose
  joint null distribution depends on the unknown underlying distribution.
- The data are **zero-inflated and heavily right-skewed**, so asymptotic
  normality arguments are unreliable at the relevant sample sizes, and the
  variance is not a known function of the mean in any form the statistic could
  exploit.
- The tail statistic depends on **only $k$ of $m$ grid nodes** — as few as one
  or two in the source project's contrasts — so its null is that of an extreme
  order statistic, where asymptotic approximations are worst.

Label permutation sidesteps all three. Under the null of exchangeability, any
assignment of labels is equally likely, so recomputing the statistic on shuffled
labels samples its null distribution directly, with no distributional assumption
and no variance model. The cost is resolution — the empirical p-value has a floor
at $1/(B+1)$, which §6 addresses — and the assumption of exchangeability itself,
which is the hinge on which the repeated-measures limitation turns
([`limits.md`](limits.md) §2.2).

One structural property is worth stating because it is easy to break in a
reimplementation: **the same shuffle is applied to all genes within a
permutation**. This preserves the gene-gene correlation structure in the null,
which is what makes the BH step defensible on correlated genes. Drawing an
independent shuffle per gene would produce a null with the correlation destroyed
and would be anti-conservative across correlated gene sets.

### 5.2 The noise draw is held out of the null

WADE applies a small uniform continuity jitter to break ties in sparse,
zero-heavy data (a gene with many zero counts has tied order statistics, and
tied quantiles make the statistic's null distribution lumpy and its
extreme-order-statistic behaviour degenerate). The jitter is drawn **once**, at
count precision, under a fixed seed; the labels are then permuted on that fixed
matrix.

Inference is therefore **conditional on the realised jitter** rather than
marginal over it. Two consequences, in order of importance:

1. **A given seed reproduces a given result exactly.** If the jitter were
   redrawn inside the permutation loop, the null would integrate over the noise
   and the same seed would still be reproducible, but each permutation would be
   comparing an observation to nulls computed on different matrices — mixing two
   sources of variation into one p-value.
2. **The comparison is apples-to-apples.** The observed statistic and every null
   statistic are computed on the *same* numbers, differing only in the label
   assignment, which is exactly the null hypothesis being tested. That is what a
   permutation test is supposed to isolate.

The honest description of what this buys is "inference conditional on the
jitter". The jitter is a nuisance perturbation introduced for numerical reasons,
and conditioning on it means the p-value does not account for the variability
that a different draw would introduce. With a jitter width of 0.01 count units,
that variability is small relative to count-level differences by construction —
but this is a **design assertion**, and this repository contains no measurement
of p-value stability across jitter seeds. A port that wants to claim seed
robustness should measure it; it would be a reasonable addition to a test suite.

---

## 6. The tail refinement, and what each of its three choices prevents

With $B$ permutations the smallest empirical p-value is $1/(B+1)$ — $5.0\times
10^{-4}$ at $B = 2000$. A gene whose observed statistic exceeds every permuted
value is known only to be "below that". Under multiple testing across thousands
of genes, that floor can be the difference between a result that survives BH and
one that cannot: BH needs p-values small relative to $\alpha \cdot (\text{rank}
/ G)$, and $1/(B+1)$ may be larger than that threshold for every gene, in which
case no gene can be declared **regardless of effect size**.

The refinement addresses the resolution limit, not the power limit: where
exceedances are too few to resolve, the upper tail of the permutation null is fit
parametrically (a generalized Pareto distribution, by method of moments, citing
Knijnenburg et al. 2009) and the p-value is read from the fitted survival
function. Extreme-value theory motivates the GPD as the limiting distribution of
threshold exceedances, which is the standard justification for this move.

Method of moments rather than maximum likelihood was chosen because it is
closed-form and dependency-free — a deliberate trade recorded in the source
([`../reference/R/wade.R`](../reference/R/wade.R) lines 140–141: "MLE-based GPD
is the production upgrade"). MoM estimators
for the GPD are less efficient than MLE and are not defined for
$\xi \ge 1/2$ (where the variance does not exist); the fallbacks of §6.3 catch
the degenerate cases in practice. An implementation upgrading to MLE should
expect small p-value changes and treat that as an intended difference, not a
parity failure.

Three choices inside the refinement each prevent a specific, identifiable
failure. They are not tuning parameters and should not be treated as such.

### 6.1 The $\xi \le 0$ exponential branch prevents machine-epsilon p-values

A generalized Pareto distribution with negative shape $\xi < 0$ has a **hard
upper bound** at $-\sigma/\xi$: it assigns probability exactly zero beyond that
point. When the moment fit returns a non-positive shape — a light or bounded
tail, which is common for a permutation null of a bounded statistic — an observed
statistic beyond the fitted bound would receive a p-value of zero, or whatever
machine epsilon the arithmetic produces.

WADE instead uses the $\xi \to 0$ **exponential limit**,
$\Pr(Y > y) = \exp(-y/\sigma)$, which is well-defined and strictly positive for
all $y \ge 0$. What this prevents is twofold: a p-value of zero, which is never a
defensible statement from a finite permutation sample; and — worse in practice —
a cluster of genes all collapsing to the same machine-epsilon value, which
**creates false ties among exactly the genes that matter most** and destroys
their relative ordering. When nomination is by rank (§8), mis-ordering the
strongest hits is the one failure the pipeline cannot absorb.

That the branch is well-chosen rather than arbitrary is checkable: on $B = 2000$
draws from a standard exponential null (true $\xi = 0$), the moment estimators
returned $\widehat{\xi} = -0.011$ and $\widehat{\sigma} = 1.019$ in the sandbox
this session — landing on the exponential branch with a scale near the truth.
A light-tailed null is the ordinary case, not the exception, so this branch
carries most of the traffic.

### 6.2 The floor at $1/(B \cdot n_{\text{tail}})$ prevents a fiction

The fitted survival function will happily return $10^{-15}$. That number is not
supported by the data: it is an extrapolation from $n_{\text{tail}} = 250$ tail
points out of $B = 2000$ draws, and the smallest tail probability such a sample
can defensibly resolve is on the order of $1/(B \cdot n_{\text{tail}})$ —
$2.0\times10^{-6}$ at the production settings.

WADE floors the returned p-value there. The floor is an **honesty constraint**:
it caps the claim at the resolution the permutation sample can support. Two
things follow, and both must travel with any port:

- **A port that "improves" the floor by returning smaller p-values is a
  regression**, not an enhancement. The source documents say so explicitly
  ([`../reference/docs/WADE_REPO_SCOPE.md`](../reference/docs/WADE_REPO_SCOPE.md),
  parity item 5).
- **A p-value sitting exactly at the floor means "beyond this cohort's
  resolution", not "very small".** It is a censored value reported at its
  detection limit. Treating it as a measured probability — in particular, feeding
  it to BH and reading the result as significance — is a specific, documented
  misreading, and the source project's own single FDR-significant result is a
  worked example of it. That example is in [`limits.md`](limits.md) §5, and it is
  the clearest thing in this repository about how to read WADE's p-values.

### 6.3 Refinement only where needed prevents replacing measurement with model

Refinement is applied only to genes with fewer than $n_{\text{exc,min}} = 10$
exceedances, and only when $B \ge 2 n_{\text{tail}}$. A gene with 10 or more
null draws at or above its observed value has a p-value that is *measured*, and
a measured p-value is never replaced by an extrapolated one.

This prevents the refinement's model error from contaminating the bulk of the
gene list — the region where the empirical estimate is perfectly adequate — and
it confines any GPD misfit to the genes where the alternative was an
uninformative $1/(B+1)$. It also keeps the two regimes distinguishable after the
fact: a p-value below $1/(B+1)$ is necessarily a fitted value, which is what
makes the diagnosis in [`limits.md`](limits.md) §5 possible at all.

The four internal fallbacks (fewer than 10 exceedances above threshold, an
observation not exceeding the threshold, a non-positive or non-finite variance,
a non-positive scale estimate) all return the empirical p-value, so a failed fit
degrades to the unrefined answer rather than to a wrong one.

---

## 7. What the permutation p-value is really protecting against

**This is the most important paragraph in this document.** If a port keeps one
piece of reasoning from it, this is the piece.

The shape statistics cannot refuse a single-outlier gene, and they were never
able to. Consider a gene whose elevated signal comes from exactly **one**
extreme case sample. Read its statistics:

- `tail.mean` is large — the tail window always contains the maximum
  ([`algorithm.md`](algorithm.md) §2.3), and one extreme sample is the maximum;
- `diff.mean` is inflated by that same sample;
- `tail.conc` sits near 1 — nearly all of the signed area is in the tail,
  because that is where the single sample's contribution lives;
- and the cumulative diagnostic curve is flat then rising at the end.

That is, on every shape statistic and on the diagnostic plot, **a one-sample
artefact is indistinguishable from a genuine subset marker**. Both look flat
then climbing. The shape statistics *describe* subset structure; description
cannot separate structure from noise, because a single outlier has the same
shape as a small subset.

The permutation p-value is what separates them, and the mechanism is simple
enough to state exactly: **no shuffled null can be beaten by one sample.** If a
gene's signal rests on one sample, then a random relabelling has probability
$n_1/N$ of putting that sample in the case group — for the source project's
primary contrast geometry (77 cases of 95 samples) that is 0.81. In 81% of
permutations the null statistic is as large as the observed one, so the empirical
p-value is near 0.81 and the gene is declined. A genuine $k$-sample subset, by
contrast, requires *all $k$* signal-carriers to land in the case group, whose
probability falls off combinatorially in $k$ — which is exactly the floor
formula of [`limits.md`](limits.md) §4, read in the favourable direction.

So the division of labour is:

> The shape statistics **describe** subset structure. The permutation p-value is
> what stops one outlier being called a discovery.

Three consequences a port must carry:

1. **The p-values must be reported even when they gate nothing.** In the source
   project no gene survives BH (§8), so the p-values did not select the
   nomination list — and they were still reported in every nomination table, for
   precisely this reason. A package that lets a user compute `tail.mean` and
   `tail.conc` without permutation p-values has handed them a tool that cannot
   tell a subtype marker from a pipetting error. If an implementation offers a
   `nperms = 0` mode (the reference does), its documentation must say what is
   lost.
2. **This is also the answer to "why not just threshold `tail.conc`".** A
   `tail.conc` cutoff is a shape filter, and shape filters admit single
   outliers by construction. It is a reasonable *descriptive* split and the
   source project used it as one in its volcano colouring; it is not a
   discovery criterion.
3. **The same argument caps what a small permutation count can do.** The
   protection works because one sample lands in the case group often enough to
   be seen in $B$ draws. It does, easily, at any usable $B$ — this protection is
   the cheapest thing the permutation test provides and the last thing it loses.

---

## 8. The rank scores are nomination, not inference

`score` and `tail.score` ([`algorithm.md`](algorithm.md) §6) are signed
rank-products built from empirical CDFs across genes: high in cases, low in
controls, and large on either the bulk axis (fold change, for `score`) or the
subset axis (`tail.mean`, for `tail.score`). Both lie in $[-1, 1]$.

They are **panel-selection heuristics**. The source describes them as "the lab's
panel-selection score, not an inferential quantity", and states the consequence
plainly: **a rank of 1 is not a claim of significance.** There is no null
distribution for these scores, no calibration, and no error control. They order
genes for follow-up; they do not test anything.

### 8.1 Why nomination fell back to rank, and what that does and does not mean

In the source project no gene survived BH on either axis, so nomination could
not be by adjusted p-value. The reason is a **power statement about group
sizes**, and it needs to be stated precisely because the alternative reading —
that the data contain no signal — is wrong and available.

The arithmetic: with roughly 2,200 genes tested and reference groups of 13–34
libraries, the permutation resolution available cannot produce p-values small
enough to survive BH across that many genes. Set against the resolution limits
of §6: the empirical floor is $1/(B+1) = 5.0\times10^{-4}$ at $B = 2000$, and
BH at $q = 0.10$ over 2,200 genes requires the smallest p-value to be at or
below $0.10/2200 \approx 4.5\times10^{-5}$ — an order of magnitude below the
empirical floor. **Every gene's empirical p-value is therefore inadmissible to
BH before the data are consulted.** Only a refined (extrapolated) p-value can
clear it, which is what makes the single FDR-significant result in the source
project an extrapolation artefact rather than a discovery
([`limits.md`](limits.md) §5).

Two conclusions, and the second is the one that gets dropped:

- **Absence of FDR-significant genes here is not evidence of no signal.** The
  source project's counter-evidence was an enrichment of nominal $p < 0.01$
  genes over the number expected by chance — a group-level signal statement that
  does not require any individual gene to survive correction.
- **The fallback to rank is a consequence of cohort size, not a property of the
  method.** A port's user with balanced groups of a few hundred samples per arm
  faces different arithmetic entirely and may well have genes surviving BH. The
  reasoning transfers; the conclusion does not. Anyone reproducing "nominate by
  rank" as a default because WADE did it in cfRNA would be importing a cohort's
  limitation as a method's convention.

### 8.2 Whether the scores belong in a general-purpose package is open

Recorded as an open question rather than resolved here. The argument for
including them is continuity: they are what the source project's nomination
actually used, and they are computed only from columns the test already
produced. The argument against is the one the source itself makes — "a package
that offers `score` next to `padj.diff` without comment invites a user to treat
a heuristic rank as inference"
([`../reference/docs/WADE_REPO_SCOPE.md`](../reference/docs/WADE_REPO_SCOPE.md)).
The prior analysis recommends moving them into a separate namespace documented
as nomination-not-inference.

There is a further design consideration a port should weigh, which follows from
the definition rather than from the source project's experience: because the
scores are built from empirical CDFs **across the genes supplied**, they are not
comparable between runs on different gene sets. The same gene, with identical
data, receives a different `score` depending on which other genes were in the
matrix. That is acceptable for selecting a panel from one fixed screen — the use
they were built for — and it is a trap for anyone comparing scores across
contrasts, cohorts, or filtering thresholds.

The decision belongs in [`design-decisions.md`](design-decisions.md); this
document's requirement is only that whatever is shipped carries the framing.

---

## 9. Evidence that the method behaves as claimed

Three simulations validate three distinct claims. They are in
[`../reference/R/validation_sims_v7.R`](../reference/R/validation_sims_v7.R) and
run standalone; the numbers below were measured in the reference R sandbox and
are recorded in
[`../reference/R/README.md`](../reference/R/README.md).

**Read the design before the numbers.** These are simulations on synthetic data
at one specific geometry: **77 cases against 18 controls**, giving an 18-point
quantile grid and a 2-point tail window, at 1,000 permutations. That geometry was
chosen to inherit the source cohort's own imbalance rather than a round number.
Every number below is conditional on it, and none is a property of the method at
other group sizes. A port's user with a different design should re-run these
simulations at their own geometry — that is what the script is for.

The source's own record notes that the control count is ambiguous in the original
notebook (18 versus 33) and that both readings were run: they agree on every
qualitative conclusion, so nothing here depends on resolving it.

### 9.1 Null calibration — the method does not manufacture significance

Poisson counts at a common rate, random per-gene normalizers, labels carrying no
signal by construction. Measured:

| axis | KS test vs Uniform(0,1) | type-I error at nominal 0.05 |
|---|---|---|
| `diff.mean` (bulk) | $p = 0.6554$ | 0.0512 |
| `tail.mean` (subset) | $p = 0.2327$ | 0.0488 |

Both axes are at nominal; the permutation p-values are not inflated. This is the
claim that WADE does not invent signal, and it is the check any port must
reproduce first.

Two caveats the source carries and this document keeps. Permutation p-values are
**discrete** — the unrefined ones are multiples of $1/(B+1)$ — so the
one-sample KS test warns about ties and its p-value is conservative rather than
exact. That warning is a property of any permutation test's calibration check,
not a defect, and a port's equivalent check will produce it too. The type-I error
rates, which make no continuity assumption, are reported alongside for that
reason and are the more direct evidence.

Independently of the simulation, the null calibration was checked once more in
the sandbox this session while investigating the quadrature bias of
[`algorithm.md`](algorithm.md) §2.1.1: on 200 null genes at the same 77-vs-18
geometry, the realised type-I error on the bulk axis was exactly 0.0500. The
bias in the effect-size estimator is common to observation and null and cancels
in the p-value.

### 9.2 Subset-vs-bulk discrimination — the tail axis does what it exists for

A lognormal null background of 600 genes, plus 25 planted whole-group location
shifts ("bulk") and 25 genes elevated in only 8% of cases ("subset"). Measured
medians, in the simulation's arbitrary abundance units:

| planted class | median `tail.mean` | median `diff.mean` | median per-gene `tail.mean`/`diff.mean` |
|---|---|---|---|
| subset (8% of cases) | 27,537 | 3,046 | **8.9** |
| bulk (whole-group shift) | 4,064 | 1,348 | **3.3** |
| null background | 535 | −46 | — |

**Read the third column carefully — it is not the quotient of the first two.**
The simulation computes it as the *median of the per-gene ratio*,
$\operatorname{median}_g(\texttt{tail.mean}_g / \texttt{diff.mean}_g)$
([`../reference/R/validation_sims_v7.R`](../reference/R/validation_sims_v7.R)
line 224), which is not the ratio of the medians: dividing the first two columns
gives 9.04 and 3.01 rather than 8.9 and 3.3. Both summaries are legitimate and
they carry the same conclusion, but a port checking its own output against these
numbers must compute the same one.
[`../reference/R/README.md`](../reference/R/README.md) quotes the two medians and
then the per-gene ratio in a single clause ("27,537 against a median `diff.mean`
of 3,046, a ratio of 8.9"), which reads as though 8.9 were the quotient; it is
not.

The discriminating quantity is the **ratio**, not either column alone. Subset
genes sit at 8.9 against bulk genes' 3.3: the tail statistic is elevated
relative to the bulk statistic roughly 2.7 times more in subset genes than in
whole-group shifts. That separation is the reason the tail statistic exists.
Note that bulk shifts also have a ratio above 1 — the tail is the high end of
the distribution, so `tail.mean` exceeds `diff.mean` for any up-in-case gene
(the $y = x$ diagonal of §4). The signal is the *excess* above that baseline.

### 9.3 Power against the combinatorial floor — the honest limit

Detection power (BH-adjusted tail $p < 0.10$) for 80 planted subset genes against
a 600-gene background, as the altered fraction of cases is swept, plotted
against the exact permutation floor. Power was **0 at every fraction through 20%
and 1 from 35% on**.

That step is not a simulation artefact and not a limit of the permutation count:
with 80 true positives among 680 genes, BH at $q = 0.10$ can only declare
p-values at or below $0.10 \times 80/680 = 0.0118$, and the combinatorial floor
crosses that threshold between the 20% and 35% sweep points. Raising the
permutation count cannot move it.

**The source documents' gloss on this is wrong in an important direction, and
the correction is developed in full in [`limits.md`](limits.md) §4.** The v7
figure caption and prose, and the S7.4 supplement text, put the collapse "below
roughly 5% of cases". The floor *formula* is exact; the "~5%" gloss understates
the constraint substantially — at 5% of cases the floor is 0.43, meaning a
shuffled null reproduces the signal 43% of the time. The independently re-run
simulation's power curve (0 through 20%, 1 from 35%) is **consistent with** the
floor arithmetic, not in conflict with it. Read the correction there before
quoting any figure for the resolution limit.

### 9.4 What these simulations do not establish

They are simulations on synthetic data. They say nothing about whether any
particular gene nomination is real, and they establish only that the statistic
behaves as designed at the design's group sizes. They also do not cover:

- **Any geometry other than 77-vs-18.** In particular they do not test the
  balanced case, where the estimator identity of
  [`algorithm.md`](algorithm.md) §2.1.1 becomes exact.
- **The tail.conc statistic.** No simulation exercises it, which is consistent
  with the defect in [`limits.md`](limits.md) §6 having been found by plotting
  real data rather than by testing.
- **Repeated measures.** Every simulated sample is independent, which is exactly
  the assumption the source cohort violates
  ([`limits.md`](limits.md) §2.2). The simulations therefore validate
  calibration under an assumption the application does not satisfy, and they
  should not be cited as evidence that the p-values are calibrated on
  multi-library-per-patient data.
- **Jitter-seed stability** (§5.2).

A port's test suite has an opportunity the source project did not take: to run
these at several geometries, including balanced ones, and to add the cases above.
