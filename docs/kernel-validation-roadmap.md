# WADE measurement kernels: deferred research and evidence

**2026-09-25 scope change:** the user has chosen a point-based 0.2.0 release.
The active scientific and release sequence is now [PLAN.md](../PLAN.md).
The dated kernel plan below is retained as a research record, including its
failed gates and previous release requirements. Statements below requiring
kernels for 0.2.0 or directing the next kernel experiment are superseded.
No kernel is selected, and further derivation is deferred until after release.

2026-09-23 — design reset following the user's review of the depth-stress study.
**This was the canonical kernel plan before the scope change above.** Earlier documents remain evidence
of experiments, including failures; their proposed next steps are superseded here.
This reset changes planning and terminology, not production code or study results.
The user accepted this sequence before requesting the workspace cleanup. No new
kernel or default is selected. New code belongs in [research/](../research/README.md);
completed reports and competing prototypes now live in the [development archive](../archive/development-20260923/README.md).

We moved too quickly from one diagnostic to the next without keeping a concise,
shared specification of the method. The reports also emphasized detection more
than estimation, and used prior names as if they were kernel families. That made
it difficult to distinguish a numerical implementation, a statistical candidate,
and a validated procedure. We need to correct all three problems.

**Last kernel gate (2026-09-24; deferred):** [alternative constructions](#alternative-constructions--2026-09-24)
now demonstrate that unchanged counts, exact individual modes, positive zero
uncertainty and conserved totals are compatible. Neither construction passes
the measurement-calibration gate. Kernel width and unseen-gene mass allocation
remain unresolved; cohort performance and production migration wait for this.
The [2026-09-25 release assessment](index.md) summarizes current software,
comparative sensitivity/specificity and the remaining release gates. Measurement
interval coverage and gene-level false discoveries are separate endpoints;
the former must not be presented as the latter or replace workflow validation.

## 1. The objective we retain

WADE should find global and subset-driven expression changes in large patient
cohorts, especially useful upward biomarkers against a large normal reserve.
It should represent finite-count uncertainty, including at zero, while preserving
precise, striking signals such as ETV1/ETV4/ERG and ERBB2/GRB7.

The intended construction remains:

1. Give each patient one abundance kernel on a common normalized expression axis.
2. Average the patient kernels within each group, with equal patient mass.
3. Calculate directional area between the group distributions; describe the
   magnitude and location of the enrichment without a user-selected percentile.

For patient CDFs K and group CDFs F, the upward statistic is

\[
F_g(a)=\frac{1}{n_g}\sum_{i\in g}K_i(a),\qquad
T_\uparrow=\int [F_{\rm control}(a)-F_{\rm case}(a)]_+\,da.
\]

The downward area reverses the groups; their sum is W1. The distribution of
patients remains nonparametric even when the measurement kernel uses a count
likelihood. Linear and log abundance serve different biological questions;
validate both before exposing them. The present depth-stress results are
**linear upward area only**, not a log-kernel comparison.

At the time of this plan, uncertainty was part of the intended 0.2.0 release;
that requirement is now deferred as stated above. Universal superiority,
small-cohort superiority over parametric methods, and fewer calls alone are not
success criteria. We need a demonstrable robustness or estimation benefit with
useful sensitivity and adequate false-discovery control. Optimization follows
that evidence. Sparse-assay certification and transcriptome-wide predictive
rarity remain later work, rather than claims borrowed from bulk validation.

## 2. What was actually compared

### Normalization was shared, including the full kernel

The executed [depth-stress runner](../archive/development-20260923/tools/kernel_depth_stress.py) fits one RLE
normalization per complete count matrix and passes the same factors to all five
arms. The downstream gene filter does not change the count-universe totals or
refit those factors. It compares normalized points, two Beta-kernel candidates,
and their own means as points, with the same labels and permutation banks.

Let C be a gene's raw count, N the library total in the declared count universe,
and s its RLE size factor. The point is x=C/s. The investigated kernels use

\[
P\mid C,N\sim\operatorname{Beta}(C+h,N-C+h),\qquad
A=\frac{N}{s}P.
\]

Thus the normalized kernel CDF is F_A(a)=F_P(sa/N), its quantiles scale by N/s,
and its variance scales by (N/s)^2. Its total probability remains one. Replacing
P by its point estimate C/N gives exactly C/s. This is the normalization rule
we should retain when evaluating fraction-based kernels; a separate normalization
algorithm is not missing from the comparison. For a log analysis, transform the
whole random variable A, not just its mean.

This is implemented in [the adapter's CDF and moment evaluation](../archive/development-20260923/tools/kernel_benchmark_adapter.py).
A fresh [bookkeeping audit](../.release/0.2.0/design-reset-20260923/normalization-audit.json)
checks all 184 saved normalization records, independently refits ten complete
matrices across five cohorts, and checks both transformed kernel CDFs in each.
It tests implementation, not calibration or the suitability of the prior.

There are still normalization questions to validate: RLE's reference assumptions,
certified internal controls, reference/test overlap, and uncertainty in estimated
factors. The current kernels condition on fitted factors; they do not propagate
factor-estimation uncertainty. Correct scaling does not equalize measurement
precision. N and s have different roles; neither establishes molecular complexity.

### “Uniform” and “Jeffreys” named priors, not kernel families

Both recent candidates use a **binomial count likelihood and a Beta abundance
kernel**. Uniform means a Beta(1,1) prior, producing Beta(C+1,N−C+1). Jeffreys
means a Beta(1/2,1/2) prior, producing Beta(C+1/2,N−C+1/2). The resulting kernels
are generally not uniform distributions. These were prior-sensitivity controls,
not a finding that these priors are optimal or new public normalization options.
Future reports should use the full model name and label the prior explicitly.

A discrete count likelihood and a continuous abundance kernel are compatible:
counts are integers, while an unknown proportion or rate need not be. The
[Beta distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html)
lives on [0,1]; the [beta-binomial distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html)
is a distribution of integer counts. They are not interchangeable objects.

| Count measurement model | What is needed to obtain an abundance kernel |
| --- | --- |
| Poisson counts with exposure s and unknown normalized rate a | Specify an inference rule for a. For example, a Gamma(α, rate β) prior gives Gamma(C+α, rate s+β). This is a derivation, not a selected prior. |
| Binomial counts with total N and unknown fraction p | A Beta prior gives the Beta abundance kernel above, then A=(N/s)P. |
| Negative-binomial counts | Specify the exposure, mean–dispersion relationship, technical meaning of dispersion, and an inference rule for the unknown abundance. An NB count law does not by itself specify the abundance kernel. |
| Beta-binomial counts | Specify the target fraction, technical dispersion and inference rule. A beta-binomial likelihood for a source fraction differs from beta-binomial prediction of future counts under a binomial–Beta model. |

Poisson/NB/binomial/beta-binomial remain the candidate **count models**. A model
card must also specify the inferred quantity and how the observed count becomes
a kernel. Resampling Poisson(C), for example, describes plug-in count noise and
collapses at C=0; it is not a complete solution for abundance uncertainty.
Do not treat real between-patient biological variation as technical dispersion.
A fitted population prior is optional research, not required by the objective.

### Location and spread both need validation

For the Beta candidates,

\[
E[A]=\frac{N}{s}\frac{C+h}{N+2h},\qquad
E[A]-\frac Cs=\frac{h(N-2C)}{s(N+2h)}.
\]

At low abundance this adjustment is approximately h/s. It therefore becomes
large in shallow libraries despite correct normalization. This connects directly
to the earlier concern about adding a raw pseudocount before scaling. Positive
width at zero necessarily entails positive mean, but its magnitude still needs
justification. That constraint does not justify collapsing zero kernels.

In the latest severe-depth study, own-mean controls reproduce much of the loss;
in an earlier weak-subset study, spreading mass itself caused a large loss.
These are different failure cases to retain, not evidence for a universal
location-only explanation. A numerically accurate kernel mixture is also not
automatically an unbiased estimate of the latent patient distribution.

## 3. Why the false-positive reports appear contradictory

We must reproduce the relevant failure with the **current matched baseline**
before claiming that a kernel repairs it. Historical results tested different
statistics, normalization assumptions, perturbations and null procedures.

| Evidence | What it tested; what must not be inferred |
| --- | --- |
| [Earlier method diagnosis](../archive/development-20260923/docs/wade-method-diagnosis.md) | Fitted bulk corrections in a two-stage subset test, plus normalization failures. The latest point comparator does not use that correction. |
| [Noise investigation](../archive/development-20260923/docs/noise-ties-and-precision.md) | Raw Poisson(C+1) perturbation introduces a normalized offset 1/s. Pure label-independent common-unit noise behaved differently. The latest points have no added count offset. |
| [Count-aware null study](../archive/development-20260923/docs/count-kernel-experiments.md) | With the same raw-area statistic, ordinary label permutations and a fitted actual-depth null gave very different rejection rates. Changing calibration is distinct from adding kernels. |
| [Latest depth stress](../archive/development-20260923/docs/kernel-depth-stress.md) | Shallow libraries were assigned independently of group. Large depth imbalance alone need not invalidate this randomized-label comparison. It does not establish robustness to condition-linked measurement differences. |

Even if biological abundance a is identical, Poisson sampling gives
Var(C/s | a,s)=a/s. Normalization fixes the conditional mean, not that variance.
Neither ordinary label permutation nor wider kernels automatically solves
condition-dependent measurement differences. Conversely, independent depth
variation is not sufficient reason to expect points to produce many false calls.

The latest selected-panel results are nominal single-gene false-positive rates;
earlier whole-family results sometimes report FDP/FDR. These are different
metrics. The latest point advantage does not certify point-based WADE generally,
and the losses of two particular Beta candidates do not reject uncertainty
modeling generally. Neither current candidate has earned production selection.

## 4. A scorecard that tests the biological quantities

Wasserstein area has abundance units; it is **not a fold change**. Detection and
estimation must be evaluated separately, including genes that were not called.
Use paired method comparisons and uncertainty across independent cohorts.

| Question | Required endpoint and truth |
| --- | --- |
| Are discoveries reliable? | Null rejection, all-null probability of any call, per-matrix FDP and mean FDP/FDR at the declared full-family threshold. Report sensitivity, precision and filtering failures alongside them. |
| Are abundances and effects recovered? | Error in patient abundances and group CDF/quantile curves; directional-area bias/RMSE; log2 fold-change bias, MAE/RMSE for explicitly defined contrasts. Separate latent finite-cohort truth from population parameters. |
| Where does enrichment occur? | Patient contributions to upward area, concentration of those contributions, and control-relative enrichment. Evaluate affected-patient precision/recall and ranking against planted memberships. Treat prevalence and component-fold recovery as separate claims; check null/shared-tail and bulk controls. |
| Does uncertainty help? | Interval coverage together with width, predictive checks, and paired degradation as reads are removed. Report coverage by true abundance/depth; observed-zero strata are useful diagnostics, not unconditional coverage certificates. |
| Is it useful on real data? | Stability and effect/ranking agreement under thinning; associations with independent fusion/DNA/protein annotations. Real unannotated genes are not known negatives. |

The existing simulator retains latent abundances and exact planted membership
for its original 36 profiles. Gate B now measures effect, curve and membership
recovery. These are assessment instruments, not validated production estimators.

Define the truth before writing its estimator. In the current simulator, b is
the unaffected-case multiplier, f the affected-case multiplier relative to the
original baseline, and π the actual planted fraction. The population mean ratio
is (1−π)b+πf. For b=0.5, f=4 and π=0.1, that ratio is 0.85, while affected cases
are 4-fold above the original baseline and 8-fold above the unaffected case
bulk. All three are different legitimate quantities. Rounding of subset size
must use the actual implanted patient count.

Planted component parameters are not always identifiable from two unpaired
nonparametric distributions: overlapping subgroups can admit several mixture
explanations. As clarified by the user, the maximum walk was intended to describe
subset-driven enrichment, not necessarily estimate π. Retain that role. Compare
its selected mass with planted membership to understand what it includes, without
making prevalence recovery its sole success criterion. Similarly, a quantile ratio
is not a causal within-patient fold change. Report observable curve contrasts;
validate component recovery in controlled mixtures with varying separation.
Use no truth-derived cutoff or known top-k fraction at evaluation time. Any
oracle localization is an explicitly labeled upper bound. Patient localization
remains descriptive unless separately calibrated for inferential claims.

## 5. Ordered work and decision gates

### A. Establish the contract and audit what exists — this reset

Agree on the target, normalization transform, model vocabulary, scorecard and
intended-use range above. The implementation audit is complete; the statistical
choices remain open. Keep the current production baseline and original studies
as identifiable comparisons. No new scientific benchmark or model selection
was performed during this reset.

### B. Build the small truth-recovery and failure-replay benchmark

Before another model sweep, implement the scorecard on the saved simulations.
Start with transparent null, broad-shift, shared-tail and minority-spike examples;
check descriptors on known latent abundances before introducing counts. Include
a lower bulk with an upward subset, zero/one/many counts, and the retained
weak-subset and singleton-collapse regressions. Record the expected answer for
each metric; the walk should cover the whole cohort for an everywhere-positive
bulk difference. Distinguish component estimation from observable descriptors.

Replay a bounded set of earlier false-positive cases with the current point
statistic, holding direction, scale, filter, factors, patient mass and null fixed.
Separate normalization error, raw-noise offsets, condition-associated depth and
legacy bulk-correction failures. Use known factors only as mechanism controls.
This is a diagnostic benchmark on reused data, not fresh confirmation.

**Gate:** we can explain what each estimator measures, verify truth accounting,
and identify which false-positive problem the proposed kernel must address.
Do not choose a new prior just because it wins a detection table.

**Executed instrument gate, 2026-09-23.** The
[runner and definitions](../research/README.md#gate-b-assessment) now implement the
[frozen protocol](../research/protocols/truth-recovery.json). The accepted
[local artifact](../.release/0.2.0/truth-recovery-20260923-v4/completion.json)
contains source/input hashes, 1,728 profile rows (including latent controls),
per-cohort scorecards, coverage/width, normalization errors and null replays.
There are four saved biological cohorts: Gamma and correlated lognormal at
250:250 and 50:500, each with all 36 original profiles at full depth and with
5% of libraries retaining 1/1,000 of their reads. This is reused-data diagnosis,
not independent confirmation; profiles within a cohort are not independent
replicates. No production behavior or statistical candidate changed.

An analytic example distinguishes cumulative enrichment from membership. Here,
90% of cases have precisely the control distribution and 10% are extreme spikes.
The maximum upper walk selects **99.1%**, because the mixture moves quantiles
throughout the distribution. With continuous first-order dominance it can select
the entire cohort. On the saved latent 10%-at-4-fold profiles, its mean selected
fraction is **94.6%**, before count noise. Above a depressed bulk it selects
10.0% on average. The latter favorable case cannot justify using the walk as a
general prevalence estimator. A selected-tail fold contrast likewise differs
from the planted component fold contrast. The
[analytic illustration](../.release/0.2.0/truth-recovery-20260923-v4/review/walk-limit.png)
has been visually checked; user approval is pending.

The current-code false-positive replay holds linear upward area, equal patient
mass and library-total scaling fixed in a two-category count experiment:

| Case/control depth ratio | Current label permutation | Saved fitted actual-depth null, same point statistic |
| --- | ---: | ---: |
| 1/4 | 23/24 null rejections | 1/24 |
| 1 | 0/24 | 0/24 |
| 4 | 23/24 | 1/24 |

Observed areas and all saved bootstrap/oracle point areas reproduce with the
current implementation. The current permutation column uses a new fixed bank;
the archived permutation bank gives the same aggregate counts. The fitted null
was reused, not refitted or selected here. These are nominal 0.05 single-gene
rejections; 1/24 has a wide exact 95% interval, approximately 0.1–21.1%.
This isolates a calibration problem when depth tracks condition. It is separate
from minority thinning independent of group and from the retired bulk correction.

With the transcriptome panels, every arm shares RLE and the same saved label
banks. At full depth all five arms call 79/96 upward alternatives and 2/32 nulls.
For the severe minority-depth intervention:

| Representation | Upward calls | Null calls | Mean-ratio log2 MAE versus latent truth |
| --- | ---: | ---: | ---: |
| Normalized points | 71/96 | 2/32 | 0.123 |
| Binomial–Beta, h=1, full kernels | 29/96 | 3/32 | 0.342 |
| Binomial–Beta, h=0.5, full kernels | 32/96 | 3/32 | 0.308 |

Mean-only controls call 29/96 and 34/96 respectively; their whole-group mean
contrasts equal those of their full kernels. RLE's case-minus-control factor
error stays below 0.009 log2 in these eight matrices. The new measurements
therefore support a location problem for these candidates in this stress case,
not a missing normalization transform. This deliberately severe intervention
is an operating-boundary diagnostic, not a typical bulk cohort or an argument
against uncertainty modeling generally.

In shallow libraries, central 95% Beta intervals cover latent truth in 43.8%
(h=1) and 84.2% (h=0.5) of the 3,744 patient–gene measurements, despite large
widths. Full-depth coverage is 93.8% and 94.8%. These are descriptive coverage
fractions across dependent measurements under the saved biological generators,
not prior-averaged calibration certificates. Positive width alone is insufficient.

The instruments caught inaccurate inverse-Beta values for a large asymmetric
shape. CDF checks and bisection now guard those requests; the initial incomplete
runs remain local evidence. Area errors use the bounded CDF integrator. Kernel
descriptions compare 64 and 128 nodes per patient, with transport bounds and
floating-point allowances. In 8/576 kernel descriptions the selected fraction
changes by more than one percentage point; the maximum is 9.2 points. These
cutoffs remain numerically unresolved as precise estimates, even aside from
their conceptual prevalence limitation. No p-value uses this quadrature.

**Completed mechanism replay, 2026-09-23.** The
[frozen replay](../research/protocols/mechanism-replay.json) and
[local results](../.release/0.2.0/mechanism-replay-20260923/completion.json)
recompute 160 normalization/test combinations: five settings, two cohorts, eight
reference choices, two scales. Current RLE reproduces the saved factors; current
unsigned W1 reproduces all observed statistics, 3,999-permutation exceedance
counts and 128-gene BH results. References contain 16,384 genes, disjoint from
the test panel. Selected identities are frozen: nomination is not retrained, and
this does not reproduce the user's complete biological control-discovery workflow.

The normalization failures persist without the retired two-stage correction.
With 60% upward-changed reference genes, the LOESS and log-GCV selections contain
only changed genes; mean linear-scale FDP is 70.2%, versus 1.6% with the valid
20-gene controls. When changed genes are especially quiet, even the independent
log-GCV selection contains only changed genes and gives 68.3% FDP. Conversely,
with 30% changed references, independent log-GCV controls improve this diagnostic
to 4.4% FDP; whole-reference RLE has a 0.30-log2 relative group-factor bias and
68.3% FDP. Neither low variance nor a majority of unchanged genes alone certifies
a reference under these generators. These are means over **two reused cohorts**,
not estimates precise enough to certify FDR. All-null replays make no BH calls.

The separate noise replay regenerates perturbations and reproduces every saved
linear/log observed upward area. New linear-upward tests use 255 permutations
with fixed known exposures. Among shallow-case null genes, nominal calls total
67/208 without perturbation, 145/208 with Poisson(C+1), and 208/208 with enormous
raw-count noise. Enormous noise added in common normalized units gives 6/208;
with equal depths it gives 7/208. In the mixed shallow-case panels, the latter
also suppresses alternative detection to 3/80. Large common-unit noise does wash
out biology; large raw noise divided by unequal exposures creates unequal group
backgrounds. These are nominal calls, **not a new BH/FDR experiment**. The old
fixed-exposure recipe is deliberately replayed, not recommended: these panels
lack the complete library needed to renormalize added counts coherently.

**Localization candidate, 2026-09-23.** The
[derivation and implementation](../research/README.md#describing-enrichment-without-estimating-prevalence)
preserve the walk and add two descriptions. First, attribute positive quantile
displacement to case patients, integrating over each entire kernel and sharing
tied abundance mass equally. Mean patient contribution equals upward W1. Its
effective contributor fraction describes concentration of area, not affected
prevalence. Second, find the abundance where case-minus-control upper-tail mass
is greatest; report case mass, control mass and their excess separately. Kernel
membership means mass above this cutoff, not probability of disease or alteration.

The [frozen localization protocol](../research/protocols/localization.json) ran
on the same four cohorts and both depth states, producing
[1,728 descriptive rows](../.release/0.2.0/localization-20260923/completion.json).
An exact separated 10% spike has perfect cutoff membership and contributes over
95% of the area. The [three-description illustration](../.release/0.2.0/localization-20260923/review/area-localization.png)
is provided for review; user approval is pending. The more variable latent
cohorts reveal important limits:

| Planted pattern | Area carried by altered cases | Contribution-ranking average precision | Case mass above cutoff | Cutoff precision / recall |
| --- | ---: | ---: | ---: | ---: |
| 10% at 4-fold | 68.0% | 0.943 | 31.9% | 40.2% / 96.8% |
| 10% at 16-fold | 92.3% | 1.000 | 32.6% | 44.9% / 100% |
| 10% spike above lower bulk | 99.7% | 0.973 | 8.7% | 100% / 87.3% |

These are equal-weight averages across the four cohorts and four baseline
abundances, using latent values before count noise; they are not independent
replicates. The continuous attribution identifies the strong contributors more
clearly than the cutoff delineates them. Even an exact additive bulk shift gives
equal contributions while the cutoff can include only half the cases. With a
proportional bulk shift, absolute area can concentrate in high-abundance cases
despite every case being altered. Independent samples from the same population
also produce apparent tails: localization alone is not a significance test.

Across all planted-subset profiles, full-depth contribution-ranking average
precision is 0.826 for points and 0.849 for h=0.5 kernels; after severe minority
thinning it is 0.765 and 0.608. These reused-data means do not establish a kernel
advantage. Numerical cutoff stability is inadequate: doubling kernel nodes
changes case tail mass by over one percentage point in 150/576 descriptions
(maximum 7.3 points), and one patient's membership by up to 48.4 points.
Contribution concentration changes by over one point in 9/576 descriptions;
ratios become unstable when total area is tiny. A regression also shows that
separating a discrete near-tie by 1e-10 can move a CDF cutoff sharply while area
and concentration barely change. No added tolerance was tuned to hide this.
Kernel cutoff errors here include the use of finite quantile nodes: a future
cutoff evaluation should use the continuous mixture CDF directly and assess
whether its maximizing location is stable. Accurate W1 integration does not
guarantee an accurate maximizing cutoff.

**Decision:** retain the maximum walk and develop area attribution as a
descriptive companion. Do not promote this hard cutoff or either Beta candidate
to production. Gate B now supplies the instruments and distinct failure
mechanisms needed for gate C. The next scientific step is a measurement-model
card and isolated zero/one/many-count location, interval and predictive checks,
before another detection sweep. Weak-subset power loss and singleton
fitted-reference collapse remain historical sentinels; neither legacy fitter
was rerun in this gate.

### C. Derive and validate one measurement candidate at a time

Write a short model card specifying count likelihood, abundance target,
normalization, inference/prior rule, zero behavior, technical assumptions and
null calibration. Begin from the four-model taxonomy, without assuming that a
fitted reference or another fixed Beta prior is the answer. Check locations,
intervals and predictions across zero/one/many counts and depths; retain precise
spikes. If estimating priors or dispersion, evaluate their estimation uncertainty
and protect biological heterogeneity. Avoid validation solely under the model's
own generative assumptions.

Use three matched controls: normalized points, the candidate's own means as
points, and full kernels. This distinguishes changing location from spreading
mass. Test linear and log transformations separately. Inferential calibration is
a separate design choice; changing it must be explicit and paired across arms.

**Gate:** justify the measurement representation and show an interpretable
benefit in the scorecard without unexplained regressions. Reducing all calls or
using very wide intervals is insufficient. Agree numerical tolerances and
acceptable loss/gain margins before fresh evaluation; none are approved here.

#### Measurement diagnosis completed, 2026-09-24

The [model card](../research/README.md#measurement-model-card-and-diagnostics),
[frozen protocol](../research/protocols/measurement.json) and
[completed run](../.release/0.2.0/measurement-20260924/completion.json) isolate the
existing h=1 and h=0.5 binomial–Beta candidates. There are six depths from 1,000
to 100 million counts; observed counts 0, 1, 2, 5, 10, 100 and 1,000; and fixed
true abundances from 0.001 to 1,000 CPM, with structural zero reported separately.
Exposures are known, with additional scaling checks. This deliberately removes
normalization estimation, group labels and gene testing from the question.
It does not simulate every biological workflow or certify transcriptome FDR.

**The individual posterior calculations are consistent; the low-count model
still needs revision.** Prior-generated checks give 94.91–95.06% central-interval
coverage, with Monte Carlo SE about 0.15 percentage points (20,000 draws per
cell). Exact fixed-truth binomial sums give very different coverage near zero.
For example, at 1 CPM and depth 1,000, central coverage is 0% for h=1 and 99.90%
for h=0.5. At still smaller positive abundances both can give 0%. A positive
central lower bound eventually excludes sufficiently small true fractions.
One-sided upper bounds retain the zero boundary and answer a different question;
they are not uniformly calibrated at 95% for every positive abundance either.
These results distinguish prior-averaged Bayesian consistency from fixed-truth
coverage; neither should be mislabeled as the other.

Zero uncertainty is positive, as intended, but its location is consequential:

| Total counts, observed C=0 | Mean, h=1 | Mean, h=0.5 | Central 95% interval, h=0.5 |
| --- | ---: | ---: | ---: |
| 1,000 | 998 CPM | 499.5 CPM | 0.491–2,508 CPM |
| 1 million | 1.000 CPM | 0.500 CPM | 0.000491–2.512 CPM |
| 100 million | 0.010 CPM | 0.005 CPM | 0.00000491–0.02512 CPM |

The density's mode is zero in all these cases. Positive width and mean do not
require a positive mode. At 100 expected reads, central coverage is about 95%
for both candidates and relative mean bias is small. The rare-fraction Gamma
limit reproduces the location problem: its mean remains (C+h)/s. On the fixed
count grid with C/N ≤ 0.001, the checked Beta and Gamma interval/median quantiles
differ by at most 0.33%. Changing from binomial to Poisson alone is not a remedy.

**There is also a whole-library prior-budget inconsistency.** For G exhaustive,
disjoint gene-count categories with counts summing to N, the current fraction
means sum to

\[
\sum_g E[p_g\mid C_g,N]=\frac{N+Gh}{N+2h}.
\]

With 16,384 genes and 1,000 counts, this is **17.35 for h=1 and 9.18 for h=0.5**,
although fractions must sum to one. At 10 million counts the totals are 1.00164
and 1.00082: the discrepancy is small at ordinary bulk depths, but cannot be
ignored in the shallow-library regime we intended to improve. Each patient
kernel still integrates to one; that separate invariant passes. This is a
collection of incompatible marginal priors, not a missing RLE scaling transform.

A coherent multinomial–Dirichlet construction would instead have marginal
Beta(C_g+α_g, N−C_g+α_total−α_g), with α_total=Σα_g. The
[Dirichlet support](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dirichlet.html)
enforces the fraction-sum constraint. This is a derivation of the required
bookkeeping, **not a selected replacement**. Simply assigning h to every gene
and changing the denominator to N+Gh would introduce thousands of prior counts
and strongly shrink shallow libraries. Total prior strength and its allocation
need explicit justification; a fitted patient-population reference is not
required by this identity.

**Prediction adds useful evidence beyond interval width.** In fresh independent
count pairs from a true 1 CPM gene at depth 1,000, the true probability of a
nonzero repeat is about 0.1%. After an observed zero, the current kernels predict
approximately 50% (h=1) or 29% (h=0.5). The plug-in point predicts zero probability
and incurs infinite log loss when a repeat is nonzero. Thus neither the point's
certainty nor these kernels' positive predictions settle the low-count problem.
The saved scorecard reports held-out log loss and zero-event Brier scores, with
Monte Carlo uncertainty, for both repeat depths. These are predictions of discrete
counts, not new gene-level significance tests.

**Spreading uncertainty is not automatically denoising the cohort distribution.**
For repeated measurements at fixed truth, the variance of a draw from the average
kernel is Var_C(E[A|C]) + E_C(Var[A|C]). At 100 expected reads in the 1 CPM example,
this is about twice the variance of normalized points. The calculation is exact
for this measurement model. It explains why averaging valid uncertainty kernels
need not recover the latent population distribution or remove depth-dependent
spread. Location, spread and calibration still require separate evaluation.

**Extra technical variation changes the target.** In the supplied 10%-CV
preparation experiment at 1,000 CPM and depth 100 million, h=0.5 intervals cover
the realized library fraction in 95.24% of replicates, but the source fraction
before preparation in only 4.81% (SE 0.15 points for each). Binomial kernels can
describe counting precision while missing preparation uncertainty. Technical
dispersion must have a measurement interpretation and supporting data; biological
variation across patients must not be relabeled as technical noise.

Numerical checks reproduce the existing kernel means/CDFs and independently
verify future-count probabilities. All checked inverse-CDF residuals are below
1e-7. The largest 128/256-node disagreement in expected log2(1+A) is 1.6e-5;
this is a quadrature check, not a log-W1 validation. The
[measurement illustration](../.release/0.2.0/measurement-20260924/review/measurement-audit.png)
is ready for review; user approval is pending.

**Recommendation from this measurement audit:** derive one candidate with a coherent whole-library allocation
of prior information and an explicit counting-versus-preparation uncertainty
target. Reuse these same location, coverage, predictive and conservation checks
before returning to cohort W1 benchmarks. Preserve positive uncertainty at zero
and precise spikes. Neither reducing h again nor renormalizing posterior means
alone constitutes a justified solution. No production kernel has been selected.

#### Preserve observations and individual kernel modes — 2026-09-24

The user clarified an additional design requirement: **each individual kernel
must peak at the observed abundance**, including zero. This is now an explicit
gate, rather than an assumption about what a Bayesian posterior will do.

- Raw integer C and the observed total N stay unchanged. Kernel shape parameters,
  inferred means and simulated future counts are separate objects. The earlier
  positive-mean calculations did not edit the observed counts or create reads.
- At C=0, positive width and positive mean are allowed; the unique density mode
  must remain zero. At C>0, require the mode at C/N in fraction space, or C/s on
  the linearly normalized axis. The prior h=0.5 does not generally satisfy this
  positive-count requirement; h=1 does for the individual binary Beta kernel.
- A fixed-total joint model must also conserve the fraction sum and hence the
  sum of marginal means. This does not require each marginal mean to equal its
  observed count. An upward mean adjustment for a zero must be balanced elsewhere.
- The requirement concerns the physical abundance axis. A nonlinear log transform
  preserves probability mass but changes density through its Jacobian; its density
  mode need not be the transformed physical mode. Keep the observed marker fixed
  and transform the full kernel without recentering it.

Mode preservation is a structural requirement, not sufficient evidence of
robustness: the existing h=1 marginal already preserves C/N while retaining the
low-count mean and predictive problems above. W1 integrates the entire kernel,
so its spread and tail mass still need the existing validation checks.

**A joint mode is not the collection of marginal modes.** We derived and tested
the natural conjugate candidate: a uniform Dirichlet prior with a multinomial
likelihood gives Dirichlet(C+1). Its most likely **whole-library vector** is C/N,
and every zero has positive marginal uncertainty. However, its individual
marginals are Beta(C_g+1, N−C_g+G−1), with mode C_g/(N+G−2).
The density and fraction constraint follow the
[Stan Dirichlet definition](https://mc-stan.org/docs/functions-reference/simplex_distributions.html);
the mode equations here follow by differentiating those densities.

For observed counts **(0, 2, 8)**, with total 10:

| Quantity, expressed in count units | Value |
| --- | --- |
| Unchanged observed counts | (0, 2, 8) |
| Joint posterior mode | (0, 2, 8) |
| Individual kernel modes | (0, 1.818, 7.273) |
| Individual kernel means | (0.769, 2.308, 6.923), summing to 10 |

Thus this candidate conserves the total and the joint mode but **fails the user's
individual-mode requirement**. At N=1,000 and G=16,384, it retains only 5.75% of
each positive count's modal value. Calling the joint MAP an individual kernel's
mode would hide this failure. Merely checking the sum of modes is insufficient,
too: the older h=0.5 marginals can sum to 10 in this example while moving the
two positive-count modes to 1.667 and 8.333.

This is not solved by another proper Dirichlet allocation. Let α_total=Σα_g and
let K be the number of positive-count genes. For any positive count strictly
below N, preserving its interior marginal mode requires

\[
\frac{C_g+\alpha_g-1}{N+\alpha_{total}-2}=\frac{C_g}{N}.
\]

Summing the resulting equations over the positive-count genes gives
Σ_{C_g=0} α_g = 2−K. With K≥3 this is impossible for positive prior parameters.
With K=2 and any zero-count category, it also conflicts with a proper positive
prior for those zeros. This rules out the **Dirichlet posterior family** for
the general exact-mode contract; it is not an impossibility result for every
uncertainty construction.

The [frozen constraint audit](../research/protocols/mode-contract.json) and
[results](../.release/0.2.0/mode-contract-20260924/decision.json) check five explicit
count vectors and 30 depth/gene-universe combinations. Numerical density checks,
checked interval calculations and read-only input assertions confirm the
derivation. Raw counts, integer dtype and totals stay unchanged throughout.
The candidate is rejected at this design gate; a full performance sweep is
unnecessary for a candidate that already violates the requested behavior.

The next derivation below keeps the exact-mode requirement. It does not replace
individual modes with joint MAP, overwrite a reported mode, or renormalize means
and assume the peaks are preserved.

#### Alternative constructions — 2026-09-24

**The constraints are compatible, but satisfying them does not determine a valid
measurement kernel.** We derived a discrete construction and a continuous one,
implemented both in [research](../research/wade_research/mode_constructions.py),
and checked them with a [frozen protocol](../research/protocols/mode-constructions.json).
Both preserve observations and totals and have positive zero variance. Neither
is selected. These are measurement diagnostics, **not gene-level false-discovery
rates or a comparison of WADE detection performance**.

**1. A discrete, fixed-total construction.** Let w be an explicit positive vector
summing to one. Define

\[
q_g=\frac{C_g+w_g}{N+1},\qquad
Z\mid C\sim\operatorname{Multinomial}(N,q).
\]

Each simulated library has exactly N counts. Its one-gene kernel is
Binomial(N,q_g); the [binomial PMF](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html)
gives a unique mode floor((N+1)q_g)=floor(C_g+w_g)=C_g.
At C_g=0, q_g>0, so the mean and variance are positive. The normalized random
variable is Z_g/s, giving mode C_g/s and variance Nq_g(1−q_g)/s². The observed
C and N are never edited.

This is **smoothed count resampling**, not a posterior for the unknown fraction.
Although q also equals the posterior mean under Dirichlet(w), fixing q and
resampling is different from integrating over that posterior. The latter gives
a [Dirichlet–multinomial predictive law](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dirichlet_multinomial.html)
with extra variance and generally different modes. Calling these the same
kernel would conceal a changed uncertainty target.

There is a useful family-wide obstruction. For any nondegenerate multinomial
whose individual modes uniquely equal C_g,

\[
C_g<(N+1)q_g<C_g+1.
\]

Thus w_g=(N+1)q_g−C_g lies in (0,1) and sums to one: the formula above
characterizes the entire family. If Z0 is the set of zero-count genes,

\[
\sum_{g\in Z0}E[Z_g]
=\frac{N}{N+1}\sum_{g\in Z0}w_g<1.
\]

Consequently, all unseen genes together receive less than one expected resampled
read. Changing the allocation cannot remove that bound. With the diagnostic
uniform allocation w_g=1/16,384 and N=1,000, a zero gene has a positive variance
but **99.9939% probability of zero and a central 95% interval [0,0]**. At one
true expected read, the corresponding percentile intervals cover the underlying
fraction in only **63.17%** of repeated libraries. These are kernel percentile
intervals assessed against latent truth, not promised exact confidence intervals.
This construction solves the modal bookkeeping but does not solve uncertainty
at low counts. The uniform allocation is an explicit diagnostic assumption,
not a learned RNA-seq prior or a normalization option.

**2. A continuous construction on the whole-library composition.** Set m=C/N.
Take a full-composition draw Q from a proper Dirichlet(C+α) posterior and an
independent U drawn uniformly from (0,1). Define

\[
P=(1-U)m+UQ.
\]

Use **one shared U for the whole library**. Every draw sums to one; all its
coordinates are positive with probability one. Each patient still contributes
one unit of probability. Q is a draw of a possible composition, not a fitted
normal-patient reference. Counts remain discrete observations; P is the unknown
fraction being represented continuously. Normalization remains A_g=(N/s)P_g.

Given Q_g, P_g is uniform on the line segment between m_g and Q_g. Mixtures of
such segments are a standard route to unimodality; see the radial representation
in [Van Parys, Goulart and Morari, §3.1](https://mitsloan.mit.edu/shared/ods/documents?DocumentID=4758).
For a base marginal density f_Q, the density to the right of m is

\[
f_P(x)=\int_x^1\frac{f_Q(q)}{q-m}\,dq,\qquad
f'_P(x)=-\frac{f_Q(x)}{x-m}<0.
\]

To the left, f_P(x)=∫₀ˣ f_Q(q)/(m−q)dq and f'_P(x)>0.
Hence **every individual mode is exactly m**, including the zero boundary.
The peak can have an integrable infinite density; it is not an atom and its
variance is not zero. This mathematical property alone does not justify that
shape for a measurement kernel.

Writing μ=E[Q] and Σ=Cov(Q), independence gives

\[
E[P]=\frac{m+\mu}{2},\qquad
\operatorname{Cov}(P)=\frac{\Sigma}{3}
+\frac{(\mu-m)(\mu-m)^T}{12}.
\]

For counts (0,2,8) and α=(1/3,1/3,1/3):

| Quantity, in original count units | Discrete construction | Continuous construction |
| --- | --- | --- |
| Unchanged observations | (0,2,8) | (0,2,8) |
| Individual kernel modes | (0,2,8) | (0,2,8) |
| Kernel means | (0.303,2.121,7.576) | (0.152,2.061,7.788) |
| Sum of means | 10 | 10 |
| Variance of the zero-count kernel | 0.294 | 0.0893 |

This establishes feasibility beyond the Dirichlet posterior family. However,
**the contraction is an additional inference rule, not Bayes' rule under the
original likelihood and prior**. It has to earn its measurement interpretation.

It fails that check in its simplest form. In a high-information regime where
Q≈Normal(m,σ²), P−m≈UσZ with Z standard normal. Its variance is σ²/3 and its
97.5th percentile is 1.258σ. Across repeated measurements, the resulting central
95% interval covers truth with probability 2Φ(1.258)−1=**79.17%**. The exact
binomial study agrees: at N=100 million, G=16,384 and 100 expected reads,
coverage is **78.90%**, compared with **94.83%** for its uncontracted
one-total-prior-count base. Preserving the peak did not preserve the width.

We evaluated total prior budgets 1 and G, with matched uncontracted controls,
to separate allocation from contraction. Neither fixes both problems. Budget G
can substantially move mass even when the peak stays put: at N=1,000, G=16,384
and observed C=100, the contracted kernel's mode is 100 but its mean is **52.905**.
W1 integrates that displaced mass; an unchanged mode does not protect the effect.
Multiplying a width until one coverage endpoint looks correct would be another
unsupported rule, so no such repair was selected.

**Evidence and numerical limits.** The
[completed artifact](../.release/0.2.0/mode-constructions-20260924b/decision.json)
contains 15 joint-library checks, 144 observed-count rows and 360 exact
repeated-binomial comparisons across six depths. Tests independently check PMF
maxima, continuous density monotonicity, joint conservation, a closed-form CDF,
simulation moments and exact finite-sum moment calculations. The first run stopped
when a CDF integration could not reach 5e-11 at depth 100 million. Its partial
artifact is retained. The documented numerical amendment compares 1e-7 and 5e-9
tolerances; maximum CDF disagreement is 2.67e-10, estimated tight integration error
is below 5.94e-10, and no coverage classification is near a threshold within the
reported numerical allowance. Omitted binomial mass is at most 1.52e-11.
These are numerical diagnostics, not mathematical error bounds on the underlying
special-function implementation. No biological data, fitted RLE, technical
dispersion or cohort significance was tested in this gate.

**Next decision gate:** retain the mode and conservation requirements, but require
a count-likelihood justification for the full kernel before another cohort sweep.
Work first on a small, exhaustively evaluable joint library: distinguish sampling
of repeat counts from uncertainty in latent fractions; account for total unseen
mass once; and check modes, likelihood support, several interval levels and
independent-count prediction together. If a mode-preserving transformation is
proposed, it must retain the high-information counting variance and pass those
checks without a fitted width multiplier. The unresolved work is deriving that
inference rule, not optimizing these two rejected measurement candidates.
No production kernel or added public option follows from this study.

### D. Validate the complete usable workflow on fresh data

Freeze one candidate and its primary claims, metrics, acceptance margins,
replication/precision target and new seeds. Run full-transcriptome null and
partial-null cohorts, including RLE/internal controls, reference/test overlap,
filtering and BH multiplicity. Use attainable permutation resolution and audit
stopping/numerical error. Single-gene pilots do not certify transcriptome FDR.
If a fitted null is required, validate the actual estimated procedure and repeat
relevant fitting in null simulations; supplied true models are only controls.

Primary designs should represent realistic large-cohort bulk RNA-seq with
moderate overlapping depths. Add isolated shallow libraries and condition-linked
depth as separately reported boundaries, not the sole definition of success.
Use multiple biological backgrounds, strong and modest effects, and repeated
cohorts. Include rare 5–10 log2 increases as well as the more modest spikes already
studied. Compare usable methods at comparable achieved error control, while also
reporting their nominal-threshold behavior. Greater information loss may rightly
reduce power; it must not create unsupported discoveries.

Compare with frozen production WADE and native DESeq2/edgeR for appropriate
gene-level targets; compare with OUTRIDER and related expression-outlier methods
for separately defined patient-by-gene targets. Do not equate their p-values or
silently aggregate different hypotheses. Sample-size curves test the proposed
large-cohort advantage; they must not assume WADE wins. Retain established data
resources and provenance from the [benchmark inventory](../archive/development-20260923/docs/kernel-performance-benchmarks.md).

**Gate:** useful global/subset estimation and detection, adequate calibration in
each claimed setting, and a reproducible kernel benefit attributable to location
and/or spread with the distinction explicit. Universal superiority is unnecessary.
If no useful benefit appears, revise the model rather than declaring validation.

### E. Finalize the release after scientific selection

Implement the selected procedure in a compact API without a prior/bandwidth/model
menu. Reuse normalization/filtering and validated resampling infrastructure.
Optimize measured bottlenecks only then; preserve a checked Python reference if
adding Rust. Revalidate all exposed directions/scales and multiplicity choices.

Rebuild every demo, benchmark and example notebook. Complete TCGA prostate
fusion-oriented and breast subtype gene-count analyses, with no splice-junction
branch. Finalize counting-pipeline provenance, inspect plots and results, obtain
user approval of the exact final artifacts, pass packaging and cross-platform CI,
and request release/tag authorization. Previous point-method notebook approval
or numerical audits cannot approve changed kernel results.

## 6. Where to look now

- This file: current scientific design, unresolved choices and next steps.
- [PLAN.md](../PLAN.md): short release queue and completion gates.
- [Documentation index](index.md): production baseline references and dated evidence.
- [Reset audit and preserved versions](../.release/0.2.0/design-reset-20260923/README.md):
  code checks, original planning documents and scope of this change.

The existing production baseline is unsigned empirical W1 on log-normalized
points; the latest kernel experiments are research tools. Neither is being
silently declared the final 0.2.0 design. All existing negative results remain
part of the evidence; we are resetting the interpretation and work sequence.
