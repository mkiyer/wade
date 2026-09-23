# WADE 0.2.0: one design and validation plan

2026-09-23 — design reset following the user's review of the depth-stress study.
**This is the canonical plan for future work.** Earlier documents remain evidence
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

Uncertainty remains part of the intended 0.2.0 release. Universal superiority,
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
| Is the right subset localized? | Subset-fraction error, subset-versus-baseline fold error, and affected-patient precision/recall and PR-AUC against planted memberships. Score false localization under null/shared-tail profiles and behavior under pure bulk shifts. |
| Does uncertainty help? | Interval coverage together with width, predictive checks, and paired degradation as reads are removed. Report coverage by true abundance/depth; observed-zero strata are useful diagnostics, not unconditional coverage certificates. |
| Is it useful on real data? | Stability and effect/ranking agreement under thinning; associations with independent fusion/DNA/protein annotations. Real unannotated genes are not known negatives. |

The existing simulator retains latent abundances and exact planted membership
for its original 36 profiles. We did measure latent-area error in the latest
study, but have **not completed fold-change, prevalence or membership recovery**.
Recovering those quantities is now a primary development task.

Define the truth before writing its estimator. In the current simulator, b is
the unaffected-case multiplier, f the affected-case multiplier relative to the
original baseline, and π the actual planted fraction. The population mean ratio
is (1−π)b+πf. For b=0.5, f=4 and π=0.1, that ratio is 0.85, while affected cases
are 4-fold above the original baseline and 8-fold above the unaffected case
bulk. All three are different legitimate quantities. Rounding of subset size
must use the actual implanted patient count.

Planted component parameters are not always identifiable from two unpaired
nonparametric distributions: overlapping subgroups can admit several mixture
explanations. The walk's selected tail fraction must therefore be evaluated
against planted π, not declared equal to it. Similarly, a quantile ratio is not
a causal within-patient fold change. Report observable curve contrasts generally;
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

### B. Build the small truth-recovery and failure-replay benchmark — next

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
