# WADE research and validation workspace

The active task is point-method validation and selection for 0.2.0. Measurement-kernel
work is deferred after the user's 2026-09-25 scope change. This workspace is
separate from `src/wade`, excluded from release distributions, and never imported
by the public package. Follow [PLAN.md](../PLAN.md) for the release sequence;
the [kernel roadmap](../docs/kernel-validation-roadmap.md) retains deferred work.
The point candidate below is under evaluation, not a new public default.
The user requires directional biomarker discovery and rejects unsigned-only
testing as the release solution. The active task is to correct the biological
scorecard and diagnose the existing directional prototype's failures.

## Active: complete point workflow

`wade_research/point_workflow.py` evaluates exact upward and downward areas
together on normalized points, with a shared label stream and one BH family
over all gene-direction pairs. It reuses production normalization, pooled
filtering and the audited stopping rule. Linear/log is an explicit, independent
choice. It is a readable NumPy reference; native optimization and public API
migration follow scientific selection.

```sh
PYTHONPATH=src:research python -m pytest -q research/tests/test_point_workflow.py research/tests/test_point_assessment.py
PYTHONPATH=src:research python research/assess_points.py --output .release/0.2.0/point-workflow-new-run
```

The [frozen first-run protocol](protocols/point-release.json) generates two fresh
16,384-gene matrices without external data. The runner stores raw counts,
latent abundances, truth, source/input hashes and complete inference records.
Both directions have a 999,999-shuffle cap; the retained unsigned log-W1 baseline
uses identical inputs, factors, test mask and seed. Effects and null discoveries
are reported together. Two matrices establish execution, not reliable FDR or
comparative sensitivity. The expanded comparison below supplies replicated
development evidence; established-input replay and fresh confirmation remain.

## Replicated point comparison

The [comparison protocol](protocols/point-comparison.json) freezes 120 independent
16,384-gene matrices: 20 null and 20 mixed replicates at each of 250:250, 50:500
and 500:50 cases:controls. Gamma–Poisson and correlated lognormal–Poisson biology
are balanced within each cell. Four point procedures separate linear/log scale
from unsigned/paired-directional area. Native DESeq2 and edgeR QL use the same
integer matrices; edgeR has its ordinary TMM analysis and a matched-RLE check.
The comparison is development evidence, not fresh release confirmation.

**Completed 2026-09-26:** all 120 datasets passed exact public-baseline parity;
independent review checked 3,271 files, native input identity, stopping/BH,
primary call counts and quantile areas. All 175 tests pass. The
[assessment](../docs/index.md#replicated-directionscale-comparison--2026-09-26)
records results and scope. The original endpoint credits any distribution
change, including an opposite-direction call; it cannot select a biomarker
method. The recommendation to retain unsigned testing is withdrawn. Preserve
these frozen outputs while adding a directional relevance assessment. Native
results must be interpreted by generating biology and target. Established-input
replay and fresh confirmation follow the corrected scorecard and diagnosis.

```sh
# Activate an environment containing Python and rustc first. The native R
# library path can be supplied with --r-library; it is not a WADE dependency.
PYTHONPATH=src:research python research/compare_points.py \
  --output .release/0.2.0/point-comparison-new-run --workers 4
PYTHONPATH=src:research python research/review_points.py \
  --source .release/0.2.0/point-comparison-new-run \
  --output .release/0.2.0/point-comparison-new-review --workers 4
```

`point_factorial.py` evaluates six statistic lanes with shared permutations.
The optional benchmark loop in `point_permutations.rs` matches the NumPy oracle;
every full dataset independently replays the public unsigned log baseline and
requires exact statistics, p-values, q-values and stopping records. This does
not change production Rust or expose new public arguments.

`point_simulation.py` saves latent abundance, count, mean-effect and membership
truth. `point_comparison_native.R` preserves native fitting, outlier handling
and inferential exclusions. `review_points.py` checks exact native input
identity, independently recomputes BH/call counts, compares profile areas to
quantile integration, and reports matrix-level intervals and figure hashes.
Figures remain pending user approval. Undefined ratios, filtered alternatives,
method-specific exclusions and generator-specific performance stay visible.
The completed inference is in
[`point-comparison-20260925`](../.release/0.2.0/point-comparison-20260925/audit.json);
the [final review](../.release/0.2.0/point-comparison-20260925-review-v3/audit.json)
adds model-stratified figures and corrects displayed matrix counts without
changing inference. Review versions preserve source and figure hashes. All six
PNG/PDF figure pairs remain pending approval.

The four point tests share the same empirical mean-ratio effect estimator.
Changing an inferential statistic does not create a different fold-change
estimate. Area recovery is assessed separately on the corresponding latent
linear/log curves; unsigned or directional area is never labeled fold change.

## Diagnostic audit of the failed upward subset

`audit_subset_failure.py` audits the balanced depressed-bulk/fourfold-subset
profiles from a completed comparison. It independently integrates all 80
profiles, regenerates two disclosed original matrices, runs fresh fixed-budget
direct-sort permutations on four selected profiles, substitutes known exposures
and latent abundances, and enumerates a clean three-level toy's label null.
It does not modify the original inference, select a new production method or
provide an independent power estimate. See the
[standalone review](../docs/independent-review.md) for findings and limitations.

```sh
PYTHONPATH=src:research python research/audit_subset_failure.py \
  --source .release/0.2.0/point-comparison-20260925 \
  --output .release/0.2.0/subset-failure-audit-new
```

The completed audit is in
[`subset-failure-audit-20260926`](../.release/0.2.0/subset-failure-audit-20260926/audit.json).
All generated evidence remains separate from the saved original comparison.

## Retained kernel instruments

| Module | Responsibility |
| --- | --- |
| `wade_research/kernels.py` | Binomial–Beta patient kernels on the normalized-count axis; equal-patient group areas with numerical bounds. |
| `wade_research/points.py` | Matched empirical directional/unsigned area controls. |
| `wade_research/resampling.py` | Group weights, label banks and bounded numerical scoring of those banks. |
| `wade_research/_cache.py` | Exact CDF reuse within one immutable kernel instance. |
| `wade_research/assessment.py` | Checked group curves, effect contrasts, descriptive walk, membership/ranking scores and kernel interval diagnostics. |
| `wade_research/localization.py` | Patient contributions to upward area, concentration, and an experimental control-relative tail cutoff. |
| `wade_research/measurement.py` | Existing Beta kernels' moments, checked quantiles and discrete future-count prediction. |
| `wade_research/mode_constructions.py` | Discrete fixed-total resampling and continuous mode-preserving contraction, isolated for structural and measurement audits. |
| `assess_kernels.py` | One frozen truth-recovery and historical-null replay runner; no imports from archived code. |
| `replay_mechanisms.py` | Current-code replay of normalization and raw-noise failures, with explicit historical limits. |
| `assess_measurement.py` | Frozen zero/one/many-count, coverage, prediction and fixed-total diagnostics. |
| `derive_mode_constructions.py` | Frozen alternative-construction audit; exact repeated-count coverage, joint invariants and analytical limits. |

The earlier adapter and prior-contrast subclass are consolidated into one
`BetaKernelArea`. Size factors and prior shape are required inputs: normalization
is explicit and no prior is silently selected. Shapes 1 and 0.5 reproduce the
two previously studied candidates; neither is a production recommendation.
The numerical routines do not fit normalization, priors or a biological null.
Label permutation remains conditional on a justified exchangeable design.

```python
import wade
from wade_research import BetaKernelArea, label_coefficients

# counts: complete genes × patients raw count matrix; condition: 1/0 labels.
norm = wade.fit_normalization(counts)
kernel = BetaKernelArea(
    counts[gene_index], counts.sum(axis=0),
    size_factors=norm.size_factors,
    prior_shape=1,  # Explicit replay of one historical candidate, not a default.
)
bounds = kernel.scores(label_coefficients(condition))
```

`bounds` measures a numerical integration interval, not biological uncertainty
or a p-value. The full kernel uses A=(N/s)P; supplying the same RLE factors as
the point control puts both on the same abundance axis. The current evaluator
supports linear area. Research descriptions measure mean ratios and selected-tail
contrasts; they do not identify latent mixture components. Log kernels, validated
subset estimators and complete kernel-workflow inference remain scientific work.

## Gate-B assessment

Run the analytic instrument tests without external data:

```sh
PYTHONPATH=src:research python -m pytest -q research/tests/test_assessment.py
```

With the original local study artifacts present, run the bounded replay:

```sh
PYTHONPATH=src:research python research/assess_kernels.py \
  --output .release/0.2.0/truth-recovery
```

The runner refuses to overwrite results. It freezes the
[protocol](protocols/truth-recovery.json), executable sources and input hashes
before evaluation. It writes per-profile effects, per-cohort scorecards, interval
coverage/width, normalization errors, and the null replay with denominators.
The source artifacts under `.release/` are required for this replay; a clean
checkout can still run every analytic and numerical test.

Points use exact empirical quantile curves. Kernel descriptions use equal-mass
quantile nodes for each patient, with 64/128-node convergence checks and an
analytic transport-error bound plus an explicit floating-point allowance.
Inverse Beta CDFs are checked against the CDF; failing requests use bisection.
Detection retains the saved bounded CDF integrator's conservative p-values,
and does not use the description approximation. Area-error scores also use that
CDF integral; curve/localization scores retain their explicit quadrature limits.

To render the exact latent counterexample for review:

```sh
PYTHONPATH=research python research/plot_assessment.py \
  --output .release/0.2.0/truth-recovery/review
```

The PNG/PDF and their hash manifest mark user approval as pending. The completed
[gate-B results](../docs/kernel-validation-roadmap.md#b-build-the-small-truth-recovery-and-failure-replay-benchmark)
record the selected run and its limits. Existing data are diagnostic evidence,
not new independent calibration or a selected production method.

The scorecard keeps three contrasts separate: the whole-group mean ratio,
the selected-tail versus control mean ratio, and the planted affected-group
versus control mean ratio. Missing zero ratios remain missing, without a hidden
pseudocount. Membership precision/recall use fractional selected patient mass;
average precision assesses patient-mean ranking separately. A null profile's
descriptive tail is not an individual significance call. Shared biological tails
are null for differential membership, and broad upshifts affect all cases.

An analytic regression exposes a crucial limit: adding a 10% spike to an
otherwise unchanged, heterogeneous case distribution can make the upper walk
select almost every case. Its cutoff is an observable curve description,
**not generally an estimator of the planted subset fraction**. This limitation
exists before any counts, normalization or kernels are introduced.

## Describing enrichment without estimating prevalence

Keep the upper walk as a cumulative enrichment description. The new prototype
asks two separate questions: **who contributes the area**, and **where is case
tail mass most in excess of control tail mass**? Neither is a patient-level test.

For case quantile u, define positive displacement
`d(u) = max(Qcase(u) − Qcontrol(u), 0)`. At a tied case abundance a, average d
across the quantile interval belonging to a. Call this average `dbar(a)`.
Patient i contributes

\[
r_i=\int \bar d(a)\,dK_i(a),\qquad
\frac{1}{n_{case}}\sum_i r_i=T_\uparrow.
\]

Each patient retains equal probability mass. Kernels distribute that patient's
contribution over abundance; they do not become confidence weights. Splitting
tied abundance mass symmetrically makes attribution independent of patient
ordering. This derives from the same one-dimensional quantile coupling used to
[compute W1](https://pythonot.github.io/_modules/ot/lp/solver_1d.html).
`mean(r)^2 / mean(r^2)` describes the effective fraction carrying the area
(zero when area is zero). It is not a mixture estimate: proportional bulk shifts
can concentrate absolute contributions despite every patient being changed.

The separate cutoff maximizes `Fcontrol(a) − Fcase(a)`, choosing the highest
observed support value on an exact tie. Patient membership is `Pr(A_i > a)`;
report case tail mass, control tail mass, their excess and area above a. This
uses the [one-sided KS discrepancy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html)
as a **description only**. Continuous-data KS p-values are not used. If the
maximum is zero, select nothing. Finite-sample null variation can give a positive
maximum; the cutoff does not make those patients discoveries. It can also be
unstable near ties and poorly separate overlapping biological components.

```sh
PYTHONPATH=src:research python research/assess_kernels.py \
  --protocol research/protocols/localization.json \
  --output .release/0.2.0/localization
```

`localization.tsv` evaluates all prespecified profiles, including noncalls.
`average_precision` ranks **patient contributions**; `membership_average_precision`
ranks mass above the cutoff. Soft precision/recall compare selected mass with
planted identities. Contribution error compares with the exact finite-cohort
latent attribution. Cutoff mass and tail-fold deviations from planted parameters
are interpretation diagnostics, not assertions that they estimate those targets.
Kernel descriptions retain 64/128-node convergence checks. The saved bounded
gene-level p-values are unchanged. Current results and unresolved numerical
limits are in the canonical plan; the cutoff is not ready for production.

Render the exact separated example for review with:

```sh
PYTHONPATH=research python research/plot_assessment.py --localization \
  --output .release/0.2.0/localization/review
```

This illustrates the definitions, not their performance on overlapping mixtures.
The artifact manifest records user approval as pending.

## Replaying the false-positive mechanisms

```sh
PYTHONPATH=src:research python research/replay_mechanisms.py \
  --output .release/0.2.0/mechanism-replay
```

The [protocol](protocols/mechanism-replay.json) fixes two historical cohorts per
setting. Current public RLE refits frozen reference identities, then current
unsigned W1 recomputes every permutation exceedance and BH result. This tests
the normalization mechanism, not a newly trained control selector. The separate
noise replay regenerates saved perturbations, checks observed linear/log areas,
and runs current linear-upward scores with a fixed 255-permutation budget.
It averages statistics across perturbations, never p-values. Historical full
BH results are explicitly separate from the new nominal rejection counts.

The noise panels contain selected genes, not full libraries. Original known
exposures stay fixed to reproduce the earlier recipe; no RLE is fit to the panel
and adding counts without updating totals is not endorsed. These studies diagnose
normalization bias and noise-unit artifacts. They do not certify FDR, implement
the retired bulk correction, or show that kernels repair either failure.

## Measurement model card and diagnostics

The [measurement protocol](protocols/measurement.json) investigates the existing
candidates without fitting a new prior or selecting a replacement. The target is
an underlying gene fraction p, transformed to abundance A=(N/s)p. N is the fixed
integer total in the declared count universe; s is a supplied positive exposure.
The study uses known exposures to isolate the measurement problem. This is not
a new normalization choice and does not propagate uncertainty in fitted RLE.

With C drawn from Binomial(N,p), the existing prior is Beta(h,h) and the kernel is
Beta(C+h,N−C+h), scaled by N/s. Here h=1 and h=0.5 are sensitivity controls.
The mean and variance are

\[
E[A\mid C]=\frac Ns\frac{C+h}{N+2h},\qquad
\operatorname{Var}(A\mid C)=\left(\frac Ns\right)^2
\frac{(C+h)(N-C+h)}{(N+2h)^2(N+2h+1)}.
\]

At zero, the mean and width are positive while the density's mode is zero.
Positive uncertainty does not require a positive mode. A future count from the
**same underlying fraction**, at depth M, has a
[beta-binomial predictive distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html).
That predictive count law is different from assuming extra technical dispersion
in the original measurement likelihood.

The rare-fraction Poisson limit is Gamma(C+h, rate s). This follows from the
Poisson likelihood and power prior proportional to A^(h−1); the prior is improper,
but the posterior is proper here. It is a diagnostic limit, not a selected new
kernel. Changing the likelihood name to Poisson does not remove the h/s location
shift. Its predictive count distribution is negative binomial; an NB predictive
law does not establish that the original measurement has NB overdispersion.

```sh
PYTHONPATH=src:research python research/assess_measurement.py \
  --output .release/0.2.0/measurement
```

The runner freezes sources and protocol, refuses overwrite, and requires no
historical artifacts. It writes observed-count locations/intervals, exact
binomial probability sums for fixed-truth coverage, conditional count strata,
fresh held-out count prediction, controlled technical overdispersion, prior
self-consistency and compositional bookkeeping. Monte Carlo endpoints include
standard errors; exact-sum coverage includes omitted probability. Predictive
log loss retains impossible plug-in outcomes as infinite loss. Mean log abundance
transforms the whole kernel and reports quadrature disagreement, rather than
substituting the log of the mean. This does not validate a log-W1 test.

```sh
python research/plot_measurement.py .release/0.2.0/measurement
```

The figure is descriptive and its manifest marks user approval as pending. The
[canonical plan](../docs/kernel-validation-roadmap.md#measurement-diagnosis-completed-2026-09-24)
records the completed run, interpretation and next decision.

Prior self-consistency is a computational check, not RNA-seq validation;
[simulation-based calibration](https://arxiv.org/abs/2211.02383) likewise separates
posterior computation from model suitability. Fixed positive abundance, a
structural-zero boundary, and conditioning on an observed zero are reported
separately. Central intervals and one-sided upper bounds answer different
questions. No coverage result alone certifies downstream W1 inference.

The overdispersion experiment draws a preparation fraction around a fixed source
fraction, then samples binomial counts. It evaluates the kernel against **both**
targets. The technical variation is supplied for diagnosis; it is not estimated
from biological patient variability. Finally, a complete disjoint count universe
requires gene fractions to sum to one. Equal patient probability mass is a
different invariant. Neither a prior-averaged calibration pass nor positive
zero-count width guarantees this compositional consistency.

The subsequent [mode contract](protocols/mode-contract.json) freezes the user's
requirement that each individual kernel peak at C/s while raw C and N remain
unchanged. It audits the natural Dirichlet(C+1) candidate and the existing Beta
comparators, separating the joint mode, marginal modes and inferred means.

```sh
PYTHONPATH=src:research python research/assess_measurement.py \
  --protocol research/protocols/mode-contract.json \
  --output .release/0.2.0/mode-contract
```

The candidate fails individual-mode preservation and is rejected before a
performance sweep. The [derivation and worked example](../docs/kernel-validation-roadmap.md#preserve-observations-and-individual-kernel-modes--2026-09-24)
show why another proper Dirichlet allocation cannot satisfy the general exact-mode
requirement. This is not a claim that all uncertainty models are impossible.
`measurement.py` contains the constraint-audit helpers; they are not a new public
API or selected kernel. Tests check immutable observations, zero uncertainty,
actual density peaks, the two-category exception, and log-coordinate effects.

## Alternative mode-preserving constructions

```sh
PYTHONPATH=src:research python research/derive_mode_constructions.py \
  --output .release/0.2.0/mode-constructions
```

The [protocol](protocols/mode-constructions.json) separates structural feasibility
from calibrated measurement uncertainty. The discrete construction resamples a
whole library from probabilities `(C+w)/(N+1)`, with positive weights summing to
one. Every marginal mode equals C, but the entire multinomial family satisfying
that condition assigns less than one expected read to all zero-count genes.
These are bootstrap kernels, not latent-fraction posteriors.

The continuous witness draws `Q ~ Dirichlet(C+alpha)` and one shared
`U ~ Uniform(0,1)`, then sets `P=(1-U)*(C/N)+U*Q`. Each marginal peaks at C/N;
zeros have positive width, and every vector sums to one. The extra contraction
changes the inference rule: it yields only about 79% high-information coverage
for central 95% intervals. It is not a selected posterior or a WADE backend.

The bounded CDF integrand uses adaptive quadrature, with a separately recorded
error estimate and tighter-tolerance replay. Direct probability calculations
avoid mistaking underflow of tiny Beta quantiles for exact zero uncertainty.
The [canonical derivation](../docs/kernel-validation-roadmap.md#alternative-constructions--2026-09-24)
records the worked example, numerical amendment, completed artifact and limits.
This audit uses known exposures and has no gene tests, normalization fitting,
technical dispersion or new public options.

## Verification and provenance

The historical `archive/` and execution `.release/` directories are local evidence,
not versioned dependencies. Tests and the numerical fixture run without them.

From the repository root, using an environment with the test dependencies:

```sh
PYTHONPATH=src:research python -m pytest -q research/tests
PYTHONPATH=src:research python -m pytest -q
```

The tests replay six pre-cleanup kernel/point cases exactly, compare areas with
independent CDF quadrature, check normalization of the whole kernel, nonzero
zero-count uncertainty, equal mass, cache isolation and unequal-group point
areas. Fixture provenance is in `tests/fixtures/kernel-baseline.json`.
These are correctness checks, not FDR calibration or model selection.

The extracted code originates in the [archived adapter](../archive/development-20260923/tools/kernel_benchmark_adapter.py),
[prior contrast](../archive/development-20260923/tools/reference_kernel.py),
[cache](../archive/development-20260923/tools/kernel_cache.py), and
[point control](../archive/development-20260923/tools/kernel_scientific_pilot.py).
Other kernel families, fitted-reference prototypes, simulators and completed
study runners remain in that archive. Reuse their ideas deliberately; do not
import archived modules into this workspace.

New studies should have one small runner and a frozen configuration, with
execution artifacts under `.release/`. Shared calculations belong here, not
inside successive experiment scripts. Add evidence to the existing plan/archive
index instead of creating another parallel roadmap. No further caching or Rust
work is needed before the scientific benefit is established.
