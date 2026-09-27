# WADE documentation

**Disk cleanup — 2026-09-27:** the local `.release/` directory was deleted at
the user's request, recovering approximately 135 GiB. References to `.release/`
artifacts below are historical; those files, including the review ZIP and local
benchmark dependencies, must be regenerated or reinstalled before reuse. Source,
protocols and written experiment summaries remain in the project.

The checkout contains a working **point-based package baseline** and a separate
research workspace. **The user narrowed 0.2.0 to a point-based release on
2026-09-25.** Measurement kernels are deferred. [PLAN.md](../PLAN.md) is the
active release and scientific plan; the kernel roadmap retains prior evidence.

The standalone [review for independent critique](independent-review.md)
documents the implemented methods, release barriers, unresolved theory and
comparative evidence. It includes a new independent audit of the 0/80 subset
result and a comparison with the inspected original HITLIB source.

**Release assessment — 2026-09-26.** WADE has a usable, tested implementation
and encouraging evidence for large-cohort distribution analysis. It is not yet
ready for the revised **point-based 0.2.0**. The replicated 120-transcriptome
comparison is complete, but its primary endpoint rewards **any distribution
change**, not necessarily the intended biomarker signal. The user explicitly
rejects unsigned-only discovery as the release objective. The recommendation
to retain it is withdrawn. Directional enrichment, meaningful effect recovery
and separation from normal expression must drive the next evaluation. The
existing directional prototype is not yet validated for that objective.

| Component | Actual status |
| --- | --- |
| Public package | Working RLE/internal-control normalization, explicit filtering, unsigned empirical log-W1, label permutations, one p-value/q-value per gene, Rust and NumPy implementations, exports and plots. |
| Direction/scale comparison | Four point procedures completed 120 full matrices alongside native DESeq2/edgeR. Aggregate distribution recovery cannot select a biomarker method. Direction-specific failures need diagnosis. Public unsigned log-W1 is a historical comparator, not the agreed release solution. |
| Abundance uncertainty | Deferred research. No kernel has demonstrated the required robustness benefit and passed selection; imminent competitiveness is not supported by current evidence. |
| Effect/subset descriptions | Baseline bulk/tail summaries work. Research attribution is descriptive; a reliable patient-enrichment cutoff or affected-fraction estimator is not established. |
| Examples and notebooks | All six were executed and visually inspected for the point baseline, including TCGA prostate and breast gene counts. A changed final method needs rebuilt results and approval. |
| Software verification | **175 tests passed**. The replicated comparison passed an independent audit of 3,271 files, exact native input identity, stopping/BH, raw call counts and quantile integration; all 120 public-baseline replays matched exactly. Final candidate, wheel/fallback, cross-platform and release checks remain pending. |

The [status audit](../.release/0.2.0/release-status-20260925/audit.json) rechecks
the saved comparison tables and confirms that all ten public inference modules
and the native binary match the benchmark snapshots byte for byte. Thus those
baseline results still apply to the current
algorithm; they do not validate the research kernels. The changed number of
tests since the archived migration reflects the workspace/test consolidation;
different suites must not be counted as independent scientific validation.

**What “point-based” means.** Each sample contributes one value, such as its
count divided by its RLE size factor. The empirical distribution puts one equal
patient mass at that value. A kernel would spread the same mass over possible
abundances. The original unsigned method was already point-based. Direction
and linear/log scale are independent choices, not consequences of removing
kernels. Label permutations still supply a null distribution for significance.

With controls `[10, 10, 10, 10]` and cases `[2, 2, 2, 30]`, upward area is 5,
downward area is 6, unsigned W1 is 11, and signed area is −1. Keeping the two
directional areas preserves the high-expression subset despite a lower group
mean. One implementation can calculate both directions using the same sorted
data and label shuffles; users need not fit normalization or prepare data twice.
The candidate corrects both directions together. These areas are not fold
changes, and an upward signal does not imply all patients are upregulated.

**Corrected biological objective.** An upward discovery should identify
reproducible expression excess in cases versus normal controls, broadly or
within a subset, even when other cases have lower expression. Reporting only
that the distributions differ does not answer that question. Sensitivity/FDP
remain necessary, but their truth labels must match the requested direction
and their interpretation must include magnitude, absolute abundance, normal
background and patient enrichment.

The saved benchmark exposes the mismatch. In the balanced depressed-bulk/
fourfold-subset pattern, joint directional log has 56/80 downward calls and
**0/80 upward calls**. Its former 70% any-direction recovery is zero upward
recovery. Directional linear likewise has 0/80 upward calls for that pattern,
while recovering 59/80 upward profiles for the 10% sixteenfold-subset pattern
(log: 15/80). These findings justify correcting the objective and diagnosing
the prototype; they do not justify replacing directional discovery with an
unsigned result or declaring the directional prototype successful.

The strongest upward factor in this study was 16-fold (4 log2), and its
depressed-bulk subset was fourfold (2 log2). It did **not** cover the motivating
5–10 log2 extreme effects. The follow-up must include that range alongside
weaker effects and realistic negatives, rather than silently treating these
simulations as ETV1/ETV4 equivalents. A spread change is neither automatically
a biomarker success nor automatically a false positive: reproducible,
useful directional separation from controls is what matters.

The [corrected roadmap](../PLAN.md#0-restore-the-biological-target-before-another-comparison)
puts a direction-specific scorecard and targeted diagnosis before another broad
comparison. Positive-part area remains the simplest starting prototype; removing
the downward contribution does not by itself prove biological usefulness or
solve its significance/power limitations. Kernels remain deferred.

An [exploratory recount](../.release/0.2.0/biomarker-objective-review-20260926/audit.json)
verified the 40 saved input/result files for the twenty balanced mixed matrices.
Re-adjusting the existing upward p-values over G genes, rather than the original
joint 2G family, still gives **0/80** upward discoveries for the depressed-bulk/
fourfold-subset pattern on either scale. Multiplicity alone does not explain
that failure. The [counts](../.release/0.2.0/biomarker-objective-review-20260926/recount.tsv)
and [replay script](../.release/0.2.0/biomarker-objective-review-20260926/recount.py)
are retained separately; this post-hoc family change is a diagnostic, not fresh
confirmation, a new default, or evidence of FDR control.

The subsequent [subset-failure audit](../.release/0.2.0/subset-failure-audit-20260926/audit.json)
independently reproduces all 80 profiles' areas (maximum discrepancy 1.71e-13)
and exactly regenerates two full matrices. Fresh fixed-budget permutations on
selected high-count profiles retain weak upward evidence with true exposures
and even with count noise removed. In contrast, a deterministic fourfold subset
above a depressed bulk has exact linear upward p≈1.13e-6. The test can detect
that structure; the completed simulations have substantial biological overlap
and weaker effects than the motivating ETV1/ETV4 examples. This is a diagnostic,
not a new power study or proof that all implementation issues are excluded.
The [review](independent-review.md#4-what-080-means-and-what-the-new-audit-establishes)
includes the exact inputs, selection of diagnostic profiles, results and limits.

## Replicated direction/scale comparison — 2026-09-26

The [frozen protocol](../research/protocols/point-comparison.json) completed
**120 independent 16,384-gene matrices**: 20 complete-null and 20 mixed matrices
at each of 250:250, 50:500 and 500:50 cases:controls. Each cell contains ten
independent Gamma–Poisson and ten correlated lognormal–Poisson matrices. The
mixed matrices contain 48 altered genes: twelve profiles at four abundance
levels. This is deliberately a **sparse-alternative** experiment; its discovery
thresholds and average sensitivity need not resemble a tumour/normal analysis
with thousands of changed genes. No systematic group depth or batch imbalance
was injected. All alternatives survived filtering.

All four point procedures fit RLE on the full raw matrix before the pooled
filter (at least three counts in at least two samples), retaining reference/test
overlap. They share label draws, exact ties, no added noise, and a 999,999-shuffle
cap. Each unsigned procedure corrects G tests; each paired directional procedure
corrects 2G tests jointly. Their q-values are **never combined by choosing the
best scale or direction**. Native R DESeq2 1.52.0 and edgeR 4.10.5 received the
same integer matrices. DESeq2 independently reproduced the RLE factors and
retained Cook handling and independent filtering. edgeR used robust modern QL
fitting with TMM, plus a matched-RLE sensitivity analysis. These are native
fits, not Python approximations.

At q≤.05, the entries below are **sensitivity / mean realized FDP**, in percent,
averaged across twenty mixed matrices. The primary truth is any planted
distribution change; mean-DE truth is scored separately. A variance-only WADE
discovery is a true distribution discovery, even though it is not mean DE.
Directional rows count unique genes called in either direction.

| Procedure | 250 cases : 250 controls | 50 cases : 500 controls | 500 cases : 50 controls |
| --- | ---: | ---: | ---: |
| Unsigned log, public baseline | 30.4 / 5.1 | 10.5 / 5.9 | 5.4 / 8.7 |
| Unsigned linear | 37.0 / 5.1 | 11.3 / 5.3 | 1.4 / 0.0† |
| Directional log, joint up/down | 27.4 / 6.9 | 8.8 / 5.6 | 3.8 / 2.1 |
| Directional linear, joint up/down | 25.3 / 4.7 | 8.0 / 5.4 | 1.8 / 3.5 |
| Native DESeq2 | 34.0 / 17.2 | 17.7 / 20.8 | 11.8 / 17.6 |
| Native edgeR QL, TMM | 34.2 / 15.8 | 20.4 / 21.1 | 9.5 / 24.6 |
| Native edgeR QL, matched RLE | 34.2 / 14.4 | 20.6 / 21.2 | 10.0 / 25.0 |

† Zero observed false discoveries is not a zero-FDR guarantee. This procedure
recovered only **13/960** planted profiles in the control-limited design.
The conservative upper interval for its mean FDP remains 35.5% with twenty
matrices. Across all rows, specificity exceeds 99.97%, yet FDP differs greatly:
there are 16,336 unchanged genes per mixed matrix. For example, the balanced
unsigned linear procedure has 355 true and 20 false calls across twenty
matrices, versus 292 and 17 for the baseline. Mean FDP averages each matrix's
FP/max(calls,1); it is not the ratio of pooled false to pooled total calls.

**The native-method results must be read by biological model.** In the balanced
Gamma–Poisson matrices, DESeq2 and edgeR-TMM have mean FDP **5.7% and 5.1%**;
in the correlated lognormal–Poisson matrices these are **28.6% and 26.4%**.
Unsigned linear is 5.2% and 5.0%, respectively. Changing this generator changes
both the abundance distribution and gene dependence, so the experiment does
not isolate which assumption causes the difference. The pooled table is not
evidence that DESeq2 or edgeR generally have inflated FDR in bulk RNA-seq.
Matched-RLE edgeR retains the model-dependent problem, so the observed native
gap cannot be attributed solely to TMM versus RLE. Established data and a
separate shape/dependence experiment are needed before a broad comparative claim.

**Aggregate distribution recovery does not select the biomarker method.**
At 250:250, unsigned linear improves sensitivity over the current baseline by
6.6 percentage points (paired matrix-bootstrap 95% interval 4.4–8.6), with
similar estimated FDP. Joint directional linear instead loses 5.1 points
(3.4–6.8). At 500:50, unsigned linear loses 4.1 points to the log baseline
(2.7–5.5). These are development comparisons, not a fresh confirmation after
selection. Testing two directions introduces a larger hypothesis family and
changes the statistic; this study evaluates those complete procedures, not an
unadjusted directional statistic against a corrected unsigned one.
The numerical comparisons remain valid for their original broad target. The
earlier inference that they favor an unsigned release procedure is withdrawn.
The scorer duplicated `altered` across directions for an equality-null endpoint;
it did not measure whether the biologically intended direction was recovered.
Gene-direction and unique-gene mean FDP agree in this run; opposite-direction
calls on the planted ordered bulk profiles were zero. Both directional tests
still use distribution equality as the permutation null. This is not a general
directional-error guarantee for crossing distributions.

**Pattern recovery explains why a single sensitivity number is insufficient.**
At 250:250 each row below has 80 planted profiles, including all four abundance
levels and all twenty matrices. Numbers are recovery percentages.

| Planted pattern | Unsigned log | Unsigned linear | DESeq2 | edgeR-TMM |
| --- | ---: | ---: | ---: | ---: |
| 10% of cases up 16-fold | 21.2 | 81.2 | 98.8 | 100.0 |
| Bulk halved, 10% up fourfold versus baseline | 72.5 | 55.0 | 3.8 | 3.8 |
| Same population mean, wider distribution | 66.2 | 38.8 | 1.2 | 1.2 |
| Mean-preserving 10% upward subset | 27.5 | 16.2 | 1.2 | 0.0 |
| One case up 16-fold | 0.0 | 0.0 | 0.0 | 0.0 |

Linear area emphasizes large abundance differences; log area helps reveal bulk
and shape changes that can accompany an upward subset. Native mean tests are
stronger on several subsets that also move the group mean. Neither scale nor
method dominates. Critically, the directional log test's 56/80 discoveries of
the depressed-bulk pattern are **all downward calls, with zero upward calls**.
The full-transcriptome experiment does not establish significant upward
enrichment for that pattern. An unsigned discovery followed by a descriptive
upward tail does not supply the directional evidence the user requires.

**Effect recovery was evaluated on all altered genes, including noncalls.**
All four point tests use the same auxiliary empirical RLE group-mean estimator;
changing the statistic does not change that estimate. Mean per-matrix log2FC
RMSE against the generating population mean ratio is:

| Effect estimator | 250:250 | 50:500 | 500:50 |
| --- | ---: | ---: | ---: |
| Empirical RLE group means | 0.153 | 0.283 | 0.249 |
| DESeq2, native unshrunk coefficient | 0.144 | 0.275 | 0.225 |
| edgeR-TMM | 0.144 | 0.260 | 0.221 |
| edgeR-RLE | 0.144 | 0.260 | 0.221 |

Errors against the realized latent patients are also saved: for balanced
cohorts the corresponding RMSEs are 0.098, 0.095, 0.089 and 0.089. There are no
missing altered-gene estimates. This supports useful effect estimation but no
WADE accuracy advantage. Matching latent linear/log areas and group-curve errors
are recorded separately; area is not fold change. Planted patient identities
are retained, but this comparison does not validate a subset-prevalence
estimator or a patient-enrichment cutoff.

**Control counts limit rare-tail evidence.** An exact two-level illustration
helps explain the 500:50 failures. Suppose 50 of 500 cases have the same high
value, everyone else has the same low value, and none of 50 controls is high.
Under label permutation, the upward-area p-value is
`choose(500,50) / choose(550,50) = 0.006715`; unsigned p is 0.016269. Raising the
high value further scales both observed and null areas without changing these
p-values. With 25 high cases among 250 cases and 250 controls, upward p is
1.59e-8. These are exact toy probabilities, **not universal bounds for continuous
gene profiles**. More cases cannot always replace information from more normal
patients. A cohort distribution test also does not estimate how rare a gene's
extreme expression is across the transcriptome.

Complete-null results remain uncertain: at 250:250, any discoveries occurred
in 3/20 baseline matrices, 2/20 unsigned-linear matrices, and 1/20 matrices for
each directional procedure. The prespecified intervals are wide; including 5%
does not certify calibration. Each biological stratum has only ten matrices.
Use the observed matrix-level variation to plan fresh confirmation with a
prespecified acceptable error tolerance, rather than claiming success from a
nonsignificant calibration check.

**Reproducibility and review.** The [runner](../research/compare_points.py),
[native adapter](../research/point_comparison_native.R), and
[independent review](../research/review_points.py) retain counts, latent truth,
normalization, eligibility, native exclusions, permutation stopping records,
versions and hashes. Every dataset matched the public baseline exactly. The
independent audit verified 3,271 files, raw native input identity, BH/stopping,
primary detection numerators, and separately integrated quantile areas
(largest numerical difference <8e-13). No production inference code changed.

- [Complete inference audit](../.release/0.2.0/point-comparison-20260925/audit.json)
  and [independent audit](../.release/0.2.0/point-comparison-20260925-review-v3/audit.json).
- Detection and FDP: [pooled](../.release/0.2.0/point-comparison-20260925-review-v3/detection-fdr.png),
  [Gamma–Poisson](../.release/0.2.0/point-comparison-20260925-review-v3/detection-gamma.png),
  [correlated lognormal–Poisson](../.release/0.2.0/point-comparison-20260925-review-v3/detection-correlated_lognormal.png).
- [Pattern recovery](../.release/0.2.0/point-comparison-20260925-review-v3/pattern-recovery.png),
  [effect recovery](../.release/0.2.0/point-comparison-20260925-review-v3/mean-effect-recovery.png),
  and [power–FDP curves](../.release/0.2.0/point-comparison-20260925-review-v3/power-fdp-curves.png).
- [Matrix-level intervals](../.release/0.2.0/point-comparison-20260925-review-v3/detection-summary.tsv),
  [paired differences](../.release/0.2.0/point-comparison-20260925-review-v3/paired-differences.tsv),
  [complete-null intervals](../.release/0.2.0/point-comparison-20260925-review-v3/complete-null-summary.tsv),
  and [effect intervals](../.release/0.2.0/point-comparison-20260925-review-v3/effect-summary.tsv).

Earlier renderings are retained. Review v3 includes model-stratified figures
and corrects the displayed matrix counts; all numerical tables and inference
are unchanged. PNG/PDF pairs are available for every figure. All results
and figures remain **pending user approval**. These are controlled development
simulations, not established-data replication, final release certification, or
clinical sensitivity/specificity.

The next bounded step is the directional biomarker diagnosis in PLAN.md,
followed by a frozen follow-up that represents the motivating effect sizes.
Established-input replay and fresh confirmation remain required. Unsigned
procedures stay as historical controls, not fallback release choices. Do not
select a scale per gene by the smallest q-value or promote the directional
prototype before it meets the corrected objective.

## Earlier evidence and remaining release gates

The [first two-matrix engineering run](../.release/0.2.0/point-workflow-20260925/audit.json)
established execution but is superseded for comparative assessment by the
replicated study above. The older pilots below have different targets and
denominators; their percentages are not directly comparable with transcriptome
BH sensitivity.

**Measured sensitivity and specificity.** The matched pilot has twenty
independent matrices per cohort design, each with a complete 20,000-gene
normalization input and a fixed 36-gene evaluation panel. Each design supplies
480 upward-pattern profiles and 160 unchanged-distribution profiles. These are
**nominal single-gene p≤.05** results, not transcriptome BH or clinical diagnostic
performance. Specificity here is the fraction of these null profiles not called;
genes within a matrix are dependent.

| Method | 250 cases : 250 controls, sensitivity / specificity | 50 cases : 500 controls, sensitivity / specificity |
| --- | ---: | ---: |
| Current public unsigned log-W1 baseline† | 81.2% / 95.6% | 69.4% / 93.1% |
| Directional linear points, research | 86.9% / 96.2% | 78.1% / 95.0% |
| Earlier Beta(C+1,N−C+1) kernels, research | 87.7% / 95.6% | 77.1% / 94.4% |
| Native DESeq2 upward test | 77.1% / 94.4% | 71.5% / 92.5% |

† The unsigned baseline detects *any* distribution change in the planted upward
profiles; its significance does not separately establish upward enrichment.
The other rows target upward behavior, with different hypotheses for mean DE
and directional area. [Recounted numerators and denominators](../.release/0.2.0/release-status-20260925/pilot-operating-points.tsv)
and the [full pilot assessment](../archive/development-20260923/docs/kernel-scientific-pilot.md)
retain all methods and uncertainty estimates.

That pilot suggested directional area captures a useful target; it did not
establish transcriptome-scale discovery power.
In the balanced design, kernels recover 54/80 profiles with an upward subset
above a depressed bulk, directional points 56/80, and DESeq2's upward test 0/80.
The generating overall mean decreases in this pattern, so it is not an upward
mean-DE alternative. Across the other five upward patterns, the kernel and
DESeq2 recover 367/400 and 370/400, respectively. There is no established general
kernel advantage or sample-size crossover. Kernel-minus-point sensitivity
differences are small and uncertain in this pilot.

On conventional compcodeR mean-DE simulations with 50 samples per group,
the current baseline recovers **75.2%**, native DESeq2 **77.0%**, and edgeR
**77.3%** of requested DE genes. Mean FDPs are 4.9%, 5.8% and 5.8%, respectively.
This is encouraging, with only three matrices at that size. At three or five
samples per group, this WADE benchmark makes no BH discoveries; that is outside
the intended large-cohort advantage. The
[reference summary](../.release/0.2.0/release-status-20260925/conventional-de.tsv)
keeps the sample-size series and its limited replication explicit.

**False-discovery behavior is conditional on the workflow.** In the
16,384-gene mild-composition confirmation, all-gene RLE plus the current test
has 57.8% sensitivity across nine patterns at 500:500 and mean FDP 5.27%
(replicate-level 95% interval 4.06–6.48%). At 500:50 these become 23.1% and
3.52% (1.29–5.76%). The 10% recurrent upper-tail pattern is recovered in
84.4% of balanced profiles but 0/160 limited-control profiles. These results
are specific to the frozen effects and abundance range, not universal power
limits. [Full-transcriptome evidence](../archive/development-20260923/docs/transcriptome-workflow-validation.md).

In the separate high-alteration 500:500 development screen, evaluated against
the generating biological-abundance truth, current WADE has
75.8% sensitivity and **20.0% mean FDP**; native DESeq2 has 60.6% and 23.8%.
Using known true exposures lowers WADE's FDP to 3.46% and sensitivity to 71.3%
on those same matrices. Known exposures are an oracle, not an available repair.
The evidence diagnoses substantial normalization/reference bias; kernel spread
does not automatically correct it. Specificity and FDP have different
denominators, so high specificity alone does not certify discovery reliability.
[Comparison refresh](../archive/development-20260923/docs/production-migration-0.2.0.md).

The earlier kernels also fail the intended depth-robustness comparison. With
5% of libraries thinned 1,000-fold in the balanced cohort, directional points
achieve 81.2% sensitivity / 96.2% specificity; the earlier h=1 kernel achieves
36.2% / 95.2%. This deliberately severe boundary test does not characterize
ordinary bulk data, but it supplies no robustness justification for promoting
that kernel. The [depth study](../archive/development-20260923/docs/kernel-depth-stress.md)
and [current measurement derivations](kernel-validation-roadmap.md#measurement-diagnosis-completed-2026-09-24)
explain the depth-dependent location effects. Later mode-preserving constructions
have measurement diagnostics only; they have no new cohort sensitivity/FDR claim.

**OUTRIDER comparisons are pilots with a different inferential unit.** Native
OUTRIDER detects aberrant expression within individual samples, as described
in its [paper](https://pubmed.ncbi.nlm.nih.gov/30503520/); WADE currently tests
cohort distributions. We have run OUTRIDER-AE, OUTRIDER-PCA, saseR and OutSingle
on the original Kremer/GTEx-skin backgrounds with custom injections. In the
Kremer singleton-upward cell, WADE recovers 0/50 genes while a case-minimum-p
aggregation of OUTRIDER-AE recovers 48/50. At approximately 20% affected cases,
the corresponding numbers are 44/50 and 0/50; OUTRIDER-PCA recovers 27/50.
These demonstrate different strengths, not a fair universal ranking. The
aggregation is an additional gene-level procedure, there is one injected
matrix per background, and unmodified real genes/cells are not known negatives.
Consequently these data do not yield a complete specificity or FDR certification.
[Exact comparison notebook](../notebooks/reference-benchmarks.qmd).

**Biological examples already show progress.** The current prostate baseline
calls ETV1 and ETV4 as distribution differences (q≈0.00251 and 0.000405), and
ERG and SPINK1 are also called. The breast baseline includes ERBB2 (q≈3e-6)
and the four subtype contrasts. Negative bulk and positive tail descriptions
can coexist. These q-values do not establish separately significant upward
tails, fusion membership or individual rarity. Very broad tumour/normal call
sets—74.1% of tested prostate genes and 95.2% of breast genes—require biological
interpretation; real per-gene truth is unavailable. Count-processing and
annotation/row-order provenance still need completion. [TCGA specification](tcga-examples.md).

**What must finish before the agreed release:**

1. Select and freeze a point procedure for large-cohort bulk data: statistic,
   direction/scale, normalization, filtering, inferential family and reported
   effects. It must answer the directional biomarker question; neither unsigned
   discovery nor the unvalidated directional prototype has met that release gate.
2. Complete established-input replay and confirm the selected procedure on fresh
   datasets/seeds, using sensitivity, specificity/FDR and effect recovery together.
   The controlled 120-matrix development comparison is complete. A general
   superiority claim or a solved measurement-uncertainty model is unnecessary.
3. Migrate the selected procedure into the simple public API and rebuild the
   six notebooks. Complete prostate/breast provenance and review marker,
   fusion/subtype and broad-call interpretations against the final method.
4. Approve the exact final release results and illustrations, then pass final
   installed-package/fallback and cross-platform CI checks and authorize release.
   Earlier technical QA and research figures do not approve changed results.

The practical priority is the finite point-method validation sequence in
PLAN.md. Kernel derivation, sparse-assay certification, automatic control
discovery and transcriptome rarity remain later work. The user authorized the
release-scope change, not publication or approval of new results and figures.

## Current documentation

| Page | Purpose |
| --- | --- |
| [Release and validation plan](../PLAN.md) | The active point-method candidate, publication benchmark and release gates. |
| [Deferred kernel roadmap](kernel-validation-roadmap.md) | Measurement derivations, completed diagnostics, negative results and unresolved gates. |
| [Manual](manual.md) | Using the current package, inputs, filtering, design and interpretation. |
| [Method](method.md) | Exact definitions of the current point-based test and inference. |
| [Normalization](normalization-standard-practice.md) | RLE, designated controls, supplied factors and per-sample conversions. |
| [TCGA examples](tcga-examples.md) | Gene-count inputs, cohort definitions and annotation provenance. |
| [Notebook builds](notebook-builds.md) | Reproduce and review the six notebooks against the current API. |
| [Change notes](changes-0.2.0.md) | Implemented public baseline; final procedure selection is pending and kernels deferred. |

[PLAN.md](../PLAN.md) is the release queue; [ISSUES.md](../ISSUES.md) lists open
issues without the experiment chronology. The [contributor guide](../CONTRIBUTING.md)
explains the code layout and verification commands.

## Research and evidence

`archive/` and `.release/` are optional local evidence, excluded from Git and
release distributions. Links into them are available only in the original
research workspace. The current specification, code and numerical fixtures
are versioned and work without either directory.

Candidate point validation and retained kernel work live in [research/](../research/README.md). Completed studies
and their competing prototypes are in the [development archive](../archive/development-20260923/README.md).
Their reports remain evidence, including negative results; they are not active
implementation instructions.

The latest point comparison is recorded above with its versioned protocol and
local artifacts. Earlier kernel comparisons are [the scientific pilot](../archive/development-20260923/docs/kernel-scientific-pilot.md),
[paired downsampling](../archive/development-20260923/docs/kernel-downsampling.md), and
[minority-depth stress](../archive/development-20260923/docs/kernel-depth-stress.md).
All existing result/figure approvals retain their original pending status.
