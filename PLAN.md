# WADE 0.2.0 release plan

**2026-09-25 — scope changed by agreement with the user.** Release a validated
point-based method for large bulk RNA-seq cohorts. Defer measurement kernels
until after this release; their unresolved derivation is no longer a blocker.
This file governs the release. The [kernel roadmap](docs/kernel-validation-roadmap.md)
retains the derivations and negative results as deferred research.

**2026-09-26 — objective correction from the user.** WADE must identify
biologically useful **directional expression enrichment**, across a cohort or
within a subset. An unsigned finding of any distribution change is not an
acceptable primary release result. The recommendation to retain unsigned
testing on the strength of aggregate recovery is withdrawn: that benchmark
endpoint rewards changes that need not answer the biomarker question.

The public package still implements unsigned log-W1 and remains a historical
comparison implementation, not the selected release procedure. The existing
directional prototype has real unresolved power limitations; a change in the
objective does not make it validated. “Point-based” describes the representation, not the direction:
each patient contributes one normalized expression value and equal patient mass.
No kernel, jitter, or fitted correction is required to calculate the area.
Permutation inference still accounts for random differences between groups
under an exchangeable design; it does not model each count's measurement error.

## 0. Restore the biological target before another comparison

The primary question is: **does this gene provide reproducible upward expression
excess in cases versus the normal background, broadly or in a subset, even if
the rest of the cases have lower expression?** Downward discovery is the
symmetric task when requested. Report magnitude, absolute abundance, normal
background and contributing patients alongside significance. A smaller p-value
for an irrelevant distribution difference is not a better biomarker result.

The simplest starting statistic remains
`A_up = integral max(Q_case(u) - Q_control(u), 0) du`, with the signs reversed
for downward analysis. It needs no fixed tail percentile. Lower case quantiles
neither add to nor cancel the observed upward area. Its current permutation
calibration uses equality of distributions; stronger claims about directional
absence or minimum useful effects need explicit validation. Positive area alone
does not guarantee an informative or technically measurable biomarker.
The ordinary [unsigned W1 definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html)
remains a distribution distance; its mathematical validity does not determine
whether it meets this biological objective.

Next steps, in order:

1. Freeze a biomarker scorecard. Recovery requires the **requested direction**,
   useful effect recovery and separation from the control background. A downward
   call for a planted upward subset is not an upward success. Keep distribution
   null FDP, wrong-direction calls and biologically unhelpful nominations distinct.
2. Reinterpret the saved direction-specific results before generating more data.
   Do not relabel old q-values, merge the best scale/direction, or call this fresh
   confirmation. Diagnose why positive-area inference misses the desired subset;
   distinguish the statistic, permutation null and multiplicity contributions.
3. Freeze a small targeted follow-up covering bulk up/down, upward subsets above
   unchanged or depressed bulk, shared normal/case tails and low-count artifacts.
   Retain weak effects and include **5–10 log2 upward changes (32–1,024-fold)**,
   the range motivating ETV1/ETV4. The completed study's strongest upward factor
   was 16-fold; its depressed-bulk subset was only fourfold. These are not the
   original extreme-effect scenario. Include spread changes, but judge their
   utility from reproducible directional separation, not their pattern label.
4. Measure directional area/effect recovery and patient enrichment/ranking,
   as well as gene-level power and error. Evaluate patient thresholds/rankings
   on held-out patients or independent cohorts. An automatically chosen cutoff
   is not a validated affected-fraction estimate; if it enters the test statistic,
   repeat its selection within each permutation.
5. Once the target and prototype are coherent, replay established benchmarks and
   perform fresh transcriptome-scale confirmation. Compare mean-DE tasks to native
   DESeq2/edgeR and the added subset task separately. One method's downward call
   must not receive upward-biomarker credit. No universal superiority is required.

This changes the acceptance criterion, not the saved experiment. Kernels remain
deferred, and a new broad sweep should not precede this targeted diagnosis.
The initial saved-result audit is in the
[assessment](docs/index.md): even an exploratory up-only BH adjustment leaves
the depressed-bulk/fourfold-subset recovery at 0/80 on both scales. The problem
cannot be dismissed as just the joint up/down multiplicity correction.

**Targeted audit completed 2026-09-26:** the
[independent-critique review](docs/independent-review.md) now documents all 80
profiles, independently recomputed areas, exact regeneration of two matrices,
fresh fixed-budget permutations, true-exposure/latent-abundance checks and an
exact clean subset control. No area/indexing discrepancy was found in those
checks. High-count failures persist with true exposures and without count
noise; a clean fourfold subset is detectable by the same linear statistic.
This narrows the next work to biologically matched effects/overlap and the
statistic/calibration target. It does not validate the prototype or resolve all
normalization issues. Reconcile successful historical HITLIB analyses on matched
inputs where available before claiming that the new benchmark contradicts them.

## 1. Freeze the candidate and verify the instruments

- [x] Select the first candidate for evaluation: exact upward and downward areas
  of **linear RLE-normalized counts**, evaluated together. Keep observed zeros
  and exact ties. Use the existing normalization and pooled filtering functions.
- [x] Implement a readable research evaluator with shared sorting/label draws,
  the existing audited sequential stopping rule, and BH over all tested
  gene-direction pairs. Production behavior remains unchanged during evaluation.
- [x] Check independent quantile integration, exact small permutation orbits,
  unequal group sizes, crossing curves, ties, direction reversal, stopping
  records, restricted permutations, filter/reference reuse and the joint family.
- [x] Complete the first frozen raw-count run on a complete null and mixed
  16,384-gene matrix. Audit raw detection counts, effect errors, exclusions,
  normalization error and permutation resolution. Two matrices check machinery;
  they cannot certify FDR or comparative performance.

The first execution completed both matrices: no calls in the complete null;
6/16 altered genes plus one false call for directional points in the mixed
matrix, versus 5/16 plus one false call for unsigned log-W1. Realized FDPs are
14.3% and 16.7%; neither one-matrix result estimates FDR. See the
[status assessment](docs/index.md) and
[run audit](.release/0.2.0/point-workflow-20260925/audit.json).
The first run passed 161 package/research tests and an independent check of 31
saved artifact hashes, stopping records, joint BH and test-mask bookkeeping.
The expanded comparison below brings the current suite to **175 passing tests**.

The executable [first-run protocol](research/protocols/point-release.json) and
[runner](research/assess_points.py) require no private or archived data. They
save raw counts, latent abundances, truth/membership, normalization, inference
records, source snapshots and hashes. They refuse to overwrite a prior run.

Upward area integrates positive case-minus-control quantile differences;
downward area integrates the negative differences in magnitude. Their sum is
unsigned W1, and their difference is the case-minus-control mean. A gene can
have both areas, including when its bulk decreases but a subset increases.
The statistic is an area in expression units, **not a fold change**.

The candidate uses exchangeability/equality of distributions as its permutation
null. An upward rejection is evidence in the upward-area statistic; it does
not assert that every patient is upregulated, that all quantiles increase, or
that an individual patient is a significant outlier. Composite stochastic-order
nulls and formal directional-error guarantees are not established by this test.

## 2. Build the publication benchmark around explicit questions

Reuse the existing benchmark inputs and native results where appropriate;
separate development/replay from fresh confirmation. Do not select a method,
filter or dataset because it makes WADE win. Freeze the expanded protocol,
replicate counts, acceptable margins and fresh seeds before confirmation.

| Question | Inputs and comparison | Required assessment |
| --- | --- | --- |
| Does the full workflow control false discoveries? | 16,384–20,000 genes; complete nulls and mixtures with mostly stable genes; independent and correlated biology; realistic independent depth variation | Per-dataset TP/FP/TN/FN, sensitivity, specificity, FDP; complete-null probability of any discovery; replicate-level intervals |
| What do direction and scale contribute? | Paired linear/log directional analyses; unsigned procedures retained as historical controls on the same inputs | Score requested-direction discovery and effect recovery; unsigned or opposite-direction calls are not upward-biomarker successes; never take the best q-value across analyses |
| How does power change with cohort size and subset structure? | Conventional mean-DE inputs plus matched bulk, 1/5/10/20% subsets, opposing shifts, shared tails and variance-only changes; 20/50/100/250/500 per group, 50:500 and 500:50 | Power by pattern, abundance and sample allocation; small cohorts provide context, not a WADE superiority requirement; singletons are a boundary |
| Are effects and enriched samples described honestly? | Saved latent abundances, planted memberships and matched controls | Up/down area error, mean-ratio error, group-curve recovery and patient-ranking accuracy; area concentration and walk cutoff are descriptions, not inferred prevalence |
| Does normalization/filtering change the answer? | All-gene RLE and prespecified internal controls; reference/test overlap and disjoint references; pooled count/detection filters; supplied true exposures as an oracle diagnostic | Both eligible-only and intention-to-test denominators; factor bias and false calls; fit once and retain factors across test masks |
| Where does the method fail? | Composition shifts, very shallow minorities, group-associated depth and batch imbalance, small control reserves | Label these as boundary studies, retain negative results, and distinguish limited power from invalid exchangeability/reference assumptions |
| How does it compare with established methods? | Native DESeq2 and edgeR on identical matrices; compcodeR and established real references, with large-cohort controlled extensions | Same target and truth where possible; nominal and achieved FDR alongside power, effect recovery, runtime and memory; retain native defaults and exclusions |
| What does it add to individual-outlier methods? | Existing Kremer/GTEx backgrounds and OUTRIDER/related methods, with independently replicated injections | Separate patient-gene and cohort-gene tasks; disclose aggregation; unchanged real genes are not certified negatives |

DESeq2's mean-DE target differs from subset enrichment above a depressed bulk.
Show common mean-DE tasks and the additional distributional tasks separately;
recovery of the latter cannot establish universal superiority. Use the native
[DESeq2 workflow](https://bioconductor.org/packages/release/bioc/vignettes/DESeq2/inst/doc/DESeq2.html)
with documented settings. The [Schurch replicate dataset](https://pubmed.ncbi.nlm.nih.gov/27022035/)
provides a recognizable reference, but an all-sample call set is not absolute
biological truth. No single dataset is beyond methodological dispute.

**Inference and reporting rules:**

- Correct the actual requested hypothesis family. The first candidate searches
  both directions and uses one BH adjustment over 2G tests. Also measure FDP
  after collapsing to unique genes; control for one unit does not automatically
  establish control for another. Assess wrong-direction calls separately.
- A predeclared up-only analysis would have G tests. Do not combine separately
  corrected up/down lists and advertise them as one 5% FDR discovery set.
- Retain the baseline's one unsigned test per gene as a historical comparison,
  not a release candidate or proof of directional discovery. Linear/log choices
  are fixed per analysis before testing, not selected by the best per-gene q.
- Use sufficient permutation resolution. With 20,000 genes and two directions,
  an isolated 5% BH discovery needs a p-value at most 1.25e-6. The first candidate
  cap is 999,999 shuffles, giving a floor of 1e-6. Record limitations for larger
  families and restricted designs; never report an interrupted run as complete.
- Replicate **whole matrices**, not just genes. Paired dataset-level intervals
  support comparisons; uncertainty across genes in one matrix is insufficient.
- A nonsignificant calibration test is not evidence of equivalence. The expanded
  confirmation protocol must specify a practical FDR tolerance and replication
  sufficient to bound excess error, using development variability for planning.
  An inconclusive interval requires more prespecified replication or a narrower
  claim, not a declaration of success.
- Publish precision–recall/power–FDR behavior and effect accuracy as well as a
  nominal-q operating point. Include false positives, exclusions, undefined
  fold changes, raw denominators and independent simulation/replay seeds.

## 3. Scientific decision, then production migration

Select a complete procedure only after it answers the directional biomarker
question and demonstrates acceptable error control, useful bulk/subset recovery
and interpretable effects. If the directional prototype fails, diagnose and
revise it under a new frozen protocol. Reverting to unsigned discovery does not
satisfy the user's release objective. Kernel research remains deferred.

- [x] Audit and freeze the expanded benchmark protocol after the instrument run.
- [x] Complete the frozen 120-matrix development grid, native DESeq2/edgeR fits,
  independent audit and paired sensitivity/FDP/effect assessment.
- [ ] Complete the directional biomarker scorecard and targeted diagnosis above.
- [ ] Replay established comparison inputs with the same instruments.
- [ ] Freeze the selected operating procedure and confirm on fresh datasets/seeds.
- [ ] Migrate the selected procedure with Python/native parity and meaningful
  inference/regression tests. Optimize only measured bottlenecks.

The [replicated comparison](research/protocols/point-comparison.json) completed
120 full transcriptomes, three sample allocations, two biological models,
four direction/scale point procedures and native DESeq2/edgeR. All 120 exact
public-baseline replays passed; the independent audit checked 3,271 files,
native input identity, stopping/BH, detection counts and quantile areas. Native
fitting retained outlier handling and exclusions; effect recovery uses both
population and realized-patient truth. Figures/results await user approval.

The [assessment](docs/index.md#replicated-directionscale-comparison--2026-09-26)
records the full table and uncertainty. At 250:250, sensitivity/mean FDP are
37.0%/5.1% for unsigned linear, 30.4%/5.1% for the log baseline and 25.3%/4.7%
for joint directional linear. At 500:50, the baseline is more sensitive than
either linear procedure. Native performance depends strongly on the generating
biology; do not generalize pooled results or claim a universal winner. Twenty
null matrices per design cannot tightly certify FDR. These sensitivities count
**any distribution change**, including opposite-direction recovery, and cannot
select the release method for the corrected biomarker objective.

**Next:** complete section 0's directional diagnosis and relevance scorecard.
Established-input replay follows, with explicit mean-DE and subset targets.
Any new follow-up uses a frozen protocol and retains negative results; it does
not replace the completed comparison or convert development into confirmation.

The design target is one call with explicit `direction="up"` or `"down"`, and
`scale="linear"` or `"log"`. Upward biomarker discovery is the primary next
evaluation; linear positive area is the starting prototype, not an approved
default. A both-direction option, if retained, must correct the joint family
and preserve separate results; it cannot mean one unsigned discovery. Final
API/default selection follows validation. These arguments are not yet available
in public `wade.wade()`.

Keep the current simple normalization/filtering API. A validated housekeeping
panel can be supplied explicitly; automatic control discovery, sparse-assay
certification and universal depth-confounding correction are outside this
release. No fixed percentile or hard patient classification is required for
an area-based gene discovery.

## 4. Complete examples, review and distribution

- [ ] Refresh TCGA prostate fusion-oriented and breast subtype analyses using
  full gene counts only. Finish count-pipeline/version, genome/annotation and
  row-order provenance. Retain independent patients, distinguish missing fusion
  annotations from negatives, and disclose expression-derived PAM50 labels.
- [ ] Investigate marker effects and broad tumour/normal call sets. Real data
  illustrate biology; they cannot supply complete sensitivity/specificity truth.
- [ ] Reexecute all six notebooks: demo, benchmarks, reference benchmarks,
  prostate, breast and private RNA100k. Review the demo's themes/probes,
  explanations, before-running case-exclusion recipe and final API examples.
- [ ] Approve the exact final results, figures and claims with the user. Prior
  QA and prior approvals do not approve changed figures or results.
- [ ] Pass source, installed-wheel, NumPy fallback, distribution and cross-platform
  CI checks. Finish release notes/citation metadata. Obtain final authorization
  before publishing or tagging.

## Evidence and deferred work

The [status assessment](docs/index.md) retains measured baseline and kernel
results. The [production migration](archive/development-20260923/docs/production-migration-0.2.0.md)
validated the existing unsigned log-W1, preprocessing, installed package and
notebooks; it does not certify a changed directional procedure. The
[cleanup archive](archive/development-20260923/README.md) retains prior plans and
negative results. Archives and local result artifacts stay excluded from Git
and release distributions.

After 0.2.0: coherent measurement kernels with demonstrated benefit, transcriptome
rarity/predictive surprise, sparse/single-cell/spatial/cell-free assays, automatic
control discovery and richer covariate handling. Kernel research remains a
valuable option, not a promised prerequisite or an imminent performance gain.
