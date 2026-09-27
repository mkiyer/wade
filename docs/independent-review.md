# WADE: implementation, evidence, unresolved questions, and readiness for release

**Prepared for independent methodological critique — 26 September 2026**

**Artifact availability — 27 September 2026:** the entire local `.release/`
directory, including the supporting review ZIP, was deleted at the author's
request to recover disk space. Results below summarize previously completed
experiments; their saved `.release/` inputs, outputs, snapshots and audit records
are no longer available locally. Repository source, protocols and this review
remain. Rechecking those numerical results requires regenerating the relevant
experiments and reinstalling any removed benchmark dependencies.

This is a development-team assessment of the current working tree, not an
independent review or a claim of validated performance. It includes negative
results and corrections to earlier interpretations. The intended reader should
not need the development conversation to understand the questions.

WADE means *Wasserstein area differential expression*. Its intended use is
biomarker discovery in large gene-expression cohorts: identify reproducible
upward or downward expression differences, including striking expression in a
minority of patients. The immediate release target is bulk RNA-seq, with TCGA
prostate and breast gene-count examples. Measurement-uncertainty kernels have
been deferred beyond 0.2.0 by agreement with the author.

## 1. Assessment

**There is substantial working software, but the current public method is not
the agreed release method.** The package implements an unsigned distribution
test on log-transformed normalized counts. The author has explicitly rejected
unsigned-only discovery as the primary result: a distribution difference alone
does not answer whether a gene is a useful directional cancer biomarker.
Upward/downward area tests exist in research code, but have not demonstrated
adequate performance for all of the intended patterns or been migrated into
the public API.

**The reported 0/80 is a real, specific failure of that directional prototype;
it is not evidence that WADE cannot detect differential expression or subsets.**
It means zero transcriptome-adjusted *upward* discoveries for one simulated
pattern across 80 gene profiles. The experiment did not recreate the large,
clean ETV1/ETV4-type signals motivating the tool. Independent checks reproduce
the areas and weak evidence in selected profiles. A clean positive control
with the same nominal subset structure is readily detectable by the same
linear-area statistic. Section 4 separates these observations.

**The main obstacle is alignment of the biological target, statistic, null
model, and validation endpoint.** It is not simply missing optimization or a
broken area calculation. Earlier development drifted toward rewarding any
distribution change. In particular, downward discoveries for a depressed-bulk
gene were counted as successful recovery of a gene whose desired feature was
an upward subset. That was an interpretation and benchmark-target error.
The numerical results are retained, but that release recommendation is withdrawn.

The present evidence supports neither abandoning directional area nor declaring
it ready. The next useful work is a bounded reconciliation with successful
historical HITLIB analyses and a biologically matched directional benchmark,
followed by fresh confirmation of a frozen procedure.

## 2. What exists today

The package distribution is `wade-rnaseq`, imported as `wade`. Its metadata says
0.2.0; that is not evidence of release approval. The public core requires NumPy
and has a Rust implementation plus a NumPy fallback. The native module named
`wade._kernel` is a computational backend, **not** an abundance-uncertainty model.

| Component | Implemented | Limit or unfinished work |
| --- | --- | --- |
| Inputs and contrasts | Gene-by-sample matrices, identifiers, count validation, named case/control selection, optional permutation strata. | Independent, exchangeable sampling units are required; this is not a general covariate-adjusted count regression. |
| Normalization | RLE median-ratio factors; designated reference genes; immutable fitted factors reused across contrasts and test masks; external size factors with provenance; library-total factors. | No certified housekeeping panel is supplied. No validated automatic low-variance control discovery. No built-in scran/TMM/TMMwsp estimator. |
| Per-sample abundance | CPM and TPM utilities. | Abundance conversion is distinct from a validated inferential workflow. These are not additional public test modes. |
| Filtering | Pooled count/detection/total-count filters and explicit gene masks. All-zero genes are excluded from testing. | No universal filter is claimed. A released operating recipe must specify sample QC, the normalization universe, and the test family. |
| Public statistic | Exact empirical unsigned Wasserstein-1 on `log2(count / size_factor + 1)`. | No public direction or scale argument. This does not meet the revised primary objective. |
| Inference | Label permutations, optional within-stratum shuffles, recorded sequential stopping, finite Monte Carlo p-values, BH adjustment. | Does not by itself handle confounded depth/batch, quantify individual measurement error, or establish a composite directional-null guarantee. |
| Results | One statistic/p/q per gene, tested/filtered status, normalization records, median shift, descriptive upper/lower decile contrasts, plots and exports. | A tail description is not a tail p-value, patient outlier probability, or estimated biological subset prevalence. |
| Directional research | Exact positive/negative areas on linear and log scales; shared label draws; joint correction over gene-direction pairs; NumPy/native comparison instruments. | Research implementation only; scientific selection and production migration remain. |
| Uncertainty research | Normalized Beta abundance kernels, group-mixture integration, caching, coverage/prediction and fixed-total diagnostics, alternative mode-preserving constructions. | No selected construction passes all requirements. Deferred from 0.2.0. |
| Benchmarks | Replicated full-transcriptome point comparison; native DESeq2/edgeR; earlier reference and OUTRIDER pilots; truth, provenance, raw inference records and audits. | Not a final publication benchmark for the revised directional target. |
| Examples | Six notebooks executed for the current baseline; TCGA prostate and breast gene-count analyses and clinical annotations. | Final directional results, input provenance, interpretation, figures and user approval remain incomplete. |
| Engineering | Current package/research suite: **175 tests passed** in the present audit. CI/wheel/source-distribution machinery exists. | Tests establish software behavior, not biological validity. Final candidate installation, fallback and cross-platform checks must follow migration. |

Production sources are [API](../src/wade/api.py),
[normalization](../src/wade/normalize.py),
[filtering](../src/wade/filtering.py), and
[area/inference](../src/wade/_wasserstein.py). The directional prototype is
[point_workflow.py](../research/wade_research/point_workflow.py); the factorial
comparison is [point_factorial.py](../research/wade_research/point_factorial.py).
Research is excluded from package distributions. Old two-stage bulk-correction
tests, tail extrapolation, jitter and count resampling are not active production
behavior.

Single-cell, spatial and cell-free RNA workflows are not certified by these
bulk studies. Importing external normalization factors does not establish that
cells, spots or repeated samples are independent patients or exchangeable units.

The implemented normalization contract allows initial sample/feature QC before
fitting. In the latest benchmark, RLE is fitted to the complete simulated matrix
before the *testing* filter. The purpose is to avoid silently changing the
normalization reference whenever the tested or displayed genes change. It is
not a claim that all RNA-seq workflows should normalize before any QC or filtering.

## 3. The methods being compared are different

### Point representation, direction, and scale

Let `x_gj = count_gj / size_factor_j`. A point representation gives every
patient equal mass at one value, either `x_gj` or `log2(x_gj + 1)`. It does not
spread that patient's mass over measurement uncertainty. This choice is
independent of direction and scale; the unsigned baseline was already point-based.

For case and control quantile functions `Q1(u)` and `Q0(u)`, define

```text
A_up   = integral_0^1 max(Q1(u) - Q0(u), 0) du
A_down = integral_0^1 max(Q0(u) - Q1(u), 0) du
W1     = A_up + A_down
Signed = A_up - A_down = mean(case) - mean(control)
```

The last mean is on the chosen expression scale. The signed integral is
therefore not a distinct general-purpose subset statistic: large subsets can
move the mean, while opposing changes can cancel. These are exact empirical
areas, not fold changes. The standard identities follow from integrated
quantile functions and the one-dimensional Wasserstein definition.
([Gushchin and Borzykh, 2017](https://www.vmsta.org/journal/VMSTA/article/99/read);
[SciPy definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html).)

For controls `[10, 10, 10, 10]` and cases `[2, 2, 2, 30]`, linear upward area
is 5, downward area 6, unsigned area 11, and signed area −1. The upward component
retains the high case despite the lower group mean. This algebra does **not**
guarantee a small p-value, accurate localization, or clinical usefulness.

### What the original HITLIB source actually does

The inspected predecessor is `probe_panel/probe_panel_v06.Rmd` in the author's
cfRNA repository, functions around lines 1490–1715. This is one historical
implementation, not a reconstruction of every successful HITLIB analysis.

| Aspect | Inspected HITLIB | Current public WADE | Directional research |
| --- | --- | --- | --- |
| Default scale | Linear | Log2(normalized count + 1) | Linear and log evaluated separately |
| Normalization | Per-library length-adjusted abundance | RLE normalized counts | Same RLE factors as public comparator |
| Statistic | Final cumulative **signed** quantile difference | Unsigned W1 | Positive and negative areas separately |
| Quantiles | R interpolated quantiles on `min(n_case, n_control)` points, including endpoints | Exact empirical area | Exact empirical area |
| Noise | Fresh uniform count noise for observed and permuted evaluations; denominator updated for that gene's draw | None | None in the completed comparison |
| Significance | Fraction of permutation scores at least as large; no BH inside this wrapper | Recorded Monte Carlo inference and one gene-level BH family | Recorded inference and joint BH over both directions |

For equal group sizes, that HITLIB quantile grid visits each order statistic,
so its final signed score is the difference of means after its normalization
and perturbation. With unequal groups, its endpoint-inclusive grid can give
extremes different influence from exact empirical integration.

The inspected example invocation uses one permutation and noise width 0.01.
It is not sufficient to reconstruct a calibrated historical discovery analysis;
the wrapper also allows zero empirical p-values. Modern Monte Carlo testing
must avoid interpreting unobserved exceedances as probability zero.
([Phipson and Smyth, 2010](https://pubmed.ncbi.nlm.nih.gov/21044043/).)
**These source observations do not invalidate the author's historical marker
discoveries.** They show that the recent benchmark is not a faithful replay
of the original data, statistic, scoring and selection procedure.

A subgroup with 32–1,024-fold expression can dominate a linear mean or area
even when the remaining cases are low. Historical success is consequently
quite plausible. A useful reconciliation needs the historical count matrix,
labels, normalization, sample allocation, marker rankings, and selection
criteria—not merely a method carrying the same name.

## 4. What 0/80 means, and what the new audit establishes

### The simulation and endpoint

The 0/80 result comes from twenty independent **250-case : 250-control** mixed
matrices, each containing 16,384 genes. Four altered genes per matrix implement
the pattern at baseline means 0.5, 5, 50 and 500, giving 80 gene profiles.
These are not 80 patient predictions or 80 independent experiments.

For each such gene:

- Patients first receive variable latent biological abundances. The baseline
  biological CV² lies between 0.2 and 1, corresponding to CVs of about 45–100%.
- A randomly selected 10% of cases are multiplied by four; the other 90% by one
  half. Membership is independent of the patient's initial abundance.
- Counts are sampled as Poisson(exposure × latent abundance). Exposure varies
  independently of condition; no systematic batch or group-depth imbalance
  was injected.
- The expected case/control arithmetic-mean ratio is
  `0.9 × 0.5 + 0.1 × 4 = 0.85`. The desired upward subgroup therefore coexists
  with a lower overall mean and overlapping biological distributions.
- Upward discoveries require q≤0.05 after testing the eligible transcriptome.
  Only 48 genes in each matrix are altered; this is a sparse discovery setting.

Ten matrices use Gamma–Poisson biology and ten correlated lognormal–Poisson
biology. All 80 profiles survive filtering. A fourfold multiplier is **2 log2**,
not the **5–10 log2** effects motivating the tool. The largest multiplier in
other profiles of this benchmark is sixteenfold, or 4 log2.

With ideal normalization, the linear signed mean contrast would be negative
in the population for this particular construction. It is therefore also a
different signal from an extreme subgroup that raises the linear mean strongly.

The original result is **0/80 upward q≤0.05 on both linear and log scales**.
However, **19/80 have unadjusted linear upward p≤0.05**. The smallest saved
linear p-value is about 0.000976, with joint q≈0.679. There is weak-to-moderate
evidence in some profiles, not uniformly zero observed signal. On log scale,
none of the 80 upward p-values is at most 0.05. Log analysis instead makes
56 downward discoveries; crediting these as upward-subset recovery was wrong.

An exploratory up-only BH adjustment over G genes, instead of the original
joint 2G family, still gives 0/80. Thus the extra direction's multiplicity
penalty is not the whole explanation. Multiplicity does matter when translating
the nominal linear evidence into transcriptome-wide discoveries.

### Checks performed specifically for this review

The new [audit script](../research/audit_subset_failure.py) leaves the original
results unchanged. It:

1. Checks the hashes of all 40 relevant saved input/result files.
2. Independently sorts quantiles and recomputes six area statistics for all 80
   profiles. The maximum discrepancy is **1.71 × 10⁻¹³**.
3. Regenerates two original full matrices exactly, including integer counts,
   latent abundances, exposure and membership arrays.
4. Runs **49,999 fresh, fixed-budget permutations** using direct sorting and an
   independent random stream, without the production scoring/stopping code.
5. Replaces fitted RLE factors with known true exposures, and separately removes
   count sampling by using the saved latent biological abundances.

The selected datasets are the first Gamma replicate (`r00`) and the replicate
containing the smallest original upward p-value (`r15`, lognormal). Selection
is disclosed and post hoc: this is a diagnostic, not a new power estimate.
For their high-abundance profiles (baseline mean 500), fresh **linear upward**
p-values are:

| Selected profile | RLE-normalized counts | Counts / true exposure | Latent abundance, no count noise |
| --- | ---: | ---: | ---: |
| First Gamma replicate | 0.03694 | 0.03810 | 0.03880 |
| Best original upward-p replicate | 0.00092 | 0.00106 | 0.00126 |

The fixed-permutation binomial 95% intervals for the first column are
approximately 0.0353–0.0386 and 0.000657–0.001204. Original sequential estimates
were 0.02215 and 0.000976; finite Monte Carlo estimates need not agree exactly.
Both implementations leave these profiles far from transcriptome-adjusted
discovery. At baseline mean 0.5, removing count noise does materially improve
the evidence, so low-count precision remains a separate issue.

These checks make an area/indexing error, RLE bias, or count noise an inadequate
explanation for the two high-count examples. They **do not rule out every
software defect or every normalization problem in other designs**.

### A clean positive control succeeds

Take 250 controls all at 100, 225 cases at 50, and 25 cases at `100 × fold`.
This is an intentionally deterministic arithmetic control with no biological
variation or count sampling. Enumerating the complete label distribution by
category counts gives:

| Upward-subset multiplier | Linear upward area | Exact upward p, linear | Exact upward p, log2(x+1) |
| --- | ---: | ---: | ---: |
| 4-fold | 30 | 1.13 × 10⁻⁶ | 9.76 × 10⁻⁴ |
| 16-fold | 150 | 1.59 × 10⁻⁸ | 1.18 × 10⁻⁵ |
| 32-fold | 310 | 1.59 × 10⁻⁸ | 3.07 × 10⁻⁶ |
| 1,024-fold | 10,230 | 1.59 × 10⁻⁸ | 1.20 × 10⁻⁷ |

These are exact conditional permutation probabilities evaluated numerically,
not Monte Carlo zeros and not a full-transcriptome RNA-seq power experiment.
The small analogue was also checked against exhaustive label enumeration.
The calculation is given in Appendix A.

**Interpretation:** the same linear-area statistic can detect an upward subset
above a depressed bulk. Broad biological overlap, a weaker-than-motivating
effect, the chosen calibration, low-count profiles, and a sparse transcriptome
testing family make the completed experiment substantially harder. The audit
does not isolate their contributions across all 80 profiles. It establishes
that “the method predicts nothing” and “the original successful algorithm has
been disproved” are unsupported conclusions.

The remaining substantive question is whether the current positive-area test
is sufficiently sensitive to the *realistic* directional signals WADE is meant
to discover. Neither the hard simulation nor the clean toy answers that alone.

## 5. What the comparative evidence supports

### Replicated full-transcriptome comparison

The [frozen protocol](../research/protocols/point-comparison.json) completed
**120 matrices**: 20 complete-null and 20 mixed at each of 250:250, 50:500 and
500:50 cases:controls. Four point procedures received identical counts, RLE
factors, test masks and label streams. Each direction/scale procedure was
evaluated separately; no minimum-p merging across procedures was used.

Native R **DESeq2 1.52.0** and **edgeR 4.10.5** received the same integer counts.
DESeq2 independently reproduced the RLE factors and retained native outlier
handling and independent filtering. edgeR used robust modern quasi-likelihood
fitting with TMM, plus a matched-RLE analysis. These are native fits, not ports.
An audit checked 3,271 artifact files, native count identity, stopping records,
BH, call counts and area integration. All 120 public-baseline replays matched.

Entries below are **sensitivity / mean realized false discovery proportion**,
in percent, at q≤0.05. They average twenty mixed matrices per column. The truth
is the original **any-distribution-change** endpoint; directional rows count a
gene called in either direction. This table cannot select the biomarker method.

| Procedure | 250:250 | 50:500 | 500:50 |
| --- | ---: | ---: | ---: |
| Public unsigned log baseline | 30.4 / 5.1 | 10.5 / 5.9 | 5.4 / 8.7 |
| Unsigned linear comparator | 37.0 / 5.1 | 11.3 / 5.3 | 1.4 / 0.0 |
| Directional log, joint up/down | 27.4 / 6.9 | 8.8 / 5.6 | 3.8 / 2.1 |
| Directional linear, joint up/down | 25.3 / 4.7 | 8.0 / 5.4 | 1.8 / 3.5 |
| DESeq2 | 34.0 / 17.2 | 17.7 / 20.8 | 11.8 / 17.6 |
| edgeR QL, TMM | 34.2 / 15.8 | 20.4 / 21.1 | 9.5 / 24.6 |
| edgeR QL, matched RLE | 34.2 / 14.4 | 20.6 / 21.2 | 10.0 / 25.0 |

FDP is `false calls / max(all calls, 1)` in each matrix. Its average estimates
FDR under the simulated ensemble; it is not pooled FP divided by pooled calls.
Zero observed FDP in a low-power procedure is not proof of zero error.
Specificity exceeds 99.97% for every row, because each mixed matrix has 16,336
unchanged genes. High specificity can therefore coexist with an unacceptable
fraction of false discoveries; both metrics are needed.

The native-method findings are strongly model-dependent. At 250:250,
DESeq2/edgeR-TMM mean FDP is **5.7%/5.1% under Gamma–Poisson**, versus
**28.6%/26.4% under correlated lognormal–Poisson**. Changing generator changes
both distribution shape and dependence. This is not an isolated experiment on
either cause, or evidence of generally poor DESeq2/edgeR performance.
The DESeq2 and edgeR count models have a different primary mean-DE target from
crossing-distribution subset enrichment.
([DESeq2 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC4302049/);
[edgeR paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC2796818/).)

Useful direction-specific observations from the balanced cell are:

| Planted feature and desired direction | Linear directional discoveries | Log directional discoveries |
| --- | ---: | ---: |
| 10% of cases up 16-fold | 59/80 up | 15/80 up |
| 10% up fourfold, remaining cases halved | 0/80 up; 7/80 down | 0/80 up; 56/80 down |
| 20% of cases up fourfold, remaining cases unchanged | 46/80 up | 20/80 up |
| All cases up 1.5-fold | 57/80 up | 53/80 up |
| All cases multiplied by two thirds | 53/80 down | 54/80 down |
| One case up 16-fold | 0/80 up | 0/80 up |

Thus the current prototype does detect bulk and subset changes. It is weaker
on some important patterns and when the control reserve is small. Large total
sample size alone is not sufficient; the number and representativeness of
normal controls matter.

Effect estimation also was measured on all altered genes, including noncalls.
At 250:250, population log2 arithmetic-mean-ratio RMSE is approximately **0.153
for the empirical RLE group-mean estimator**, **0.144 for DESeq2**, and **0.144
for edgeR**. All four point procedures share that same auxiliary empirical
mean estimator; changing the test does not create a new fold-change estimate.
The public `bulk_shift` is instead a median log-expression contrast. Neither
quantity estimates the fold change of an unidentified biological subgroup.

Twenty matrices per cell, split between two generators, are development
evidence—not enough to certify precise FDR equivalence near 5%. Fresh seeds,
prespecified tolerances and adequate whole-matrix replication must follow
method selection. The current simulation is custom and sparse in alternatives;
it must be complemented by established references and realistic effect profiles.

### Earlier benchmarks and real data

On conventional compcodeR mean-DE simulations with 50 samples per group, the
public baseline recovered **75.2%** of requested DE genes versus **77.0% for
native DESeq2** and **77.3% for edgeR**, with mean FDPs of **4.9%, 5.8% and
5.8%**, respectively. There were only three matrices at that size. This is
encouraging evidence of conventional DE capability, not a certification or a
validation of the newer directional prototype.

Earlier large-cohort point-baseline studies do not show universal failure:
one separate 500:500 full-transcriptome confirmation achieved 57.8% sensitivity
across nine patterns and mean FDP 5.27% (matrix-level 95% interval 4.06–6.48%).
Those designs differ in truth and signal composition and cannot be pooled with
the present 48-alternative experiment. Earlier high-alteration studies also
showed normalization-induced false discoveries; replacing estimated factors
with true exposures helped there. That mechanism must not be assumed to explain
every later failure.

Native OUTRIDER and related-method pilots exist, but are not a completed
publication comparison. OUTRIDER's patient-gene aberrant-expression task differs
from WADE's cohort-gene task; aggregation into a common endpoint requires care.
Unmodified genes in a real cohort are not certified negative controls.
([OUTRIDER paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6288422/).)

The current TCGA baseline calls ETV1 and ETV4 as **distribution differences**
(q≈0.00251 and 0.000405), and detects familiar prostate and breast markers.
These are not validated upward-subset p-values or fusion classifiers. The
independent tumour/normal comparisons use 445:52 prostate samples and 971:112
female breast samples, with matched-patient exclusions. Prostate and breast
examples use gene counts; splice-junction analyses are outside this release.

Broad tumour/normal discovery sets—74.1% of tested prostate genes and 95.2% of
breast genes—need biological interpretation. Known marker recovery does not
measure genome-wide specificity. PAM50 labels are expression-derived, missing
fusion annotations are not fusion negatives, and adjacent normal tissue is
not a universal healthy reference. Local TCGA count-pipeline/annotation
provenance must be finalized before publication-quality use.

## 6. Unresolved issues in the point method

### 6.1 What precisely does a directional p-value claim?

The implemented permutation null is equality/exchangeability of the normalized
distributions. Using upward area makes the statistic directional, but the
current work has not established a general test of the composite statement
“there is no upward enrichment of a useful size.” Neither a rejection nor a
positive area means that every case increases, that an individual patient is
an outlier, or that a diagnostic panel will separate cancer from controls.

For crossing distributions, the lower bulk no longer subtracts from the observed
upward area. However, those low observations still enter the pooled permutation
distribution. Random label splits mix both low and high observations, changing
the null spread. Rectifying the observed area is therefore not a guarantee of
high power against an upward minority above a depressed bulk.

We need an explicit estimand, null and claim before selecting a new statistic
or calibration. If a data-selected cutoff enters the score, the cutoff search
must be repeated within every permutation. Otherwise the p-value ignores the
selection that made the observed score large.

### 6.2 Biological importance and statistical surprise differ

Linear area emphasizes absolute normalized abundance; log area emphasizes
relative changes and compresses very large expression. Neither is a fold change,
and neither by itself incorporates assay detectability or normal-background
specificity. A release can offer a predeclared scale choice, but should teach
one justified default and avoid selecting the best per-gene q-value across scales.

Per-gene label permutation conditions on that gene's observed values. It does
not learn that an enormous fold change is rare across the transcriptome. As a
simple example, with 500 cases, 50 controls, and exactly 50 high observations
all in cases while all others share one low value, the upward-area permutation
p-value is `choose(500,50) / choose(550,50) ≈ 0.00672`. Increasing the common
high value scales every permutation score equally and leaves this p unchanged.
This is an exact two-level example, **not a universal lower bound** for real data.
It shows why amplitude and cohort allocation need separate consideration.

A calibrated transcriptome-wide rarity model has **not** been implemented.
Borrowing across genes could supply information absent from a within-gene
permutation, but would add assumptions about comparable genes, abundance,
dispersion and technical artifacts. This should not be hidden inside an
unexplained significance score or made a prerequisite for the narrower release.

### 6.3 Normalization and sampling assumptions remain substantive

RLE assumes a sufficiently stable reference population; a small targeted panel
or a coordinated majority shift can violate that assumption. Validated internal
controls can help and are supported. Low variance alone does not prove that a
gene is invariant to the biology, composition or protocol of interest. Automatic
reference nomination, including mean–variance-adjusted approaches, remains
research rather than a production default.

Count normalization changes the scale, not the information content. One read
from a shallow library can have a large normalized value. Systematically
different depths can yield different normalized measurement distributions even
when latent expression is identical; unrestricted labels need not be exchangeable
under that biological null. Strata help only where a valid design permits
within-stratum exchanges. Neither added noise nor a kernel automatically
repairs biological/batch confounding. A bulk release needs a clear sample-QC
and design recipe with documented boundary behavior.

### 6.4 Calibration, multiplicity, and Monte Carlo precision

The production algorithm stops at its 20th inclusive exceedance, reporting
`20 / stopping_time`, or at the permutation cap, reporting `(K+1)/(B+1)`.
The public cap is 499,999; the research comparison uses 999,999. Records are
saved and small-orbit/backend behavior is tested. Fixed permutations in this
audit provide a separate check for selected failures; they do not revalidate
the entire stopping procedure under every design.

For 20,000 genes and two directions, an isolated 5% BH discovery needs
p≤1.25×10⁻⁶. Finite resolution and Monte Carlo variability must be reported;
more permutations refine evidence, but will not generally turn p≈0.001 into
p≈10⁻⁶. Sequential early stopping can make individual estimated p-values
noticeably variable even when the scientific conclusion is unchanged.

Joint BH over gene-direction pairs, BH over a predeclared up-only family, and
error after collapsing directions to unique genes are different reporting
contracts. Gene dependence and shared normalization also matter. The present
studies do not prove arbitrary-dependence or general directional-error control.
These limits need explicit statements and practical validation, not an assurance
that a nominal q threshold alone settles them.

### 6.5 Localization and biomarker performance are unfinished

The research code can attribute upward area to patients and describe how
concentrated it is. A maximum walk or tail cutoff can select almost all cases
even when the planted affected fraction is 10%; it is not automatically a
mixture-proportion estimator. Ranks, descriptive contributions, subgroup fold
change, significant patient calls and clinical sensitivity/specificity are
distinct outputs.

A minimal release need not solve every one of these tasks. It does need to
show which patients drive a discovery, keep unsupported patient probabilities
and prevalence claims out of the API, and validate any advertised selection
rule on held-out or independent patients. Gene-discovery sensitivity/specificity
in a simulation is not diagnostic-test performance in patients.

## 7. Uncertainty-kernel derivation: progress and remaining failures

The proposed extension replaces each patient's point by a probability kernel
over abundance, averages these kernels within groups with equal patient mass,
then calculates area between the resulting distributions. This remains a
reasonable research proposal; it has not demonstrated a generally superior
or adequately calibrated production procedure.

The tested Beta kernels **were normalized**. For raw gene count C, library
total N, size factor s and abundance fraction P, the expression axis was
`A = (N/s) P`, matching the point `C/s`. “Uniform” and “Jeffreys” named prior
choices in `P | C,N ~ Beta(C+h, N-C+h)`, with h=1 or 1/2. They were not an
unannounced replacement of count likelihoods by unrelated kernel families.

Several distinct questions must be separated:

1. A binomial/Poisson model describes observed counts given a latent fraction
   or rate and an exposure.
2. A posterior or confidence distribution for that latent abundance requires
   further construction; it is not obtained just by centering the count
   likelihood at the observed count.
3. Beta-binomial or negative-binomial distributions can describe future counts
   under mixing/overdispersion. A predictive count kernel is not automatically
   uncertainty about the latent abundance used for comparing patients.
4. Across-patient biological heterogeneity is not interchangeable with technical
   measurement overdispersion. Separating them requires assumptions or additional
   information, such as technical replication.

The author's constraint is that raw counts remain unchanged, a zero observation
retains mode zero, and uncertainty remains positive. A nonnegative distribution
with mean exactly zero must be a point mass at zero; hence positive uncertainty
at zero implies a positive **kernel mean**, even though its mode can remain
zero. That distinction is mathematical, not a reason to rewrite observed counts.

| Construction or requirement | What the derivation/audit established | Unresolved consequence |
| --- | --- | --- |
| Independent symmetric Beta priors per gene | Sum of posterior fractions is `(N + G h)/(N + 2 h)`. For 16,384 genes and N=1,000 it is 17.35 with h=1 and 9.18 with h=1/2. | Not a coherent allocation of one library's abundance across all genes. Prior contribution also changes with depth. |
| One Dirichlet prior over the library | Allocates prior information once and preserves total mass. | Within this family, positive priors cannot preserve every marginal mode at C/N when three or more categories have positive counts. Joint and marginal modes are different. |
| Discrete fixed-total construction | Multinomial probabilities `(C_g+w_g)/(N+1)`, with positive weights summing to one, preserve each marginal count mode C_g, the realized total N, and positive zero-count uncertainty. | It is smoothed count resampling. Its unseen-mass budget and use as latent-abundance uncertainty still need justification and calibration. |
| Continuous mode-preserving contraction | Can preserve modes and joint total by contracting a coherent draw toward observed fractions. | The tested construction is too narrow: nominal 95% coverage tends to about 79.17% in a high-information limit; measured coverage was 78.90% versus 94.83% without contraction. |
| Average patient posterior kernels, then compare | Conserves patient mass and produces computable group distributions. | Adds within-kernel spread to variation of estimated locations; it is not automatically deconvolution of measurement noise or a less noisy estimate of the biological distribution. |

These results reject particular constructions or claims, not all possible
uncertainty models. There is still no agreed combination of location, width,
unseen-mass allocation, overdispersion, normalized scale and null calibration.
Severe minority-library downsampling experiments did not show the hoped-for
robustness benefit for the tested kernels. They do not establish that kernels
can never help or that ordinary balanced bulk data have the same behavior.

The derivations and instruments are retained in the
[kernel roadmap](kernel-validation-roadmap.md). Kernel correctness is now a
**deferred research issue**, not a requirement that should indefinitely block
the agreed point-based 0.2.0 release.

## 8. Concrete barriers to a defensible 0.2.0

| Order | Required completion | Evidence needed to close it |
| --- | --- | --- |
| 1 | Freeze the directional biomarker claim and scorecard. | Requested-direction recovery; useful magnitude and normal-background separation; explicit distinction between statistical null errors and biologically unhelpful nominations. |
| 2 | Reconcile historical success and the failed subset pattern. | Matched historical replay where data are available; controlled varying biological overlap, abundance, subgroup size and 4–1,024-fold signals; retain difficult/negative profiles. Independently audited scoring and calibration. |
| 3 | Select and confirm one simple complete procedure. | A frozen statistic, default scale, normalization/filtering recipe, actual hypothesis family, permutation settings and limits; fresh whole-transcriptome replication with prespecified error/power tolerances. |
| 4 | Establish comparative scope. | Native DESeq2/edgeR on common mean-DE tasks and established inputs; separate upward-subset tasks and effect recovery; patient-gene methods only on matched outlier endpoints. Report failures and achieved error as well as sensitivity. |
| 5 | Migrate the selected procedure. | Public directional API/results, meaningful descriptive patient contributions, independent numerical/backend checks, no undocumented selection across scales or directions. |
| 6 | Rebuild and review all user material. | Demo, benchmark, reference and example notebooks on the final API; TCGA gene-count provenance; prostate fusion and breast subtype interpretation; all plots/results explicitly approved. |
| 7 | Finish release engineering. | Clean install from wheel and source; NumPy fallback and platform CI for the actual candidate; API/document consistency, distribution manifest, version/changelog, final release authorization. |

Steps 1–4 are scientific barriers; 5–7 are implementation and delivery barriers.
The existing software is useful for research, but **release readiness under
the newly agreed claim has not been established**. This is not a requirement
to outperform every parametric method at small sample sizes. WADE can have a
defensible role if its large-cohort directional utility and limitations are
demonstrated honestly.

The immediate follow-up should be small and diagnostic. Retain the 0/80 inputs;
vary effect magnitude and biological overlap separately, including the original
extreme-effect range. Compare historical signed scoring, exact positive area
and native mean tests with the same normalization where possible. A historical
algorithm replay and a modern, calibrated inference comparison must be labeled
separately. Add shared tails and ordinary bulk shifts as controls. Score upward
discovery, effect recovery and patient ranking together with false discoveries.
Do not start another broad method search or optimize Rust before this is clear.

## 9. Questions for the independent reviewer

1. Is positive quantile area the right simple estimand for upward biomarker
   enrichment above potentially depressed bulk? What important signals or
   nuisance patterns does it inherently emphasize or miss?
2. Is equality-null label permutation an adequate inferential contract for
   that estimand? If a stronger directional null is necessary, what minimal
   derivation and validation would make its meaning defensible?
3. Does the 0/80 simulation represent a meaningful biological challenge, an
   unnecessarily difficult stress test, or both? Which parameters should be
   anchored to real normal/cancer distributions before further comparison?
4. Which historical HITLIB differences are likely to matter most: scale,
   signed versus positive area, normalization, quantile-grid extremes,
   control-reserve size, perturbation, or marker-ranking versus FDR selection?
5. What is the smallest credible benchmark that establishes added large-cohort
   subset utility while fairly representing native DESeq2/edgeR and patient
   outlier methods? What precision in estimated FDR should be required?
6. Can useful patient enrichment be described without promising identifiable
   mixture prevalence or patient-level p-values? Which descriptions should
   be mandatory for a biomarker discovery result?
7. Are the reference, filtering, exchangeability and multiplicity assumptions
   sufficiently explicit? What essential failure mode is currently absent?
8. For deferred uncertainty work, what inferential object should a patient
   kernel represent, and which location/mode constraints should be relaxed if
   they conflict with calibrated uncertainty and coherent library mass?

## Appendix A. Reproducible calculation for the clean subset control

For the 250:250 toy, the pooled values contain 225 low observations at 50,
250 middle observations at 100, and 25 high observations at `100f`. In a random
case group of size 250, let l and h be the numbers of low and high observations.
Its label probability is

```text
Pr(l,h) = choose(225,l) choose(250,250-l-h) choose(25,h) / choose(500,250).
```

The linear upward area for this allocation is

```text
[50 max(225-2l,0) + (100f-100) max(2h-25,0)] / 250.
```

The observed allocation has l=225 and h=25. Sum `Pr(l,h)` over allocations
whose area is at least the observed area. For log scale, replace the two gaps
by `log2(101)-log2(51)` and `log2(100f+1)-log2(101)`. This enumerates category
counts with their correct combinatorial weights, rather than trying to list
every individual label permutation.

## Appendix B. Evidence and reproducibility

This review describes a working tree based on commit `d59bb0d` with uncommitted
development files. **That commit alone does not contain the reviewed state.**
No production method was changed during this review. The new diagnostic is
exploratory and is not counted as independent confirmation of the chosen method.
The 175-test suite passed during this review; the diagnostic additionally checks
its exact toy against exhaustive enumeration.

Relevant repository files:

- [Public method](method.md), [normalization contract](normalization-standard-practice.md),
  [manual](manual.md), [release plan](../PLAN.md), [issues](../ISSUES.md).
- [Point comparison protocol](../research/protocols/point-comparison.json),
  [generator](../research/wade_research/point_simulation.py),
  [runner](../research/compare_points.py),
  [native comparator](../research/point_comparison_native.R),
  [comparison audit](../research/review_points.py).
- [Subset-failure diagnostic](../research/audit_subset_failure.py),
  [research instructions](../research/README.md),
  [TCGA examples](tcga-examples.md), [notebook builds](notebook-builds.md).

Saved local artifacts, excluded from Git/distributions:

| Directory under `.release/0.2.0/` | Contents |
| --- | --- |
| `point-comparison-20260925/` | Complete 120-matrix inference, raw/latent inputs, protocol, source snapshots, hashes. |
| `point-comparison-20260925-review-v3/` | Audited comparisons, model-stratified summaries and unapproved figures. |
| `biomarker-objective-review-20260926/` | Post-hoc up-only-family recount, kept separate from original inference. |
| `subset-failure-audit-20260926/` | New independent area checks, selected fixed permutations/oracle comparisons, exact toy, protocol and hashes. |

The full study can be regenerated from the protocol and source without private
patient data; the diagnostic requires the saved or regenerated study first:

```sh
PYTHONPATH=src:research python research/compare_points.py \
  --output .release/0.2.0/point-comparison-new --workers 4
PYTHONPATH=src:research python research/audit_subset_failure.py \
  --source .release/0.2.0/point-comparison-new \
  --output .release/0.2.0/subset-failure-audit-new
```

Native comparator reproduction additionally requires the documented R packages;
the subset diagnostic itself uses Python/NumPy/SciPy/pandas and does not run R.
The scripts refuse to overwrite an existing output directory.

Selected SHA-256 fingerprints fix the reviewed versions:

```text
src/wade/api.py
999902b45c80712ed45c112bc2354491d3bf46246a1d18df581162dcd3fa0667
research/protocols/point-comparison.json
503ba11f3aeb5d2d53e363d7b62c393aaf9834d11f85ba5dd96b850ab61b1592
research/wade_research/point_simulation.py
30cdda830008e285f8fbe1139cd0e7f26b50b4247af25bd544d8a0660035a831
research/audit_subset_failure.py
59a662a2d99ae4ef713ebf8f1d9a6e11d2f5010770e0c551611df4554bdab727
point-comparison-20260925/checksums.json
655713f82d5b32edd997c1aa7c5221ad6862acee3489afd983de864671d50790
Inspected historical probe_panel_v06.Rmd (external repository)
3136e5bcbd3cfe78be73399bf870202e35ff629fc4a738501ab0cf124b2346ad
```

This document is self-contained for conceptual critique. Reproduction or a code
audit also requires the cited source snapshot and artifacts; the external
historical cohort and all past successful HITLIB configurations are not part
of this assessment. Figures and final scientific results remain pending the
author's approval.
