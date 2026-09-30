# Validation

What the release procedure was measured to do, on what, and where it stops
working. Every number here comes from a script in `research/`; their outputs
are kept in `.release/0.2.0/`, which is not distributed.

## Confirmation on fresh simulated transcriptomes

`research/confirm.py` ran the public call, `wade.wade(direction="both",
scale="linear")` with its defaults (`pvalues="auto"`, `relative_error=0.2`,
the 999,999-shuffle budget), on 120 transcriptomes of 16,384 genes generated
after the procedure was frozen ([protocol](../research/protocols/confirmation.json),
version 6). Twenty matrices per cell: complete nulls and mixed matrices at
250:250, 500:50 and 50:500 cases:controls, half Gamma–Poisson and half
correlated lognormal–Poisson biology. Mixed matrices plant 15 patterns at
baseline means 0.5, 5, 50 and 500 counts (60 changed genes) among unchanged
genes and shared-tail null genes. A two-sided Wilcoxon rank-sum, with the same
normalized counts, test mask and BH family, is the comparator for the mean
test. A median matrix took 29 s on a laptop shared with other jobs; the mean
test was shuffled for a median of one gene per matrix and computed for the
rest.

**Calibration** at q ≤ 0.05. A mean-test call counts as false on an unchanged
gene; a subset call counts as false on a gene with no planted subset,
including whole-cohort shifts.

| Cases : controls | Mean test: complete nulls with any call | Mean test: false discovery proportion (calls) | Wilcoxon | Subset test: complete nulls with any call | Subset test: false discovery proportion (calls) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 250 : 250 | 1 of 20 | 5.1% (24.5) | 5.1% (20.6) | 1 of 20 | 5.2% (29.7) |
| 500 : 50 | 2 of 20 | 3.8% (1.1) | 5.0% (6.0) | 0 of 20 | 7.0% (12.1) |
| 50 : 500 | 0 of 20 | 3.3% (13.8) | 4.5% (7.2) | 1 of 20 | 3.6% (23.2) |

No whole-cohort shift was called in the wrong direction, and no planted subset
on the wrong side. The protocol's earlier runs are the record of two defects
this confirmation caught and the fixes it checks: without count jitter
(version 3) the subset test called a false subset in 9 of 20 complete nulls at
500:50 ([below](#the-subset-test-and-jitter)); and version 5, whose
saddlepoint lost precision when a total sat within a hair of its average,
called a false mean shift in 5 of 20.

**Power**: share of changed genes, baseline mean ≥ 5 counts, called by the
subset test and in the intended direction by the mean test and Wilcoxon (the
share Wilcoxon called in the opposite direction in brackets):

| Pattern | Design | Subset test | Mean test | Wilcoxon |
| --- | --- | ---: | ---: | ---: |
| Every case × ½, 10% of cases × 8 / 32 / 128 / 1,024 | 250 : 250 | 100 / 100 / 100 / 100% | 0 / 65 / 100 / 100% | 0% [85–92%] |
| | 50 : 500 | 80 / 100 / 100 / 100% | 0 / 8 / 93 / 98% | 0% [30–45%] |
| | 500 : 50 | 8 / 48 / 93 / 100% | 0% | 0% [27–42%] |
| 10% of cases × 16 | 250 : 250 | 100% | 97% | 2% |
| | 50 : 500 | 87% | 67% | 0% |
| 20% of cases × 4 | 250 : 250 | 52% | 78% | 18% |
| Same mean, wider spread | 250 : 250 | 85% | 0% | 60% |
| 20% × 4 and 20% × ¼ | 250 : 250 | 77% | 33% | 0% |
| Every case × 1.5 / × ⅔ | 250 : 250 | no subset | 90 / 90% | 88 / 83% |

A raised tenth above a halved majority is the pattern WADE is built for: the
subset test found 80–100% of these genes at every fold from 8× at 250:250 and
50:500, where the mean moves little until the fold is large, and Wilcoxon
called most of them down. No design detected 5% of cases × 4, 10% of cases × ¼
or a single case × 16 in more than 3% of genes: the first two are near the information
limit, and the lowered subset is the linear scale's weakness (see
[subsets](#subsets)). With 50 controls against 500 cases the subset test needs
a large fold, because the 50 controls must stand in for the whole case
distribution once the shift is removed.

## The computed mean-test p-value

`research/pvalue_accuracy.py` simulates one transcriptome per design with the
confirmation's generator, multiplies every case by 2–8 in a quarter of the
genes and one or two cases by 20 in a tenth of them (lumpy totals, where a
saddlepoint strains), and compares the computed two-sided p-value of up to ten
genes per decade of p, from 0.1 to 10⁻⁶, with the exact permutation p-value:
every labelling where a design has at most 3 million, otherwise 20 million
random labellings. 771 genes were checked across nine designs, from 3 v 52 to
445 v 10, on both scales.

| Estimated error | Genes | Median actual error | Actual error beyond 0.67–1.5× |
| --- | ---: | ---: | ---: |
| below 0.05 | 365 | 1% | 0% |
| 0.05–0.1 | 210 | 2% | 0% |
| 0.1–0.2 | 113 | 6% | 1% |
| 0.2–0.5 | 65 | 22% | 26% |
| above 0.5 | 18 | 43% | 78% |

The estimate tracks the error, so the default `relative_error=0.2` keeps the
computed p-value where it is accurate and shuffles where it is not: it kept
688 of the 771 genes, of which 0.3% lay beyond 0.67–1.5× of the exact value;
37% of the genes it sent to shuffling would have. The rule kept 99–100% of
genes at 445 v 5, 445 v 10 and 50 v 500, and 68–83% in the smallest designs
(3 v 52, 5 v 5, 8 v 8, 12 v 12), where the estimate, which includes the exact
distribution's granularity, sends more of the tail to shuffling.

## The subset test and jitter

`research/subset_calibration.py` runs the subset test (99,999 shuffles) on five
complete-null transcriptomes per design, on count / size_factor as it stands
and with WADE's jitter, and reports observed over expected p-values by mean
count:

| Design | Mean count | Without jitter, p ≤ 0.01 / 0.001 | With jitter |
| --- | --- | --- | --- |
| 500 : 50 | below 5 | 1.24× / 1.78× | 0.96× / 0.73× |
| 500 : 50 | 5–50 | 0.94× / 1.06× | 0.97× / 1.33× |
| 500 : 50 | 50 and above | 0.99× / 1.01× | 0.98× / 1.05× |
| 50 : 500 | any | 0.33–1.00× | 0.63–0.99× |
| 250 : 250 | any | 0.78–1.13× | 0.69–1.17× |

Without jitter, low-count genes are anti-conservative when cases outnumber
controls: zero counts cannot move with the mean shift, but the null moves them.
In the confirmation run without jitter (protocol v3) this put a false subset
call in 9 of 20 complete-null matrices at 500:50 and a mean false discovery
proportion of 13.0% among subset calls; with jitter (protocol v4) the same
design gave 1 of 20 and 5.6%. At p ≤ 0.001 each ratio above rests on 19–27
expected genes.

## Subsets

`research/subset_recovery.py` regenerates the confirmation's 60 mixed matrices
and scores every planted gene with baseline mean ≥ 5 counts: how often the
split chose the planted side, the share of cases selected, precision, recall,
and the subset test at 999 shuffles. On the linear scale at 250:250:

| Planted pattern | Side right | Share | Precision | Recall | Subset p ≤ 0.05 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Every case × ½, 10% × 8–1,024 above | 100% | 0.09 | 0.97–1.00 | 0.87–0.95 | 100% |
| 10% of cases × 16 | 100% | 0.09 | 0.97 | 0.88 | 100% |
| 20% of cases × 4 | 100% | 0.17 | 0.77 | 0.62 | 97% |
| 5% of cases × 4 | 97% | 0.12 | 0.39 | 0.66 | 47% |
| 10% of cases × ¼ | 25% | 0.25 | 0.07 | 0.22 | 17% |
| Every case × 1.5 or × ⅔ | no subset | | | | 5–7% |

- Raised subsets above a halved majority are found on their side, nearly
  exactly, in all three designs (precision 0.96–1.00, significant in 97–100% of
  genes).
- A change every case carries is called a subset at the null rate.
- On unchanged genes the split takes 15–21% of the cases and the subset test
  is significant in 4.7–5.1%, on either scale and in every design.
- Lowered subsets are the linear scale's weakness: the split chose the lowered
  side in 25–28% of genes, against 65–72% on the log2 scale, where the test was
  also more often significant (20–43% against 10–17%).
- The log2 scale selects more cases around a raised subset (precision
  0.53–0.97 where linear has 0.96–1.00).

`research/subset_significance.py` plants a share of cases raised by a fold in
250:250 Gamma–Poisson genes (200 per row, 999 shuffles). The subset test is
calibrated under the nulls (3–5% for no change and for every case doubled or
halved; 0–8% when every case carries the fold), and its significance falls as
the subset approaches the whole cohort: with tight biological spread, 100% of
genes at 5–80% of cases, 60% (×4) and 100% (×100) at 95%; with realistic
spread, 64–100% from 5% to 80% of cases, 20–29% at 95%. Past half the cases the
smaller side is the unchanged remainder, reported as a subset below the mean
shift.

## On TCGA tumours

The example notebooks run the default procedure on independent TCGA patients
and compare each subset with annotations WADE never sees: fusion calls for the
prostate ETS genes, and GISTIC high-level amplification for ERBB2 and GRB7,
which share an amplicon. Only tumours annotated positive or negative enter
precision and recall.

| Gene | Mean shift q | Subset | Subset q | Positives in subset | Negatives in subset | Precision | Recall |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| ERG (prostate, 445 v 52) | 1.8 × 10⁻⁴ | 38% above | 0.046 | 122 of 130 | 0 | 1.00 | 0.94 |
| ETV1 | 0.20 | 8% above | 4.0 × 10⁻⁴ | 16 of 17 | 0 | 1.00 | 0.94 |
| ETV4 | 0.53 | 3% above | 1.6 × 10⁻⁴ | 8 of 9 | 0 | 1.00 | 0.89 |
| FLI1 | 0.66 | 1% above | 0.66 | 2 of 2 | 0 | 1.00 | 1.00 |
| ERBB2 (breast, 971 v 112) | 0.0024 | 10% above | 5.9 × 10⁻⁵ | 86 of 107 | 9 | 0.91 | 0.80 |
| GRB7 | 0.016 | 9% above | 1.5 × 10⁻⁵ | 72 of 107 | 9 | 0.89 | 0.67 |

ETV1 and ETV4 are the case WADE is built for: fusion-driven expression in a
few tumours, a mean shift no test calls, and a subset that holds only
fusion-positive tumours. ERBB2 and GRB7 are raised in most breast tumours; the
mean shift carries that broad rise, and the subset is what departs from it,
the amplified tumours.

Across the transcriptome, the prostate defaults called 7,877 mean shifts up and
6,891 down, and 5,679 subsets above the mean shift and 414 below, among 24,246
tested genes. In breast, 19,952 of 25,328 genes have a subset at q ≤ 0.05: with
971 tumours, purity and subtype make almost every gene's tumours depart from a
uniform shift, so the subset test rejects nearly everywhere, and the subset's
fold and share, not its q-value, rank the genes. Such call sets need review of
tissue mixture and technical effects, which marker recovery does not provide.
The breast analysis took 89 minutes on a 16-core laptop shared with other
jobs, nearly all of it the subset test spending its full budget on genes whose
subsets were all but certain.

## Limits of this evidence

- The confirmation is simulated, with two biological models and independent
  exposures. Batch or depth confounded with the groups is not simulated.
- Twenty matrices per cell bound the mean false discovery proportion to within
  about ±2–3 percentage points.
- DESeq2/edgeR and outlier statistics were not refitted on the confirmation
  matrices.
- Real-data behaviour is illustrated by the TCGA notebooks, where no complete
  truth exists.
