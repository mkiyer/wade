# Validation

What the release procedure was measured to do, on what, and where it stops
working. Every number here comes from a script in `research/`.

## Confirmation on fresh simulated transcriptomes

`research/confirm.py` ran the public call, `wade.wade(direction="both",
scale="linear")` at its default 999,999-shuffle cap, on 120 transcriptomes of
16,384 genes generated after the procedure was frozen
([protocol](../research/protocols/confirmation.json)). Twenty matrices per cell:
complete nulls and mixed matrices at 250:250, 500:50 and 50:500 cases:controls,
half Gamma–Poisson and half correlated lognormal–Poisson biology. Mixed
matrices plant 15 patterns at baseline means 0.5, 5, 50 and 500 counts (60
changed genes) among unchanged genes and shared-tail null genes. A one-sided
Wilcoxon rank-sum per direction, with the same normalized counts, test mask
and BH family, is the comparator. The whole run took 31 minutes on a
16-core laptop (median 15 s per matrix).

**Calibration** at q ≤ 0.05, counting gene-direction calls:

| Cases : controls | Complete nulls with any call | Mixed: mean false discovery proportion [95% CI] | Mean calls |
| --- | ---: | ---: | ---: |
| 250 : 250 | 1 of 20 | 5.4% [3.4, 7.4] | 27.1 |
| 500 : 50 | 1 of 20 | 2.5% [0.0, 5.9] | 1.5 |
| 50 : 500 | 0 of 20 | 4.7% [2.3, 7.1] | 15.3 |

Wilcoxon: 4.8%, 2.7% and 4.3%. No whole-cohort shift was called in the wrong
direction.

**Power**: share of changed genes called in the intended direction, at 250:250
and 50:500, baseline means 5 / 50 / 500 counts (Wilcoxon in brackets):

| Pattern | 250 : 250 | 50 : 500 |
| --- | --- | --- |
| All cases × 1.5 | 90 / 95 / 100% [85 / 90 / 95] | 15 / 35 / 25% [0 / 45 / 20] |
| All cases × ⅔ | 75 / 95 / 100% [85 / 85 / 90] | 25 / 35 / 30% [10 / 20 / 20] |
| 20% of cases × 4 | 90 / 80 / 85% [15 / 5 / 25] | 35 / 45 / 40% [0 / 0 / 0] |
| 10% of cases × 16 | 80 / 100 / 90% [5 / 5 / 0] | 60 / 65 / 55% [0 / 5 / 0] |
| Bulk × ½, 10% of cases × 32 | 95 / 100 / 95% [0 / 0 / 0] | 30 / 50 / 55% [0 / 0 / 0] |
| Bulk × ½, 10% of cases × 128 | 100 / 100 / 100% [0 / 0 / 0] | 85 / 95 / 90% [0 / 0 / 0] |
| Bulk × ½, 10% of cases × 1,024 | 100 / 100 / 100% [0 / 0 / 0] | 100 / 95 / 100% [0 / 0 / 0] |
| Bulk × ½, 10% of cases × 8 | 0 / 0 / 10% [0 / 0 / 0] | 0 / 0 / 0% [0 / 0 / 0] |

Wilcoxon called 83–92% of the raised-subset, lowered-bulk genes at 250:250
*down*. WADE detected no pattern of 5% of cases × 4, 10% of cases × ¼, or a
single case × 16 in any design. With 20% of cases × 4 and another 20% × ¼, it
called 45–60% of genes up and none down at 250:250: on the linear scale the
raised subset masks the lowered one (see the [manual](manual.md#direction-and-scale)).
A same-mean variance increase was called in at most 5% of genes; Wilcoxon
called 35–70% at 250:250, through the median shift that skewed variation
brings. At 500:50 no subset pattern was detected by either method:
that is the control floor below. At baseline mean 0.5 counts detection drops
for every pattern (for example 75% for the × 32 subset at 250:250).

## Where upward detection is possible

`research/subset_limits.py` simulates single genes (250:250, baseline mean
50, 8 genes per cell, both biologies) and asks where each statistic reaches an
upward p ≤ 10⁻⁴. Its `oracle` picks the best expression cutoff *with
hindsight*, deliberately anti-conservative: when it fails, no cutoff-type
upward test can succeed. For 10% of cases raised above a halved bulk (Gamma /
lognormal; biological CV² is the squared coefficient of variation):

| Subset fold | CV² 0.2 | CV² 0.5 | CV² 1.0 |
| --- | --- | --- | --- |
| 4× | 0 / 0% | 0 / 0% | 0 / 0% |
| 8× | 88 / 100% | 0 / 0% | 0 / 0% |
| 16× | 100 / 100% | 88 / 100% | 38 / 50% |
| ≥ 32× | 100% | 100% | 100% |
| hindsight oracle at 4× | 100 / 100% | 25 / 0% | 0 / 0% |

At 4× and CV² ≥ 0.5 even the oracle fails; only 26–57% of such a "subset"
lies above the control 99th percentile, so it is not separable from normal
variation. This is why the 4× pattern of the development benchmark (0 of 80
upward discoveries) was retired from the gate: it measured an information
limit, not the statistic. Log-scale areas detected none of these genes at 32×
at any CV², and a one-sided Wilcoxon had median p = 1.0 in every cell.

A tail statistic that searches the best top-k cutoff inside every permutation
detected more 4–16× subsets above a halved bulk (at 4×, CV² 0.2: 62% vs 0%;
at 8×, CV² 0.5: 75 / 50% vs 0%) but fewer whole-cohort shifts (below). The
linear area is the better default for the stated purpose.

### Whole-cohort shifts

The same script shifts every case (Gamma / lognormal, p ≤ 10⁻⁴):

| Shift, CV² | Linear area | Log area | Tail statistic | Wilcoxon |
| --- | --- | --- | --- | --- |
| × 1.25, 0.5 | 25 / 25% | 25 / 50% | 25 / 12% | 25 / 50% |
| × 1.5, 0.5 | 100 / 88% | 100 / 100% | 100 / 88% | 100 / 100% |
| × 1.5, 1.0 | 88 / 100% | 75 / 100% | 75 / 88% | 75 / 100% |
| × 2, any | 100% | 100% | 100% | 100% |

For whole-cohort shifts the linear area, the log area and Wilcoxon are about
equal; log and Wilcoxon were slightly more sensitive to small shifts under
lognormal biology. Under the null (no shift, 240 genes across both
biologies), 17, 20, 13 and 15 genes had p ≤ 0.05 for the four statistics; the permutation
tests are exact under exchangeability, and the confirmation's false discovery
proportions above are the calibration evidence.

## Learned subsets

`research/subset_recovery.py` regenerates the confirmation's 60 mixed matrices
from their frozen seeds and compares, for every planted gene, the cases each
walk places in the planted direction's subset with the planted cases. Share of
cases selected / precision / recall, baseline mean ≥ 5 counts, all planted
genes (60 per row):

| Planted pattern | Standardized walk (WADE) | Running sum | Cumulative area walk |
| --- | --- | --- | --- |
| 250 : 250, bulk × ½, 10% × 32–1,024 (up) | 0.10 / 1.00 / 0.95–1.00 | 0.10 / 1.00 / 0.96–1.00 | 0.11 / 0.88–0.91 / 0.99–1.00 |
| 250 : 250, 10% × 16 over unchanged bulk | 0.11 / 0.88 / 0.91 | 0.26 / 0.50 / 0.97 | 0.79 / 0.15 / 0.99 |
| 250 : 250, 20% × 4 over unchanged bulk | 0.17 / 0.80 / 0.62 | 0.35 / 0.55 / 0.84 | 0.96 / 0.21 / 0.99 |
| 250 : 250, every case × 1.5 | 0.51 / 1.00 / 0.51 | 0.58 / 1.00 / 0.58 | 1.00 / 1.00 / 1.00 |
| 500 : 50, bulk × ½, 10% × 32–1,024 (up) | 0.10 / 0.99–1.00 / 0.97–1.00 | 0.10 / 0.99–1.00 / 0.97–1.00 | 0.11–0.12 / 0.87–0.91 / 0.99–1.00 |
| 500 : 50, 10% × 16 over unchanged bulk | 0.34 / 0.54 / 0.96 | 0.38 / 0.38 / 0.98 | 0.85 / 0.13 / 1.00 |
| 50 : 500, 10% × 16 over unchanged bulk | 0.09 / 0.95 / 0.88 | 0.40 / 0.36 / 0.98 | 0.87 / 0.13 / 1.00 |

- The standardized walk is exact on raised subsets above a lowered bulk in
  every design, and precise over an unchanged bulk when controls are plentiful.
- With 50 controls it over-calls a subset over an unchanged bulk.
- For a whole-cohort shift it reports the most-shifted part (31–89% of cases
  across designs), not every case.
- On unchanged genes its peak is noise: across 10,000 unchanged genes per
  design the upward subset held 30–39% of cases on average and was empty for
  only 16–17% of genes, so subsets are read with the q-value.
- Without standardization, the running sum's peak drifts into the bulk. The
  cumulative area walk selects nearly every case whenever the rest of the cases
  match the controls, because a raised subset then lifts the whole case
  distribution.

### On TCGA tumours

The example notebooks compare each learned upward subset with annotations WADE
never sees: fusion calls for the prostate ETS genes, and GISTIC amplification
calls for ERBB2 and GRB7, which share an amplicon. Only tumours annotated
positive or negative enter precision and recall; the rest of each subset has
neither annotation (53 of ERG's 186 tumours).

| Gene | Subset | Positives in subset | Negatives in subset | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| ERG (prostate, 445 tumours) | 186 | 127 of 130 | 6 | 0.95 | 0.98 |
| ETV1 | 43 | 17 of 17 | 2 | 0.89 | 1.00 |
| ETV4 | 22 | 9 of 9 | 1 | 0.90 | 1.00 |
| FLI1 | 4 | 2 of 2 | 0 | 1.00 | 1.00 |
| ERBB2 (breast, 971 tumours) | 588 | 106 of 107 | 465 | 0.19 | 0.99 |
| GRB7 | 963 | 107 of 107 | 830 | 0.11 | 1.00 |

The fusion subsets recover the fusion-positive tumours. ERBB2 and GRB7 are
raised in most breast tumours, and the amplified tumours form a higher mode
within them. The walk places one boundary, at the broad rise, which contains
the amplified tumours but is not them: telling a high mode from other raised
cases compares cases with each other, which the walk does not do.

## How many controls

A random relabelling puts k given samples all among the cases with
probability C(n_case, k) / C(n_case + n_control, k). For a pure subset (k cases
above every control, the other cases no higher than the controls) the upward
p-value stays near or above this value:

| Cases : controls | k = 18 | k = 36 | k needed for p ≤ 2.5 × 10⁻⁶ |
| --- | ---: | ---: | ---: |
| 250 : 250 | 2.8 × 10⁻⁶ | 3.7 × 10⁻¹² | 19 (8% of cases) |
| 50 : 500 | 7.2 × 10⁻²¹ | 2.5 × 10⁻⁴⁵ | 6 (12%) |
| 445 : 52 (TCGA prostate) | 0.13 | 0.016 | 104 (23%) |
| 971 : 112 (TCGA breast) | 0.14 | 0.018 | 112 (12%) |

The values are exact arithmetic. In the TCGA prostate example, 22 tumours lie
above every normal for ETV4 (value 0.083) and its upward p-value is 0.15. ERG,
whose other tumours are raised too, reaches 10⁻⁶, although its 76 tumours above
every normal would alone give 1.1 × 10⁻⁴.

## Comparison with DESeq2 and edgeR

The development comparison (preserved in history at commit `f07d5f9`) fitted
native DESeq2 1.52.0 and edgeR 4.10.5 to 120 matrices from the same generator.
At 250:250 their mean false discovery proportion was 5.7% / 5.1% under
Gamma–Poisson biology but 28.6% / 26.4% under correlated lognormal–Poisson
biology, consistent with Li et al. (*Genome Biology*, 2022) on large human
cohorts. Their mean tests target a different question: a subset above a lowered
bulk lowers the mean. The development comparison also measured directional
linear areas against directional log areas on those matrices: 10% of cases ×16
was called up in 75–95% vs 15–30% of genes (baseline mean ≥ 5), 20% of cases
×4 in 70–80% vs 25–40%, with whole-cohort shifts about equal.

## Limits of this evidence

- The confirmation is simulated, with two biological models and independent
  exposures. Batch or depth confounded with the groups is not simulated.
- Twenty matrices per cell bound the mean false discovery proportion to within
  about ±2–3 percentage points.
- DESeq2/edgeR were not refitted on the confirmation matrices.
- Real-data behaviour is illustrated by the TCGA notebooks, where no complete
  truth exists.
