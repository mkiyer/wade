# Validation

What the procedure was measured to do, on what, and where it stops working.
Every number here comes from a script in `research/`, or, for research since
retired, from the scripts at commit `e4fd67f`. Their outputs are kept
in `.release/0.2.0/`, which is not distributed.

## The count-model reference

RNA-seq counts follow, gene by gene, the count model

\[C_j\mid\Lambda_j\sim\text{Poisson}(s_j f^{L_j}\Lambda_j),\]

with s_j sample j's depth, L_j its label (1 for a case), f the gene's fold,
and the latent levels Λ_j drawn independently from one distribution.
`research/count_model.py` builds an exact reference for it from real data.

- **The source.** The 445 TCGA prostate tumours.
- **The split.** Each gene's tumours are split at random into pseudo-cases
  and pseudo-controls. One split shared by every gene would make the share of
  genes rejected swing with that split; a split per gene makes it a binomial
  estimate of each test's size.
- **The thinning.** Every sample is thinned binomially with a probability
  fixed by the design alone, never by an estimate:
  - a depth q for every sample: 1, 1/8 or 1/64 of the original libraries
    (median 47M, 5.9M and 0.7M reads);
  - group shares with p₁/p₀ = f·d, where d is the cases' depth over the
    controls'.
- **Why it is exact.** A thinned Poisson mixture is a Poisson mixture with its
  mean scaled, so the thinned counts follow the count model exactly, with the
  tumours' own depths and biology as the latent levels.
- **What the tests see.** Each sample's size factor is RLE of the unthinned
  tumours, fitted on the other half of the genes, times d for the cases.
- **The bins.** Genes are binned by the sparser group's expected reads per
  sample: [0, 0.5), [0.5, 2), [2, 8), [8, 32), [32, 128) and ≥ 128.
- **The arms:**
  - designs 222:223, 40:405 and 405:40 at all three depths, and 20:20 at
    1/8;
  - equal depths (d = 1);
  - the TCGA designs' own imbalance, the median tumour size factor over the
    median normal one: d = 0.88 (prostate, tumours shallower) and d = 1.07
    (breast);
  - a 2× gap (d = 0.5 and 2).

## The area test

**Calibration** (`count-model-shipped-20260930.txt`, section `shipped`). The
table gives shares of null genes (no fold) rejected, as ranges over cells: at
p ≤ 0.01 by expected reads per sample, then overall at 0.05 / 0.01 / 0.001.
The test without depth equalization, the same statistic and tail on the
unthinned counts, is the contrast.

| Arm (cells) | Test | < 0.5 | 0.5–2 | 2–8 | 8–32 | 32–128 | ≥ 128 | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Equal depths (10) | area test | 0.9–1.1 | 0.8–1.2 | 0.6–1.3 | 0.8–1.3 | 0.7–1.2 | 0.6–1.4 | 4.86–5.29 / 0.92–1.08 / 0.07–0.12 |
| | unthinned | 1.0 | 0.9–1.1 | 0.8–1.3 | 0.9–1.4 | 0.7–1.3 | 0.7–1.4 | 4.87–5.19 / 0.94–1.09 / 0.07–0.13 |
| TCGA imbalance (20) | area test | 0.9–1.2 | 0.8–1.1 | 0.7–1.2 | 0.8–1.4 | 0.8–1.2 | 0.6–1.3 | 4.87–5.30 / 0.91–1.11 / 0.07–0.16 |
| | unthinned | 0.9–8.5 | 0.5–2.3 | 0.8–1.5 | 0.7–1.3 | 0.7–1.2 | 0.4–1.3 | 4.84–10.5 / 0.90–3.41 / 0.07–0.71 |
| 2× depth gap (20) | area test | 0.9–1.3 | 0.7–1.4 | 0.7–1.3 | 0.7–1.2 | 0.5–1.3 | 0.0–1.6 | 4.80–5.24 / 0.89–1.11 / 0.07–0.14 |
| | unthinned | 9.5–95.6 | 3.1–39.9 | 0.4–6.6 | 0.6–2.5 | 0.7–1.5 | 0.3–1.6 | 4.73–48.4 / 0.81–41.5 / 0.06–36.2 |

Shares are in percent, over bins holding at least 200 genes. The area test
holds its level in all 50 cells, at every expression level. Without
equalization, a group sequenced half as deep makes most null genes below
0.5 expected reads significant. Even the TCGA designs' own imbalance lifts the
unthinned test to 3.4% at p ≤ 0.01 overall at 1/64 depth.

**What equalization costs** (same section: a 1.5-fold difference, equal
depths, 10 cells). The table gives points of power lost at p ≤ 0.01 against
the unthinned test, as the median and range over cells:

| Expected reads | < 0.5 | 0.5–2 | 2–8 | 8–32 | 32–128 | ≥ 128 |
| --- | --- | --- | --- | --- | --- | --- |
| Points lost | 3.9 (0.5–25.6) | 14.5 (1.5–35.2) | 12.1 (4.4–25.6) | 4.3 (0.3–7.1) | 0.8 (0.0–2.6) | 0.1 (0.0–0.8) |

Genes with fewer than about 32 reads per sample pay for the depth
equalization. That is by design: counts there are dominated by Poisson noise
and by depth.

**On TCGA prostate with its own labels** (`count-model-real-20260930.txt`,
section `real`: 445 tumours v 52 normals, 24,246 tested genes).

- **The depth kept.** The smallest RLE size factor is 0.295 of the median,
  so the median sample keeps 29.5% of its reads. Nine tumours, and no
  normals, have size factors below half the median.
- **The calls.** At BH q ≤ 0.05 the area test calls 14,266 genes, against
  14,770 for the unthinned test. The calls lost are concentrated at low
  expression:

  | Lower group mean, reads | Genes | Unthinned | Area test (seed 1 / seed 2) |
  | --- | ---: | ---: | ---: |
  | 0–8 | 3,381 | 1,855 | 1,619 / 1,629 |
  | 8–32 | 4,291 | 2,285 | 2,135 / 2,117 |
  | 32–128 | 3,382 | 2,004 | 1,968 / 1,962 |
  | ≥ 128 | 13,192 | 8,626 | 8,544 / 8,561 |

- **The thinning draw.** Two seeds call 14,266 and 14,269 genes, 13,728 of
  them by both: 92.7% of the genes either calls.
  - Agreement grows with expression: 80% of the genes called by either seed
    below 8 reads, 86% at 8–32, 93% at 32–128 and 97% at 128 or more.
  - Among the genes either calls, the two seeds' p-values have a Spearman
    correlation of 0.971.
  - The 200 smallest p-values share 186 genes.
  - A run is reproducible from its seed. Rankings near a threshold move with
    the draw, as any permutation p-value does.

## The computed area-test p-value

`research/pvalue_accuracy.py` (`pvalue-accuracy-20260930.txt`) simulates one
transcriptome per design with the confirmation's generator.

- **The genes.** It multiplies every case by 2–8 in a quarter of the genes,
  and one or two cases by 20 in a tenth of them: lumpy totals, where a
  saddlepoint strains.
- **The values.** It thins and jitters the counts as the area test does.
- **The check.** It compares the computed two-sided p-value of up to ten
  genes per decade of p, from 0.1 to 10⁻⁶, with the exact permutation
  p-value. That comes from every labelling where a design has at most 3
  million, and otherwise from 20 million random labellings.
- **Coverage.** 383 genes across nine designs, from 3 v 52 and 5 v 5 to
  445 v 10 and 50 v 500.

| Estimated error | Genes | Median actual error | Actual error beyond 0.67–1.5× |
| --- | ---: | ---: | ---: |
| below 0.05 | 183 | 1% | 0% |
| 0.05–0.1 | 94 | 2% | 0% |
| 0.1–0.2 | 55 | 9% | 2% |
| 0.2–0.5 | 39 | 29% | 36% |
| above 0.5 | 12 | 25% | 42% |

The estimate tracks the error. The default `relative_error=0.2` therefore keeps
the computed p-value where it is accurate and shuffles where it is not.

- It kept 332 of the 383 genes, of which 0.3% lay beyond 0.67–1.5× of the
  exact value. Of the genes it sent to shuffling, 37% would have.
- It kept 98–100% of the genes at 445 v 5, 445 v 10 and 50 v 500.
- It kept 65–92% in the smallest designs. There the estimate, which includes
  the exact distribution's granularity, sends more of the tail to shuffling.

## Normalization when many genes change

Section `composition` (`count-model-composition-area-20260930.txt`):

- the design is 222:223 at 1/8 depth, with one split shared by every gene and
  4 splits;
- a share of the genes is changed 2× in one group;
- RLE is refitted on each data set, and the area test runs on every 8th
  unchanged gene;
- shares are at p ≤ 0.05 / 0.01, pooled over the splits.

| Genes changed 2× | RLE, cases over controls | Design size factors | Refitted RLE |
| --- | --- | --- | --- |
| 10%, all up | 1.051–1.063 | 6.3% / 1.2% | 19.6% / 9.8% |
| 30%, all up | 1.189–1.204 | 5.5% / 1.2% | 59.5% / 48.3% |
| 10%, half up and half down | 0.994–1.002 | 4.7% / 1.0% | 4.5% / 1.0% |
| 30%, half up and half down | 0.998–1.005 | 4.0% / 0.7% | 4.2% / 0.8% |

When many genes rise together, RLE absorbs part of the rise, and the unchanged
genes appear lowered. This holds for any test built on size factors. Balanced
changes leave RLE and the area test unaffected. The subset test is unaffected
in every row (3.0–5.4% at 0.05; tag's `count-model-composition-20260930.txt`),
because a global fold does not move the Lorenz gap.

## The subset test under the count model

The subset test is unchanged by the 30 September 2026 build except for the
removed options. Its count-model measurements ran from the scripts at commit
`e4fd67f`: sections `reference`, `laplace`, `realistic` and
`summary` of `research/count_model.py`, in `count-model-20260929.txt`,
`count-model-laplace-20260929.txt`, `count-model-realistic-20260929.txt` and
`count-model-summary-20260929.txt`.

- **With no fold and equal depths** it is calibrated at every level: 0.6–1.2%
  at p ≤ 0.01 by level, and 0.058–0.087% overall at 0.001.
- **Under a fold or a depth imbalance** it is calibrated only where the
  sparser group expects at least 128 reads per sample. The table gives the
  worst cell at p ≤ 0.01, by expected reads per sample:

  | Condition | 2–8 | 8–32 | 32–128 | ≥ 128 |
  | --- | --- | --- | --- | --- |
  | Equal depths, folds 1.5–4 (cases higher) | 2.1–8.4% | 1.3–1.4% | 0.8–1.1% | 0.9–1.4% |
  | Equal depths, fold 0.5 (cases lower) | 6.1% | 8.0% | 2.8% | 1.1% |
  | The TCGA designs' own imbalance, no fold | 2.1% | 1.9% | 1.2% | 1.4% |
  | A 2× depth gap | 19–24% | 3.0–15% | 1.6–4.5% | 1.3% |

- **Why it fails below 128.** The test registers the cases by a fold
  estimated from the same data and then holds it fixed through the shuffles.
  At low counts a fold changes Poisson noise and zeros as well as levels, and
  the extra noise among the lower or shallower cases reads as concentration.
  - It fails most when the cases are the lower or the shallower group.
  - 128 is the lowest bin edge at which every arm holds: a measured bound,
    not a parameter of the method.
  - `wade()` warns how many tested genes fall below it.
- **Zero-inflated and bimodal genes** hold under random splits at their real
  depth (`count-model-realbimodal-20260930.txt`), in the designs 20:20, 10:40,
  40:10, 50:50 and 100:100:
  - GSTM1, deleted on both chromosomes in about half of people:
    4.4–5.3% / 0.68–0.88% / 0.00–0.14%;
  - ERAP2: 4.3–5.2% / 0.48–1.32% / 0.00–0.12%.
- **Small-sample residual.** On a simulated bimodal null (two log-normal modes
  of equal weight), the test is slightly anti-conservative, falling with
  sample size ([research ideas](research-ideas.md), section 4.9):
  - at 20 v 20: 5.7–5.9% / 1.8% / 0.32–0.37%;
  - at 250 v 250: 5.06% / 1.03% / 0.114%.

## Subsets recovered in simulation

`research/subset_recovery.py` (`subset-recovery-20260930.txt`) regenerates the
60 mixed matrices of the protocol-version-6 confirmation (Gamma–Poisson and
lognormal–Poisson biology; baseline means 0.5, 5, 50 and 500 counts). It scores
every planted gene with baseline mean ≥ 5 counts:

- how often the split chose the planted side;
- the share of cases selected;
- precision and recall;
- the subset test at 999 shuffles.

At 250:250:

| Planted pattern | Side right | Share | Precision | Recall | Subset p ≤ 0.05 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Every case × ½, 10% × 8–1,024 above | 100% | 0.09 | 0.97–1.00 | 0.87–0.95 | 100% |
| 10% of cases × 16 | 100% | 0.09 | 0.97 | 0.88 | 100% |
| 20% of cases × 4 | 100% | 0.17 | 0.76 | 0.62 | 97% |
| 5% of cases × 4 | 97% | 0.12 | 0.39 | 0.66 | 48% |
| 10% of cases × ¼ | 27% | 0.24 | 0.08 | 0.23 | 22% |
| Every case × 1.5 or × ⅔ | no subset | | | | 7–8% |

- **Raised subsets are found on their side, nearly exactly, in all three
  designs** (250:250, 500:50 and 50:500). A tenth of the cases raised 8× or
  more, alone or above a halved majority, is recovered with precision
  0.96–1.00 and significant in 97–100% of genes.
- **Lowered subsets are the test's weakness.** The split chose the lowered
  side in 27–32% of genes.
- **On unchanged genes** the split takes 15–16% of the cases, and the subset
  test is significant in 4.6–4.9%.
- **A change every case carries** is called a subset in 2–10% of genes across
  the designs. Two of the three baseline levels tested (5 and 50 counts) lie
  below the validated range of 128 reads per sample, where an excess is
  expected.

`research/subset_significance.py` (`subset-significance-20260930.txt`) plants
a share of cases raised by a fold in 250:250 Gamma–Poisson genes (mean 50
counts, 200 genes per row, 999 shuffles).

- **Under the nulls** the test calls 3–5% for no change and for every case
  doubled or halved. When every case carries the fold it calls 0% with tight
  biological spread and 6–8% with realistic spread, at 50 reads, below the
  validated range.
- **Significance falls as the subset approaches the whole cohort.**
  - Tight spread: 100% of genes at 5–80% of cases; at 95%, 57% (×4) and
    100% (×100).
  - Realistic spread: 62–100% from 5% to 80% of cases; 19–28% at 95%.
- **Past half the cases** the smaller side is the unchanged remainder,
  reported as a subset below the rest.

## On TCGA tumours

The prostate notebook (`notebooks/examples.qmd`, rendered 30 September 2026)
runs the defaults on 445 independent TCGA prostate tumours and 52 normals,
with 24,246 tested genes. It compares each subset with fusion annotations
WADE never sees. Only tumours annotated fusion-positive or negative enter
precision and recall.

| Gene | log2fc | Area fraction | Area q | Subset | Subset q | Positives in subset | Negatives in subset | Precision | Recall |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| ERG | 2.42 | 0.68 | 1.8 × 10⁻⁴ | 38% above | 0.073 | 122 of 130 | 0 | 1.00 | 0.94 |
| ETV1 | 2.30 | 0.66 | 0.20 | 8% above | 0.032 | 16 of 17 | 0 | 1.00 | 0.94 |
| ETV4 | 3.96 | 0.88 | 0.54 | 3% above | 0.044 | 8 of 9 | 0 | 1.00 | 0.89 |
| FLI1 | −0.16 | −0.06 | 0.62 | 1% above | 0.67 | 2 of 2 | 0 | 1.00 | 1.00 |

- **ETV1 and ETV4 are the case WADE is built for.** A few fusion-positive
  tumours carry large mean folds (log2fc 2.30 and 3.96) that the area test
  cannot call, because a few tumours carry the whole difference. Their
  subsets hold only fusion-positive tumours and are called at q ≤ 0.05.
- **Few controls cost the subset test power.** With 52 controls these small
  subsets are called near the threshold. Under the subset test's previous
  construction, withdrawn on 29 September 2026, they reached q ≈ 10⁻⁴.
- **ERG** is fused in 130 of the 290 annotated tumours. Its area is called, and
  its subset holds 122 of the 130 fusion-positive tumours and none of the
  negatives, at q = 0.073.
- **Across the transcriptome** the area test called 7,503 genes up and 6,765
  down at q ≤ 0.05. The subset test found 5,528 subsets above the rest of
  the tumours and 407 below.
- **These are broad call sets.** They need review of tissue mixture and
  technical effects, which marker recovery does not provide.
- **The breast notebook** has not been rerun for this build.

## Earlier confirmation (protocol version 6)

Protocol version 6 confirmed the procedure as it stood before this build:

- the mean test on unthinned counts;
- the subset test's previous construction.

It ran on 120 fresh simulated transcriptomes (`research/confirm.py`, outputs
in `.release/0.2.0/confirmation-*`). Its numbers do not describe this build.
The confirmation of this build, protocol version 7, is pending
([issues](../ISSUES.md)).

## Limits of this evidence

- The count-model reference takes its latent biology from one tissue, prostate
  tumours, and its size factors from RLE on the same tumours.
- Batch effects and changes of composition are measured only as described
  above.
- The subset test's validity is measured, not proven. Its bound of 128 reads
  per sample is the coarsest statement every measured arm supports.
- DESeq2/edgeR and outlier statistics were not refitted on these data.
- Real-data behaviour beyond calibration is illustrated by the TCGA notebooks,
  where no complete truth exists.
