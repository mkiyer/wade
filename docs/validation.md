# Validation

What the procedure was measured to do, on what, and where it stops working.
Each section names the sources of its numbers, which are of three kinds:
- the outputs of the scripts in `research/`, or, for research since retired,
  of the scripts at commit `e4fd67f`, kept in `.release/0.2.0/`;
- the confirmation's log, `.release/0.2.0/confirmation-v7.log`, which ends
  with the `--evaluate` output;
- the TCGA notebooks as rendered on 25 October 2026, and their tables in
  `notebooks/results/`.

None of these outputs is distributed.

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
  - a depth q for every sample: 1, 1/8 or 1/64 of the original libraries;
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
  so the median sample keeps 29.5% of its reads.
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

## Decision-based stopping

`research/stopping.py` (`stopping-4draws-20261025.txt`) runs the TCGA
examples' independent designs (prostate, 445 tumours v 52 normals; breast,
971 v 112) two ways:
- with `fdr=0.05`, where each gene's shuffles stop once its BH decision is
  settled;
- with `fdr=None`, where every run finishes.

**Exactness on a random 3,000-gene family.** In both cohorts and both tests:
- the decisions were identical at BH levels 0.05, 0.1 and 0.2;
- every gene not called had exactly its finished p-value and q-value;
- every called gene reported a p-value at least its finished one, and a
  q-value of at most 0.05;
- no gene used more shuffles than its finished run.

| Cohort | Called (area / subset) | Subset-test shuffles, finished | With `fdr=0.05` | Wall time, finished → `fdr=0.05` |
| --- | ---: | ---: | ---: | ---: |
| Prostate | 1,746 / 697 | 143 million | 2.1 million (67× fewer) | 144 s → 7 s |
| Breast | 1,975 / 2,406 | 1.52 billion | 1.54 million (987× fewer) | 4,479 s → 12 s |

The area test is mostly computed by its saddlepoint, so its few shuffled
genes cost little either way.

**The whole transcriptome with `fdr=0.05`:**
- **Prostate** (24,246 genes): 59 s. The subset test used 17 million
  shuffles, at most 2,165 for any gene.
- **Breast** (25,328 genes): 158 s. The subset test used 13 million shuffles,
  at most 628 for any gene: 25 exceedances over BH's threshold of about 0.040.

**The whole prostate transcriptome, both ways**
(`stopping-whole-prostate-4draws-20261025.txt`):
- the decisions were identical in both tests at 0.05, 0.1 and 0.2, over all
  24,246 genes;
- every gene not called kept its finished p- and q-value;
- the subset test used 63× fewer shuffles;
- it took 90 s against 1,656 s, beside another project's jobs.

Timings come from a 16-core laptop shared with other jobs. The breast runs
overlapped another project's jobs, so their wall times are inflated; the
shuffle counts are not.

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
changes leave RLE and the area test unaffected.

The subset test has not been measured here since it began thinning. As it
stood at commit `e4fd67f`, the rescaled test was unaffected in every row
(3.0–5.4% at 0.05; `count-model-composition-20260930.txt`), because a global
fold does not move the Lorenz gap.

## The subset test under the count model

Since 25 October 2026 the subset test thins rather than rescales, and
averages its walk over four thinning draws (see
[method](method.md#the-subset-test)). It is measured on the count model's
reference: every 4th gene of 48 data sets, from TCGA prostate tumours split
at random per gene and thinned binomially by design.
- **Four draws:** the shipped test. **One draw:** the thinned test as first
  built. Both are the package's own (`research/subset_pooled.py`,
  `subset-pooled-native-{grid,bimodal,power}-20261025.txt`).
- **Rescaled:** the test before, with the cases carried back by the fold on
  unthinned counts. **Exact:** the true fold. Both are from
  `research/subset_range.py` (`subset-range-20261025.txt`), on the same
  data sets.

The 48 data sets cross three designs (222:223, 40:405 and 405:40) and two
depths (1 and 1/8) with eight arms:
- no fold;
- every case 2×, 4× or 0.5×;
- a 2× depth gap either way;
- the TCGA prostate design's own imbalance;
- a fold of 2 with a depth gap of 0.5.

**Worst share at p ≤ 0.01** over the 48 data sets, by the sparser group's
expected reads per sample:

| Construction | 0.5–2 | 2–8 | 8–32 | 32–128 | ≥ 128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Four draws (shipped) | 88.8% | 29.4% | 2.42% | 1.48% | 1.78% |
| One draw | 74.5% | 15.4% | 1.75% | 1.79% | 1.94% |
| Rescaled (before 25 October) | 61.2% | 17.3% | 3.69% | 2.56% | 1.30% |
| An exact test (the true fold) | 1.56% | 1.75% | 1.70% | 1.81% | 1.34% |

- **At 32 reads and above the shipped test is within chance.** With about a
  thousand genes per cell, an exact test's worst of 48 cells is about 1.8%.
  32 is the validated range: a measured bound, not a parameter.
- **Depth no longer reads as a subset.** With no fold and a 2× depth gap, the
  rescaled test reached 22% below 0.5 reads and 2.6% at 32–128. With every
  sample thinned to the smallest size factor, every cell of the shipped
  test in those arms lies between 0.4% and 2.1%, most near 1%.
- **Below 32, a fold carried by every case reads as a subset,** more so with
  four draws than with one (the table's first two rows). The fold is
  estimated and then held fixed, and averaging the draws sharpens the
  statistic, its error included. A valid test there would need the fold's
  estimation inside its null ([research ideas](research-ideas.md), section
  4.4).
- **The W1 fixed point** (section 4.2 there) held to 8 reads in this grid:
  worst 1.6% at 8–32. It fails on zero-inflated genes (below), so it was set
  aside.

**Zero-inflated and bimodal genes** under 2,000 random splits of the tumours
per design (20:20, 10:40, 40:10 and 100:100), at p ≤ 0.05 / 0.01 / 0.001
(section `bimodal`):

| Gene | Zero in | Four draws (shipped) | W1 fixed point |
| --- | ---: | --- | --- |
| GSTM1 | 32% of tumours | 4.75–5.80% / 0.65–1.25% / 0.00–0.25% | 3.95–5.85% / 0.30–1.20% / 0.00–0.20% |
| ERAP2 | 0% | 4.30–5.65% / 0.30–1.35% / 0.00–0.20% | 4.15–5.55% / 0.40–1.35% / 0.00–0.20% |
| PSPHP1 | 50% | 4.75–6.35% / 1.10–1.25% / 0.10–0.30% | 5.05–9.35% / 0.60–4.10% / 0.00–1.20% |
| UGT2B17 | 24% | 3.80–5.15% / 0.55–1.00% / 0.00–0.15% | 3.45–5.45% / 0.35–0.90% / 0.00–0.10% |

**Power** (section `power`: 222:223, a tenth of the cases at 8× or a fifth at
4×, at depths 1 and 1/8; share called at p ≤ 0.01):
- At 32 reads and above, four draws, one draw and the rescaled test call
  within 1 point of one another.
- At 8–32, four draws are within 1.5 points of the rescaled test, and up to
  3.4 points above one draw.
- Below 8, four draws recover most of what one draw lost: at depth 1/8, a
  tenth at 8× is called in 66% of genes at 0.5–2 reads, against 47% with
  one draw and 70% rescaled. Neither the rescaled test nor the shipped one
  is calibrated there.

The small-sample residual on a simulated bimodal null
([research ideas](research-ideas.md), section 4.9) was measured on the
rescaled test and has not been measured again.

**The thinning draw on TCGA prostate** (445 tumours v 52 normals, 24,246
tested genes, BH q ≤ 0.05). Seeds 1 and 2 draw the jitter, the thinning and
the shuffles. The share of the genes either seed calls that both call, by the
lower group's mean reads:

| Run | Calls (seed 1 / seed 2) | All | 0–8 | 8–32 | 32–128 | ≥ 128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Four draws (shipped) | 5,598 / 5,575 | 81.0% | 57% | 57% | 75% | 90% |
| Eight draws | 5,743 / 5,741 | 84.2% | 65% | 66% | 80% | 91% |
| One draw (25 October) | 4,936 / 4,898 | 70.3% | 39% | 36% | 58% | 81% |
| One draw, `relative_error=0.1` | 4,949 / 4,848 | 71.4% | 38% | 37% | 58% | 83% |
| One draw's pool, each seed's shuffles | 4,936 / 5,005 | 92.8% | 88% | 92% | 92% | 94% |
| Rescaled test (before 25 October) | 5,793 / 5,909 | 92.6% | 86% | 91% | 93% | 94% |

Sources:
- four, eight and one draw: `subset-pooled-native-seeds-20261025.txt`
  (`research/subset_pooled.py`, section `seeds`). Every run finished at
  9,999 shuffles, which leaves the calls those of the default budget: the
  four-draw calls are `wade()`'s.
- the last three rows: `subset-seeds-20261026.txt`
  (`research/subset_range.py`, section `seeds`, every run finished), which
  also reproduces the one-draw row exactly.

- **One draw's thinning moved the calls.** With one thinned pool, a new
  shuffle stream changes 7% of them, as it did for the rescaled test. A new
  thinning changed 30%, and 19% at 128 reads and above, where the area
  test's calls agree on 97%. More precise p-values did not help: 100
  exceedances instead of 25 agree on 71%.
- **Four draws halve the disagreement.** Averaging each shuffle's walk over four draws
  takes the agreement at 128 reads and above from 81% to 90%; eight take it
  to 91%. New shuffles of one pool, at 94%, are the floor at 25 exceedances.
- **Why.** The median sample keeps 29.5% of its reads, and the higher group
  loses more to the fold. Which reads are kept is random, and near BH's
  threshold one draw decided the call (ETV4:
  [On TCGA tumours](#on-tcga-tumours)).
- **The subset itself barely moves.** Its side, share and fold are read on
  jittered normalized counts, not thinned ones, so they move with the seed
  only through the jitter. Rarity's Spearman correlation between the seeds
  is 1.000 ([rarity](#rarity)).
- **Cost.** Each draw is one walk per shuffle: the four-draw subset test took
  4.2 times as long as one draw.

## Subsets recovered in simulation

`research/subset_recovery.py` (`subset-recovery-4draws-20261025.txt`)
regenerates the 60 mixed matrices of the protocol-version-6 confirmation
(Gamma–Poisson and lognormal–Poisson biology; baseline means 0.5, 5, 50 and
500 counts). It scores every planted gene with baseline mean ≥ 5 counts:

- how often the split chose the planted side;
- the share of cases selected;
- precision and recall;
- the subset test at 999 shuffles.

At 250:250:

| Planted pattern | Side right | Share | Precision | Recall | Subset p ≤ 0.05 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Every case × ½, 10% × 8–1,024 above | 100% | 0.09 | 0.97–1.00 | 0.87–0.95 | 100% |
| 10% of cases × 16 | 100% | 0.09 | 0.97 | 0.88 | 100% |
| 20% of cases × 4 | 100% | 0.17 | 0.76 | 0.62 | 95% |
| 10% of cases × 4 | 100% | 0.12 | 0.60 | 0.62 | 78% |
| 5% of cases × 4 | 97% | 0.12 | 0.39 | 0.66 | 38% |
| 10% of cases × ¼ | 27% | 0.24 | 0.08 | 0.23 | 17% |
| Every case × 1.5 or × ⅔ | no subset | | | | 3–10% |

- **Raised subsets are found on their side, nearly exactly, in all three
  designs** (250:250, 500:50 and 50:500). A tenth of the cases raised 8× or
  more, alone or above a halved majority, is recovered with precision
  0.96–1.00 and significant at p ≤ 0.05 in 98–100% of genes. At 500:50 few
  pass BH over a whole transcriptome
  ([confirmation](#confirmation-protocol-version-7)). The split and its
  scores read normalized counts, not thinned ones.
- **Modest subsets are called less often.** At 250:250 a tenth of the cases
  at 4× is called in 78% of genes; at 500:50 and 50:500 a fifth at 4× is
  called in 72% and 78%. The thinning costs power at low counts, and these
  matrices include baseline means of 5 counts.
- **Lowered subsets are the test's weakness.** The split chose the lowered
  side in 27–32% of genes.
- **On unchanged genes** the split takes 15–16% of the cases, and the subset
  test is significant in 4.8–5.1%.
- **A change every case carries** is called a subset in 2–12% of genes across
  the designs.
  - The 12% is ×1.5 at 500:50: 20% at a baseline of 5 counts, 15% at 50 and
    0% at 500. At 250:250 ×1.5 is 10%: 15%, 15% and 0%.
  - Each baseline cell holds about 20 genes, too few to measure calibration.
    The count-model grid above measures it.

`research/subset_significance.py`
(`subset-significance-4draws-20261025.txt`) plants a share of cases raised
by a fold in 250:250 Gamma–Poisson genes (mean 50 counts, 200 genes per row,
999 shuffles).

- **Under the nulls the test calls 4–6%:** for no change, for every case
  doubled or halved, and when every case carries the fold, with tight or
  realistic biological spread.
- **Significance falls as the subset approaches the whole cohort.**
  - Tight spread: 100% of genes at 5–80% of cases; at 95%, 90% (×4) and
    100% (×100).
  - Realistic spread: 62–100% from 5% to 80% of cases; 14–24% at 95%.
- **Past half the cases** the smaller side is the unchanged remainder,
  reported as a subset below the rest.

## On TCGA tumours

The prostate notebook (`notebooks/examples.qmd`, rendered 25 October 2026)
runs the defaults on 445 independent TCGA prostate tumours and 52 normals,
with 24,246 tested genes. It compares each subset with fusion annotations
WADE never sees. Only tumours annotated fusion-positive or negative enter
precision and recall. The numbers are from its tables in
`notebooks/results/prostate/`.

| Gene | log2fc | Area fraction | Area q | Subset | Subset q | Positives in subset | Negatives in subset | Precision | Recall |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| ERG | 2.42 | 0.68 | 1.8 × 10⁻⁴ | 38% above | 0.071 | 122 of 130 | 0 | 1.00 | 0.94 |
| ETV1 | 2.30 | 0.66 | 0.20 | 8% above | ≤ 0.05 (bound) | 16 of 17 | 0 | 1.00 | 0.94 |
| ETV4 | 3.96 | 0.88 | 0.54 | 3% above | 0.054 | 8 of 9 | 0 | 1.00 | 0.89 |
| FLI1 | −0.16 | −0.06 | 0.62 | 1% above | 0.82 | 2 of 2 | 0 | 1.00 | 1.00 |

- **ETV1 and ETV4 are the case WADE is built for.** A few fusion-positive
  tumours carry large mean folds (log2fc 2.30 and 3.96) that the area test
  cannot call, because a few tumours carry the whole difference. Their
  subsets, 36 tumours for ETV1 and 12 for ETV4, hold no fusion-negative
  tumour: every annotated member is fusion-positive (16 and 8).
- **With 52 controls, these small subsets sit at BH's threshold.** With
  `fdr=0.05` a called subset's q-value is a bound. With every run finished
  (`subset-pooled-native-seeds-20261025.txt`, seeds 1 and 2):
  - ETV1's subset is called under both seeds (q = 0.030 and 0.032), as in
    the notebook.
  - ETV4's is called under both (q = 0.039 and 0.031), but not in the
    notebook (q = 0.054). The notebook's genes sit in other rows, so they
    draw other thinnings.
  - With one thinning draw, the seed decided ETV4's call: q = 0.061 under
    seed 1 and 0.032 under seed 2. This is why the test averages four draws
    ([the thinning draw](#the-subset-test-under-the-count-model)).
- **ERG** is fused in 130 of the 290 annotated tumours. Its area is called, and
  its subset holds 122 of the 130 fusion-positive tumours and none of the
  negatives, at q = 0.071.
- **Across the transcriptome** the area test called 7,503 genes up and 6,765
  down at q ≤ 0.05. The subset test found 5,170 subsets above the rest of
  the tumours and 389 below. The research scripts' runs of the same design
  count a few different calls (14,266 area, 5,598 subset): their genes sit
  in other rows, so they draw other jitter and thinning.
- **These are broad call sets.** They need review of tissue mixture and
  technical effects, which marker recovery does not provide.
- **The breast notebook** (`notebooks/examples-breast.qmd`, 971 tumours v
  112 normals, 25,328 tested genes; tables in `notebooks/results/breast/`)
  also runs four subtype contrasts. Against high-level ERBB2 amplification
  (GISTIC), which WADE never sees:
  - ERBB2's subset, 10% above the rest, holds 86 of the 107 amplified tumours
    and 9 negatives (precision 0.91, recall 0.80);
  - GRB7's, on the same amplicon, holds 72 of 107 and 9 negatives (precision
    0.89, recall 0.67).
- **Breast across the transcriptome:** the subset test called 19,740 subsets
  above the rest and 395 below. With 971 tumours, purity and subtype make
  nearly every gene's tumours more unequal than its normals, so there rarity
  (`subset_rarity`), not the q-value, ranks the subsets ([rarity](#rarity)).

## Rarity

`research/rarity.py` (`rarity-4draws-20261025.txt`) runs both TCGA designs
with the defaults (seed 1, `fdr=0.05`). Each gene is ranked within its group:
the subset test's validated range (the lower group averages at least 32
reads per sample), or the genes below it. Rank 1 is the rarest, and tied
genes share the better rank. The baseline is the rank by the gap alone,
without size matching.

**Known biology.** Ranks in the validated range: 16,574 genes in prostate,
16,956 in breast.

| Cohort | Gene | Subset (tumours) | Rarity rank | Gap-alone rank |
| --- | --- | ---: | ---: | ---: |
| Prostate | ETV4 | 12 above | 1 | 1 |
| | ETV1 | 36 above | 3 | 12 |
| | ETV5 | 13 above | 80 | 72 |
| | TP63 | 125 above | 96 | 290 |
| | ERG | 170 above | 470 | 770 |
| | FLI1 | 4 above | 1,116 | 2,320 |
| Breast | KRT5 | 110 above | 403 | 358 |
| | ERBB2 | 97 above | 746 | 610 |
| | GRB7 | 83 above | 759 | 611 |
| | ESR1 | 299 above | 4,727 | 4,458 |

- **The ETS fusions are among the rarest subsets in prostate:** ETV4 first,
  tied with one tumour's BBOX1, and ETV1 third.
- **Size matching helps some genes and costs others.**
  - It lifts ETV1, the larger prostate subsets (TP63, ERG) and FLI1's four
    tumours.
  - It ranks ETV5 a little lower, and the breast ERBB2 amplicon about a
    fifth lower, because many breast genes have a subset of about a tenth of
    the tumours.
- **Several of the other 20 rarest are normal-tissue programmes,** from
  tissue in or beside the sample. Their genes share members, which
  `rare_subsets()` shows:
  - prostate: seminal vesicle (SEMG1 and SEMG2, 6 tumours in common);
  - breast: lactation (the milk proteins CSN1S1, CSN1S2AP, CSN2, LALBA and
    BTN1A1, in the same 1–2 tumours) and muscle (MYH7 and MYL1, in one
    tumour).
  - A driver shares members too: ETV1's 36 tumours include 8 of ISG15's 58.
- **Expression.** Within the validated range, rarity still rises with
  expression: Spearman 0.30 in prostate and 0.37 in breast.

**Planted subsets.** Subsets were planted into 1% of the validated genes per
replicate, by exact binomial thinning. k of the 445 prostate tumours (1, 3,
12 or 30) were raised or lowered f-fold, f from 1.5 to 32. Each host gene
keeps at least 32 reads per sample in its lower group after the thinning, so
none left the validated range. There were 8 replicates: about 41 genes per
design at folds 1.5–8 and 83 at 16–32. The rarity and gap columns give the
planted genes' median percentile in the validated group; lower is rarer. The
last column is the share the subset test called at q ≤ 0.05.

| Planted | Rarity | Gap alone | Side right | Subset test calls |
| --- | ---: | ---: | ---: | ---: |
| 1 tumour × 32, above | 14.1% | 19.7% | 100% | 29% |
| 3 tumours × 8, above | 26.1% | 29.1% | 100% | 35% |
| 12 tumours × 16, above | 2.5% | 1.9% | 100% | 28% |
| 12 tumours × 32, above | 0.6% | 0.6% | 100% | 25% |
| 30 tumours × 8, above | 2.0% | 1.7% | 100% | 57% |
| 30 tumours × 32, above | 0.1% | 0.1% | 100% | 65% |
| 30 tumours × 8, below | 16.5% | 27.8% | 50% | 50% |
| 30 tumours × 32, below | 13.6% | 26.3% | 56% | 60% |

- **Rarity lifts lowered subsets and small raised ones**, whose gaps run
  small. A lowered subset's gap is capped by its share of the cases. A raised
  subset's gap grows with its share unless the fold is extreme: 0.065 for one
  tumour at 32× over an even group, against 0.30 for 30 tumours at 8×.
- **For mid-sized raised subsets the gap alone does as well or slightly
  better.** At 12 tumours × 16 the planted genes' median percentile was 2.5%
  by rarity and 1.9% by the gap; at 30 tumours × 16, 88% reached the top 1%
  by rarity, against 93% by the gap.
- **Strong small subsets rank near the top but are not called.**
  - Any shuffle that puts none of the planted tumours among the 52 controls
    repeats the planted split. That is 89.5% of all labellings for 1 tumour,
    71.7% for 3, 26.1% for 12 and 3.3% for 30.
  - At 16–32×, subsets of 1–12 tumours were called in 12–39% of genes, no
    more often than at 1.5× (20–36%). Subsets of 30 were called in 60–65%.
  - Nearly every raised 12-tumour subset at 32× (94%) was nevertheless in
    the top 1%.
- **Lowered subsets are often reported on the raised side.** At 8–32× they
  were reported on their own side in 17–66% of genes, more often the larger
  the subset; raised subsets in 96–100%. The walk takes each gene's largest
  gap, and a lowered subset's gap is capped by its share of the cases.

**DNA alterations.** Take each validated gene that has a subset and at
least one tumour carrying its own alteration: high-level amplification for a
raised subset, deep deletion for a lowered one. Calls are GISTIC, from the
TCGA PanCancer Atlas via cBioPortal, for 436 of 445 prostate and 945 of 971
breast tumours. That leaves 10,840 prostate and 13,197 breast genes. The
relative risk is the alteration's rate among the subset's members over its
rate in the other tumours.

| Rarity rank | Prostate | Breast |
| --- | ---: | ---: |
| 1–100 | 2.3 | 1.5 |
| 101–500 | 1.3 | 1.7 |
| 501–2,000 | 1.7 | 2.3 |
| 2,001–5,000 | 2.9 | 3.0 |
| 5,001 on | 3.8 | 4.0 |

- **Rarer subsets are less often the gene's own copy-number change:** a
  relative risk of 1.3–2.3 among the 2,000 rarest in either cohort, against
  3.8–4.0 past rank 5,000. Among the genes ranked 1–100, only 17 in prostate
  and 27 in breast had an altered member. The rarest come from other
  tissues, or in prostate from the ETS fusions, which copy number does not
  see.
- Rarity ranks what is unusual in the cohort, whatever the cause. It is not a
  marker of DNA events.

**Sample load.**
- **The heaviest libraries carry another tissue's or cell type's programme,
  or an artefact.**
  - prostate: skeletal muscle (MB, XIRP1, TRIM63), gut Paneth-cell
    defensins (DEFA5, DEFA6) and immunoglobulins;
  - breast: muscle (MYH7, MYL2, CKM) and small nuclear and 7SL RNAs
    (RNU4-1, RNU4-2, RN7SL1).
- **A light contamination does not stand out.** One breast tumour's library
  was mixed into each of five prostate tumours, as 0.5–5% of its reads.
  - Each host's load rose with the share: at 5%, its rank of 445 moved from
    18, 288, 59, 330 and 362 to 12, 183, 34, 203 and 215.
  - None reached the top ten.
- **Weights.** The excess over the null mean, max(0, −log rarity − 1),
  ranked the hosts 19, 143, 25, 134 and 198 at 5%. It put three of the five
  nearer the top at 0.5–2% and four at 5%, the heaviest host further down at
  every share, and none in the top ten. The load keeps −log(rarity), which
  needs no offset.

**Stability.** Seeds 1 and 2 change the jitter, the thinning and the
shuffles. Over the 16,409 validated genes with a subset under both seeds,
rarity's Spearman correlation between them was 1.000. 101 genes ranked in
the top 100 under both (tied genes share a rank), and 998 in the top 1,000.
ETV4 and ETV1 ranked first and third under both.

## Strata

`research/strata.py` vets the construction. The record is
`strata-20261024.txt`; the subset test's thinned construction is in
`strata-thinned-20261025.txt`, and the shipped test (four draws, as `wade()`
runs it) in `strata-shipped-20261025.txt`. The constructions set aside are in
[research ideas](research-ideas.md#9-strata-constructions-set-aside).

**The data.**
- *The count model with studies.* Each gene's TCGA prostate tumours are
  split at random into studies whose case shares differ. Each study is
  thinned binomially to its depth (1 or 1/4), with a gene-specific batch
  scale per study (lognormal, sd 0.5 in the batch arms). Binomial thinning
  keeps the count model exact. There are two designs: two studies (176:44
  and 44:176), and twenty studies of 22 (15:7 and 7:15).
- *Real nulls,* each a random relabelling within strata, with case shares
  that differ between strata:
  - TCGA prostate tumours within 20 tissue source sites;
  - the tumours of the real contrast's five sites, with their real case
    shares (one site 22:1);
  - TCGA and GTEx prostate normals within study.

**Exactness.**
- **One stratum** is the unstratified procedure, bit for bit: the stream, the
  shuffled area test and the subset test.
- **Stratified shuffles** match exact enumeration of every within-stratum
  labelling on four small designs (2,475 to 109,760 labellings). The median
  deviation is 2.0–2.3%, against a standard error of 5%, and none exceeds
  three standard errors.
- **The stratified saddlepoint's error estimate** sends the inaccurate
  genes to shuffles.
  - Small designs: it kept 62–84% of genes, at most 0.9% of them beyond
    0.67–1.5× of exact.
  - Five realistic designs, against 20 million labellings: it kept 98–100% of
    267 genes, none beyond, with a median error of 0.7–2.1%. It shuffled
    0.5–4.3% of all genes.

**Calibration.** Share of null genes at p ≤ 0.01: the area test over all
genes (`strata-20261024.txt`, sections `model`, `sites`, `layout` and
`gtex`), the subset test at 32 reads per sample and above.

| Null | Area, unstratified | Area, stratified | Subset, unstratified | Subset, stratified |
| --- | ---: | ---: | ---: | ---: |
| Two studies, batch | 59.8% | 1.5% | 9.4–14.9% | 1.4% |
| Two studies, batch, every case 1.5× or 3× | | | 6.1–18.3% | 0.6–1.0% |
| Twenty studies, batch | 4.8% | 0.9% | 2.2–4.4% | 0.4–0.8% |
| Twenty studies, batch, every case 1.5× or 3× | | | 1.5–4.9% | 1.1–1.5% |
| TCGA tumours within 20 sites | 1.8% | 0.85% | 0.57–0.66% | 0.88–1.15% |
| The real contrast's five sites | 1.1% | 1.1% | 0.5–1.5% | 0.8–1.3% |
| TCGA and GTEx normals by study | 42.5% | 0.78% | 29.3–30.9% | 0.74–0.76% |

Both subset columns are the shipped test, four thinning draws, in the 32–128
and ≥ 128 reads bins (`strata-shipped-20261025.txt`). The stratified subset
test's worst cell at 32 reads and above, over every null and fold measured,
is 1.73%. Over all genes, below 32 reads included, the TCGA and GTEx null
gives the subset test 27.68% unstratified and 0.78% stratified (same file).

**Power.**
- **The area test.** Thinning within strata gains 3–8 points over thinning
  to the global minimum. Inverse-variance weights gain nothing consistent
  over the Cochran–Mantel–Haenszel weights.
- **The subset test,** with a tenth of each stratum's cases at 8×: at 32
  reads and above, the shipped construction (four draws) matches the
  rescaled one within 1 point (`strata-shipped-20261025.txt`,
  `strata-thinned-20261025.txt`).

**On TCGA prostate stratified by site** (5 of 32 sites hold both groups:
207 tumours and 52 normals; section `real`, `wade(..., strata=site)`,
`strata-real-4draws-20261025.txt`):
- **The area test** calls 13,167 genes, against 13,197 unstratified on the
  same 259 samples; 12,634 are called by both. `research/strata.py`,
  calling the package's stratified area functions outside `wade()` with
  every run finished, called the same 13,167.
- **The subset test** calls 5,033, against 4,532 unstratified on the same
  samples; 3,735 are called by both.
- **The fusion markers.** ETV1's subset is called either way (q ≤ 0.05).
  ETV4's weakens: q = 0.089, against 0.06 unstratified.

**TCGA normal-adjacent against GTEx normals**: the area test
calls 18,849 of 24,246 genes (78%), 8,060 of them at least 2-fold, and the
subset test 2,986 (12%; `strata-gtex-4draws-20261025.txt`). The two studies'
normals differ in most genes, so GTEx normals pooled as controls would
confound study with condition. Strata do not help: a GTEx stratum holds only
controls and is left out.

## Confirmation (protocol version 7)

**The build passed all 27 checks** of a confirmation on fresh seeds. Its
protocol and script were frozen at commit `74c0752` on 26 October 2026,
before any of its data existed (`research/protocols/confirmation-v7.json`,
`research/confirm.py`; outputs in `.release/0.2.0/confirmation-v7`). The
cleanup that followed removed unused API and changed only genes without
reads in the samples the tests use, which no arm tests.

- **The rule.** Each check compares an error rate with an acceptable and an
  unacceptable value:
  - per-gene null rates and false discovery proportions: the nominal level,
    and 1.5× it;
  - the chance of any discovery in a transcriptome with no change: the
    level, and 2× it;
  - the false discovery proportion of the area and subset lists combined:
    2× and 3× the level.
- **The decision.** A check fails if its estimate lies significantly above
  the acceptable value, one-sided at 0.05/27, so the 27 checks share a 5%
  chance of any false failure. Each also needed power 0.9 at the
  unacceptable value at its realized size; all had 0.905 or more.
- **The arms:**
  - *count-model nulls:* designs 222:223, 40:405 and 405:40 at depths 1 and
    1/8, crossed with eight arms of folds (½ to 4) and depth gaps (½ to 2×,
    and TCGA prostate's own), every 4th gene;
  - *TCGA genes under random labels:* prostate at 40:405, 222:223 and
    405:40, breast at 88:883, 485:486 and 883:88, with a split per gene;
  - *strata:* the two simulated-study designs with batch effects of 0 and
    0.5, TCGA prostate relabelled within its sites, the real contrast's site
    layout, and TCGA with GTEx normals, 16,000 genes per set;
  - *simulated transcriptomes* of 16,384 genes through `wade()` itself, with
    decision-based stopping: 480 with no change and 60 with 960 planted
    changes each, at 250:250, 500:50 and 50:500;
  - *biology*, descriptive: the notebooks' fusion and amplicon subsets
    ([On TCGA tumours](#on-tcga-tumours)).
- **The per-gene arms** give each gene its own labelling, which `wade()`
  cannot take, so they run `wade()`'s functions with its settings and every
  run finished. A preflight found them equal to `wade()`, with and without
  strata.
- **Cost** (`confirmation-v7.log`). About 4.2 hours with six threads on a
  shared 16-core laptop: count model 54 minutes, real labels 5, strata 24,
  and 2.9 hours of `wade()` time for the 540 transcriptomes (about 19
  seconds each).

**Per-gene null rates** (`checks.tsv`). The area test's nulls have no fold.
The subset test's are counted in its validated range (the lower group
averages at least 32 reads per sample) and, in the count-model and strata
arms, include folds carried by every case.

| Arm | Test | Null genes | p ≤ 0.05 | p ≤ 0.01 | p ≤ 0.001 |
| --- | --- | ---: | ---: | ---: | ---: |
| Count model | area | 145,488 | 5.00% | 0.99% | 0.084% |
| | subset | 158,625 | 4.93% | 0.96% | 0.086% |
| TCGA, random labels | area | 148,722 | 5.03% | 1.06% | 0.115% |
| | subset | 105,056 | 4.96% | 0.99% | 0.079% |
| Strata | area | 112,000 | 5.05% | 1.00% | 0.107% |
| | subset | 147,821 | 5.01% | 0.98% | 0.097% |

**Lists over a transcriptome** (`checks.tsv`). q is BH's adjusted p-value.
The subset
lists keep genes in the validated range, and count a call on a gene changed
in every case as false.

| Quantity | Test | Matrices | q ≤ 0.05 | q ≤ 0.1 |
| --- | --- | ---: | ---: | ---: |
| Any discovery, no change | area | 480 | 5.6% | 10.2% |
| | subset | 480 | 1.5% | 1.9% |
| False discovery proportion, changes planted | area | 60 | 4.8% | 9.3% |
| | subset | 60 | 2.2% | 5.1% |
| | both lists | 60 | 4.4% | |

- **On real genes the area test runs a little above its level at small
  p:** 1.06% at p ≤ 0.01 and 0.115% at 0.001, inside the tolerance. The
  model and strata arms give 0.99–1.00% and 0.084–0.107%.
- **Below its range the subset test is anti-conservative under a fold,** as
  measured above. In the count-model arm it rejected 21% of nulls at
  p ≤ 0.05 below 8 reads and 5.35% at 8–32. On real genes under random
  labels, with no fold, it rejected 5.44% and 5.02% (per-gene rates by
  reads, the `--evaluate` output in `confirmation-v7.log`).
- **The subset lists run below their level:** 1.5% of the transcriptomes
  with no change had any subset discovery at q ≤ 0.05.

**Power** (`power.tsv`). These are the shares of planted genes called on
their side at q ≤ 0.05, for genes with baseline means of 5 counts or more
(960 per cell). These matrices plant 16 times as many changes as protocol
version 6's, which loosens BH's cutoff, so the two versions' power is not
comparable.

| Planted change | Test | 250:250 | 500:50 | 50:500 |
| --- | --- | ---: | ---: | ---: |
| Every case × 1.5 | area | 94% | 9% | 62% |
| | Wilcoxon | 96% | 48% | 51% |
| Every case × ⅔ | area | 88% | 32% | 26% |
| | Wilcoxon | 94% | 46% | 46% |
| A tenth of the cases × 16 | area | 99% | 0% | 90% |
| | subset | 100% | 1% | 70% |
| A fifth of the cases × 4 | area | 94% | 0% | 65% |
| | subset | 72% | 5% | 33% |
| Every case × ½, a tenth × 8 | subset | 100% | 1% | 82% |
| Same mean, wider spread | subset | 92% | 38% | 53% |

- **With ten cases per control, the area test misses most modest shifts.**
  At 500:50 it called 9% of genes raised 1.5× in every case, against 62% at
  50:500 and 48% for the Wilcoxon comparator.
  - Raising every case's level also widens the cases' spread. A shuffled
    difference of means stays exact, but it loses power when the larger
    group is the more variable one, and gains it in the reverse design
    ([research ideas](research-ideas.md), section 2.4).
  - TCGA's tumour-to-normal designs are of this kind.
- **At 500:50 the subset test called almost none of the planted subsets:**
  1–2% of the raised tenths, at any fold. This is the floor seen with TCGA
  prostate's 52 controls ([rarity](#rarity)).
  - Any shuffle that puts none of the 50 raised cases among the 50 controls
    repeats the planted split: 0.67% of labellings.
  - A test valid for exchangeable labels can therefore call such a subset
    at level α with probability no more than about α/0.0067.
  - BH's cutoff in these transcriptomes was about 2 × 10⁻⁴ (53 subset calls
    of 14,480 tested genes in the first), which allows about 3%.
  - With the full budget, the planted genes' p-values sit at that chance:
    medians 0.0032–0.0072 (`research/subset_recovery.py`, section `floor`,
    `subset-recovery-floor-20261026.txt`).
  - At 250:250 and 50:500 the chance is 1.6 × 10⁻⁸ and 5.1 × 10⁻⁶. At
    250:250 the planted genes' median p-values reach the budget's limit,
    10⁻⁶ to 2 × 10⁻⁶; at 50:500 so do those at 32× and 1,024× above a
    halved majority. At 50:500 a tenth at 16× alone has a median of
    3.1 × 10⁻⁴, ranging to 0.072.
  - The rescaled test of protocol version 6 called more at 500:50 than that
    bound allows a valid test: all of a tenth at 1,024× above a halved
    majority, and 20% of a tenth at 16×
    (`confirmation-v6-20260928/power.tsv`).

Protocol version 6 confirmed the procedure as it stood before this build
(the mean test on unthinned counts and the rescaled subset test), on 120
simulated transcriptomes (`.release/0.2.0/confirmation-v6-20260928`). Its
numbers do not describe this build.

## Limits of this evidence

- The count-model reference takes its latent biology from one tissue, prostate
  tumours, and its size factors from RLE on the same tumours.
- Batch effects and changes of composition are measured only as described
  above. The subset test under a change of composition was measured only
  before it began thinning.
- The subset test's validity is measured, not proven. Its bound of 32 reads
  per sample is the coarsest statement every measured arm supports. Its
  calls near the threshold depend on the thinning draw, measured on one
  cohort.
- Strata are measured on two simulated designs with batch effects and on
  three real relabelling nulls (TCGA sites, the real contrast's sites, TCGA
  with GTEx).
- Paired designs and continuous covariates are not measured.
- DESeq2/edgeR and outlier statistics were not refitted on these data.
- Real-data behaviour beyond calibration is illustrated by the TCGA notebooks,
  where no complete truth exists.
- Rarity's evidence is two TCGA cohorts, subsets planted into one, and one
  contamination design. It ranks relative to the cohort's own genes.
