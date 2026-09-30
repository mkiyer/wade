# WADE's two tests: design and implementation plan

*29 September 2026, after three external statistical reviews, updated the
same night with the first results of step 1 (the count model, section 3.5).
This document is the handoff: it records the method as implemented,
everything measured and why, the decisions taken, the claims retracted, and
the next steps in order. Nothing described here is committed.*

**Related material.**

- **The reviewed derivation:** [subset-null-derivation.md](subset-null-derivation.md),
  kept as reviewed, with a status note.
- **The full review-by-review record** of the previous version of this plan:
  `.release/0.2.0/subset-null-plan-before-rewrite-20260929.md`.
- **Research scripts:** `research/subset_null_study.py`,
  `research/location_tests.py`, `research/welch_saddlepoint.py` and
  `research/count_model.py`.
- **Outputs:** `.release/0.2.0/` (section 8).

## 1. Scope

- **Two groups, independent samples.** WADE tests group A (cases) against
  group B (controls).
- **Out of scope for now:** strata (blocks such as batches, studies or
  patients within which labels would be shuffled), pairing, repeated measures
  and covariates. Passing `strata` is an error.
  - These come later, with covariates.
  - `notebooks/rna100k.qmd`, which blocks the RNA100K compendium by study,
    waits for that work.
- **Two tests per gene.**
  1. **A DE (location) test.** Today this is the unstudentized permutation
     test of the difference in means, with a double-saddlepoint tail. It is
     decided to become a studentized rank test (section 6, step 2).
  2. **A subset test:** is the case distribution more unequal than the
     controls' after one common fold? This is implemented (section 2).
- **0.2.0 is held** until the method is finished and confirmed (section 4).

## 2. The method as implemented

### 2.1 The subset test

Notation: $z = x + 1$ for cases and $w = y + 1$ for controls, on normalized,
jittered expression (linear scale).

- **The null.** Every case carries one common fold of the controls'
  distribution: $z \overset{d}{=} f\,w$. On the log2 scale, one common
  difference.
- **The walk.** At case boundaries $t = k/n_1$, with the mean-fold
  prediction, the walk is $R(t) = \bar z\,\{L_z(t) - L_w(t)\}$, where L is a
  group's Lorenz curve of count + 1. Checked numerically to 5.6 × 10⁻¹⁶.
- **The statistic, `lorenz_gap`.**
  - Linear scale: $\max_k \{L_w(k/n_1) - L_z(k/n_1)\}_+$, the largest amount
    by which the cases' Lorenz curve falls below the controls'.
  - Log2 scale: the same gap between absolute Lorenz curves,
    $A(t) = \int_0^t (Q - \mu)$. Checked to 3.6 × 10⁻¹⁵.
  - Either gap is exactly unchanged by a fold on the cases.
- **The cut and the subset.**
  - The cut is the highest boundary among the walk's minima.
  - The smaller side of the cut is the subset: `subset_side` "up" is the
    upper part of the group, "down" the lower.
  - Boundaries fall only between distinct values (100 ulps).
- **The registration, `alignment_log2fc`.**
  - It is the weighted median of $a(u) = \log\{(Q_1(u) + 1)/(Q_0(u) + 1)\}$
    over the merged breakpoints, with integer weights in units of
    $1/(n_1 n_0)$.
  - When the weight below a value is exactly half, the midpoint to the next
    value is taken, so swapping the groups inverts the registration exactly.
  - This is the Wasserstein-1 alignment of the log(count + 1) quantile
    functions: an alignment, not necessarily any subgroup's fold.
- **The test.**
  - Pool the cases divided by the alignment (on count + 1) with the controls.
  - Walk label shuffles of that pool directly; the gap needs no
    re-application.
  - Score the observed labelling with the same kernel.
  - Stop by Besag–Clifford: 25 exceedances at `relative_error` 0.2, or
    `max_permutations`.
- **`subsets="cases"` (default) or `"either"`.**
  - "either" also walks the controls against the cases, on the swapped grid,
    and takes the larger direction-signed score. The choice between groups is
    thereby calibrated jointly.
  - `direction` keeps its meaning. "up" asks whether the cases are higher: a
    subset of cases above the rest, or a subset of controls below the rest.
    "down" is the reverse; "both" is either.
- **The subset's fold, `subset_log2fc`.**
  $\log_2$ of the subset's mean of count + 1, over the alignment times the
  mean of the matching quantiles of the other group. It can point against the
  side in genes whose extreme values dominate both groups' means: at most 1.8%
  of TCGA genes, one of them significant (SEMG1). Read the side from
  `subset_side`.

### 2.2 Code

- **`rust/src/lib.rs`.**
  - Building blocks: `Grid` (`walk`, `below`), `totals`, `cut`, `gap`,
    `Registration::fit` (integer weights, midpoint), `describe`, and `Side`
    with its scoring rule.
  - `split(case, control, linear, both_groups, direction)` returns size,
    side, group, fold, gap and registration.
  - `score(pooled, order, linear, direction, both_groups, labels)` scores the
    observed labelling.
  - `tally(...)` walks the shuffles.
- **`rust/src/study.rs`.** Research only: `split_study` and `tally_study`, the
  old constructions that `research/subset_null_study.py` needs. Delete them
  with the study.
- **Python.**
  - `src/wade/_subsets.py`: `split`, `subset_test`.
  - `src/wade/_means.py`: the present mean test and saddlepoint (to be
    replaced).
  - `src/wade/api.py`:
    - `wade(..., subsets="cases", ...)`, where `strata` raises;
    - `WadeResult`;
    - `Subset(side, group, samples, fraction, log2fc, pvalue, padj)`.
  - `src/wade/io.py`: `RESULT_COLUMNS` is gene, status, mean_log2fc,
    mean_pvalue, mean_padj, alignment_log2fc, lorenz_gap, subset_group,
    subset_side, subset_fraction, subset_log2fc, subset_pvalue and
    subset_padj. The manifest counts calls as "case up", "case down",
    "control up" and "control down".
  - Plots draw the mean shift (dashed) and the alignment (dotted), and take
    the side from `Subset.side`.
- **Tests: 105 pass** (`tests/test_core.py`).
  - An exact rational oracle for the split, the gap and the registration, and
    the Lorenz identities.
  - The registration inverts when the groups are swapped.
  - Every p-value is exactly unchanged by a fold on the cases.
  - The two-sided test is exactly symmetric in the two groups.
  - The contraction case, cases more even than controls, is found only with
    "either".
  - The fold does not saturate.
  - A gene with one extreme sample, and a bimodal gene, are calibrated under
    400 random labellings.
  - The new columns match `split` and `Subset`; `strata` raises.
- **Build and test** (conda env `wade`, never base):

  ```
  export PATH=~/sw/miniforge3/envs/wade/bin:$PATH
  python -m pip install -e . --no-build-isolation
  PYTHONPATH=src:research python -m pytest -q
  ```
- **Stale, to rewrite after the freeze:** README, `docs/method.md`,
  `docs/manual.md`, `docs/validation.md`, `docs/changes-0.2.0.md`,
  `docs/tcga-examples.md`, and the notebooks. They describe the released
  subset test, the old columns and strata. `notebooks/demo.qmd` has only
  `s.side` updated.

## 3. What we learned

Rates are shares of null genes at p ≤ 0.05 / 0.01 / 0.001; designs are cases v
controls.

### 3.1 Why the released subset test had to go

- **The failure.** It registered by the mean fold. A few extreme values
  dominate the mean, which pushes the other cases below the controls.
- **Real genes.** On TCGA prostate (445 v 52) under random labels:
  - genes whose top sample is at least 4× the next rejected at 18× the
    nominal rate at 0.001;
  - MC4R's own values gave p ≤ 0.001 in 27% of random labellings.
- **Simulated sparse-tail nulls.** 12–20% at 5% in small designs, and 15–41%
  at 500 v 50.
- **Matched size.** Over simulated null shapes (Gamma, lognormal, 1% ×100
  and 0.5% ×500 spikes, folds 1 and 2), no p-value cut keeps the released test
  at or below 1% at the resolution p ≥ 10⁻⁴. So it has zero power at matched
  size, over these families and at this resolution; that is not a theorem.
- **Its extra power is real only on light-tailed genes.** On the
  confirmation generator at 50 v 500, BH q ≤ 0.05, with 999,999 shuffles, it
  calls 83% of "every case ×½, a tenth at 8×", against 17% (median-scaled)
  and 22% (Lorenz). That power comes from comparing the bulk with the mean
  fold, which is valid only when the mean estimates the fold well: the
  failure above. The loss is accepted (section 4).

### 3.2 The Lorenz statistic

- **The reformulation (review 2).** The depth over the case mean + 1 is
  exactly the Lorenz gap.
- **Against the median-scaled normalizer:** as valid and nearly as powerful.
  - Simulated sparse-tail grid: at most 7.6% at 5% (10 v 100).
  - Real random labels: equal to exact permutation. At 445 v 52, 3.51 /
    0.59 / 0.049 against exact 3.53 / 0.60 / 0.056.
  - Small real designs (20 v 20, 10 v 40, 40 v 10): conservative, 3.1–4.4% at
    5%.
  - Weaker on a raised tenth ×16 with few cases, for example 80% against 88%
    at 0.001 at 50 v 500.
- **It fixes the median version's real-gene failure.**
  - GSTM1 is deleted on both chromosomes in about half of people; half of
    TCGA prostate tumours sit in its lower mode.
  - There the median version rejects 1.5–1.7% at 1%, from 20 v 20 to
    100 v 100, against 0.9–1.4% for exact permutation. The Lorenz version is
    nominal.
  - ERAP2 (a common null allele) is near nominal for both.
- **What it targets.** The cases' Lorenz curve falls below the controls'
  somewhere.
  - The curves may cross. It does not show that every selected individual
    changed.
  - Moving a minority does not always raise inequality.
  - Cases more even than the controls (a contraction) give a gap of 0;
    `subsets="either"` finds that as a subset of controls.

### 3.3 The registration

- **W1 against the alternatives.** Standard deviation of the log fold at fold
  1: 0.24 at 20 v 20, 0.11 at 50 v 500, 0.07 at 250 v 250.
  - Hodges–Lehmann is as good.
  - The difference of log-medians is about 20% worse on unimodal genes, and
    erratic on bimodal ones (0.68–0.91).
- **The midpoint of the median interval** changes calibration little, and
  gives exact inversion when the groups are swapped.
- **A bug, fixed.** The study code once summed float weights, so at exact
  half splits it picked between the two ends of the median interval by
  rounding. Every point of the interval minimizes the area, so the
  conclusions stand. All review-2 sections were rerun with exact weights; the
  old outputs are in `.release/0.2.0/superseded-float-registration/`.
- **Under an alternative it is an alignment, not the majority's fold.** A
  tenth of cases far above an unchanged majority on U[1, 2] moves the
  population alignment to 28/27 (reproduced: 1.037).

### 3.4 The small-sample residual (the open calibration issue)

The bimodal null, $\log X \sim \tfrac12 N(3, 0.3^2) + \tfrac12 N(5, 0.3^2)$ in
both groups:

| Design (genes) | Lorenz, W1 midpoint | True-fold oracle |
| --- | --- | --- |
| 20 v 20 (2 × 30,000) | 5.7–5.9 / 1.8 / 0.32–0.37 | 4.7–4.8 / 1.0 / 0.09–0.11 |
| 50 v 50 (6,000) | 4.9 / 1.32 / 0.23 | 4.5 / 1.02 / 0.13 |
| 100 v 100 (6,000) | 5.2 / 1.23 / 0.20 | 5.0 / 1.05 / 0.15 |
| 250 v 250 (200,000, paired) | 5.064 / 1.030 / 0.114 | 4.982 / 0.972 / 0.0985 |

- **At 250 v 250 the excess is still there.** The paired differences are
  +0.082 / +0.058 / +0.016 percentage points, with standard errors of
  0.011 / 0.007 / 0.003 across 80 batches. At 0.1%, 32 genes were rejected by
  the package alone and 1 by the oracle alone.
- **The mechanism** (section `mechanism2`).
  - A chance excess of lower-mode cases looks like an upper subset.
  - The registration reads part of it as a fold: the log fold correlates
    0.84 with the imbalance in mode counts.
  - The pool built on that fold is less bimodal than the population: the mode
    gap shrinks from 1.96 to 1.77, and each mode widens from 0.28 to 0.36. So
    the shuffled reference is too narrow.
  - All of the excess is in the "up" direction.
- **The oracle is exact everywhere.** The statistic is fine; the registration
  is the cause.
- **Remedies tried and rejected as defaults.**
  - *A centered Lorenz bootstrap* is invalid on spikes (21% at 0.1% at
    100 v 100, 51% at 200 v 20). It never asks whether an extreme value could
    have fallen among the controls.
  - *Berger–Boos over a grid of 9 registrations* reached 6.0% at 5% on a
    sharper mixture: the grid misses the supremum. It also costs most of the
    power with few cases (85.5% → 1.5% at 0.001, 50 v 500).
  - An exact Berger–Boos must track, inside each interval of fixed pool
    order, where each shuffle's comparison with the observed gap changes.
- **Not yet tried.**
  - An exact, registration-free benchmark. Pair samples within each group,
    by a pairing fixed independently of expression, and take
    $H = |a - b|/(a + b)$ on count + 1. Permuting pair labels is then exact
    under the observed-scale null, at the cost of half the data.
  - A screened valid reference: an expensive valid p-value for screened
    candidates, and p = 1 elsewhere, which keeps validity.

### 3.5 The count model (step 1, measured 29 September 2026)

- **The null that RNA-seq data need** is the count model

  $$C_j \mid \Lambda_j \sim \text{Poisson}(s_j f^{L_j}\Lambda_j), \qquad \Lambda_j \overset{\text{iid}}{\sim} G.$$

  - A common latent fold is not a common fold of count + 1: Poisson noise and
    the pseudocount do not scale with it.
  - Unequal exposures between the groups change the noise of normalized
    values even at f = 1.
- **The exact reference** (`research/count_model.py`; outputs
  `count-model-*-20260929.txt`).
  - The 445 TCGA prostate tumours are split at random into pseudo-cases and
    pseudo-controls, then thinned binomially.
  - Each probability is fixed by the design alone: a depth q for every
    sample, and group shares with $p_1/p_0 = f d$.
  - Thinning keeps any Poisson mixture a Poisson mixture with its mean
    scaled. The count model therefore holds exactly, with the tumours' own
    depths and biology as the latent levels.
  - No estimate enters a thinning probability. The plan's first sketch, which
    thinned each sample to a target exposure, needed estimated size factors
    inside the probabilities; this design avoids that.
  - The tests get each sample's size factor times d for the cases. The size
    factor is RLE of the unthinned tumours, taken from the other half of the
    genes so that it is independent of the tested gene.
  - **A random split per gene.** One split shared by every gene makes an exact
    test's share of genes swing with that split: genes respond together to
    where a few unusual tumours land.
    - Four shared 222:223 splits gave an exact permutation test of the
      subset statistic 0.35–0.88% at p ≤ 0.01, and the rank test 0.25–0.70%.
    - A split per gene gives 1.01% and 0.93% (section `splits`).
    - With a split per gene, the share over genes is a binomial estimate of
      each test's size.
  - **Arms:**
    - designs 222:223, 40:405 and 405:40;
    - depths 1, 1/8 and 1/64 (median library 47M, 5.9M and 0.7M reads);
    - f ∈ {1, 1.5, 2, 4, 0.5} with d = 1, and d ∈ {0.5, 2} with f ∈ {1, 2}.
  - Genes are binned by the sparser group's expected count per sample:
    [0, 0.5), [0.5, 2), [2, 8), [8, 32), [32, 128), ≥ 128.
- **Everything is calibrated with no fold and equal exposures** (a random
  split): 0.6–1.5% at p ≤ 0.01 in every level bin, and 0.05–0.13% overall at
  0.001. All numbers below are ranges over cells from section `summary`, for
  bins holding at least 200 genes.
- **What fails.**
  - *The package's subset test under a fold, with equal exposures,* at
    p ≤ 0.01:

    | Expected count | < 0.5 | 0.5–2 | 2–8 | 8–32 | 32–128 | ≥ 128 |
    | --- | --- | --- | --- | --- | --- | --- |
    | Share rejected | up to 85% | 46% | 8.4% | 8.0% | 2.8% | 0.5–1.4% |

    Overall, up to 21.9% at 0.001.
  - *The subset test with unequal group exposures and no fold* reaches 24% at
    2–8 counts, 15% at 8–32 and 4.5% at 32–128. The shallower group's extra
    Poisson noise reads as concentration.
  - *The DE tests with unequal group exposures and no fold.*
    - The prototype rank test (jittered normalized values) reaches 98% below
      0.5 counts, 42% at 0.5–2, 7.0% at 2–8, 2.3% at 8–32 and 1.4–1.6% above.
    - The mean test reaches 96%, 40%, 6.6% and 2.5% in the same bins.
    - Part of the cause is the jitter, whose expected 0.5/s_j differs by
      group; the rest is Poisson noise.
- **What holds.**
  - *Thinning the higher group by the true fold, then shuffling with no
    registration,* gives 0.5–1.6% with equal exposures. With unequal
    exposures it fails like the package: up to 24% at 2–8 counts and 21% at
    8–32.
  - *Thinning every sample to its gene's smallest exposure* (by its size
    factor and the true fold, "equalized") gives 0.3–1.9% in every arm and
    0.05–0.14% overall at 0.001.
  - *The rank test after equalizing* gives 0.3–1.6%.
- **The fixed-point fold** (thin, re-register, multiply) is calibrated from
  2 counts with equal exposures. It rejects up to 26% below 0.5 counts and
  3.1% at 0.5–2. Its fold is biased at low counts: 1.91–1.92 for a true 2 at
  1 count in simulation (section `thinfixed` of the study).
- **The Laplace fold** (prototype; section `laplace`).
  - *The identity.* A sample thinned to exposure e shows no reads with chance
    $(1 - e/s_j)^{C_j}$. Its mean over a group is the latent Laplace
    transform at e·f, $E\,e^{-e f \Lambda}$, whatever G and the exposures.
  - *The estimator.* The exposures at which the cases' and the controls' mean
    zero chance reach one level have the ratio 1/f.
    - The level is 1/2, or higher where a group cannot reach 1/2 down to its
      smallest exposure.
    - The estimator is exact for any latent distribution and exactly
      reciprocal under a group swap.
    - A few extreme samples barely move it: their zero chance is near 0
      whatever the fold.
  - *Measured* (section `laplace`, 81 cells; condensed in section `summary`).
    - The fold's median error over genes, in log2, is at most 0.10 below
      0.5 expected counts, 0.016 at 0.5–2 and 0.012 above, in every cell.
    - *Thinning the higher group by it:*
      - equal exposures: 0.2–1.6% at p ≤ 0.01 by level, and 0.05–0.13% at
        0.001; conservative below 0.5 counts;
      - unequal exposures: it fails like thinning by the true fold (up to
        24% at 2–8 counts). The exposures break it, not the fold estimate.
    - *Equalizing every sample* by its size factor and the Laplace fold:
      0.1–1.9% by level in every group of arms, and 0.04–0.15% at 0.001.
- **Count-exact DE statistics** (prototypes).
  - *Zero chance:* each sample's zero chance at the exposure where the pooled
    mean is 1/2, compared by a Welch-studentized difference in means over
    shuffles. Under f = 1 every sample's mean is the same, whatever its
    exposure.
    - Over the grid it gives 0.6–1.4% by level with equal exposures.
    - With a 2× exposure gap it gives 0.1–2.2%, and up to 0.24% at 0.001
      overall: the far-tail problem below.
  - *Mid-rank scores* (section `scores`): sample j's chance that its count,
    thinned to the gene's smallest exposure, exceeds another sample's thinned
    count, plus half the chance of a tie, averaged over the other samples.
    - Under f = 1 every score has mean 1/2 whatever the exposures.
    - The scores do not change under shuffles, so the test costs what the
      mean test costs.
  - *Measured on the genes below 8 expected counts* at 1/8 depth, in the
    three designs (section `scores`; shares at p ≤ 0.05 / 0.01 / 0.001):

    | No DE (f = 1) | Jittered rank | Rank after equalizing | Zero chance | Mid-rank scores |
    | --- | --- | --- | --- | --- |
    | equal exposures | 4.4–5.5 / 1.0–1.2 / 0.06–0.13 | 5.0–5.3 / 1.0–1.1 / 0.06–0.07 | 4.5–5.2 / 1.0–1.1 / 0.08–0.18 | 4.7–5.2 / 1.0–1.1 / 0.08–0.16 |
    | one group at half depth | 27–52 / 14.5–39 / 5.4–26 | 5.0–5.4 / 0.9–1.1 / 0.08–0.12 | 4.6–5.2 / 0.9–1.1 / 0.03–0.19 | 4.7–5.2 / 0.9–1.1 / 0.04–0.18 |

    - Both count-exact statistics hold at 0.05 and 0.01 in every arm.
    - At 0.001 they reach 0.11–0.19% when the smaller group is the
      shallower one: 0.18–0.19% at 40:405 and 0.11–0.14% at 405:40. With
      the larger group shallower they give 0.03–0.11%. This is the
      Behrens–Fisher problem: with unequal variances no two-sample test of
      equal means is exact, and a studentized permutation test converges
      slowly in the far tail.
    - At equal exposures the test is exact. Its highest cell, 0.18% at
      405:40, is 16 genes against 8.7 expected (P ≈ 0.014) and is read as
      chance.
    - Power at a fold of 1.5 with equal exposures, p ≤ 0.001:

      | Design | Mid-rank scores | Zero chance | Jittered rank | Rank after equalizing |
      | --- | --- | --- | --- | --- |
      | 222:223 | 55.05% | 53.55% | 48.69% | 21.74% |
      | 405:40 | 16.59% | 14.54% | 14.16% | 4.12% |
      | 40:405 | 9.93% | 11.33% | 12.25% | 3.38% |

      At 40:405 the order reverses at 0.001 (at 0.05, 45.5% against 45.4%).
      Equalizing costs about half the power.
  - *A numerical point.* The Welch statistic is computed from matrix
    products. Its values are put on a grid of $2^{-k}$,
    $k = \lfloor (53 - \log_2 n)/2 \rfloor$, on which every sum is exact in
    any order. Without that, labellings with the same case values gave
    different statistics, ties were lost, and sparse genes rejected too
    often in the far tail. In a direct check, 2,000 such labellings gave 3
    distinct sums.
- **Power** (section `power`).
  - The subsets are planted by thinning, so the alternative is itself the
    count model. Every 4th gene is tested; the rates are shares at p ≤ 0.001
    by the controls' expected count.
  - *At 8 or more counts,* thinning by the Laplace fold matches the package
    within a point, or beats it.
    - Every pattern, design and depth. For example, 222:223, a fifth ×4 at
      full depth: 50.7% against 50.9% at 8–32 counts, 89.3% against 89.1%
      above 128.
    - With a fold on the majority, where the package's registration is
      biased, it is ahead: every case ×½ with a tenth ×8, 1/8 depth, 85.6%
      against 82.6% at 2–8 counts.
    - With few controls (405:40), a tenth ×16: 10.9% against 5.7% at 32–128.
  - *Below 8 counts* the package's own null is inflated, so raw power is not
    comparable there. At the sparsest levels a fold and a subset both mainly
    add samples with reads, and the Laplace fold absorbs part of a strong
    subset: 222:223, a tenth ×16, full depth, below 0.5 counts, 21.6% against
    68%.
  - *Equalizing every sample* costs power at low and moderate counts. For
    example, 405:40, a fifth ×4, full depth: 10.7% against 17.7% at 8–32
    counts.
- **The imbalance real designs carry** (section `realistic`).
  - The arms use d = 0.880 and 1.072: the TCGA prostate and breast designs'
    median tumour size factor over their median normal one. They run at
    222:223 and 40:405, at 1/8 and 1/64 depth.
  - *At prostate's 0.88, 1/64 depth, no fold* (222:223; overall shares at
    0.05 / 0.01 / 0.001):

    | Test | Share rejected |
    | --- | --- |
    | Package mean test | 10.5% / 3.4% / 0.71% |
    | Jittered rank test | 11.1% / 3.6% / 0.67% |
    | Package subset test | 8.4% / 2.2% / 0.26% |
    | Higher group thinned by the Laplace fold | 7.0% / 1.5% / 0.15% |

    At 1/8 depth the same arm gives the rank test 6.0% / 1.45% / 0.18% and
    the subset test 5.8% / 1.3% / 0.14%.
  - *What holds in every realistic cell:*
    - equalized thinning by the Laplace fold: 4.1–5.3% / 0.73–1.06% /
      0.05–0.14%;
    - the zero-chance test: 4.7–5.2% / 0.84–1.08% / 0.08–0.12%;
    - mid-rank scores, on the genes below 8 expected counts at 1/64 depth:
      4.5–5.1% / 0.7–1.1% / 0.05–0.11%. There the jittered rank test
      reaches 15.3% / 5.5% / 1.09%.
  - *Conclusion.* Even these small differences move every construction that
    does not equalize away from nominal at low counts.
    - With the cases shallower (prostate), the subset tests are
      anti-conservative.
    - With the cases deeper (breast), thinning the higher group by the
      Laplace fold gives 2.9–5.2% at 0.05 and 0.04–0.10% at 0.001:
      conservative or nominal.
    - The DE tests are anti-conservative either way; at breast's 1.072 and
      1/64 depth the rank test gives 6.7% / 1.65% / 0.21%.
    - Only the exposure-aware constructions hold.

- **How thinning should treat depth** (30 September 2026; sections
  `depthchoice` and `seeds`).
  - *Setting samples aside is not exact in real data.* The optimized common
    depth thins to the depth a that keeps the most reads,
    $a \cdot \#\{j : e_j \ge a\}$, and sets aside the shallower samples.
    - With no fold it is calibrated.
    - At a fold of 2 (1/8 depth) it rejects up to 11.4% / 3.5% / 0.61%,
      worst at high counts: 18.7% at 0.05 above 128.
    - The cause: the fold enters the samples' effective exposures, so the
      cut sets aside different depth ranges in the two groups. A library's
      size factor is not independent of its expression profile, since
      composition moves both, so selecting by depth selects on biology.
    - With few cases it also loses the subset itself. At 40:405, every case
      ×½ with a tenth ×8, it finds 6–18% where the others find 53–91%.
  - *Equalizing every sample* is calibrated in all 18 calibration cells,
    including the prostate design's imbalance: 4.3–5.2% / 0.58–1.02% /
    0.05–0.15%.
    - Its power cost, measured against thinning only the higher group (not
      valid once depths differ), over every planted pattern, design and
      depth:

      | Expected count | 0.5–2 | 2–8 | 8–32 | 32–128 | ≥ 128 |
      | --- | --- | --- | --- | --- | --- |
      | Median cost (points) | 17 | 6 | 1.3 (worst 12) | 0.2 | 0 |

      The worst at 8–32 counts is 40:405, a fifth ×4.
  - *The thinning draw.* Two thinning seeds on the same data give the same
    share called, within 0.2 points.
    - On planted subsets, 90–95% of the calls agree (Spearman 0.92–0.95 on
      log p).
    - Under the null the calls are noise; 8–48% agree.
    - A fixed seed makes a run reproducible, as the jitter's does.
- **Pairwise-depth scores** (section `pairwise`; every 4th gene at every
  level, 1/8 depth).
  - Each pair is compared at its own common depth: the shallower sample's
    count against the deeper one's thinned to it. That is one binomial per
    pair, so the scores run at every level.
  - *Calibrated in every arm:* 3.6–6.0% at 0.05, 0.6–1.5% at 0.01 and
    0.00–0.16% at 0.001. Under a 2× depth gap, below 8 counts, the jittered
    rank test gives 26–50% at 0.05.
  - *Power at 0.001:*
    - equal to the jittered rank test at 8 or more counts (fold 1.5,
      405:40: 79.09% against 78.95%);
    - above it below 8 counts (15.56% against 13.98%);
    - against the smallest-depth scores, ahead at 40:405 (11.56% against
      9.33% at a fold of 1.5), and slightly behind at 222:223 (54.71%
      against 56.28%) and 405:40 (15.56% against 16.40%).

- **Small designs and the estimated fold** (30 September 2026; sections
  `small`, `bimodal`, `realbimodal` and `robust`).
  - *On average over real genes the adopted constructions hold in small
    designs* (20:20, 50:50, 10:100, 100:10, six exposure arms, two depths).
    - The subset test: 0.78–1.14% at 0.01 and 0.05–0.14% at 0.001; up to
      6.0% at 0.05 at 10:100. Without jitter, slightly more conservative.
    - The pairwise DE scores: 0.80–1.32% at 0.01. One far-tail cell reaches
      0.32% at 0.001 (100:10).
  - *The adopted subset registration fails on zero-inflated and bimodal
    genes.* With the true fold, thinning is exact. The failure comes from
    estimating the fold on the same data and then shuffling as if it were
    known.

    | Gene or null | Design | Laplace fold, equalized | W1 fixed point, equalized | Oracle (true fold) |
    | --- | --- | --- | --- | --- |
    | GSTM1: 41% of tumours at ≤ 1 read, deletion | 10:40 | 19.2 / 14.3 / 7.86 | 6.40 / 1.22 / 0.26 | 5.20 / 1.00 / 0.08 |
    | PSPHP1: 56% at ≤ 1 read | 100:100 | 29.6 / 22.8 / 8.92 | 6.06 / 1.44 / 0.26 | 5.02 / 0.88 / 0.10 |
    | Review 2's bimodal null, as counts | 50:50 | 6.44 / 2.19 / 0.52 | 5.48 / 1.49 / 0.26 | 4.88 / 1.04 / 0.10 |
    | Review 2's bimodal null, as counts | 250:250 | 5.47 / 1.43 / 0.19 | 5.15 / 1.20 / 0.11 | 4.94 / 1.08 / 0.08 |

    Shares are at p ≤ 0.05 / 0.01 / 0.001, in percent, over 5,000 random
    splits of the tumours or 50,000 simulated genes.
    - The Laplace fold matches the chance of zero reads. For these genes
      that chance is set largely by how many zero-mode samples (for GSTM1,
      deletion homozygotes) land in each group, so a chance imbalance
      becomes a false fold.
    - About 289 genes in TCGA prostate have a zero mode in 5–95% of
      tumours and a positive mode with median ≥ 20 reads (59 with median
      ≥ 50).
    - UGT2B17, SOX14, SMN2 and ERAP2 are conservative or nominal.
  - *The robust registration does not rescue it.* The W1 fixed point on
    equalized counts (thin, re-register by W1, multiply) resists the modes
    better but not fully (the table above). It fails under a fold below
    about 2 expected counts:
    - 18.7% at 0.01 below 0.5 counts, at a fold of 2 and 1/8 depth;
    - 31.6% at 0.5–2 counts at a fold of 4.
  - *The tension.* At low counts only smooth functionals, such as the
    chance of zero reads, can identify a fold. Those same functionals move
    with a chance imbalance of a zero or second mode. Quantile matching
    ignores that imbalance but cannot see a fold at low counts. No
    registration estimated from the data and then held fixed through the
    shuffles is valid across these cases. The null distribution has to
    include the fold's estimation (section 6, step 1).

### 3.6 The DE test

- **The present unstudentized mean test** is exact only when the groups share
  a distribution. With equal means and unequal spreads, when the smaller
  group is the wider one, it rejects about 33% at 5%.
- **Welch studentization** (permutation distribution, not a t reference)
  brings that to 8.8–10.5%.
- **No test of arithmetic means** is uniformly valid over distributions with
  arbitrarily large rare tails (Bahadur and Savage, 1956).
- **Four permutation tests, each under its own null**
  (`research/location_tests.py`; lognormal values, 2,000 genes per cell):
  - *The studentized Brunner–Munzel rank test* (relative effect
    P(case > control) + P(tie)/2 = 1/2) holds 4.5–5.4% at 5% and 0.8–1.5% at
    1% at 50 v 500, 500 v 50 and 250 v 250.
  - *The unstudentized rank sum* gives 0.4–14.1% in the unbalanced designs.
  - *A pure subset* (a tenth of cases at 8×, no shift) is called 16–23% by
    the rank test with 50 cases and 46% with 250, since it really moves the
    relative effect to about 0.55. The present mean test calls it 98–100%.
  - *A shift* (every case ×1.5): the rank test 98–100%, as powerful as the
    mean.
- **The Welch saddlepoint prototype** (`research/welch_saddlepoint.py`,
  paused).
  - It matches exact enumeration to within 0.93–1.07 from 10⁻¹ to 10⁻³.
  - It matches 2,000,000 shuffles to within 0.88–1.18 down to 10⁻⁵.
  - Its continuation solver fails on 3% of heavy-tailed genes, where the
    minimizer switches branches. A global search along the level curve fixes
    that.
  - Kept as a reference for a fast tail for the rank test.

### 3.7 Effect semantics

- **Side and fold can disagree.** Cases [1, 1.1, 2, 199, 999] against
  controls [0, 0.1, 99, 99.1, 999]: the subset is the two largest cases (side
  up), yet its fold against the aligned controls is −0.81.
- **Folds are pseudocount-regularized.** Every linear-scale fold is of
  count + 1: (M + 1)/(b + 1), not the biological M/b.

## 4. Decisions (29 and 30 September 2026)

1. **0.2.0 is held** until the method is finished and confirmed.
2. **The subset test** uses the Lorenz statistic. Its registration was the
   midpoint W1 fold; decision 10 replaces it.
3. **Both subset tests are offered:** `subsets="cases"` (default) and
   `"either"`.
4. **The subset fold** is measured against the aligned controls.
5. **The few-case power loss** of the registered tests is accepted, not won
   back.
6. **The sequence of section 6 is adopted.** The count model and the DE test
   precede the freeze.
7. **The DE test becomes the studentized rank test** (Brunner–Munzel,
   calibrated by shuffles).
   - Its effect is reported as the alignment fold; `mean_log2fc` stays as a
     description.
   - Its column names are to be settled with the author when it is
     implemented.
   - This replaces the earlier choice of a Welch mean with a saddlepoint.
   - Decision 11 computes it under the count model.
8. **Effect columns** as in section 2.2 (the third reviewer's set).
9. **Scope: two independent groups.** Strata, pairing and covariates come
   later; passing `strata` is an error.
10. **The subset test registers by thinning** (30 September 2026, after step
    1's results; section 3.5). *Withdrawn the same day (decision 14):* the
    Laplace fold, estimated on the same data, fails on zero-inflated and
    bimodal genes (section 3.5, "Small designs and the estimated fold").
    - Estimate the Laplace fold, thin by it, and shuffle with no
      registration.
    - This replaces dividing count + 1 by the W1 fold, which fails under the
      count model below about 32 expected counts.
    - Candidate A (the bootstrap) is on hold.
11. **The DE statistic is the count-exact mid-rank score** (30 September
    2026).
    - Each sample's expected placement after thinning to a common depth,
      Welch-studentized and shuffled.
    - It is decision 7's rank test, computed under the count model.

12. **The subset test equalizes depth** (30 September 2026, after the
    measurements in section 3.5). *Superseded by decision 14:* the subset
    test no longer thins.
    - Every sample is thinned to its gene's shallowest effective depth: its
      size factor, times the Laplace fold for a case.
    - It is the only exact handling measured. Thinning only the higher group
      fails at real imbalances, and setting shallow samples aside selects on
      biology.
    - Its power cost is concentrated below 8 expected counts.
13. **The mid-rank scores compare each pair at its own common depth**
    (30 September 2026): the shallower sample's count against the deeper
    one's thinned to it.

14. **The subset test keeps the W1 registration and limits its claim**
    (30 September 2026). This replaces decisions 10 and 12.
    - The estimated Laplace fold fails on zero-inflated and bimodal genes.
      The W1 registration is robust to them (GSTM1 at 10:40: 5.3% / 0.88% /
      0.10%) but not count-consistent at low counts. The principled
      alternatives (a bootstrap, Berger–Boos) are not pursued now.
    - *The claim.* Under the count model the subset test is calibrated when
      the sparser group expects at least 128 reads per sample. Across every
      measured fold (0.5–4) and depth arm (matched, the TCGA designs' own
      imbalance, a 2× gap), its worst share at p ≤ 0.01 there is 1.4%.
    - Below 128 it is anti-conservative. The table gives the worst cell at
      p ≤ 0.01, by the sparser group's expected reads per sample:

      | Condition | 2–8 | 8–32 | 32–128 | ≥ 128 |
      | --- | --- | --- | --- | --- |
      | matched depths, folds 1.5–4 (cases higher) | 2.1–8.4% | 1.3–1.4% | 0.8–1.1% | 0.9–1.4% |
      | matched depths, fold 0.5 (cases lower) | 6.1% | 8.0% | 2.8% | 1.1% |
      | the TCGA designs' own imbalance, no fold | 2.1% | 1.9% | 1.2% | 1.4% |
      | a 2× depth gap | 19–24% | 3.0–15% | 1.6–4.5% | 1.3% |

    - It fails most when the cases are the lower or shallower group: extra
      Poisson noise among the cases reads as concentration.
    - 128 is the lowest bin edge of the reference at which every arm holds.
      It is a measured validity bound, not a parameter of the method.

The DE test is unaffected: its count-exact pairwise scores (decisions 11 and
13) estimate no fold. Whether the reported effect becomes the Laplace fold
rather than the W1 alignment is left to step 3.

## 5. Retracted claims (do not repeat)

1. **"The excess is gone by 250 v 250."** It is still measurable
   (section 3.4).
2. **A p-value floor of $\binom{n_1}{k}/\binom{n_1+n_0}{k}$** for k extreme
   cases, and "no test has power at 20 v 20".
   - Counterexample: cases [0, 1.1, 33/7, 100] against controls [1, 2, 3, 9]
     give exact p = 22/70, below the claimed 1/2.
   - Other samples move the Lorenz curves too. The probability is intuition,
     not a limit.
3. **"A restricted null is the only valid route to the lost power."** That is
   not shown; there is no optimality result.
4. **"Sparse tails need pooling, and pooling needs registration."** The pair
   test is a registration-free exact test.
5. **"A count-consistent fixed point."** Not established.
6. **An asymptotic validity theorem.** It is a proof strategy (section 6,
   step 4).
   - Also withdrawn: the claim that a registration error of order $n^{-1/2}$
     moves the shuffled gap only at order $n^{-1}$.
7. **"Welch fails because of the t approximation."** The permutation test
   was used; its finite-sample error is the issue.
8. **"Run the labels both ways and take the smaller p."** Only a jointly
   calibrated maximum is valid, which is what `subsets="either"` computes.
9. **"Zero discoveries in 60 null matrices" as a pass rule.** A calibrated
   procedure passes with probability $0.95^{60} = 0.046$.
10. **"Typical-case fold."** It is an alignment. Likewise, "(M + 1)/(b + 1)
    is the biological fold" is wrong.
11. **The study's pattern "every case ×½, a tenth ×8".** It put the tenth at
    4× baseline; it is now 8×, as in the generator, and rerun.

## 6. Next steps, in order

### Step 1. Define the supported RNA-seq null: the count model

Goal: a subset test, and a DE test, whose validity holds under the count model
of section 3.5 with real exposures, not only on observed count + 1.

1. **Done: the exact count-model reference** (`research/count_model.py`,
   section 3.5).
   - Design-constant thinning of real tumours.
   - A random split per gene.
   - Given size factors from the other half of the genes.
2. **Done: the current tests measured under it** (section 3.5).
   - They fail below about 32 expected counts whenever there is a fold or the
     groups' exposures differ.
   - All of them are calibrated for a random split.
3. **Decided: the subset test keeps the W1 registration and limits its
   claim** (decision 14). It is calibrated when the sparser group expects at
   least 128 reads per sample.
   - The thinning constructions are retired from the subset test. Thinning
     by the Laplace fold is exact under a fold at every level, but fails on
     zero-inflated and bimodal genes once the fold is estimated. Thinning by
     the W1 fixed point fails below about 2 counts.
   - They stay in `research/count_model.py` as references.
   - The principled routes (a bootstrap of the whole procedure, Berger–Boos)
     are recorded in section 3.5 and not pursued now.
4. **The DE test under the count model.** The jittered rank test fails with
   unequal exposures.
   - Two count-exact statistics, prototyped:
     - the zero chance, which tests a Laplace fold of 1;
     - mid-rank scores, the count model's version of the relative effect.
   - What holds (section 3.5):
     - 0.05 and 0.01 in every arm;
     - 0.001 at equal exposures and at the realistic imbalances;
     - with a 2× exposure gap and the smaller group shallower, the far tail
       reaches about twice nominal (Behrens–Fisher).
   - Power: ahead of the jittered rank test at low counts, except in the far
     tail at 40:405.
   - *Depth:* each pair at its own common depth (decision 13). The pairwise
     scores are calibrated, need one binomial per pair at any level, and
     match the jittered rank test's power at high counts (section 3.5).
   - Open: an O(n) or O(n log n) computation of the pairwise scores for large
     cohorts (they are O(n²) per gene); small designs; the far tail under
     large exposure gaps.
   - This bears on decision 7 (section 4): the rank test would become a test
     of mid-rank scores.
5. **Candidate A** (a nonparametric mixture fit with a bootstrap) is on hold
   (decision 14). It remains the principled route should the subset test's
   claim need to reach low counts.
6. **Exposures.** Size factors are estimates.
   - State the relative-expression interpretation.
   - Validate the normalization under widespread differential expression
     and changes of composition. That needs a reference arm in which only a
     share of the genes carries the fold, with RLE refitted on each data set.
7. **Jitter.** The subset test keeps it: its registered pool needs distinct
   values. The DE scores need none, since they are exact expectations over
   the counts.

Deliverable: the supported null, its estimator and test, the count-model
references, and their measured calibration.

### Step 2. Settle the calibration procedures

1. **The DE test: count-exact pairwise mid-rank scores** (decisions 11 and
   13; prototype `pairwise_scores` and `welch_test` in
   `research/count_model.py`).
   - *Scores.* Each pair is compared at its own common depth: the shallower
     sample's count against the deeper one's thinned to it, (B(C − 1) +
     B(C))/2 with B the thinned count's binomial distribution function. A
     sample's score is its mean over the other samples.
     - The scores need raw counts and size factors, not normalized values.
     - They are O(n²) binomials per gene: fine at hundreds of samples. Large
       cohorts need an O(n log n) method, for example grouping samples by
       depth or a normal approximation where counts are high, validated
       against the exact scores.
   - *Test.* The scores do not change under shuffles, so the Welch-studentized
     difference in mean score is shuffled as cheaply as the mean test (one
     matrix product per batch).
     - Put the scores on the exact-sum grid of section 3.5 so that ties stay
       ties.
     - Use Besag–Clifford stopping.
   - *Tail.* The far tail runs up to about 2× nominal at 0.001 when the
     smaller group is also the shallower one (Behrens–Fisher). Document the
     measured bound, or find a better-calibrated studentization.
   - *Fast tail.* The Welch saddlepoint prototype is a starting point:
     studentized statistics need a constrained saddlepoint, and branch
     switches need a global search.
   - Settle the column names with the author. The effect column is step 3's
     question.
2. **The subset test's small-sample residual.** Either document it with
   measured bounds by design and shape, or add a screened valid reference
   (section 3.4).
3. **The subset test under the count model: decided** (decision 14).
   Document its claim, at least 128 expected reads per sample in the
   sparser group, and the table behind it.

### Step 3. Effect semantics

The columns are done. Still to do:

- plots that show the side from `subset_side`;
- a within-gene diagnostic, the curve
  $u \mapsto \log\{(Q_1(u) + 1)/(Q_0(u) + 1)\}$ with bands, whose weighted
  median is the alignment;
- optionally, cut stability: apply each bootstrap resample's cut to the
  original cases. This is descriptive, not a membership probability.

### Step 4. State the theory the release relies on

- **What validity at a fixed level needs.** With
  $r_N = \sqrt{n_1 n_0/(n_1 + n_0)}$, the sampling Lorenz process
  $r_N(\hat L_0 - \hat L_1)$ and the conditional permutation process of the
  registered pool must converge to the same Gaussian limit, in a topology that
  supports the maximum.
- **What is proven.** The observed gap is invariant to the fold, and W1 is
  consistent under H₀.
- **What is open.**
  - The conditional limit: moment control, tightness, regularity.
  - The signed (argmax-direction) statistics.
  - Uniformity over gene distributions.
  - Accuracy at 10⁻⁶.
- **The rule.** Claim only what is proven, and label the rest as measured.

### Step 5. Freeze everything, then confirm (protocol version 7)

- **Freeze first:** normalization, filtering, count calibration, direction
  selection, Monte Carlo stopping and the statistic.
- **Predeclare, for each arm:** the quantity, a tolerance, and the
  simulation size needed to tell acceptable from unacceptable calibration
  with stated power. There are no zero-event rules.
- **Exact references by model.**
  - Observed scale: the same statistic without registration where the fold
    is 1, and the pair test.
  - Count model: known-fold, known-exposure thinning.
- **Uncertainty across whole matrices or labellings,** with paired
  comparisons against the oracle on shared shuffles.
- **The arms.**
  1. Real genes under random labels: TCGA prostate and breast, designs from
     1:10 to 10:1, including heavy-tailed and genotype-bimodal genes.
  2. Simulated observed-scale nulls, including the bimodal nulls.
  3. Count-model nulls.
  4. Complete-null matrices (the chance of any discovery, with an interval)
     and partial-null matrices (the false discovery proportion).
  5. Power at matched size: thresholds fitted on one set of simulations and
     evaluated on another, including the failure families.
  6. Biological recovery, separately: ERG and ETV fusions, ERBB2
     amplification.
- **The thresholds actually used.** Validate at the BH thresholds users
  meet, not only at 0.001.
- **BH and BY** both need valid p-values. A combined list of mean and subset
  discoveries needs its own error target.

### After the method: documents, then the remaining issues

1. **Documents.**
   - Rewrite the derivation around the Lorenz identity.
   - Rewrite the method, manual, validation, changes and TCGA-examples pages
     and the README, and rerun the notebooks.
2. **Clean-up.** Delete `rust/src/study.rs`, and freeze or retire the study
   scripts.
3. **The remaining ISSUES items, in order:** rarity (item 3), the dataset-wide
   model (item 4), performance (item 6), and covariates, strata and pairing
   (later).

## 7. Notes for whoever continues

- **Environment.** Run with `PYTHONPATH=src:research` and the conda env
  `wade`.
- **Commits.** Do not commit unless the author asks. Ask before writing to
  the shared lab Dropbox.
- **Numbers.** Every documented number must come from a research script run.
  Check `uptime` before timing anything.
- **Style.** No tuned magic numbers: derive thresholds from principle, and
  prefer deleting to adding.
- **The machine is shared.** The author runs other jobs on it; cap Rust
  threads (`RAYON_NUM_THREADS=6`) on long runs.
- **Research-only code.** `research/subset_null_study.py` depends on
  `rust/src/study.rs` for the old constructions; the package does not.
- **Random numbers.** numpy's Poisson sampler consumes a variable number of
  draws. Changing one pattern's rates shifts every later random draw in a
  script, so rows after it change within Monte Carlo noise.

## 8. Reproduction

| Script, section | Output in `.release/0.2.0/` | Content |
| --- | --- | --- |
| `subset_null_study.py` default sections | `subset-null-study-20260929.txt` | The reviewed derivation's evidence |
| `normalizers`, `thinnednull`, `thinfixed` | `subset-null-review1-*-20260929.txt` | Other normalizers; thinning |
| `review2` | `subset-null-review2-20260929.txt` | Bimodal nulls, small real designs, bootstrap, Welch |
| `mechanism2` | `subset-null-review2-mechanism-20260929.txt` | Why the registration fails on bimodal nulls |
| `lorenz` | `subset-null-review2-lorenz-20260929.txt` | Lorenz: generator power, validity grid, real labels |
| `bimodalgenes`, `bergerboos` | `subset-null-review2-bb-20260929.txt` | GSTM1 and ERAP2; Berger–Boos |
| `registrations`, `matched` | `subset-null-review2-registrations-20260929.txt` | Registrations; power at matched size |
| `budget` | `subset-null-review2-budget-20260929.txt` | Generator power at 999,999 shuffles |
| `checks` | `subset-null-review2-checks-20260929.txt` | Invariance and the Lorenz identities |
| `large250` | `subset-null-review3-large250-20260929.txt` | 200,000-gene bimodal null at 250 v 250 |
| `location_tests.py` | `location-tests-20260929.txt` | The DE-test comparison |
| `welch_saddlepoint.py` (`exact`, `shuffles`, `solver`) | `welch-saddlepoint-20260929.txt`, `welch-saddlepoint-solver-20260929.txt` | The Welch saddlepoint prototype |
| `count_model.py` `splits`, `reference` | `count-model-20260929.txt` | The count-model reference: shared splits against a split per gene; the package's tests and the thinning oracles |
| `count_model.py` `laplace` | `count-model-laplace-20260929.txt` | The Laplace fold, its thinned subset tests and the zero-chance DE test |
| `count_model.py` `power` | `count-model-power-20260929.txt` | Power on subsets planted by thinning |
| `count_model.py` `scores` | `count-model-scores-20260929.txt` | The DE tests below 8 expected counts, with count-exact mid-rank scores |
| `count_model.py` `realistic` | `count-model-realistic-20260929.txt` | The TCGA designs' own exposure imbalance |
| `count_model.py` `summary` | `count-model-summary-20260929.txt` | Both grids condensed by group of arms |
| `count_model.py` `depthchoice` | `count-model-depthchoice-20260930.txt` | Equalized against optimized-depth thinning: calibration and power |
| `count_model.py` `seeds` | `count-model-seeds-20260930.txt` | How much the thinned test's calls depend on the thinning draw |
| `count_model.py` `pairwise` | `count-model-pairwise-20260930.txt` | Mid-rank scores with each pair at its own depth |
| `count_model.py` `small` | `count-model-small-20260930.txt` | The adopted constructions in small designs, with and without jitter |
| `count_model.py` `bimodal` | `count-model-bimodal-20260930.txt` | Review 2's bimodal null under the count model |
| `count_model.py` `realbimodal` | `count-model-realbimodal-20260930.txt` | GSTM1 and ERAP2 under random splits |
| `count_model.py` `robust` | `count-model-robust-20260930.txt` | The W1 fixed point on equalized counts against the Laplace fold |

Commands: `research/README.md`. The TCGA sections need
`WADE_TCGA_CONFIG=.release/0.2.0/tcga-config.json`.

## 9. References

- Bahadur, R. R. and Savage, L. J. (1956). *Annals of Mathematical Statistics*
  27, 1115–1122.
- Barrett, G. F., Donald, S. G. and Bhattacharya, D. (2014). Consistent
  nonparametric tests for Lorenz dominance. *Journal of Business and Economic
  Statistics* 32, 1–13.
- Berger, R. L. and Boos, D. D. (1994). *Journal of the American Statistical
  Association* 89, 1012–1016.
- Brunner, E. and Munzel, U. (2000). *Biometrical Journal* 42, 17–25.
- Chung, E. and Romano, J. P. (2013). *Annals of Statistics* 41, 484–507.
- Gandy, A. and Hahn, G. (2014). *Scandinavian Journal of Statistics* 41,
  1083–1101.
- Gerard, D. (2020). *BMC Bioinformatics* 21, 206.
- Janssen, A. (1997). *Statistics and Probability Letters* 36, 9–21.
- Neubert, K. and Brunner, E. (2007). *Computational Statistics and Data
  Analysis* 51, 5192–5204.
