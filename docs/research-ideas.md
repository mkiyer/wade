# Research ideas

*30 September 2026.* WADE's research branched faster than the method could
absorb it. On 30 September 2026 the package was simplified to two tests: the
area test on depth-equalized counts, and the subset test. The research
scripts were pruned to what supports them. This ledger keeps every idea from
that research that is worth remembering: what it is, why it looked
promising, what was measured, and why it stopped where it did.

**Where the code is.** Everything named here is in git at commit `e4fd67f`,
the analysis freeze of 30 September 2026, on `main`. That tree's
`docs/subset-null-plan.md` holds the full record (sections 3 to 5). The
outputs cited are in `.release/0.2.0/`. To get a script back:

```bash
git show e4fd67f:research/count_model.py > count_model_frozen.py
```

or check out the whole frozen tree beside this one:

```bash
git worktree add ../wade-analysis e4fd67f
```

**Status.**

- **Adopted:** in the package.
- **Parked:** worth resuming; the obstacle is named.
- **Rejected:** measured to fail; the reason is named.
- **Untried:** an idea with no measurement yet.

**Notation.** Rates are shares of null genes at p ≤ 0.05 / 0.01 / 0.001.
Designs are cases:controls. "Expected reads" is per sample in the sparser
group, for one gene. The count model is
C_j | Λ_j ~ Poisson(s_j f^{L_j} Λ_j), with s_j a sample's depth, L_j its
label and f the gene's fold ([validation](validation.md)).

## 1. Adopted

- **Depth equalization before the area test.** Every sample is thinned
  binomially until its size factor equals the smallest, and then the difference in
  means is tested by permutation. Under the count model with no fold and
  known depths, the thinned samples are exchangeable whatever their depths,
  so the test is exact at every expression level. With estimated size
  factors it was measured calibrated in every arm
  ([validation](validation.md)). It washes out genes whose counts are too low
  to trust. This matches the author's original HITLIB practice of adding
  noise to wash out low counts.
- **The area and the area fraction** (from HITLIB's `diff.frac`):
  - the area is case mean minus control mean, in normalized reads per sample;
  - the area fraction is (case mean − control mean)/(case mean + control
    mean), in [−1, 1].
  - Neither needs a pseudocount. They replace the +1 fold `mean_log2fc`.
- **The Lorenz subset statistic with the midpoint W1 registration.** Since
  25 October 2026 the fold is taken out by thinning rather than rescaling.
  Its claim is limited to genes whose sparser group expects at least 32
  reads per sample (128 before), and `wade()` warns about genes below that.
  The walk's gap is averaged over four thinning draws (section 3.4).
- **The count-model reference** as the validation instrument. It thins real
  tumours by the design alone, with a random split per gene. It is kept in
  `research/count_model.py`.

## 2. The DE test

### 2.1 Count-exact pairwise mid-rank scores — parked

- **What.** Each pair of samples is compared at the pair's own common depth:
  the shallower sample's count against the deeper one's count thinned
  binomially to it.
  - A sample's score is its mean, over the other samples, of P(its count is
    higher) + ½ P(tie).
  - Scores are exact expectations over the thinning, so no random draws are
    involved.
  - The test is the Welch-studentized difference in mean score, calibrated by
    shuffles. Scores do not change under shuffles, so each batch costs one
    matrix product, as the mean test does.
- **Why promising.** Every sample competes head to head with every other.
  The test is valid at every count level with no jitter and no random
  thinning. It is the count model's version of Brunner and Munzel's relative
  effect.
- **Evidence.** Sources: `count-model-pairwise-20260930.txt`,
  `count-model-scores-20260929.txt` and `count-model-small-20260930.txt`.
  - Calibrated in every arm: 3.6–6.0% at 0.05, 0.6–1.5% at 0.01 and
    0.00–0.16% at 0.001. Under a 2× depth gap below 8 expected reads, the
    jittered rank test gives 26–50% at 0.05.
  - Power at 0.001:
    - equal to the jittered rank test at 8 or more expected reads (fold 1.5,
      405:40: 79.09% against 78.95%);
    - above it below 8 expected reads (15.56% against 13.98%).
  - Small designs (20:20 to 100:10): 0.80–1.32% at 0.01; one far-tail cell
    reaches 0.32% at 0.001 (100:10).
  - When the smaller group is also the shallower one, the far tail runs up
    to about twice nominal at 0.001. This is the Behrens–Fisher problem: with
    unequal variances no two-sample test of equal means is exact.
- **Why it stopped.**
  1. Its effect is a probability, like an AUC, and ignores magnitude. Two
     genes with every case above every control score alike whether the cases
     have 2 reads or 2,000 (the author's point). Ranking biomarkers needs a
     magnitude.
  2. It costs O(n²) binomials per gene. That is fine at hundreds of samples
     and slow at biobank scale. A rough extrapolation from the prototype's
     timing, not a research-script measurement, put 20,000 controls and
     50,000 cases at days of compute, or hours with a fixed panel of 1,000
     opponents per sample.
  3. At low counts the author wants fewer false positives, not more power.
     Depth equalization delivers that directly.
- **If resumed:**
  - an O(n log n) computation (group samples by depth; use a normal
    approximation where counts are high), validated against the exact scores;
  - a better-calibrated studentization for the far tail;
  - pair the score test with a magnitude effect such as the area.
- **Code.** `research/count_model.py` at the tag, sections `scores`
  (scores at the gene's smallest depth) and `pairwise` (each pair at its own
  depth), with its Welch permutation test.

### 2.2 The zero-chance test — parked

- **What.** Each sample's chance of zero reads when thinned to the exposure
  at which the pooled mean zero chance is ½. The groups are compared by a
  Welch-studentized difference in means over shuffles. Under f = 1 every
  sample has the same expectation whatever its depth: the latent Laplace
  transform at one common exposure.
- **Evidence.** Sources: `count-model-laplace-20260929.txt`,
  `count-model-scores-20260929.txt` and `count-model-realistic-20260929.txt`.
  - With equal exposures: 0.6–1.4% at 0.01, by level.
  - With a 2× exposure gap: 0.1–2.2%, and up to 0.24% at 0.001 overall.
  - At the TCGA designs' own imbalance: 4.7–5.2% / 0.84–1.08% / 0.08–0.12%.
  - Power at a fold of 1.5, at 0.001, below 8 expected reads:
    - 53.55% at 222:223;
    - 14.54% at 405:40;
    - 11.33% at 40:405.
- **Status.** It tests a Laplace fold of 1, not a mean, and so suits genes
  where most samples have no reads. It is magnitude-blind, as the scores are.

### 2.3 The jittered rank test (studentized Brunner–Munzel) — rejected

- **What.** The relative effect P(case > control) + ½ P(tie) = ½,
  studentized and calibrated by shuffles, on jittered normalized values.
- **Evidence.**
  - *Lognormal values* (`location-tests-20260929.txt`):
    - it holds 4.5–5.4% at 0.05 and 0.8–1.5% at 0.01 at 50 v 500, 500 v 50
      and 250 v 250;
    - the unstudentized rank sum gives 0.4–14.1% in the unbalanced designs;
    - a pure subset (a tenth of the cases at 8×) is called 16–23% with 50
      cases and 46% with 250. The mean test calls it 98–100%.
  - *The count model with unequal exposures*: it rejects 98% below 0.5
    expected reads, 42% at 0.5–2 and 7.0% at 2–8, at p ≤ 0.01. Part of the
    cause is the jitter, whose expected 0.5/s_j differs by group.
  - *After equalizing depth* it holds (0.3–1.6%), but costs about half the
    power below 8 expected reads: 21.74% against 48.69% at a fold of 1.5,
    222:223, 0.001.
- **Status.** Rejected under the count model. Its effect is magnitude-blind.

### 2.4 Welch studentization of the mean test — rejected as a fix

- **Evidence.**
  - With equal means and unequal spreads, when the smaller group is the
    wider, the unstudentized mean test rejects about 33% at 0.05.
  - Shuffling the Welch statistic brings that to 8.8–10.5%.
  - No test of arithmetic means is uniformly valid over distributions with
    arbitrarily heavy rare tails (Bahadur and Savage, 1956).
- **Status.** The area test's null is "the groups share a distribution", so
  it also rejects when they differ in spread alone. The documentation says
  so. Studentization would be needed only for a claim about means alone.

### 2.5 A saddlepoint for studentized permutation statistics — parked

- **What.** A constrained double saddlepoint for the Welch-studentized
  permutation statistic.
- **Evidence** (`welch-saddlepoint-20260929.txt`,
  `welch-saddlepoint-solver-20260929.txt`):
  - within 0.93–1.07 of exact enumeration from 10⁻¹ to 10⁻³;
  - within 0.88–1.18 of 2,000,000 shuffles down to 10⁻⁵;
  - its continuation solver fails on 3% of heavy-tailed genes, where the
    minimizer switches branches. A global search along the level curve fixes
    that.
- **Status.** Parked. It is the starting point for a fast tail for any
  studentized test (sections 2.1 and 2.2).
- **Code.** `research/welch_saddlepoint.py` at the tag (sections `exact`,
  `shuffles` and `solver`).

### 2.6 The exact-sum grid — parked with the Welch test

- **What.** Before matrix-product shuffles, put every value on a grid of
  2^−k with k = ⌊(53 − log₂ n)/2⌋. Every sum of n values is then exact in any
  order, so labellings that pick the same values give identical statistics
  and ties stay ties.
- **Evidence.** Without it, 2,000 labellings with the same case values gave
  3 distinct sums, and sparse genes rejected too often in the far tail.
- **Status.** The shipped area test handles ties with a rounding tolerance
  instead. Reuse the grid for any studentized statistic computed from matrix
  products.

## 3. Depth and thinning

### 3.1 Equalized thinning under a fold — the area test's construction, extended

- **What.** Every sample is thinned to its gene's smallest effective
  exposure (its size factor, times the fold for a case), and then shuffled
  with no registration. With f = 1 its thinning is the shipped area test's;
  here it served the subset statistic.
- **Evidence** for the subset null. Sources:
  `count-model-20260929.txt`, `count-model-laplace-20260929.txt`,
  `count-model-realistic-20260929.txt`, `count-model-depthchoice-20260930.txt`
  and `count-model-seeds-20260930.txt`.
  - With the true fold: 0.3–1.9% at 0.01 in every arm, and 0.05–0.14% at
    0.001.
  - At the TCGA designs' own imbalance: 4.1–5.3% / 0.73–1.06% / 0.05–0.14%.
  - Its power cost, against thinning only the higher group, in median
    points by expected reads:

    | 0.5–2 | 2–8 | 8–32 | 32–128 | ≥ 128 |
    | --- | --- | --- | --- | --- |
    | 17 | 6 | 1.3 (worst 12) | 0.2 | 0 |

  - The thinning draw: two seeds on the same data give the same share
    called, within 0.2 points. On planted subsets 90–95% of the calls agree.
    Under the null the calls are noise, and 8–48% agree.
- **Status.** Exact with a known fold. The subset test would need the fold
  estimated, and every estimate measured fails somewhere (sections 4.1 and
  4.2).

### 3.2 An optimized common depth — rejected

- **What.** Thin to the depth a that keeps the most reads,
  a · #{j : e_j ≥ a}, and set the shallower samples aside.
- **Evidence** (`count-model-depthchoice-20260930.txt`).
  - With no fold it is calibrated.
  - At a fold of 2 (1/8 depth) it rejects up to 11.4% / 3.5% / 0.61%, and
    18.7% at 0.05 above 128 expected reads.
  - At 40:405, with every case ×½ and a tenth ×8, it finds 6–18% of the
    subsets where the other constructions find 53–91%.
- **Why it fails.** A library's size factor is not independent of its
  expression profile, because composition moves both. Selecting samples by
  depth therefore selects on biology. In this construction the fold entered
  the effective exposures, so the cut set aside different depth ranges in
  the two groups.
- **Bearing on QC.** A depth floor applied as quality control was not
  measured. Set it before testing, and note that it changes which
  populations are compared whenever depth tracks biology.

### 3.3 Thinning only the higher group — rejected

- **What.** For the subset null, thin the higher group by the fold so the
  groups match, then shuffle with no registration.
- **Evidence.**
  - With the true fold and equal exposures: 0.5–1.6% at 0.01.
  - With unequal exposures it fails like the package's subset test: up to
    24% at 2–8 expected reads and 21% at 8–32.
- **Status.** Rejected. The exposures break it, not the fold.

### 3.4 Pooled thinning draws — adopted: the mean gap of four draws

- **What.** The shipped subset test thins once, and near BH's threshold that
  one draw decides calls: on TCGA prostate two seeds agree on 70% of them
  ([validation](validation.md#the-subset-test-under-the-count-model)).
  `research/subset_pooled.py` combines K independent draws of the thinned
  pool, all walked under the run's one shuffle stream.
  - *Mean p:* each draw's p-value, averaged. Each draw alone is the shipped
    test, and an average of valid p-values is valid within a factor of 2
    under any dependence.
  - *Mean gap:* the walk's score averaged over the draws, in one
    Besag–Clifford run per gene. Not exact in principle: given its level, a
    sample's draws are correlated by the share of its reads it keeps, which
    differs between samples and groups.
- **Evidence** (`subset-pooled-{seeds,grid,bimodal,power}-20261025.txt`).
  - *Calibration,* worst share at p ≤ 0.01 over the count model's 48 data
    sets:

    | Construction | 2–8 | 8–32 | 32–128 | ≥ 128 |
    | --- | ---: | ---: | ---: | ---: |
    | One draw (shipped) | 15.4% | 1.75% | 1.79% | 1.94% |
    | Mean p, 4 / 8 draws | 6.5% / 5.5% | 1.02% / 0.85% | 0.57% / 0.67% | 1.09% / 1.06% |
    | Mean gap, 4 / 8 draws | 29.8% / 33.5% | 2.26% / 2.38% | 1.43% / 1.33% | 1.94% / 1.58% |

  - *Zero-inflated genes* (GSTM1, ERAP2, PSPHP1, UGT2B17): every
    construction near its level; the mean gap within 0.2 points of one
    draw, the mean p slightly conservative.
  - *Power,* planted subsets at p ≤ 0.01: equal within a point at 32 reads
    and above. Below 32 the mean gap calls more and the mean p less: a
    tenth of the cases at 8×, 2–8 reads, 72% / 66% / 61% (mean gap 8, one
    draw, mean p 8).
  - *TCGA prostate, seeds 1 and 2,* the share of either seed's calls that
    both make:

    | Construction | Calls (seed 1 / 2) | All | ≥ 128 reads | Time |
    | --- | ---: | ---: | ---: | ---: |
    | One draw | 4,936 / 4,898 | 70.3% | 81.0% | 1× |
    | Mean p, 4 draws | 3,986 / 4,021 | 83.4% | 88.2% | 3.9× |
    | Mean p, 8 draws | 3,845 / 3,851 | 86.9% | 90.0% | 7.9× |
    | Mean gap, 4 draws | 5,579 / 5,685 | 80.2% | 89.5% | 10.2× |
    | Mean gap, 8 draws | 5,724 / 5,815 | 84.7% | 91.5% | 19.9× |

    New shuffles of one thinned pool agree on 93.6% at 128 reads and above:
    the floor any number of draws can reach at 25 exceedances. The mean
    gap's times are of a Python loop; a native tally would cost about K
    times the shipped test. ETV4's subset is called under both seeds by the
    mean gap (q 0.029–0.034); under the mean p of 8 draws it sits at the
    threshold (0.050 and 0.051).
- **Status.** The author adopted the mean gap of four draws on 25 October
  2026, built into the native walk (`wade._subsets.DRAWS`); the shipped
  test's own measurements are in [validation](validation.md). The mean gap
  keeps the one-draw test's level and power where the test is validated,
  and steadies its calls there; below 32 reads it is more anti-conservative
  under a fold than one draw. The mean p, conservative everywhere at a fifth
  of the calls, was set aside. `research/subset_pooled.py --prototype` reruns
  the measurements above.

## 4. The subset test's registration

### 4.1 The Laplace fold — parked

- **What.** A sample thinned to exposure e shows no reads with chance
  (1 − e/s_j)^{C_j}. Its mean over a group is the latent Laplace transform at
  e·f, E exp(−e f Λ), whatever G and the exposures. The exposures at which
  the cases' and the controls' mean zero chance reach one level have the
  ratio 1/f. The level is ½, or higher where a group cannot reach ½ down to
  its smallest exposure.
- **Why promising.**
  - It is exact for any latent distribution, and exactly reciprocal under a
    group swap.
  - A few extreme samples barely move it.
  - Its median error over genes is at most 0.10 log₂ below 0.5 expected
    reads, 0.016 at 0.5–2 and 0.012 above.
- **Why it failed as a registration.** In zero-inflated and bimodal genes the
  zero chance is set largely by how many zero-mode samples (for GSTM1, the
  deletion homozygotes) land in each group. A chance imbalance then becomes
  a false fold (`count-model-realbimodal-20260930.txt`, equalized):
  - GSTM1 at 10:40: 19.2 / 14.3 / 7.86%;
  - PSPHP1 at 100:100: 29.6 / 22.8 / 8.92%.
  - About 289 TCGA prostate genes have such a zero mode: 5–95% of tumours,
    with a positive mode whose median is at least 20 reads.
- **Possible uses.** A low-count effect estimate reported beside the area,
  or the zero-chance test (section 2.2).
- **Code.** `research/count_model.py` at the tag, sections `laplace`,
  `realbimodal` and `robust`.

### 4.2 The W1 fixed point on equalized counts — rejected

- **What.** Thin, re-register by W1, multiply, and repeat.
- **Evidence** (`count-model-robust-20260930.txt`,
  `subset-null-review1-thinfixed-20260929.txt`).
  - It resists the modes better: GSTM1 at 10:40, 6.40 / 1.22 / 0.26%.
  - It fails under a fold below about 2 expected reads: 18.7% at 0.01 below
    0.5 expected reads (fold 2, 1/8 depth), and 31.6% at 0.5–2 (fold 4).
  - Its fold is biased at low counts: 1.91–1.92 for a true 2 at 1 count, in
    simulation.

### 4.3 The lesson of sections 4.1 and 4.2

At low counts only smooth functionals, such as the chance of zero reads, can
identify a fold. Those same functionals move with a chance imbalance of a
zero mode or a second mode. Quantile matching ignores that imbalance, but
cannot see a fold at low counts. No registration that is estimated from the
data and then held fixed through the shuffles is valid across these cases.
A valid test below 128 expected reads must include the fold's estimation in
its null.

### 4.4 Candidate A: a mixture fit with a bootstrap — parked

- **What.** Fit the latent distribution nonparametrically (a Poisson
  mixture). Then calibrate the subset statistic by simulating the whole
  procedure under the fitted null, fold estimation included.
- **Status.** Parked. It is the principled route should the subset test's
  claim need to reach below 128 expected reads.

### 4.5 Berger–Boos over registrations — rejected as a default

- **What.** Take the largest p-value over a confidence set of registrations,
  plus the set's chance of missing the true one.
- **Evidence** (`subset-null-review2-bb-20260929.txt`).
  - Over a grid of 9 registrations it reached 6.0% at 0.05 on a sharper
    mixture: the grid misses the supremum.
  - It costs most of the power with few cases: 85.5% → 1.5% at 0.001, at
    50 v 500.
- **An exact version** must track, inside each interval of fixed pool order,
  where each shuffle's comparison with the observed gap changes.

### 4.6 The centered Lorenz bootstrap — rejected

It is invalid on spikes: 21% at 0.001 at 100 v 100, and 51% at 200 v 20.
It never asks whether an extreme value could have fallen among the controls
(`subset-null-review2-20260929.txt`).

### 4.7 The pair test — untried

Pair the samples within each group by a pairing fixed independently of
expression, and take H = |a − b|/(a + b) on count + 1. Permuting pair labels
is then exact under the observed-scale null, at the cost of half the data.
It would be a registration-free exact benchmark for the subset test.

### 4.8 A screened valid reference — untried

Compute an expensive valid p-value for screened candidates only, and set
p = 1 elsewhere. That keeps validity.

### 4.9 The small-sample residual — open

The subset test is slightly anti-conservative on a bimodal null,
log X ~ ½ N(3, 0.3²) + ½ N(5, 0.3²) in both groups
(`subset-null-review2-20260929.txt`, `subset-null-review3-large250-20260929.txt`):

| Design | Lorenz, W1 midpoint | True-fold oracle |
| --- | --- | --- |
| 20 v 20 | 5.7–5.9 / 1.8 / 0.32–0.37 | 4.7–4.8 / 1.0 / 0.09–0.11 |
| 250 v 250 | 5.064 / 1.030 / 0.114 | 4.982 / 0.972 / 0.0985 |

- **The mechanism.** A chance excess of lower-mode cases looks like an upper
  subset.
  - The registration reads part of that excess as a fold: the log fold
    correlates 0.84 with the imbalance in mode counts.
  - The pool built on that fold is less bimodal than the population, so the
    shuffled reference is too narrow.
  - All of the excess is in the "up" direction.
- **Remedies:** sections 4.4, 4.7 and 4.8.

## 5. Options removed on 30 September 2026

- **`subsets="either"`.**
  - *What it did.* It also walked the controls against the cases, on the
    swapped grid, and took the larger direction-signed score, so the choice
    between the groups was calibrated jointly.
  - *What it found.* A contraction, where the cases are more even than the
    controls, showed as a subset of controls. Without it a contraction has
    no subset.
  - *Why removed.* Simplicity (the author's choice).
  - *Code.* `rust/src/lib.rs` at the tag (`both_groups`),
    `src/wade/_subsets.py`, and `wade(subsets=...)`.
- **The log2 scale.**
  - *What it did.* Both tests ran on log2(x + 1). The subset statistic was
    the gap between absolute Lorenz curves, A(t) = ∫₀ᵗ (Q − μ), which is
    exactly unchanged by a shift on the log scale (checked to 3.6 × 10⁻¹⁵).
  - *Why removed.* One scale, the linear one: the area in reads is the
    effect the author wants.
  - *Code.* The `linear` flag in `rust/src/lib.rs` at the tag, and
    `SCALES` in `src/wade/_means.py`.
- **The +1 mean fold, `mean_log2fc`:** log2 of (case mean + 1)/(control
  mean + 1). The area and the area fraction replace it.

## 6. Normalization under a change of composition — measured, remedies untried

- **The finding.** When many genes change one way, RLE size factors absorb
  part of that change. Unchanged genes then look changed, in any DE tool
  that uses the size factors. Source: `count-model-composition-20260930.txt`
  (222:223, 1/8 depth, a share of genes changed 2×, 4 shared splits), with
  the pairwise scores:

  | Genes changed 2× | RLE bias, cases over controls | Scores, design factors | Scores, refitted RLE |
  | --- | --- | --- | --- |
  | 10%, all up | 1.051–1.063 | 7.0% / 1.4% | 23.8% / 12.2% |
  | 30%, all up | 1.189–1.204 | 5.9% / 1.2% | 69.5% / 57.8% |
  | 10%, half up and half down | 0.994–1.002 | 4.1% / 0.8% | 4.1% / 0.8% |
  | 30%, half up and half down | 0.998–1.005 | 2.8% / 0.4% | 2.9% / 0.4% |

- **Balanced changes are harmless.**
- **The subset test is unaffected.** Its registration absorbs a global fold:
  3.0–5.4% at 0.05 in every row.
- **The area test's own numbers:** `count-model-composition-area-20260930.txt`
  and [validation](validation.md).
- **Remedies (untried):**
  - a trimmed or reference-gene normalization (TMM; housekeeping genes;
    spike-ins);
  - a diagnostic that reports how asymmetric the RLE ratios are between the
    groups.

## 7. Ranking biomarkers

- **HITLIB's composite rank — untried in WADE.** The author's probe-panel
  analysis (`~/proj/cfrna/probe_panel/probe_panel_v06.Rmd`) ranks genes by
  sign(diff.frac) · ecdf(|log2 fold|) · ecdf(case mean) · (1 − ecdf(control
  mean)).
  - It rewards a large fold, high expression in cases and low expression in
    controls: how detectable a marker is, not only how significant.
  - WADE's area, area fraction and group means supply its ingredients.
- **Rarity** ([ISSUES.md](../ISSUES.md)). This asks how rare a gene's case
  subset is among genes of its size. It is a separate question from DE. A
  prototype on prostate, before the Lorenz statistic, ranked ETV4 10th of
  24,246 genes.

## 8. Older ideas

- **The two-stage design:** stage 1 and stage 2, thinning, the saddlepoint,
  GPD tail refinement. It was retired between 19 and 23 September 2026
  (commit `9bf10d3`).
- **Earlier research scripts:** in git at commit `f07d5f9`.
