# WADE's two tests: decisions and next steps

*30 September 2026.* This page records what was decided about WADE's two
tests, the claims withdrawn along the way, and what comes next.

- **The full record** of the research behind the decisions, with every
  measurement, the three external reviews and the prototypes, is the previous
  version of this page at commit `e4fd67f` (sections 3 to 5).
- **The ideas** from that research worth resuming are in
  [research ideas](research-ideas.md).
- **The method as built** is in [method](method.md).
- **Its measured performance** is in [validation](validation.md).

## 1. Scope

- **Two groups, independent samples.** WADE tests cases against controls.
- **Strata** are supported since 25 October 2026 ([strata plan](strata-plan.md)).
  Pairing, repeated measures and continuous covariates are not yet supported.
- **Two tests per gene:**
  - the area test (DE): the difference in means, on counts thinned until
    every sample's size factor equals the smallest;
  - the subset test: are the cases more unequal than the controls, after one
    common fold?

## 2. Decisions

Decisions 1 to 14 were taken on 29 and 30 September 2026, during the
research; 15 to 18 on 30 September 2026, for the build the author uses; 19
and 20 on 25 October 2026.

1. **0.2.0 is held** until the method is confirmed (section 4, step 1).
2. **The subset test uses the Lorenz statistic,** registered by the midpoint
   Wasserstein-1 fold of log(count + 1).
3. *Replaced by 17.* Both subset tests were offered: `subsets="cases"` and
   `"either"`.
4. **The subset's fold** is measured against the aligned controls.
5. **The few-case power loss** of the registered subset test is accepted.
6. **The count model precedes the freeze.** The DE test and the subset test
   are measured under the count model before anything is frozen.
7. *Replaced by 11, then by 15.* The DE test was to become the studentized
   Brunner–Munzel rank test.
8. **The effect columns** follow the third reviewer's set; decision 16
   replaces the DE effect.
9. **Scope: two independent groups** (section 1).
10. *Withdrawn by 14.* The subset test was to register by thinning with the
    Laplace fold. That fold, estimated on the same data, fails on
    zero-inflated and bimodal genes.
11. *Replaced by 15.* The DE statistic was to be the count-exact mid-rank
    score.
12. *Superseded by 14.* The subset test was to equalize depth.
13. *Replaced by 15.* The mid-rank scores were to compare each pair at its
    own depth.
14. *Replaced by 19.* **The subset test keeps the W1 registration and limits its claim.** Under
    the count model it is calibrated where the sparser group expects at least
    128 reads per sample: its worst share at p ≤ 0.01 there is 1.4%, across
    folds from 0.5 to 4 and every depth arm measured. Below that it is
    anti-conservative. 128 is the lowest bin edge of the reference at which
    every arm holds: a measured validity bound, not a parameter.
15. **The DE test is the area test on depth-equalized counts.** Every sample
    is thinned binomially until its size factor equals the smallest, then the
    difference in means is tested by permutation with the saddlepoint tail.
    - It is exact under the count model at every expression level, with no
      estimate inside the thinning.
    - It washes out low counts, which matches the author's goal: rank the
      best biomarkers with confidence, and keep false positives out of the
      count-noise range, rather than gain power there.
    - Its effect keeps magnitude. A rank or score effect, such as the
      mid-rank scores', gives two genes with every case above every control
      the same effect whether the cases hold 2 reads or 2,000.
    - The scores' O(n²) cost at biobank scale was a further reason.
16. **The DE effects are the area and the area fraction** (from HITLIB). The
    area is case mean − control mean, in normalized reads per sample. The
    area fraction is the area over case mean + control mean, in [−1, 1]. They
    replace the +1 fold `mean_log2fc`.
17. **One scale and one subset construction.** `scale="log2"` and
    `subsets="either"` are removed.
18. **The subset test's range is stated, not enforced.** `wade()` warns how
    many tested genes fall below the validated range in their lower group
    (32 reads per sample since decision 19), and reports every gene (the
    author's choice).
19. **The subset test thins** (25 October 2026, the author's choice after
    rejecting the 128-read range).
    - Every sample is thinned to the smallest size factor, as for the area
      test.
    - The Wasserstein-1 fold of those counts is taken out of the higher group
      by thinning, not by rescaling.
    - Its validated range is 32 reads per sample (`research/subset_range.py`,
      `subset-range-20261025.txt`).
    - The W1 fixed point reached 8 reads, but failed on zero-inflated genes.
20. **Strata** (25 October 2026): the design in [strata plan](strata-plan.md),
    built.

## 3. Retracted claims (do not repeat)

1. **"The subset test's small-sample excess is gone by 250 v 250."** It is
   still measurable there: 0.114% against 0.0985% at 0.001 on a bimodal null.
2. **A p-value floor of C(n₁, k)/C(n₁ + n₀, k)** for k extreme cases, and "no
   test has power at 20 v 20". Other samples move the Lorenz curves too.
   Counterexample: cases [0, 1.1, 33/7, 100] against controls [1, 2, 3, 9]
   give exact p = 22/70.
3. **"A restricted null is the only valid route to the lost power."** No
   optimality result shows it.
4. **"Sparse tails need pooling, and pooling needs registration."** The pair
   test ([research ideas](research-ideas.md), section 4.7) is exact without
   either.
5. **"A count-consistent fixed point."** Not established.
6. **An asymptotic validity theorem.** It is a proof strategy (section 4,
   step 3). Also withdrawn: that a registration error of order n^−1/2 moves
   the shuffled gap only at order n^−1.
7. **"Welch fails because of the t approximation."** The permutation test was
   used; its finite-sample error is the issue.
8. **"Run the labels both ways and take the smaller p."** Only a jointly
   calibrated maximum is valid.
9. **"Zero discoveries in 60 null matrices" as a pass rule.** A calibrated
   procedure passes with probability 0.95⁶⁰ = 0.046.
10. **"Typical-case fold."** The registration is an alignment. Likewise,
    "(M + 1)/(b + 1) is the biological fold" is wrong.
11. **The study's pattern "every case ×½, a tenth ×8"** once put the tenth at
    4× baseline. It was corrected to 8× and rerun.

## 4. Next steps

1. **Confirm the build (protocol version 7), then release.** *Confirmed 26
   October 2026: all 27 checks passed
   ([validation](validation.md#confirmation-protocol-version-7)).*
   - Freeze first: the normalization, the thinning, the filtering, the
     direction, the Monte Carlo stopping and both statistics.
   - Predeclare, for each arm, the quantity, a tolerance and the simulation
     size needed to tell acceptable from unacceptable calibration with stated
     power. There are no zero-event rules.
   - The arms:
     - real genes under random labels (TCGA prostate and breast, designs from
       1:10 to 10:1);
     - count-model nulls with known depths;
     - complete-null matrices (the chance of any discovery) and partial-null
       matrices (the false discovery proportion);
     - power at matched size;
     - biological recovery, separately (ERG and ETV fusions, ERBB2
       amplification).
   - Validate at the BH thresholds users meet, not only at 0.001. A combined
     list of area and subset discoveries needs its own error target.
2. **The subset test below 128 reads per sample:** see
   [research ideas](research-ideas.md), section 4. A valid test there must
   include the fold's estimation in its null; candidate A (a mixture fit with
   a bootstrap) is the principled route.
3. **The theory the release relies on.**
   - What validity needs, with r_N = √(n₁n₀/(n₁ + n₀)): the sampling Lorenz
     process r_N(L̂₀ − L̂₁) and the conditional permutation process of the
     registered pool must converge to the same Gaussian limit, in a topology
     that supports the maximum.
   - Proven: the observed gap is invariant to the fold, W1 is consistent under
     the null, and the thinned area test is exact under the count model when
     the exposures are known. With estimated size factors it is measured, not
     proven.
   - Open: the conditional limit (moment control, tightness, regularity), the
     signed statistics, uniformity over gene distributions, and accuracy at
     10⁻⁶.
   - The rule: claim only what is proven, and label the rest as measured.
4. **Effect diagnostics.** A within-gene plot of
   u ↦ log{(Q₁(u) + 1)/(Q₀(u) + 1)} with bands, whose weighted median is the
   alignment. Optionally, the stability of the cut under resampling
   (descriptive, not a membership probability).
5. **Then the remaining [issues](../ISSUES.md).**

## 5. Reproduction

Scripts in `research/` today, with outputs in `.release/0.2.0/`:

| Script, section | Output | Content |
| --- | --- | --- |
| `count_model.py` `shipped` | `count-model-shipped-20260930.txt` | The area test, and the same test unthinned, under the count model |
| `count_model.py` `composition` | `count-model-composition-area-20260930.txt` | The area test when a share of genes changes, with RLE refitted |
| `count_model.py` `real` | `count-model-real-20260930.txt` | The area test on TCGA prostate with its own labels: depth kept, calls by level, the thinning seed |
| `rarity.py` | `rarity-4draws-20261025.txt` | Rarity against the gap alone: markers, planted subsets, DNA alterations, contamination, stability |
| `stopping.py` | `stopping-4draws-20261025.txt`, `stopping-whole-prostate-4draws-20261025.txt` | Decision-based stopping: exactness against finished runs, shuffles and time |
| `pvalue_accuracy.py` | `pvalue-accuracy-20260930.txt` | The computed area-test p-value against exact enumeration or 20 million shuffles |
| `subset_recovery.py` | `subset-recovery-4draws-20261025.txt` | Planted subsets recovered on the confirmation's matrices |
| `subset_significance.py` | `subset-significance-4draws-20261025.txt` | Calibration and significance as a subset grows |
| `subset_range.py` | `subset-range-20261025.txt`, `subset-seeds-20261025.txt` | The subset test's constructions under the count model (rescaled, thinned, the W1 fixed point, exact); its calls under two seeds |
| `subset_pooled.py` | `subset-pooled-native-{seeds,grid,bimodal,power}-20261025.txt`; with `--prototype`, `subset-pooled-{seeds,grid,bimodal,power}-20261025.txt` | The shipped test (four thinning draws) against one and eight: calibration, power, calls under two seeds; the prototype's mean p and mean gap |
| `strata.py` | `strata-20261024.txt`; `thinned`, `shipped`, `real`, `gtex`: `strata-thinned-20261025.txt`, `strata-shipped-20261025.txt`, `strata-real-4draws-20261025.txt`, `strata-gtex-4draws-20261025.txt` | Strata: exactness, calibration on simulated studies and real relabellings, the real contrast by site, the TCGA–GTEx bridge |

The research since retired ran from the scripts at commit `e4fd67f`:

| Script, section | Output | Content |
| --- | --- | --- |
| `count_model.py` `splits`, `reference` | `count-model-20260929.txt` | Shared splits against a split per gene; the package's tests and the thinning oracles |
| `count_model.py` `laplace` | `count-model-laplace-20260929.txt` | The Laplace fold and the zero-chance test |
| `count_model.py` `power` | `count-model-power-20260929.txt` | Power on subsets planted by thinning |
| `count_model.py` `scores` | `count-model-scores-20260929.txt` | DE tests below 8 expected reads, with count-exact scores |
| `count_model.py` `realistic` | `count-model-realistic-20260929.txt` | The TCGA designs' own exposure imbalance |
| `count_model.py` `summary` | `count-model-summary-20260929.txt` | Both grids condensed by group of arms: the subset test's claim |
| `count_model.py` `depthchoice`, `seeds` | `count-model-depthchoice-20260930.txt`, `count-model-seeds-20260930.txt` | Equalized against optimized-depth thinning; the thinning draw |
| `count_model.py` `pairwise` | `count-model-pairwise-20260930.txt` | Mid-rank scores with each pair at its own depth |
| `count_model.py` `small`, `bimodal`, `realbimodal`, `robust` | `count-model-{small,bimodal,realbimodal,robust}-20260930.txt` | Small designs; bimodal and zero-inflated genes; the W1 fixed point |
| `count_model.py` `composition` | `count-model-composition-20260930.txt` | The normalization under composition change, with the scores and the subset test |
| `subset_null_study.py` | `subset-null-*-20260929.txt` | The reviewed derivation's evidence and the review sections |
| `location_tests.py` | `location-tests-20260929.txt` | Which location test the DE test can calibrate |
| `welch_saddlepoint.py` | `welch-saddlepoint-*20260929.txt` | The Welch saddlepoint prototype |

Commands: `research/README.md`. The TCGA sections need
`WADE_TCGA_CONFIG=.release/0.2.0/tcga-config.json`.

## 6. Notes for whoever continues

- **Environment.** Run with `PYTHONPATH=src:research` and the conda env
  `wade`.
- **Commits.** Do not commit unless the author asks. Ask before writing to the
  shared lab Dropbox.
- **Numbers.** Every documented number must come from a research script run.
  Check `uptime` before timing anything.
- **Style.** No tuned magic numbers: derive thresholds from principle, and
  prefer deleting to adding.
- **The machine is shared.** The author runs other jobs on it; cap threads
  (`RAYON_NUM_THREADS=6`) on long runs.
- **Random numbers.** numpy's Poisson sampler consumes a variable number of
  draws. Changing one pattern's rates shifts every later random draw in a
  script, so later rows change within Monte Carlo noise.

## 7. References

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
