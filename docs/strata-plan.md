# Strata: design and vetting

*24 October 2026; adopted and built 25 October 2026.* This page records the
design for testing within strata (study, site, batch or any other factor), the
prototype that vets it (`research/strata.py`), and the author's decisions
(section 6). The method as built is in [method](method.md#strata), and its
measurements in [validation](validation.md#strata).

## 1. What strata are

- **Strata are the groups one or more factors define.** With several
  factors, each combination of their levels is a stratum. Factors are
  categorical. A continuous covariate needs another construction (section 8).
- **The stratified null.** Within each stratum, the case and control labels
  are exchangeable. Strata may differ in anything: depth, composition, a
  gene-specific batch scale, biology.
- **Every shuffle exchanges labels only within strata.** Both tests are then
  exact under the stratified null, whatever differs between strata (Hothorn
  2006; Winkler 2014).
- **A stratum with only cases or only controls carries no information.** It
  cannot move a label, so it is left out of both tests and reported.
  - This is why strata cannot bring external controls in. GTEx normals
    stratified by study form a stratum with no cases, and drop out.
  - Using them needs an assumption that the studies' normals are
    exchangeable, which strata cannot supply (section 5, the bridge).
- **The cost.** The effective size is N_eff = Σ n₁ₛn₀ₛ/nₛ, against n₁n₀/n
  unstratified. The smallest p-value is 1/Π C(nₛ, n₁ₛ).

## 2. Interface

```python
wade(counts, cond, strata=study)                    # one label per sample
wade(counts, cond, strata=[study, sex])             # several factors, crossed
cond = wade.condition(meta, key="library", column="group", case="tumor",
                      control="normal", strata=["study", "sex"])
wade(counts, cond)                                  # strata travel with the condition
```

- Labels are matched by position (arrays) or by sample name (a `Condition`),
  as `cond` is. Missing labels are an error.
- `params` records each stratum's cases and controls, which strata carry
  information, N_eff, the samples left out and the permutation space.
- **Warnings:** the samples in single-group strata, left out of both tests;
  an error only if no stratum has both groups.
- `strata=None` is the present procedure exactly: one stratum reproduces the
  package's stream, saddlepoint and shuffles.

## 3. The area test

- **The statistic is unchanged.** The cases' total, shuffled within strata,
  is the Cochran–Mantel–Haenszel sum of within-stratum differences in means:
  T − E(T) = Σₛ (n₁ₛn₀ₛ/nₛ)(mean₁ₛ − mean₀ₛ). Values are centred within their
  stratum first, so that the two-sided test is centred.
- **Thinning within each stratum.** Each stratum is thinned to its own
  smallest size factor, and its values are divided by it. Exchangeability
  within a stratum needs equal depths only there. Size factors then need to
  be right only up to one constant per stratum.
- **The p-value.**
  - *Shuffled:* the run's one stream, each block permuting each stratum's
    labels in turn; position-indexed as now, so decision-based stopping is
    unchanged.
  - *Computed:* a stratified double saddlepoint. Independent Bernoulli
    selectors are conditioned on one count per stratum, an
    (S+1)-dimensional saddlepoint whose Hessian is an arrowhead. Every term,
    including the second-order error estimate, is a sum over samples: O(n)
    per gene whatever the number of strata.
  - Where the error estimate exceeds `relative_error`, the gene is shuffled,
    as now.
- **The effects.** The Mantel–Haenszel standardized group means,
  case_mean = Σ wₛ mean₁ₛ / Σ wₛ and control_mean likewise, with
  wₛ = n₁ₛn₀ₛ/nₛ. The area, the area fraction and log2fc follow from them as
  now, so area = case_mean − control_mean still holds, and log2fc is the
  Mantel–Haenszel fold.

## 4. The subset test

**Adopted: one fold, a level per stratum (S4b), by thinning.** As built, each
stratum is first thinned to its own smallest size factor, and the fold is
taken out of the higher group by thinning, as the unstratified test does since
25 October 2026 (`strata-thinned-20261025.txt`; the measurements in section 5
below are of S4b as first prototyped, by rescaling). Like the unstratified
test, it averages its walk over four thinning draws
(`strata-shipped-20261025.txt`).
- **The null.** Within each stratum, every case carries one fold of that
  stratum's controls, the same fold in every stratum. Strata may differ in
  level: depth, batch scale, composition.
- **The fold** is the weighted median of the strata's own Wasserstein-1
  folds on count + 1, with weights n₁ₛn₀ₛ/nₛ.
- **Each stratum's level** is the median of its log(count + 1) once that
  fold is taken out of its cases. It rests on all the stratum's samples, so
  no group is centred on a few of its members.
- **The test.** Every stratum is moved to the cohort's level, the cases are
  carried back by the fold, and one walk compares the pooled cases with the
  pooled controls, as now. Labels are shuffled within strata.
- **What carries over.** The observed gap is unchanged by the fold, so it is
  the gap between the level-aligned groups. One subset per gene, so
  `subset()`, rarity and the sample load carry over. The native walk is
  unchanged; only the stream is stratified.

**Tried and set aside** (section 5 gives the measurements):
- **Combined (S1):** each stratum walked on its own, the gaps added with
  weights √(n₁ₛn₀ₛ/nₛ). A fold fitted inside a small stratum absorbs chance
  differences while the observed gap ignores it, so the shuffles run too
  narrow. It is anti-conservative with small strata and under a fold.
- **Pooled (S2):** each stratum's cases registered onto its own controls. A
  stratum with one or a few controls centres them, and the shuffles, which
  pick other controls, run too narrow. It is anti-conservative on TCGA's own
  site layout, where one site has 22 tumours and 1 normal.
- **Label-free alignment (S3):** each stratum moved by its median over all
  samples, then one fold. A stratum's median mixes its case share into its
  level, so under a fold, strata with different case shares misalign. It is
  anti-conservative under a fold.
- **Alternating two-way fit (S4):** the same model as S4b, fitted by
  alternating medians. It is as well calibrated, but 18% of genes never
  settle, because the medians cycle. It costs 50 registrations, 25 s on TCGA
  prostate.

## 5. What the prototype measures

`research/strata.py`, every section in `strata-20261024.txt` (seeds fixed).

**The data.**
- *The count model with studies.* Each gene's TCGA prostate tumours are split
  at random into studies whose case shares differ. Each study is thinned
  binomially to its depth (1 or 1/4), with a gene-specific batch scale per
  study (lognormal, sd 0.5) and, in one arm, extra noise in alternate
  studies. Binomial thinning keeps the count model exact. There are two
  designs: two studies (176:44 and 44:176), and twenty studies of 22 (15:7
  and 7:15).
- *Real nulls,* each a random relabelling within strata, with case shares
  that differ between strata:
  - TCGA prostate tumours within 20 tissue source sites;
  - the tumours of the real contrast's five sites, with their real case
    shares (one site 22:1);
  - TCGA and GTEx prostate normals within study.

**Exactness.**
- **One stratum is the package.** The stream, the shuffled area test and the
  subset test reproduce it bit for bit. The saddlepoint agrees to 4 × 10⁻¹⁰,
  and its error estimate to 2 × 10⁻¹⁴.
- **Stratified shuffles are exact.** Four small designs have 2,475 to 109,760
  labellings, and one has a cases-only stratum. Against every labelling, the
  shuffles (stopped at the 400th exceedance, standard error 5%) deviate by a
  median 2.0–2.3%. None deviates by more than three standard errors.
- **The stratified saddlepoint's error estimate tracks its error.**

  | Designs | Genes the estimate keeps (≤ 0.2) | Kept genes beyond 0.67–1.5× of exact |
  | --- | ---: | ---: |
  | Four small designs, every labelling | 62–84% | 0–0.9% (2 genes in one design) |
  | TCGA prostate and breast sites, two studies, twenty small studies, one large and ten tiny; 20 million labellings | 98–100% of 267 | 0% |

  - On the realistic designs it sends 0.5–4.3% of genes to shuffles. Its
    median error is 0.7–2.1% where it keeps them.
  - On the small designs the median error is 1.9% where the estimate is
    0.1–0.2, and 7.9% at 0.2–0.5.

**Calibration.** Share of null genes at p ≤ 0.01. The area columns come from
sections `model`, `sites`, `layout` and `gtex` (3,000–4,000 genes). The
subset columns come from section `common`, which runs every construction on
the same 3,000 genes.

| Null | Area, unstratified | Area, stratified | Subset, unstratified | Subset, S4b |
| --- | ---: | ---: | ---: | ---: |
| Two studies, batch | 59.8% | 1.5% | 8.7% | 1.1% |
| Two studies, batch, every case 1.5× | | | 9.1% | 1.1% |
| Two studies, batch and extra noise | 61.1% | 0.7% | 2.4% | |
| Twenty studies, batch | 4.8% | 0.9% | 2.4% | 1.0% |
| Twenty studies, batch, every case 1.5× | | | 3.8% | 1.6% (1.25% at ≥ 128 reads) |
| TCGA tumours within 20 sites | 1.8% | 0.85% | 0.80% | 0.83% |
| The same, every case 1.5× | | | 0.97% | 1.07% |
| The real contrast's five sites | 1.1% | 1.1% | 1.2% | 0.87% |
| The same, every case 1.5× | | | 1.4% | 1.3% |
| TCGA and GTEx normals by study | 42.5% | 0.78% | 28.7% | 0.70% |

- Without batch effects every arm run there holds. The area test calls
  0.7–1.1%, unstratified or stratified. The subset test calls 0.7–1.1%,
  unstratified, S1 or S2. S4b was run on the batch arms.
- Folds that differ between strata (1.5× in alternate studies, none in the
  others) are not called subsets by S4b: 0.30% (two studies) and 1.27%
  (twenty), against 11.1% and 1.6% unstratified.

**The subset constructions set aside**, at p ≤ 0.01 / 0.05:

| Construction | Where it fails |
| --- | --- |
| Combined (S1) | TCGA sites 1.5% / 6.4%; twenty studies, batch, 1.5× fold, 2.3% / 8.7%. Without the registration the same statistic holds (0.95% / 4.95%), so a fold fitted inside a small stratum is the cause |
| Pooled (S2) | The real contrast's five sites 1.9% / 8.3%; with a 1.5× fold, 2.5% / 9.0% |
| Label-free alignment (S3) | TCGA sites, 1.5× fold, 2.3% / 8.8%; twenty studies, batch, 1.5× fold, 3.0% / 11.4% |

**Power.**
- **Area test, every case 1.5×, at p ≤ 0.01.**

  | Design | Thinned to the global minimum | Thinned within strata | Inverse-variance weights |
  | --- | ---: | ---: | ---: |
  | Two studies | 71.9% | 74.9% | 76.2% |
  | Two studies, batch | 69.7% | 72.7% | 72.9% |
  | Twenty studies | 77.4% | 82.1% | 84.1% |
  | Twenty studies, batch | 69.0% | 76.7% | 75.5% |

  - Thinning within strata gains 3–8 points.
  - Inverse-variance weights gain −1 to +2 points over the CMH weights: no
    consistent gain.
  - Unstratified, without batch, the test reaches 80.7% and 80.1%.
- **Subset test, a tenth of each stratum's cases 8×, batch, at p ≤ 0.01.**
  - S4b reaches 68.3% (two studies) and 78.1% (twenty studies), as S2 and
    S4 do.
  - Unstratified reaches 74.6% and 71.9%, on top of its false calls.

**Real data.**
- **The bridge.** TCGA normal-adjacent prostate against GTEx normals: the
  area test calls 18,849 of 24,246 genes (78%), 8,060 of them at least
  2-fold, and the subset test 3,054 (13%). Pooling GTEx normals as controls
  without strata would test this study difference. With strata, they drop
  out.
- **The real contrast stratified by site.**
  - *The design.* 5 of the 32 sites have both groups: 207 tumours and 52
    normals. N_eff is 40.6 stratified, against 46.6 for all 497 samples.
  - *Area test.* On those 259 samples it calls 13,197 genes unstratified and
    13,167 stratified.
  - *Subset test.* It calls 4,638 genes unstratified, 5,224 by S4b (3,846 in
    common), and 8,002 by S2, the construction set aside.
  - *Fusion markers under S4b.* ETV1's subset strengthens (q 0.012) and
    ETV4's weakens (0.088).

**Cost** on TCGA prostate (24,246 genes × 497 samples), as a ratio to the
unstratified piece in the same run, with 1 to 128 random strata:
- the stratified saddlepoint: 0.5–1.0×;
- the stream: 0.6–1.1×;
- the S4b subset test: 0.96×, with its pool built for every gene in 0.5 s,
  against 0.2 s for one registration.

## 6. Decisions for the author

*Adopted 25 October 2026:*
- 1 (the area test), 2 (S4b), 3 (the interface), 4 (Mantel–Haenszel effects),
  5 (single-group strata left out with a warning) and 7 (stratified designs in
  confirmation v7).
- *Instead of 6:* the subset test equalizes depth and registers by thinning,
  and its validated range falls from 128 to 32 reads per sample
  (`research/subset_range.py`, `subset-range-20261025.txt`). S4b takes the same
  construction within strata, vetted in `strata-thinned-20261025.txt`.

1. **The stratified area test** as in section 3:
   - shuffles and the saddlepoint within strata;
   - thinning within each stratum;
   - CMH weights, since inverse-variance weights gained nothing consistent;
   - Mantel–Haenszel effects.
2. **S4b for the subset test,** with its null: one fold shared by every
   stratum, a level per stratum. A 1.5× difference in fold between strata
   was not called. A larger one would be, as cases departing from the common
   fold.
3. **The interface** of section 2: `strata=` for arrays and lists of
   factors, and `condition(..., strata=[...])` for metadata columns.
4. **The effects under strata.**
   - *Proposed:* the Mantel–Haenszel standardized means as case_mean and
     control_mean, so that area = case_mean − control_mean still holds.
   - *The alternative:* keep the plain group means, at the price of an area
     that is not their difference.
5. **Single-group strata** are left out of both tests with a warning. It is
   an error only if no stratum has both groups.
6. **The subset test's validated range** stays at 128 reads per sample in
   the lower group. Under a fold, S4b's excess lies below it, as the
   unstratified test's does.
7. **The confirmation** (protocol version 7) adds stratified designs: the
   model's two designs, the TCGA site layout and the TCGA–GTEx merge.

## 7. Build plan

- **Shuffles.** `permutation.stream_block` takes the strata. One stratum
  keeps today's draws, so results without strata are bit for bit unchanged.
- **`_means`.**
  - Thinning within strata.
  - The stratified saddlepoint, with today's code kept for one stratum.
  - The shuffled test centred within strata.
- **`_subsets`.** The S4b pool, and the stratified stream fed to the native
  walk, which is unchanged.
- **`api`.**
  - Strata from arrays, lists of factors or a `Condition`.
  - The design report and warnings.
  - The informative strata.
  - The Mantel–Haenszel effects.
  - The run record.
- **`_rarity`.** Gap curves on the level-aligned values.
- **`io.condition(..., strata=...)`.**
- **Tests against oracles.**
  - The stratified saddlepoint and shuffles against exact enumeration.
  - One stratum identical to today.
  - Every shuffle keeping each stratum's counts.
  - The Mantel–Haenszel effects.
  - Single-group strata dropped.
  - A `Condition`'s strata aligned by sample name.
- **Documents.** Manual, method, validation, changes and README, and
  ISSUES.

## 8. Out of scope

- **Continuous covariates.** Exchangeability blocks are exact only for a
  nuisance constant within blocks. Residualizing counts breaks the count
  model that makes thinning exact. Binning a continuous covariate into
  strata is possible, with the bins chosen before looking.
- **Two-step batch correction** (ComBat-seq, then the test) is
  anti-conservative in unbalanced designs (Nygaard 2016; Li 2023) and breaks
  permutation validity. It is not offered.
- **Confounded designs,** where study and condition coincide, identify only
  biology plus study for each gene (Leek 2010). Strata cannot separate them.
- **Pairing and repeated measures** are strata of two or a few samples. The
  construction covers them, but they need their own validation.
  - For pairs, the area test becomes the paired permutation test.
  - S4b's fold becomes the median of the paired log ratios.
