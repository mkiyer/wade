# Open issues

## Before releasing 0.2.0

Work these in order; each later item waits on the subset test.

1. **The subset test's null, and the method around it.**
   - **The failure.** The released null removed the cases' mean fold, pooled
     them with the controls and shuffled. When a few extreme cases dominate
     the mean, the other cases land below the controls, and the test is badly
     anti-conservative. Measured on TCGA prostate (445 v 52) under random
     labels:
     - genes whose top sample is at least 4× the next reject at 18× the
       nominal rate at p ≤ 0.001;
     - MC4R's own values give p ≤ 0.001 in 27% of random labellings.
   - **The replacement,** after a derivation and three external reviews
     ([docs/subset-null-derivation.md](docs/subset-null-derivation.md),
     [docs/subset-null-plan.md](docs/subset-null-plan.md)). Implemented
     2026-09-29; uncommitted, and not yet in the documentation pages:
     - the statistic: the largest gap by which the cases' Lorenz curve of
       count + 1 falls below the controls';
     - the registration: the midpoint Wasserstein-1 alignment of
       log(count + 1);
     - `subsets="cases"|"either"`;
     - new columns `typical_log2fc` and `subset_group`.
   - **Decided 2026-09-29.**
     - 0.2.0 is held until the method is resolved.
     - The Lorenz statistic is adopted.
     - Both tests are offered.
     - The few-case power loss is accepted.
   - **Decided after review 3** (2026-09-29).
     - The plan's sequence (section 8.5) is adopted.
     - The DE test becomes a studentized rank test (Brunner–Munzel), with the
       alignment fold as its effect; the mean fold stays descriptive.
     - The effect columns are `alignment_log2fc`, `lorenz_gap`,
       `subset_group`, `subset_side`, `subset_fraction` and `subset_log2fc`
       (implemented).
     - Scope is two independent groups (cases v controls). Strata, pairing
       and covariates come later, and passing `strata` is an error
       (implemented).
       - The pages that describe strata or paired designs need rewriting.
       - `notebooks/rna100k.qmd` blocks the compendium by study, so it waits
         for the covariate work.
   - **Decided after step 1's first results** (2026-09-30; plan, section 4,
     decisions 10 and 11).
     - The subset test keeps the W1 registration, with its claim limited to
       at least 128 expected reads per sample in the sparser group
       (decision 14, replacing 10 and 12).
     - The DE statistic is the count-exact mid-rank score, Welch-studentized.
     - The mid-rank scores compare each pair at its own depth (decision
       13).
   - **The handoff plan** is docs/subset-null-plan.md, section 6: the next
     steps in order.
   - **Open after review 3,** in the plan's order (section 8.5):
     - the count-model null (item 2);
     - the registered test's small-sample residual, still measurable at
       250 v 250 over 200,000 bimodal genes (0.114% against 0.099% at
       p ≤ 0.001);
     - the studentized rank test: calibrated under the count model, made
       fast, and replacing the mean test (plan, section 4.8). As a test of
       arithmetic means the present test rejects 33% at p ≤ 0.05 under equal
       means with unequal spreads, and Welch studentization 10%;
     - the theory;
     - protocol v7 with acceptance criteria (item 5).
2. **Low counts and library depth: the count model.** Review 3 moved this
   before the protocol freeze: the observed-scale null on count + 1 is not
   the RNA-seq null. Step 1 of the plan is under way; its first results are
   in plan section 3.5, and the revised steps in section 6.
   - **The exact reference** (`research/count_model.py`).
     - Real tumours are split at random, one split per gene, and thinned
       binomially with probabilities fixed by the design.
     - Known fold and exposure arms span three designs and three depths.
   - **Measured.**
     - The package's subset test fails below about 32 expected counts
       whenever there is a fold: at p ≤ 0.01, up to 8.4% at 2–8 counts and
       2.8% at 32–128.
     - Unequal group exposures break the subset test and the DE tests at low
       counts. That includes the TCGA designs' own 7–14% imbalance at shallow
       depth: the rank test gives 3.6% at 0.01 at 1/64 depth.
     - Every test is calibrated for a random split.
   - **Found (prototypes).**
     - The *Laplace fold* matches the groups' chance of zero reads at a
       common exposure. It identifies the fold exactly under the count
       model, with median error at most 0.016 log2 above 0.5 counts.
     - Thinning every sample to one exposure by it (with the size factors)
       calibrates the subset test in every arm. It costs no power at 8 or
       more counts when only the higher group is thinned; equalizing costs
       some.
     - *Count-exact DE statistics* (zero chance; mid-rank scores) hold at
       0.05 and 0.01 in every arm. With a 2× exposure gap in unbalanced
       designs their far tail reaches about twice nominal.
   - **Measured 30 September.**
     - Setting shallow samples aside is not exact in real data, because
       size factors correlate with composition.
     - Equalizing every sample is the only exact depth handling measured.
       Its power cost is concentrated below 8 expected counts.
     - 90–95% of calls agree between thinning seeds.
     - Pairwise-depth mid-rank scores are calibrated at every level and
       match the rank test's power at high counts.
     - The Laplace fold, estimated on the same data, fails on
       zero-inflated and bimodal genes: 19% at 0.05 for GSTM1 at 10:40, 30%
       for PSPHP1 at 100:100. About 289 prostate genes share their shape.
       The robust W1 fixed point fails below about 2 counts.
     - **Decided (decision 14):** the subset test keeps the W1 registration.
       Its claim is limited to genes whose sparser group expects at least
       128 reads per sample: worst 1.4% at p ≤ 0.01 across every measured
       arm.
   - **Open.**
     - The thinning's randomization and theory.
     - A fast computation of the pairwise scores for large cohorts.
     - The normalization under composition change, and the jitter.
3. **Transcriptome-wide rarity.** A `subset_rarity` column: the share of
   genes whose k most extreme cases depart at least as far from their own mean
   shift, k the subset's size (a conformal rank; no shuffles). Prototype on
   prostate: ETV4 10th, ETV1 36th, ETV5 46th of 24,246 genes; the most
   extreme singleton (MC4R) 2nd. Add a per-sample count of rare subsets to
   flag contaminated or artefact libraries (one prostate tumour holds six of
   the forty most extreme singletons). Conditioning on abundance helps known
   biology but needs a model, not a window.
4. **A dataset-wide model of "expected".** A model of how a gene's
   distribution depends on its mean (mean–variance and shape, fitted across
   genes) is what defines the expected case quantiles at a given mean shift,
   and so bears directly on item 1. It also gives rarity a principled
   reference and supplies tails that a few controls cannot show.
5. **Release claims.** The ETV1/ETV4 subset q-values and the 500:50 subset
   power in the README and validation no longer hold. Rerun the notebooks and
   the confirmation, protocol v7 (plan, section 5):
   - predeclared acceptance criteria, with no zero-event rules;
   - references exact for each null model;
   - complete-null and partial-null matrices;
   - uncertainty across whole matrices and labellings;
   - thresholds fitted apart from evaluation;
   - the whole procedure frozen first.
6. **Performance at 20,000–80,000 samples**, once the algorithm is final.
   At saturation every gene runs the full shuffle budget (about 40 core-days
   at 80,000 samples). Stop each gene when its BH decision is settled
   (Gandy and Hahn) rather than when its p-value is precise; port the
   saddlepoint to Rust; vectorize the walk; stream gene chunks from Arrow.
7. **Review.** The author approves the rendered demo, prostate and breast
   notebooks and the documents' figures, tables and claims. Record the TCGA
   count-pipeline provenance (workflow, genome and annotation versions, row
   order).
8. **Publish.** Tag `v0.2.0` on `main`; CI tests every platform and publishes.

## After 0.2.0

- **Subset p-values below the shuffle floor.** Multilevel splitting was
  prototyped on the native walk (2026-09-28, outside the repository) and is not
  safe as built: on outlier genes such as ETV4 its local swap moves cannot cross
  between the tail's modes, so some runs collapsed by many decades, and larger
  populations shrank but did not remove a low bias, at a cost near brute force.
  Ties were not the cause, so jitter would not help. It would only rank the
  strongest subsets, never change a call at the 10⁻⁶ floor.
- **A stratified saddlepoint.** With strata the mean test is shuffled. The
  double saddlepoint extends to restricted exchange with one count constraint
  per stratum; worth doing if stratified designs become common.
- **Two subsets in one gene.** One walk makes one split, so a raised and a
  lowered subset in the same gene share it. A second split of each side
  (binary segmentation) would report both.
- **More controls.** Methods for pooling external normals (e.g. GTEx) with
  batch strata, and guidance on when that is exchangeable.
- **Native comparators.** Rerun DESeq2/edgeR and a classic outlier statistic
  (COPA or outlier sum) on the confirmation matrices for a publication.
- **Sparse assays.** Single-cell, spatial and cell-free RNA need their own
  exchangeability design and validation.
- **Measurement uncertainty.** Per-sample count kernels were explored and
  deferred; the derivations are in history at `f07d5f9`.
