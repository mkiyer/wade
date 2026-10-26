# Open issues

The work queue. Finished work keeps a one-line note; its evidence is in
[validation](docs/validation.md) and its description in
[changes](docs/changes-0.2.0.md).

## Before releasing 0.2.0

1. **Confirm the build.** *Done 26 October 2026:* protocol version 7 passed
   all 27 checks
   ([validation](docs/validation.md#confirmation-protocol-version-7)).
2. **Review.** *Done 26 October 2026:* the author approved the rendered demo,
   prostate and breast notebooks, for now. Left:
   - the author's review of README and `docs/` after the 0.2.0 cleanup;
   - the TCGA count pipeline's provenance, in the
     [data contract](docs/tcga-examples.md#inputs-and-provenance): the
     versions of the `mibulkrnaseq` workflow and of STAR. The release,
     genome, annotation and gene-ID join are recorded there already.
3. **Re-render the notebooks.** The 0.2.0 cleanup changed the package
   source and the notebooks' text (subtitles, captions, the settings they
   state), so the demo, prostate and breast notebooks need a render from the
   release commit; their results should not change.
4. **Publish.** With the author, tag `v0.2.0` on `main`; CI tests every
   platform and publishes to PyPI.

## After 0.2.0

### Evidence to gather

- **The bimodal small-sample residual.** It was measured on the rescaled
  subset test ([research ideas](docs/research-ideas.md), section 4.9).
  Measure it on the shipped test, with four thinning draws.
- **A second tissue for the count model.** The reference takes its latent
  biology and size factors from TCGA prostate tumours. Build it from a second
  tissue and repeat the calibration grid.
- **Paired designs and repeated measures: not validated.** They are strata
  of two or a few samples, which `strata=` accepts, but no such design has
  been measured. For pairs the area test becomes the paired permutation
  test, and the subset test's common fold the median of the paired log
  ratios. Validate them before claiming them. Continuous covariates are not
  supported; binning one into strata chosen before looking is possible but
  not validated.

### Method research

The ideas and their evidence are in
[research ideas](docs/research-ideas.md).

- **The subset test below 32 reads per sample.**
  - A fold that every case carries reads as a subset there. The shipped
    test's worst share at p ≤ 0.01 is 2.4% at 8–32 reads and 29% at 2–8
    ([validation](docs/validation.md#the-subset-test-under-the-count-model)).
  - A valid test there must include the fold's estimation in its null. The
    principled route is a mixture fit with a bootstrap, candidate A (ideas,
    section 4.4).
  - The count-exact pairwise scores are a valid DE statistic at every level,
    but blind to magnitude (ideas, section 2.1).
- **Normalization under composition change.**
  - When a tenth of the genes double in one group, RLE moves 5–6% and the
    area test calls 19.6% of unchanged genes at p ≤ 0.05
    ([validation](docs/validation.md#normalization-when-many-genes-change)).
  - Candidates: a diagnostic of how asymmetric the RLE ratios are between
    the groups; trimmed or reference-gene normalization (ideas, section 6).
  - That the subset test is unaffected was measured on the rescaled test
    (`count-model-composition-20260930.txt`, at commit `e4fd67f`). Measure
    it on the shipped test.
- **Calls near BH's threshold depend on the seed.**
  - On TCGA prostate two seeds agree on 81% of the subset test's calls, and
    on 90% at 128 reads or more. ETV4 sits at the threshold.
  - Candidate: report each gene's share of K seeds that call it. Try it
    first as a recipe in the manual (`seed=` exists). Add code only if it
    changes a decision a user makes.
- **Rarity.**
  - *Programmes and drivers.* Several of the rarest subsets are
    normal-tissue programmes, whose genes share members, and
    `rare_subsets()` lists the sharing. Scoring it is open: a driver can
    share members too (ETV1's 36 tumours include 8 of ISG15's 58).
  - *Crowded sizes.* Rarity ranks the breast ERBB2 amplicon about a fifth
    lower than the gap alone does: many breast genes have a subset of about a
    tenth of the tumours.
  - *Lowered subsets.* A lowered subset's gap is capped by its share of the
    cases, so the walk often reports the gene's raised side instead. Planted
    into prostate at 8–32×, lowered subsets were reported on their own side
    in 17–66% of genes. Scaling each gap by its cap would change the subset
    statistic, which would then need validating again.
  - *Light contamination.* A breast library at 0.5–5% of a prostate
    tumour's reads did not bring the host into the sample load's top ten. A
    detector would need a reference for the tissue expected.
  - *Abundance.* Within the validated range, rarity still rises with
    expression (Spearman 0.30 in prostate, 0.37 in breast).
- **A dataset-wide model of "expected".** A model of how a gene's
  distribution depends on its mean (mean–variance and shape, fitted across
  genes). It would give rarity a principled reference, and supply tails that
  a few controls cannot show.
- **A proof of the subset test's validity.** Its validity is measured, not
  proven.
  - Needed: the sampling Lorenz process r_N(L̂₀ − L̂₁), with
    r_N = √(n₁n₀/(n₁ + n₀)), and the conditional permutation process share
    a Gaussian limit, in a topology that supports the maximum.
  - Proven: the gap is unchanged by a fold; W1 is consistent under the
    null; the thinned area test is exact when the exposures are known.
    With estimated size factors it is measured.
  - Open: the conditional limit (moment control, tightness, regularity),
    the signed statistics, uniformity over gene distributions, and accuracy
    at 10⁻⁶.

### Design and scale

- **Performance at thousands of samples.** Measure 5,000–80,000 samples. A
  lone discovery, or genes right at BH's threshold, still need the full
  budget. Then:
  - vectorize the walk;
  - stream gene chunks from Arrow;
  - mind the memory: the result keeps genes × samples of expression.
- **One saddlepoint.** `mean_saddlepoint` keeps its unstratified code beside
  `strata_saddlepoint`, which computes the same tail with one stratum.
  Merging them changes p-values slightly, not bit for bit, so it needs a new
  confirmation.
- **The RNA100K workbench.** Render `notebooks/rna100k.qmd` (the
  compendium's plasma libraries, blocked by study) with the compendium. It
  has not been rendered with strata.
- **Subset p-values below the shuffle floor.** Multilevel splitting was
  prototyped on the native walk (2026-09-28, outside the repository) and is
  not safe as built. On outlier genes such as ETV4 its local swap moves
  cannot cross between the tail's modes, so some runs collapsed by many
  decades; larger populations shrank but did not remove a low bias, at a
  cost near brute force. It would only rank the strongest subsets, never
  change a call at the 10⁻⁶ floor.
- **Two subsets in one gene.** One walk makes one split, so a raised and a
  lowered subset in the same gene share it. A second split of each side
  (binary segmentation) would report both.
- **More controls.** Methods for pooling external normals (e.g. GTEx) with
  batch strata, and guidance on when that is exchangeable. Few controls cap
  small subsets: with 52 controls, a quarter of all labellings put a given 12
  tumours wholly among the cases ([validation](docs/validation.md#rarity)).
- **Native comparators.** Rerun DESeq2/edgeR and a classic outlier statistic
  (COPA or outlier sum) on the confirmation matrices for a publication.
- **Sparse assays.** Single-cell, spatial and cell-free RNA need their own
  exchangeability design and validation.
- **Measurement uncertainty.** Per-sample count kernels were explored and
  deferred; the derivations are in history at `f07d5f9`.

## Done for 0.2.0

- *23 October 2026:* rarity, with `subset_rarity`, `rare_subsets()` and
  `sample_load()` ([validation](docs/validation.md#rarity)).
- *23 October 2026:* decision-based stopping; TCGA breast's whole
  transcriptome in 158 s, beside another project's jobs
  ([validation](docs/validation.md#decision-based-stopping)).
- *25 October 2026:* the subset test thins rather than rescales, validated
  from 32 reads per sample (128 before), and averages its walk over four
  thinning draws ([validation](docs/validation.md#the-subset-test-under-the-count-model)).
- *25 October 2026:* strata in both tests
  ([validation](docs/validation.md#strata); constructions set aside in
  [research ideas](docs/research-ideas.md#9-strata-constructions-set-aside)).
