# Open issues

## Before releasing 0.2.0

1. **Confirm the build (protocol version 7).** *Done 26 October 2026: all 27
   checks passed*
   ([validation](docs/validation.md#confirmation-protocol-version-7)).
   - *The build:*
     - the area test on counts thinned until every sample's size factor
       equals the smallest;
     - the area and the area fraction as effects;
     - the subset test on depth-equalized counts, its fold taken out of the
       higher group by thinning and its walk averaged over four thinning
       draws (25 October 2026), with its validated range of 32 reads per
       sample and a warning below it;
     - strata (25 October 2026): shuffles within strata, the stratified
       saddlepoint, thinning within each stratum, Mantel–Haenszel effects,
       and the subset test's one fold and level per stratum;
     - no log2 scale and no `subsets="either"`.
   - *The confirmation* (`research/protocols/confirmation-v7.json`) was
     predeclared and frozen at commit `74c0752` before any of its data
     existed. Its 27 checks, each with power 0.9 at an unacceptable error
     rate, share a 5% chance of any false failure.
     - Per-gene null rates over more than 400,000 genes per test:
       4.93–5.05% at p ≤ 0.05 and 0.079–0.115% at 0.001.
     - Simulated transcriptomes: any discovery at q ≤ 0.05 in 5.6% (area)
       and 1.5% (subset) of 480 with no change; false discovery proportions
       4.8% and 2.2% over 60 with planted changes.
     - Power, descriptive: with ten cases per control the area test called
       9% of genes raised 1.5× in every case (Wilcoxon 48%), and the subset
       test 1–2% of raised tenths, a floor any valid shuffle test shares.
     - `research/confirm.py --output .release/0.2.0/confirmation-v7`, about
       4.2 hours with six threads; `--evaluate` scores the checks.
2. **Review.** The author approves:
   - the rendered demo, prostate and breast notebooks, all re-rendered on
     25 October 2026 with strata and the subset test's four thinning draws
     (breast in 454 s, where it took 89 minutes before decision-based
     stopping);
   - the documents' figures, tables and claims.

   Record the TCGA count-pipeline provenance: workflow, genome and
   annotation versions, and row order.
3. **Publish.** Tag `v0.2.0` on `main`; CI tests every platform and publishes.

## Method research

The ideas and their evidence are in
[docs/research-ideas.md](docs/research-ideas.md).

- **The subset test below 32 reads per sample.**
  - Since 25 October 2026 the test thins rather than rescales, which took
    its validated range from 128 to 32 reads (`research/subset_range.py`).
  - Below 32, a fold that every case carries still reads as a subset (2.2%
    at p ≤ 0.01 at 8–32 reads, 16% at 2–8).
  - A valid test there must include the fold's estimation in its null. The
    principled route is a mixture fit with a bootstrap, candidate A (ideas,
    section 4.4).
  - The count-exact pairwise scores are a valid DE statistic at every level,
    but blind to magnitude (ideas, section 2.1).
- **The subset test's small-sample residual** on bimodal nulls (ideas,
  section 4.9).
- **Normalization under composition change.**
  - When a tenth of the genes double in one group, RLE moves about 5% and the
    area test calls 20% of unchanged genes at p ≤ 0.05.
  - Candidates: a diagnostic of how asymmetric the RLE ratios are between
    the groups; trimmed or reference-gene normalization (ideas, section 6).
- **The thinning draw.**
  - The area test's calls below about 32 reads per sample move with the
    seed: on TCGA prostate, 80–86% of the genes one seed calls there are
    called by another, against 97% at 128 reads or more.
  - The subset test's calls move more. With one thinning draw 70% agreed,
    81% at 128 reads or more, against 93% for the rescaled test and for new
    shuffles of one thinned pool (`subset-seeds-20261025.txt`); more precise
    p-values did not help. Four draws take it to 81% and 90%
    (`subset-pooled-native-seeds-20261025.txt`).
  - *Done 25 October:* the subset test averages its walk over four
    thinning draws ([ideas](docs/research-ideas.md), section 3.4), natively,
    at about four times its cost. It keeps the level and power at 32 reads
    and above and steadies the calls there; below 32 it is more
    anti-conservative under a fold than one draw.
  - Whether to offer a stability check is open.
- **Rarity.**
  - **Done 2026-10-23:** `subset_rarity`, `rare_subsets()` and
    `sample_load()` ([validation](docs/validation.md#rarity)).
  - *Programmes and drivers.* About half of the rarest subsets are a
    normal-tissue programme, whose genes share members, and `rare_subsets()`
    lists the sharing. Scoring it is open: a driver can share members too
    (ETV1's 36 tumours include 8 of ISG15's 58).
  - *Crowded sizes.* Rarity ranks the breast ERBB2 amplicon about a fifth
    lower than the gap alone does: many breast genes have a subset of about a
    tenth of the tumours.
  - *Lowered subsets.* A lowered subset's gap is capped by its share of the
    cases, so the walk often reports the gene's raised side instead. Planted
    into prostate at 8–32×, lowered subsets were reported on their own side
    in 17–66% of genes. Scaling each gap by its cap would change the subset
    statistic, which would then need validating again.
  - *Light contamination.* A breast library at 5% of a prostate tumour's reads
    did not top the sample load. A detector would need a reference for the
    tissue expected.
  - *Abundance.* Within the validated range, rarity still rises with
    expression (Spearman 0.30 in prostate, 0.37 in breast, with the range
    at 32 reads; 0.21 and 0.29 at 128).
- **A dataset-wide model of "expected".** A model of how a gene's
  distribution depends on its mean (mean–variance and shape, fitted across
  genes). It would give rarity a principled reference, and supply tails that
  a few controls cannot show.
- **Performance at thousands of samples.**
  - **Done 2026-10-23:** decision-based stopping. Each gene's shuffles stop
    once its BH decision at `fdr` is settled, with exactly the decisions of
    finished runs (Fischer, Barry and Ramdas 2026). TCGA breast (25,328 genes
    × 1,083 samples) went from 89 minutes to 27 seconds; with the subset
    test's four thinning draws it took 158 seconds beside another project's
    jobs ([validation](docs/validation.md#decision-based-stopping)).
  - **Remaining.** Measure 5,000–80,000 samples. A lone discovery, or genes
    right at BH's threshold, still need the full budget. Then:
    - vectorize the walk;
    - stream gene chunks from Arrow;
    - mind the memory: the result keeps genes × samples of expression.
- **Strata: built 25 October 2026** ([plan](docs/strata-plan.md), vetting
  `research/strata.py`, [validation](docs/validation.md#strata)).
  - *The area test.* Shuffles and a stratified saddlepoint, thinning within
    each stratum, and Mantel–Haenszel effects. It is exact against
    enumeration, and calibrated where the unstratified test calls up to 60%
    of null genes.
  - *The subset test.* One fold shared by every stratum and a level per
    stratum, by thinning (S4b, thinned). Four other constructions failed on
    small or lopsided strata, or under a fold.
  - *Confirmed* with stratified designs (item 1). *Left:*
    `notebooks/rna100k.qmd`, which blocks the RNA100K compendium by study,
    can now run.
  - Pairing and continuous covariates remain out of scope.

## After 0.2.0

- **Subset p-values below the shuffle floor.** Multilevel splitting was
  prototyped on the native walk (2026-09-28, outside the repository) and is not
  safe as built: on outlier genes such as ETV4 its local swap moves cannot cross
  between the tail's modes, so some runs collapsed by many decades, and larger
  populations shrank but did not remove a low bias, at a cost near brute force.
  Ties were not the cause, so jitter would not help. It would only rank the
  strongest subsets, never change a call at the 10⁻⁶ floor.
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
