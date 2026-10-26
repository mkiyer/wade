# TCGA gene-count examples

The [prostate](../notebooks/examples.qmd) and
[breast](../notebooks/examples-breast.qmd) notebooks run WADE on TCGA gene
counts. This page is their data contract: the inputs, who is eligible, the
gene filter and the settings. The [manual](manual.md) explains the method and
its limits.

## Inputs and provenance

- **Counts.** STAR gene counts from the RNA100K compendium, release
  `rna100k_2026-01-28`: 62,846 rows, the 62,754 genes of GENCODE v44 on the
  GRCh38 primary assembly and 92 ERCC spike-ins.
  - The lab processed the TCGA reads itself, with its `mibulkrnaseq`
    pipeline. The counts are not a GDC count release and are not
    numerically identical to one.
  - The examples test every supplied row that passes the filter below.
- **Metadata.** Harmonized by `rnaseq-metadata-harmonize`.
- **Bundles.** The lab's `rna100k` package extracts one bundle per cancer:
  TCGA primary tumours and normal-adjacent tissue, and GTEx normal tissue,
  one library per patient and sample class.
  - Each bundle records the release, the query, the annotation version read
    from the GTF, and checksums of its files. The loader verifies them.
  - Rows are joined by GENCODE v44 ID. The bundle's unique `label` (the
    symbol, or `symbol|gene_id` where symbols repeat) names genes in WADE.
- **GTEx normals.** The notebooks use the TCGA libraries only. Validation
  uses the prostate bundle's GTEx normals ([strata](validation.md#strata)).
- **Open gap.** The count pipeline's own record, its workflow and genome
  versions, is not yet part of the bundles ([ISSUES](../ISSUES.md)).

```sh
rna100k extract cohorts/prostate-tcga-gtex --one-per patient_uid,sample_class \
  --where "tissue_of_origin = 'Prostate' AND specimen_class = 'in_vivo' AND ((study = 'TCGA'
           AND sample_class IN ('tumor', 'normal_adjacent')) OR (study = 'gtex' AND sample_class = 'control'))"
```

The breast bundle uses the same query with `'Breast'`.

**Rendering.** Copy [the template](../notebooks/tcga-config.example.json) and
set `WADE_TCGA_CONFIG` to it. The configuration names the two bundles and the
annotation snapshot; relative paths resolve against its folder.

```sh
python tools/fetch_tcga_annotations.py /path/to/frozen-tcga-annotations
cd notebooks
WADE_TCGA_CONFIG=/path/to/tcga-config.json quarto render examples.qmd
WADE_TCGA_CONFIG=/path/to/tcga-config.json quarto render examples-breast.qmd
```

**Annotations.** `tools/fetch_tcga_annotations.py` freezes them.

- It reads only public annotations, and never sends local sample lists or
  expression values to the service.
- It records URLs, UTC retrieval times, response checksums and record counts.
- It reuses verified snapshots; a changed or unrecorded file fails
  verification. A new snapshot goes in a new directory.
- Sources: [TCGA prostate, Cell 2015](https://doi.org/10.1016/j.cell.2015.10.025),
  [TCGA breast, Nature 2012](https://www.nature.com/articles/nature11412), and
  the cBioPortal studies
  [prostate publication](https://www.cbioportal.org/study/summary?id=prad_tcga_pub),
  [breast PanCancer](https://www.cbioportal.org/study/summary?id=brca_tcga_pan_can_atlas_2018)
  and [breast TCGA clinical](https://www.cbioportal.org/study/summary?id=brca_tcga).
- The PanCancer GISTIC profile supplies the discrete ERBB2 copy-number
  value. `2` is its high-level amplification category, not an absolute copy
  count.

## Eligibility

- Primary tumours and normal-adjacent tissue from TCGA; no metastases.
- One library per patient and sample class, after sorting by patient, sample
  and library. Excluded duplicates are exported.
- A TCGA sample ID is the first 15 barcode characters, and its first 12 must
  equal the patient ID.
- Annotation joins are many-to-one and keep the count columns' order.
  Missing clinical labels are reported, never imputed as negatives.
- **Prostate:** every eligible patient.
- **Breast:** female patients only. Sex comes from the TCGA clinical
  annotations, with agreement checked across sources. Male and unknown-sex
  records are exported as exclusions.
- Histology, protocol, preservation and annotation coverage are reported, not
  used to exclude samples.
- **The independent design** keeps every normal-adjacent sample and the
  tumours of the patients without one, so every patient appears once.

| Cohort | Eligible tumours / normals | Independent design, tumours / normals | Tested genes |
| --- | ---: | ---: | ---: |
| Prostate | 497 / 52 | 445 / 52 | 24,246 |
| Female breast | 1,083 / 112 | 971 / 112 | 25,328 |

The independent breast design also holds 1,083 samples: each of the 112
normals has a patient with an eligible tumour, which the design leaves out.

## Size factors and the gene filter

- **Size factors.** RLE on every supplied row, fitted after sample
  eligibility and before the filter.
  - The primary design has its own fit. The four breast subtype contrasts
    share one, fitted on their tumours.
  - Selecting samples from a fit keeps its factors; nothing is re-centred.
- **The filter.** A gene is tested if it has at least 10 raw counts in at
  least five samples and a mean CPM of at least 0.1, over all eligible
  samples pooled. No label enters it.
  - Every comparison within a cancer tests the same genes.
  - This is the examples' recipe, not a universal threshold.
- **Library QC** reads raw totals over all supplied rows, independent of the
  size factors. No genomic span is used as a transcript length.
- **What the values are.** As in every WADE run, both tests use thinned
  counts, and the subset columns read jittered normalized counts
  ([depth equalization](manual.md#depth-equalization)). Figures plot
  log2 of the jittered normalized count + 1. What RLE assumes, and when it fails, is in
  the [manual](manual.md#size-factors).

## Settings

- **Primary comparisons,** tumours against normals: `wade()`'s defaults.
  - `direction="both"`, `pvalues="auto"`, `relative_error=0.2`, `fdr=0.05`,
    `max_permutations=999_999`, seed 1.
  - Each test has one BH family over the tested genes.
  - With `fdr=0.05`, the called genes report p-value bounds
    ([p-values](manual.md#p-values)).
- **Records.** Every run writes its full table and manifest: the stopping
  records, the depth kept and the normalization's provenance.
- **Breast subtype contrasts.** Each of Luminal A, Luminal B, HER2-enriched
  and Basal-like against the other three.
  - The tumours: every eligible tumour annotated with one of the four, one
    per patient. That is 945 tumours: 499, 197, 78 and 171. Normal-like and
    unknown labels are left out of these contrasts only.
  - The settings: the subtype size factors, the same tested genes, seed 1,
    `fdr=0.05` and `max_permutations=99_999`.
  - Each contrast decides at `fdr=0.05` within its own family, so its called
    genes report bounds certified for that family.
  - The notebook then pools the four contrasts into one BH family per test:
    4 × 25,328 = 101,312 hypotheses, exported as `q_area_four_contrasts`
    and `q_subset_four_contrasts`. Every reported p-value is at least its
    finished one, so the pooled BH calls a subset of what it would call on
    finished runs. The pooled family is conservative, and its decisions are
    not those of finished runs.
  - The shuffled floor is 10⁻⁵. A subset discovery in the pooled family
    needs at least 21 genes at the floor (101,312 × 10⁻⁵ / 0.05 = 20.3).
  - This is an exploratory association with expression-derived labels.

## Labels and their reading

- **Prostate fusions.** Fusion-confirmed, expression-high, negative and
  unknown labels stay separate. SPINK1 is an expression-high comparator.
  - The notebook reports expression and the supplied annotations; it does
    not assign fusion membership.
  - Only tumours annotated fusion-positive or negative enter precision and
    recall.
- **Breast.** PAM50 is expression-derived, so marker agreement is not
  independent validation.
  - Receptor status and DNA copy-number calls are complementary measurements
    in the same patients, not an independent cohort.
  - HER2-enriched is not clinical HER2 positivity, and basal-like is not
    triple-negative disease.
  - ERBB2 amplification is GISTIC category `2`; other calls count as
    negatives, and missing calls are left out.
- **What WADE sees.** The notebooks compare the ERG, ETV1, ETV4 and FLI1
  subsets with the fusion annotations, and the ERBB2 and GRB7 subsets with
  the GISTIC calls. WADE never sees either. What a subset does and does not
  say is in the [manual](manual.md#limits).
