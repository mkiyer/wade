# TCGA gene-count examples for WADE 0.2.0

Design frozen on 2026-09-11 before inspecting the new marker p-values.
The user selected full gene counts for both cancers and omitted all
splice-junction analyses from these examples. Results and figures require
the author's approval before release.

## Inputs and provenance

Copy [the configuration template](../notebooks/tcga-config.example.json)
to a local location, fill in paths, and set `WADE_TCGA_CONFIG` to that JSON.
Relative paths resolve against the configuration's directory. Large data
and local paths stay outside tracked source.

- `counts`: Arrow IPC/Feather, genes in rows and library IDs as columns;
  finite, nonnegative integer gene counts. Only the selected TCGA columns
  are loaded. These are raw counts, not precomputed TPM/log expression.
- `genes`: an Excel table or tab-delimited table with unique `gene_id` and
  unique nonmissing `gene_name`, in exactly the count matrix's row order.
  Other gene annotations are retained in the filtering export.
- `samples`: Parquet with `library`, `sample`, `patient`, `study`,
  `tissue_of_origin`, `sample_class`, `library_protocol`, `preservation`,
  and `oncotree_code`. Study is `TCGA`; tissue is `Prostate` or `Breast`;
  sample classes are `tumor` and `normal_adjacent`.
- `annotations`: a frozen directory from the downloader below.
- `processing_reference`: document the count-producing workflow/version,
  reference genome and gene annotation, counting/assignment conventions,
  upstream gene exclusions, and the matrix/annotation row-order contract.

```sh
python tools/fetch_tcga_annotations.py /path/to/frozen-tcga-annotations
cd notebooks
WADE_TCGA_CONFIG=/path/to/tcga-config.json quarto render examples.qmd
WADE_TCGA_CONFIG=/path/to/tcga-config.json quarto render examples-breast.qmd
```

The downloader reads only public annotations and never sends local sample
lists or expression values to the service. It records URLs, UTC retrieval
times, response checksums and record counts. It reuses verified snapshots;
a changed or unrecorded file fails verification. Request a new snapshot by
using a new directory rather than overwriting the old one.

Sources: [TCGA prostate, Cell 2015](https://doi.org/10.1016/j.cell.2015.10.025),
[TCGA breast, Nature 2012](https://www.nature.com/articles/nature11412), and
the cBioPortal studies [prostate publication](https://www.cbioportal.org/study/summary?id=prad_tcga_pub),
[breast PanCancer](https://www.cbioportal.org/study/summary?id=brca_tcga_pan_can_atlas_2018)
and [breast TCGA clinical](https://www.cbioportal.org/study/summary?id=brca_tcga).
The PanCancer GISTIC profile supplies the discrete ERBB2 copy-number value;
`2` denotes its high-level amplification category, not an absolute copy count.

The local matrices are previously harmonized TCGA data, not a newly
downloaded GDC count release. The exact upstream processing manifest has
been requested and remains a release documentation requirement. Matching
row counts cannot establish row identity in an unlabeled matrix. Candidate
outputs therefore record that row-order assumption, plus hashes of the
selected count values, gene order, sample order and annotation inputs.
An exported NPZ contains all selected raw counts and labels, allowing the
numerical input to be preserved without hashing or copying unrelated cohorts.
An independently downloaded TCGA count release should not be advertised as
numerically identical to these harmonized inputs.

## Fixed preprocessing and primary populations

Use primary tumours and normal-adjacent tissue, excluding metastases.
Sort by patient, sample and library; keep one library per patient and
sample class. Export excluded duplicates. TCGA sample IDs are the first
15 barcode characters, and the first 12 must agree with the patient ID.
Annotation joins are many-to-one and preserve the exact count-column order.
Missing clinical labels are reported, not imputed as negatives.

Prostate includes all eligible patients. Breast is restricted to female
patients, with sex obtained from TCGA clinical annotations and agreement
checked across sources. Male and unknown-sex records are exported as
exclusions. This defines the target population before inspecting marker
effects. Histology, protocol, preservation and annotation coverage are
reported; they are not silently used for additional outcome-driven exclusion.

The production recipe fits DESeq2-compatible RLE on **all supplied rows before
its testing filter**, after sample eligibility and patient separation. The
primary comparison has its own fitted object; the four breast subtype
comparisons share one fit on their eligible tumour-only cohort. Subsetting a
fitted object never re-centers its factors. The count denominator is that
relative size factor, and the test uses `count / size_factor` without added
noise; figures and descriptions use log2. Raw full-input totals accompany library QC, independently of the
RLE factors. No genomic span is used as a transcript length.

This ordering distinguishes initial sample/reference QC from the subsequent
testing filter; it is not a general prohibition on prefiltering. RLE's
reference assumption must be justified. It does not remove depth-dependent
sampling variance or confounding, and when many genes change together the
reference itself can shift.

A gene is eligible if it has at least 10 raw counts in at least five pooled
samples and pooled mean CPM of at least 0.1. No case/control or marker label
enters this filter. Keep the resulting gene universe constant across primary
and sensitivity comparisons within each cancer, and keep factors fixed within
each analysis family as described above.
“Full gene counts” describes the assay and whole supplied feature universe;
it does not promise that every annotated human gene was in the upstream matrix.

The independent primary analysis keeps normals and excludes the corresponding
tumours. Every patient appears once. Paired analyses in earlier development
are not part of the migrated example; they are preserved as historical work.

| Cohort | Unique eligible tumours / normals | Independent primary | Tested genes |
|---|---:|---:|---:|
| Prostate | 497 / 52 | 445 / 52 | 20,517 |
| Female breast | 1,083 / 112 | 971 / 112 | 21,431 |

These are design counts, not approved results.

## Inference and interpretation

Both primary comparisons use the defaults: both directions, the linear scale,
seed 1 and the 999,999 permutation cap, with one BH family over every
gene-direction pair. Every output includes the stopping records and the
fitted normalization provenance.

The four breast subtype-versus-other-three contrasts use all eligible
annotated tumours, one per patient. Normal-like and unknown labels are excluded
only from those secondary contrasts. They use the same test-gene universe,
tumour reference, seed and prespecified 99,999 cap. One additional BH family
covers all four contrasts × both directions × all tested genes; with its 1e-5
floor a discovery needs several genes at the floor together. This is an
exploratory expression-label association analysis.

Prostate fusion-confirmed, expression-high, negative and unknown labels stay
separate. SPINK1 is an expression-high comparator. The notebook reports observed
expression and supplied annotations; it does not assign fusion membership.
There is no selected ERG-negative inference in this migrated example.
PAM50 is expression-derived, so marker agreement is not independent validation.
Receptor status and DNA copy-number calls provide complementary measurements
in the same patients, not an independent cohort. HER2-enriched is not
synonymous with clinical HER2 positivity, nor basal-like with triple-negative
disease.

`bulk_shift` is a log2 median difference. `upper_tail` and `lower_tail` are
decile-mean differences minus that shift, so a lowered bulk and a raised
upper tail can coexist. `padj_up` tests the upward area; it does not infer an
affected fraction or an individual's rarity. With 52 prostate normals, k
tumours above every normal cannot reach an upward p-value below
C(445, k) / C(497, k); the prostate notebook reports that floor for its
fusion markers.
