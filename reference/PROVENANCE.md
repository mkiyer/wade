# PROVENANCE — where the reference material in this repository came from

This file records every WADE-bearing location in the cfRNA analysis repository,
what each one is, and what was done with it: **copied verbatim**, **extracted**,
or **deliberately left behind**. Everything under `reference/` is cfRNA material
— reproduced byte-for-byte, or transcribed with its origin stamped in its own
header — except the orientation prose and the renv scaffolding listed in
[Files written for this repository](#files-written-for-this-repository).

## The source

| | |
|---|---|
| Source repository | `mkiyer/cfrna`, local path `/Users/mkiyer/proj/cfrna` |
| Git SHA | `828f2f1c6a10afd182fee0e119ba3595d98b6a72` |
| Commit date | 2026-08-17 01:07:40 −0400 |
| Staging date | 2026-08-17 |
| Working tree | Clean for every file staged here. `git status` reported two untracked directories, `scratch_liver/` and `wade_check2/`; neither is a modification to a tracked file, and neither was staged (see [Left behind](#left-behind)). |

The source repository was treated as strictly read-only. No file in it was
created, modified, moved or deleted.

### Diffing this material against the evolving cfRNA repo

Every file staged here has a fixed origin path and a fixed SHA, so a future
reader can recover the exact bytes this repository was built from without
relying on the copies here:

```sh
# the state of a source file at the staging SHA
git -C /path/to/cfrna show 828f2f1c6a10afd182fee0e119ba3595d98b6a72:cfrna_analysis/wade.R

# what has changed in it since
git -C /path/to/cfrna diff 828f2f1c6a10afd182fee0e119ba3595d98b6a72..HEAD -- cfrna_analysis/wade.R

# confirm a staged copy is still byte-identical to its origin
git -C /path/to/cfrna show 828f2f1c6a10afd182fee0e119ba3595d98b6a72:cfrna_analysis/wade.R \
  | shasum -a 256    # compare against the table below
```

For an extracted notebook section, the header of the extracted file names the
source path, the line range and the sha256 of the **whole** source notebook, so
the same `git show` recovers the notebook and the line range locates the
section within it. Line numbers are only valid at this SHA; if the notebook has
changed, locate the section by its heading text instead.

## Inventory

Checksums are sha256 of the file as it exists at the staging SHA. For copied
files the same checksum is the checksum of the staged copy — that is what
"verbatim" means here, and it was verified with `cmp` as well as by digest.

### Copied verbatim

Method and downstream code. Staged by the R track; checksums verified
independently here.

| Source path | Staged as | sha256 | Size |
|---|---|---|---|
| `cfrna_analysis/wade.R` | `reference/R/wade.R` | `19dbef718d213b7dd306703d82b9404d330162fbc8e06b06b7a0bd8373d32059` | 317 lines, 15,957 B |
| `cfrna_analysis/R/wade_contrasts.R` | `reference/R/downstream/wade_contrasts.R` | `cdfde7f993daa7add4ca972eeff6fea4955565ef23491bec8bb4f087556b7867` | 583 lines, 31,035 B |
| `cfrna_analysis/R/wade_figures.R` | `reference/R/downstream/wade_figures.R` | `565f9d5da38629e438a07aaa371e2bd3d53f363845929b84bfda076655d5d87a` | 573 lines, 30,269 B |
| `cfrna_analysis/R/control_strata.R` | `reference/R/downstream/control_strata.R` | `1d677c07b11ff9cd4667cdc674aa8fbfc62765161d9e3a300f264209b47c3cce` | 266 lines, 13,911 B |

Source documents. Staged by this track.

| Source path | Staged as | sha256 | Size |
|---|---|---|---|
| `WADE_REPO_SCOPE.md` | `reference/docs/WADE_REPO_SCOPE.md` | `cb1515c34eb1a040437ec1461a2696c089be2d31b4d33f1a2eac76834fe2ef67` | 215 lines, 12,552 B |
| `cfrna_analysis/supplement_wade.qmd` | `reference/docs/supplement_wade.qmd` | `78d55c39197d181a48f925c893f3a37d3941752103fbcde40c60dc8cf20fe33b` | 231 lines, 12,581 B |
| `cfrna_analysis/section4_wade.qmd` | `reference/docs/section4_wade.qmd` | `869a6673a47d0defdd8d5c8119f38712f78135fb466c689974269903ffc46ca3` | 286 lines, 17,205 B |
| `cfrna_analysis/section4_wade_setup.qmd` | `reference/docs/section4_wade_setup.qmd` | `7be7fcf53822cb1390098c2c734846cb3272d57b485dff9b4ad22da26a4c9f8c` | 359 lines, 17,978 B |

What each one carries:

- **`wade.R`** — the method module. Ten top-level function definitions (eight
  public, two dot-private); see [Discrepancies](#discrepancies-found-during-staging)
  on the "nine functions" count. No knowledge of diseases, panels, cohorts or
  the store; declares `matrixStats`, `tibble` and `dplyr` as its only
  dependencies.
- **`wade_contrasts.R`** — the cfRNA consumer: contrast registry, stratum
  resolution, the pre-run feasibility and confounding screen, cohort plumbing
  and cache-key discipline. Staged as evidence of what the method's API is
  actually asked for, not as method code.
- **`wade_figures.R`** — the figure grammar for the discovery section, including
  the display-layer guard on `tail.conc`. Staged as evidence of which
  visualizations the method has to support.
- **`control_strata.R`** — the control taxonomy. Entirely clinical vocabulary;
  see [Clinical vocabulary](#clinical-vocabulary-a-judgement-for-the-user).
- **`WADE_REPO_SCOPE.md`** — the extraction analysis written before this
  repository existed: which functions are method and which are plumbing, where
  the normalization boundary should fall, five named cross-language parity
  hazards, and a recommended sequence. This is the closest thing to a design
  brief that exists.
- **`supplement_wade.qmd`** — the method supplement (S7.1–S7.7): why not a
  t-test, the statistic, why permutation with a tail refinement, what the
  permutation test is protecting against, the quantile grid as a sample-size
  constraint, what WADE does not do, implementation and cost.
- **`section4_wade.qmd`** — the discovery-section fragment: why this cohort has
  no single control group, and the screen that runs before any contrast does.
- **`section4_wade_setup.qmd`** — the computation block behind both fragments.
  Staged because it is the operational definition of every quantity the prose
  quotes: what was computed, from what, once.

### Extracted

Notebook prose and chunk code, transcribed to markdown. These are the only
files under `reference/` that are neither byte-identical copies nor orientation
prose, so their transformation is stated exactly: the line range was taken
verbatim and the code-fence info string was rewritten from `` ```{r} `` to
`` ```r `` so the R highlights as R in a plain markdown viewer. Nothing else
changed. Chunk options (`#|`), Quarto div syntax, cross-references and inline
`` `r ... ` `` expressions are all preserved unevaluated. Each file states its
source path, line range and the source notebook's sha256 in its own header.

| Source | Lines | Staged as | Staged file sha256 |
|---|---|---|---|
| `cfrna_analysis/analysis_cfrna_v7.qmd` §9 (9 preamble, 9.1–9.6) | 3727–4415 | `reference/docs/notebook_sections/v7_section9_wade_discovery.md` | `c289063d2037b242acd45d15065fad689d509a6901d615a0d0e2261f8d3894e9` |
| `cfrna_analysis/analysis_cfrna_v8.qmd` §4.4 (4.4.1–4.4.4) | 3784–4282 | `reference/docs/notebook_sections/v8_section4.4_wade_discovery.md` | `349cf32c4ba6cc751248f7a488ee68e1d46bcc883a3606743aea463844c26b9a` |
| `cfrna_analysis/analysis_cfrna_v8.qmd` S7 (S7.1–S7.7) | 4979–5211 | `reference/docs/notebook_sections/v8_S7_wade_method_supplement.md` | `1bca11e8b4ee83944373781ab208693abfdf61bb860422339e8dc223e5ef41a9` |

The R track additionally extracted the two §9.6 chunks — `wade-validation-sim`
and `fig-wade-validation`, within the v7 range above — into
`reference/R/validation_sims_v7.R`, a standalone runnable script, together with
its outputs (`validation_sims_v7.png` and two small CSVs). That extraction is
the R track's to describe; the same material is present as chunk code inside
`v7_section9_wade_discovery.md`, so the two are cross-checkable. Note the
transformation there is heavier than in this directory: the script parameterises
the cohort geometry the notebook read from live objects, so it is
verbatim-in-substance rather than byte-identical.

Source notebook checksums at the staging SHA — needed to verify a line range,
since the extracted files quote them:

| Notebook | sha256 | Size |
|---|---|---|
| `cfrna_analysis/analysis_cfrna_v7.qmd` | `16bb7bffda7ef9a44a11e5eb67598340960744391f4e0dccf67287c40e8319cb` | 4,627 lines |
| `cfrna_analysis/analysis_cfrna_v8.qmd` | `2aacf2ae50b82415da671c78bdc45e5a2717fe018663504c083572e4aa02bae1` | 5,450 lines |

Why these ranges and not others:

- **v7 §9** is where WADE was first applied to a cohort and where its operating
  regime was established: three contrasts on one clinical grouping, the finding
  that no gene survives BH-FDR on either axis, the decision to nominate by rank
  score instead, and the §9.6 simulations that are the method's only
  ground-truth validation. The range ends at the close of §9's prose. The
  `ss-load` chunk that follows belongs to §10 (single-sample analysis) and is
  excluded.
- **v8 §4.4** is the rewrite. Where v7 ran three contrasts on one grouping, v8
  enumerates nine candidates and screens each before running it — §4.4.2 is
  where the method acquires a refusal step, which the v7 material has no trace
  of. §4.4.3 reads the nominations; §4.4.4 repeats the screen on a second
  cohort.
- **v8 S7** is the supplement as the reader of the assembled notebook saw it.

### Overlap between the extracted notebook sections and the verbatim fragments

The v8 notebook does **not** use `{{< include >}}`; it was checked for the
directive and has none. The fragments were pasted into the notebook, so the
notebook ranges overlap the verbatim fragments — but not exactly, and the
differences are the reason both are staged rather than one being dropped as a
duplicate.

- **`section4_wade.qmd` vs v8 §4.4.** The fragment's 286 lines correspond to
  the first ~137 lines of the notebook's 499-line §4.4. Within the overlap the
  two differ on section numbering (the fragment writes `4.x`, the notebook
  `4.4`), on callout-title syntax (the fragment uses a `##` heading inside the
  div, the notebook bold text — this renderer is rmarkdown, not quarto), and on
  one cross-reference (`@sec-wade-method` in the fragment, an explicit
  `[S7. WADE](#sec-wade-method)` link in the notebook, for the same reason).
  **Everything from §4.4.3 onward — the nomination reading, the recurrence
  audit, the figure gallery, the both-cohorts repeat — exists only in the
  notebook.** That is the interpretive material, and it is why the extraction
  was worth doing.
- **`supplement_wade.qmd` vs v8 S7.** Substantively identical: the diff is two
  lines — the same callout-title syntax difference, and one trailing blank line.
  The extracted copy is kept anyway so the notebook range is auditable without
  a diff, and because the S7.1–S7.7 numbering in the notebook is what the
  discovery sections cross-reference. **The verbatim fragment is the source of
  record**; the extracted file shows what the reader actually saw.
- **`section4_wade_setup.qmd` vs v8.** The setup fragment was inlined into the
  notebook as chunk `wade-setup` (v8 lines 2718–3066). Of the fragment's 170
  substantive code lines, 160 appear verbatim inside that chunk and all 170
  appear somewhere in the notebook; all 101 of its substantial comment lines
  appear in the chunk. The fragment is staged verbatim and the chunk is **not**
  separately extracted — it is the same code, and the fragment is the version
  with the paste-in instructions attached.

### Left behind

Nothing in this list is staged. For each, the reason.

| Location | What it is | Why not staged |
|---|---|---|
| `cfrna_analysis/cache/wade/*.rds` | **76 files, 18.9 MB.** Serialized WADE result frames — per-gene statistics computed from real patient cohorts, one file per (contrast × cohort × cache key), spanning 22 distinct contrast/cohort combinations. | Patient-derived. See [Patient data](#patient-derived-data-and-the-public-release). |
| `cfrna_analysis/cache/wade_results.rds` | **5.0 MB.** An earlier combined results object, same character. | Patient-derived. |
| `cfrna_analysis/disc_tables/wade_*.csv` | **27 files, 7.5 MB.** The exported discovery products: per-gene nomination tables (3,041 rows each), cross-contrast classifications, and the contrast-screen summaries. | Patient-derived at gene level. The screen summaries are aggregate-only, but they were left behind with the rest rather than split — nothing in this repository needs them. |
| `wade_check2/` (untracked) | Four PNGs: `curves.png`, `volcano.png`, `screen.png`, `cross.png`. | Rendered figures built from cohort data. Also untracked scratch. |
| `scratch_liver/` (untracked) | Untracked scratch analysis, including `liver_program_scores_by_library.csv` and a rendered PNG. Not WADE. | Per-library patient data; not WADE material. |
| `cfrna_analysis/analysis_cfrna_v7.qmd`, `analysis_cfrna_v8.qmd` | The full notebooks, 4,627 and 5,450 lines. | Only the WADE ranges are relevant; those are extracted above. The rest is cfRNA QC, normalization, cohort definition and per-patient reporting. |
| `cfrna_analysis/analysis_cfrna_v7.html`, `analysis_cfrna_v8.html` | Rendered notebooks. Mention WADE 4 and 209 times. | Rendered output containing cohort figures, per-library tables and resolved cohort numbers throughout. |
| `cfrna_analysis/R/group_structure.R` | Mentions WADE three times, all in prose: whether two groups are separable at all, how much a nomination list from a marginal contrast is worth, and what a drill-down needs. | Not WADE code — PCA of the analysis matrix. The three comments are context, not method. |
| `cfrna_analysis/R/normalize.R` | One mention: a note that handing a `(counts, normalizer)` pair to WADE expects sizes computed a particular way. | Not WADE code. Relevant only to the normalization-boundary question, which `WADE_REPO_SCOPE.md` already states. |
| `cfrna_analysis/R/patient_html.R` | One mention: adjusted *p* is uninformative for overlapping sets, as with WADE. | Not WADE code. |
| `cfrna_analysis/render_v8.R` | Two mentions: a render gate that exists because a WADE fragment shipped quarto cross-reference syntax into an rmarkdown pipeline, and a gene-symbol allowlist containing the string `WADE`. | Notebook build tooling. Recorded here because it explains the callout/cross-reference differences noted in [Overlap](#overlap-between-the-extracted-notebook-sections-and-the-verbatim-fragments). |
| `cfrna_analysis/R/README.md` | Two table rows describing `wade_contrasts.R` and `wade_figures.R`. | A module index for the cfRNA repo. Superseded by this file and by `reference/R/README.md`. |
| `in_silico_blood_mixing/.../batch_0002_counts.npz` | A case-insensitive grep for "wade" matches inside this compressed binary. | **Not a reference.** The match is the byte sequence `WaDe` inside deflate-compressed data — coincidence, verified by inspecting the surrounding bytes. Recorded so a future audit does not re-discover it as a lead. |

## Patient-derived data and the public release

`mkiyer/wade` is intended for public release. The staged tree was audited
directly rather than assumed clean. The audit covered every file staged under
`reference/` by both tracks, excluding `reference/R/renv/library/` — a vendored
copy of renv itself, gitignored by `reference/R/renv/.gitignore` and never
tracked. It is re-runnable:

```sh
git status --porcelain --untracked-files=all reference/ | awk '{print $2}' \
  | grep -v 'renv/library/' \
  | xargs grep -aoIhE 'mctp_LBX[0-9]+|mctp_MI_[0-9]+|LBX[0-9]{3,}|SI_[0-9]{5}|\bMRN\b|\bDOB\b'
```

**Identifier scan — zero hits.** Every staged file was searched for library
identifiers (`mctp_LBX####`, `mctp_MI_####`, bare `LBX####`), the
flowcell/specimen suffixes those ids carry (`SI_#####`, run-id-shaped tokens),
`MRN`, `accession`, and dates of birth or collection. **No pattern matched in
any staged file.** The one hit the command above now returns is in this
document, in the sentence above, naming the patterns it searched for. For
calibration: the whole cfRNA repository contains exactly one literal library id
in a text file — in `cfrna_analysis/METADATA_ALIAS_BRIDGE.md`, which is not
WADE-bearing and is not staged.

**No cohort fingerprints.** Seven staged lines mention the word "fingerprint",
all of them describing the cache-key discipline (`cd$fingerprint`, "store
fingerprint, cohort name, gene count"). No literal hash value is present: a
scan for hex strings of 12+ characters returns only the git SHAs and file
checksums written by this staging process, in the headers of the three extracted
files.

**No cohort data files, and no figure built from cohort data.** Every staged
file is text (`text/*` or `application/json`) with one exception:
`reference/R/validation_sims_v7.png`, the three-panel method-validation figure
produced by `reference/R/validation_sims_v7.R`. That script reads **no data at
all** — it takes the two group sizes (77 cases, 18 controls) as scalar
parameters and simulates every count it plots, from a fixed seed. Its two
companion CSVs (`_calibration.csv`, 4 rows; `_power.csv`, 7 rows) are likewise
simulation output. Nothing in this trio derives from a patient. Beyond it: no
`.rds`, no `.npz`, no archive, and no figure rendered from the cohort.

**No store paths.** Nothing staged references the Dropbox project tree, the
`hulkrna` Zarr store, or any absolute path into the user's filesystem.

### Aggregate numbers were kept, deliberately

The copied and extracted documents do contain cohort numbers, and this line was
drawn on purpose rather than by omission. What is kept:

- **Group sizes** — "77 cases, 18 controls", "33 controls", "5–7 patients in the
  smaller group". These are the reason the method's limits are what they are:
  the quantile grid is `min(n0, n1)` wide, the tail window is
  `max(1, ceiling(0.10 · min(n0, n1)))` order statistics, and the combinatorial
  floor on a *k*-sample subset is `C(n1, k) / C(n1 + n0, k)`. A specification
  that omits them cannot explain why per-gene FDR failed on this cohort or why a
  one-point tail window is not usable.
- **Gene counts** — 3,041 on-target genes in v7; ~2,200–2,800 in v8, depending
  on cohort. Sets the multiple-testing burden the p-values were adjusted
  against.
- **Enrichment counts** — genes clearing nominal *p* < 0.01 against the number
  expected by chance. This is the evidence that the signal is real given that no
  gene survives BH; dropping it would leave the nomination-by-rank decision
  looking arbitrary.
- **The measured defect** — 120 of 2,219 genes with `|tail.conc| > 2`, largest
  149. Quantified in `WADE_REPO_SCOPE.md`; it is the size of the bug the port
  is meant to fix.

Every one of these is an **aggregate statistic over a group**, not a record
about a person. None of them is a count small enough to identify anyone, none
is tied to an identifier, and none can be inverted to recover a patient's data.
The smallest number staged is a group size of five, stated without any
accompanying attribute, in a sentence whose point is that the group is too small
to nominate from.

What was **not** kept, and would have been a different decision: any per-library
row, any per-gene statistic computed from the real cohort, any figure rendered
from cohort data. Those are the `cache/`, `disc_tables/` and `wade_check2/`
entries in [Left behind](#left-behind), and they are the reason that list is
long.

### Clinical vocabulary: a judgement for the user

Two staged files encode clinical **vocabulary** — not patient data, but the
cohort's design expressed as code. This is flagged rather than decided.

`reference/R/downstream/control_strata.R` defines the control taxonomy: four
strata (`hr_surveil`, `screen_pos`, `precursor_panc`, `benign_other`) mapped to
disease labels including `pancreas_high_risk`, `prostate_high_psa`,
`pancreas_ipmn`, `pancreas_mcn`, `pancreatic_cyst`, `adrenal_adenoma`,
`thyroid_nodule`, `intramuscular_myxoma`, `hepatitis`,
`liver_angiomyolipoma_benign`. Its header prose explains the clinical logic:
that a surveillance cohort is cancer-free but risk-enriched, that
screened-positive patients may have undiagnosed cancer, that same-organ
precursors are real lesions and not normal tissue.

`reference/R/downstream/wade_contrasts.R` names the contrasts built from those
strata: `cancer_vs_pooled`, `cancer_vs_nopsa`, `cancer_vs_hr`, `panc_vs_hr`,
`nonpanc_vs_hr`, `precursor_vs_hr`, `screenpos_vs_hr`, `benign_vs_hr`,
`precursor_vs_benign`, with a prose rationale for each.

**My assessment.** This is study design, not patient data, and I judge it
acceptable for public release — with the reasoning stated so it can be
overruled:

- Every stratum name is a **disease category**, not a patient. Nothing maps a
  category to a library, a specimen or a person. `control_strata.R` contains no
  token shaped like an identifier at all.
- The categories are the kind of thing a methods section publishes. "We
  compared pancreatic cancer plasma against high-risk surveillance plasma,
  excluding screen-positive prostate patients from the reference" is a sentence
  that belongs in a paper.
- The vocabulary is **load-bearing for the port**. `WADE_REPO_SCOPE.md`
  identifies the contrast screen as the one component with no analogue in a
  generic package, and the screen is inseparable from the strata it screens.
  Staging the screen without its vocabulary would stage a function whose purpose
  is unreadable.

**What the user should weigh, and what I cannot judge from inside the
repository:**

1. Together, these two files disclose the **shape of an unpublished cohort** —
   which diseases were enrolled, which were used as controls, and that a
   prostate arm exists alongside a pancreatic one. Group sizes appear elsewhere
   in the staged documents. If the cohort's composition is not yet public, this
   publishes it ahead of the paper.
2. `control_strata.R` documents a **bug and its clinical consequence** (v7's
   `grepl("control", disease)` silently returned zero controls on the
   2026-08-02 store, and every v7 contrast was therefore run against a
   misassigned reference). That is a candid and useful engineering record. It is
   also a public statement about a prior analysis of this cohort.
3. Whether an IRB protocol or data-use agreement treats enrolment categories as
   disclosable is not something this repository can answer.

If any of the three is a problem, the remedy is narrow: `control_strata.R` and
the registry portion of `wade_contrasts.R` can be withheld or reduced to
abstract stratum names (`ref_a`, `ref_b`) without touching `wade.R`, the
supplement, the algorithm specification, or the extracted method-validation
material. The screen's *logic* — design confounding, depth confounding,
quantile-grid floor — survives that substitution intact. Nothing else staged
depends on the clinical labels.

## Discrepancies found during staging

Recorded so a future reader is not surprised by them.

1. **"Nine functions" is a count of nine, but `wade.R` defines ten.**
   `WADE_REPO_SCOPE.md` opens by describing `wade.R` as "16 kB, 9 exported
   functions". The file defines **ten** top-level functions: eight public
   (`wade_lib_size`, `wade_normalize`, `wade_stats`, `wade_perm_pvalues`,
   `wade`, `wade_score`, `wade_run`, `wade_gene`) and two dot-private
   (`.wade_null_stats`, `.gpd_tail_p`). The scope document's own tables account
   for nine of them — seven under "What moves: the method" plus
   `wade_lib_size`/`wade_normalize` under "What moves but should be
   reconsidered". The tenth, **`wade_run()`**, appears in neither table; it is
   mentioned once in passing, as the signature the proposed interface is "close
   to". It is the convenience driver the v7 notebook actually calls
   (`wade_run(Ccnt, Cnorm, lib_cancer, lib_noncancer, gene_names = gn)`), so a
   port that works through the scope document's tables alone will omit the
   entry point the historical usage depends on. Also note "exported" is
   aspirational: `wade.R` is `source()`d, not a package, so nothing is exported
   in the R sense.
2. **The `.qmd` fragments are quarto-flavoured but the notebook renders through
   rmarkdown.** `render_v8.R` carries a gate (its "gate 9") that fails the
   build if quarto cross-reference syntax reaches the output, added because a
   WADE fragment shipped `@sec-`/`@fig-` references that this pipeline does not
   resolve. This is why the verbatim fragments and the extracted notebook
   ranges differ on cross-reference and callout syntax. Neither is wrong; they
   target different renderers.
3. **The cache holds several entries per contrast.** 76 cache files across 22
   contrast/cohort combinations: 18 combinations have four files each, and four
   combinations — all of them on a `v3v4v5v6` cohort — have one each. The cache
   key includes the store fingerprint and gene count, so the repeats are most
   likely successive store refreshes rather than distinct analyses. Not
   investigated further; nothing here depends on it.

## Files written for this repository

Everything else under `reference/` is cfRNA material. These four are not:

| File | What it is |
|---|---|
| `reference/PROVENANCE.md` | This file. |
| `reference/docs/README.md` | Why the source documents are unresolved, and the rule that they adjudicate disagreements with the clean specs in top-level `docs/`. |
| `reference/R/README.md` | Written by the R track: how to run the staged R in the pinned renv sandbox. |
| `reference/R/CHECKSUMS.txt` | Written by the R track: checksums for the staged R files. |
| `reference/R/downstream/README.md` | Written by the R track: why the three downstream files are evidence rather than porting targets. |
| `reference/R/validation_sims_v7.R` | Written by the R track: the v7 §9.6 simulations as a standalone script, with the cohort geometry parameterised. Verbatim-in-substance, not byte-identical. |
| `reference/R/validation_sims_v7.png`, `_calibration.csv`, `_power.csv` | Its outputs. Simulation only — no patient data reaches them. |

Two helper scripts used to produce the extractions are in `tools/`
(`qmd_outline.py`, a fence-aware Quarto heading lister, and
`extract_notebook_section.py`, the line-range-to-markdown extractor). They are
staging tooling, not part of the reference material, and nothing in the
repository depends on them.

`reference/R/.Rprofile`, `reference/R/renv.lock`, `reference/R/renv/` and
`reference/R/downstream/` were staged by the R track; their contents and
provenance are that track's to describe, and the four R source files it copied
are inventoried above with checksums verified independently here.
