# reference/docs — WADE source documents, as written in the cfRNA repo

These files are **raw source material**, copied out of the cfRNA analysis
repository without modification. They are not documentation of the WADE package
and they are not a specification. They are the record of what the method was,
how it was described, and how it was used, at the moment the extraction began.

Everything here came from `mkiyer/cfrna` at git SHA
`828f2f1c6a10afd182fee0e119ba3595d98b6a72` (clean working tree, 2026-08-17).
Per-file paths, checksums and disposition are in
[`reference/PROVENANCE.md`](../PROVENANCE.md).

## What is here

| File | What it is |
|---|---|
| `WADE_REPO_SCOPE.md` | The extraction analysis: which functions are method and which are cfRNA plumbing, where the normalization boundary should fall, and the five cross-language parity hazards. Written before this repo existed. |
| `supplement_wade.qmd` | The method supplement (S7): why not a t-test, the statistic, why permutation with a tail refinement, the quantile-grid sample-size constraint, what WADE does not do. |
| `section4_wade.qmd` | The v8 discovery-section fragment: why one cohort yields no single control group, and the pre-run contrast screen. |
| `section4_wade_setup.qmd` | The computation block behind both fragments — every quantity the prose and the supplement quote, computed once. |
| `notebook_sections/` | WADE prose and chunk code extracted from the v7 and v8 notebooks, as markdown. Each file states its source, line range and checksum in its own header. |

## These files are unresolved on purpose

The `.qmd` files are Quarto sources. They carry:

- **Unresolved inline expressions** of the form `` `r some_r_expression` ``.
  In the rendered notebook these evaluate to numbers. Here they do not
  evaluate to anything — the text reads `` `r wade_n_runs` `` where the reader
  of the rendered document saw an integer. No value has been substituted,
  interpolated or guessed anywhere in this directory.
- **cfRNA-specific quantities** — group sizes, gene counts, contrast names,
  cohort labels. These describe one cohort at one point in time. They are
  evidence of how the method behaved on real data, not properties of the
  method.
- **Quarto-only syntax** — `::: {.callout-note}` divs, `@sec-` and `@fig-`
  cross-references, `#|` chunk options, `{#sec-wade-method}` anchors. Several
  cross-references point at sections of the cfRNA notebook that do not exist
  in this repository.

All three are correct for a raw source and none of them should be repaired
here. Editing these files would destroy the only thing they are for: being
exactly what the cfRNA repo said.

**Read them as text, not as documents to render.** No attempt should be made
to knit, render or execute anything in this directory. The R code in the
chunks references cohort objects (`cds`, `mats`, `cohort`, `ot`) and a Zarr
store that are not part of this repository, and the data they read is
patient-derived and deliberately absent (see `PROVENANCE.md`).

## Relationship to `docs/`

The clean, language-agnostic specifications live in the repository's top-level
`docs/` directory. They are **derived from these files**: the algorithm
specification, the rationale, the stated limits and the porting hazards are all
restatements of material that first appeared here, with the cfRNA cohort
specifics removed and the unresolved expressions either resolved from a cited
source or dropped.

That derivation direction matters for one practical reason:

> **Where a document in `docs/` and a document in `reference/docs/` disagree,
> the file in `reference/docs/` is correct and the file in `docs/` has a bug.**

These are the originals. A clean spec is a transcription, and transcriptions
drift. If a future reader finds the specification asserting something the
supplement does not support, the specification is what changes.

The one exception is a defect the source documents themselves identify as a
defect — `WADE_REPO_SCOPE.md` §"Cross-language parity" item 4 describes the
`tail.conc` guard as a bug to fix in the port rather than a behaviour to
reproduce. Where a source document says "this is wrong, fix it," the spec is
entitled to specify the fix. It is not entitled to silently disagree.
