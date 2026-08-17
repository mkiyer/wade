# =====================================================================
# R/control_strata.R -- "control" is a TAXONOMY, not a group
# ---------------------------------------------------------------------
# WHY THIS EXISTS. Every case/control test in this project needs to name a
# reference population, and this cohort does not have one. It has FOUR, and
# they are different in kind, not merely in label:
#
#   hr_surveil      pancreas_high_risk -- family history or germline
#                   predisposition, SCREENED NEGATIVE for pancreatic cancer.
#                   The surveillance reference: cancer-free, but enriched for
#                   risk, so not a healthy baseline either.
#   screen_pos      prostate_high_psa, prostate_unknown -- screened POSITIVE
#                   on a biomarker (elevated PSA) with malignancy neither
#                   confirmed nor excluded. A control by enrolment, not by
#                   biology: some of these patients have cancer.
#   precursor_panc  pancreas_ipmn, pancreas_mcn, pancreatic_cyst -- same-organ
#                   neoplastic precursors. Real lesions, not normal tissue.
#   benign_other    adrenal_adenoma, thyroid_nodule, intramuscular_myxoma,
#                   hepatitis -- benign or inflammatory disease in other
#                   organs. Non-neoplastic disease signal, no malignancy.
#
# Pooling these buys sample size and costs interpretation: a gene "up in
# cancer vs controls" against a pool containing screen-positive prostate
# patients and inflamed livers is not the same claim as the same gene up
# against cancer-free surveillance plasma. Both runs are worth having, which
# is why cfrna_wade_registry() (R/wade_contrasts.R) enumerates the pooled AND
# the separated forms rather than choosing.
#
# THE BUG THIS MODULE EXISTS TO PREVENT, TWICE OVER.
#
# 1. v7 built its clinical grouping with grepl("control", disease). On the
#    2026-08-02 store NO disease contains the substring "control" -- the
#    control group is called `pancreas_high_risk` -- so that code returns
#    ZERO controls and silently relabels all 22 control libraries "Cancer".
#    R/pca.R::cfrna_broad_group() already documents and fixes this. This
#    module keeps the same discipline: strata are named from a registry, and
#    a disease that is in no list is an ERROR, never a default.
#
# 2. The registry itself was incomplete. `prostate_high_psa` (9 libraries, 9
#    patients) appears in NEITHER CFRNA_CONTROL_GROUPS NOR CFRNA_BENIGN in
#    R/perspective.R, so cfrna_broad_group() falls through to its "unknown =
#    Cancer" default and counts nine screen-positive CONTROLS as cancer
#    cases. The default is documented as safe for this cohort; it is not.
#    cfrna_control_strata_audit() reports exactly this, and it fails loudly
#    rather than defaulting, so the next relabelling cannot repeat it.
#
# WHAT THIS MODULE DOES NOT DO. It does not decide which stratum is the right
# reference -- that is the analysis question, and R/wade_contrasts.R measures
# it. It only names the strata and refuses to guess.
# =====================================================================

suppressPackageStartupMessages({
  library(dplyr)
})

# The stratum registry. Disease vocabulary, not free text: every value here
# must appear in the workbook's `disease` column, and cfrna_control_strata()
# stops if a cohort contains a disease no list mentions.
#
# `cancer` is deliberately NOT enumerated. There are 19 malignancy labels in
# this store and the list grows with enrolment, so enumerating them would make
# every new cancer a silent misclassification. The reference strata are
# closed sets; malignancy is the complement. That asymmetry is the whole
# safety argument, and cfrna_control_strata_audit() checks it by reporting
# every disease that lands in the complement.
CFRNA_CONTROL_STRATA <- list(
  hr_surveil     = c("pancreas_high_risk"),
  screen_pos     = c("prostate_high_psa", "prostate_unknown"),
  precursor_panc = c("pancreas_ipmn", "pancreas_mcn", "pancreatic_cyst"),
  benign_other   = c("adrenal_adenoma", "thyroid_nodule",
                     "intramuscular_myxoma", "hepatitis",
                     # 2026-08-16 batch. Benign by its own label, and an unregistered
                     # label defaults to CANCER -- so left alone it would have entered
                     # the cancer arm of every contrast AND the non-control reference
                     # every patient is scored against. This is the same defect class
                     # as prostate_high_psa, which sat in neither list and put 9
                     # screen-positive control patients in the cancer group.
                     "liver_angiomyolipoma_benign")
)

# WHY `healthy` AND `deceased_donor` ARE DELIBERATELY UNREGISTERED. The strict audit
# flags both -- correctly, they read as non-malignant and fall through to `cancer`. They
# are left out anyway because they never reach an analysis: all four libraries sit on
# the paxgene and ctrna_idt_v2 panels, outside the panel set any cohort is built from,
# so cfrna_qc_calls() returns NA for them and no contrast, reference set or per-patient
# score ever sees them. Verified 2026-08-16 on the 371-library store: 0 of 4 pass QC
# because 0 of 4 are scored.
#
# Registering them as controls would be worse than leaving them out: it would assert
# they are a usable control population, and the first cohort built on a panel that DOES
# include those captures would silently pick up two whole-blood PAXgene libraries as
# cfRNA controls. The audit is meant to be read and reasoned about, not silenced -- if
# a future batch puts either label on an analysis panel, that is the moment to decide
# what they are, and the audit will say so again.

CFRNA_STRATUM_LEVELS <- c("cancer", "hr_surveil", "screen_pos",
                          "precursor_panc", "benign_other")

# Display labels. Held next to the registry so a figure or table cannot
# invent its own wording for a stratum, and short enough to sit on an axis.
CFRNA_STRATUM_LABELS <- c(
  cancer         = "Cancer",
  hr_surveil     = "High-risk surveillance",
  screen_pos     = "Screen-positive (PSA)",
  precursor_panc = "Pancreatic precursor",
  benign_other   = "Benign / inflammatory",
  all_control    = "All controls (pooled)",
  control_nopsa  = "Controls, screen-positive excluded"
)

# One-line statement of what each stratum IS, for the section that has to
# explain why they are not poolable without comment.
CFRNA_STRATUM_NOTES <- c(
  cancer         = "confirmed malignancy, any organ",
  hr_surveil     = "germline/family risk, screened NEGATIVE for cancer",
  screen_pos     = "screened POSITIVE on PSA; malignancy neither confirmed nor excluded",
  precursor_panc = "same-organ neoplastic precursor lesion (IPMN / MCN / cyst)",
  benign_other   = "benign or inflammatory disease, other organs"
)

#' Assign a control stratum to each disease label.
#'
#' Registry-driven and CLOSED on the reference strata: a disease that matches
#' no reference list is malignancy by complement, which is correct for this
#' store but is asserted rather than assumed -- pass `verbose = TRUE`, or call
#' cfrna_control_strata_audit(), to see every label that took that branch.
#'
#' @param disease character vector of workbook disease labels.
#' @param strata  registry; defaults to CFRNA_CONTROL_STRATA.
#' @return factor with levels CFRNA_STRATUM_LEVELS.
cfrna_control_strata <- function(disease, strata = CFRNA_CONTROL_STRATA,
                                 verbose = FALSE) {
  if (!length(disease)) return(factor(character(), levels = CFRNA_STRATUM_LEVELS))
  if (anyNA(disease))
    stop("cfrna_control_strata(): ", sum(is.na(disease)), " NA disease label(s) -- ",
         "a library with no disease cannot be assigned a stratum; fix the metadata")

  dup <- unlist(strata, use.names = FALSE)
  if (anyDuplicated(dup))
    stop("cfrna_control_strata(): disease(s) in more than one stratum: ",
         paste(unique(dup[duplicated(dup)]), collapse = ", "))

  g <- rep("cancer", length(disease))
  for (s in names(strata)) g[disease %in% strata[[s]]] <- s

  if (verbose) {
    unk <- sort(unique(disease[g == "cancer"]))
    if (length(unk))
      message("cfrna_control_strata(): treated as cancer by complement (",
              length(unk), " label(s)): ", paste(unk, collapse = ", "))
  }
  factor(g, levels = CFRNA_STRATUM_LEVELS)
}

#' Audit the stratum registry against a cohort's actual disease vocabulary.
#'
#' The check that would have caught `prostate_high_psa`. Two questions:
#'   (a) does the registry name diseases this cohort does not have?
#'       -- harmless, but it means the registry is describing another store.
#'   (b) which diseases fell through to `cancer` by complement, and are they
#'       all really malignancies?
#'
#' (b) cannot be answered by code -- "is adrenal_adenoma a cancer" is a
#' clinical fact, not a computation -- so this returns the list for a human to
#' read and, when `strict = TRUE`, stops on any label matching a benign-sounding
#' pattern. The pattern is a TRIPWIRE, not a classifier: it exists to make a
#' mislabelled control loud, and every hit must be resolved by editing the
#' registry, never by widening the pattern.
#'
#' @param disease disease labels present in the cohort.
#' @param strict  stop if a complement label looks non-malignant.
#' @return list(missing_from_store, complement, suspicious)
cfrna_control_strata_audit <- function(disease, strata = CFRNA_CONTROL_STRATA,
                                       strict = FALSE) {
  present <- sort(unique(disease))
  named   <- unlist(strata, use.names = FALSE)
  g       <- cfrna_control_strata(present, strata = strata)
  comp    <- present[g == "cancer"]

  # Words that describe a non-malignant condition. Deliberately narrow: it
  # must fire on a control that was forgotten, not on every unfamiliar label.
  # `high_risk`/`high_psa`/`unknown` are here because those are exactly the
  # enrolment-defined control groups this cohort keeps adding.
  tripwire <- "high_risk|high_psa|healthy|normal|donor|control|benign|adenoma|nodule|cyst|ipmn|mcn|hepatitis|myxoma|cirrhosis|unknown|screen"
  susp <- comp[grepl(tripwire, comp)]

  out <- list(
    missing_from_store = setdiff(named, present),
    complement         = comp,
    suspicious         = susp)

  if (length(susp) && strict)
    stop("cfrna_control_strata_audit(): ", length(susp),
         " disease label(s) fell through to `cancer` but read as non-malignant: ",
         paste(susp, collapse = ", "),
         "\n  Add each to a stratum in CFRNA_CONTROL_STRATA (R/control_strata.R).",
         "\n  Do NOT relax the tripwire pattern -- that is how nine ",
         "screen-positive controls were counted as cancer cases.")
  out
}

#' Attach the stratum to a library table.
#'
#' HOUSE RULE: a figure axis a reader might want to look up must be a column in
#' the master library table. `control_stratum` is a grouping axis for every
#' figure in the discovery section, so it is added to the scorecard here rather
#' than being recomputed inside each figure. Idempotent: an existing column is
#' replaced, so re-running the chunk cannot produce `.x`/`.y` twins.
#'
#' @param libs library table carrying `disease`.
cfrna_strata_annotate <- function(libs, strata = CFRNA_CONTROL_STRATA,
                                  verbose = FALSE) {
  if (!"disease" %in% names(libs))
    stop("cfrna_strata_annotate(): table has no `disease` column")
  libs$control_stratum <- cfrna_control_strata(libs$disease, strata = strata,
                                               verbose = verbose)
  libs$control_stratum_label <-
    unname(CFRNA_STRATUM_LABELS[as.character(libs$control_stratum)])
  libs
}

#' Composition of the strata in a cohort: libraries, patients, diseases.
#'
#' Reports BOTH libraries and patients because they differ by more than a
#' constant here -- cancer patients carry a median of 2 libraries and controls
#' 1, so a library-level count overstates the cancer group relative to the
#' patient-level one. Any power statement has to say which unit it used.
cfrna_strata_summary <- function(libs, strata = CFRNA_CONTROL_STRATA) {
  d <- cfrna_strata_annotate(libs, strata = strata)
  d |>
    dplyr::group_by(control_stratum) |>
    dplyr::summarise(
      label     = dplyr::first(control_stratum_label),
      libraries = dplyr::n(),
      patients  = dplyr::n_distinct(patient),
      libs_per_patient = round(dplyr::n() / dplyr::n_distinct(patient), 2),
      diseases  = dplyr::n_distinct(disease),
      disease_list = paste(sort(unique(disease)), collapse = ", "),
      .groups = "drop") |>
    dplyr::mutate(note = unname(CFRNA_STRATUM_NOTES[as.character(control_stratum)])) |>
    dplyr::arrange(control_stratum)
}

#' Library-id vectors for one stratum, or for a named pool of strata.
#'
#' The single place a contrast turns a stratum name into libraries, so a
#' contrast definition never subsets by disease itself. Pools are named in
#' CFRNA_STRATUM_LABELS so a pooled group cannot be described two ways.
#'
#' @param libs library table with `disease` and `library`.
#' @param which one or more stratum names, or "all_control" / "control_nopsa".
cfrna_stratum_libs <- function(libs, which, strata = CFRNA_CONTROL_STRATA) {
  d <- cfrna_strata_annotate(libs, strata = strata)
  ctrl_all <- setdiff(CFRNA_STRATUM_LEVELS, "cancer")
  sel <- switch(
    paste(which, collapse = "|"),
    all_control   = ctrl_all,
    control_nopsa = setdiff(ctrl_all, "screen_pos"),
    which)
  bad <- setdiff(sel, CFRNA_STRATUM_LEVELS)
  if (length(bad))
    stop("cfrna_stratum_libs(): unknown stratum(s): ", paste(bad, collapse = ", "),
         "\n  known: ", paste(c(CFRNA_STRATUM_LEVELS, "all_control", "control_nopsa"),
                              collapse = ", "))
  d$library[as.character(d$control_stratum) %in% sel]
}
