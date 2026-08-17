# =====================================================================
# R/wade_contrasts.R -- which contrasts this cohort can support, and why
# ---------------------------------------------------------------------
# WHAT THIS IS. The cfRNA-specific layer around wade.R: it names the
# contrasts, screens each one for confounding BEFORE it is run, runs the
# survivors, and caches them. wade.R itself knows nothing about diseases,
# panels or probe designs -- keeping it that way is what makes it extractable
# (see WADE_REPO_SCOPE.md).
#
# WHY A REGISTRY AND NOT THREE FUNCTION CALLS. v7 ran three contrasts written
# inline as three `wade_run()` calls on three hand-built library vectors. That
# was reproducible but not auditable: nothing recorded WHY those three, what
# else was possible, or what was rejected. On this cohort the rejected ones are
# the interesting part -- one candidate contrast is confounded with probe design
# strongly enough to be uninterpretable -- so the enumeration, the screen, and
# the verdict are data structures the notebook prints, not prose the notebook
# asserts.
#
# THE SCREEN, AND WHAT EACH TEST IS FOR.
#
#   panel   Cramer's V and a Fisher exact p on group x DESIGN VERSION.
#           Design version, not capture label: v6f shares v6's gene targets
#           (cohort_def.R), so splitting them invents a batch the design does
#           not have. The distinction is load-bearing. `prostate_high_psa` is
#           100% v6f, so ANY contrast putting it against a panel-mixed group
#           reaches capture-label V = 1.00 -- capture label predicts the group
#           perfectly -- while the design-version V for the same libraries is
#           far lower (0.45 for cancer vs screen-positive at index level, 0.69
#           for screen-positive vs high-risk surveillance). The first number is
#           an artefact of labelling v6f separately from v6; the second is the
#           real exposure. Screening on capture label would refuse contrasts
#           that are merely v6f-heavy, and 0.69 is high enough to refuse
#           screenpos_vs_hr on the honest metric alone.
#   depth   Median on-target spliced yield ratio and a Wilcoxon p. WADE's
#           statistic is a quantile area on normalized abundance, so a
#           systematic depth difference shifts every quantile of the deeper
#           group and manufactures signed area.
#   size    min(n_case, n_ctrl) IS the quantile-grid resolution -- WADE
#           compares the groups at min(n0,n1) probabilities and the tail
#           window is ceiling(0.10 * that). A contrast with 6 in the smaller
#           group has a ONE-POINT tail: its "subset axis" is a single order
#           statistic, and calling that a subset detector is a category error.
#
# The screen classifies, it does not gate silently. Four verdicts, and the
# distinction between the middle two is the point:
#   run         both groups reach the size floor, no screen fires
#   caution     runnable and nominatable, but a screen fired -- name the caveat
#   descriptive TOO SMALL TO NOMINATE FROM, but the comparison is still
#               interpretable: run it, report effect sizes and the top genes,
#               and exclude it from the nomination product. This tier exists
#               because "underpowered" and "uninterpretable" are different
#               problems with different remedies -- a small contrast is fixed
#               by enrolment, and refusing to run it hides what the current
#               data does say. The control-vs-control comparisons live here.
#   refuse      the design confound makes the result uninterpretable at ANY
#               sample size, so running it would produce a gene list whose
#               most likely explanation is the probe panel.
# Every verdict is recorded WITH ITS NUMBERS and printed. A contrast that fails
# silently is a contrast that gets re-proposed next quarter.
#
# For a panel-refused contrast the screen also reports whether restricting to a
# single design version would rescue it (cfrna_wade_panel_matched()), because
# "not testable" and "not testable yet, and here is the missing sample" are
# different messages to a lab that is still enrolling.
#
# DEPENDENCE ON THE INDEX-LIBRARY RULE, MEASURED. The index unit comes from
# cfrna_index_libraries() (R/perspective.R), whose rule is "earliest timepoint,
# then best on-target spliced yield". A separate strand of this project uses a
# STRICTER rule for the deduplicated pancreatic analysis -- t1/s1 only, patients
# lacking either excluded -- and the two disagree for the 7 t0, 2 surgery and 7
# missing-timepoint libraries here. Since every verdict is decided at the index
# level, that disagreement could in principle change what runs.
#
# It does not. Re-screening all 10 candidates under the strict t1/s1-only rule
# leaves every nprobs and every verdict identical (checked on the all-panels
# cohort; Cramer's V moves by at most 0.015). The reason is structural rather
# than lucky: the libraries the two rules disagree about belong to patients who
# have only those libraries, so both rules select the same row -- the choice
# only matters when a patient has BOTH a t1/s1 and an earlier or unlabelled
# timepoint. The screen is therefore reported as-is, and this note exists so
# the next reader does not have to re-derive that.
#
# WHAT THE SCREEN CANNOT DO. It detects ASSOCIATION between group and a
# nuisance variable; it cannot remove one. There is no panel term in WADE --
# the statistic is a two-group quantile area, and adding a covariate to it is
# a method change, not a call-site change. So `caution` means "the reader must
# know", not "handled".
# =====================================================================

# LOAD ORDER (per R/README.md's contract that each module assumes the ones
# above it): this module needs R/control_strata.R for the stratum registry,
# R/group_structure.R for cfrna_gs_design(), R/perspective.R for
# cfrna_index_libraries(), and wade.R for the statistic itself. Checked rather
# than assumed -- sourcing this file alone and getting "object not found" three
# calls deep is a worse error message than this one.
suppressPackageStartupMessages({
  library(dplyr)
})

local({
  need <- c(cfrna_strata_annotate = "R/control_strata.R",
            cfrna_gs_design       = "R/group_structure.R",
            cfrna_index_libraries = "R/perspective.R",
            wade_run              = "wade.R")
  miss <- need[!vapply(names(need), exists, logical(1), mode = "function")]
  if (length(miss))
    stop("R/wade_contrasts.R requires: ",
         paste(sprintf("%s() from %s", names(miss), miss), collapse = ", "),
         " -- source them first")
})

# Smaller group must reach this, or the quantile grid is too coarse to carry a
# tail window. 10 gives ceiling(0.10*10) = 1 -- still a single point, which is
# why 10 is the FLOOR for running at all and CFRNA_WADE_TAIL_MIN_N below is the
# threshold for believing the subset axis.
CFRNA_WADE_MIN_N <- 10L

# Below this in the smaller group the tail window is 1-2 order statistics, so
# tail.mean is reported as descriptive only and nomination falls back to the
# bulk axis. 20 gives a 2-point window; 30 gives 3.
CFRNA_WADE_TAIL_MIN_N <- 20L

# Cramer's V on group x design version above this and panel largely predicts
# group; the contrast is refused. 0.5 is a conventional "strong association"
# mark and is stated here rather than inline so the threshold is one edit.
CFRNA_WADE_PANEL_V_MAX <- 0.50

# v7's settings, preserved: 2000 label permutations, top-10% tail window.
CFRNA_WADE_NPERMS <- 2000L
CFRNA_WADE_TAIL_Q <- 0.10

#' Cramer's V for a two-way table, NA when a margin is degenerate.
cfrna_wade_cramer_v <- function(tb) {
  tb <- tb[rowSums(tb) > 0, colSums(tb) > 0, drop = FALSE]
  if (any(dim(tb) < 2)) return(NA_real_)
  chi <- suppressWarnings(stats::chisq.test(tb)$statistic)
  as.numeric(sqrt(chi / (sum(tb) * (min(dim(tb)) - 1))))
}

#' The candidate contrast registry.
#'
#' Each entry names its case and control groups as STRATA (or "all_control" /
#' "control_nopsa" pools, or a disease vector for the organ-specific ones), so
#' a contrast never subsets by disease string itself -- R/control_strata.R owns
#' that mapping. `rationale` is what the contrast is FOR, and it is printed in
#' the section table: a contrast that cannot be explained in one line does not
#' belong in the registry.
#'
#' The organ-focused entries (`case_disease`) are deliberately literal disease
#' vectors rather than perspectives, because a perspective's control group is
#' fixed at `pancreas_high_risk` while the point here is to VARY the reference.
CFRNA_WADE_CONTRASTS <- list(
  list(id = "cancer_vs_pooled", label = "Cancer vs all controls (pooled)",
       case = "cancer", ctrl = "all_control",
       family = "detection",
       rationale = paste("Maximum-power detection contrast: every non-cancer",
                         "library as one reference. Pooling buys n and costs",
                         "interpretation -- the reference contains screened-positive",
                         "and precursor-lesion plasma.")),
  list(id = "cancer_vs_nopsa", label = "Cancer vs controls, screen-positive excluded",
       case = "cancer", ctrl = "control_nopsa",
       family = "detection",
       rationale = paste("The same detection contrast with the screen-positive",
                         "prostate group removed, since those patients may",
                         "harbour undiagnosed cancer. Isolates how much the",
                         "pooled result depends on including them.")),
  list(id = "cancer_vs_hr", label = "Cancer vs high-risk surveillance",
       case = "cancer", ctrl = "hr_surveil",
       family = "detection",
       rationale = paste("The cleanest reference this cohort has: cancer-free",
                         "surveillance plasma, screened negative. Smallest",
                         "reference group but the only one with a confirmed",
                         "cancer-free status.")),
  list(id = "panc_vs_hr", label = "Pancreatic cancer vs high-risk surveillance",
       case_disease = "pancreatic_cancer", ctrl = "hr_surveil",
       family = "organ",
       rationale = paste("The organ-matched contrast: pancreatic cancer against",
                         "the surveillance population enrolled for pancreatic",
                         "risk. Case and control share the target organ and the",
                         "referral pathway.")),
  list(id = "nonpanc_vs_hr", label = "Non-pancreatic cancer vs high-risk surveillance",
       case_disease = "__cancer_except_pancreatic__", ctrl = "hr_surveil",
       family = "organ",
       rationale = paste("The complement of panc_vs_hr. Pancreatic cancer is 59",
                         "of 134 cancer libraries, so a pooled nomination could",
                         "be a pancreatic list wearing a pan-cancer label; genes",
                         "nominated here as well are not pancreas-specific.")),
  list(id = "precursor_vs_hr", label = "Pancreatic precursor vs high-risk surveillance",
       case = "precursor_panc", ctrl = "hr_surveil",
       family = "continuum",
       rationale = paste("Does a same-organ neoplastic precursor (IPMN / MCN /",
                         "cyst) already carry plasma signal? Compared against",
                         "panc_vs_hr this separates early markers from",
                         "malignancy-specific ones.")),
  list(id = "screenpos_vs_hr", label = "Screen-positive (PSA) vs high-risk surveillance",
       case = "screen_pos", ctrl = "hr_surveil",
       family = "control_vs_control",
       rationale = paste("CONTROL vs CONTROL. Two enrolment-defined reference",
                         "populations that are both nominally cancer-free.",
                         "Signal here is either undiagnosed malignancy or a",
                         "technical difference between the cohorts -- the screen",
                         "says which is testable.")),
  list(id = "benign_vs_hr", label = "Benign / inflammatory vs high-risk surveillance",
       case = "benign_other", ctrl = "hr_surveil",
       family = "control_vs_control",
       rationale = paste("CONTROL vs CONTROL. Non-neoplastic disease against",
                         "cancer-free surveillance: what does benign organ",
                         "disease alone put in plasma? Genes shared with the",
                         "cancer contrasts are disease-response, not cancer",
                         "markers.")),
  list(id = "precursor_vs_benign", label = "Pancreatic precursor vs benign / inflammatory",
       case = "precursor_panc", ctrl = "benign_other",
       family = "control_vs_control",
       rationale = paste("CONTROL vs CONTROL. Neoplastic precursor against",
                         "non-neoplastic disease -- is there a specifically",
                         "neoplastic component to the precursor signal?")),
  # PANEL-MATCHED BY CONSTRUCTION. Every screen-positive library is v6-design,
  # so restricting the cancer group to v6 removes the design confound instead
  # of caveating it -- the one contrast involving screen_pos that the panel
  # screen cannot object to. Written as an explicit design restriction because
  # the restriction IS the design of the comparison, not a filter applied to it.
  # FAMILY IS `detection`, NOT `control_vs_control`. Its case group is cancer, so
  # mechanically it is a detection contrast against an unusual reference -- what
  # it teaches about the screen-positive group is read from how its nominations
  # compare to the other detection contrasts, not from the contrast being
  # control-vs-control. Labelling it control_vs_control would colour it with the
  # comparisons that have no cancer in them and invite the wrong reading.
  list(id = "cancer_vs_screenpos_v6", label = "Cancer vs screen-positive (PSA), v6 design only",
       case = "cancer", ctrl = "screen_pos", design = "v6",
       family = "detection",
       rationale = paste("Do screen-positive PSA patients carry cancer-like",
                         "plasma signal? Restricted to v6-design libraries so",
                         "case and control are panel-matched by construction",
                         "rather than by adjustment -- all screen-positive",
                         "libraries are v6-design, which is exactly what makes",
                         "the unrestricted version of this contrast refusable."))
)

#' Resolve one registry entry to case / control library ids.
#'
#' A `design` field on the spec restricts BOTH groups to those design versions
#' before resolving, which is how a panel-matched contrast is built. The
#' restriction is applied here, once, so no caller can apply it to one group and
#' forget the other.
cfrna_wade_resolve <- function(spec, libs) {
  d <- cfrna_strata_annotate(libs)
  if (!is.null(spec$design))
    d <- d[cfrna_gs_design(d$capture) %in% spec$design, , drop = FALSE]
  case <- if (!is.null(spec$case_disease)) {
    if (identical(spec$case_disease, "__cancer_except_pancreatic__"))
      d$library[d$control_stratum == "cancer" & d$disease != "pancreatic_cancer"]
    else d$library[d$disease %in% spec$case_disease]
  } else cfrna_stratum_libs(d, spec$case)
  ctrl <- if (!is.null(spec$ctrl_disease)) d$library[d$disease %in% spec$ctrl_disease]
          else cfrna_stratum_libs(d, spec$ctrl)
  ov <- intersect(case, ctrl)
  if (length(ov))
    stop("cfrna_wade_resolve(): contrast '", spec$id, "' puts ", length(ov),
         " librar(ies) in BOTH groups")
  list(case = case, ctrl = ctrl)
}

#' Screen every registry contrast for feasibility and confounding.
#'
#' Runs at BOTH units. The library level is what WADE will actually be handed
#' (it has no random effect, so it cannot use repeated libraries properly); the
#' index level -- one library per patient -- is the honest test of whether the
#' association is driven by independent patients or by replicate structure. A
#' contrast whose panel association is significant at library level but not at
#' index level is one where correlated replicates are doing the work, and the
#' section says so.
#'
#' @param libs      library table (cohort_def$libraries).
#' @param contrasts registry; defaults to CFRNA_WADE_CONTRASTS.
#' @return tibble, one row per contrast x unit, with the verdict and its reason.
cfrna_wade_screen <- function(libs, contrasts = CFRNA_WADE_CONTRASTS,
                              min_n = CFRNA_WADE_MIN_N,
                              tail_min_n = CFRNA_WADE_TAIL_MIN_N,
                              panel_v_max = CFRNA_WADE_PANEL_V_MAX,
                              tail_q = CFRNA_WADE_TAIL_Q, seed = 1L) {
  idx <- cfrna_index_libraries(libs, metric = libs, by = "on_spliced")
  rows <- lapply(contrasts, function(spec) {
    dplyr::bind_rows(lapply(c("library", "index"), function(unit) {
      d  <- if (unit == "library") libs else idx
      rr <- cfrna_wade_resolve(spec, d)
      n1 <- length(rr$case); n0 <- length(rr$ctrl)
      nprobs <- min(n1, n0)
      k_tail <- max(1L, ceiling(tail_q * nprobs))

      sub <- d[d$library %in% c(rr$case, rr$ctrl), ]
      g   <- ifelse(sub$library %in% rr$case, "case", "ctrl")
      des <- cfrna_gs_design(sub$capture)
      tb  <- table(g, des)
      V   <- cfrna_wade_cramer_v(tb)
      tbf <- tb[rowSums(tb) > 0, colSums(tb) > 0, drop = FALSE]
      pp  <- if (any(dim(tbf) < 2)) NA_real_ else {
        set.seed(seed)
        tryCatch(stats::fisher.test(tbf, simulate.p.value = TRUE, B = 20000)$p.value,
                 error = function(e) NA_real_)
      }
      dep <- sub$on_spliced
      dfc <- if (n1 > 0 && n0 > 0)
        stats::median(dep[g == "case"]) / stats::median(dep[g == "ctrl"]) else NA_real_
      dpv <- if (min(n1, n0) >= 3)
        tryCatch(stats::wilcox.test(dep ~ g)$p.value, error = function(e) NA_real_)
        else NA_real_

      # --- verdict ---------------------------------------------------
      # ORDER MATTERS. A design confound is fatal at any sample size, so it is
      # checked first and cannot be downgraded by the size tiers below. Size
      # limits are NOT fatal -- they cap what may be claimed, which is the
      # `descriptive` tier, not a refusal.
      reasons <- character()
      verdict <- "run"

      if (!is.na(V) && V > panel_v_max) {
        verdict <- "refuse"
        reasons <- c(reasons, sprintf(
          "probe design predicts group (Cramer V=%.2f > %.2f) -- uninterpretable at any n",
          V, panel_v_max))
      }

      if (verdict != "refuse") {
        if (nprobs < min_n) {
          verdict <- "descriptive"
          reasons <- c(reasons, sprintf(
            "smaller group n=%d below the %d nomination floor: %d-point quantile grid, %d-point tail -- effect sizes only, no nomination",
            nprobs, min_n, nprobs, k_tail))
        } else if (nprobs < tail_min_n) {
          verdict <- "caution"
          reasons <- c(reasons, sprintf(
            "tail window is %d order statistic(s) at n=%d -- subset axis descriptive only, nominate on the bulk axis",
            k_tail, nprobs))
        }
        if (!is.na(pp) && pp < 0.05) {
          if (verdict == "run") verdict <- "caution"
          reasons <- c(reasons, sprintf("design-version association p=%.3g (V=%.2f)", pp, V))
        }
        if (!is.na(dpv) && dpv < 0.05) {
          if (verdict == "run") verdict <- "caution"
          reasons <- c(reasons, sprintf("depth differs %.2fx (p=%.3g)", dfc, dpv))
        }
      }
      tibble::tibble(
        id = spec$id, label = spec$label, family = spec$family, unit = unit,
        design = if (is.null(spec$design)) "all" else paste(spec$design, collapse = "+"),
        n_case = n1, n_ctrl = n0, nprobs = nprobs, k_tail = k_tail,
        panel_v = V, panel_p = pp, depth_fc = dfc, depth_p = dpv,
        verdict = verdict,
        # The subset axis needs enough order statistics to mean anything; this
        # flag is what cfrna_wade_nominate(tail_ok=) consumes, so the threshold
        # is decided once here rather than at each call site.
        tail_usable = nprobs >= tail_min_n,
        nominatable = verdict %in% c("run", "caution"),
        reason = if (length(reasons)) paste(reasons, collapse = "; ") else "no screen fired",
        rationale = spec$rationale)
    }))
  })
  dplyr::bind_rows(rows)
}

#' The contrasts to actually run: those whose INDEX-level verdict is not refuse.
#'
#' Index level decides. A contrast that passes only because replicate libraries
#' inflate its group sizes is not feasible, it is pseudoreplicated -- and the
#' index level is where that shows. `descriptive` contrasts ARE returned: they
#' get run, they just do not contribute nominations (`nominatable` is FALSE).
cfrna_wade_runnable <- function(screen) {
  keep <- screen |> dplyr::filter(unit == "index", verdict != "refuse")
  screen |>
    dplyr::filter(unit == "library", id %in% keep$id) |>
    dplyr::left_join(dplyr::select(keep, id, index_verdict = verdict,
                                   index_reason = reason,
                                   index_tail_usable = tail_usable,
                                   index_nominatable = nominatable,
                                   index_n_case = n_case, index_n_ctrl = n_ctrl),
                     by = "id")
}

#' Would restricting to one design version rescue a panel-confounded contrast?
#'
#' For each design version, the group sizes a contrast would have if it were run
#' within that version alone. This turns "refused" into a specific, actionable
#' statement -- which panel, how many samples short -- instead of a dead end,
#' and it is what identified `cancer_vs_screenpos_v6` as buildable.
#'
#' @return tibble: contrast, design, n_case, n_ctrl, nprobs, feasible
cfrna_wade_panel_matched <- function(libs, spec, min_n = CFRNA_WADE_MIN_N) {
  d <- cfrna_strata_annotate(libs)
  d$design <- cfrna_gs_design(d$capture)
  dplyr::bind_rows(lapply(sort(unique(d$design)), function(v) {
    s2 <- modifyList(spec, list(design = v))
    rr <- tryCatch(cfrna_wade_resolve(s2, d), error = function(e) NULL)
    if (is.null(rr)) return(NULL)
    tibble::tibble(id = spec$id, design = v,
                   n_case = length(rr$case), n_ctrl = length(rr$ctrl),
                   nprobs = min(length(rr$case), length(rr$ctrl)),
                   feasible = min(length(rr$case), length(rr$ctrl)) >= min_n)
  }))
}

#' Run one contrast through wade.R, with caching.
#'
#' CACHE KEY. The key must change whenever the RESULT would, or a stale cache
#' silently ships the wrong numbers -- this project has been burned by stale
#' cached scalars before (see the v8 setup chunk's rarefaction note). The key
#' therefore covers: contrast id, cohort NAME and fingerprint (store refresh),
#' gene count, the sorted library ids of BOTH groups, nperms, tail_q and seed. A
#' weekly store refresh, a min_detect change, or one library moving between
#' groups all produce a different file.
#'
#' The key uses the cohort's `name` rather than its panel list: a cohort is now
#' a gene list plus a library list (cohort_def.R), and two cohorts can share a
#' panel set while differing in genes or libraries. Name plus gene count plus
#' the library ids pins the actual inputs; `spec$panels` would not.
#'
#' @param counts     genes x libs count matrix (rigel total counts).
#' @param normalizer genes x libs effective-length matrix.
#' @param cd         the cohort_def these came from.
#' @param spec       a registry entry.
#' @param cache_dir  directory for the .rds files; NULL disables caching.
#' @return the scored wade frame, with contrast metadata as attributes.
cfrna_wade_contrast <- function(counts, normalizer, cd, spec,
                                cache_dir = NULL, nperms = CFRNA_WADE_NPERMS,
                                tail_q = CFRNA_WADE_TAIL_Q, seed = 1L,
                                verbose = TRUE) {
  rr <- cfrna_wade_resolve(spec, cd$libraries)
  cohort_name <- cd$name %||% "unnamed"
  key <- paste(spec$id, cohort_name, cd$fingerprint,
               length(cd$gene_idx), nperms, tail_q, seed,
               paste(sort(rr$case), collapse = ","),
               paste(sort(rr$ctrl), collapse = ","), sep = "|")
  hash <- substr(digest_string(key), 1, 16)
  # Cohort names are human strings ("all panels", "v3 omitted"), so slugify for
  # the filename; the hash carries the exact identity either way.
  slug <- gsub("[^a-z0-9]+", "-", tolower(cohort_name))
  f <- if (!is.null(cache_dir))
    file.path(cache_dir, sprintf("wade_%s_%s_%s.rds", spec$id, slug, hash)) else NULL

  if (!is.null(f) && file.exists(f)) {
    res <- readRDS(f)
    if (verbose) cat("  [cache]", spec$id, "->", basename(f), "\n")
    attr(res, "cached") <- TRUE
    return(res)
  }

  gi <- cd$gene_idx
  Cc <- counts[gi, , drop = FALSE]
  Cn <- normalizer[gi, , drop = FALSE]
  t0 <- Sys.time()
  res <- wade_run(Cc, Cn, rr$case, rr$ctrl, gene_names = cd$gene_name,
                  nperms = nperms, tail_q = tail_q, seed = seed, verbose = FALSE)
  el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  attr(res, "contrast_id") <- spec$id
  attr(res, "contrast_label") <- spec$label
  attr(res, "family") <- spec$family
  attr(res, "elapsed_sec") <- el
  attr(res, "nperms") <- nperms
  attr(res, "tail_q") <- tail_q
  attr(res, "cohort") <- cohort_name
  attr(res, "n_genes") <- length(gi)
  attr(res, "fingerprint") <- cd$fingerprint
  attr(res, "cached") <- FALSE
  if (!is.null(f)) {
    dir.create(dirname(f), showWarnings = FALSE, recursive = TRUE)
    saveRDS(res, f)
  }
  if (verbose) cat(sprintf("  [run]   %s  %d vs %d  %.1fs\n", spec$id,
                           length(rr$case), length(rr$ctrl), el))
  res
}

#' Small stable string hash for cache keys.
#'
#' `digest` is not in this renv and a cache key does not warrant adding a
#' dependency, so this is a hand-rolled polynomial hash over the code points.
#'
#' WHY NOT FNV-1a, WHICH IS THE OBVIOUS CHOICE. FNV needs a 32-bit XOR, and R's
#' bitwXor() takes INTEGERS: the running FNV state exceeds .Machine$integer.max
#' almost immediately, so bitwXor() coerces and returns NA (with a warning that
#' is easy to miss), and sprintf("%08x") then fails on the double. A first
#' version of this function did exactly that. Two 31-bit modular polynomial
#' hashes with different bases stay inside double's exact-integer range
#' (2^53) at every step, need no bit operations, and give a 62-bit key -- ample
#' for a filename discriminator that also carries the contrast id and panel set.
#'
#' Deterministic across sessions and platforms: no hashing of R objects, no
#' serialisation, just arithmetic on utf8ToInt().
digest_string <- function(s) {
  b <- utf8ToInt(as.character(s))
  h1 <- 2147483647; h2 <- 1103515245        # two 31-bit moduli-safe seeds
  M  <- 2147483647                          # 2^31 - 1, Mersenne prime
  for (ch in b) {
    h1 <- (h1 * 131 + ch) %% M
    h2 <- (h2 * 137 + ch) %% M
  }
  sprintf("%08x%08x", as.integer(h1 %% 2147483647L), as.integer(h2 %% 2147483647L))
}

#' Run every runnable contrast, returning a named list of scored frames.
cfrna_wade_run_all <- function(counts, normalizer, cd, ids,
                               contrasts = CFRNA_WADE_CONTRASTS,
                               cache_dir = NULL, ...) {
  specs <- contrasts[vapply(contrasts, function(s) s$id %in% ids, logical(1))]
  if (!length(specs)) stop("cfrna_wade_run_all(): no registry entry matches ", 
                           paste(ids, collapse = ", "))
  out <- lapply(specs, function(s)
    cfrna_wade_contrast(counts, normalizer, cd, s, cache_dir = cache_dir, ...))
  names(out) <- vapply(specs, function(s) s$id, character(1))
  out
}

#' Nomination set for one contrast: top-N on either axis.
#'
#' v7's rule, preserved: nomination is by RANK, not by p-value. With ~2,200
#' genes and 22 controls the permutation p-values cannot survive BH, so the
#' ranked score is the discovery product and p is effect-strength annotation.
#' `tail_ok = FALSE` drops the subset axis from nomination, which is what the
#' screen's `caution` verdict on a small group requires.
cfrna_wade_nominate <- function(res, top_n = 100L, tail_ok = TRUE) {
  hit <- res$rank <= top_n
  if (tail_ok) hit <- hit | res$tail.rank <= top_n
  res$nominated <- hit
  res
}

#' Chance-expectation check on the raw signal.
#'
#' The number v7 quoted to show the signal is real despite failing BH: how
#' many genes clear nominal p < cut against how many are expected. Returned as
#' a tibble so a table can print all contrasts at once.
cfrna_wade_signal_check <- function(res_list, cut = 0.01) {
  dplyr::bind_rows(lapply(names(res_list), function(nm) {
    r <- res_list[[nm]]
    ng <- nrow(r)
    tibble::tibble(
      id = nm, cohort = attr(r, "cohort") %||% NA_character_,
      label = attr(r, "contrast_label"),
      # n_case / n_ctrl are set by wade_run() itself, not by the contrast
      # wrapper -- reading them from the frame keeps the reported group sizes
      # the ones the statistic actually used.
      n_case = attr(r, "n_case"), n_ctrl = attr(r, "n_ctrl"),
      genes = ng,
      n_bulk = sum(r$p.diff < cut, na.rm = TRUE),
      n_tail = sum(r$p.tail < cut, na.rm = TRUE),
      expected = round(cut * ng),
      enrich_bulk = round(sum(r$p.diff < cut, na.rm = TRUE) / (cut * ng), 2),
      enrich_tail = round(sum(r$p.tail < cut, na.rm = TRUE) / (cut * ng), 2),
      min_padj_bulk = min(r$padj.diff, na.rm = TRUE),
      min_padj_tail = min(r$padj.tail, na.rm = TRUE),
      elapsed_sec = attr(r, "elapsed_sec"))
  }))
}

#' Cross-contrast gene classification.
#'
#' Generalises v7's cancer-specific / shared / benign-specific split to an
#' arbitrary set of contrasts sharing a reference: for each gene, which
#' contrasts nominated it. Returns one row per gene with a membership string,
#' so the section can ask "what did ONLY the pancreatic contrast find" without
#' another bespoke join.
cfrna_wade_cross <- function(res_list, top_n = 100L, tail_ok = TRUE) {
  nom <- lapply(res_list, function(r)
    cfrna_wade_nominate(r, top_n = top_n, tail_ok = tail_ok))
  genes <- sort(unique(unlist(lapply(nom, function(r) r$gene))))
  out <- tibble::tibble(gene = genes)
  for (nm in names(nom)) {
    r <- nom[[nm]]
    out[[paste0(nm, ".rank")]] <- r$rank[match(genes, r$gene)]
    out[[paste0(nm, ".tail_rank")]] <- r$tail.rank[match(genes, r$gene)]
    out[[paste0(nm, ".diff_mean")]] <- r$diff.mean[match(genes, r$gene)]
    out[[paste0(nm, ".nom")]] <- r$nominated[match(genes, r$gene)]
  }
  nom_cols <- grep("\\.nom$", names(out), value = TRUE)
  out$n_nominating <- rowSums(as.matrix(out[, nom_cols, drop = FALSE]), na.rm = TRUE)
  out$nominated_by <- apply(as.matrix(out[, nom_cols, drop = FALSE]), 1, function(v) {
    v[is.na(v)] <- FALSE
    if (!any(v)) return("")
    paste(sub("\\.nom$", "", nom_cols)[v], collapse = "+")
  })
  dplyr::arrange(out, dplyr::desc(n_nominating))
}
