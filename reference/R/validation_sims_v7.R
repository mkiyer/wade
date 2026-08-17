# =====================================================================
# validation_sims_v7.R -- WADE method validation on simulated data
# ---------------------------------------------------------------------
# PROVENANCE. Extracted verbatim-in-substance from
# cfrna_analysis/analysis_cfrna_v7.qmd, section "## 9.6 Method validation"
# (cfRNA git SHA 828f2f1c6a10afd182fee0e119ba3595d98b6a72), which contained
# two chunks: `wade-validation-sim` (the three simulations) and
# `fig-wade-validation` (the three-panel figure). Both are reproduced here as
# one standalone script. See ../PROVENANCE.md and ./CHECKSUMS.txt.
#
# WHAT IT SHOWS. Three claims about the statistic, each with its own
# simulation and its own panel:
#
#   (a) NULL CALIBRATION. Poisson counts, random per-gene normalizers, labels
#       carrying no signal. The permutation p-values on both axes should be
#       Uniform(0,1); tested by a one-sample Kolmogorov-Smirnov test and by
#       the realised type-I error at the nominal 0.05 level. This is the
#       claim that WADE does not manufacture significance.
#
#   (b) SUBSET-VS-BULK DISCRIMINATION. A lognormal null background, plus two
#       planted gene classes: whole-group location shifts ("bulk") and genes
#       elevated in only 8% of cases ("subset"). The subset axis (tail.mean)
#       should separate the planted subset genes from the bulk shifts, which
#       is the entire reason the tail statistic exists -- diff.mean alone
#       collapses both to the same number.
#
#   (c) POWER VS SUBSET FRACTION, AGAINST THE COMBINATORIAL FLOOR. Detection
#       power (BH-adjusted tail p < 0.10) as the altered fraction of cases is
#       swept, plotted against the exact permutation floor
#           exp(lchoose(n1, k) - lchoose(nn, k)),
#       the probability that a random relabelling puts all k
#       signal-carrying samples in the case group. That probability is the
#       smallest p-value any label-permutation test can return for a k-sample
#       subset, so it is a property of the DESIGN, not of the implementation:
#       no amount of permutations, and no better tail fit, moves it. This is
#       the honest limit on subtype discovery at these group sizes.
#
# DISCREPANCY WITH THE NOTEBOOK'S NARRATIVE, MEASURED HERE. The v7 figure
# caption and prose put the collapse of (c) "below ~5% of cases", and the prose
# computed the resolution limit as the first swept fraction whose power exceeds
# 0.5. Re-running the chunk in this sandbox, power is 0 at every fraction up to
# and including 20% and 1 from 35% on -- so the first fraction over 0.5 power is
# 35% (about 27 of 77 cases), not ~5%. This was reproduced exactly across two
# runs and at both candidate control counts (see N_CONTROLS below).
#
# The ~5% figure is not supported by what this script produces. The transition
# location is fully explained by the BH arithmetic noted at the power table
# below, so the honest statement of the limit at these group sizes is the one
# printed at run time, not the notebook's. Whether the notebook's ~5% came from
# a different sweep grid, a different detection rule, or a stale narrative
# cannot be determined from the source alone; the numbers this script prints
# are the ones to trust, because they are the ones it computed.
#
# WHAT IT IS NOT. This is a simulation study on synthetic data. It says
# nothing about whether any particular cfRNA gene nomination is real; it
# establishes that the statistic behaves as designed at the design's group
# sizes.
#
# HOW TO RUN. From this directory, with the pinned renv library active:
#   export PATH="/usr/local/bin:$PATH"
#   RENV_CONFIG_SANDBOX_ENABLED=FALSE Rscript validation_sims_v7.R
# See README.md for the offline-cache constraint.
# =====================================================================

# ---------------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------------
# COHORT GEOMETRY. The notebook read these from the live cohort as the inline
# values `n_cancer` and `n_noncancer` (v7 §9 chunk `wade-inputs`), so the
# chunk could not run outside the notebook. They are hardcoded here.
#
# These are the cfRNA v7 primary contrast's own group sizes, chosen so the
# simulation inherits the real design's geometry rather than a round number.
# They are NOT re-derivable here: this repo carries no cohort data, which is
# exactly why they became parameters.
#
# N_CASES = 77 is unambiguous: the `fig-wade-validation` caption says "the real
# group sizes (77 cases, 18 controls)", and the §9 narrative independently uses
# 77 ("a gene elevated in 15 of 77 cases").
#
# N_CONTROLS IS AMBIGUOUS IN THE SOURCE, AND THE CONFLICT IS RECORDED RATHER
# THAN SILENTLY RESOLVED. The notebook's simulation chunk began
#     n0v <- n_noncancer; n1v <- n_cancer
# so the control count it actually ran with was `n_noncancer` -- the POOLED
# non-cancer reference (healthy controls + benign/precancer), not the healthy
# controls alone (`n_ctrl`). Three pieces of internal evidence indicate those
# two were different numbers, and that n_noncancer was 33:
#   * "With heterogeneous cancers and at most `r n_noncancer` reference
#     libraries" -- "at most" makes n_noncancer the LARGEST reference group,
#     so n_noncancer > n_ctrl.
#   * A hand-written code comment in the same section: "18-33 controls cannot
#     resolve permutation p-values small enough to survive BH". Two numbers,
#     spanning the three contrasts' reference groups.
#   * The contrast list gives Cancer vs Non-Cancer as `n_cancer` vs
#     `n_noncancer` and Cancer vs Control as `n_cancer` vs `n_ctrl`, i.e. the
#     18 in "18-control cohort" is n_ctrl.
# On that reading the figure caption's "18 controls" names n_ctrl while the
# code ran on n_noncancer = 33 -- a mislabelled caption, not a second design.
#
# The default below is the caption's 18, because that is the only control
# count stated as a literal anywhere in the source. Set N_CONTROLS <- 33L to
# reproduce the other reading. MEASURED: the two settings were both run in
# this sandbox and agree on every qualitative conclusion, including the exact
# location of the power transition in (c) (35% of cases in both). Details:
#   n0 = 18: KS p = 0.6554 (bulk) / 0.2327 (subset); type-I 0.0512 / 0.0488
#   n0 = 33: KS p = 0.2671 (bulk) / 0.4948 (subset); type-I 0.0600 / 0.0638
# So nothing this script claims depends on resolving the ambiguity.
#
# The geometry matters more than the totals. min(n0, n1) IS the quantile grid,
# so the top-10% tail window is ceiling(0.10 * min(n0, n1)) order statistics:
# 2 points at n0 = 18, 4 at n0 = 33. Every result below is conditional on that.
N_CASES    <- 77L
N_CONTROLS <- 18L

# Label permutations per wade() call. The notebook used 1000 here (its
# production contrasts used 2000; see CFRNA_WADE_NPERMS in
# downstream/wade_contrasts.R). Kept at the notebook's value: nperms sets the
# resolution of every p-value below, so lowering it would change the KS
# statistics and the power curve rather than just the runtime. Nine wade()
# calls at this setting run in a few minutes on one core.
NPERMS <- 1000L

# Genes per simulation, and the planted-signal geometry. All from the
# notebook; named here so the script is readable without cross-referencing.
G_NULL       <- 800L    # (a) null-calibration genes
G_BACKGROUND <- 600L    # (b),(c) shared null background
N_PLANTED    <- 25L     # (b) genes per planted class (bulk, subset)
N_SUBSET_PWR <- 80L     # (c) planted subset genes per sweep point
SUBSET_FRAC  <- 0.08    # (b) fraction of cases carrying the planted subset
SPIKE_MU     <- 6.2     # (b),(c) lognormal meanlog of the subset spike
LEN_CONST    <- 4       # (b),(c) constant per-gene normalizer
FRACS        <- c(0.03, 0.05, 0.08, 0.12, 0.20, 0.35, 0.50)  # (c) sweep

OUT_FIG <- "validation_sims_v7.png"

# ---------------------------------------------------------------------
# DEPENDENCIES
# ---------------------------------------------------------------------
# wade.R itself needs matrixStats, tibble and dplyr. The plotting block
# additionally needs ggplot2, tidyr, patchwork and scales; all four are in
# the pinned lockfile, but the block is guarded so the statistical results
# still print in a leaner library.
source("wade.R")
suppressPackageStartupMessages({
  library(dplyr)
})

PLOT_PKGS <- c("ggplot2", "tidyr", "patchwork", "scales")
can_plot  <- all(vapply(PLOT_PKGS, requireNamespace, logical(1), quietly = TRUE))

n0v <- N_CONTROLS
n1v <- N_CASES
nn  <- n0v + n1v
condv <- c(rep(1L, n1v), rep(0L, n0v))                 # case-first

cat(sprintf("WADE validation: %d cases vs %d controls (nprobs = %d, tail window = %d)\n",
            n1v, n0v, min(n0v, n1v), max(1L, ceiling(0.10 * min(n0v, n1v)))))
cat(sprintf("nperms = %d per contrast\n\n", NPERMS))

# =====================================================================
# (a) NULL CALIBRATION
# ---------------------------------------------------------------------
# Poisson counts at a common rate, random per-gene normalizer (the vector
# form of wade_normalize()'s `normalizer` argument), labels carrying no
# signal by construction. Any departure from Uniform(0,1) here is the method
# inventing signal.
# =====================================================================
cat("(a) null calibration ...\n")
set.seed(2024)
Cnull <- matrix(rpois(G_NULL * nn, lambda = 30), G_NULL, nn)
Lnull <- sample(1:12, G_NULL, replace = TRUE)
resN  <- wade(Cnull, Lnull, wade_lib_size(Cnull, Lnull), condv,
              nperms = NPERMS, seed = 1, verbose = FALSE)
# CAVEAT, carried from the notebook unchanged. Permutation p-values are
# DISCRETE -- the unrefined ones are multiples of 1/(nperms+1) -- so the
# one-sample KS test warns that ties should not be present, and its p-value is
# conservative rather than exact. The warning is expected, not a defect; it is
# a property of any permutation test's calibration check, and a port's
# equivalent check will produce it too. Read the KS p-values as "no detectable
# departure from uniform at this resolution", and read the type-I error rates
# below, which do not assume continuity, alongside them.
ks_diff <- ks.test(resN$p.diff, "punif")$p.value
ks_tail <- ks.test(resN$p.tail, "punif")$p.value
t1_diff <- mean(resN$p.diff < 0.05); t1_tail <- mean(resN$p.tail < 0.05)

cat(sprintf("    KS vs Uniform(0,1):  diff.mean p = %.4f   tail.mean p = %.4f\n",
            ks_diff, ks_tail))
cat(sprintf("    type-I error @0.05:  diff.mean = %.4f     tail.mean = %.4f\n\n",
            t1_diff, t1_tail))

# =====================================================================
# (b) SUBSET-VS-BULK DISCRIMINATION
# ---------------------------------------------------------------------
# `mk_bulk` shifts the whole case group's lognormal meanlog; `mk_subset`
# leaves the case group at the null and replaces a random `frac` of it with a
# high-expressing spike. Both draw from the ambient RNG state, so the
# set.seed() calls below (and the order of the three matrix constructions)
# are what make this reproducible -- do not reorder them.
# =====================================================================
cat("(b) subset vs bulk discrimination ...\n")
mk_bulk   <- function(shift) c(rlnorm(n1v, 3 + shift, 0.6), rlnorm(n0v, 3, 0.6))
mk_subset <- function(frac, spikemu = SPIKE_MU) {
  x <- c(rlnorm(n1v, 3, 0.6), rlnorm(n0v, 3, 0.6))
  sp <- sample(seq_len(n1v), max(1, round(frac * n1v)))
  x[sp] <- rlnorm(length(sp), spikemu, 0.3); x
}
set.seed(11)
BG   <- t(replicate(G_BACKGROUND, c(rlnorm(n1v, 3, 0.6), rlnorm(n0v, 3, 0.6))))
BULK <- t(sapply(seq_len(N_PLANTED), function(i) mk_bulk(runif(1, 0.5, 0.9))))
SUB  <- t(sapply(seq_len(N_PLANTED), function(i) mk_subset(SUBSET_FRAC)))
MM <- rbind(BG, BULK, SUB); LL <- rep(LEN_CONST, nrow(MM))
lab <- rep(c("null", "bulk", "subset"), c(G_BACKGROUND, N_PLANTED, N_PLANTED))
rB <- wade(MM, LL, wade_lib_size(MM, LL), condv, nperms = NPERMS,
           seed = 1, verbose = FALSE) |>
  dplyr::mutate(label = lab)

# Separation summary. The figure shows the same thing geometrically; these
# numbers are what a reader can check without opening the PNG.
sep <- rB |>
  dplyr::group_by(label) |>
  dplyr::summarise(n = dplyr::n(),
                   median_diff_mean = stats::median(diff.mean),
                   median_tail_mean = stats::median(tail.mean),
                   median_tail_over_diff = stats::median(tail.mean / diff.mean),
                   .groups = "drop")
print(as.data.frame(sep), digits = 4)
cat("\n")

# =====================================================================
# (c) POWER VS SUBSET FRACTION, AGAINST THE COMBINATORIAL FLOOR
# ---------------------------------------------------------------------
# `pfloor(k)` is the probability that a random relabelling assigns all k
# signal-carrying samples to the case group. No label-permutation test can
# report a p-value below it, so it is the design's own detection floor.
# Reuses BG from (b): the background is held fixed across sweep points so
# only the planted subset fraction varies.
# =====================================================================
cat("(c) power vs subset fraction ...\n")
pfloor <- function(k) exp(lchoose(n1v, k) - lchoose(nn, k))   # all-k-in-case probability
powtab <- dplyr::bind_rows(lapply(FRACS, function(fr) {
  k <- max(1, round(fr * n1v)); set.seed(200 + k)
  S <- t(sapply(seq_len(N_SUBSET_PWR), function(i) mk_subset(fr)))
  MMp <- rbind(BG, S); LLp <- rep(LEN_CONST, nrow(MMp))
  rr <- wade(MMp, LLp, wade_lib_size(MMp, LLp), condv, nperms = NPERMS,
             seed = 1, verbose = FALSE)
  is_sub <- c(rep(FALSE, G_BACKGROUND), rep(TRUE, N_SUBSET_PWR))
  padj_tail <- p.adjust(rr$p.tail, "BH")
  tibble::tibble(frac = fr, k = k, floor = pfloor(k),
                 power = mean(padj_tail[is_sub] < 0.10))
}))
# NOTE ON THE NOTEBOOK'S IDIOM: the original used purrr::map_dfr(), which is
# superseded in purrr 1.x. dplyr::bind_rows(lapply(...)) is the same
# computation and drops the purrr dependency from the statistical core.
print(as.data.frame(powtab), digits = 4)

# WHY THE CURVE IS A STEP RATHER THAN A RAMP, and why that is the floor
# speaking. Detection here is BH(q = 0.10) over G_BACKGROUND + N_SUBSET_PWR
# genes, of which N_SUBSET_PWR are true positives. If the planted genes' tail
# p-values are pinned at the combinatorial floor, BH can declare them only when
#     floor * (G_BACKGROUND + N_SUBSET_PWR) / N_SUBSET_PWR  <=  0.10,
# i.e. only when floor <= 0.10 * 80/680 = 0.0118. Comparing that threshold
# against the `floor` column predicts the observed 0->1 transition exactly at
# every swept fraction. So the step is not a simulation artefact and not a
# resolution limit of NPERMS: it is the design's own floor crossing the
# multiple-testing threshold. Raising nperms cannot move it.
first_over_half <- which(powtab$power > 0.5)[1]
if (!is.na(first_over_half)) {
  cat(sprintf("\n    power first exceeds 0.5 at %.0f%% of cases (~%d samples of %d)\n",
              100 * powtab$frac[first_over_half],
              round(powtab$frac[first_over_half] * n1v), n1v))
} else {
  cat("\n    power does not exceed 0.5 at any swept subset fraction\n")
}

# =====================================================================
# FIGURE (from the notebook's `fig-wade-validation` chunk)
# ---------------------------------------------------------------------
# Kept because it is the evidence, not decoration: (a) is a P-P plot against
# the uniform diagonal, (b) reads the vertical distance above y = x as the
# subset signal, and (c) puts the power curve and the combinatorial floor on
# one axis so the collapse point is legible.
# =====================================================================
if (!can_plot) {
  cat(sprintf("\n[figure skipped: missing %s]\n",
              paste(setdiff(PLOT_PKGS, rownames(installed.packages())),
                    collapse = ", ")))
} else {
  suppressPackageStartupMessages(library(ggplot2))
  col_bulk <- "#B9770E"; col_sub <- "#1F618D"; col_null <- "grey70"
  th_v <- theme_minimal(base_size = 10) +
    theme(plot.title = element_text(size = 10), panel.grid.minor = element_blank(),
          legend.position = "top", legend.title = element_blank(),
          legend.text = element_text(size = 8))

  qdf <- tibble::tibble(theo = ppoints(nrow(resN)),
                        diff = sort(resN$p.diff), tail = sort(resN$p.tail)) |>
    tidyr::pivot_longer(c(diff, tail), names_to = "stat", values_to = "obs")
  pA <- ggplot(qdf, aes(theo, obs, colour = stat)) +
    geom_abline(slope = 1, intercept = 0, colour = "grey55", linewidth = 0.4,
                linetype = "2222") +
    geom_step(linewidth = 0.7) +
    scale_colour_manual(values = c(diff = "#7D3C98", tail = "#148F77"),
                        labels = c(diff = "diff.mean", tail = "tail.mean")) +
    coord_equal() +
    labs(title = "(a) null calibration", x = "theoretical quantile",
         y = "observed p-value") + th_v

  db <- rB |> dplyr::mutate(lab = factor(label, levels = c("null", "bulk", "subset")),
                            dm = pmax(diff.mean, 0.1), tm = pmax(tail.mean, 0.1))
  pB <- ggplot(db, aes(dm, tm, colour = lab)) +
    geom_abline(slope = 1, intercept = 0, colour = "grey70", linewidth = 0.4,
                linetype = "2222") +
    geom_point(data = dplyr::filter(db, label == "null"), size = 0.5, alpha = 0.22) +
    geom_point(data = dplyr::filter(db, label != "null"), size = 1.6, alpha = 0.85) +
    scale_x_log10() + scale_y_log10() +
    scale_colour_manual(values = c(null = col_null, bulk = col_bulk, subset = col_sub)) +
    labs(title = "(b) subset vs bulk discrimination",
         x = "diff.mean (bulk)", y = "tail.mean (subset)") + th_v

  pC <- ggplot(powtab, aes(frac)) +
    geom_line(aes(y = power, colour = "detection power"), linewidth = 0.8) +
    geom_point(aes(y = power, colour = "detection power"), size = 1.8) +
    geom_line(aes(y = floor, colour = "permutation floor"), linewidth = 0.6,
              linetype = "2222") +
    scale_colour_manual(values = c("detection power" = col_sub,
                                   "permutation floor" = "grey45")) +
    scale_x_continuous(labels = scales::percent) + ylim(0, 1) +
    labs(title = "(c) power vs subset size", x = "fraction of cases altered",
         y = "power / floor") + th_v

  fig <- patchwork::wrap_plots(pA, pB, pC, nrow = 1)
  ggsave(OUT_FIG, plot = fig, width = 12, height = 4, dpi = 150)
  cat(sprintf("\nfigure written: %s\n", OUT_FIG))
}

# ---------------------------------------------------------------------
# Machine-readable summary, so a port can diff against these numbers.
# ---------------------------------------------------------------------
summary_tbl <- tibble::tibble(
  quantity = c("ks_p_diff", "ks_p_tail", "type1_diff_0.05", "type1_tail_0.05"),
  value    = c(ks_diff, ks_tail, t1_diff, t1_tail))
utils::write.csv(as.data.frame(summary_tbl), "validation_sims_v7_calibration.csv",
                 row.names = FALSE)
utils::write.csv(as.data.frame(powtab), "validation_sims_v7_power.csv",
                 row.names = FALSE)
cat("wrote validation_sims_v7_calibration.csv, validation_sims_v7_power.csv\n")
