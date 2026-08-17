# =====================================================================
# R/wade_figures.R -- figures for the WADE discovery section
#
# RULES carried from the other figure modules (R/qc_figures.R,
# R/pca_figures.R, R/discovery_figures.R):
#   * NO CROSS-MODULE GLOBALS. This module defines every colour and every
#     geometry itself. Group display strings are NOT resolved here either: they
#     arrive as the `label` column of the screen frame, which R/wade_contrasts.R
#     built from the contrast registry, so a figure cannot invent its own
#     wording for a contrast. (CFRNA_STRATUM_LABELS in R/control_strata.R is
#     the equivalent constant for a bare STRATUM name; no figure in this module
#     labels a stratum directly, so nothing here reads it.)
#   * ONE COLOUR SCALE PER FIGURE. Never two continuous scales in one panel.
#   * A subtitle quoting a number computes it from the data ACTUALLY DRAWN,
#     rather than taking it as an argument that can go stale.
#   * Gene symbols are italicised; group names are not.
#   * Abundance is log2(CPM+1) on a linear axis. WADE's own quantities
#     (diff.mean, tail.mean) are TPM-scale differences, NOT abundances, and
#     they legitimately span four orders of magnitude -- so the volcano uses
#     log axes on x+1 / y+1. The +1 is what makes zero representable, which is
#     the same reason the house rule bans a bare log axis on abundance.
#   * EVERY LOG AXIS GETS AN EXPLICIT RANGE FROM ITS DATA. Build gate 9/10
#     enforces this for the interactive panels; it is right for the static ones
#     too, so the limits here are computed from the plotted values rather than
#     left to the default whole-decade rounding.
#
# GENE-LEVEL, SO STATIC. The crosstalk stack in R/qc_interactive.R is keyed on
# `library` and shares one selection document-wide. These panels have genes as
# the unit, so they cannot join that selection and are plain ggplot on purpose.
# =====================================================================

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr)
})

# Verdict palette. Ordinal severity, so a single-hue ramp from neutral to
# alarm, NOT a categorical rainbow -- the four verdicts are ordered
# (run < caution < descriptive < refuse) and the colour should say so.
# `refuse` gets the one alarm hue in this module and no data series reuses it.
WADE_VERDICT_PAL <- c(
  run         = "#1B7837",
  caution     = "#B8860B",
  descriptive = "#4A6FA5",
  refuse      = "#A50F15")

# Contrast-family palette, for the cross-contrast panels. Okabe-Ito derived so
# it survives deuteranopia; deliberately shares no hue with the verdict ramp,
# because both can appear in the same figure sequence.
WADE_FAMILY_PAL <- c(
  detection          = "#0072B2",
  organ              = "#D55E00",
  continuum          = "#009E73",
  control_vs_control = "#CC79A7")

# Nomination classes for the volcano. Grey is the unnominated reference class
# and is never used for a nominated gene. `nominated (bulk axis)` is the class
# used when the tail window is too small to support the subset/bulk split at
# all -- it gets its own neutral-dark hue rather than borrowing either of the
# split classes' colours, so a reader cannot mistake it for a measured "broad"
# call. Deuteranopia-safe against the other two.
WADE_NOM_PAL <- c(
  `subset-concentrated`   = "#1F618D",
  `broad / bulk`          = "#B9770E",
  `nominated (bulk axis)` = "#4A4A4A",
  `not nominated`         = "grey80")

# Case/control colours for the per-gene diagnostic. Threaded from v7 §9.3 so a
# reader comparing the two documents sees the same colour for the same role.
WADE_CASE_COL <- "#C0392B"
WADE_CTRL_COL <- "#2C3E50"
WADE_AREA_COL <- "#E67E22"

wade_theme <- function(base = 11) {
  theme_bw(base_size = base) +
    theme(plot.title = element_text(face = "bold", size = base + 0.5),
          plot.subtitle = element_text(size = base - 2.2, colour = "grey30"),
          panel.grid.minor = element_blank(),
          legend.key.size = grid::unit(0.85, "lines"),
          legend.title = element_text(size = base - 2.4),
          legend.text = element_text(size = base - 2.8))
}

#' Guard `tail.conc` for DISPLAY. Returns NA where the ratio is uninterpretable.
#'
#' WHY THIS IS NEEDED, MEASURED. `tail.conc` is a share: the signed area in the
#' tail window over the TOTAL signed area. When the total is near zero -- which
#' happens whenever the lower quantiles' differences cancel the upper ones --
#' the denominator vanishes and the ratio explodes. On the primary contrast,
#' 120 of 2,219 genes (5.4%) come back with |tail.conc| > 2, one as high as 149,
#' which is not a "share" of anything.
#'
#' wade.R already NA-guards this, but its threshold is |diff.mean * nprobs| <
#' 1e-8, i.e. |diff.mean| < 5e-10 on this contrast -- about nine orders of
#' magnitude too tight to catch the cases that actually occur. Rather than
#' change the statistic (which would invalidate every cached result for a
#' cosmetic column), the guard is applied at the DISPLAY layer and the method-
#' level fix is recorded in WADE_REPO_SCOPE.md as a porting item.
#'
#' Note that tail.conc slightly ABOVE 1 is legitimate -- it means the tail
#' carries more than the total because the bulk partially cancels -- so the
#' ceiling is deliberately loose rather than 1. Checked: no NOMINATED gene in
#' any contrast on either cohort is affected, so this guard changes no
#' nomination; it only stops an uninterpretable number being printed.
wade_tail_conc_display <- function(tail_conc, ceiling = 1.5) {
  ifelse(is.na(tail_conc) | abs(tail_conc) > ceiling, NA_real_, tail_conc)
}

#' Explicit limits for a log axis, from the data, with a little headroom.
#'
#' The house rule: a log axis without an explicit range lets the renderer round
#' out to whole decades and waste half the panel. Returns c(lo, hi) on the
#' PLOTTED scale (so pass the +1 values).
wade_log_limits <- function(x, pad_lo = 0.9, pad_hi = 1.25) {
  x <- x[is.finite(x) & x > 0]
  if (!length(x)) return(c(1, 10))
  c(max(min(x) * pad_lo, 1), max(x) * pad_hi)
}

#' Human-readable log breaks inside a range (1, 10, 100, 1k, 10k, ...).
wade_log_breaks <- function(lim) {
  p <- seq(floor(log10(lim[1])), ceiling(log10(lim[2])))
  b <- 10^p
  b <- b[b >= lim[1] / 2 & b <= lim[2] * 2]
  if (length(b) < 2) b <- pretty(lim, 4)
  b
}
wade_log_labels <- function(b) {
  ifelse(b >= 1e6, paste0(b / 1e6, "M"),
    ifelse(b >= 1e3, paste0(b / 1e3, "k"), as.character(b)))
}

# ---------------------------------------------------------------------
#' FIGURE: the contrast screen -- what is runnable, and what the screen said.
#'
#' The point of the figure is that feasibility has TWO independent axes (group
#' size and design confounding) and a contrast can fail on either. So: the
#' SMALLER group's size on x, Cramer's V on y, verdict as colour, and the two
#' thresholds as reference lines. A reader can see at a glance which constraint
#' bit which contrast.
#'
#' The x axis is log because the plotted quantity -- nprobs, the smaller group,
#' which is what sets the quantile grid -- spans 5 to 55 across both cohorts and
#' both units, an order of magnitude. Note this is NOT the case-group size (that
#' reaches 134): a contrast's resolution is set by its smaller group, so the
#' larger one is deliberately not on the axis.
#'
#' Both units are drawn -- library level as an open ghost marker, index level
#' filled -- with a segment joining them, because the DIFFERENCE between the two
#' is the pseudoreplication statement. Verdict colour is taken from the index
#' level, which is the one that decides.
#'
#' @param screen output of cfrna_wade_screen().
#' @param panel_v_max,min_n thresholds, for the reference lines.
disc_fig_wade_screen <- function(screen, panel_v_max = CFRNA_WADE_PANEL_V_MAX,
                                 min_n = CFRNA_WADE_MIN_N,
                                 tail_min_n = CFRNA_WADE_TAIL_MIN_N,
                                 cohort_label = NULL) {
  # NA V means the group x design table was DEGENERATE -- a panel-matched
  # contrast has one design version, so there is no association to measure.
  # That is not the same as V = 0 ("measured, and perfectly balanced"), and
  # plotting it at 0 would claim a measurement that was never made. Such
  # contrasts are drawn on a separate baseline row below the axis with an open
  # square, and the subtitle says how many.
  d <- screen
  n_nomeasure <- dplyr::n_distinct(d$id[is.na(d$panel_v) & d$unit == "index"])
  v_floor <- -0.055 * max(1, max(d$panel_v, na.rm = TRUE))
  d <- dplyr::mutate(d,
    v_plot = ifelse(is.na(panel_v), v_floor, panel_v),
    v_measured = !is.na(panel_v))
  idx <- dplyr::filter(d, unit == "index")
  lib <- dplyr::filter(d, unit == "library")
  seg <- dplyr::left_join(
    dplyr::select(idx, id, label, verdict, x1 = nprobs, y1 = v_plot),
    dplyr::select(lib, id, x0 = nprobs, y0 = v_plot), by = "id")

  # Order the label column by index-level group size so the text does not
  # collide and reads as a ranking.
  idx <- dplyr::arrange(idx, nprobs)
  lim <- wade_log_limits(c(lib$nprobs, idx$nprobs), pad_lo = 0.75, pad_hi = 1.9)
  br  <- wade_log_breaks(lim)

  ggplot(idx, aes(nprobs, v_plot)) +
    annotate("rect", xmin = lim[1], xmax = min_n, ymin = -Inf, ymax = Inf,
             fill = "grey92", alpha = 0.7) +
    annotate("rect", xmin = lim[1], xmax = Inf, ymin = panel_v_max, ymax = Inf,
             fill = WADE_VERDICT_PAL[["refuse"]], alpha = 0.06) +
    geom_hline(yintercept = panel_v_max, colour = WADE_VERDICT_PAL[["refuse"]],
               linetype = "2222", linewidth = 0.4) +
    geom_hline(yintercept = 0, colour = "grey80", linewidth = 0.3) +
    geom_vline(xintercept = min_n, colour = "grey45", linetype = "2222", linewidth = 0.4) +
    geom_vline(xintercept = tail_min_n, colour = "grey65", linetype = "dotted", linewidth = 0.4) +
    geom_segment(data = seg, aes(x = x0, y = y0, xend = x1, yend = y1),
                 colour = "grey65", linewidth = 0.3) +
    geom_point(data = lib, aes(nprobs, v_plot), shape = 21, size = 1.9,
               stroke = 0.4, fill = NA, colour = "grey55") +
    geom_point(data = dplyr::filter(idx, !v_measured), aes(fill = verdict),
               shape = 22, size = 3.1, stroke = 0.35, colour = "grey20") +
    geom_point(data = dplyr::filter(idx, v_measured), aes(fill = verdict),
               shape = 21, size = 3.1, stroke = 0.35, colour = "grey20") +
    ggrepel::geom_text_repel(aes(label = label), size = 2.6, colour = "grey15",
                             max.overlaps = 30, box.padding = 0.45,
                             segment.size = 0.25, segment.colour = "grey60",
                             min.segment.length = 0, seed = 1) +
    scale_x_log10(limits = lim, breaks = br, labels = wade_log_labels(br)) +
    scale_y_continuous(
      limits = c(if (n_nomeasure > 0) v_floor * 1.5 else 0,
                 max(1, max(d$panel_v, na.rm = TRUE) * 1.15)),
      breaks = seq(0, 1, 0.25),
      expand = expansion(mult = c(0.02, 0.04))) +
    scale_fill_manual(values = WADE_VERDICT_PAL, name = "screen verdict",
                      breaks = names(WADE_VERDICT_PAL)) +
    annotate("text", x = lim[1] * 1.05, y = panel_v_max, hjust = 0, vjust = -0.6,
             size = 2.7, colour = WADE_VERDICT_PAL[["refuse"]], fontface = "italic",
             label = sprintf("design predicts group (V > %.2f) \u2014 refuse", panel_v_max)) +
    annotate("text", x = min_n, y = Inf, hjust = -0.06, vjust = 1.6, size = 2.7,
             colour = "grey35", fontface = "italic",
             label = sprintf("n < %d: describe only", min_n)) +
    labs(x = "smaller group size = quantile-grid resolution (log)",
         y = "Cram\u00e9r's V, group \u00d7 probe design version",
         title = "Which contrasts this cohort can support, and what stops the rest",
         subtitle = paste0(
           if (!is.null(cohort_label)) paste0(cohort_label, " \u00b7 ") else "",
           sprintf("%d candidate contrasts \u00b7 circle = V measured, square = single-design contrast (no association to measure) \u00b7 filled = one library per patient, open = all libraries",
                   dplyr::n_distinct(d$id)),
           sprintf("\n%s",
             paste(vapply(names(WADE_VERDICT_PAL), function(v)
               sprintf("%s %d", v, sum(idx$verdict == v)), character(1)),
               collapse = "  \u00b7  ")))) +
    wade_theme() +
    theme(legend.position = "right")
}

# ---------------------------------------------------------------------
#' FIGURE: subset-detection volcano, bulk effect vs subset effect.
#'
#' v7's fig-wade-volcano, with two changes. (1) The axis limits are computed
#' from the plotted data rather than left to whole decades. (2) Nomination
#' classing respects `tail_ok`: when the tail window is 1-2 order statistics the
#' subset axis is not a nomination axis, so the classes collapse to
#' nominated/not and the subtitle says why. Drawing a "subset-concentrated"
#' class off a 2-point tail would be the figure asserting something the sample
#' size cannot support.
#'
#' Every gene sits above y = x because the tail is the high end of the
#' distribution; the distance above the diagonal IS the subset signal, which is
#' why the diagonal is drawn as the reference rather than a fitted line.
disc_fig_wade_volcano <- function(res, top_n = 100L, tail_ok = TRUE,
                                  label_n = 16L, extra_labels = character(),
                                  tail_conc_cut = 0.6) {
  # When the tail axis is not a nomination axis, the subset/bulk SPLIT is not
  # supportable either -- it is a threshold on tail.conc, which is computed from
  # the same too-small tail window. So the classes collapse to a single
  # "nominated" class rather than labelling every gene "broad / bulk", which
  # would assert that none of them is subset-driven when the truth is that this
  # contrast cannot tell.
  d <- cfrna_wade_nominate(res, top_n = top_n, tail_ok = tail_ok) |>
    dplyr::filter(diff.mean > 0, tail.mean > 0) |>
    dplyr::mutate(dm1 = diff.mean + 1, tm1 = tail.mean + 1,
                  cls = dplyr::case_when(
                    !nominated ~ "not nominated",
                    !tail_ok ~ "nominated (bulk axis)",
                    tail.conc >= tail_conc_cut ~ "subset-concentrated",
                    TRUE ~ "broad / bulk"))
  labg <- d |> dplyr::filter(nominated) |> dplyr::arrange(rank) |>
    dplyr::slice(seq_len(min(label_n, dplyr::n()))) |> dplyr::pull(gene)
  labg <- unique(c(labg, intersect(extra_labels, d$gene)))
  d$lab <- ifelse(d$gene %in% labg, d$gene, NA_character_)

  xl <- wade_log_limits(d$dm1); yl <- wade_log_limits(d$tm1)
  xb <- wade_log_breaks(xl);    yb <- wade_log_breaks(yl)
  pal <- WADE_NOM_PAL[names(WADE_NOM_PAL) %in% unique(d$cls)]

  ggplot(d, aes(dm1, tm1)) +
    geom_abline(slope = 1, intercept = 0, colour = "grey70",
                linetype = "2222", linewidth = 0.4) +
    geom_point(data = dplyr::filter(d, cls == "not nominated"),
               colour = WADE_NOM_PAL[["not nominated"]], size = 0.5, alpha = 0.3) +
    geom_point(data = dplyr::filter(d, cls != "not nominated"),
               aes(colour = cls), size = 1.8, alpha = 0.9) +
    ggrepel::geom_text_repel(aes(label = lab), size = 2.6, colour = "grey15",
                             fontface = "italic", max.overlaps = 30,
                             segment.size = 0.25, segment.colour = "grey55",
                             box.padding = 0.3, min.segment.length = 0, seed = 1) +
    scale_x_log10(limits = xl, breaks = xb, labels = wade_log_labels(xb)) +
    scale_y_log10(limits = yl, breaks = yb, labels = wade_log_labels(yb)) +
    scale_colour_manual(values = pal, name = NULL, breaks = names(pal)) +
    guides(colour = guide_legend(override.aes = list(size = 2.4, alpha = 1))) +
    labs(x = "bulk effect: diff.mean + 1  (case \u2212 control signed area, TPM, log)",
         y = "subset effect: tail.mean + 1  (mean difference in top-10% quantiles, TPM, log)",
         title = attr(res, "contrast_label") %||% "WADE subset-detection volcano",
         subtitle = sprintf(
           "%s \u00b7 %d vs %d libraries \u00b7 %s genes \u00b7 %d nominated (top %d by %s)%s",
           attr(res, "cohort") %||% "", attr(res, "n_case"), attr(res, "n_ctrl"),
           format(nrow(res), big.mark = ","), sum(d$nominated), top_n,
           if (tail_ok) "either axis" else "bulk axis only",
           if (!tail_ok)
             "\ntail window too small to nominate on the subset axis \u2014 distance above the diagonal is descriptive"
           else "")) +
    wade_theme() +
    theme(legend.position = c(0.985, 0.02), legend.justification = c(1, 0),
          legend.background = element_rect(fill = "white", colour = NA))
}

# ---------------------------------------------------------------------
#' FIGURE: per-gene diagnostic -- quantile functions and cumulative area.
#'
#' v7's fig-wade-curves. Kept because it is the panel that makes the statistic
#' legible: the cumulative signed area's SHAPE distinguishes a broad shift
#' (steady rise) from a rare high subset (flat, then a climb inside the tail
#' window) from a single outlier (flat, then one spike).
#'
#' @param tpm normalized matrix WADE ran on, gene symbols as rownames.
#' @param cond 1 = case, 0 = control, columns aligned to `tpm`.
#' @param genes genes to draw, in order.
#' @param res the scored frame, for the per-gene annotation.
disc_fig_wade_curves <- function(tpm, cond, genes, res, tail_q = 0.10,
                                 titles = NULL) {
  miss <- setdiff(genes, rownames(tpm))
  if (length(miss))
    stop("disc_fig_wade_curves(): gene(s) not in the matrix: ",
         paste(miss, collapse = ", "))
  fmtk <- function(x) ifelse(abs(x) >= 1000,
                             paste0(round(x / 1000, 1), "k"), as.character(round(x)))
  curves <- dplyr::bind_rows(lapply(genes, function(g)
    dplyr::mutate(wade_gene(tpm[g, ], cond, tail_q = tail_q), gene = g)))
  st <- res |> dplyr::filter(gene %in% genes)
  xmin_tail <- 1 - tail_q

  base9 <- theme_minimal(base_size = 10) +
    theme(plot.title = element_text(size = 9.5),
          plot.subtitle = element_text(size = 7.6, colour = "grey35"),
          axis.title = element_text(size = 8.5), axis.text = element_text(size = 7.5),
          panel.grid.minor = element_blank(), legend.position = "none",
          plot.margin = margin(3, 6, 3, 3))

  mk_q <- function(g, first) {
    dd <- curves |> dplyr::filter(gene == g) |> dplyr::select(p, y1, y0) |>
      tidyr::pivot_longer(c(y1, y0), names_to = "grp", values_to = "q")
    s <- st |> dplyr::filter(gene == g)
    ttl <- if (!is.null(titles) && !is.na(titles[g])) titles[[g]] else g
    p <- ggplot(dd, aes(p, q, colour = grp)) +
      annotate("rect", xmin = xmin_tail, xmax = 1, ymin = -Inf, ymax = Inf,
               fill = "grey88", alpha = 0.55) +
      geom_step(linewidth = 0.75, direction = "hv") +
      scale_colour_manual(values = c(y1 = WADE_CASE_COL, y0 = WADE_CTRL_COL)) +
      scale_y_continuous(labels = fmtk) +
      labs(title = ttl,
           subtitle = sprintf("diff.mean=%s  tail.conc=%s  rank=%d",
                              fmtk(s$diff.mean),
                              # Guarded: an exploded ratio prints as "n/a" rather
                              # than a number the reader would try to interpret.
                              ifelse(is.na(wade_tail_conc_display(s$tail.conc)), "n/a",
                                     sprintf("%.2f", s$tail.conc)),
                              s$rank),
           x = NULL, y = if (first) "expression\nquantile (TPM)" else NULL) +
      base9 +
      theme(plot.title = element_text(face = "italic"))
    if (first) p <- p +
      annotate("text", x = 0.04, y = Inf, vjust = 1.6, hjust = 0, size = 2.7,
               colour = WADE_CASE_COL, label = "case") +
      annotate("text", x = 0.04, y = Inf, vjust = 3.1, hjust = 0, size = 2.7,
               colour = WADE_CTRL_COL, label = "control")
    p
  }
  mk_c <- function(g, first) {
    dd <- curves |> dplyr::filter(gene == g)
    ggplot(dd, aes(p, cum)) +
      annotate("rect", xmin = xmin_tail, xmax = 1, ymin = -Inf, ymax = Inf,
               fill = "grey88", alpha = 0.55) +
      geom_hline(yintercept = 0, colour = "grey70", linewidth = 0.3) +
      geom_line(linewidth = 0.85, colour = WADE_AREA_COL) +
      geom_point(data = dd[nrow(dd), ], size = 1.7, colour = WADE_AREA_COL) +
      scale_y_continuous(labels = fmtk) +
      labs(x = "quantile p",
           y = if (first) "cumulative\nsigned area" else NULL) + base9
  }
  r1 <- lapply(seq_along(genes), function(i) mk_q(genes[i], i == 1))
  r2 <- lapply(seq_along(genes), function(i) mk_c(genes[i], i == 1))
  patchwork::wrap_plots(c(r1, r2), ncol = length(genes), byrow = TRUE,
                        heights = c(1, 0.82)) +
    patchwork::plot_annotation(
      title = "WADE per-gene diagnostic: quantile functions and cumulative signed area",
      subtitle = sprintf(paste("Top: case vs control quantile functions.",
                               "Bottom: cumulative signed area (endpoint = diff.mean).",
                               "Grey band = top-%.0f%% tail window."), 100 * tail_q),
      theme = theme(plot.title = element_text(size = 11.5, face = "bold"),
                    plot.subtitle = element_text(size = 8.7, colour = "grey35")))
}

# ---------------------------------------------------------------------
#' FIGURE: cross-contrast nomination membership.
#'
#' Which contrasts nominated each gene. An upset-style dot matrix rather than a
#' Venn: there are more than three sets, and the question is per-gene
#' membership, which a Venn cannot show at this size.
#'
#' Genes are ordered by how many contrasts nominated them, then by their best
#' rank, so the top of the panel is the consensus set and the bottom is
#' contrast-specific. Only genes nominated by at least one contrast appear --
#' the ~2,100 unnominated genes are the reference class and would be 95% of the
#' rows.
disc_fig_wade_cross <- function(cross, screen = NULL, max_genes = 40L,
                                cohort_label = NULL) {
  nom_cols <- grep("\\.nom$", names(cross), value = TRUE)
  ids <- sub("\\.nom$", "", nom_cols)
  rank_cols <- paste0(ids, ".rank")

  d <- cross |> dplyr::filter(n_nominating > 0)
  best <- apply(as.matrix(d[, rank_cols, drop = FALSE]), 1, min, na.rm = TRUE)
  d$best_rank <- best
  d <- d |> dplyr::arrange(dplyr::desc(n_nominating), best_rank) |>
    dplyr::slice(seq_len(min(max_genes, dplyr::n())))

  long <- dplyr::bind_rows(lapply(seq_along(ids), function(i) {
    tibble::tibble(gene = d$gene, id = ids[i],
                   nominated = as.logical(d[[nom_cols[i]]]),
                   rank = d[[rank_cols[i]]],
                   n_nominating = d$n_nominating)
  })) |> dplyr::mutate(nominated = ifelse(is.na(nominated), FALSE, nominated))

  fam <- if (!is.null(screen))
    setNames(screen$family[match(ids, screen$id)], ids) else setNames(rep("detection", length(ids)), ids)
  long$family <- unname(fam[long$id])
  lab <- if (!is.null(screen))
    setNames(screen$label[match(ids, screen$id)], ids) else setNames(ids, ids)

  long$gene <- factor(long$gene, levels = rev(d$gene))
  long$id   <- factor(long$id, levels = ids)

  ggplot(long, aes(id, gene)) +
    geom_point(data = dplyr::filter(long, !nominated), colour = "grey88", size = 1.6) +
    geom_point(data = dplyr::filter(long, nominated), aes(colour = family), size = 2.9) +
    scale_x_discrete(labels = function(v) wade_wrap_labels(unname(lab[v]), 22),
                     position = "top") +
    scale_colour_manual(values = WADE_FAMILY_PAL, name = "contrast family") +
    labs(x = NULL, y = NULL,
         title = "Which contrasts nominate each gene",
         subtitle = sprintf(
           "%s%d genes nominated by \u22651 contrast, top %d shown \u00b7 ordered by number of nominating contrasts, then best rank \u00b7 grey = not nominated",
           if (!is.null(cohort_label)) paste0(cohort_label, " \u00b7 ") else "",
           sum(cross$n_nominating > 0), nrow(d))) +
    wade_theme() +
    theme(axis.text.y = element_text(size = 7.2, face = "italic"),
          axis.text.x = element_text(size = 7.4, angle = 0, hjust = 0.5),
          panel.grid.major.y = element_line(colour = "grey94", linewidth = 0.25),
          panel.grid.major.x = element_blank(),
          legend.position = "bottom")
}

#' Wrap long axis labels to a width, for the cross-contrast columns.
wade_wrap_labels <- function(x, width = 22) {
  vapply(x, function(s) paste(strwrap(s, width), collapse = "\n"), character(1))
}

`%||%` <- function(a, b) if (is.null(a)) b else a

# --------------------------------------------------------------------
# EVERY contrast gets a volcano, and every point is identifiable
# --------------------------------------------------------------------
# The section rendered ONE volcano for the primary contrast while 18 result
# tables sat in the cache. A reader with no per-contrast view has no way to ask
# why a gene was nominated in one comparison and not another, which is the first
# question a nomination list provokes.

#' Faceted volcano over every contrast in one cohort.
#'
#' One panel per contrast, shared axes so effect sizes are comparable BY EYE
#' across contrasts -- which is the point of a gallery rather than nine separate
#' figures with nine different scales. Panels are ordered by the screen verdict
#' so the runnable contrasts come first and the descriptive ones are visibly
#' set apart.
disc_fig_wade_gallery <- function(res, screen = NULL, top_n = 12L,
                                  label_n = 4L, p_col = "padj.diff") {
  nm <- names(res)
  d <- dplyr::bind_rows(lapply(nm, function(k) {
    t <- res[[k]]
    t <- if (is.data.frame(t)) t else t$tbl
    if (is.null(t) || !nrow(t)) return(NULL)
    t$contrast <- k
    t
  }))
  if (!nrow(d)) stop("no contrast in `res` carries a result table")

  # AXES. x is log2 fold change, NOT diff.mean: diff.mean is a difference of mean
  # CPM spanning +/-9,000, so plotting it collapses every point onto a vertical
  # line at zero (the first version of this figure did exactly that and was
  # unreadable). y is -log10 of the adjusted p-value, which is what makes this a
  # volcano rather than an effect-size strip.
  stopifnot("log2fc" %in% names(d))
  if (!p_col %in% names(d))
    stop("`", p_col, "` not in the result table; available: ",
         paste(grep("^p", names(d), value = TRUE), collapse = ", "))
  d$x <- d$log2fc
  d$y <- -log10(pmax(d[[p_col]], .Machine$double.xmin))
  d <- d[is.finite(d$x) & is.finite(d$y), ]

  # Rank within contrast, so "top" means top FOR THAT COMPARISON.
  d <- d |>
    dplyr::group_by(.data$contrast) |>
    dplyr::mutate(rk = dplyr::min_rank(dplyr::desc(abs(.data$score)))) |>
    dplyr::ungroup()
  d$is_top <- d$rk <= top_n
  d$lab <- ifelse(d$rk <= label_n, d$gene, NA_character_)

  if (!is.null(screen)) {
    key <- if ("id" %in% names(screen)) screen$id else screen$contrast
    stopifnot(!is.null(key))
    v <- screen$verdict[match(nm, key)]
    v[is.na(v)] <- "not screened"
    ord <- nm[order(factor(v, levels = c("run", "caution", "descriptive",
                                         "refuse", "not screened")), nm)]
    lv <- sprintf("%s [%s]", ord, v[match(ord, nm)])
    d$contrast <- factor(sprintf("%s [%s]", d$contrast,
                                 v[match(d$contrast, nm)]), levels = lv)
  } else {
    d$contrast <- factor(d$contrast, levels = nm)
  }

  # Symmetric x limits from a robust quantile: a handful of extreme fold changes
  # on near-zero denominators would otherwise squeeze the bulk into the centre.
  xq <- stats::quantile(abs(d$x), 0.995, na.rm = TRUE)
  xl <- c(-xq, xq) * 1.05

  ggplot2::ggplot(d, ggplot2::aes(.data$x, .data$y)) +
    ggplot2::geom_point(ggplot2::aes(colour = .data$is_top),
                        size = 0.8, alpha = 0.7) +
    ggrepel::geom_text_repel(ggplot2::aes(label = .data$lab), size = 2.4,
                             min.segment.length = 0, segment.size = 0.22,
                             max.overlaps = 25, na.rm = TRUE) +
    ggplot2::geom_vline(xintercept = 0, linewidth = 0.3, colour = "#2b3138") +
    ggplot2::geom_hline(yintercept = -log10(0.05), linetype = "22",
                        linewidth = 0.35, colour = "#8a949e") +
    ggplot2::facet_wrap(~contrast, ncol = 3) +
    ggplot2::scale_colour_manual(values = c(`TRUE` = "#c1443f",
                                            `FALSE` = "#aab3bb"),
                                 guide = "none") +
    ggplot2::coord_cartesian(xlim = xl) +
    ggplot2::labs(x = expression(log[2]~fold~change*","~case/control),
                  y = expression(-log[10]~adjusted~italic(p))) +
    wade_theme(10)
}

#' Per-contrast nomination overlap: which genes recur, and where.
#'
#' A gene nominated by one contrast and not another is the interesting case; a
#' gene nominated everywhere is usually a burden proxy. This is the table that
#' distinguishes them.
disc_tbl_wade_recurrence <- function(res, top_n = 25L) {
  nm <- names(res)
  hits <- lapply(nm, function(k) {
    t <- res[[k]]; t <- if (is.data.frame(t)) t else t$tbl
    if (is.null(t) || !nrow(t)) return(character())
    t$gene[order(abs(t$score), decreasing = TRUE)][seq_len(min(top_n, nrow(t)))]
  })
  names(hits) <- nm
  all_g <- sort(unique(unlist(hits)))
  if (!length(all_g))
    return(tibble::tibble(gene = character(), n_contrasts = integer(),
                          contrasts = character()))
  # Build the membership matrix explicitly. vapply + apply on a single-column
  # result drops to a vector and the tibble loses `gene` -- which is how this
  # failed the first time.
  m <- matrix(FALSE, nrow = length(all_g), ncol = length(nm),
              dimnames = list(all_g, nm))
  for (k in seq_along(nm)) m[, k] <- all_g %in% hits[[k]]
  out <- tibble::tibble(
    gene = all_g,
    n_contrasts = as.integer(rowSums(m)),
    contrasts = vapply(seq_len(nrow(m)),
                       function(i) paste(nm[m[i, ]], collapse = ", "),
                       character(1)))
  out[order(-out$n_contrasts, out$gene), ]
}
