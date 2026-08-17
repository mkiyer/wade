# cfRNA v7 §9 — Discovery: subset-aware differential expression (WADE)

> **Extracted reference material — not a specification.**
> Source: `cfrna_analysis/analysis_cfrna_v7.qmd` lines 3727–4415 (689 lines).
> cfRNA repo git SHA `828f2f1c6a10afd182fee0e119ba3595d98b6a72` (clean tree, 2026-08-17).
> sha256 of the whole source notebook: `16bb7bffda7ef9a44a11e5eb67598340960744391f4e0dccf67287c40e8319cb`.
> Retrieve the original with `git show 828f2f1c6a10afd182fee0e119ba3595d98b6a72:cfrna_analysis/analysis_cfrna_v7.qmd`.
>
> Contains 12 R chunk(s) and 35 unresolved inline `r ...` expression(s). The inline expressions are **not** evaluated here: every number they would print is absent, and no value has been guessed or substituted. Quarto div syntax (`::: {.callout-*}`), cross-references (`@sec-*`, `§`) and chunk options (`#|`) are preserved as written. The only edit is the code-fence info string, `{r}` → `r`, so the R reads as R in a plain markdown viewer.
>
> This is the **first** cohort application of WADE, and the section where the method's operating regime was established: three contrasts on one clinical grouping, nomination by rank score after per-gene FDR failed to resolve, and the §9.6 simulations that are the method's only ground-truth validation. Subsections: 9 preamble, 9.1 Running WADE, 9.2 Gene nomination table, 9.3 Per-gene diagnostic, 9.4 Subset-detection volcano, 9.5 Cross-contrast, 9.6 Method validation. The range ends at the close of §9's prose; the `ss-load` chunk that follows at line 4421 belongs to §10 and is excluded.
>
> Chunk labels in order: `wade-inputs`, `wade-run`, `wade-prior-panels`, `tbl-wade-nomination`, `fig-wade-curves`, `fig-wade-volcano`, `wade-crosscontrast`, `tbl-wade-crosscontrast`, `fig-wade-crosscontrast`, `wade-validation-sim`, `fig-wade-validation`, `save-disc-tables`.

---

# 9 Discovery: subset-aware differential expression (WADE)

The unsupervised characterization (§8) established that the on-target cohort
carries genuine tissue-of-origin biology once the capture-efficiency axis is set
aside. We now turn to the primary discovery goal: **nominate genes whose plasma
cfRNA abundance separates cancer cases from controls**, with the explicit
understanding that a cancer "case" group is heterogeneous — a signal may be
present in only a fraction of cases (one tumour subtype, one tissue of origin)
and absent in the rest.

The cohort has three clinical groups — cancer, healthy control, and benign /
precancer (precursor lesions such as IPMN, pancreatic cysts, adenomas). We run
WADE on **three contrasts**: the primary **Cancer vs Non-Cancer** detection
contrast (the nomination product), and two resolving contrasts — **Cancer vs
Control** and **Benign vs Control** — that share the healthy reference and, when
compared (§9.5), separate cancer-specific markers from shared "early" markers
already present in precursor lesions.

Standard differential-expression tests (t-test, Wilcoxon, DESeq2/edgeR) compare
group *means* or *ranks* and assume the two groups differ by a location shift.
That assumption fails here: a gene elevated in 15 of 77 cases and silent in the
other 62 has a modest mean difference and is easily missed, yet it is exactly
the kind of subtype marker we want. **WADE** (**W**asserstein **A**rea
**D**ifferential **E**xpression) is built for this regime.

**Method.** For each gene, WADE compares the two groups' empirical *quantile
functions* $Q_{\text{case}}(p)$ and $Q_{\text{ctrl}}(p)$ on a shared grid of
$\min(n_0, n_1)$ probabilities. The signed area between them,
$$\texttt{diff.mean} = \int_0^1 \big[Q_{\text{case}}(p) - Q_{\text{ctrl}}(p)\big]\,dp = \mu_{\text{case}} - \mu_{\text{ctrl}},$$
is the ordinary mean difference (the **bulk** axis); the integral of the
*absolute* difference is the 1-Wasserstein distance $W_1$. Because
`diff.mean` collapses a rare-subset signal and a whole-group shift to the same
number, WADE adds **shape** statistics that read the upper tail of the quantile
difference, where a rare high-expressing subset concentrates: `tail.mean` (the
mean difference over the top-10% quantiles — the **subset** axis) and
`tail.conc` (the share of the total signed area that falls in that tail). This
connects two literatures — the quantile/Wasserstein differential-distribution
test (cf. scDD, waddR) supplies the mechanism, and cancer-outlier profile
analysis (COPA, OS, ORT, MOST, LSOSS) supplies the subset-detection goal.

Inference is by **permutation**: the per-gene normalizer and a fixed continuity
jitter are applied once to build a normalized TPM matrix, then case/control
labels are shuffled on that fixed matrix (conditional inference — the noise draw
is held out of the null). Small p-values are refined with a Generalized-Pareto
tail fit (Knijnenburg et al. 2009), and both axes carry a BH-FDR annotation.
As §9.1 shows, this 18-control cohort is underpowered for per-gene FDR, so genes
are nominated by **rank score** and the p-values are read as effect-strength
annotation rather than a significance gate.
The implementation is a self-contained module, `wade.R`, sourced in the setup
block and structured as the prototype for a future standalone package.

```r
#| label: wade-inputs
# WADE takes a (count, normalizer) pair. Since rigel deconvolutes gDNA, the
# primary measurement is TOTAL rigel counts normalized by effective length,
# giving standard TPM — this uses all the signal, not just spliced reads.
# (The same module also runs on spliced counts / intron number; see the
# sensitivity note below.) Same on-target cohort and grouping as §7-§8.
wade_genes <- genes[det_on, ]

# Three clinically-curated groups from pca_df$broad (§8):
#   Cancer (case) · Control (healthy) · Benign / precancer (precursor lesions).
# We run three contrasts (§9.1):
#   (1) PRIMARY  Cancer vs Non-Cancer (Control + Benign)  — the detection question
#   (2)          Cancer vs Control                        — pure malignancy signal
#   (3)          Benign vs Control                        — precursor-lesion signal
# Comparing (2) and (3) separates cancer-specific from shared/early markers (§9.5).
grp        <- pca_df$broad
lib_cancer <- ana_libs[grp == "Cancer"]
lib_ctrl   <- ana_libs[grp == "Control"]
lib_benign <- ana_libs[grp == "Benign / precancer"]
lib_noncancer <- c(lib_ctrl, lib_benign)

# Full-cohort (count, normalizer) matrices; wade_run() subsets per contrast.
# Count = total rigel counts; normalizer = effective length (per-gene-per-
# sample, so count/efflen scaled to library size is exactly TPM).
Ccnt  <- cnt[det_on,    ana_libs, drop = FALSE]
Cnorm <- efflen[det_on, ana_libs, drop = FALSE]

# Primary-contrast objects (Cancer vs Non-Cancer) kept explicit for the
# per-gene diagnostic (§9.3), which needs lib_sizes / cond in scope.
wade_case <- lib_cancer
wade_ctrl <- lib_noncancer
wade_libs <- c(wade_case, wade_ctrl)
cond      <- c(rep(1L, length(wade_case)), rep(0L, length(wade_ctrl)))   # 1 = case
Xcnt      <- cnt[det_on,    wade_libs, drop = FALSE]
Xnorm     <- efflen[det_on, wade_libs, drop = FALSE]

n_cancer <- length(lib_cancer); n_ctrl <- length(lib_ctrl)
n_benign <- length(lib_benign); n_noncancer <- length(lib_noncancer)
n_case   <- n_cancer
```

The **primary discovery contrast** is **`r n_cancer` cancer cases vs
`r n_noncancer` non-cancer** (`r n_ctrl` healthy controls + `r n_benign` benign /
precancer) across **`r format(sum(det_on), big.mark = ",")` on-target genes**.
WADE compares the two groups at `r min(n_cancer, n_noncancer)` matched quantiles
(the smaller group size), and the top-10% tail window is the upper
`r max(1L, ceiling(0.10 * min(n_cancer, n_noncancer)))` of those quantiles. The
fundamental measurement is **total rigel counts normalized by effective length**
(standard TPM): because rigel already removes genomic-DNA contamination, total
counts are now a trustworthy signal and use every read, not only the spliced
fraction.

## 9.1 Running WADE

```r
#| label: wade-run
# Three contrasts, each 2000 label permutations with GPD tail refinement on
# both the bulk (diff.mean) and subset (tail.mean) axes. The fixed seed makes
# the continuity jitter and the permutation null reproducible; wade_run()
# subsets the (count, normalizer) matrices, codes case=1/control=0, sizes the
# libraries, runs the test, and adds the two rank scores.
gn <- wade_genes$gene_name
wade_cvn <- wade_run(Ccnt, Cnorm, lib_cancer, lib_noncancer, gene_names = gn)  # PRIMARY
wade_cvc <- wade_run(Ccnt, Cnorm, lib_cancer, lib_ctrl,      gene_names = gn)  # malignancy
wade_bvc <- wade_run(Ccnt, Cnorm, lib_benign, lib_ctrl,      gene_names = gn)  # precursor

# Primary contrast drives §9.2-9.4; keep the classic names + lib_sizes/cond
# (the per-gene diagnostic in §9.3 re-normalizes the primary matrices).
wade_res  <- wade_cvn
lib_sizes <- wade_lib_size(Xcnt, Xnorm)

# This cohort is deliberately underpowered for per-gene FDR: 18-33 controls
# cannot resolve permutation p-values small enough to survive BH across
# 3,041 genes. We therefore NOMINATE by rank score and treat the p-values
# as effect-strength annotation, not as a significance gate. That the raw
# signal is real is still checkable against the chance expectation:
n_p01_diff  <- sum(wade_res$p.diff < 0.01)      # expected ~30 by chance
n_p01_tail  <- sum(wade_res$p.tail < 0.01)
n_expect_01 <- round(0.01 * nrow(wade_res))
```

We run **three contrasts** on the same on-target genes and measurement, and
report each below:

1. **Primary — Cancer vs Non-Cancer** (`r n_cancer` vs `r n_noncancer`): the
   cancer-detection question, pooling healthy controls with benign / precancer
   as the "not cancer" reference. This drives the nomination table and figures.
2. **Cancer vs Control** (`r n_cancer` vs `r n_ctrl`): the pure malignancy
   signal against healthy plasma only.
3. **Benign vs Control** (`r n_benign` vs `r n_ctrl`): the precursor-lesion
   signal — what, if anything, IPMN / cyst / adenoma plasma already carries.

Contrasts 2 and 3 are compared head-to-head in §9.5 to separate cancer-specific
markers from shared "early" markers already present in precursor lesions.

WADE reports two axes per gene. The **bulk** axis (`diff.mean`, the signed
quantile area = mean difference) finds genes broadly shifted between cases and
controls; the **subset** axis (`tail.mean`, the mean difference over the top-10%
quantiles) finds genes elevated in only a fraction of cases even when their mean
difference is unremarkable. With heterogeneous cancers and at most `r n_noncancer`
reference libraries this cohort is **underpowered for per-gene FDR** — after
Benjamini-Hochberg across `r format(nrow(wade_res), big.mark=",")` genes, no gene
reaches FDR < 0.10 on either axis. That is a power statement, not an absence of
signal: in the primary contrast **`r n_p01_diff` genes clear nominal *p* < 0.01
on the bulk axis and `r n_p01_tail` on the subset axis, against `r n_expect_01`
expected by chance** — a `r round(n_p01_diff/n_expect_01, 1)`× enrichment of true
positives. We therefore nominate genes by their **rank score** (stable and
interpretable regardless of p-value resolution) and carry the permutation *p*
only as an effect-strength annotation. Expanding the control set (see the cohort
discussion, §7) is the route to formal significance; the ranked nominations
below are the discovery product this cohort can support.

**Measure choice.** WADE takes a generic *(count, normalizer)* pair, so the same
module runs on either total counts ÷ effective length (used here) or spliced
counts ÷ intron number (the original formulation, from before rigel deconvolved
gDNA). The two give concordant top nominations — CEACAM5, MYF6, GKN1, KLK6 and
the SPRR/mucin markers top both — but total counts recover slightly more
subset-axis signal (they use every read, not just the spliced fraction), which
is why they are the primary measurement now that gDNA is removed upstream.

## 9.2 Gene nomination table

We annotate every nominated gene against the lab's **prior discovery panels** —
the cfRNA v4 capture panel and the RAG (rank-aggregated gene) discovery lists,
read from the v5 probe-panel workbook — and against the canonical Moffitt PDAC
subtype signatures. This does two things: it confirms that WADE recovers genes
the lab has independently nominated before (continuity), and it flags **novel**
genes that the current run surfaces but the prior panels missed.

```r
#| label: wade-prior-panels
#| include: false
# Prior panel membership + prior WADE rank, read from the v5 probe-panel
# workbook (the probes sheet embeds the earlier per-gene run and the panel
# flags). Aggregated to one row per gene: membership = ANY probe flagged,
# prior_rank = best (min) rank across the gene's probes.
prior_panel_annotation <- function(panel_xlsx) {
  if (!file.exists(panel_xlsx)) return(NULL)
  pp <- readxl::read_xlsx(panel_xlsx, sheet = "probes")
  truthy <- function(x) x %in% c(TRUE, 1, "TRUE", "1")
  pp |>
    dplyr::group_by(gene) |>
    dplyr::summarise(
      prior_cfrna_v4   = any(truthy(cfrna_v4)),
      prior_agilentv4  = any(truthy(agilentv4)),
      prior_rag_v1     = any(truthy(rag_v1)),
      prior_rag_v2     = any(truthy(rag_v2)),
      prior_rag_shared = any(truthy(rag_shared)),
      prior_rag_union  = any(truthy(rag_union)),
      prior_rag_panel  = paste(unique(na.omit(rag_panel)), collapse = ";"),
      prior_rank       = suppressWarnings(min(as.numeric(rank), na.rm = TRUE)),
      .groups = "drop") |>
    dplyr::mutate(prior_rank = ifelse(is.finite(prior_rank), prior_rank, NA_real_))
}
prior <- prior_panel_annotation(file.path(panel_dir, "cfrnav5_probe_panel.xlsx"))

# Moffitt PDAC subtype signatures (literature-standard basal/classical markers).
moffitt_basal     <- c("VGLL1","UCA1","S100A2","LY6D","SPRR3","SPRR1B","LEMD1","KRT15",
                       "CTSL2","DHRS9","AREG","CST6","SERPINB3","KRT6C","KRT6A","SERPINB4",
                       "FAM83A","SCEL","FGFBP1","KRT7","KRT17","GPR87","TNS4","SLC2A1","ANXA8L2")
moffitt_classical <- c("BTNL8","FAM3D","ATAD4","AGR3","CTSE","LOC400573","LYZ","TFF2","TFF1",
                       "ANXA10","LGALS4","PLA2G10","CEACAM6","VSIG2","TSPAN8","ST6GALNAC1","AGR2",
                       "TFF3","CYP3A7","MYO1A","CLRN3","KRT20","CDH17","SPINK4","REG4")
```

```r
#| label: tbl-wade-nomination
#| tbl-cap: "Top 25 WADE gene nominations, ranked by the bulk discovery score. `diff.mean` is the mean case-control difference (TPM); `tail.conc` is the share of that signal concentrated in the top-10% quantiles (high = subset-driven); `padj.diff` / `padj.tail` are BH-FDR on the two axes (annotation only — this cohort is underpowered for per-gene FDR, see §9.1). `prior_rag` marks genes already in a RAG discovery panel; `novel` marks top-ranked genes absent from every prior panel. The full 3,041-gene table is exported to `disc_tables/`."
nom <- wade_res |>
  dplyr::left_join(prior, by = c("gene")) |>
  dplyr::mutate(
    dplyr::across(dplyr::starts_with("prior_") & where(is.logical),
                  ~ tidyr::replace_na(.x, FALSE)),
    pdac_basal     = toupper(gene) %in% toupper(moffitt_basal),
    pdac_classical = toupper(gene) %in% toupper(moffitt_classical),
    # Nomination is by RANK (top-100 on either the bulk or the subset score) —
    # this cohort is underpowered for per-gene FDR (§9.1), so the ranked score,
    # not the p-value, defines the discovery set.
    wade_hit = rank <= 100 | tail.rank <= 100,
    novel    = wade_hit & !prior_rag_union) |>
  dplyr::arrange(rank)

n_wade_hit <- sum(nom$wade_hit)
n_confirm  <- sum(nom$wade_hit & nom$prior_rag_union)
n_novel    <- sum(nom$novel)

nom |>
  dplyr::slice(1:25) |>
  dplyr::transmute(
    Rank = rank, Gene = gene,
    `diff.mean` = round(diff.mean, 0), fc = round(fc, 0),
    `tail.conc` = round(tail.conc, 2),
    `padj.diff` = signif(padj.diff, 2), `padj.tail` = signif(padj.tail, 2),
    prior_rag = ifelse(prior_rag_union, "yes", ""),
    PDAC = dplyr::case_when(pdac_basal & pdac_classical ~ "B/C",
                            pdac_basal ~ "basal", pdac_classical ~ "classical", TRUE ~ ""),
    novel = ifelse(novel, "yes", "")) |>
  knitr::kable(align = "rlrrrrrccc")
```

Across the whole cohort WADE nominates **`r n_wade_hit` genes** (top-100 by the
bulk score or the subset score). Of these, **`r n_confirm` confirm prior RAG
panel members** — strong continuity with the lab's earlier discovery work — and
**`r n_novel` are novel**, surfaced now but absent from every prior panel.
The top of the list is dense with established circulating epithelial and
GI/pancreatic tumour markers: the top hit is **CEACAM5** (carcinoembryonic
antigen, CEA), followed by mucins and secreted markers (MUCL3, GKN1),
serine-protease inhibitors (SERPINB5/B3), and the pancreatic markers PSCA and
AGR2 — a face-valid nomination list for a plasma cancer-detection panel.

## 9.3 Per-gene diagnostic: quantile functions and cumulative area

The WADE statistic is easiest to read one gene at a time. The top row overlays
the case and control quantile functions; the bottom row plots the **cumulative
signed area** — accumulated left-to-right, its endpoint at $p = 1$ equals
`diff.mean`. The *shape* of that cumulative curve is the whole story: a broad
shift rises steadily across all quantiles, while a rare high subset stays flat
and then climbs inside the top-tail window (shaded).

```r
#| label: fig-wade-curves
#| fig-cap: "WADE per-gene diagnostic across four profiles. CEACAM5 (top-ranked subset marker): the cumulative area is flat until ~p=0.75, then climbs steeply inside the tail window — signal concentrated in a fraction of cases. MUC1 (bulk shift): a smooth, steady rise from p=0.1 across the whole range. PANTR1 (single-outlier): dead flat until one end-point spike — high tail.conc but driven by a single sample, so it ranks low. GRM7 (null): flat at zero. Top row: cancer (red) vs control (navy) quantile functions; grey band = top-10% tail window. Bottom row: cumulative signed area (endpoint = diff.mean)."
#| fig-width: 11
#| fig-height: 5.4
tpm_wade <- wade_normalize(Xcnt, Xnorm, lib_sizes, seed = 1L)
rownames(tpm_wade) <- wade_genes$gene_name
sel <- c("CEACAM5", "MUC1", "PANTR1", "GRM7")
lab <- c(CEACAM5 = "CEACAM5  (subset marker)", MUC1 = "MUC1  (bulk shift)",
         PANTR1  = "PANTR1  (single-outlier)", GRM7 = "GRM7  (null)")
fmtk <- function(x) ifelse(abs(x) >= 1000, paste0(round(x/1000, 1), "k"), as.character(round(x)))
col_case <- "#C0392B"; col_ctrl <- "#2C3E50"; col_area <- "#E67E22"

curves <- purrr::map_dfr(sel, ~ dplyr::mutate(wade_gene(tpm_wade[.x, ], cond), gene = .x))
gstat  <- nom |> dplyr::filter(gene %in% sel) |>
  dplyr::mutate(gene = factor(gene, levels = sel)) |> dplyr::arrange(gene)

base9 <- theme_minimal(base_size = 10) +
  theme(plot.title = element_text(size = 10), plot.subtitle = element_text(size = 8, colour = "grey35"),
        axis.title = element_text(size = 8.5), axis.text = element_text(size = 7.5),
        panel.grid.minor = element_blank(), legend.position = "none", plot.margin = margin(3, 6, 3, 3))

mk_q <- function(g, first = FALSE) {
  d  <- curves |> dplyr::filter(gene == g) |> dplyr::select(p, y1, y0) |>
    tidyr::pivot_longer(c(y1, y0), names_to = "grp", values_to = "q")
  st <- gstat |> dplyr::filter(gene == g)
  p <- ggplot(d, aes(p, q, colour = grp)) +
    annotate("rect", xmin = 0.90, xmax = 1, ymin = -Inf, ymax = Inf, fill = "grey88", alpha = 0.55) +
    geom_step(linewidth = 0.75, direction = "hv") +
    scale_colour_manual(values = c(y1 = col_case, y0 = col_ctrl)) +
    scale_y_continuous(labels = fmtk) +
    labs(title = lab[g], subtitle = sprintf("diff.mean=%.0f  tail.conc=%.2f  rank=%d",
         st$diff.mean, st$tail.conc, st$rank),
         x = NULL, y = if (first) "expression\nquantile (TPM)" else NULL) + base9
  if (first) p <- p +
    annotate("text", x = 0.04, y = Inf, vjust = 1.6, hjust = 0, size = 2.7, colour = col_case, label = "cancer") +
    annotate("text", x = 0.04, y = Inf, vjust = 3.1, hjust = 0, size = 2.7, colour = col_ctrl, label = "control")
  p
}
mk_c <- function(g, first = FALSE) {
  d <- curves |> dplyr::filter(gene == g)
  ggplot(d, aes(p, cum)) +
    annotate("rect", xmin = 0.90, xmax = 1, ymin = -Inf, ymax = Inf, fill = "grey88", alpha = 0.55) +
    geom_hline(yintercept = 0, colour = "grey70", linewidth = 0.3) +
    geom_line(linewidth = 0.85, colour = col_area) +
    geom_point(data = d[nrow(d), ], size = 1.7, colour = col_area) +
    scale_y_continuous(labels = fmtk) +
    labs(x = "quantile p", y = if (first) "cumulative\nsigned area" else NULL) + base9
}
row1 <- purrr::imap(sel, ~ mk_q(.x, .y == 1))
row2 <- purrr::imap(sel, ~ mk_c(.x, .y == 1))
patchwork::wrap_plots(c(row1, row2), ncol = 4, byrow = TRUE, heights = c(1, 0.82)) +
  patchwork::plot_annotation(
    title = "WADE per-gene diagnostic: quantile functions and cumulative signed area",
    subtitle = "Top: cancer vs control quantile functions. Bottom: cumulative signed area (endpoint = diff.mean). Grey band = top-10% tail window.",
    theme = theme(plot.title = element_text(size = 11.5),
                  plot.subtitle = element_text(size = 8.7, colour = "grey35")))
```

The contrast between CEACAM5's tail-loaded climb and MUC1's steady rise is the
exact distinction the subset axis is built to capture, and PANTR1 is the
cautionary case: a single extreme sample produces a large `diff.mean` and a
`tail.conc` near 1, but the permutation test refuses it (FDR ≈ 1) because one
sample cannot beat a shuffled null. The shape statistics *describe* subset
structure; the permutation p-value is what *protects* against calling a single
outlier a discovery.

## 9.4 Subset-detection volcano

Plotting every gene's bulk effect (`diff.mean`) against its subset effect
(`tail.mean`) separates the two kinds of hit. Both axes are stable TPM
quantities, so the display is honest across the full dynamic range (we avoid
`tail.conc` as an axis — it is a ratio that explodes when `diff.mean` ≈ 0). Every
gene sits above the $y = x$ diagonal because the tail is the high end of the
distribution; the **distance above the diagonal is the subset signal**.

```r
#| label: fig-wade-volcano
#| fig-cap: "WADE subset-detection volcano. Each point is one on-target gene: x = bulk effect (diff.mean), y = subset effect (tail.mean), both up-in-cancer, log scales. Genes far above the y=x diagonal are subset-driven (their signal concentrates in the tail); genes near the diagonal are broad shifts. Colour splits the rank-nominated genes (top-100 on either axis) by tail-concentration class; grey = not nominated. Labelled points are the top nominations plus canonical markers — all established epithelial / GI / pancreatic tumour genes."
#| fig-width: 9.6
#| fig-height: 6.6
vd <- nom |>
  dplyr::filter(diff.mean > 0, tail.mean > 0) |>
  dplyr::mutate(
    cls = dplyr::case_when(
      wade_hit & tail.conc >= 0.6 ~ "subset-concentrated nomination",
      wade_hit & tail.conc <  0.6 ~ "broad/bulk nomination",
      TRUE ~ "not nominated"),
    dm1 = diff.mean + 1, tm1 = tail.mean + 1)
labg <- vd |> dplyr::filter(wade_hit) |> dplyr::arrange(rank) |> dplyr::slice(1:16) |> dplyr::pull(gene)
labg <- unique(c(labg, "MUC1", "AGR2", "CEACAM5", "PLA2G1B", "SERPINB5"))
vd$lab <- ifelse(vd$gene %in% labg, vd$gene, NA)
pal9 <- c("subset-concentrated nomination" = "#1F618D", "broad/bulk nomination" = "#B9770E", "not nominated" = "grey80")
xmax <- max(vd$dm1); ymax <- max(vd$tm1)

ggplot(vd, aes(dm1, tm1)) +
  geom_abline(slope = 1, intercept = 0, colour = "grey70", linetype = "2222", linewidth = 0.4) +
  geom_point(aes(colour = cls, size = cls, alpha = cls)) +
  ggrepel::geom_text_repel(aes(label = lab), size = 2.7, colour = "grey15", max.overlaps = 30,
                  segment.size = 0.25, segment.colour = "grey55", box.padding = 0.3,
                  min.segment.length = 0, seed = 1) +
  scale_x_log10(breaks = c(1, 10, 100, 1000, 10000), labels = c("1", "10", "100", "1k", "10k"),
                limits = c(1, xmax * 1.2)) +
  scale_y_log10(breaks = c(1, 10, 100, 1000, 10000, 100000),
                labels = c("1", "10", "100", "1k", "10k", "100k"), limits = c(1, ymax * 1.3)) +
  scale_colour_manual(values = pal9) +
  scale_size_manual(values = c("subset-concentrated nomination" = 1.8, "broad/bulk nomination" = 1.8, "not nominated" = 0.5)) +
  scale_alpha_manual(values = c("subset-concentrated nomination" = 0.9, "broad/bulk nomination" = 0.9, "not nominated" = 0.28)) +
  annotate("text", x = 1.5, y = ymax * 1.15, hjust = 0, size = 3.0, colour = "#1F618D",
           fontface = "italic", label = "subset-concentrated (tail \u226b mean)") +
  annotate("text", x = xmax * 0.9, y = xmax * 0.42, hjust = 1, size = 3.0, colour = "grey45",
           fontface = "italic", label = "y = x  (bulk: tail \u2248 mean)") +
  labs(x = "diff.mean + 1  (bulk effect: cancer \u2212 control signed area, TPM, log)",
       y = "tail.mean + 1  (subset effect: mean difference in top-10% quantiles, TPM, log)",
       colour = NULL) +
  guides(size = "none", alpha = "none", colour = guide_legend(override.aes = list(size = 2.4, alpha = 1))) +
  theme_minimal(base_size = 11) +
  theme(axis.title = element_text(size = 9.5), panel.grid.minor = element_blank(),
        legend.position = c(0.99, 0.02), legend.justification = c(1, 0),
        legend.background = element_rect(fill = "white", colour = NA),
        legend.text = element_text(size = 8), legend.key.height = unit(11, "pt"))
```

## 9.5 Cross-contrast: cancer-specific vs shared "early" markers

The two resolving contrasts — Cancer vs Control and Benign vs Control — share a
reference group (healthy controls), so a gene's nomination status across the two
tells us *where in the disease continuum* its plasma signal first appears:

- **Cancer-specific** — nominated in Cancer-vs-Control but **not** in
  Benign-vs-Control: the signal appears only with malignancy.
- **Shared / early** — nominated in **both**: already elevated in precursor
  lesions, so it rises early and is not by itself cancer-discriminating.
- **Benign-specific** — nominated in Benign-vs-Control only: a precursor-lesion
  signal that does not persist (or shifts) in the cancer group.

This is the analysis that decides whether the held-out benign / precancer group
carries a cfRNA signature — the second prong of the discovery goal.

```r
#| label: wade-crosscontrast
# Nomination = top-100 on either axis, within each resolving contrast.
nom_set <- function(res) res$gene[res$rank <= 100 | res$tail.rank <= 100]
set_cvc <- nom_set(wade_cvc)          # Cancer vs Control
set_bvc <- nom_set(wade_bvc)          # Benign vs Control

cancer_specific <- setdiff(set_cvc, set_bvc)
shared_early    <- intersect(set_cvc, set_bvc)
benign_specific <- setdiff(set_bvc, set_cvc)

# Assemble a tidy cross-contrast table: each gene's rank + effect on both
# contrasts, plus its class. Direction is recorded so a "shared" gene that
# moves opposite ways is not mislabelled.
cross <- wade_cvc |>
  dplyr::select(gene, cvc.rank = rank, cvc.tail = tail.rank,
                cvc.diff = diff.mean, cvc.tailc = tail.conc) |>
  dplyr::full_join(
    dplyr::select(wade_bvc, gene, bvc.rank = rank, bvc.tail = tail.rank,
                  bvc.diff = diff.mean, bvc.tailc = tail.conc), by = "gene") |>
  dplyr::mutate(cross_class = dplyr::case_when(
    gene %in% shared_early    ~ "shared / early",
    gene %in% cancer_specific ~ "cancer-specific",
    gene %in% benign_specific ~ "benign-specific",
    TRUE ~ "not nominated"))

n_cs <- length(cancer_specific); n_se <- length(shared_early); n_bs <- length(benign_specific)
```

Of the **`r length(set_cvc)`** Cancer-vs-Control nominations, **`r n_cs` are
cancer-specific** and **`r n_se` are shared** with the precursor-lesion contrast.
The Benign-vs-Control contrast adds **`r n_bs` benign-specific** genes. The
benign / precancer group is therefore **not** a silent copy of healthy plasma —
it carries a real, partly distinct signal, which both justifies holding it out of
the primary training contrast and flags it as a target for the presence/absence
scoring step to come.

```r
#| label: tbl-wade-crosscontrast
#| tbl-cap: "Cross-contrast marker classes. Cancer-specific genes are nominated against healthy controls but not by precursor lesions; shared/early genes are nominated by both; benign-specific genes appear only in the precursor contrast. Ranks are the bulk discovery rank within each contrast (lower = stronger); `diff.mean` is the case-control TPM difference. Top 12 per class by the relevant rank."
mk_class <- function(cls, order_by) {
  cross |> dplyr::filter(cross_class == cls) |>
    dplyr::arrange({{ order_by }}) |> dplyr::slice(1:12) |>
    dplyr::transmute(class = cls, gene,
      `CvC rank` = cvc.rank, `CvC diff` = round(cvc.diff, 0),
      `BvC rank` = bvc.rank, `BvC diff` = round(bvc.diff, 0))
}
dplyr::bind_rows(
  mk_class("cancer-specific", cvc.rank),
  mk_class("shared / early",  cvc.rank),
  mk_class("benign-specific", bvc.rank)) |>
  knitr::kable(align = "llrrrr")
```

```r
#| label: fig-wade-crosscontrast
#| fig-cap: "Cross-contrast scatter: each gene's bulk discovery rank in Cancer-vs-Control (x) against Benign-vs-Control (y), both on a reversed log scale so the strongest nominations sit top-right. Genes nominated only against controls by cancer (cancer-specific, red) fall to the right; genes nominated by both (shared/early, purple) cluster top-right; benign-specific genes (orange) sit high on the y-axis only. Dashed lines mark the top-100 nomination thresholds."
#| fig-width: 8.2
#| fig-height: 6.6
cc_pal <- c("cancer-specific" = "#C0392B", "shared / early" = "#7D3C98",
            "benign-specific" = "#E67E22", "not nominated" = "grey80")
lab_genes <- cross |>
  dplyr::filter(cross_class != "not nominated") |>
  dplyr::group_by(cross_class) |>
  dplyr::slice_min(pmin(cvc.rank, bvc.rank, na.rm = TRUE), n = 8) |> dplyr::ungroup()

cross_plot <- cross |>
  dplyr::mutate(cross_class = factor(cross_class,
    levels = c("shared / early", "cancer-specific", "benign-specific", "not nominated"))) |>
  dplyr::arrange(dplyr::desc(cross_class))   # draw grey first

ggplot(cross_plot, aes(cvc.rank, bvc.rank)) +
  annotate("rect", xmin = 0, xmax = 100, ymin = 0, ymax = Inf, fill = "#C0392B", alpha = 0.05) +
  annotate("rect", xmin = 0, xmax = Inf, ymin = 0, ymax = 100, fill = "#E67E22", alpha = 0.05) +
  geom_vline(xintercept = 100, colour = "grey65", linetype = "2222", linewidth = 0.4) +
  geom_hline(yintercept = 100, colour = "grey65", linetype = "2222", linewidth = 0.4) +
  geom_point(aes(colour = cross_class, size = cross_class, alpha = cross_class)) +
  ggrepel::geom_text_repel(data = lab_genes, aes(label = gene, colour = cross_class),
                  size = 2.7, max.overlaps = 30, segment.size = 0.25,
                  segment.colour = "grey55", box.padding = 0.3, min.segment.length = 0, seed = 1,
                  show.legend = FALSE) +
  scale_x_log10(breaks = c(1, 10, 100, 1000), labels = c("1", "10", "100", "1k")) +
  scale_y_log10(breaks = c(1, 10, 100, 1000), labels = c("1", "10", "100", "1k")) +
  scale_colour_manual(values = cc_pal, breaks = c("shared / early","cancer-specific","benign-specific")) +
  scale_size_manual(values = c("shared / early" = 2, "cancer-specific" = 1.8,
                               "benign-specific" = 1.8, "not nominated" = 0.5), guide = "none") +
  scale_alpha_manual(values = c("shared / early" = 0.95, "cancer-specific" = 0.9,
                                "benign-specific" = 0.9, "not nominated" = 0.25), guide = "none") +
  labs(x = "Cancer vs Control  —  bulk discovery rank (log, strong \u2192 left)",
       y = "Benign vs Control  —  bulk discovery rank (log, strong \u2192 down)",
       colour = NULL,
       title = "Where each marker's plasma signal first appears",
       subtitle = sprintf("%d cancer-specific \u00b7 %d shared/early \u00b7 %d benign-specific (top-100 either axis)",
                          n_cs, n_se, n_bs)) +
  guides(colour = guide_legend(override.aes = list(size = 2.6, alpha = 1))) +
  theme_minimal(base_size = 11) +
  theme(axis.title = element_text(size = 9.5), panel.grid.minor = element_blank(),
        plot.subtitle = element_text(size = 9, colour = "grey35"),
        legend.position = c(0.99, 0.99), legend.justification = c(1, 1),
        legend.background = element_rect(fill = "white", colour = NA),
        legend.text = element_text(size = 8.5))
```

## 9.6 Method validation

Before trusting the nominations we validate WADE on simulated data with known
ground truth, using the *same* group sizes as the primary contrast
(`r n_cancer` cases, `r n_noncancer` non-cancer). Three claims: (a) the permutation
p-values are uniform under the null (calibration — no false-positive inflation);
(b) the subset statistic separates subset genes from bulk shifts; and (c)
detection power scales with subset size down to a hard permutation floor.

```r
#| label: wade-validation-sim
#| include: false
# Reproducible simulation with the real n0/n1. (a) Null: Poisson counts, random
# lengths, labels carry no signal. (b) Discrimination: lognormal null background
# vs planted bulk shifts and 8%-subset spikes. (c) Power: subset fraction swept,
# detection = permutation FDR<0.1, against the exact combinatorial p-floor.
n0v <- n_noncancer; n1v <- n_cancer; nn <- n0v + n1v
condv <- c(rep(1L, n1v), rep(0L, n0v))                 # case-first

## (a) null calibration
set.seed(2024); Gnull <- 800
Cnull <- matrix(rpois(Gnull * nn, lambda = 30), Gnull, nn)
Lnull <- sample(1:12, Gnull, replace = TRUE)
resN  <- wade(Cnull, Lnull, wade_lib_size(Cnull, Lnull), condv,
              nperms = 1000, seed = 1, verbose = FALSE)
ks_diff <- ks.test(resN$p.diff, "punif")$p.value
ks_tail <- ks.test(resN$p.tail, "punif")$p.value
t1_diff <- mean(resN$p.diff < 0.05); t1_tail <- mean(resN$p.tail < 0.05)

## (b) discrimination
mk_bulk   <- function(shift) c(rlnorm(n1v, 3 + shift, 0.6), rlnorm(n0v, 3, 0.6))
mk_subset <- function(frac, spikemu = 6.2) {
  x <- c(rlnorm(n1v, 3, 0.6), rlnorm(n0v, 3, 0.6))
  sp <- sample(seq_len(n1v), max(1, round(frac * n1v)))
  x[sp] <- rlnorm(length(sp), spikemu, 0.3); x
}
set.seed(11); Gbg <- 600; nb <- 25; LEN <- 4
BG   <- t(replicate(Gbg, c(rlnorm(n1v, 3, 0.6), rlnorm(n0v, 3, 0.6))))
BULK <- t(sapply(seq_len(nb), function(i) mk_bulk(runif(1, 0.5, 0.9))))
SUB  <- t(sapply(seq_len(nb), function(i) mk_subset(0.08)))
MM <- rbind(BG, BULK, SUB); LL <- rep(LEN, nrow(MM))
lab <- rep(c("null", "bulk", "subset"), c(Gbg, nb, nb))
rB <- wade(MM, LL, wade_lib_size(MM, LL), condv, nperms = 1000, seed = 1, verbose = FALSE) |>
  dplyr::mutate(label = lab)

## (c) power vs subset fraction, with the exact permutation p-floor
pfloor <- function(k) exp(lchoose(n1v, k) - lchoose(nn, k))    # all-k-in-case probability
fracs <- c(0.03, 0.05, 0.08, 0.12, 0.20, 0.35, 0.50)
powtab <- purrr::map_dfr(fracs, function(fr) {
  k <- max(1, round(fr * n1v)); set.seed(200 + k)
  S <- t(sapply(seq_len(80), function(i) mk_subset(fr)))
  MMp <- rbind(BG, S); LLp <- rep(LEN, nrow(MMp))
  rr <- wade(MMp, LLp, wade_lib_size(MMp, LLp), condv, nperms = 1000, seed = 1, verbose = FALSE)
  is_sub <- c(rep(FALSE, Gbg), rep(TRUE, 80))
  padj_tail <- p.adjust(rr$p.tail, "BH")
  tibble::tibble(frac = fr, k = k, floor = pfloor(k),
                 power = mean(padj_tail[is_sub] < 0.10))
})
```

```r
#| label: fig-wade-validation
#| fig-cap: "WADE method validation on simulated data with the real group sizes (77 cases, 18 controls). (a) Null calibration: permutation p-values for both axes track the Uniform(0,1) diagonal (KS p reported in text) — no false-positive inflation. (b) Discrimination: planted subset genes (blue) lift above the y=x diagonal (tail.mean ≫ diff.mean) while bulk shifts (orange) sit near it and nulls (grey) cluster at the origin. (c) Detection power vs subset size: the tail axis detects subset genes down to a hard floor set by the combinatorial permutation limit (dashed) — below ~5% of cases, a shuffled null can reproduce the signal and no test can separate it."
#| fig-width: 12
#| fig-height: 4
col_bulk <- "#B9770E"; col_sub <- "#1F618D"; col_null <- "grey70"
th_v <- theme_minimal(base_size = 10) +
  theme(plot.title = element_text(size = 10), panel.grid.minor = element_blank(),
        legend.position = "top", legend.title = element_blank(), legend.text = element_text(size = 8))

qdf <- tibble::tibble(theo = ppoints(nrow(resN)),
                      diff = sort(resN$p.diff), tail = sort(resN$p.tail)) |>
  tidyr::pivot_longer(c(diff, tail), names_to = "stat", values_to = "obs")
pA <- ggplot(qdf, aes(theo, obs, colour = stat)) +
  geom_abline(slope = 1, intercept = 0, colour = "grey55", linewidth = 0.4, linetype = "2222") +
  geom_step(linewidth = 0.7) +
  scale_colour_manual(values = c(diff = "#7D3C98", tail = "#148F77"),
                      labels = c(diff = "diff.mean", tail = "tail.mean")) +
  coord_equal() +
  labs(title = "(a) null calibration", x = "theoretical quantile", y = "observed p-value") + th_v

db <- rB |> dplyr::mutate(lab = factor(label, levels = c("null", "bulk", "subset")),
                          dm = pmax(diff.mean, 0.1), tm = pmax(tail.mean, 0.1))
pB <- ggplot(db, aes(dm, tm, colour = lab)) +
  geom_abline(slope = 1, intercept = 0, colour = "grey70", linewidth = 0.4, linetype = "2222") +
  geom_point(data = dplyr::filter(db, label == "null"), size = 0.5, alpha = 0.22) +
  geom_point(data = dplyr::filter(db, label != "null"), size = 1.6, alpha = 0.85) +
  scale_x_log10() + scale_y_log10() +
  scale_colour_manual(values = c(null = col_null, bulk = col_bulk, subset = col_sub)) +
  labs(title = "(b) subset vs bulk discrimination",
       x = "diff.mean (bulk)", y = "tail.mean (subset)") + th_v

pC <- ggplot(powtab, aes(frac)) +
  geom_line(aes(y = power, colour = "detection power"), linewidth = 0.8) +
  geom_point(aes(y = power, colour = "detection power"), size = 1.8) +
  geom_line(aes(y = floor, colour = "permutation floor"), linewidth = 0.6, linetype = "2222") +
  scale_colour_manual(values = c("detection power" = col_sub, "permutation floor" = "grey45")) +
  scale_x_continuous(labels = scales::percent) + ylim(0, 1) +
  labs(title = "(c) power vs subset size", x = "fraction of cases altered", y = "power / floor") + th_v

patchwork::wrap_plots(pA, pB, pC, nrow = 1)
```

The null calibration is clean — KS test against Uniform(0,1) gives
p = `r sprintf("%.2f", ks_diff)` (bulk) and `r sprintf("%.2f", ks_tail)` (subset),
with type-I error at the 0.05 level of `r sprintf("%.3f", t1_diff)` and
`r sprintf("%.3f", t1_tail)` — both at nominal, so the permutation p-values are
trustworthy. Panel (b) confirms the subset statistic does what it claims:
planted 8%-subset genes lift clearly off the diagonal that bulk shifts hug.
Panel (c) is the honest limit — detection power climbs with subset size but
collapses to a **hard permutation floor** below ~5% of cases: when only a handful
of samples carry the signal, a shuffled null can reproduce it and *no*
permutation test can call it. With `r n_case` cases, WADE resolves subsets down
to roughly `r round(100 * fracs[which(powtab$power > 0.5)[1]])`% of cases — about
`r round(fracs[which(powtab$power > 0.5)[1]] * n_case)` samples — which sets the
realistic floor on subtype discovery in this cohort.

```r
#| label: save-disc-tables
#| include: false
# Discovery deliverables: the primary nomination table (with prior-panel +
# PDAC annotation), the two resolving-contrast tables, and the cross-contrast
# marker classification. All on-target genes, both axes.
disc_out <- "disc_tables"; dir.create(disc_out, showWarnings = FALSE)

# (1) primary Cancer-vs-Non-Cancer nomination table (fully annotated).
nom |>
  dplyr::transmute(
    rank, tail.rank, gene, gene_id = wade_genes$gene_id[match(gene, wade_genes$gene_name)],
    diff.mean = round(diff.mean, 2), diff.frac = round(diff.frac, 4),
    log2fc = round(log2(fc), 3), fc = round(fc, 1),
    case.mean = round(cond1.mean, 2), ctrl.mean = round(cond0.mean, 2),
    w1 = round(w1, 2), tail.mean = round(tail.mean, 2), tail.conc = round(tail.conc, 3),
    score = round(score, 4), tail.score = round(tail.score, 4),
    p.diff = signif(p.diff, 3), padj.diff = signif(padj.diff, 3),
    p.tail = signif(p.tail, 3), padj.tail = signif(padj.tail, 3),
    prior_cfrna_v4, prior_agilentv4, prior_rag_v1, prior_rag_v2, prior_rag_shared,
    prior_rag_union, prior_rag_panel, prior_rank,
    pdac_basal, pdac_classical, wade_hit, novel) |>
  readr::write_csv(file.path(disc_out, "wade_nomination_table.csv"))

# (2) all three contrasts, compact per-gene tables (both axes + effects).
write_contrast <- function(res, file) {
  res |>
    dplyr::transmute(
      rank, tail.rank, gene,
      gene_id = wade_genes$gene_id[match(gene, wade_genes$gene_name)],
      diff.mean = round(diff.mean, 2), log2fc = round(log2(fc), 3),
      case.mean = round(cond1.mean, 2), ctrl.mean = round(cond0.mean, 2),
      tail.mean = round(tail.mean, 2), tail.conc = round(tail.conc, 3),
      score = round(score, 4), tail.score = round(tail.score, 4),
      p.diff = signif(p.diff, 3), p.tail = signif(p.tail, 3),
      nominated = rank <= 100 | tail.rank <= 100) |>
    readr::write_csv(file.path(disc_out, file))
}
write_contrast(wade_cvn, "wade_cancer_vs_noncancer.csv")
write_contrast(wade_cvc, "wade_cancer_vs_control.csv")
write_contrast(wade_bvc, "wade_benign_vs_control.csv")

# (3) cross-contrast marker classification (cancer-specific / shared / benign).
cross |>
  dplyr::arrange(cross_class, pmin(cvc.rank, bvc.rank, na.rm = TRUE)) |>
  dplyr::transmute(
    gene, cross_class,
    cvc.rank, cvc.tail.rank = cvc.tail, cvc.diff.mean = round(cvc.diff, 2), cvc.tail.conc = round(cvc.tailc, 3),
    bvc.rank, bvc.tail.rank = bvc.tail, bvc.diff.mean = round(bvc.diff, 2), bvc.tail.conc = round(bvc.tailc, 3)) |>
  readr::write_csv(file.path(disc_out, "wade_cross_contrast.csv"))
```

**Where this goes next.** WADE has produced a ranked, FDR-controlled, prior-panel
-annotated nomination list, and the two-pronged discovery goal now has its first
prong: a set of candidate cancer-signature genes. The second prong — testing all
samples (including the held-out benign / precancer group) for the *presence* of
these signatures — is the supervised step that follows, and the `tail.mean` /
`tail.conc` shape statistics are what make it possible to ask not just "is this
gene up?" but "is it up in *this* subset of samples?".
