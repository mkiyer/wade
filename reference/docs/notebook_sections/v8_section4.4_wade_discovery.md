# cfRNA v8 §4.4 — Discovery: subset-aware differential expression (WADE)

> **Extracted reference material — not a specification.**
> Source: `cfrna_analysis/analysis_cfrna_v8.qmd` lines 3784–4282 (499 lines).
> cfRNA repo git SHA `828f2f1c6a10afd182fee0e119ba3595d98b6a72` (clean tree, 2026-08-17).
> sha256 of the whole source notebook: `2aacf2ae50b82415da671c78bdc45e5a2717fe018663504c083572e4aa02bae1`.
> Retrieve the original with `git show 828f2f1c6a10afd182fee0e119ba3595d98b6a72:cfrna_analysis/analysis_cfrna_v8.qmd`.
>
> Contains 15 R chunk(s) and 34 unresolved inline `r ...` expression(s). The inline expressions are **not** evaluated here: every number they would print is absent, and no value has been guessed or substituted. Quarto div syntax (`::: {.callout-*}`), cross-references (`@sec-*`, `§`) and chunk options (`#|`) are preserved as written. The only edit is the code-fence info string, `{r}` → `r`, so the R reads as R in a plain markdown viewer.
>
> The v8 rewrite of the discovery section. Where v7 ran three contrasts on one grouping, v8 enumerates nine candidate contrasts and screens each **before** running it; §4.4.2 is where the method acquires a refusal step. §4.4.3 reads the nominations and §4.4.4 repeats the whole screen on a second cohort. Lines 3784–3920 of this range overlap `section4_wade.qmd` (staged verbatim in `reference/docs/`) with §-numbering and callout-title syntax adapted for the rmarkdown renderer; §4.4.3 onward exists only in the notebook.
>
> Chunk labels in order: `tbl-wade-strata`, `fig-wade-screen`, `tbl-wade-screen`, `wade-gallery-setup`, `wade-gal-caption`, `fig-wade-gallery`, `tbl-wade-recurrence`, `tbl-wade-signal`, `fig-wade-volcano`, `wade-curve-caption`, `fig-wade-curves`, `fig-wade-cross`, `tbl-wade-nomination`, `tbl-wade-cohorts`, `wade-export`.

---

## 4.4 Discovery: subset-aware differential expression (WADE)

The unsupervised views above establish what varies across this cohort. This
section asks the discovery question directly: **which genes' plasma cfRNA
abundance separates one clinical group from another**, knowing that a "case"
group is heterogeneous — a signal may be present in a fraction of cases and
absent in the rest.

The method is **WADE** (Wasserstein Area Differential Expression), the lab's
subset-aware test. It compares two groups' empirical quantile functions rather
than their means, and reports a **bulk** axis (the signed area between them,
which equals the mean difference) alongside a **subset** axis (the mean
difference over the top-10% quantiles, where a rare high-expressing subset
concentrates). How the statistic works, why inference is by permutation with a
Generalized-Pareto tail refinement, and what the method does and does not
guarantee are in [S7. WADE](#sec-wade-method); this section is what the
contrasts found.

### 4.4.1 There is no single control group, so there is no single contrast

The cohort has one cancer group and **four different kinds of control**, and
they are different in kind rather than in label:

```r
#| label: tbl-wade-strata
wade_strata[[WADE_COHORT_MAIN]] |>
  dplyr::transmute(Group = label, Libraries = libraries, Patients = patients,
                   `Libs/patient` = libs_per_patient, `What it is` = note) |>
  knitr::kable(align = "lrrrl",
    caption = sprintf(paste("Clinical strata in the %s cohort. The four reference",
      "groups are not interchangeable: one is screened cancer-free, one is",
      "screened biomarker-POSITIVE with malignancy unresolved, one carries",
      "same-organ neoplastic precursor lesions, and one carries benign disease",
      "in other organs. Pooling them buys sample size and costs interpretation."),
      WADE_COHORT_MAIN))
```

This is a discovery section, so the taxonomy is treated as something to learn
from rather than a nuisance to average over. We therefore run WADE **three
ways**: with all controls pooled (maximum power), with the strata separated
(interpretable reference), and **control against control** — which asks what
each kind of reference population actually contributes, and is the only way to
find out whether a "cancer" signal is really a
disease-versus-cancer-free-plasma signal.

::: {.callout-important}
**v7's grouping code returns zero controls on this store**


v7 built its clinical groups with `grepl("control", disease)`. **No disease
label in the 2026-08-02 store contains the substring "control"** — the
surveillance group is `pancreas_high_risk` — so that expression silently
assigns all `r wade_n_hr` control libraries to "Cancer" and leaves the reference
group empty. `R/pca.R::cfrna_broad_group()` already caught and documented this.

Auditing the registry against this cohort's vocabulary found a second, live
instance: **`prostate_high_psa` appears in neither `CFRNA_CONTROL_GROUPS` nor
`CFRNA_BENIGN`**, so it falls through the "unknown is Cancer" default and
`r wade_n_psa` screen-positive control patients are counted as cancer cases.
Both are the same failure mode — a clinical category inferred from a string
pattern or a default instead of a registry — and
`cfrna_control_strata_audit()` now fails loudly on it rather than defaulting.
:::

### 4.4.2 Screening the contrasts before running them

Nine candidate contrasts are enumerated with their group sizes, and each is
screened on three axes **before** it is run. The screen is not a formality: it
refuses one contrast outright and caps what may be claimed from three more.

- **Probe design.** Panels were introduced over time and enrolment was not
  balanced across them, so a group difference can be a panel difference.
  Measured as Cramér's V on group × **design version** — not capture label,
  because v6f shares v6's gene targets and splitting them invents a batch the
  design does not have.
- **Sequencing depth.** WADE's statistic is a quantile area on normalized
  abundance, so a systematic depth difference shifts every quantile of the
  deeper group and manufactures signed area.
- **Quantile-grid resolution.** WADE compares the groups at
  `min(n_case, n_ctrl)` probabilities and the tail window is the top 10% of
  those. A contrast with 6 in the smaller group has a **one-point** tail: its
  "subset axis" is a single order statistic, and calling that subset detection
  is a category error rather than a weak result.

Every test is reported at two units — all libraries, and one library per
patient. The patient-level unit is the one that decides, because libraries are
not independent: within-patient library pairs correlate at median Spearman
`r sprintf("%.2f", wade_exch$within_patient)` against
`r sprintf("%.2f", wade_exch$same_disease)` for same-disease
different-patient pairs (*p* = `r sprintf("%.1e", wade_exch$p_value)`,
`r wade_exch$n_within` vs `r wade_exch$n_same_disease` pairs), so a
library-level test counts correlated replicates as independent observations.

```r
#| label: fig-wade-screen
#| fig-cap: "Contrast feasibility has two independent axes and a candidate can fail on either. x = size of the smaller group, which IS the quantile-grid resolution (log); y = Cramér's V between group and probe design version. Filled markers are one library per patient, open markers all libraries, joined by a segment — the gap between them is the pseudoreplication effect. Grey band: below the nomination floor. Red band: probe design predicts group, which no sample size fixes. Colour is the index-level verdict."
#| fig-width: 9.6
#| fig-height: 6.2
disc_fig_wade_screen(wade_screen[[WADE_COHORT_MAIN]], cohort_label = WADE_COHORT_MAIN)
```

```r
#| label: tbl-wade-screen
wade_screen[[WADE_COHORT_MAIN]] |>
  dplyr::filter(unit == "index") |>
  dplyr::transmute(Contrast = label, Family = family,
                   Case = n_case, Ctrl = n_ctrl, `Tail pts` = k_tail,
                   `V design` = round(panel_v, 2),
                   `Depth fc` = round(depth_fc, 2),
                   Verdict = verdict, Why = reason) |>
  knitr::kable(align = "llrrrrrll",
    caption = paste("Contrast screen, one library per patient. `Tail pts` is the",
      "number of order statistics in the top-10% window — the subset axis is",
      "believable only when this is 2 or more. Verdicts: `run` (no screen fired),",
      "`caution` (runnable, caveat named), `descriptive` (too small to nominate",
      "from, but the comparison is still interpretable — effect sizes only),",
      "`refuse` (design confounding makes the result uninterpretable at any",
      "sample size)."))
```

**One contrast is refused.** Screen-positive-vs-surveillance is confounded with
probe design at V = `r sprintf("%.2f", wade_v_screenpos)`: every
screen-positive library is v6-design while the surveillance group spans v3–v6,
so a gene list from it would have the panel as its most likely explanation.
That is not fixable by enrolling more patients into the existing design mix, and
restricting to v6-design alone leaves only
`r wade_pm_v6_nctrl` surveillance libraries — below the floor. What *is*
buildable is the panel-matched complement: **cancer vs screen-positive within
v6-design only**, where the confound is removed by construction rather than by
adjustment. That contrast is in the run set.

**Three control-vs-control contrasts are `descriptive`.** They survive the
design screen but have 5–7 patients in the smaller group. Rather than drop
them, we run them and report effect sizes and top genes while excluding them
from the nomination product — "underpowered" and "uninterpretable" are
different problems, and refusing to run a small comparison hides what the
current data does say. Each is a specific, quantified enrolment target rather
than a dead end.

### 4.4.3 What the contrasts found

```r
#| label: wade-gallery-setup
#| include: false
# wade_res[[cohort]] IS the named list of contrast tables -- no $res wrapper.
# (The standalone cache RDS I prototyped against has an extra level; the live
# object does not, and the section's other uses at 2269/2312 confirm the flat
# shape.) Assert it so a future shape change fails here rather than in a figure.
wade_gal_res <- wade_res[[WADE_COHORT_MAIN]]
stopifnot(is.list(wade_gal_res), length(wade_gal_res) > 0,
          is.data.frame(wade_gal_res[[1]]) || !is.null(wade_gal_res[[1]]$tbl))
wade_rec <- disc_tbl_wade_recurrence(wade_gal_res, 25L)

# THE AUDIT THAT CHANGES HOW THIS SECTION READS. §4.3.1 established that a gene
# whose reads carry no splice junction is genomic-DNA read-through rather than
# transcript. These contrasts were computed on TOTAL counts, so the question is
# whether the nomination list is enriched for such genes. It is, heavily.
wade_art <- local({
  z <- open_cohort(cd_total$counts_path)
  q <- read_rigel_quant(z, measures = c("count", "count_spliced"))
  C <- q$count[cd_total$gene_idx, cd_total$library_ids]
  S <- q$count_spliced[cd_total$gene_idx, cd_total$library_ids]
  gf <- rowSums(S) / pmax(rowSums(C), 1)
  names(gf) <- cd_total$gene_name
  gd <- qcc$gdna_pct[match(cd_total$library_ids, qcc$library)]
  cpm <- t(t(C) / (colSums(C) / 1e6))
  art <- gf < 0.10; spl <- gf > 0.50
  nom <- wade_rec$gene[wade_rec$n_contrasts >= 2]
  nom <- nom[nom %in% names(gf)]
  a <- sum(gf[nom] < 0.10); b <- length(nom) - a
  cc <- sum(art) - a; d <- sum(!art) - b
  ft <- stats::fisher.test(matrix(c(a, b, cc, d), 2))
  ctl <- stats::cor.test(1 - colSums(S) / colSums(C), gd, method = "spearman")
  ex <- rowMeans(C)
  lowq <- ex < stats::quantile(ex, 0.25); hiq <- ex > stats::quantile(ex, 0.75)
  list(gf = gf, n_nom = length(nom), n_art = a,
       frac_nom = a / length(nom), frac_bg = mean(art),
       or = unname(ft$estimate), p = ft$p.value,
       # The three tests that walked back the contamination reading. RAW counts,
       # not CPM: CPM is compositional, so a fall in the spliced genes forces a
       # rise in the complement and manufactures the effect being tested for.
       rho_raw_art = stats::cor(colSums(C[art, , drop = FALSE]), gd,
                                method = "spearman", use = "pair"),
       rho_cpm_art = stats::cor(colSums(cpm[art, , drop = FALSE]), gd,
                                method = "spearman", use = "pair"),
       rho_cpm_spl = stats::cor(colSums(cpm[spl, , drop = FALSE]), gd,
                                method = "spearman", use = "pair"),
       rho_lib = unname(ctl$estimate), p_lib = ctl$p.value,
       rho_expr = stats::cor(gf, rowMeans(cpm), method = "spearman"),
       spl_lowq = stats::median(gf[lowq]), spl_hiq = stats::median(gf[hiq]),
       mean_cpm = stats::setNames(rowMeans(cpm), cd_total$gene_name),
       surviving = sum(nom[gf[nom] < 0.10] %in% cd_analysis$gene_name))
})

wade_null <- local({
  # THE SECTION'S REAL HEADLINE, and it must be self-consistent: an earlier
  # version stated "exactly 0 results reach 0.05" and then named SPATA3 as one,
  # because the count came from this cohort while the example came from the other.
  # Everything here is derived from ONE cohort's tables, and the sentence about
  # the other cohort is generated from its own count.
  main <- wade_res[[WADE_COHORT_MAIN]]
  grab <- function(co) {
    p <- unlist(lapply(co, function(d) {
      d <- if (is.data.frame(d)) d else d$tbl
      if (is.null(d)) NULL else c(d$padj.diff, d$padj.tail)
    }))
    p[is.finite(p)]
  }
  ps <- grab(main)
  others <- setdiff(names(wade_res), WADE_COHORT_MAIN)
  osig <- vapply(others, function(ch) sum(grab(wade_res[[ch]]) < 0.05), numeric(1))
  sentence <- if (!length(others) || sum(osig) == 0) {
    "No other cohort definition produces one either."
  } else {
    sprintf(paste("The %s cohort yields %d, the SPATA3 tail statistic, which",
                  "[S7. WADE](#sec-wade-method) shows sits on the tail-fit",
                  "extrapolation floor with a one-point window."),
            names(osig)[which.max(osig)], max(osig))
  }
  lb <- wade_cds[[WADE_COHORT_MAIN]]$libraries
  list(n_contrast = length(main), n_tests = length(ps),
       n_sig = sum(ps < 0.05), min_overall = min(ps),
       n_genes = length(wade_cds[[WADE_COHORT_MAIN]]$gene_idx),
       n_surveil = dplyr::n_distinct(
         lb$patient[cfrna_control_strata(lb$disease) == "hr_surveil"]),
       sentence = sentence)
})

# The claim and its own counterexample must not contradict: if this cohort has no
# significant result, the prose must not simultaneously name one from it.
stopifnot(wade_null$n_sig == 0 || !grepl("SPATA3", wade_null$sentence))

wade_rec_tbl <- wade_rec |>
  dplyr::filter(n_contrasts >= 3) |>
  dplyr::mutate(`spliced fraction` = round(wade_art$gf[gene], 3),
                `mean CPM` = round(wade_art$mean_cpm[gene], 2)) |>
  dplyr::select(gene, n_contrasts, `spliced fraction`, `mean CPM`) |>
  utils::head(16)
```

```r
#| label: wade-gal-caption
#| include: false
WADE_GAL_CAP <- sprintf(paste0("Volcano for every contrast in the all-panel ",
  "cohort: difference in mean log2(CPM+1) against |WADE score|, one panel per ",
  "contrast on shared x limits. Red marks each contrast's own top %d by score, ",
  "and the %d strongest are labelled. The bracketed word is the screen verdict ",
  "from 4.4.2. Read this figure together with the note below it -- %d of the %d ",
  "genes nominated by two or more contrasts carry almost no splice-junction ",
  "reads, which the note below shows reflects low abundance rather than the ",
  "genomic-DNA contamination it first resembles."), 12L, 4L, wade_art$n_art,
  wade_art$n_nom)
```

:::: {.callout-note}
**Read the dashed line first: essentially nothing clears FDR 0.05**


Across `r wade_null$n_contrast` contrasts in this cohort, over both the mean and
the tail statistic, `r wade_null$n_sig` of `r format(wade_null$n_tests, big.mark = ",")`
gene-level tests reach an adjusted p below 0.05, and the smallest value anywhere
is `r sprintf("%.2f", wade_null$min_overall)`. `r wade_null$sentence`

That is the honest headline of this section, and the gallery is what makes it
visible: the significance line sits above every point in all
`r wade_null$n_contrast` panels. The ranked nominations below are therefore an
**ordering**, not a set of discoveries --- useful for choosing what to follow up,
not evidence that any individual gene differs between groups. §4.3.4 says why:
the group this cohort separates unsupervisedly is cancer against surveillance,
with `r wade_null$n_surveil` surveillance patients, and a two-group comparison at
that size does not survive multiplicity correction over
`r format(wade_null$n_genes, big.mark = ",")` genes.
::::

```r
#| label: fig-wade-gallery
#| fig-cap: !expr WADE_GAL_CAP
#| fig-width: 13
#| fig-height: 11
disc_fig_wade_gallery(wade_gal_res, wade_screen[[WADE_COHORT_MAIN]])
```

Nine contrasts on shared axes, so an effect size in one panel means the same
thing as an effect size in another. The screen verdict is printed on each panel:
the `descriptive` ones are shown because a reader should see how little separates
those groups, not because they support nomination.

:::: {.callout-warning}
**The recurring nominations are low-splice genes --- but that is not proof of contamination**


`r wade_art$n_art` of the `r wade_art$n_nom` genes nominated by two or more
contrasts carry almost no splice-junction reads (under 10% of their counts),
against `r sprintf("%.0f%%", 100 * wade_art$frac_bg)` of genes overall --- odds
ratio `r sprintf("%.1f", wade_art$or)`, p = `r sprintf("%.0e", wade_art$p)`. The
list is heavily enriched for genes whose signal is unspliced.

**What that does and does not license.** I first read this as genomic-DNA
read-through inflating the nominations, and the direct test does not support
that reading:

- raw summed counts of those genes against the independent gDNA estimate:
  $\rho$ = `r sprintf("%+.2f", wade_art$rho_raw_art)`. **No association.** An
  earlier CPM-based version of this test gave
  $\rho$ = `r sprintf("%+.2f", wade_art$rho_cpm_art)`, but CPM is a share: the
  well-spliced genes fall with contamination
  ($\rho$ = `r sprintf("%+.2f", wade_art$rho_cpm_spl)`), so the complement is
  forced upward. The apparent effect was compositional.
- the library-level prediction also fails: unspliced share against the gDNA
  estimate is $\rho$ = `r sprintf("%+.2f", wade_art$rho_lib)`
  (p = `r sprintf("%.2f", wade_art$p_lib)`), consistent with the pipeline
  deconvoluting genomic DNA upstream as documented.

**The confound that does hold is abundance.** Spliced fraction tracks expression
at $\rho$ = `r sprintf("%+.2f", wade_art$rho_expr)` --- a lowly expressed gene
has fewer junction-spanning reads by sampling alone, median
`r sprintf("%.2f", wade_art$spl_lowq)` in the bottom expression quartile against
`r sprintf("%.2f", wade_art$spl_hiq)` in the top. So "low spliced fraction" means
"lowly expressed" at least as much as it means "not a transcript", and these
nominations are concentrated among genes near the detection floor where a
quantile statistic has least to work with.

**The recurrence pattern is the durable warning.** A gene nominated in six of
nine contrasts, including comparisons between two *control* groups, is not a
cancer marker under any reading --- it is tracking whatever differs between
libraries generally. That is an argument for treating the recurring names as
low-confidence regardless of mechanism, and it is why the table below reports
recurrence next to splice fraction rather than either alone.
::::

```r
#| label: tbl-wade-recurrence
knitr::kable(wade_rec_tbl, align = "lrrl",
  caption = paste("Genes nominated by three or more contrasts, with the",
                  "fraction of their reads carrying a splice junction.",
                  "Recurrence across unrelated contrasts is a warning sign, not",
                  "corroboration."))
```



```r
#| label: tbl-wade-signal
wade_sig[[WADE_COHORT_MAIN]] |>
  dplyr::transmute(Contrast = label, Case = n_case, Ctrl = n_ctrl,
                   `p<0.01 bulk` = n_bulk, `p<0.01 tail` = n_tail,
                   Expected = expected, `Enrich bulk` = enrich_bulk,
                   `Min FDR` = signif(pmin(min_padj_bulk, min_padj_tail), 2),
                   `Secs` = round(elapsed_sec)) |>
  knitr::kable(align = "lrrrrrrrr",
    caption = paste("Raw signal against chance expectation.",
      "With ~2,200-2,800 genes and 13-34 reference libraries, exactly one gene",
      "across all", wade_n_runs, "runs clears FDR < 0.10 on either axis -- and",
      "that one is an extrapolation artefact rather than a discovery (S7.4).",
      "`Min FDR` is therefore a power statement, not an absence of signal: the",
      "enrichment of nominal p<0.01 genes over the number expected by chance is",
      "what says the signal is real. Nomination is by rank score throughout."))
```

The framing v7 established still holds and is worth restating plainly: **with
this many genes and this few reference libraries the p-values are not the
product**. Across all `r wade_n_runs` runs, exactly one gene clears FDR < 0.10 —
`r wade_gpd_gene` on the subset axis of the pancreatic-precursor contrast — and
its *p*-value sits precisely at the tail-fit extrapolation floor on a one-point
tail window, so it is a resolution limit being reported rather than a
discovery (see [S7. WADE](#sec-wade-method)). What the permutation test does
provide is the
enrichment above chance, and the protection against calling a single extreme
sample a discovery. Genes are therefore **nominated by rank score**, and where
the tail window is one or two order statistics the subset axis is dropped from
nomination entirely.

```r
#| label: fig-wade-volcano
#| fig-cap: "Bulk effect against subset effect for the primary detection contrast. Each point is one on-target gene, both axes up-in-case, log scales with explicit ranges from the data. Every gene sits above the y=x diagonal because the tail is the high end of the distribution, so the distance above the diagonal is the subset signal. Colour splits nominated genes by tail concentration; grey is unnominated."
#| fig-width: 9.6
#| fig-height: 6.6
disc_fig_wade_volcano(wade_res[[WADE_COHORT_MAIN]][[WADE_PRIMARY]],
                      tail_ok = wade_tail_ok[[WADE_PRIMARY]],
                      extra_labels = WADE_CANON_GENES)
```

```r
#| label: wade-curve-caption
#| include: false
# A multi-line `!expr` is not valid YAML -- knitr hands the chunk header to
# the YAML parser first, which stops at the first newline inside the string.
# Build the caption here and let the header reference it by name.
WADE_CURVE_CAP <- sprintf("The statistic one gene at a time, on the primary contrast. Top row: case and control quantile functions, grey band = top-10%% tail window. Bottom row: cumulative signed area, whose endpoint at p=1 equals diff.mean, and whose SHAPE is the whole distinction the subset axis exists to capture. The four genes are selected from the data by rule, not by hand, and the profile claims are verified rather than asserted: *%s* is elevated above every control in %d patients (a genuine subset), *%s* rises steadily from the middle quantiles (a bulk shift), *%s* clears the control maximum in only %d patient(s) and the permutation test therefore refuses it despite tail.conc = 0.99, and *%s* is a null. The outlier panel is the important one: shape statistics alone cannot tell it from the subset marker, and the permutation p-value is what separates them.", WADE_CURVE_GENES[1], WADE_CURVE_NPAT[[WADE_CURVE_GENES[1]]], WADE_CURVE_GENES[2], WADE_CURVE_GENES[3], WADE_CURVE_NPAT[[WADE_CURVE_GENES[3]]], WADE_CURVE_GENES[4])
```

```r
#| label: fig-wade-curves
#| fig-cap: !expr WADE_CURVE_CAP
#| fig-width: 11
#| fig-height: 5.4
disc_fig_wade_curves(wade_tpm, wade_cond, WADE_CURVE_GENES,
                     wade_res[[WADE_COHORT_MAIN]][[WADE_PRIMARY]],
                     titles = WADE_CURVE_TITLES)
```

The outlier panel is the cautionary case that justifies carrying a p-value the
nomination does not use. Its `tail.conc` is 0.99 — on the shape statistics alone
it is indistinguishable from the top-ranked subset marker — but it clears the
control maximum in only
`r WADE_CURVE_NPAT[[WADE_CURVE_GENES[3]]]` patient(s), and the permutation test
declines it (*p* = `r sprintf("%.2f", dplyr::filter(wade_res[[WADE_COHORT_MAIN]][[WADE_PRIMARY]], gene == WADE_CURVE_GENES[3])$p.tail)`).
The shape statistics describe subset structure; the permutation p-value is what
stops one extreme sample being called a discovery.

```r
#| label: fig-wade-cross
#| fig-cap: "Cross-contrast nomination membership. Rows are genes nominated by at least one contrast, ordered by how many contrasts nominated them and then by best rank, so the top of the panel is the consensus set and the bottom is contrast-specific. Columns are contrasts, coloured by family. Grey = not nominated by that contrast. Reading down a single column gives that contrast's private nominations; reading across the top rows gives the genes that do not depend on which reference group was used."
#| fig-width: 9.2
#| fig-height: 8.4
disc_fig_wade_cross(wade_cross[[WADE_COHORT_MAIN]],
                    screen = dplyr::filter(wade_screen[[WADE_COHORT_MAIN]], unit == "index"),
                    cohort_label = WADE_COHORT_MAIN)
```

The cross-contrast panel is what the control taxonomy buys. A gene nominated
against every reference group is a gene whose signal does not depend on which
control population it was measured against; a gene nominated only against the
pooled reference is one whose apparent signal may be carried by whichever
stratum dominates that pool. Genes nominated by a **control-vs-control**
contrast are the diagnostic cases: they mark axes on which two nominally
cancer-free populations differ, so a cancer nomination sharing them is
measuring disease or referral pathway rather than malignancy.

```r
#| label: tbl-wade-nomination
wade_nom_table[[WADE_COHORT_MAIN]] |>
  knitr::kable(align = "rlrrrrrl",
    caption = paste("Top nominations for the primary detection contrast, ranked",
      "by the bulk discovery score. `diff.mean` is the mean case-control",
      "difference in TPM; `tail.conc` is the share of that signal concentrated",
      "in the top-10% quantiles, so a high value marks a subset-driven gene;",
      "`other contrasts` counts how many of the remaining contrasts also",
      "nominate the gene. Full per-gene tables for every contrast are exported",
      "to `disc_tables/`."))
```

### 4.4.4 Both cohorts

Every contrast is run on both cohort definitions — all panels, and v3 omitted —
and the results are keyed by cohort throughout. The two differ in libraries and
genes together, so a nomination present in one and absent from the other may be
a gene the other cohort cannot measure rather than a disagreement. The
head-to-head comparison belongs downstream, once the other discovery features
exist for both; here both sets are simply produced and cached.

```r
#| label: tbl-wade-cohorts
wade_cohort_compare |>
  knitr::kable(align = "llrrrrr",
    caption = paste("The same contrasts on both cohort definitions. `nominated`",
      "is the top-100 union across axes where the tail axis is usable;",
      "`shared` counts genes nominated in both cohorts, and `measurable`",
      "is how many of the other cohort's nominations exist in this cohort's",
      "gene set at all -- the denominator that makes `shared` interpretable."))
```

```r
#| label: wade-export
#| include: false
# Per-contrast per-gene tables, one CSV per contrast per cohort, plus the
# cross-contrast membership table. Written from the cached results, so this
# chunk is cheap and the CSVs cannot disagree with the figures above.
wade_out <- cfrna_here("disc_tables"); dir.create(wade_out, showWarnings = FALSE)
for (ch in names(wade_res)) {
  slug_ch <- gsub("[^a-z0-9]+", "-", tolower(ch))
  for (id in names(wade_res[[ch]])) {
    cfrna_wade_nominate(wade_res[[ch]][[id]], tail_ok = isTRUE(wade_tail_ok[[id]])) |>
      dplyr::transmute(rank, tail.rank, gene,
        diff.mean = round(diff.mean, 2), log2fc = round(log2(fc), 3),
        case.mean = round(cond1.mean, 2), ctrl.mean = round(cond0.mean, 2),
        w1 = round(w1, 2), tail.mean = round(tail.mean, 2),
        tail.conc = round(tail.conc, 3),
        score = round(score, 4), tail.score = round(tail.score, 4),
        p.diff = signif(p.diff, 3), padj.diff = signif(padj.diff, 3),
        p.tail = signif(p.tail, 3), padj.tail = signif(padj.tail, 3),
        nominated) |>
      readr::write_csv(file.path(wade_out, sprintf("wade_%s_%s.csv", slug_ch, id)))
  }
  readr::write_csv(wade_cross[[ch]],
                   file.path(wade_out, sprintf("wade_cross_contrast_%s.csv", slug_ch)))
  readr::write_csv(dplyr::filter(wade_screen[[ch]], unit == "index"),
                   file.path(wade_out, sprintf("wade_contrast_screen_%s.csv", slug_ch)))
}
```

**Where this goes next.** The discovery product from this section is a set of
ranked nominations, each annotated with which reference population it was
measured against and which others reproduce it. The screen's `descriptive` and
`refuse` verdicts are the enrolment specification that follows from it: more
surveillance controls on the current panel would move three control-vs-control
comparisons from descriptive to nominatable, and a surveillance arm on the
v6 design is what would make the screen-positive contrast testable at all.
