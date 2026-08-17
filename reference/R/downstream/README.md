# Downstream consumers of the WADE API — reference only, not porting targets

These three files are the cfRNA-specific layers that sat *above* `wade.R` in the
source project. They are here as **evidence**, not as specification:

1. **What a real caller actually needs from the WADE API.** `wade.R` on its own
   does not tell you which of its arguments matter in practice, which return
   columns get consumed, what a caller has to build before it can call, or what
   it has to remember afterwards. `wade_contrasts.R` is the only worked example
   of all four.
2. **What the statistic looks like when visualized.** A quantile-area statistic
   is hard to reason about from its formula. `wade_figures.R` is the record of
   which views made it legible to readers — and, in two places, of defects that
   only surfaced when someone tried to plot the output.

Nothing in this directory should be ported. The Python package's own consumer
layer will be written against its own data model, and its figures against
whichever plotting stack that package adopts. Read these to learn what the API
has to support and what the numbers do; then close them.

All three are byte-identical copies of the cfRNA originals — see
`../CHECKSUMS.txt` for the digests and `../PROVENANCE.md` for source paths and
the cfRNA git SHA.

---

## What each file contains, and what generalizes

### `control_strata.R` — the control taxonomy the contrasts resolve against

**cfRNA-specific in its content; generalizable in its discipline.** Every
disease string, stratum name and display label is specific to one plasma cfRNA
cohort and carries no meaning outside it.

What does generalize is the pattern, and it is worth stating because it is a
*correctness* pattern rather than a stylistic one. The module's own header
documents two defects it exists to prevent: a substring match (`grepl("control",
disease)`) that silently returned zero controls on a store where the control
group is named `pancreas_high_risk`, and an incomplete registry that let nine
screen-positive control libraries fall through a default into the cancer arm.
The fix in both cases was the same: name the reference groups in a closed
registry, treat the complement as the case class, make an unregistered label an
error rather than a default, and provide an audit function
(`cfrna_control_strata_audit()`) that reports every label taking the complement
branch. That is a group-assignment discipline any two-group test needs, and it
is the reason the cfRNA project keeps group definition *outside* `wade.R`.

Functions: `cfrna_control_strata()`, `cfrna_control_strata_audit()`,
`cfrna_strata_annotate()`, `cfrna_strata_summary()`, `cfrna_stratum_libs()`.
All operate on the cohort's own disease vocabulary; none is portable as written.

### `wade_contrasts.R` — registry, feasibility screen, cohort plumbing, cache

This is the file to read for API design. Four distinct concerns, with different
degrees of generality:

**cfRNA-specific.** The contrast registry `CFRNA_WADE_CONTRASTS` (ten named
candidate comparisons with a one-line rationale each), `cfrna_wade_resolve()`
(turns a registry entry into two library-id vectors via the stratum module),
`cfrna_wade_panel_matched()` (would restricting to one probe-design version
rescue a confounded contrast?), and `cfrna_wade_contrast()` /
`cfrna_wade_run_all()` (which know about cohort objects, effective-length
matrices, and an on-disk `.rds` cache). Every threshold constant
(`CFRNA_WADE_MIN_N = 10`, `CFRNA_WADE_TAIL_MIN_N = 20`,
`CFRNA_WADE_PANEL_V_MAX = 0.50`, `CFRNA_WADE_NPERMS = 2000`,
`CFRNA_WADE_TAIL_Q = 0.10`) is a decision about one cohort, not a property of
the method.

**Potentially generalizable — and this is the important part.** Two things in
this file are really statements about WADE itself that happen to be written in
cfRNA terms:

- **The sample-size tier logic.** `cfrna_wade_screen()` encodes the fact that
  `min(n0, n1)` *is* the quantile-grid resolution and that the tail window is
  `ceiling(tail_q * min(n0, n1))` order statistics. Its comments spell out the
  consequence: a contrast with 6 in the smaller group has a one-point tail, and
  "calling that a subset detector is a category error". It therefore reports
  `nprobs` and `k_tail` per contrast and emits a `tail_usable` flag that
  downstream nomination consumes. This is arithmetic on WADE's own parameters —
  any port that exposes `tail_q` should be able to tell a caller how many order
  statistics that buys, and any port's documentation needs the same warning.
- **The four-verdict classification, and specifically that it classifies rather
  than gates.** `run` / `caution` / `descriptive` / `refuse`, where the
  distinction between the middle two is that an underpowered comparison is
  still *interpretable* (run it, report effect sizes, exclude it from the
  nomination product) while a design-confounded one is not interpretable *at
  any sample size*. Every verdict is recorded with the numbers that produced
  it. The specific tests are cohort-specific (Cramér's V on probe-design
  version, a Wilcoxon on sequencing depth), but the shape — screen before you
  run, classify rather than silently drop, keep the numbers — is worth
  carrying.
- **`cfrna_wade_nominate()` and the rank-vs-p-value decision.** The file
  records that with roughly 2,200 genes and 22 controls the permutation
  p-values cannot survive BH correction, so nomination is by rank score and the
  p-values are read as effect-strength annotation. That is a statement about
  what WADE's output can support at small reference-group sizes.
- **`cfrna_wade_cross()` and `cfrna_wade_signal_check()`.** Cross-contrast
  membership ("which comparisons nominated this gene") and a
  chance-expectation check (how many genes clear nominal *p* < cut versus how
  many are expected). Both are generic post-processing over a set of WADE
  result frames and would be reasonable additions to a package's own analysis
  layer.

**Worth reading even though it is not portable:** `digest_string()`, a
hand-rolled polynomial hash for cache keys, whose docstring explains why the
obvious FNV-1a choice fails in R (the running state exceeds
`.Machine$integer.max`, `bitwXor()` coerces and returns `NA` with an easily
missed warning). Also read the cache-key discussion in
`cfrna_wade_contrast()`: the key covers contrast id, cohort name and
fingerprint, gene count, the sorted library ids of both groups, `nperms`,
`tail_q` and `seed` — the stated rule being that the key must change whenever
the result would.

### `wade_figures.R` — what the statistic looks like when visualized

**Almost entirely cfRNA-specific as code**, and all of it dependent on ggplot2
(plus `ggrepel` for labels, `patchwork` and `tidyr` inside the per-gene
diagnostic). The palettes, verdict vocabulary, gene-symbol italicisation and
axis wording are house conventions of one project.

**Generalizable as views of the statistic.** Five figures, of which three are
about WADE rather than about cfRNA:

- `disc_fig_wade_volcano()` — bulk effect (`diff.mean`) against subset effect
  (`tail.mean`), both on log axes of *x*+1. The file notes that every gene sits
  above *y* = *x* because the tail is the high end of the distribution, so the
  **distance above the diagonal is the subset signal**; that is why the
  diagonal is drawn rather than a fitted line. It also collapses its
  nomination classes to a single class when the tail window is too small to
  support the subset/bulk split, rather than labelling every gene "broad /
  bulk" and implying the contrast ruled subset structure out.
- `disc_fig_wade_curves()` — the per-gene diagnostic, and the panel that makes
  the statistic legible. Case and control quantile functions on top, cumulative
  signed area below (endpoint = `diff.mean`, the identity `wade_gene()` is
  built to satisfy). Its documented reading: a broad shift rises steadily, a
  rare high subset stays flat then climbs inside the tail window, a single
  outlier stays flat then spikes. Any port should be able to reproduce this
  panel; if it cannot, it is missing `wade_gene()`'s per-gene detail output.
- `disc_fig_wade_gallery()` — one volcano per contrast on shared axes. Its
  comment records a real failure: the first version plotted `diff.mean` on *x*,
  which spans roughly ±9,000 in these units and collapsed every point onto a
  vertical line at zero; it was changed to log2 fold change. A porting note in
  disguise — WADE's signed-area quantities are not fold changes and do not plot
  like them.
- `disc_fig_wade_screen()` and `disc_fig_wade_cross()` — feasibility screen
  (smaller-group size against Cramér's V, verdict as colour) and a
  cross-contrast nomination dot matrix. Both are views of the cfRNA registry's
  own output, not of the statistic.
- `disc_tbl_wade_recurrence()` — per-contrast nomination overlap as a table.
  Generic over a named list of result frames.

**Read `wade_tail_conc_display()` before porting `tail.conc`.** It is the one
place in this directory that documents a measured defect in the statistic
itself. `tail.conc` is a *share*: signed tail area over total signed area. When
the lower quantiles' differences cancel the upper ones the denominator vanishes
and the ratio explodes. On the cfRNA primary contrast the file reports 120 of
2,219 genes (5.4%) returning `|tail.conc| > 2`, one as high as 149. `wade.R`
does guard this, but its threshold is `|diff.mean * nprobs| < 1e-8` — which the
file computes as about `|diff.mean| < 5e-10` on that contrast, roughly nine
orders of magnitude tighter than the cases that actually occur. The guard was
therefore applied at the display layer rather than in the statistic, to avoid
invalidating cached results for a cosmetic column, and the method-level fix was
recorded as a porting item. The file also notes that `tail.conc` slightly above
1 is legitimate (the tail carries more than the total because the bulk
partially cancels), so a fix must not simply clamp to 1.

---

## These files do not run standalone here — what was actually verified

Each file was sourced on its own in this repo's pinned renv sandbox
(`Rscript -e "source('downstream/<file>')"`). The results, and the precise
mechanism, are worth stating because the mechanism is *not* `source()` — **none
of these three files contains a `source()` call.** They assume the caller has
already loaded their dependencies, which in the cfRNA project was a documented
load-order contract.

| file | sourcing on its own | why |
| --- | --- | --- |
| `control_strata.R` | **succeeds, and its functions run** | Self-contained apart from dplyr. Verified working on a synthetic six-library table: strata assigned, audit reports the complement labels, `cfrna_stratum_libs()` and `cfrna_strata_summary()` both return. |
| `wade_figures.R` | **succeeds** (definitions load) | But the figure functions fail at *call* time. Verified absent in a bare session: `cfrna_wade_nominate()` (needed by the volcano), `wade_gene()` (needed by the per-gene diagnostic), and the three threshold constants that are the screen figure's default arguments. `ggrepel` — used by three of the figures — is also not in this sandbox's lockfile. The pure helpers do run standalone; each was invoked on test input — `wade_tail_conc_display()` (NA-guards a 149 to `NA`, passes 0.5 through), `wade_log_limits()`, `wade_log_breaks()` (whole decades inside a range, falling back to `pretty()` on a sub-decade range) with `wade_log_labels()`, and `disc_tbl_wade_recurrence()` on two synthetic contrasts. |
| `wade_contrasts.R` | **fails immediately, by design** | Its own header block checks for four functions and stops with the list of what is missing: `cfrna_strata_annotate()` from `R/control_strata.R`, `cfrna_gs_design()` from `R/group_structure.R`, `cfrna_index_libraries()` from `R/perspective.R`, and `wade_run()` from `wade.R`. The module's comment explains the choice: failing here is a better error message than "object not found" three calls deep. |

Two of those four dependencies — `cfrna_gs_design()` (probe-design version from
a capture label) and `cfrna_index_libraries()` (one index library per patient:
earliest timepoint, then best on-target spliced yield) — live in cfRNA modules
that were **not** copied into this repo, because they are cohort plumbing with
no bearing on the port. So `wade_contrasts.R` cannot be made to run here even
by sourcing the other two files in this directory, and it needs cohort data
that this repo does not contain in any case.

Read them as documents.
