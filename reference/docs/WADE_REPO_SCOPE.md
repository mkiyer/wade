# Extracting WADE into a standalone repo — scope analysis

**Status: analysis only. Nothing created, nothing moved.** This note is the
inventory and the proposed cut line for a separate repository carrying Python
and R implementations of WADE. It is written against
`cfrna_analysis/wade.R` (16 kB, 9 exported functions) and
`cfrna_analysis/R/wade_contrasts.R` (the cfRNA wrapper written for v8 §4).

## The short version

`wade.R` is already close to a clean method module: it has no knowledge of
diseases, probe panels, cohorts, or the Zarr store, and its only dependencies
are `matrixStats`, `tibble` and `dplyr`. **Seven of its nine functions move
essentially as-is.** The work is not disentangling method from plumbing inside
`wade.R` — that separation was made when it was written. The work is (a)
deciding what the method's *input contract* is, since two of the nine functions
encode a normalization choice that arguably is not the method's business, and
(b) reproducing the R numerics in Python exactly enough that the two can be
tested against each other.

The cfRNA-specific layer already lives outside `wade.R`, in
`R/wade_contrasts.R`. That file **stays here in full** and is the model for
what a downstream consumer looks like.

## What moves: the method

| Function | Role | Notes for the port |
|---|---|---|
| `wade_stats()` | The statistic. Empirical quantile functions on a shared `min(n0,n1)` grid; signed area (`diff.mean`), `w1`, `tail.mean`, `tail.conc`. | The core. `matrixStats::rowQuantiles` → `numpy.quantile(..., axis=1)`. **Quantile convention is a hard compatibility point** — see below. |
| `.wade_null_stats()` | Lean permutation-loop variant returning only the two null quantities. | A performance specialisation, not separate science. Keep as a private fast path; in Python it is the same function with a flag. |
| `.gpd_tail_p()` | GPD tail refinement for small permutation p (Knijnenburg 2009), method-of-moments, with the ξ≤0 exponential fallback and the `1/(B·n_tail)` floor. | Pure scalar math, ports directly. The documented upgrade to MLE-based GPD belongs in the new repo, not here. |
| `wade_perm_pvalues()` | Empirical p, then GPD refinement where exceedances are too few. | Ports directly. |
| `wade()` | Driver: normalize → observed stats → permutation loop → BH-FDR → tidy frame. | Ports directly. The permutation loop is the obvious target for vectorisation/parallelism in the new repo; the R version is a serial `for`. |
| `wade_score()` | The two rank scores (`score`, `tail.score`) from empirical CDFs of `|log2fc|`, case mean, control mean. | Ports directly, but see the caveat below — these are *panel-selection heuristics*, not part of the test. |
| `wade_gene()` | Single-gene quantile + cumulative-area detail for the diagnostic plot. | Ports directly. Small and worth keeping: it is what makes the statistic legible. |

## What moves but should be reconsidered on the way

`wade_lib_size()` and `wade_normalize()` are the (count, normalizer) → TPM-like
step, plus a fixed-seed uniform continuity jitter to break ties in
zero-heavy data.

These are *usable* as method code, and the jitter in particular is genuinely
part of how the test behaves on sparse counts — it is applied once, before
permutation, so the null is conditional on the noise draw. Keeping it is
defensible.

But they also encode a normalization opinion that a general-purpose package
should probably not impose. Two concrete reasons, both specific to this
project rather than hypothetical:

1. **It is not the normalizer this project uses anywhere else.** The cfRNA
   toolkit settled on `ic_spl_sum` (spliced counts over 20 internal control
   genes) after a 16-candidate evaluation, and every other analysis in v8
   reads `cfrna_cd_matrix()`. `wade()` instead applies its own
   count/normalizer TPM. Measured on this cohort, the residual dependence of
   per-library mean abundance on sequencing depth is ρ = 0.41 for WADE's
   internal TPM against ρ = 0.36 for `ic_spl_sum`, both against ρ = 0.79
   uncorrected (all-panels cohort; 0.30 / 0.28 / 0.70 on the v3-omitted
   cohort). So WADE's normalizer is *not* the source of a problem here — it
   corrects about as well as the house normalizer — but it is a second,
   redundant normalization decision living inside a test.
2. **A method package that normalizes for you cannot be handed an
   already-normalized matrix**, which is the common case for anyone with an
   existing pipeline.

**Proposed cut:** the new repo's primary entry point takes a *matrix that is
already on a comparable scale* plus the condition vector. `wade_normalize()` /
`wade_lib_size()` ship as an optional convenience layer
(`wade.normalize.tpm_like()`), clearly documented as one choice among many, with
the jitter exposed as a parameter and its rationale in the docstring. The
cfRNA notebook keeps calling the convenience layer, so nothing changes here.

## What stays: cfRNA plumbing

Everything in `R/wade_contrasts.R`, none of which the method needs to know:

- **`CFRNA_WADE_CONTRASTS`** — the contrast registry (case/control strata,
  family, rationale). Clinical vocabulary.
- **`cfrna_wade_resolve()`** — stratum names → library ids, including the
  design-version restriction for panel-matched contrasts.
- **`cfrna_wade_screen()` / `cfrna_wade_cramer_v()` / `cfrna_wade_panel_matched()`**
  — the feasibility and confounding screen (probe design, sequencing depth,
  quantile-grid size) and its four verdicts. *This is the part with no
  analogue in a generic package*: "is this contrast confounded with the
  capture panel it was run on" is a cfRNA question.
- **`cfrna_wade_contrast()` / `cfrna_wade_run_all()`** — cohort-object plumbing
  and the cache-key discipline (store fingerprint, cohort name, gene count,
  both library-id lists).
- **`digest_string()`** — cache-key hash. Incidental; delete if the new repo
  ever gains a real dependency set.
- **`cfrna_wade_signal_check()` / `cfrna_wade_cross()`** — reporting layers.
  Borderline: the chance-expectation check is generic enough to move, the
  cross-contrast classification is about *this* set of contrasts.
- **`R/control_strata.R`** — the control taxonomy. Entirely clinical.

### The rank scores are a judgement call, and the caveat has to travel

`wade_score()` is method-shaped (it reads only columns `wade()` produced) but is
not part of the *test*: it is the lab's panel-selection heuristic, and it is
what nomination actually uses here, because with ~2,200 genes and 13–34
reference libraries no gene survives BH on either axis. If it moves, it must
move with that framing attached — a package that offers `score` next to
`padj.diff` without comment invites a user to treat a heuristic rank as
inference. Recommend: move it, in a `wade.rank` submodule, documented as
nomination-not-inference.

## The interface across the boundary

What the notebook would call, once WADE is external. This is close to the
current `wade_run()` signature, minus the normalization it currently does for
you:

```r
# R
wade::wade(x, cond,                    # matrix (genes x samples), 0/1 vector
           nperms = 2000, tail_q = 0.10,
           seed = 1L)                  # -> tidy frame, one row per gene
wade::wade_rank(res)                   # adds score / tail.score / rank
wade::wade_gene(x[g, ], cond)          # single-gene diagnostic curves
wade::normalize_tpm_like(counts, normalizer, lib_sizes, noise = 0.01)  # optional
```

```python
# Python
wade.wade(X, cond, nperms=2000, tail_q=0.10, seed=1) -> pandas.DataFrame
wade.rank(res)
wade.gene_detail(X[g], cond)
wade.normalize.tpm_like(counts, normalizer, lib_sizes, noise=0.01)
```

And on this side, `cfrna_wade_contrast()` becomes a thin adapter: resolve the
contrast to library ids, build or fetch the matrix, call `wade::wade()`, attach
the cohort/contrast metadata, cache. The screen and the registry do not change
at all.

## Cross-language parity: what will actually bite

These are the places the two implementations will silently disagree, listed
because a parity test suite should target them specifically:

1. **Quantile definition.** `stats::quantile` / `matrixStats::rowQuantiles`
   default to type 7; `numpy.quantile` defaults to linear interpolation, which
   *is* type 7 — so the defaults agree. But R offers nine types and several
   published quantile-based DE methods use type 1 or 2, so the new repo should
   pin and test the convention explicitly rather than inherit it. Every
   WADE statistic is a function of these quantiles, so a type mismatch changes
   every number without erroring.
2. **The RNG.** The jitter and the label permutations use R's Mersenne-Twister
   through `set.seed()`; NumPy's `default_rng` is PCG64. Bit-identical results
   across languages are **not achievable** with a shared seed. Parity tests
   must therefore either (a) supply the permutation index matrix and the jitter
   matrix as inputs, or (b) compare distributions rather than values. Option
   (a) makes exact cross-language tests possible and is worth designing for.
3. **BH-FDR.** `p.adjust(method="BH")` and `statsmodels`
   `multipletests(method="fdr_bh")` agree, but scipy's `false_discovery_control`
   has a different signature. Pin one.
4. **`tail.conc`'s NA guard is ~9 orders of magnitude too loose, and this is a
   real bug to fix in the port rather than a parity concern.** `wade()` guards
   the ratio at `abs(diff.mean * nprobs) < 1e-8`, which on the primary contrast
   (nprobs = 22) means `|diff.mean| < 5e-10`. Measured on that contrast, **120
   of 2,219 genes (5.4%) return `|tail.conc| > 2`**, the largest being 149 — a
   "share of the signed area" of 149 is not a share of anything. The cause is
   structural, not numerical: `tail.conc = sum(D[tail]) / sum(D)` and the
   denominator vanishes whenever the lower quantiles' differences cancel the
   upper ones, which happens at `diff.mean` values that are small but nowhere
   near 1e-10.

   Checked: no *nominated* gene in any of the 18 runs is affected, so no
   nomination in this notebook depends on it — but that is luck, not
   construction, and any threshold on `tail.conc` (the volcano's subset/bulk
   split is one) sits directly on top of it. The notebook guards at the display
   layer (`wade_tail_conc_display()`, ceiling 1.5) rather than changing the
   statistic, because editing `wade()` would invalidate every cached result for
   a column that is descriptive. **The new repo should fix it properly**: guard
   on the ratio's magnitude, or report `tail.conc` only when
   `sum(|D|)` and `|sum(D)|` are within a stated factor. Note that values
   slightly above 1 are legitimate (the tail can carry more than the total when
   the bulk partially cancels), so the fix is not a clamp to [0, 1].
5. **The GPD moment fit.** `.gpd_tail_p()` branches on `xi <= 0` and floors at
   `1/(B·n_tail)`. Both branches and the floor need explicit tests; the floor
   in particular is a deliberate honesty constraint, not a numerical
   convenience, and a port that "improves" it by returning smaller p-values
   would be a regression.

## Suggested repo shape

```
wade/
  README.md            # the statistic, the two literatures it joins, when NOT to use it
  python/wade/         # wade.py, rank.py, normalize.py, gpd.py
  R/                   # DESCRIPTION, NAMESPACE, R/wade.R ...
  tests/
    fixtures/          # small matrices + FIXED permutation and jitter matrices
    test_parity.py     # R vs Python on the fixtures, exact
    test_calibration.* # null p-values uniform; the v7 §9.6 simulation
  docs/
```

The calibration and power simulations currently inline in v7 §9.6 (null
uniformity by KS, subset-vs-bulk discrimination, power against the
combinatorial permutation floor) are the natural seed for `tests/` — they are
method validation, they do not mention cfRNA, and they are the strongest
evidence the method behaves as claimed. Moving them out of a notebook and into
a test suite is most of the argument for doing this extraction at all.

## Recommended sequence

1. Port `wade_stats()` + `wade()` + `wade_perm_pvalues()` + `.gpd_tail_p()` to
   Python against fixed-permutation fixtures; get exact parity with R.
2. Move the v7 §9.6 simulations into `tests/`.
3. Decide the normalization boundary (recommendation above: matrix in, optional
   convenience layer).
4. Only then switch this notebook to the external package, leaving
   `R/wade_contrasts.R` and `R/control_strata.R` untouched.
