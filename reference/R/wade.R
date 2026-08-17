# =====================================================================
# WADE — WAsserstein Area Differential Expression
# ---------------------------------------------------------------------
# A quantile-area differential-distribution test built for SUBSET
# detection: it compares a heterogeneous "case" group against a
# "control" group without assuming the two are drawn from the same
# distribution, and it is designed to surface genes that are altered in
# only a fraction of cases (e.g. a rare cancer subtype).
#
# Statistical basis. For each gene, WADE compares the two groups'
# empirical quantile functions Q_case(p), Q_ctrl(p) on a common grid of
# min(n0, n1) probabilities. The integral of their difference is the
# signed area
#     diff.mean = \int_0^1 [Q_case(p) - Q_ctrl(p)] dp  =  mu_case - mu_ctrl,
# i.e. the first moment (mean difference); the integral of the ABSOLUTE
# difference is the 1-Wasserstein distance W1. diff.mean alone collapses
# a rare-subset signal and a whole-group shift to the same number, so
# WADE also reports SHAPE statistics that read the tail of the quantile
# difference, where a rare high-expressing subset concentrates.
#
# This connects two literatures: the quantile/Wasserstein
# differential-distribution test (cf. scDD, waddR) supplies the
# mechanism; cancer-outlier profile analysis (COPA, OS, ORT, MOST,
# LSOSS) supplies the subset-detection goal.
#
# Prototype for a future standalone package (CRAN/PyPI). Dependencies:
# matrixStats (fast row quantiles), and tibble for the result frame.
# =====================================================================

suppressPackageStartupMessages({
  library(matrixStats)
})

# ---------------------------------------------------------------------
# Normalization: normalizer- and library-scaled TPM-like units.
#
# WADE takes a COUNT measure and a NORMALIZER measure per gene:
#   * splice-junction counts  normalized by number of introns  -> "sjTPM"
#   * total (rigel) counts     normalized by effective length    -> standard TPM
# The math is identical; only the pair changes. Each gene's count is
# divided by its normalizer, then by a per-library size factor, x 1e6.
#
# The `normalizer` argument may be EITHER:
#   * a length-`nrow(counts)` vector  (per-gene; e.g. intron count),
#     recycled down the rows of the count matrix, OR
#   * a genes x samples MATRIX        (per-gene-per-sample; e.g. rigel
#     effective length, which varies with each library's fragment-length
#     distribution). Matrix division is element-wise, so counts/efflen
#     scaled to library size reproduces standard TPM exactly.
#
# A small uniform continuity jitter breaks ties in sparse/zero-heavy data;
# it is applied ONCE at the count-precision level with a fixed seed so the
# whole analysis is reproducible (permutations then shuffle labels on this
# fixed matrix — conditional inference).
#   counts     : genes x samples matrix of raw counts
#   normalizer : per-gene vector OR genes x samples matrix (see above)
#   lib_sizes  : per-sample size vector (sum_g counts[g,]/normalizer[g,])
# ---------------------------------------------------------------------
wade_lib_size <- function(counts, normalizer) {
  colSums(as.matrix(counts) / normalizer)
}

wade_normalize <- function(counts, normalizer, lib_sizes,
                           noise = 0.01, norm_factor = 1e6, seed = 1L) {
  counts <- as.matrix(counts)
  g <- nrow(counts); n <- ncol(counts)
  if (!is.null(seed)) set.seed(seed)
  nz    <- matrix(stats::runif(g * n, 0, noise), g, n)
  y     <- (counts + nz) / normalizer                 # per-gene (vec) or element-wise (mat)
  denom <- sweep(nz / normalizer, 2, lib_sizes, "+")  # + lib_sizes[j] per column
  norm_factor * y / denom
}

# ---------------------------------------------------------------------
# Core statistics from a normalized matrix. n0, n1 are shared by all
# genes, so the quantile grid q is computed once. Quantiles are ordered
# high -> low, so the FIRST `k` columns are the upper tail where a rare
# high subset lives.
#   tail_q : upper-tail fraction defining the "subset" window (0.10)
#   weight : >1 upweights CONTROLS (stringency), as in the original
# Returns per-gene vectors plus the full quantile grids (for plotting).
# ---------------------------------------------------------------------
wade_stats <- function(tpm, cond, tail_q = 0.10, log2_scale = FALSE, weight = 1) {
  cond <- as.integer(cond)
  i1 <- which(cond == 1L); i0 <- which(cond == 0L)
  n1 <- length(i1); n0 <- length(i0)
  stopifnot(n1 > 0, n0 > 0)
  X <- tpm
  if (weight != 1) X <- sweep(X, 2, ifelse(cond == 0L, weight, 1), "*")
  if (log2_scale)  X <- log2(X + 1)
  nprobs <- min(n0, n1)
  q  <- seq(1, 0, length.out = nprobs)                 # high -> low
  Q1 <- matrixStats::rowQuantiles(X[, i1, drop = FALSE], probs = q, useNames = FALSE)
  Q0 <- matrixStats::rowQuantiles(X[, i0, drop = FALSE], probs = q, useNames = FALSE)
  if (is.null(dim(Q1))) { Q1 <- matrix(Q1, nrow = 1); Q0 <- matrix(Q0, nrow = 1) }
  D  <- Q1 - Q0
  k  <- max(1L, ceiling(tail_q * nprobs))              # upper-tail window
  s1 <- rowSums(Q1); s0 <- rowSums(Q0); sD <- rowSums(D)
  list(
    q = q, nprobs = nprobs, k = k, n1 = n1, n0 = n0, Q1 = Q1, Q0 = Q0, D = D,
    diff.mean = sD / nprobs,                            # = mu_case - mu_ctrl
    w1        = rowMeans(abs(D)),                       # 1-Wasserstein distance
    tail.mean = rowMeans(D[, 1:k, drop = FALSE]),       # mean upper-tail difference (SUBSET axis)
    tail.conc = rowSums(D[, 1:k, drop = FALSE]) / sD,   # share of signed area in the tail
    fc         = s1 / s0,
    cond1.mean = s1 / nprobs,
    cond0.mean = s0 / nprobs,
    tot.mean   = (s1 + s0) / nprobs
  )
}

# ---------------------------------------------------------------------
# Lean null statistics for the permutation loop: returns ONLY the two
# quantities that build the null (diff.mean, tail.mean), skipping the
# derived means / fold-change / tibble in wade_stats. weight/log2_scale
# are not supported here (the driver falls back to full wade_stats when
# either is non-default), so the hot path stays a single pair of
# rowQuantiles calls. Identical numerics to wade_stats on the defaults.
# ---------------------------------------------------------------------
.wade_null_stats <- function(tpm, cond, nprobs, q, k) {
  i1 <- which(cond == 1L); i0 <- which(cond == 0L)
  Q1 <- matrixStats::rowQuantiles(tpm[, i1, drop = FALSE], probs = q, useNames = FALSE)
  Q0 <- matrixStats::rowQuantiles(tpm[, i0, drop = FALSE], probs = q, useNames = FALSE)
  if (is.null(dim(Q1))) { Q1 <- matrix(Q1, nrow = 1); Q0 <- matrix(Q0, nrow = 1) }
  D <- Q1 - Q0
  list(diff.mean = rowSums(D) / nprobs,
       tail.mean = rowMeans(D[, 1:k, drop = FALSE]))
}

# ---------------------------------------------------------------------
# Permutation p-values with a Generalized Pareto (GPD) tail refinement
# for small p (Knijnenburg et al. 2009). The empirical p is
# (1 + #{null >= obs}) / (B + 1); when exceedances are too few to
# resolve, the upper tail of the permutation null is fit by a GPD (here
# by method-of-moments, closed-form and dependency-free) and p is read
# from its CDF. When the moment fit gives a non-positive shape (xi <= 0,
# a light/bounded tail), the xi -> 0 EXPONENTIAL limit is used instead of
# the GPD's hard upper bound, so a strong observed statistic never
# collapses to a machine-epsilon p-value. p is floored at 1/(B * n_tail),
# the credible resolution limit for B permutations. MLE-based GPD is the
# production upgrade.
# ---------------------------------------------------------------------
.gpd_tail_p <- function(o, null, n_tail = 250) {
  B <- length(null)
  n_tail <- min(n_tail, floor(B / 2))
  s   <- sort(null, decreasing = TRUE)
  thr <- s[n_tail + 1]
  exc <- s[s > thr] - thr
  emp <- (1 + sum(null >= o)) / (B + 1)
  if (length(exc) < 10 || o <= thr) return(emp)
  m <- mean(exc); v <- stats::var(exc)
  if (!is.finite(v) || v <= 0) return(emp)
  y <- o - thr
  # Credible extrapolation floor: with B permutations and n_tail tail points,
  # the smallest defensible tail probability is ~1/(B * n_tail). Never return
  # machine-epsilon p-values — 2000 permutations cannot support p < ~1e-6, and
  # a degenerate GPD bound collapsing to .Machine$double.eps creates false ties
  # and mis-orders strong hits. This floor is honest about the resolution limit.
  p_floor <- 1 / (B * n_tail)
  xi    <- 0.5 * (1 - m^2 / v)
  sigma <- 0.5 * m * (1 + m^2 / v)
  if (!is.finite(sigma) || sigma <= 0) return(emp)
  if (xi <= 0) {
    # Light / bounded tail (mom shape non-positive): use the xi -> 0 exponential
    # limit, which is well-defined for all y >= 0 and appropriately conservative
    # rather than the GPD's hard upper bound at -sigma/xi.
    tail_prob <- exp(-y / sigma)
  } else {
    tail_prob <- (1 + xi * y / sigma)^(-1 / xi)
  }
  max((n_tail / B) * tail_prob, p_floor)
}

wade_perm_pvalues <- function(obs, perm, n_exc_min = 10, n_tail = 250) {
  B <- ncol(perm)
  nexc <- rowSums(perm >= obs)                          # obs recycled per row
  p <- (1 + nexc) / (B + 1)
  refine <- which(nexc < n_exc_min & B >= 2 * n_tail)
  for (i in refine) p[i] <- .gpd_tail_p(obs[i], perm[i, ], n_tail = n_tail)
  p
}

# ---------------------------------------------------------------------
# Driver: run WADE across all genes and return a tidy result frame.
# Takes a (count, normalizer) pair — see wade_normalize() for the two
# supported forms (per-gene vector, e.g. introns; or genes x samples
# matrix, e.g. rigel effective length). Reports the bulk axis (diff.mean,
# fc) and the subset axis (tail.mean, tail.conc, w1), permutation
# p-values and BH-FDR for both. Permuting labels on the fixed normalized
# matrix keeps the noise draw out of the null, so results are
# reproducible for a given seed.
# ---------------------------------------------------------------------
wade <- function(counts, normalizer, lib_sizes, cond,
                 nperms = 1000, tail_q = 0.10, noise = 0.01,
                 log2_scale = FALSE, weight = 1, seed = 1L,
                 gene_names = rownames(counts), verbose = TRUE) {
  tpm <- wade_normalize(counts, normalizer, lib_sizes, noise = noise, seed = seed)
  obs <- wade_stats(tpm, cond, tail_q = tail_q, log2_scale = log2_scale, weight = weight)
  g <- nrow(tpm)
  if (nperms > 0) {
    perm_dm <- matrix(NA_real_, g, nperms)
    perm_tm <- matrix(NA_real_, g, nperms)
    if (!is.null(seed)) set.seed(seed + 1L)
    lean <- (weight == 1 && !log2_scale)               # hot path when defaults
    for (b in seq_len(nperms)) {
      if (lean) {
        st <- .wade_null_stats(tpm, sample(cond), obs$nprobs, obs$q, obs$k)
      } else {
        st <- wade_stats(tpm, sample(cond), tail_q = tail_q,
                         log2_scale = log2_scale, weight = weight)
      }
      perm_dm[, b] <- st$diff.mean
      perm_tm[, b] <- st$tail.mean
      if (verbose && b %% 500 == 0) cat("  perm", b, "/", nperms, "\n")
    }
    p.diff <- wade_perm_pvalues(obs$diff.mean, perm_dm)
    p.tail <- wade_perm_pvalues(obs$tail.mean, perm_tm)
  } else {
    p.diff <- rep(NA_real_, g); p.tail <- rep(NA_real_, g)
  }
  tibble::tibble(
    gene       = gene_names,
    diff.mean  = obs$diff.mean,
    diff.frac  = obs$diff.mean / obs$tot.mean,
    w1         = obs$w1,
    tail.mean  = obs$tail.mean,
    tail.conc  = ifelse(abs(obs$diff.mean * obs$nprobs) < 1e-8, NA_real_, obs$tail.conc),
    fc         = obs$fc,
    cond1.mean = obs$cond1.mean,
    cond0.mean = obs$cond0.mean,
    tot.mean   = obs$tot.mean,
    p.diff     = p.diff,
    p.tail     = p.tail,
    padj.diff  = stats::p.adjust(p.diff, "BH"),
    padj.tail  = stats::p.adjust(p.tail, "BH")
  )
}

# ---------------------------------------------------------------------
# Scoring / nomination. Two complementary rank scores on a wade() frame:
#
#   score       — the ORIGINAL panel-selection score, preserved for
#                 continuity. A signed rank-product rewarding genes that
#                 are high in cases, LOW in controls, and large fold-
#                 change: sign(diff.frac) * F(|log2fc|) * F(case.mean) *
#                 (1 - F(ctrl.mean)), where F is the empirical CDF of
#                 each quantity across genes. Detects the BULK case-vs-
#                 control axis.
#   tail.score  — a SUBSET-aware score that replaces the bulk fold-change
#                 term with the tail statistic F(tail.mean): sign(tail.mean)
#                 * F(tail.mean) * F(case.mean) * (1 - F(ctrl.mean)).
#                 Surfaces genes elevated in only a fraction of cases,
#                 which `score` (a first-moment quantity) can miss.
#
# Both are in [-1, 1]. exp arguments let a user tune term emphasis.
# ---------------------------------------------------------------------
wade_score <- function(df, log2fc_exp = 1, case_exp = 1, ctrl_exp = 1, tail_exp = 1) {
  Ffc   <- stats::ecdf(abs(log2(df$fc)))
  Fcase <- stats::ecdf(df$cond1.mean)
  Fctrl <- stats::ecdf(df$cond0.mean)
  Ftail <- stats::ecdf(df$tail.mean)
  df$log2fc     <- log2(df$fc)
  df$score      <- sign(df$diff.frac) *
                   Ffc(abs(df$log2fc))^log2fc_exp *
                   Fcase(df$cond1.mean)^case_exp *
                   (1 - Fctrl(df$cond0.mean))^ctrl_exp
  df$tail.score <- sign(df$tail.mean) *
                   Ftail(df$tail.mean)^tail_exp *
                   Fcase(df$cond1.mean)^case_exp *
                   (1 - Fctrl(df$cond0.mean))^ctrl_exp
  df$rank       <- dplyr::dense_rank(dplyr::desc(df$score))
  df$tail.rank  <- dplyr::dense_rank(dplyr::desc(df$tail.score))
  df
}

# ---------------------------------------------------------------------
# Convenience wrapper: run a full WADE contrast from a (count, normalizer)
# matrix pair and two library-id vectors (case, control). Handles column
# sub-setting, condition coding (1 = case, 0 = control), library sizing,
# the permutation test, and rank scoring in one call, and records the
# group sizes as attributes. Returns the scored data frame.
# ---------------------------------------------------------------------
wade_run <- function(counts, normalizer, case_libs, ctrl_libs,
                     gene_names = NULL, nperms = 2000, tail_q = 0.10,
                     seed = 1L, verbose = FALSE, ...) {
  libs <- c(case_libs, ctrl_libs)
  cond <- c(rep(1L, length(case_libs)), rep(0L, length(ctrl_libs)))
  Xc   <- counts[, libs, drop = FALSE]
  Xn   <- normalizer[, libs, drop = FALSE]
  gn   <- if (is.null(gene_names)) rownames(counts) else gene_names
  ls   <- wade_lib_size(Xc, Xn)
  res  <- wade(Xc, Xn, ls, cond, nperms = nperms, tail_q = tail_q,
               seed = seed, gene_names = gn, verbose = verbose, ...) |>
          wade_score()
  attr(res, "n_case") <- length(case_libs)
  attr(res, "n_ctrl") <- length(ctrl_libs)
  res
}

# ---------------------------------------------------------------------
# Single-gene detail for the diagnostic plot: returns the quantile grid
# and cumulative signed-area curve alongside the scalar summaries.
# ---------------------------------------------------------------------
wade_gene <- function(tpm_row, cond, tail_q = 0.10, log2_scale = FALSE, weight = 1) {
  st <- wade_stats(matrix(tpm_row, nrow = 1), cond,
                   tail_q = tail_q, log2_scale = log2_scale, weight = weight)
  # reverse to ascending p (0 -> 1) FIRST, then accumulate, so that the
  # cumulative curve reads left-to-right and its endpoint at p=1 equals
  # diff.mean (the total signed area). A bulk shift rises steadily; a
  # rare high subset stays flat then jumps inside the top-tail window.
  y1 <- rev(as.numeric(st$Q1)); y0 <- rev(as.numeric(st$Q0))
  data.frame(
    p    = rev(st$q),                         # ascending 0 -> 1
    y1   = y1, y0 = y0,
    cum  = cumsum(y1 - y0) / st$nprobs        # cumulative signed area, endpoint = diff.mean
  )
}
