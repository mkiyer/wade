# =====================================================================
# generate_fixtures.R — golden fixtures from the R reference
# ---------------------------------------------------------------------
# Run from reference/R/ (the renv project root):
#
#   export PATH="/usr/local/bin:$PATH"
#   cd reference/R
#   RENV_CONFIG_SANDBOX_ENABLED=FALSE \
#   RENV_PATHS_CACHE="$HOME/Library/Caches/org.R-project.R/R/renv/cache" \
#     Rscript ../../tools/r/generate_fixtures.R
#
# Emits tests/fixtures/*.json. Every double is written at 17 significant
# digits and round-trip-verified before the file is written.
#
# WHAT IS EMITTED, AND WHY INTERMEDIATES
# --------------------------------------
# docs/implementation-notes.md: "intermediate quantities localize a
# disagreement; endpoint quantities only detect one." So each scenario
# carries the whole computation, not just the result frame:
#
#   inputs      counts, normalizer (vector or matrix form), cond, params
#   injected    the jitter matrix and the permutation label matrix
#   layer 1     nprobs, k, n1, n0, and the probability grid q
#   layer 2     lib_sizes and the normalized matrix tpm
#   layer 3     the Q1, Q0 and D grids, in full
#   layer 4     every per-gene reduction, plus tail.conc's NUMERATOR and
#               DENOMINATOR separately (hazard 5: the ratio itself is a
#               place a correct port must disagree with the R)
#   layer 5     the full g x nperms null matrices for BOTH axes
#   layer 6     exceedance counts, empirical p, which genes were refined,
#               and the GPD internals (thr, n_exc, m, v, xi, sigma,
#               branch) for each refined gene
#   layer 7     the result frame as wade() actually returns it
#   layer 8     the wade_score() columns
#   diagnostic  wade_gene() output for gene 1
#
# INTEGRITY CHECKS RUN ON EVERY SCENARIO
# --------------------------------------
# The intermediates are produced by replicating wade()'s body step by
# step. That replication is then checked against a real wade() call under
# the same injection, column by column, with identical(). If they ever
# disagree the emitted intermediates would not belong to the emitted
# endpoints, so the generator aborts rather than writing a fixture that
# quietly describes two different computations.
# =====================================================================

.here <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
if (is.na(.here) || !nzchar(.here)) .here <- "../../tools/r"
source(file.path(.here, "json_emit.R"))
source(file.path(.here, "wade_seam.R"))

WADE_R  <- normalizePath("wade.R", mustWork = TRUE)
OUT_DIR <- normalizePath(file.path(.here, "..", "..", "tests", "fixtures"), mustWork = FALSE)
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

cat("wade.R :", WADE_R, "\n")
cat("output :", OUT_DIR, "\n\n")

seam_selftest(WADE_R)
cat("\n")

env <- wade_seam_env(WADE_R)

# ---------------------------------------------------------------------
# .gpd_tail_p, instrumented.
#
# Mirrors wade.R lines 143-172 exactly and additionally reports the
# internals a port needs to test branch by branch. Its returned p is
# asserted equal to the real .gpd_tail_p() on every call, so the mirror
# cannot drift from the reference without the generator failing.
# ---------------------------------------------------------------------
gpd_introspect <- function(o, null, n_tail = 250) {
  B <- length(null)
  n_tail_c <- min(n_tail, floor(B / 2))
  s   <- sort(null, decreasing = TRUE)
  thr <- s[n_tail_c + 1]
  exc <- s[s > thr] - thr
  emp <- (1 + sum(null >= o)) / (B + 1)
  out <- list(B = B, n_tail_requested = n_tail, n_tail_used = n_tail_c,
              thr = thr, n_exc = length(exc), emp = emp,
              m = NA_real_, v = NA_real_, xi = NA_real_, sigma = NA_real_,
              y = NA_real_, tail_prob = NA_real_,
              p_floor = 1 / (B * n_tail_c), branch = NA_character_, p = NA_real_)
  if (length(exc) < 10 || o <= thr) {
    out$branch <- if (length(exc) < 10) "bail_few_exceedances" else "bail_obs_at_or_below_thr"
    out$p <- emp
    return(out)
  }
  m <- mean(exc); v <- stats::var(exc)
  out$m <- m; out$v <- v
  if (!is.finite(v) || v <= 0) {
    out$branch <- "bail_degenerate_variance"; out$p <- emp; return(out)
  }
  y <- o - thr
  out$y <- y
  xi    <- 0.5 * (1 - m^2 / v)
  sigma <- 0.5 * m * (1 + m^2 / v)
  out$xi <- xi; out$sigma <- sigma
  if (!is.finite(sigma) || sigma <= 0) {
    out$branch <- "bail_degenerate_sigma"; out$p <- emp; return(out)
  }
  tail_prob <- if (xi <= 0) exp(-y / sigma) else (1 + xi * y / sigma)^(-1 / xi)
  out$tail_prob <- tail_prob
  scaled <- (n_tail_c / B) * tail_prob
  out$p <- max(scaled, out$p_floor)
  out$branch <- if (out$p == out$p_floor && scaled < out$p_floor) {
    "floor"
  } else if (xi <= 0) "exponential" else "gpd"
  out
}

j_gpd <- function(gi) {
  j_obj(
    B = j_int(gi$B), n_tail_requested = j_int(gi$n_tail_requested),
    n_tail_used = j_int(gi$n_tail_used), n_exc = j_int(gi$n_exc),
    thr = j_dbl(gi$thr, "thr"), emp = j_dbl(gi$emp, "emp"),
    m = j_dbl(gi$m, "m"), v = j_dbl(gi$v, "v"),
    xi = j_dbl(gi$xi, "xi"), sigma = j_dbl(gi$sigma, "sigma"),
    y = j_dbl(gi$y, "y"), tail_prob = j_dbl(gi$tail_prob, "tail_prob"),
    p_floor = j_dbl(gi$p_floor, "p_floor"),
    branch = .j_str(gi$branch), p = j_dbl(gi$p, "p")
  )
}

# ---------------------------------------------------------------------
# One scenario: replicate wade() step by step under injection, verify the
# replication against a real wade() call, and emit everything.
# ---------------------------------------------------------------------
emit_scenario <- function(name, description, counts, normalizer, cond,
                          nperms, tail_q = 0.10, noise = 0.01, seed = 1L,
                          weight = 1, log2_scale = FALSE,
                          gene_names = rownames(counts), notes = "") {
  cat(sprintf("  %-16s ", name)); flush(stdout())
  g <- nrow(counts); n <- ncol(counts)
  norm_is_matrix <- is.matrix(normalizer)

  J <- seam_native_jitter(g, n, noise = noise, seed = seed)
  P <- if (nperms > 0) seam_native_perms(cond, nperms, seed = seed) else
       matrix(integer(0), nrow = 0, ncol = n)

  lib_sizes <- env$wade_lib_size(counts, normalizer)

  # --- replicate wade()'s body -------------------------------------
  seam_arm(jitter = J, perms = P)
  tpm <- env$wade_normalize(counts, normalizer, lib_sizes, noise = noise, seed = seed)
  seam_disarm()

  obs <- env$wade_stats(tpm, cond, tail_q = tail_q,
                        log2_scale = log2_scale, weight = weight)
  lean <- (weight == 1 && !log2_scale)

  perm_dm <- matrix(NA_real_, g, max(nperms, 1L))
  perm_tm <- matrix(NA_real_, g, max(nperms, 1L))
  if (nperms > 0) {
    perm_dm <- matrix(NA_real_, g, nperms)
    perm_tm <- matrix(NA_real_, g, nperms)
    for (b in seq_len(nperms)) {
      cb <- P[b, ]
      st <- if (lean) env$.wade_null_stats(tpm, cb, obs$nprobs, obs$q, obs$k)
            else       env$wade_stats(tpm, cb, tail_q = tail_q,
                                      log2_scale = log2_scale, weight = weight)
      perm_dm[, b] <- st$diff.mean
      perm_tm[, b] <- st$tail.mean
    }
    nexc_dm <- rowSums(perm_dm >= obs$diff.mean)
    nexc_tm <- rowSums(perm_tm >= obs$tail.mean)
    p_diff  <- env$wade_perm_pvalues(obs$diff.mean, perm_dm)
    p_tail  <- env$wade_perm_pvalues(obs$tail.mean, perm_tm)
    refine_dm <- which(nexc_dm < 10 & nperms >= 2 * 250)
    refine_tm <- which(nexc_tm < 10 & nperms >= 2 * 250)
  } else {
    nexc_dm <- integer(0); nexc_tm <- integer(0)
    p_diff <- rep(NA_real_, g); p_tail <- rep(NA_real_, g)
    refine_dm <- integer(0); refine_tm <- integer(0)
  }

  # --- the real wade(), same injection, as the integrity check ------
  seam_arm(jitter = J, perms = P)
  frame <- env$wade(counts, normalizer, lib_sizes, cond, nperms = nperms,
                    tail_q = tail_q, noise = noise, log2_scale = log2_scale,
                    weight = weight, seed = seed, gene_names = gene_names,
                    verbose = FALSE)
  fired <- seam_counts()
  seam_disarm()

  stopifnot(fired$runif_calls == 1L)
  stopifnot(fired$perm_calls == nperms)
  # rep_len is the identity when the statistic has one value per gene. It is
  # NOT the identity in the nprobs == 1 scenario, where wade_stats() returns a
  # single value for a many-gene input and tibble() recycles it up the frame —
  # that recycling IS the defect of hazard 11, so the check has to model it
  # rather than trip over it.
  for (nm in c("diff.mean", "w1", "tail.mean", "fc",
               "cond1.mean", "cond0.mean", "tot.mean")) {
    if (!identical(frame[[nm]], rep_len(as.numeric(obs[[nm]]), nrow(frame)))) {
      stop(sprintf("scenario %s: replicated %s differs from wade()'s", name, nm))
    }
  }
  if (nperms > 0) {
    if (!identical(frame$p.diff, p_diff) || !identical(frame$p.tail, p_tail)) {
      stop(sprintf("scenario %s: replicated p-values differ from wade()'s", name))
    }
  }

  scored <- env$wade_score(frame)
  gene1  <- env$wade_gene(tpm[1, ], cond, tail_q = tail_q,
                          log2_scale = log2_scale, weight = weight)

  # --- GPD internals for every refined gene ------------------------
  gpd_records <- character(0)
  if (length(refine_dm) > 0 || length(refine_tm) > 0) {
    mk <- function(axis, idx, obsv, perm) {
      vapply(idx, function(i) {
        gi <- gpd_introspect(obsv[i], perm[i, ], n_tail = 250)
        ref <- env$.gpd_tail_p(obsv[i], perm[i, ], n_tail = 250)
        if (!identical(gi$p, ref)) {
          stop(sprintf("scenario %s: gpd_introspect disagrees with .gpd_tail_p at %s gene %d",
                       name, axis, i))
        }
        j_obj(axis = .j_str(axis), gene_index0 = j_int(i - 1L), detail = j_gpd(gi))
      }, character(1))
    }
    gpd_records <- c(mk("diff", refine_dm, obs$diff.mean, perm_dm),
                     mk("tail", refine_tm, obs$tail.mean, perm_tm))
  }

  sD  <- rowSums(obs$D)
  num <- rowSums(obs$D[, seq_len(obs$k), drop = FALSE])

  json <- j_obj(
    name        = .j_str(name),
    description = .j_str(description),
    notes       = .j_str(notes),
    params = j_obj(
      nperms = j_int(nperms), tail_q = j_dbl(tail_q, "tail_q"),
      noise = j_dbl(noise, "noise"), norm_factor = j_dbl(1e6, "norm_factor"),
      seed = j_int(seed), weight = j_dbl(weight, "weight"),
      log2_scale = j_bool(log2_scale), lean_path = j_bool(lean),
      n_exc_min = j_int(10L), n_tail = j_int(250L)
    ),
    shapes = j_obj(
      g = j_int(g), n = j_int(n), nprobs = j_int(obs$nprobs), k = j_int(obs$k),
      n1 = j_int(obs$n1), n0 = j_int(obs$n0), B = j_int(nperms)
    ),
    gene_names = if (is.null(gene_names)) "null" else j_str_vec(gene_names),
    cond       = j_int_vec(cond),
    counts     = j_dbl_mat(counts, "counts"),
    normalizer = j_obj(
      form = .j_str(if (norm_is_matrix) "matrix" else "vector"),
      data = if (norm_is_matrix) j_dbl_mat(normalizer, "normalizer")
             else j_dbl_vec(normalizer, "normalizer")
    ),
    jitter = j_dbl_mat(J, "jitter"),
    perms  = j_int_mat(P),
    lib_sizes = j_dbl_vec(lib_sizes, "lib_sizes"),
    tpm       = j_dbl_mat(tpm, "tpm"),
    q  = j_dbl_vec(obs$q, "q"),
    Q1 = j_dbl_mat(obs$Q1, "Q1"),
    Q0 = j_dbl_mat(obs$Q0, "Q0"),
    D  = j_dbl_mat(obs$D,  "D"),
    stats = j_obj(
      diff.mean     = j_dbl_vec(obs$diff.mean, "diff.mean"),
      w1            = j_dbl_vec(obs$w1, "w1"),
      tail.mean     = j_dbl_vec(obs$tail.mean, "tail.mean"),
      tail_num      = j_dbl_vec(num, "tail_num"),
      tail_den      = j_dbl_vec(sD,  "tail_den"),
      tail.conc_raw = j_dbl_vec(obs$tail.conc, "tail.conc_raw"),
      fc            = j_dbl_vec(obs$fc, "fc"),
      cond1.mean    = j_dbl_vec(obs$cond1.mean, "cond1.mean"),
      cond0.mean    = j_dbl_vec(obs$cond0.mean, "cond0.mean"),
      tot.mean      = j_dbl_vec(obs$tot.mean, "tot.mean")
    ),
    null = j_obj(
      perm_dm = if (nperms > 0) j_dbl_mat(perm_dm, "perm_dm") else "null",
      perm_tm = if (nperms > 0) j_dbl_mat(perm_tm, "perm_tm") else "null"
    ),
    pvalues = j_obj(
      nexc_diff        = j_int_vec(nexc_dm),
      nexc_tail        = j_int_vec(nexc_tm),
      p_diff           = j_dbl_vec(p_diff, "p_diff"),
      p_tail           = j_dbl_vec(p_tail, "p_tail"),
      refined_diff_i0  = j_int_vec(refine_dm - 1L),
      refined_tail_i0  = j_int_vec(refine_tm - 1L),
      gpd_details      = paste0("[", paste0(gpd_records, collapse = ","), "]")
    ),
    frame = j_obj(
      columns        = j_str_vec(names(frame)),
      has_gene_col   = j_bool("gene" %in% names(frame)),
      nrow           = j_int(nrow(frame)),
      diff.frac      = j_dbl_vec(frame$diff.frac, "diff.frac"),
      tail.conc      = j_dbl_vec(frame$tail.conc, "tail.conc"),
      padj.diff      = j_dbl_vec(frame$padj.diff, "padj.diff"),
      padj.tail      = j_dbl_vec(frame$padj.tail, "padj.tail")
    ),
    score = j_obj(
      columns    = j_str_vec(names(scored)),
      log2fc     = j_dbl_vec(scored$log2fc, "log2fc"),
      score      = j_dbl_vec(scored$score, "score"),
      tail.score = j_dbl_vec(scored$tail.score, "tail.score"),
      rank       = j_int_vec(scored$rank),
      tail.rank  = j_int_vec(scored$tail.rank)
    ),
    wade_gene = j_obj(
      gene_index0 = j_int(0L),
      p   = j_dbl_vec(gene1$p, "wade_gene.p"),
      y1  = j_dbl_vec(gene1$y1, "wade_gene.y1"),
      y0  = j_dbl_vec(gene1$y0, "wade_gene.y0"),
      cum = j_dbl_vec(gene1$cum, "wade_gene.cum")
    )
  )
  j_write(json, file.path(OUT_DIR, paste0(name, ".json")))
  cat(sprintf("g=%-4d n=%-3d m=%-3d k=%-2d B=%-4d  ok\n", g, n, obs$nprobs, obs$k, nperms))
  invisible(NULL)
}

# ---------------------------------------------------------------------
# Scenario construction helpers. All synthetic; nothing patient-derived.
# ---------------------------------------------------------------------
mk_counts <- function(g, n, lambda = 20, seed) {
  set.seed(seed)
  m <- matrix(rpois(g * n, lambda), g, n)
  storage.mode(m) <- "double"
  m
}
mk_norm_mat <- function(g, n, seed) {
  set.seed(seed + 7777L)
  matrix(runif(g * n, 0.5, 8), g, n)
}
mk_norm_vec <- function(g, seed) {
  set.seed(seed + 13131L)
  as.numeric(sample.int(12, g, replace = TRUE))
}

cat("emitting scenarios\n")

# --- 1. tiny: hand-checkable. 6 x 7, 3 cases vs 4 controls -> m=3, k=1
#     q = (1, 0.5, 0). Non-square in g vs n and g vs B. Vector normalizer.
{
  counts <- mk_counts(6, 7, 8, 101)
  counts[1, ] <- c(0, 0, 3, 3, 3, 7, 11)          # ties and zeros
  counts[2, ] <- c(0, 0, 0, 1, 2, 2, 5)
  rownames(counts) <- paste0("g", 1:6)
  emit_scenario("tiny",
    "6 genes x 7 samples, 3 cases vs 4 controls: nprobs=3, q=(1,0.5,0), k=1. Small enough to check by hand. Vector normalizer.",
    counts, mk_norm_vec(6, 101), c(1L,1L,1L,0L,0L,0L,0L), nperms = 11L,
    notes = "Non-square in genes vs samples (6 vs 7) and genes vs permutations (6 vs 11).")
}

# --- 2. main: the workhorse. 40 x 27, 15 cases vs 12 controls -> m=12, k=2
{
  counts <- mk_counts(40, 27, 25, 202)
  counts[3, ] <- 0                                  # all-zero gene
  counts[4, 5:14] <- 0                              # zero run
  counts[5, ] <- rep(c(0, 0, 4, 4, 4, 9, 17, 17, 30), 3)  # heavy ties
  counts[6, ] <- 12                                 # constant gene -> tail.conc NaN
  rownames(counts) <- paste0("gene", sprintf("%02d", 1:40))
  emit_scenario("main",
    "40 genes x 27 samples, 15 cases (odd) vs 12 controls (even): nprobs=12, k=2. Matrix normalizer. Contains an all-zero gene, a zero run, a heavy-ties gene and a constant gene.",
    counts, mk_norm_mat(40, 27, 202), c(rep(1L,15), rep(0L,12)), nperms = 37L,
    notes = "Non-square everywhere: 40 genes, 27 samples, 37 permutations, 12 probabilities.")
}

# --- 3. even_larger: larger group has EVEN size (hazard 1 needs both parities)
{
  counts <- mk_counts(13, 19, 15, 303)
  counts[2, ] <- c(rep(0, 6), rep(5, 7), 5, 9, 9, 14, 22, 40)
  rownames(counts) <- paste0("G", 1:13)
  emit_scenario("even_larger",
    "13 genes x 19 samples, 7 cases (odd, smaller) vs 12 controls (even, larger): nprobs=7, k=1. The interpolated group has even length.",
    counts, mk_norm_mat(13, 19, 303), c(rep(1L,7), rep(0L,12)), nperms = 23L,
    notes = "Complements 'main', where the larger group is odd-sized.")
}

# --- 4. vecnorm: the per-gene vector normalizer form, with zeros
{
  counts <- mk_counts(17, 15, 10, 404)
  counts[c(1, 9), ] <- 0
  counts[2, 1:8] <- 0
  rownames(counts) <- paste0("v", 1:17)
  emit_scenario("vecnorm",
    "17 genes x 15 samples, 8 cases vs 7 controls: nprobs=7, k=1. Per-gene VECTOR normalizer (the second supported form), zero-heavy counts.",
    counts, mk_norm_vec(17, 404), c(rep(1L,8), rep(0L,7)), nperms = 29L)
}

# --- 5. m2: the nprobs == 2 degenerate case, grid = {1, 0}
{
  counts <- mk_counts(9, 8, 18, 505)
  rownames(counts) <- paste0("d", 1:9)
  emit_scenario("m2",
    "9 genes x 8 samples, 6 cases vs 2 controls: nprobs=2, grid={1,0}, k=1. diff.mean is the difference of mid-ranges.",
    counts, mk_norm_mat(9, 8, 505), c(rep(1L,6), rep(0L,2)), nperms = 13L)
}

# --- 6. onesample: nprobs == 1. The R's reshape defect lives here.
{
  counts <- mk_counts(5, 4, 22, 606)
  rownames(counts) <- paste0("s", 1:5)
  emit_scenario("onesample",
    "5 genes x 4 samples, 1 case vs 3 controls: nprobs=1. THE R IS WRONG HERE — rowQuantiles drops to a length-5 vector, the guard reshapes it to 1x5, and wade() recycles a single value across all five genes. A correct port MUST disagree. See docs/implementation-notes.md hazard 11.",
    counts, mk_norm_mat(5, 4, 606), c(1L, 0L, 0L, 0L), nperms = 7L,
    notes = "DO NOT assert parity on the statistics here. This fixture records a defect.")
}

# --- 7. nperms0: statistics only, all p-values NA
{
  counts <- mk_counts(8, 11, 14, 707)
  rownames(counts) <- paste0("z", 1:8)
  emit_scenario("nperms0",
    "8 genes x 11 samples, 6 cases vs 5 controls, nperms = 0: a supported mode returning effect sizes with all four p-value columns NA.",
    counts, mk_norm_mat(8, 11, 707), c(rep(1L,6), rep(0L,5)), nperms = 0L)
}

# --- 8. gate_closed: B = 300 < 2*n_tail, so refinement can NEVER fire
{
  counts <- mk_counts(12, 27, 25, 808)
  counts[1, 1:15] <- counts[1, 1:15] * 40           # a very strong gene
  rownames(counts) <- paste0("c", 1:12)
  emit_scenario("gate_closed",
    "12 genes x 27 samples, 15 vs 12, nperms = 300. Gene 1 is planted strongly up so its exceedance count is 0, but B < 2*n_tail = 500 so NO gene is refined and the minimum p-value is 1/301. Tests the refinement gate's B condition.",
    counts, mk_norm_mat(12, 27, 808), c(rep(1L,15), rep(0L,12)), nperms = 300L,
    notes = "refined_diff_i0 and refined_tail_i0 must both be empty.")
}

# --- 9. refine: B = 501 >= 500, planted genes -> GPD refinement fires
{
  counts <- mk_counts(25, 27, 25, 909)
  counts[1, 1:15] <- counts[1, 1:15] * 50           # bulk shift
  counts[2, 1:3]  <- counts[2, 1:3] * 300           # rare high subset
  counts[3, 1:15] <- counts[3, 1:15] * 12
  rownames(counts) <- paste0("r", 1:25)
  emit_scenario("refine",
    "25 genes x 27 samples, 15 vs 12, nperms = 501 (just over the 2*n_tail = 500 gate; n_tail caps at floor(501/2) = 250). Genes 1-3 are planted so the GPD refinement fires; gpd_details carries thr, n_exc, m, v, xi, sigma and the branch for every refined gene.",
    counts, mk_norm_mat(25, 27, 909), c(rep(1L,15), rep(0L,12)), nperms = 501L)
}

# --- 10/11. the non-lean path: weight and log2_scale (open question O5)
{
  counts <- mk_counts(10, 15, 16, 1010)
  rownames(counts) <- paste0("w", 1:10)
  nm <- mk_norm_mat(10, 15, 1010)
  cd <- c(rep(1L,8), rep(0L,7))
  emit_scenario("weighted",
    "10 genes x 15 samples, 8 vs 7, weight = 2.0. Controls are multiplied by 2 before quantiles, which drops wade() off the lean permutation path onto full wade_stats() per permutation.",
    counts, nm, cd, nperms = 13L, weight = 2.0,
    notes = "Under permutation the WEIGHTED group changes membership each iteration; see ROADMAP.md O5.")
  emit_scenario("log2scaled",
    "10 genes x 15 samples, 8 vs 7, log2_scale = TRUE. log2(x+1) applied after weighting; also the non-lean path.",
    counts, nm, cd, nperms = 13L, log2_scale = TRUE)
}

# --- 12. zerolib: an all-zero sample column -> every gene returns exactly 1e6
{
  counts <- mk_counts(6, 8, 12, 1212)
  counts[, 3] <- 0
  rownames(counts) <- paste0("q", 1:6)
  emit_scenario("zerolib",
    "6 genes x 8 samples with sample 3 entirely zero. lib_sizes[3] = 0 makes numerator and denominator equal, so EVERY gene normalizes to exactly 1e6 in that column — an artefact of the algebra, not a sensible value.",
    counts, mk_norm_mat(6, 8, 1212), c(rep(1L,4), rep(0L,4)), nperms = 11L,
    notes = "A port should refuse or document this; see docs/implementation-notes.md section 2.")
}

# --- 13. nonames: rownames(counts) is NULL -> tibble drops the gene column
{
  counts <- mk_counts(7, 9, 13, 1313)
  emit_scenario("nonames",
    "7 genes x 9 samples with NO rownames. gene_names defaults to rownames(counts) = NULL and tibble() DROPS the column, so the frame has 13 columns and no gene identifier at all.",
    counts, mk_norm_mat(7, 9, 1313), c(rep(1L,5), rep(0L,4)), nperms = 11L,
    gene_names = NULL,
    notes = "A port should synthesize positional identifiers instead; assert the difference deliberately.")
}

# --- 14. tailconc: deliberate bulk/tail cancellation -> |tail.conc| >> 1
{
  set.seed(1414)
  g <- 200; n1 <- 11; n0 <- 11; n <- n1 + n0
  counts <- matrix(rlnorm(g * n, meanlog = 3, sdlog = 1), g, n)
  for (i in seq_len(g)) {
    up <- sample.int(n1, 3)
    dn <- setdiff(seq_len(n1), up)
    counts[i, up] <- counts[i, up] * runif(3, 1.4, 3.0)
    counts[i, dn] <- counts[i, dn] * runif(length(dn), 0.30, 0.85)
  }
  counts <- round(counts)
  rownames(counts) <- paste0("t", 1:g)
  emit_scenario("tailconc",
    "200 genes x 22 samples, 11 vs 11 (nprobs = 11, k = 2), constructed so the lower quantiles partially cancel the upper ones. Many genes have |tail.conc| well above 1. THE RATIO ITSELF IS NOT A PARITY TARGET — assert tail_num and tail_den separately. See docs/implementation-notes.md hazard 5.",
    counts, mk_norm_vec(g, 1414), c(rep(1L, n1), rep(0L, n0)), nperms = 17L,
    notes = "The R's guard (|diff.mean * nprobs| < 1e-8) will NA out essentially none of these.")
}

# --- 15. tiesheavy: sparse, zero-heavy, many exact ties (quantile stress)
{
  set.seed(1515)
  g <- 15; n <- 13
  counts <- matrix(rbinom(g * n, 3, 0.35), g, n)
  storage.mode(counts) <- "double"
  counts[1, ] <- 0
  counts[2, ] <- c(rep(0, 7), rep(2, 6))
  counts[3, ] <- rep(1, n)
  rownames(counts) <- paste0("h", 1:15)
  emit_scenario("tiesheavy",
    "15 genes x 13 samples, 7 cases vs 6 controls: nprobs=6, k=1. Counts are 0-3 with heavy ties and whole zero rows — the regime the continuity jitter exists for, and where quantile conventions diverge most.",
    counts, mk_norm_vec(15, 1515), c(rep(1L,7), rep(0L,6)), nperms = 19L)
}

# =====================================================================
# Standalone unit fixtures
# =====================================================================
cat("\nemitting unit fixtures\n")

# --- .gpd_tail_p, all six branches, at the constructions documented in
#     docs/implementation-notes.md hazard 4.
{
  cat("  gpd_branches     "); flush(stdout())
  cases <- list()
  add <- function(label, null, o, n_tail = 250) {
    gi  <- gpd_introspect(o, null, n_tail = n_tail)
    ref <- env$.gpd_tail_p(o, null, n_tail = n_tail)
    stopifnot(identical(gi$p, ref))
    cases[[length(cases) + 1L]] <<- j_obj(
      label = .j_str(label),
      null  = j_dbl_vec(null, paste0("null.", label)),
      o     = j_dbl(o, paste0("o.", label)),
      n_tail_arg = j_int(n_tail),
      detail = j_gpd(gi)
    )
  }

  # A heavy-tailed null gives a positive moment shape. Note that pushing the
  # observed value far out FLOORS the result, which is a different branch: to
  # observe the GPD form's own value the observation has to sit inside the
  # range the fit can still resolve, so these use order statistics of the null
  # rather than a large multiple of its maximum.
  set.seed(1); n1v <- rt(2000, 2.5)
  add("gpd_positive_shape", n1v, sort(n1v, decreasing = TRUE)[5])
  add("gpd_positive_shape_2nd", n1v, sort(n1v, decreasing = TRUE)[2])
  add("gpd_positive_shape_floored", n1v, 5 * max(n1v))

  set.seed(2); n2v <- rexp(2000, 1);   add("exponential_branch", n2v, 1.2 * max(n2v))
  set.seed(3); n3v <- rnorm(2000);     add("exponential_normal_null", n3v, 1.6 * max(n3v))
  set.seed(4); n4v <- runif(2000);     add("exponential_uniform_null", n4v, max(n4v))
  set.seed(5); n5v <- rexp(2000, 1);   add("floor_binding", n5v, 20 * max(n5v))
  set.seed(6); n6v <- rexp(2000, 1);   add("obs_at_or_below_thr", n6v, sort(n6v, decreasing = TRUE)[300])
  set.seed(7); n7v <- c(rep(5, 400), rexp(1600, 5)); add("few_exceedances_via_ties", n7v, 9)

  # An all-constant null ties at the threshold and so bails on the exceedance
  # COUNT, never reaching the variance test. To reach the zero-variance bail
  # the exceedances must be numerous (>= 10) and identical.
  n8v <- rep(2.5, 2000)
  add("all_tied_bails_on_count", n8v, 10)
  n8b <- c(rep(9, 20), rep(1, 1980))
  add("degenerate_zero_variance", n8b, 50)

  set.seed(9); n9v <- rexp(100, 1);    add("n_tail_capped_at_B_over_2", n9v, 3 * max(n9v))

  json <- j_obj(
    name = .j_str("gpd_branches"),
    description = .j_str("Constructed nulls exercising every branch and bail-out of .gpd_tail_p(). Each case carries the full null vector, the observed value, and the internals (thr, n_exc, m, v, xi, sigma, tail_prob, floor, branch, p)."),
    n_tail_default = j_int(250L),
    cases = paste0("[", paste0(unlist(cases), collapse = ","), "]")
  )
  j_write(json, file.path(OUT_DIR, "gpd_branches.json"))
  cat(sprintf("%d cases  ok\n", length(cases)))
}

# --- p.adjust(., "BH"), including the NA-dropping behaviour
{
  cat("  bh_padjust       "); flush(stdout())
  v1 <- c(0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074,
          0.205, 0.212, 0.216, 0.222, 0.45, 0.6, 0.74, 0.99)
  v2 <- c(0.9, 0.01, NA, 0.5, 0.01, 1.0)
  v3 <- rep(NA_real_, 5)
  set.seed(31); v4 <- round(runif(23), 6); v4[c(3, 17)] <- v4[c(2, 16)]  # ties
  v5 <- c(1/2001, 2/2001, 2e-06, 2e-06, 0.5)
  mk <- function(lbl, v) j_obj(
    label = .j_str(lbl), p = j_dbl_vec(v, paste0("p.", lbl)),
    padj  = j_dbl_vec(stats::p.adjust(v, "BH"), paste0("padj.", lbl))
  )
  json <- j_obj(
    name = .j_str("bh_padjust"),
    description = .j_str("stats::p.adjust(p, 'BH') targets. R DROPS NA, adjusts using the reduced n, and puts NA back in position; ties receive equal adjusted values; the step-up cumulative minimum is enforced."),
    cases = paste0("[", paste0(c(mk("basic15", v1), mk("with_na", v2),
                                 mk("all_na", v3), mk("with_ties", v4),
                                 mk("at_floors", v5)), collapse = ","), "]")
  )
  j_write(json, file.path(OUT_DIR, "bh_padjust.json"))
  cat("5 cases  ok\n")
}

# --- ecdf and dense_rank conventions, for the rank scores
{
  cat("  ecdf_denserank   "); flush(stdout())
  e1 <- c(3, 1, 4, 1, 5)
  e2 <- c(-2.5, 0, 0, 7.25, 7.25, 7.25, 100)
  d1 <- c(0.5, 0.5, 0.2, NA, 0.9)
  d2 <- c(3, 1, 4, 1, 5, 9, 2, 6)
  mk_e <- function(lbl, v) j_obj(
    label = .j_str(lbl), x = j_dbl_vec(v, lbl),
    ecdf_at_x = j_dbl_vec(stats::ecdf(v)(v), paste0("F.", lbl))
  )
  mk_d <- function(lbl, v) j_obj(
    label = .j_str(lbl), x = j_dbl_vec(v, lbl),
    dense_rank_desc = j_int_vec(dplyr::dense_rank(dplyr::desc(v)))
  )
  json <- j_obj(
    name = .j_str("ecdf_denserank"),
    description = .j_str("R's stats::ecdf is F(t) = #{x <= t}/n (right-continuous, ties shared, F(max) = 1, never 0 at an observed point). dplyr::dense_rank(desc(x)) gives ties the same rank with no gaps and propagates NA."),
    ecdf_cases = paste0("[", paste0(c(mk_e("small_with_tie", e1), mk_e("runs_and_negatives", e2)), collapse = ","), "]"),
    dense_rank_cases = paste0("[", paste0(c(mk_d("ties_and_na", d1), mk_d("plain", d2)), collapse = ","), "]")
  )
  j_write(json, file.path(OUT_DIR, "ecdf_denserank.json"))
  cat("ok\n")
}

# --- the method.md section 2.7 worked example, straight from wade_stats
{
  cat("  worked_example   "); flush(stdout())
  X <- rbind(gA = c(10, 12, 14, 16, 60, 10, 12, 14, 16),
             gB = c(20, 22, 24, 26, 28, 10, 12, 14, 16))
  cond <- c(rep(1L, 5), rep(0L, 4))
  st <- env$wade_stats(X, cond)
  json <- j_obj(
    name = .j_str("worked_example"),
    description = .j_str("The deterministic fixture from docs/method.md section 2.7: 2 genes, 5 cases vs 4 controls, no jitter and no normalization — wade_stats() called directly on the values. nprobs=4, q=(1,2/3,1/3,0), k=1."),
    X = j_dbl_mat(X, "X"), cond = j_int_vec(cond),
    nprobs = j_int(st$nprobs), k = j_int(st$k),
    q = j_dbl_vec(st$q, "q"),
    Q1 = j_dbl_mat(st$Q1, "Q1"), Q0 = j_dbl_mat(st$Q0, "Q0"), D = j_dbl_mat(st$D, "D"),
    stats = j_obj(
      diff.mean = j_dbl_vec(st$diff.mean, "diff.mean"),
      w1 = j_dbl_vec(st$w1, "w1"),
      tail.mean = j_dbl_vec(st$tail.mean, "tail.mean"),
      tail.conc = j_dbl_vec(st$tail.conc, "tail.conc"),
      fc = j_dbl_vec(st$fc, "fc"),
      cond1.mean = j_dbl_vec(st$cond1.mean, "cond1.mean"),
      cond0.mean = j_dbl_vec(st$cond0.mean, "cond0.mean"),
      tot.mean = j_dbl_vec(st$tot.mean, "tot.mean")
    ),
    sample_mean_difference = j_dbl_vec(
      rowMeans(X[, cond == 1L, drop = FALSE]) - rowMeans(X[, cond == 0L, drop = FALSE]),
      "sample_mean_difference")
  )
  j_write(json, file.path(OUT_DIR, "worked_example.json"))
  cat("ok\n")
}

# --- type-7 quantile conventions on the vectors from hazard 1
{
  cat("  quantile_type7   "); flush(stdout())
  v_odd  <- c(0, 0, 0, 1, 2, 2, 5, 13, 100)
  v_even <- c(0, 0, 3, 3, 3, 7, 11, 40)
  probs5 <- seq(1, 0, length.out = 5)
  mk <- function(lbl, v) {
    grids <- lapply(c(2, 3, 4, 5, 7, 8), function(m) {
      pr <- seq(1, 0, length.out = m)
      j_obj(nprobs = j_int(m), q = j_dbl_vec(pr, "q"),
            rowQuantiles = j_dbl_vec(
              as.numeric(matrixStats::rowQuantiles(matrix(v, nrow = 1), probs = pr, useNames = FALSE)),
              paste0("rq.", lbl, ".", m)),
            stats_quantile_type7 = j_dbl_vec(
              as.numeric(stats::quantile(v, probs = pr, type = 7, names = FALSE)),
              paste0("sq.", lbl, ".", m)))
    })
    j_obj(label = .j_str(lbl), x = j_dbl_vec(v, lbl),
          grids = paste0("[", paste0(unlist(grids), collapse = ","), "]"))
  }
  json <- j_obj(
    name = .j_str("quantile_type7"),
    description = .j_str("Type-7 quantiles on WADE's own descending grid shape seq(1,0,length.out=m), for an odd-length and an even-length vector with ties. matrixStats::rowQuantiles and stats::quantile(type=7) are emitted separately and must agree bitwise. The even-length vector is the one that separates type 7 from types 1, 2, 4 and 6 — a test built only on the odd vector passes with type 1 substituted."),
    cases = paste0("[", paste0(c(mk("v_odd", v_odd), mk("v_even", v_even)), collapse = ","), "]")
  )
  j_write(json, file.path(OUT_DIR, "quantile_type7.json"))
  cat("ok\n")
}

cat("\ndone. fixtures in:", OUT_DIR, "\n")
print(data.frame(file = list.files(OUT_DIR, pattern = "\\.json$"),
                 kb = round(file.size(list.files(OUT_DIR, pattern = "\\.json$", full.names = TRUE)) / 1024, 1)))
