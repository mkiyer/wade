# =====================================================================
# wade_seam.R — the deterministic seam
# ---------------------------------------------------------------------
# Sources reference/R/wade.R UNMODIFIED and makes its two sources of
# randomness injectable:
#
#   * the continuity jitter   — stats::runif() inside wade_normalize()
#   * the label permutations  — sample(cond) inside wade()'s loop
#
# Exact cross-language parity is impossible without this: R's
# Mersenne-Twister and NumPy's PCG64 cannot be made to agree on a shared
# seed (docs/porting-hazards.md, hazard 2). The fixtures must therefore
# carry the realised jitter matrix and permutation matrix as data.
#
# HOW THE INJECTION WORKS, and why it does not touch wade.R or any
# namespace:
#
#   wade.R is sourced into a private environment whose PARENT holds two
#   shadowing bindings. R resolves free variables lexically through the
#   enclosing environment chain, so:
#
#     * `sample(cond)` (unqualified) resolves to our shim.
#     * `stats::runif(...)` resolves `::` FIRST, and base's `::` is an
#       ordinary closure, so shadowing `::` intercepts the lookup. Our
#       `::` returns the shim for stats::runif when armed and delegates
#       everything else (matrixStats::rowQuantiles, tibble::tibble,
#       dplyr::dense_rank, stats::var, stats::p.adjust, stats::ecdf) to
#       getExportedValue() unchanged.
#
#   Nothing is assigned into any package namespace, nothing global is
#   mutated, and reference/R/wade.R stays byte-identical. Confirm with
#   `shasum -a 256 -c sha256sums.txt` from reference/R/.
#
# The seam is VERIFIED TRANSPARENT by seam_selftest(): a native wade()
# run and a seam run injected with R's own realised draws must agree
# bitwise. If that ever fails the seam is lying and every fixture it
# emits is suspect.
# =====================================================================

# ---------------------------------------------------------------------
# Injection state. Kept in its own environment so nothing leaks into the
# caller's workspace and so the shims can mutate it.
# ---------------------------------------------------------------------
.seam <- new.env(parent = emptyenv())
.seam$armed_jitter <- NULL   # g x n matrix, or NULL to pass through
.seam$armed_perms  <- NULL   # B x n matrix of labels, or NULL
.seam$runif_calls  <- 0L     # how many times the runif shim fired
.seam$perm_calls   <- 0L     # how many times the sample shim fired

seam_arm <- function(jitter = NULL, perms = NULL) {
  if (!is.null(jitter)) {
    stopifnot(is.matrix(jitter), is.numeric(jitter))
  }
  if (!is.null(perms)) {
    stopifnot(is.matrix(perms))
    storage.mode(perms) <- "integer"
  }
  .seam$armed_jitter <- jitter
  .seam$armed_perms  <- perms
  .seam$runif_calls  <- 0L
  .seam$perm_calls   <- 0L
  invisible(NULL)
}

seam_disarm <- function() {
  .seam$armed_jitter <- NULL
  .seam$armed_perms  <- NULL
  invisible(NULL)
}

seam_counts <- function() {
  list(runif_calls = .seam$runif_calls, perm_calls = .seam$perm_calls)
}

# ---------------------------------------------------------------------
# The two shims.
# ---------------------------------------------------------------------

# wade_normalize() calls: stats::runif(g * n, 0, noise)
# and immediately does matrix(., g, n) — R fills COLUMN-MAJOR, so the
# vector we return must be the column-major flattening of the supplied
# jitter matrix. as.vector() on a matrix is exactly that.
.seam_runif <- function(n, min = 0, max = 1) {
  J <- .seam$armed_jitter
  if (is.null(J)) return(stats::runif(n, min, max))
  if (length(J) != n) {
    stop(sprintf("seam: injected jitter has %d cells, wade.R asked for %d draws",
                 length(J), n))
  }
  .seam$runif_calls <- .seam$runif_calls + 1L
  as.vector(J)
}

# wade() calls: sample(cond), once per permutation, in order.
.seam_sample <- function(x, size, replace = FALSE, prob = NULL) {
  P <- .seam$armed_perms
  if (is.null(P) || !missing(size) || !identical(replace, FALSE) || !is.null(prob)) {
    return(base::sample(x, size = if (missing(size)) length(x) else size,
                        replace = replace, prob = prob))
  }
  b <- .seam$perm_calls + 1L
  if (b > nrow(P)) {
    stop(sprintf("seam: wade.R requested permutation %d but only %d supplied",
                 b, nrow(P)))
  }
  if (ncol(P) != length(x)) {
    stop(sprintf("seam: permutation row has %d labels, cond has %d",
                 ncol(P), length(x)))
  }
  .seam$perm_calls <- b
  P[b, ]
}

# `::` is an ordinary closure in base R, so it can be shadowed lexically.
# Delegate everything except the one binding we are intercepting.
.seam_ns_get <- function(pkg, name) {
  p <- as.character(substitute(pkg))
  n <- as.character(substitute(name))
  if (p == "stats" && n == "runif" && !is.null(.seam$armed_jitter)) {
    return(.seam_runif)
  }
  getExportedValue(p, n)
}

# ---------------------------------------------------------------------
# Build the seam environment.
# ---------------------------------------------------------------------
wade_seam_env <- function(wade_path) {
  if (!file.exists(wade_path)) stop("wade.R not found at: ", wade_path)
  shim <- new.env(parent = globalenv())
  assign("::",     .seam_ns_get,  envir = shim)
  assign("sample", .seam_sample,  envir = shim)
  env <- new.env(parent = shim)
  sys.source(wade_path, envir = env)
  env
}

# ---------------------------------------------------------------------
# Reconstruct the draws wade.R makes natively for a given seed.
#
# These are not guesses: wade_normalize() does set.seed(seed) then one
# runif(g*n, 0, noise); wade() does set.seed(seed + 1L) then exactly one
# sample(cond) per iteration, in order, consuming nothing else from the
# stream. Reproducing that sequence reproduces the draws exactly, which
# seam_selftest() then confirms bitwise.
# ---------------------------------------------------------------------
seam_native_jitter <- function(g, n, noise = 0.01, seed = 1L) {
  set.seed(seed)
  matrix(stats::runif(g * n, 0, noise), g, n)
}

seam_native_perms <- function(cond, nperms, seed = 1L) {
  set.seed(seed + 1L)
  P <- matrix(NA_integer_, nperms, length(cond))
  for (b in seq_len(nperms)) P[b, ] <- base::sample(cond)
  P
}

# ---------------------------------------------------------------------
# Prove the seam is transparent: a native wade() run and a seam run fed
# R's own realised draws must be bitwise identical.
# ---------------------------------------------------------------------
seam_selftest <- function(wade_path, verbose = TRUE) {
  native_env <- new.env(parent = globalenv())
  sys.source(wade_path, envir = native_env)
  seam_env <- wade_seam_env(wade_path)

  set.seed(20260817)
  g <- 9L; n <- 11L; B <- 23L
  counts <- matrix(rpois(g * n, 12), g, n)
  counts[2, ] <- 0L                      # an all-zero gene
  counts[3, 4:7] <- 0L                   # zeros and ties
  rownames(counts) <- paste0("g", seq_len(g))
  normalizer <- matrix(runif(g * n, 0.5, 6), g, n)
  cond <- c(rep(1L, 6L), rep(0L, 5L))
  noise <- 0.01; seed <- 1L

  ls_ <- native_env$wade_lib_size(counts, normalizer)

  seam_disarm()
  native <- native_env$wade(counts, normalizer, ls_, cond, nperms = B,
                            noise = noise, seed = seed, verbose = FALSE)

  J <- seam_native_jitter(g, n, noise = noise, seed = seed)
  P <- seam_native_perms(cond, B, seed = seed)

  seam_arm(jitter = J, perms = P)
  injected <- seam_env$wade(counts, normalizer, ls_, cond, nperms = B,
                            noise = noise, seed = seed, verbose = FALSE)
  cnts <- seam_counts()
  seam_disarm()

  ok_fired <- cnts$runif_calls == 1L && cnts$perm_calls == B
  ok_ident <- identical(as.data.frame(native), as.data.frame(injected))

  if (verbose) {
    cat("seam self-test\n")
    cat("  runif shim fired  :", cnts$runif_calls, "(expected 1)\n")
    cat("  sample shim fired :", cnts$perm_calls, "(expected", B, ")\n")
    cat("  bitwise identical to native run:", ok_ident, "\n")
    if (!ok_ident) {
      for (nm in names(native)) {
        a <- native[[nm]]; b <- injected[[nm]]
        if (!identical(a, b)) cat("    DIFFERS:", nm, "\n")
      }
    }
  }
  if (!ok_fired) stop("seam self-test: shims did not fire the expected number of times")
  if (!ok_ident) stop("seam self-test: injected run differs from native run")
  invisible(TRUE)
}
