# =====================================================================
# json_emit.R — dependency-free JSON writer for golden fixtures
# ---------------------------------------------------------------------
# jsonlite is not in the pinned renv closure (reference/R/renv.lock has
# 31 packages and jsonlite is not one of them) and there is no install
# path in this sandbox, so the writer is hand-rolled.
#
# THREE DECISIONS THAT MATTER FOR A PARITY FIXTURE
#
# 1. Every double is emitted TWICE: as a 17-significant-digit decimal
#    string ("dec") and as a C99 hex float ("hex").
#
#    The decimal channel is what the roadmap asks for and what a human
#    reads. The hex channel is what makes the fixture exact: "%a" is a
#    lossless base-2 rendering of the IEEE-754 double, so it cannot lose
#    a bit no matter what parser reads it.
#
#    Both are needed because R cannot verify its own decimal output.
#    Measured in this sandbox: R's sprintf("%.17g") produces correctly
#    rounded digits, but R's as.numeric() is NOT a correctly-rounded
#    parser and reads 8 of 42 uniform draws back one ULP off — and it
#    still does at %.18g, %.19g and %.20g, so this is the reader, not
#    the digits. Hex round-trips through as.numeric() exactly.
#
#    So verification is split: R checks the hex channel round-trips
#    here, and the Python loader checks dec == hex on every value it
#    reads (tests/conftest.py). Hex is authoritative; a disagreement is
#    a real defect and fails the test suite rather than passing quietly.
#
# 2. Numbers are strings, never bare JSON numbers. That sidesteps
#    reader-dependent float parsing entirely and lets NA / NaN / Inf /
#    -Inf be represented at all — bare NaN and Infinity are not legal
#    JSON. NA and NaN are kept DISTINCT: R's NA_real_ is a NaN with a
#    payload, and wade() produces NA (not NaN) from its tail.conc guard
#    and from nperms = 0, so collapsing them would erase the
#    distinction a port is being tested on.
#
# 3. Every 2-D array carries "shape": [nrow, ncol] and is nested as row
#    arrays. docs/implementation-notes.md hazard 8 is precisely the failure
#    of shipping a flat vector plus dimensions and letting two languages
#    disagree about fill order; nested rows cannot be silently
#    transposed.
# =====================================================================

.j_esc <- function(s) {
  s <- gsub("\\", "\\\\", s, fixed = TRUE)
  s <- gsub("\"", "\\\"", s, fixed = TRUE)
  s
}

.j_str <- function(s) paste0("\"", .j_esc(as.character(s)), "\"")

# The four non-finite states R can produce, spelled explicitly.
# is.nan() must be tested before is.na() because is.na(NaN) is TRUE.
.nonfinite_label <- function(v) {
  if (is.nan(v))       "NaN"
  else if (is.na(v))   "NA"
  else if (is.infinite(v)) (if (v > 0) "Inf" else "-Inf")
  else                 NA_character_
}

.fmt_dec <- function(x) {
  x <- as.numeric(x)
  vapply(x, function(v) {
    lab <- .nonfinite_label(v)
    if (!is.na(lab)) lab else sprintf("%.17g", v)
  }, character(1), USE.NAMES = FALSE)
}

.fmt_hex <- function(x) {
  x <- as.numeric(x)
  vapply(x, function(v) {
    lab <- .nonfinite_label(v)
    if (!is.na(lab)) lab else sprintf("%a", v)
  }, character(1), USE.NAMES = FALSE)
}

# Verify the hex channel is lossless. R's as.numeric parses C99 hex
# floats exactly, so this is a real check rather than a formality.
.check_hex_roundtrip <- function(x, what) {
  x <- as.numeric(x)
  fin <- is.finite(x)
  if (!any(fin)) return(invisible(TRUE))
  back <- as.numeric(.fmt_hex(x)[fin])
  if (!identical(back, x[fin])) {
    i <- which(back != x[fin])[1]
    stop(sprintf("json_emit: hex round-trip failed for '%s': %s -> %s",
                 what, sprintf("%a", x[fin][i]), format(back[i], digits = 22)))
  }
  invisible(TRUE)
}

.arr <- function(v) paste0("[", paste0("\"", v, "\"", collapse = ","), "]")

# double vector -> {"dec":[...],"hex":[...]}
j_dbl_vec <- function(x, what = "value") {
  x <- as.numeric(x)
  .check_hex_roundtrip(x, what)
  paste0("{\"dec\":", .arr(.fmt_dec(x)), ",\"hex\":", .arr(.fmt_hex(x)), "}")
}

j_dbl <- function(x, what = "value") {
  stopifnot(length(x) == 1)
  j_dbl_vec(x, what)
}

# double matrix -> {"shape":[r,c],"dec":[[...],...],"hex":[[...],...]}
j_dbl_mat <- function(m, what = "matrix") {
  m <- as.matrix(m)
  storage.mode(m) <- "double"
  .check_hex_roundtrip(as.vector(m), what)
  rowsd <- vapply(seq_len(nrow(m)), function(i) .arr(.fmt_dec(m[i, ])), character(1))
  rowsh <- vapply(seq_len(nrow(m)), function(i) .arr(.fmt_hex(m[i, ])), character(1))
  paste0("{\"shape\":[", nrow(m), ",", ncol(m), "],",
         "\"dec\":[", paste0(rowsd, collapse = ","), "],",
         "\"hex\":[", paste0(rowsh, collapse = ","), "]}")
}

j_int_vec <- function(x) {
  x <- as.integer(x)
  paste0("[", paste0(ifelse(is.na(x), "null", as.character(x)), collapse = ","), "]")
}

j_int <- function(x) {
  stopifnot(length(x) == 1)
  if (is.na(x)) "null" else as.character(as.integer(x))
}

j_int_mat <- function(m) {
  m <- as.matrix(m)
  storage.mode(m) <- "integer"
  rows <- vapply(seq_len(nrow(m)), function(i) {
    paste0("[", paste0(as.character(m[i, ]), collapse = ","), "]")
  }, character(1))
  paste0("{\"shape\":[", nrow(m), ",", ncol(m), "],\"data\":[",
         paste0(rows, collapse = ","), "]}")
}

j_str_vec <- function(x) paste0("[", paste0(.j_str(x), collapse = ","), "]")

j_bool <- function(x) if (isTRUE(x)) "true" else "false"

j_obj <- function(...) {
  kv <- list(...)
  nms <- names(kv)
  if (is.null(nms) || any(nms == "")) stop("j_obj: every element must be named")
  paste0("{", paste0(.j_str(nms), ":", unlist(kv, use.names = FALSE), collapse = ","), "}")
}

j_write <- function(json, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  con <- file(path, open = "wt")
  on.exit(close(con))
  writeLines(json, con)
  invisible(path)
}
