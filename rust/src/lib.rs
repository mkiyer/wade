//! The permutation loop, in Rust.
//!
//! # Why this and nothing else
//!
//! The serial permutation loop is the **entire cost** of the method:
//! `nperms` iterations, each doing two row-quantile passes over a genes ×
//! samples slice plus a subtraction and three reductions. Everything else
//! in WADE is a single pass. Any boundary that puts this loop on the Rust
//! side captures essentially all of the available speedup; any boundary
//! that does not, captures almost none.
//!
//! # What crosses the boundary, and why it matters more than what is behind it
//!
//! This kernel returns the **full null matrix**, not p-values. That is a
//! deliberate testability decision: asserting the two `g × nperms` nulls
//! elementwise against the R's serial loop is the only place a kernel bug
//! is cleanly separable from a p-value bug. A kernel that returned only
//! p-values would be harder to test whatever its speed. It also keeps the
//! GPD refinement possible at all, since that needs each refined gene's
//! *full* null vector.
//!
//! The permutation matrix is an **input**, on the same argument path
//! production uses. R's Mersenne-Twister and NumPy's PCG64 cannot agree on
//! a shared seed, so exact cross-language parity is only achievable by
//! passing the realised labels as data — and if the fixture path went
//! through a separate entry point, the parity suite would be validating
//! code nobody runs.
//!
//! # Numerical contract
//!
//! Two choices here are about numerical fidelity rather than speed, and
//! neither should be "optimized" away:
//!
//! * **Type-7 quantiles are reimplemented here** rather than delegated,
//!   term for term against R's `quantile.default` — including its
//!   no-interpolation guard, which returns `x[lo]` untouched when the two
//!   bracketing order statistics are equal. Without that guard
//!   `(1-h)*a + h*a` can drift off `a` by an ulp on a run of equal values,
//!   and WADE's target regime is zero-heavy count data.
//!
//! * **Summation is sequential, left to right**, matching R's `rowSums`.
//!   NumPy's `sum` uses pairwise summation, which is *more* accurate and
//!   therefore guaranteed to differ from R on some inputs. Since the R is
//!   the parity oracle, matching its association is what keeps the null
//!   matrices bitwise identical. Do not replace these loops with anything
//!   that reassociates.
//!
//! Both kernels are parallelized across **genes** with rayon, because both
//! sort each gene's row once and re-partition it per permutation (see the
//! subset kernel's notes below — the mean-shift kernel adopted the same
//! design 2026-08-20, ~10× over its original permutation-major loop). That
//! changes nothing numerically: each gene's arithmetic is self-contained
//! and no accumulation crosses threads.
//!
//! # The second kernel: the subset test
//!
//! [`subset_null`] is the permutation loop of the *shape* test
//! (`docs/method.md` §3), validated elementwise against
//! `wade.permutation._subset_null_numpy`, which is the contract. Its
//! design differs from the mean-shift kernel in three ways, each for a
//! reason:
//!
//! * **Parallel over genes, not permutations.** The test needs two passes
//!   over the permutations — the first estimates the null moments of the
//!   bridge at every width, the second standardizes against them before
//!   maximizing. Per gene, those moment accumulators are tiny (`m` doubles)
//!   and the second pass can reuse what the first computed; per
//!   permutation, every thread would need its own `(genes × m)` accumulator
//!   and a cross-thread reduction that would also change the summation
//!   order. So each gene runs both passes on one thread, and the order of
//!   accumulation over permutations is exactly the NumPy path's.
//!
//! * **One sort per gene, not one per permutation per group.** Every
//!   permutation only re-partitions the same `n` values into two groups. Sort
//!   the gene's row once; then the sorted case values are the subsequence of
//!   that sorted row whose labels are 1, obtained by one O(n) walk. Equal
//!   values are interchangeable, so ties make no difference to what is
//!   read. This is where most of the speedup comes from.
//!
//! * **Almost no `log2` calls.** The log-ratio curve is
//!   `log2(q1) - log2(q0)`. Where type 7 does not interpolate — `h == 0`, or
//!   the two bracketing values are equal — the quantile *is* a group value
//!   whose `log2` was computed once when the row was sorted, and because
//!   `log2` is monotone the group's sorted logs are the logs of its sorted
//!   values elementwise. Only where interpolation fires is `log2` called, on
//!   the interpolated value, exactly as the NumPy path does. On the full
//!   grid, `m = min(n1, n0)`, the smaller group's nodes land on its order
//!   statistics and need no `log2` (bar the handful where NumPy's `linspace`
//!   puts the node an ulp off an integer); on a balanced design that is both
//!   groups, and on an unbalanced one only the larger group's nodes
//!   interpolate. A grid capped below the design (`max_probs`,
//!   `docs/method.md` §1) interpolates more nodes and saves fewer calls, but
//!   either way the inputs to `log2` are the same as the NumPy path's, so
//!   the outputs are the same bits.
//!
//! Everything else — the type-7 guard, the sequential cumulative sum, the
//! `k/m` division before the multiplication, the `±0.0` that an unusable
//! width contributes to the maximum — is mirrored term for term, and the
//! test file holds the two paths to `1e-15` relative.

use numpy::ndarray::{Array1, Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// The mean-shift statistics of one gene under every permutation.
///
/// One sort per gene, not one per permutation per group: a permutation only
/// re-partitions the same values, so the row is sorted once and each
/// permutation's two sorted groups fall out of a single O(n) two-ended
/// partition walk — the same trick, and the same buffer layout, as
/// [`gene_bridge`] in the subset kernel (cases fill `part` from the front
/// ascending, controls from the back descending, and the two [`Type7Plan`]s
/// know where their group sits). Ties are equal values, so the partition
/// reads exactly what per-permutation sorts would.
///
/// The quantile arithmetic is [`Type7Plan`]'s term for term — the plans hold
/// the same `lo`/`hi`/`h` R's definition produces — and the accumulation over
/// the grid is sequential left to right, so the outputs are bitwise what a
/// per-permutation sort produced.
fn one_gene_mean_shift(
    row: &[f64],
    masks: &[u8],
    plan1: &Type7Plan,
    plan0: &Type7Plan,
    order: &mut [usize],
    sorted: &mut [f64],
    part: &mut [f64],
    out_row: &mut [f64],
) {
    let n = row.len();
    let m = plan1.h.len();
    for (i, slot) in order.iter_mut().enumerate() {
        *slot = i;
    }
    order.sort_unstable_by(|&a, &b| {
        row[a].partial_cmp(&row[b]).expect("non-finite value in kernel input")
    });
    for i in 0..n {
        sorted[i] = row[order[i]];
    }

    for (b, slot) in out_row.iter_mut().enumerate() {
        let mask = &masks[b * n..(b + 1) * n];
        let mut front = 0usize;
        let mut back = n;
        for i in 0..n {
            let c = (mask[order[i]] == 1) as usize;
            back -= 1 - c;
            let pos = c * front + (1 - c) * back;
            part[pos] = sorted[i];
            front += c;
        }
        debug_assert_eq!(front, back);

        // Sequential accumulation, matching R's rowSums association.
        let mut total = 0.0_f64;
        for k in 0..m {
            let a1 = part[plan1.lo[k]];
            let b1 = part[plan1.hi[k]];
            let h1 = plan1.h[k];
            let q1 = if h1 > 0.0 && b1 != a1 { (1.0 - h1) * a1 + h1 * b1 } else { a1 };
            let a0 = part[plan0.lo[k]];
            let b0 = part[plan0.hi[k]];
            let h0 = plan0.h[k];
            let q0 = if h0 > 0.0 && b0 != a0 { (1.0 - h0) * a0 + h0 * b0 } else { a0 };
            total += q1 - q0;
        }
        *slot = total / m as f64;
    }
}

/// The mean-shift null for a supplied set of label permutations.
///
/// Returns the `(n_genes, n_perms)` matrix directly — the kernel is parallel
/// over genes (each gene sorts its row once and walks every permutation), so
/// gene-major writes are the contiguous ones.
#[pyfunction]
fn null_statistics<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    perms: PyReadonlyArray2<'py, i64>,
    probs: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let xv = x.as_array();
    let pv = perms.as_array();
    let probs_slice: Vec<f64> = probs.as_array().iter().copied().collect();

    let g = xv.nrows();
    let n = xv.ncols();
    let n_perms = pv.nrows();
    let nprobs = probs_slice.len();

    if pv.ncols() != n {
        return Err(PyValueError::new_err(format!(
            "perms has {} columns but the matrix has {} samples",
            pv.ncols(),
            n
        )));
    }
    if nprobs == 0 {
        return Err(PyValueError::new_err("probs must be non-empty"));
    }
    if !probs_slice.iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p)) {
        return Err(PyValueError::new_err("probs must be finite and within [0, 1]"));
    }
    if !xv.iter().all(|v| v.is_finite()) {
        return Err(PyValueError::new_err(
            "the normalized matrix contains non-finite values",
        ));
    }

    // Group sizes are preserved by label exchange, so nprobs is a constant of
    // the run. The grid may be *capped* below min(n1, n0) (`max_probs`,
    // docs/method.md §1) but never exceeds it: a denser-than-design grid is
    // never produced by WADE and signals a caller bug.
    let mut n1 = 0usize;
    let mut n0 = 0usize;
    let mut masks = vec![0u8; n_perms * n];
    for (b, row) in pv.rows().into_iter().enumerate() {
        let mut r1 = 0usize;
        let mut r0 = 0usize;
        for (j, &lab) in row.iter().enumerate() {
            match lab {
                1 => {
                    r1 += 1;
                    masks[b * n + j] = 1;
                }
                0 => r0 += 1,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "perms must contain only 0 and 1; row {} has {}",
                        b, other
                    )))
                }
            }
        }
        if b == 0 {
            n1 = r1;
            n0 = r0;
        } else if r1 != n1 || r0 != n0 {
            return Err(PyValueError::new_err(format!(
                "permutation row {} has group sizes (n1, n0) = ({}, {}) but row 0 has \
                 ({}, {}); label exchange preserves the group sizes",
                b, r1, r0, n1, n0
            )));
        }
        if r1.min(r0) < nprobs {
            return Err(PyValueError::new_err(format!(
                "permutation row {} gives min(n1, n0) = {} but probs has {} entries; \
                 the grid may be capped below the design but never above it",
                b,
                r1.min(r0),
                nprobs
            )));
        }
    }
    if n1 == 0 || n0 == 0 {
        return Err(PyValueError::new_err("both groups must be non-empty"));
    }

    let plan1 = Type7Plan::new(n1, &probs_slice, 0, false);
    let plan0 = Type7Plan::new(n0, &probs_slice, n1, true);

    let mut out = vec![0.0_f64; g * n_perms];

    py.detach(|| {
        out.par_chunks_mut(n_perms).enumerate().for_each_init(
            || (vec![0usize; n], vec![0.0_f64; n], vec![0.0_f64; n], vec![0.0_f64; n]),
            |(order, sorted, part, row_buf), (gene, out_row)| {
                // Copy through a buffer so the hot loop sees a contiguous
                // slice whatever the input strides.
                for (slot, v) in row_buf.iter_mut().zip(xv.row(gene).iter()) {
                    *slot = *v;
                }
                one_gene_mean_shift(
                    row_buf, &masks, &plan1, &plan0, order, sorted, part, out_row,
                );
            },
        );
    });

    let arr = Array2::from_shape_vec((g, n_perms), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

// ---------------------------------------------------------------------
// The subset test
// ---------------------------------------------------------------------

/// Which end of the bridge the scan looks at. Mirrors `reduce()` in
/// `_subset_null_numpy`: `greater` keeps `z`, `less` negates it, `two-sided`
/// takes its magnitude.
#[derive(Clone, Copy)]
enum Alternative {
    TwoSided,
    Greater,
    Less,
}

impl Alternative {
    fn parse(s: &str) -> PyResult<Self> {
        match s {
            "two-sided" => Ok(Alternative::TwoSided),
            "greater" => Ok(Alternative::Greater),
            "less" => Ok(Alternative::Less),
            other => Err(PyValueError::new_err(format!(
                "alternative must be one of ('two-sided', 'greater', 'less'); got {:?}",
                other
            ))),
        }
    }

    #[inline(always)]
    fn reduce(self, z: f64) -> f64 {
        match self {
            Alternative::Greater => z,
            Alternative::Less => -z,
            Alternative::TwoSided => z.abs(),
        }
    }
}

/// The type-7 bracketing for a group of fixed size, computed once.
///
/// Mirrors R's `quantile.default(type = 7)`, which is the definition both
/// kernels read through:
///
/// ```text
/// index <- 1 + (n - 1) * p
/// lo <- floor(index); hi <- ceiling(index)
/// qs  <- x[lo]
/// i   <- which(index > lo & x[hi] != qs)
/// qs[i] <- (1 - h) * qs[i] + h * x[hi][i]
/// ```
///
/// The 1-based index arithmetic is kept as-is and shifted to 0-based only at
/// the point of indexing, so the floor/ceil see exactly the values R's do.
/// The `x[hi] != x[lo]` guard is load-bearing on tied data: without it
/// `(1-h)*a + h*a` can drift off `a` by an ulp on a run of equal values, and
/// WADE's target regime is zero-heavy count data.
///
/// `lo`, `hi` and `h` depend only on the group size and the grid, both of
/// which are constants of the run, so the floor/ceil are done once rather
/// than once per permutation. The arithmetic is the same as
/// this type's own (`index <- 1 + (n - 1) * p`, 1-based, shifted at the
/// point of indexing) and the stored `h` is the same `index - lo` it would
/// have computed inline, so nothing changes numerically.
///
/// The indices are positions in the *partition buffer* (see
/// [`gene_bridge`]): the case group occupies `[0, n1)` ascending, so its
/// `lo`/`hi` are used as they are; the control group occupies `[n1, n)`
/// **descending**, so its order statistic `t` sits at `n - 1 - t` and the
/// plan is built with that map applied. Which of `a` and `b` is the larger
/// does not matter to `(1 - h) * a + h * b` — `a` is always `x[lo]` and
/// `b` always `x[hi]`, exactly as before the map.
struct Type7Plan {
    lo: Vec<usize>,
    hi: Vec<usize>,
    h: Vec<f64>,
}

impl Type7Plan {
    /// `n` is the group size; `offset` and `reversed` describe where the
    /// group's sorted values live in the partition buffer.
    fn new(n: usize, probs: &[f64], offset: usize, reversed: bool) -> Self {
        let nf = n as f64;
        let place = |t: usize| if reversed { offset + n - 1 - t } else { offset + t };
        let mut lo = Vec::with_capacity(probs.len());
        let mut hi = Vec::with_capacity(probs.len());
        let mut h = Vec::with_capacity(probs.len());
        for &p in probs {
            let index = 1.0 + (nf - 1.0) * p;
            let lo_f = index.floor();
            let hi_f = index.ceil();
            lo.push(place(lo_f as usize - 1));
            hi.push(place(hi_f as usize - 1));
            h.push(index - lo_f);
        }
        Type7Plan { lo, hi, h }
    }
}

/// `log2` of the type-7 quantiles of a group, written into `out`, given a
/// buffer holding the group's sorted values *and* one holding their logs,
/// addressed through a [`Type7Plan`] that knows where the group sits.
///
/// The branch is the whole point. R's guard interpolates only where
/// `h > 0` and `x[hi] != x[lo]`; everywhere else the quantile is exactly
/// `x[lo]`, and `log2(x[lo])` is `logsorted[lo]` — the same libm call on the
/// same input, made once per gene instead of once per permutation. Where
/// the guard fires, the interpolated value is formed with R's
/// `(1 - h) * a + h * b` and `log2` is called on it, which is what the NumPy
/// path does. The results are therefore the same bits by construction, not
/// by tolerance.
///
/// On NumPy's `linspace` grid the nodes of the smaller group land within an
/// ulp of integers rather than on them, so `h` is occasionally a few
/// `1e-16` rather than zero and the guard does fire on a handful of nodes
/// even on a balanced design. That is the NumPy path's behaviour and it is
/// reproduced, not corrected.
#[inline]
fn type7_sorted_log(sorted: &[f64], logsorted: &[f64], plan: &Type7Plan, out: &mut [f64]) {
    for (k, slot) in out.iter_mut().enumerate() {
        let lo = plan.lo[k];
        let a = sorted[lo];
        let b = sorted[plan.hi[k]];
        let h = plan.h[k];
        *slot = if h > 0.0 && b != a {
            ((1.0 - h) * a + h * b).log2()
        } else {
            logsorted[lo]
        };
    }
}

/// Per-thread working memory for the gene loop. Allocated once per rayon
/// job, never per gene.
struct SubsetScratch {
    order: Vec<usize>,
    vals: Vec<f64>,
    sorted: Vec<f64>,
    logsorted: Vec<f64>,
    /// The two-ended partition of one permutation: cases in `[0, n1)`
    /// ascending, controls in `[n1, n)` descending. And their logs.
    part: Vec<f64>,
    logpart: Vec<f64>,
    lq1: Vec<f64>,
    lq0: Vec<f64>,
    bridge: Vec<f64>,
    s1: Vec<f64>,
    s2: Vec<f64>,
    safe: Vec<f64>,
    /// Every permutation's bridge from pass 1, `(B × m)`, so pass 2 can
    /// re-read instead of recompute. `None` when that would exceed
    /// [`STORE_CAP_BYTES`], in which case pass 2 recomputes.
    store: Option<Vec<f64>>,
}

/// Default per-thread cap on the pass-1 bridge store. 64 MB: at `B = 2000`
/// and `m = 100` the store is 1.6 MB and pass 2 is a re-read; at
/// `B = 10000` and `m = 2000` it would be 160 MB per thread and pass 2
/// recomputes instead. Either way the numbers are the same; only the work is.
const STORE_CAP_BYTES: usize = 64 << 20;

impl SubsetScratch {
    fn new(n: usize, m: usize, n_perms: usize, store_cap_bytes: usize) -> Self {
        let store_bytes = n_perms
            .checked_mul(m)
            .and_then(|x| x.checked_mul(std::mem::size_of::<f64>()));
        let store = match store_bytes {
            Some(bytes) if bytes <= store_cap_bytes => Some(vec![0.0; n_perms * m]),
            _ => None,
        };
        SubsetScratch {
            order: (0..n).collect(),
            vals: vec![0.0; n],
            sorted: vec![0.0; n],
            logsorted: vec![0.0; n],
            part: vec![0.0; n],
            logpart: vec![0.0; n],
            lq1: vec![0.0; m],
            lq0: vec![0.0; m],
            bridge: vec![0.0; m],
            s1: vec![0.0; m],
            s2: vec![0.0; m],
            safe: vec![0.0; m],
            store,
        }
    }
}

/// The bridge of one gene under one permutation, into `scr.bridge`.
///
/// `mask` is the permutation's label row (1 = case). The walk over `order`
/// visits the gene's values in ascending order and routes each to its
/// group, so both groups come out sorted without a sort, and the logs
/// travel alongside. The partition is two-ended — cases fill `part` from
/// the front, controls from the back — so the destination is a select
/// rather than a branch: the labels are random, so a branch here would
/// mispredict half the time, and it sits inside the innermost loop.
/// Controls therefore land in descending order, which the control group's
/// [`Type7Plan`] is built to read.
///
/// Then `R = log2 q1 - log2 q0` (two logs, then a subtraction — not
/// `log2(q1 / q0)`, which rounds differently), the sequential cumulative
/// sum, and `B_k = S_k - (k/m) S_m` with the division done first, as NumPy
/// evaluates `(k / m) * s[:, -1]`.
#[inline]
fn gene_bridge(
    scr: &mut SubsetScratch,
    mask: &[u8],
    plan1: &Type7Plan,
    plan0: &Type7Plan,
    frac: &[f64],
) {
    let n = scr.order.len();
    let m = scr.bridge.len();
    let mut front = 0usize;
    let mut back = n;
    for i in 0..n {
        let c = (mask[scr.order[i]] == 1) as usize;
        back -= 1 - c;
        let pos = c * front + (1 - c) * back;
        scr.part[pos] = scr.sorted[i];
        scr.logpart[pos] = scr.logsorted[i];
        front += c;
    }
    debug_assert_eq!(front, back);

    type7_sorted_log(&scr.part, &scr.logpart, plan1, &mut scr.lq1);
    type7_sorted_log(&scr.part, &scr.logpart, plan0, &mut scr.lq0);

    // R, then its running total, in place. Sequential, left to right.
    let mut s = 0.0_f64;
    for k in 0..m {
        s += scr.lq1[k] - scr.lq0[k];
        scr.bridge[k] = s;
    }
    let s_m = s;
    for (b_k, &f_k) in scr.bridge.iter_mut().zip(frac.iter()) {
        *b_k -= f_k * s_m;
    }
}

/// `max_k reduce((br_k - mu_k) / safe_k)` and the first `k` attaining it —
/// `numpy.max` and `numpy.argmax` on the same row, which take the first of
/// tied maxima because the comparison is strict.
#[inline(always)]
fn scan_max(br: &[f64], mu: &[f64], safe: &[f64], alt: Alternative) -> (f64, usize) {
    let mut best = alt.reduce((br[0] - mu[0]) / safe[0]);
    let mut best_k = 0usize;
    for (k, ((&b, &u), &s)) in br.iter().zip(mu.iter()).zip(safe.iter()).enumerate().skip(1) {
        let z = alt.reduce((b - u) / s);
        if z > best {
            best = z;
            best_k = k;
        }
    }
    (best, best_k)
}

/// The subset test's two passes for one gene. Writes the gene's null row,
/// `mu`, `sd`, and returns `(statistic, argmax_k)` with `argmax_k` 1-based.
#[allow(clippy::too_many_arguments)]
fn one_gene_subset(
    row: &[f64],
    b_obs_row: &[f64],
    masks: &[u8],
    n_perms: usize,
    plan1: &Type7Plan,
    plan0: &Type7Plan,
    frac: &[f64],
    alt: Alternative,
    scr: &mut SubsetScratch,
    null_row: &mut [f64],
    mu: &mut [f64],
    sd: &mut [f64],
) -> (f64, i64) {
    let n = row.len();
    let m = mu.len();

    // Sort the row once. Ties are equal values, so an unstable sort changes
    // nothing any permutation reads.
    scr.vals.copy_from_slice(row);
    for (i, slot) in scr.order.iter_mut().enumerate() {
        *slot = i;
    }
    {
        let vals = &scr.vals;
        scr.order.sort_unstable_by(|&a, &b| {
            vals[a].partial_cmp(&vals[b]).expect("non-finite value in kernel input")
        });
    }
    for i in 0..n {
        let v = scr.vals[scr.order[i]];
        scr.sorted[i] = v;
        scr.logsorted[i] = v.log2();
    }

    // Pass 1: the null moments of the bridge at every width, accumulated
    // over permutations in order — `s1 += bb; s2 += bb * bb`.
    for v in scr.s1.iter_mut() {
        *v = 0.0;
    }
    for v in scr.s2.iter_mut() {
        *v = 0.0;
    }
    for b in 0..n_perms {
        gene_bridge(scr, &masks[b * n..(b + 1) * n], plan1, plan0, frac);
        for k in 0..m {
            let bb = scr.bridge[k];
            scr.s1[k] += bb;
            scr.s2[k] += bb * bb;
        }
        if let Some(store) = scr.store.as_mut() {
            store[b * m..(b + 1) * m].copy_from_slice(&scr.bridge);
        }
    }
    let bf = n_perms as f64;
    for (k, (mu_k, sd_k)) in mu.iter_mut().zip(sd.iter_mut()).enumerate() {
        *mu_k = scr.s1[k] / bf;
        *sd_k = (scr.s2[k] / bf - *mu_k * *mu_k).max(0.0).sqrt();
    }
    // B_m is identically zero, so the last width carries no information and
    // is excluded; so is any width whose null spread is zero. An excluded
    // width divides by +inf and contributes ±0.0 to the maximum — that is
    // what the NumPy path does, and it is reproduced rather than skipped,
    // because a scan whose every usable z is negative has maximum 0.
    for (k, (safe_k, &sd_k)) in scr.safe.iter_mut().zip(sd.iter()).enumerate() {
        *safe_k = if sd_k > 0.0 && k != m - 1 { sd_k } else { f64::INFINITY };
    }

    // Pass 2: the standardized maximum, from the stored bridges or by
    // recomputing them.
    match scr.store.take() {
        Some(store) => {
            for (b, slot) in null_row.iter_mut().enumerate() {
                *slot = scan_max(&store[b * m..(b + 1) * m], mu, &scr.safe, alt).0;
            }
            scr.store = Some(store);
        }
        None => {
            for (b, slot) in null_row.iter_mut().enumerate() {
                gene_bridge(scr, &masks[b * n..(b + 1) * n], plan1, plan0, frac);
                *slot = scan_max(&scr.bridge, mu, &scr.safe, alt).0;
            }
        }
    }

    // The observed statistic and the (1-based) width it peaked at.
    let (stat, best_k) = scan_max(b_obs_row, mu, &scr.safe, alt);
    (stat, best_k as i64 + 1)
}

/// The subset test's permutation loop, on the shift-corrected matrix.
///
/// Returns `(statistic, null, mu, sd, argmax_k)`: the per-gene standardized
/// maximum, the `(genes × permutations)` null, the bridge's null moments at
/// every width `(genes × m)`, and the 1-based width at which the observed
/// scan peaked. Identical to `wade.permutation._subset_null_numpy`, which is
/// the contract; see the module docs for how that is achieved at speed.
///
/// `xs` must already be divided by the fitted fold change — the caller owns
/// that — and must be finite and strictly positive, because the curve is a
/// difference of logarithms. `b_obs` is the observed bridge, `perms` the
/// `(B × n)` label matrix, `probs` the descending grid.
///
/// `store_cap_bytes` overrides the per-thread cap on the pass-1 bridge
/// store (default [`STORE_CAP_BYTES`]). It exists so the recompute path can
/// be tested at small sizes; production callers leave it alone.
#[pyfunction]
#[pyo3(signature = (xs, b_obs, perms, probs, alternative, store_cap_bytes = None))]
#[allow(clippy::type_complexity)]
fn subset_null<'py>(
    py: Python<'py>,
    xs: PyReadonlyArray2<'py, f64>,
    b_obs: PyReadonlyArray2<'py, f64>,
    perms: PyReadonlyArray2<'py, i64>,
    probs: PyReadonlyArray1<'py, f64>,
    alternative: &str,
    store_cap_bytes: Option<usize>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<i64>>,
)> {
    let alt = Alternative::parse(alternative)?;
    let xv = xs.as_array();
    let bv = b_obs.as_array();
    let pv = perms.as_array();
    let probs_slice: Vec<f64> = probs.as_array().iter().copied().collect();

    let g = xv.nrows();
    let n = xv.ncols();
    let n_perms = pv.nrows();
    let m = probs_slice.len();

    if m == 0 {
        return Err(PyValueError::new_err("probs must be non-empty"));
    }
    if !probs_slice.iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p)) {
        return Err(PyValueError::new_err("probs must be finite and within [0, 1]"));
    }
    if pv.ncols() != n {
        return Err(PyValueError::new_err(format!(
            "perms has {} columns but the matrix has {} samples",
            pv.ncols(),
            n
        )));
    }
    if n_perms == 0 {
        return Err(PyValueError::new_err(
            "perms must have at least one row; the null moments are undefined otherwise",
        ));
    }
    if bv.nrows() != g || bv.ncols() != m {
        return Err(PyValueError::new_err(format!(
            "b_obs must have shape (genes, len(probs)) = ({}, {}); got ({}, {})",
            g,
            m,
            bv.nrows(),
            bv.ncols()
        )));
    }
    // The curve is log2 q1 - log2 q0, so a zero or a negative value would
    // produce -inf or NaN silently and the maximum would be meaningless.
    // The NumPy path raises on a non-positive quantile; this raises on a
    // non-positive value, which is the same condition one step earlier.
    for (idx, &v) in xv.iter().enumerate() {
        if !v.is_finite() {
            return Err(PyValueError::new_err(format!(
                "the shift-corrected matrix contains non-finite values (gene {}, sample {})",
                idx / n.max(1),
                idx % n.max(1)
            )));
        }
        if v <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "the subset test needs a strictly positive matrix, because the log-ratio \
                 curve is a difference of logarithms; gene {} has a non-positive value at \
                 sample {}. The continuity jitter normally guarantees this — if you passed \
                 an already-normalized matrix, it has exact zeros in it.",
                idx / n.max(1),
                idx % n.max(1)
            )));
        }
    }

    // Group sizes are preserved by label exchange, so n1 and n0 are
    // constants of the run and the bracketing plans can be built once.
    // Verified rather than assumed, as the mean-shift kernel does.
    let mut n1 = 0usize;
    let mut n0 = 0usize;
    let mut masks = vec![0u8; n_perms * n];
    for (b, row) in pv.rows().into_iter().enumerate() {
        let mut r1 = 0usize;
        let mut r0 = 0usize;
        for (j, &lab) in row.iter().enumerate() {
            match lab {
                1 => {
                    r1 += 1;
                    masks[b * n + j] = 1;
                }
                0 => r0 += 1,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "perms must contain only 0 and 1; row {} has {}",
                        b, other
                    )))
                }
            }
        }
        if b == 0 {
            n1 = r1;
            n0 = r0;
        } else if r1 != n1 || r0 != n0 {
            return Err(PyValueError::new_err(format!(
                "permutation row {} has group sizes (n1, n0) = ({}, {}) but row 0 has \
                 ({}, {}); label exchange preserves the group sizes",
                b, r1, r0, n1, n0
            )));
        }
        if r1.min(r0) < m {
            return Err(PyValueError::new_err(format!(
                "permutation row {} gives min(n1, n0) = {} but probs has {} entries; \
                 the grid may be capped below the design but never above it",
                b,
                r1.min(r0),
                m
            )));
        }
    }
    if n1 == 0 || n0 == 0 {
        return Err(PyValueError::new_err("both groups must be non-empty"));
    }

    let plan1 = Type7Plan::new(n1, &probs_slice, 0, false);
    let plan0 = Type7Plan::new(n0, &probs_slice, n1, true);
    let store_cap = store_cap_bytes.unwrap_or(STORE_CAP_BYTES);
    // `(k / m)` for k = 1..m, divided first as NumPy does, once.
    let frac: Vec<f64> = (1..=m).map(|k| k as f64 / m as f64).collect();

    let mut statistic = vec![0.0_f64; g];
    let mut null = vec![0.0_f64; g * n_perms];
    let mut mu = vec![0.0_f64; g * m];
    let mut sd = vec![0.0_f64; g * m];
    let mut argmax = vec![0_i64; g];

    py.detach(|| {
        null.par_chunks_mut(n_perms)
            .zip(mu.par_chunks_mut(m))
            .zip(sd.par_chunks_mut(m))
            .zip(statistic.par_iter_mut())
            .zip(argmax.par_iter_mut())
            .enumerate()
            .for_each_init(
                || (SubsetScratch::new(n, m, n_perms, store_cap), vec![0.0_f64; n], vec![0.0_f64; m]),
                |(scr, row_buf, obs_buf), (gene, ((((null_row, mu_row), sd_row), stat_slot), argmax_slot))| {
                    // Copy through a buffer so the hot loop sees contiguous
                    // slices whatever the input strides.
                    for (slot, v) in row_buf.iter_mut().zip(xv.row(gene).iter()) {
                        *slot = *v;
                    }
                    for (slot, v) in obs_buf.iter_mut().zip(bv.row(gene).iter()) {
                        *slot = *v;
                    }
                    let (stat, k) = one_gene_subset(
                        row_buf, obs_buf, &masks, n_perms, &plan1, &plan0, &frac, alt, scr,
                        null_row, mu_row, sd_row,
                    );
                    *stat_slot = stat;
                    *argmax_slot = k;
                },
            );
    });

    let null = Array2::from_shape_vec((g, n_perms), null)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mu = Array2::from_shape_vec((g, m), mu)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let sd = Array2::from_shape_vec((g, m), sd)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        Array1::from_vec(statistic).into_pyarray(py),
        null.into_pyarray(py),
        mu.into_pyarray(py),
        sd.into_pyarray(py),
        Array1::from_vec(argmax).into_pyarray(py),
    ))
}

// ---------------------------------------------------------------------
// The fold-change fit
// ---------------------------------------------------------------------

/// Mean of the middle half of `buf` (positions `n/4 .. n - n/4` of the
/// sorted order), by selection: pin the upper quartile bound over the whole
/// buffer, then the lower bound within the left partition, and sum what
/// lies between. The same middle set `_middle_mean` selects on the NumPy
/// side; the sum order differs, which is within the fit paths' documented
/// tolerance (they already differ through the draws).
///
/// `total_cmp`, not `partial_cmp`: a sample with a zero library size gives
/// that column an infinite `alpha`, so its zero counts scale to NaN — an
/// input the NumPy path tolerates because `np.partition` orders NaN last.
/// `total_cmp` orders NaN last the same way, so such a column falls in the
/// discarded top quarter exactly as it does on the NumPy side instead of
/// panicking the rayon workers.
#[inline]
fn middle_mean(buf: &mut [f64]) -> f64 {
    let n = buf.len();
    let lo = n / 4;
    let hi = n - n / 4;
    if hi < n {
        buf.select_nth_unstable_by(hi, f64::total_cmp);
    }
    if lo > 0 {
        buf[..hi].select_nth_unstable_by(lo, f64::total_cmp);
    }
    let mut total = 0.0_f64;
    for &v in &buf[lo..hi] {
        total += v;
    }
    total / (hi - lo) as f64
}

/// One gene's bisection for the thinning-matched fold change — the loop of
/// `wade.thinning._fit_fold_change_alpha`, given that gene's active-side
/// counts and scale and the untouched side's interquartile mean. Returns
/// `0.5 * (lo + hi)` in natural log.
///
/// The generator is re-seeded identically at every bisection step (common
/// random numbers, as the NumPy paths do), and the seed is a function of
/// `(seed, global gene index)` alone — so the result does not depend on how
/// the genes were chunked or which thread ran them.
fn one_gene_fit(
    c_active: &[i64],
    a_active: &[f64],
    m_static: f64,
    seed: u64,
    gene_index: u64,
    iters: usize,
    max_log_fold: f64,
    thinned: &mut Vec<f64>,
) -> f64 {
    use rand::SeedableRng;
    use rand_distr::Distribution;

    // splitmix64 over (seed, gene) to decorrelate the per-gene streams.
    let mut z = seed ^ gene_index.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut next = || {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut x = z;
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    };
    let gene_seed: [u8; 16] = {
        let a = next().to_le_bytes();
        let b = next().to_le_bytes();
        let mut s = [0u8; 16];
        s[..8].copy_from_slice(&a);
        s[8..].copy_from_slice(&b);
        s
    };

    let mut lo = 0.0_f64;
    let mut hi = max_log_fold;
    for _ in 0..iters {
        let mid = 0.5 * (lo + hi);
        let keep = (-mid).exp();
        let mut rng = rand_pcg::Pcg64Mcg::from_seed(gene_seed);
        thinned.clear();
        for (&c, &a) in c_active.iter().zip(a_active.iter()) {
            let t = if c > 0 {
                rand_distr::Binomial::new(c as u64, keep)
                    .expect("keep probability out of range in fit kernel")
                    .sample(&mut rng) as f64
            } else {
                0.0
            };
            thinned.push(t * a);
        }
        let m_thin = middle_mean(thinned);
        if m_thin > m_static {
            lo = mid; // the thinned side is still higher: thin more
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// The fold-change fit's bisection, parallel over genes — `docs/scaling.md`
/// §3.3. Opt-in (`fit_backend="rust"`): unlike the two permutation kernels it
/// is **not** bitwise against the NumPy path, because no two binomial
/// samplers consume randomness alike; it fits the same objective and is held
/// to the NumPy path statistically. Deterministic given `seed` and the
/// global gene indices, whatever the chunking or thread schedule.
///
/// `counts` and `alpha` are the chunk's rows; `active` marks each gene's
/// thinned side's columns (row-wise bool mask); `m_static` is the untouched
/// side's interquartile mean. Returns `0.5 * (lo + hi)` per gene in natural
/// log; the caller applies `exp`, the both-middles-empty override and the
/// direction convention.
#[pyfunction]
fn fit_bisect<'py>(
    py: Python<'py>,
    counts: PyReadonlyArray2<'py, f64>,
    alpha: PyReadonlyArray2<'py, f64>,
    active: PyReadonlyArray2<'py, bool>,
    m_static: PyReadonlyArray1<'py, f64>,
    seed: u64,
    gene_offset: u64,
    iters: usize,
    max_log_fold: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let cv = counts.as_array();
    let av = alpha.as_array();
    let actv = active.as_array();
    let msv = m_static.as_array();
    let g = cv.nrows();
    let n = cv.ncols();
    if av.nrows() != g || av.ncols() != n || actv.nrows() != g || actv.ncols() != n {
        return Err(PyValueError::new_err(
            "counts, alpha and active must share one (genes, samples) shape",
        ));
    }
    if msv.len() != g {
        return Err(PyValueError::new_err("m_static must have one value per gene"));
    }
    for &v in cv.iter() {
        if !v.is_finite() || v < 0.0 || v != v.round() {
            return Err(PyValueError::new_err(
                "the fit kernel needs finite, non-negative integer counts",
            ));
        }
    }

    let ms: Vec<f64> = msv.iter().copied().collect();
    let mut out = vec![0.0_f64; g];
    py.detach(|| {
        out.par_iter_mut().enumerate().for_each_init(
            || (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n)),
            |(c_buf, a_buf, thinned), (gene, slot)| {
                c_buf.clear();
                a_buf.clear();
                for j in 0..n {
                    if actv[(gene, j)] {
                        c_buf.push(cv[(gene, j)] as i64);
                        a_buf.push(av[(gene, j)]);
                    }
                }
                *slot = one_gene_fit(
                    c_buf, a_buf, ms[gene], seed, gene_offset + gene as u64, iters,
                    max_log_fold, thinned,
                );
            },
        );
    });
    Ok(Array1::from_vec(out).into_pyarray(py))
}

#[pymodule]
fn _kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(null_statistics, m)?)?;
    m.add_function(wrap_pyfunction!(subset_null, m)?)?;
    m.add_function(wrap_pyfunction!(fit_bisect, m)?)?;
    m.add("__doc__", "Native permutation kernel for WADE.")?;
    Ok(())
}
