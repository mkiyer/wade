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
//! Permutations are independent, so the loop is parallelized across them
//! with rayon. That changes nothing numerically: each permutation's
//! arithmetic is self-contained and no accumulation crosses iterations.

use numpy::ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Type-7 quantiles of an already-sorted ascending slice, written into `out`.
///
/// Mirrors R's `quantile.default(type = 7)`:
///
/// ```text
/// index <- 1 + (n - 1) * p
/// lo <- floor(index); hi <- ceiling(index)
/// qs  <- x[lo]
/// i   <- which(index > lo & x[hi] != qs)
/// qs[i] <- (1 - h) * qs[i] + h * x[hi][i]
/// ```
///
/// The 1-based index arithmetic is kept as-is and shifted to 0-based only
/// at the point of indexing, so the floor/ceil see exactly the values R's
/// do.
#[inline]
fn type7_sorted(sorted: &[f64], probs: &[f64], out: &mut [f64]) {
    let n = sorted.len() as f64;
    for (slot, &p) in out.iter_mut().zip(probs.iter()) {
        let index = 1.0 + (n - 1.0) * p;
        let lo = index.floor();
        let hi = index.ceil();
        let a = sorted[lo as usize - 1];
        let b = sorted[hi as usize - 1];
        let h = index - lo;
        *slot = if h > 0.0 && b != a {
            (1.0 - h) * a + h * b
        } else {
            a
        };
    }
}

/// Ascending sort. The inputs are normalized abundances and are finite by
/// construction (the caller validates), so a total order exists and
/// `partial_cmp` cannot fail.
#[inline]
fn sort_ascending(buf: &mut [f64]) {
    buf.sort_unstable_by(|a, b| a.partial_cmp(b).expect("non-finite value in kernel input"));
}

fn one_permutation(
    x: &ArrayView2<'_, f64>,
    labels: &[i64],
    probs: &[f64],
    idx1: &mut Vec<usize>,
    idx0: &mut Vec<usize>,
    buf1: &mut Vec<f64>,
    buf0: &mut Vec<f64>,
    q1: &mut Vec<f64>,
    q0: &mut Vec<f64>,
    out: &mut [f64],
) {
    idx1.clear();
    idx0.clear();
    for (j, &lab) in labels.iter().enumerate() {
        if lab == 1 {
            idx1.push(j);
        } else if lab == 0 {
            idx0.push(j);
        }
    }

    let nprobs = probs.len();
    buf1.resize(idx1.len(), 0.0);
    buf0.resize(idx0.len(), 0.0);
    q1.resize(nprobs, 0.0);
    q0.resize(nprobs, 0.0);

    for gene in 0..x.nrows() {
        let row = x.row(gene);
        for (slot, &j) in buf1.iter_mut().zip(idx1.iter()) {
            *slot = row[j];
        }
        for (slot, &j) in buf0.iter_mut().zip(idx0.iter()) {
            *slot = row[j];
        }
        sort_ascending(buf1);
        sort_ascending(buf0);
        type7_sorted(buf1, probs, q1);
        type7_sorted(buf0, probs, q0);

        // Sequential accumulation, matching R's rowSums association.
        let mut total = 0.0_f64;
        for i in 0..nprobs {
            total += q1[i] - q0[i];
        }
        out[gene] = total / nprobs as f64;
    }
}

/// The mean-shift null for a supplied set of label permutations.
///
/// Returns a `(n_perms, n_genes)` matrix. The Python wrapper transposes it to
/// the `(genes, permutations)` orientation the rest of the package uses;
/// transposing a view is free, and building permutation-major here keeps each
/// permutation's writes contiguous, which is what makes the parallel loop cheap.
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
    if !xv.iter().all(|v| v.is_finite()) {
        return Err(PyValueError::new_err(
            "the normalized matrix contains non-finite values",
        ));
    }

    // Group sizes are preserved by label exchange, so nprobs is a constant of
    // the run. Verified rather than assumed: a malformed matrix would
    // otherwise index past the end of a quantile buffer.
    for (b, row) in pv.rows().into_iter().enumerate() {
        let mut n1 = 0usize;
        let mut n0 = 0usize;
        for &lab in row.iter() {
            match lab {
                1 => n1 += 1,
                0 => n0 += 1,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "perms must contain only 0 and 1; row {} has {}",
                        b, other
                    )))
                }
            }
        }
        if n1.min(n0) != nprobs {
            return Err(PyValueError::new_err(format!(
                "permutation row {} gives min(n1, n0) = {} but probs has {} entries",
                b,
                n1.min(n0),
                nprobs
            )));
        }
    }

    let mut out = vec![0.0_f64; n_perms * g];

    py.detach(|| {
        let labels: Vec<Vec<i64>> = pv
            .rows()
            .into_iter()
            .map(|r| r.iter().copied().collect())
            .collect();

        out.par_chunks_mut(g).enumerate().for_each(|(b, chunk)| {
            let mut idx1 = Vec::with_capacity(n);
            let mut idx0 = Vec::with_capacity(n);
            let mut buf1 = Vec::with_capacity(n);
            let mut buf0 = Vec::with_capacity(n);
            let mut q1 = Vec::with_capacity(nprobs);
            let mut q0 = Vec::with_capacity(nprobs);
            one_permutation(
                &xv, &labels[b], &probs_slice, &mut idx1, &mut idx0, &mut buf1,
                &mut buf0, &mut q1, &mut q0, chunk,
            );
        });
    });

    let arr = Array2::from_shape_vec((n_perms, g), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// Type-7 quantiles, exposed so the kernel's own implementation can be
/// tested directly against the NumPy one rather than only through the
/// null matrices.
#[pyfunction]
fn type7_quantiles<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    probs: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let xv = x.as_array();
    let probs_slice: Vec<f64> = probs.as_array().iter().copied().collect();
    let g = xv.nrows();
    let n = xv.ncols();
    if n == 0 {
        return Err(PyValueError::new_err("cannot take quantiles of an empty group"));
    }
    let m = probs_slice.len();
    let mut out = vec![0.0_f64; g * m];
    let mut buf = vec![0.0_f64; n];
    for gene in 0..g {
        for (slot, v) in buf.iter_mut().zip(xv.row(gene).iter()) {
            *slot = *v;
        }
        sort_ascending(&mut buf);
        type7_sorted(&buf, &probs_slice, &mut out[gene * m..(gene + 1) * m]);
    }
    let arr = Array2::from_shape_vec((g, m), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// Number of threads rayon will use, for reporting.
#[pyfunction]
fn num_threads() -> usize {
    rayon::current_num_threads()
}

#[pymodule]
fn _kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(null_statistics, m)?)?;
    m.add_function(wrap_pyfunction!(type7_quantiles, m)?)?;
    m.add_function(wrap_pyfunction!(num_threads, m)?)?;
    m.add("__doc__", "Native permutation kernel for WADE.")?;
    Ok(())
}
