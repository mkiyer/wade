//! Exact empirical W1 and the audited h-th-exceedance stopping rule.
//! Each gene's sum is ordered; parallelism cannot change its stopping time.
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn update_w1<'py>(
    py: Python<'py>, order: PyReadonlyArray2<'py, u64>,
    gaps: PyReadonlyArray2<'py, f64>, labels: PyReadonlyArray2<'py, u8>,
    thresholds: PyReadonlyArray1<'py, f64>, exceed: PyReadonlyArray1<'py, u64>,
    tried: PyReadonlyArray1<'py, u64>, cases: usize, stop: u64,
) -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u64>>)> {
    let (genes, samples) = order.as_array().dim();
    if samples < 2 || cases == 0 || cases >= samples || stop == 0
        || gaps.as_array().dim() != (genes, samples - 1)
        || labels.as_array().ncols() != samples
        || thresholds.as_array().len() != genes || exceed.as_array().len() != genes || tried.as_array().len() != genes
        || samples.checked_mul(samples).is_none_or(|n| n > i64::MAX as usize)
    {
        return Err(PyValueError::new_err("invalid W1 state dimensions or group sizes"));
    }
    let indices = order.as_slice()?;
    let delta = gaps.as_slice()?;
    let lab = labels.as_slice()?;
    let threshold = thresholds.as_slice()?;
    let hits = exceed.as_slice()?;
    let draws = tried.as_slice()?;
    if indices.iter().any(|&i| i >= samples as u64)
        || delta.iter().any(|&v| !v.is_finite() || v < 0.0)
        || threshold.iter().any(|&v| !v.is_finite() || v < 0.0)
        || hits.iter().zip(draws).any(|(&k, &t)| k > stop || t < k)
        || lab.chunks(samples).any(|row| {
            row.iter().any(|&v| v > 1) || row.iter().map(|&v| v as usize).sum::<usize>() != cases
        })
    {
        return Err(PyValueError::new_err("invalid W1 state values or permutations"));
    }
    let mut result: Vec<(u64, u64)> = hits.iter().copied().zip(draws.iter().copied()).collect();
    py.detach(|| {
        result.par_iter_mut().enumerate().for_each(|(gene, (k, t))| {
            if *k == stop { return; }
            let order = &indices[gene * samples..(gene + 1) * samples];
            let gap = &delta[gene * (samples - 1)..(gene + 1) * (samples - 1)];
            for row in lab.chunks(samples) {
                let mut nc = 0i64;
                let mut area = 0.0;
                for j in 0..samples - 1 {
                    nc += row[order[j] as usize] as i64;
                    let mass = ((samples as i64) * nc - (cases * (j + 1)) as i64).abs();
                    area += (mass as f64) * gap[j];
                }
                *t += 1;
                if area >= threshold[gene] {
                    *k += 1;
                    if *k == stop { break; }
                }
            }
        });
    });
    let (hits, draws): (Vec<_>, Vec<_>) = result.into_iter().unzip();
    Ok((Array1::from_vec(hits).into_pyarray(py), Array1::from_vec(draws).into_pyarray(py)))
}

#[pymodule]
fn _kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(update_w1, m)?)?;
    Ok(())
}
