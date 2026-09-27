//! Exact empirical upward/downward quantile areas and the audited h-th-exceedance
//! stopping rule. Each gene's sums are ordered and each direction keeps its own
//! stopping record under the shared label draws; parallelism cannot change either.
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn update_areas<'py>(
    py: Python<'py>, order: PyReadonlyArray2<'py, u64>,
    gaps: PyReadonlyArray2<'py, f64>, labels: PyReadonlyArray2<'py, u8>,
    thresholds: PyReadonlyArray2<'py, f64>, exceed: PyReadonlyArray2<'py, u64>,
    tried: PyReadonlyArray2<'py, u64>, active: (bool, bool), cases: usize, stop: u64,
) -> PyResult<(Bound<'py, PyArray2<u64>>, Bound<'py, PyArray2<u64>>)> {
    let (genes, samples) = order.as_array().dim();
    if samples < 2 || cases == 0 || cases >= samples || stop == 0
        || gaps.as_array().dim() != (genes, samples - 1)
        || labels.as_array().ncols() != samples
        || thresholds.as_array().dim() != (genes, 2)
        || exceed.as_array().dim() != (genes, 2) || tried.as_array().dim() != (genes, 2)
        || samples.checked_mul(samples).is_none_or(|n| n > i64::MAX as usize)
    {
        return Err(PyValueError::new_err("invalid area state dimensions or group sizes"));
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
        return Err(PyValueError::new_err("invalid area state values or permutations"));
    }
    let active = [active.0, active.1];
    // One (exceedances, draws) record per gene and direction: up, then down.
    let mut records: Vec<[(u64, u64); 2]> = (0..genes)
        .map(|g| [(hits[2 * g], draws[2 * g]), (hits[2 * g + 1], draws[2 * g + 1])])
        .collect();
    py.detach(|| {
        records.par_iter_mut().enumerate().for_each(|(gene, record)| {
            let live = |r: &[(u64, u64); 2], d: usize| active[d] && r[d].0 < stop;
            if !live(record, 0) && !live(record, 1) { return; }
            let order = &indices[gene * samples..(gene + 1) * samples];
            let gap = &delta[gene * (samples - 1)..(gene + 1) * (samples - 1)];
            let limit = &threshold[2 * gene..2 * gene + 2];
            for row in lab.chunks(samples) {
                let mut nc = 0i64;
                let mut area = [0.0f64; 2];
                for j in 0..samples - 1 {
                    nc += row[order[j] as usize] as i64;
                    // Control CDF minus case CDF, scaled: positive where cases are higher.
                    let mass = (cases * (j + 1)) as i64 - (samples as i64) * nc;
                    if mass > 0 {
                        area[0] += (mass as f64) * gap[j];
                    } else {
                        area[1] += ((-mass) as f64) * gap[j];
                    }
                }
                for d in 0..2 {
                    if live(record, d) {
                        record[d].1 += 1;
                        if area[d] >= limit[d] { record[d].0 += 1; }
                    }
                }
                if !live(record, 0) && !live(record, 1) { break; }
            }
        });
    });
    let mut k = Array2::<u64>::zeros((genes, 2));
    let mut t = Array2::<u64>::zeros((genes, 2));
    for (gene, record) in records.iter().enumerate() {
        for d in 0..2 {
            k[[gene, d]] = record[d].0;
            t[[gene, d]] = record[d].1;
        }
    }
    Ok((k.into_pyarray(py), t.into_pyarray(py)))
}

#[pymodule]
fn _kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(update_areas, m)?)?;
    Ok(())
}
