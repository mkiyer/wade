//! Research only: the subset-test constructions compared in docs/subset-null-derivation.md and
//! docs/subset-null-plan.md, kept so research/subset_null_study.py reproduces those documents.
//! Each shuffle removes a given registration from the cases, and each labelling's cases carry it
//! again; `relative` picks the normalizer of the depth (0 none, 1 the median case + 1, 2 the
//! majority side's mean + 1, 3 the cases' geometric mean of count + 1, 4 the labelling's own
//! W1 fold times the median control + 1, 5 the case mean + 1). Delete with the study.
//!
//! Also the prototype of the rank test (docs/subset-null-plan.md, step 2), measured under the
//! count model by research/count_model.py before it moves into the package's kernel.
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use super::{cases_in, cut, Grid, Side, TIE_RTOL};

/// The released split: size, log2 fold over the mean shift's prediction, and the walk's depth.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn split_study<'py>(
    py: Python<'py>, case: PyReadonlyArray2<'py, f64>, control: PyReadonlyArray2<'py, f64>, linear: bool,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (genes, n1) = case.as_array().dim();
    let n0 = control.as_array().ncols();
    if n1 == 0 || n0 == 0 || control.as_array().nrows() != genes
        || !case.is_c_contiguous() || !control.is_c_contiguous()
    {
        return Err(PyValueError::new_err("cases and controls need one nonempty C-ordered row per gene"));
    }
    let (c, d) = (case.as_slice()?, control.as_slice()?);
    let grid = Grid::new(n1, n0);
    let found: Vec<(i64, f64, f64)> = py.detach(|| {
        (0..genes).into_par_iter().map_init(|| vec![0.0; n1], |running, g| {
            let case = &c[g * n1..(g + 1) * n1];
            let (low, rounding, _) = grid.walk(case, &d[g * n0..(g + 1) * n0], linear, &mut running[..n1 - 1]);
            if -low <= rounding {
                return (0, f64::NAN, 0.0);
            }
            let (cut, depth) = (cut(&running[..n1 - 1], low, rounding), -low);
            let (total, below) = (case.iter().sum::<f64>(), case[..cut].iter().sum::<f64>());
            let due_below = below / n1 as f64 - running[cut - 1];
            let (size, seen, due) = if n1 - cut <= cut {
                let size = (n1 - cut) as f64;
                (n1 - cut, (total - below) / size, (total / n1 as f64 - due_below) * n1 as f64 / size)
            } else {
                (cut, below / cut as f64, due_below * n1 as f64 / cut as f64)
            };
            let fold = if linear { ((seen + 1.0) / (due + 1.0)).log2() } else { seen - due };
            (size as i64, fold, depth)
        }).collect()
    });
    let (mut cases, mut folds, mut depths) = (Vec::with_capacity(genes), Vec::with_capacity(genes), Vec::with_capacity(genes));
    for (k, f, h) in found {
        cases.push(k);
        folds.push(f);
        depths.push(h);
    }
    Ok((cases.into_pyarray(py), folds.into_pyarray(py), depths.into_pyarray(py)))
}

struct StudyWalker<'a> {
    grid: &'a Grid,
    values: &'a [f64],
    at: &'a [i64],
    shift: f64,
    linear: bool,
    relative: u8,
    side: Side,
    case: Vec<f64>,
    control: Vec<f64>,
    running: Vec<f64>,
    logs: Vec<f64>,
    case_log: Vec<f64>,
    control_log: Vec<f64>,
    segments: Vec<(f64, u64)>,
}

impl<'a> StudyWalker<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(grid: &'a Grid, values: &'a [f64], at: &'a [i64], shift: f64, linear: bool, relative: u8, side: Side) -> Self {
        let (n1, n0) = (grid.n1, grid.n0);
        let logged = linear && relative >= 3;
        let logs = if logged { values.iter().map(|&v| (v + 1.0).ln()).collect() } else { Vec::new() };
        let (cl, kl) = if logged { (vec![0.0; n1 + 1], vec![0.0; n0 + 1]) } else { (Vec::new(), Vec::new()) };
        StudyWalker { grid, values, at, shift, linear, relative, side,
                      case: vec![0.0; n1 + 1], control: vec![0.0; n0 + 1], running: vec![0.0; n1],
                      logs, case_log: cl, control_log: kl, segments: Vec::with_capacity(n1 + n0) }
    }

    fn score(&mut self, labels: &[u8]) -> f64 {
        let (n1, n0) = (self.grid.n1, self.grid.n0);
        let (mut a, mut b) = (0, 0);
        let logged = !self.logs.is_empty();
        for (idx, (&v, &i)) in self.values.iter().zip(self.at).enumerate() {
            let is_case = labels[i as usize] as usize;
            self.case[a] = if self.linear { (v + 1.0) * self.shift - 1.0 } else { v + self.shift };
            self.control[b] = v;
            if logged {
                self.case_log[a] = self.logs[idx];
                self.control_log[b] = self.logs[idx];
            }
            a += is_case;
            b += 1 - is_case;
        }
        let (low, rounding, mean1) = self.grid.walk(&self.case[..n1], &self.control[..n0], self.linear, &mut self.running[..n1 - 1]);
        if -low <= rounding {
            return 0.0;
        }
        let k = cut(&self.running[..n1 - 1], low, rounding);
        let up = n1 - k <= k;
        let depth = match (self.relative, self.linear) {
            (1, true) => -low / (0.5 * (self.case[(n1 - 1) / 2] + self.case[n1 / 2]) + 1.0),
            (2, true) => {
                let (from, to) = if up { (0, k) } else { (k, n1) };
                let majority = self.case[from..to].iter().sum::<f64>() / (to - from) as f64;
                -low / (majority + 1.0)
            }
            (3, true) => {
                let mean_log = self.case_log[..n1].iter().sum::<f64>() / n1 as f64;
                -low / (self.shift.ln() + mean_log).exp()
            }
            (5, true) => -low / (mean1 + 1.0),
            (4, true) => {
                let fold = (self.shift.ln() + w1_log_shift(&self.case_log[..n1], &self.control_log[..n0], &mut self.segments)).exp();
                -low / (fold * (0.5 * (self.control[(n0 - 1) / 2] + self.control[n0 / 2]) + 1.0))
            }
            _ => -low,
        };
        match (self.side, up) {
            (Side::Both, _) => depth,
            (Side::Up, true) | (Side::Down, false) => depth,
            _ => -depth,
        }
    }
}

/// The lower weighted median of the log quantile difference over the merged breakpoints.
fn w1_log_shift(case_log: &[f64], control_log: &[f64], segments: &mut Vec<(f64, u64)>) -> f64 {
    let (n1, n0) = (case_log.len(), control_log.len());
    segments.clear();
    let (mut i, mut j, mut at) = (0usize, 0usize, 0u64);
    while i < n1 && j < n0 {
        let (ci, cj) = ((i as u64 + 1) * n0 as u64, (j as u64 + 1) * n1 as u64);
        let next = ci.min(cj);
        segments.push((case_log[i] - control_log[j], next - at));
        at = next;
        if ci == next { i += 1; }
        if cj == next { j += 1; }
    }
    segments.sort_unstable_by(|p, q| p.0.partial_cmp(&q.0).unwrap_or(std::cmp::Ordering::Equal));
    let total = (n1 * n0) as u64;
    let mut cum = 0u64;
    for &(value, weight) in segments.iter() {
        cum += weight;
        if 2 * cum >= total {
            return value;
        }
    }
    segments.last().map_or(0.0, |s| s.0)
}

/// The study's shuffles: `shift` removed from the pooled cases and re-applied to each labelling's.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn tally_study<'py>(
    py: Python<'py>, pooled: PyReadonlyArray2<'py, f64>, order: PyReadonlyArray2<'py, i64>,
    shift: PyReadonlyArray1<'py, f64>, linear: bool, direction: &str, labels: PyReadonlyArray2<'py, u8>,
    observed: PyReadonlyArray1<'py, f64>, exceed: PyReadonlyArray1<'py, u64>,
    tried: PyReadonlyArray1<'py, u64>, stop: u64, maximum: u64, relative: u8,
) -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u64>>)> {
    let side = Side::parse(direction)?;
    let (genes, n) = pooled.as_array().dim();
    if !pooled.is_c_contiguous() || !order.is_c_contiguous() || order.as_array().dim() != (genes, n) {
        return Err(PyValueError::new_err("pooled values and order must be C-ordered genes x samples"));
    }
    let (values, at, shift, observed) = (pooled.as_slice()?, order.as_slice()?, shift.as_slice()?, observed.as_slice()?);
    if n == 0 || shift.len() != genes || observed.len() != genes || at.iter().any(|&i| i < 0 || i as usize >= n) {
        return Err(PyValueError::new_err("invalid walk inputs: shapes or sample order"));
    }
    let lab = labels.as_slice()?;
    let (hits, draws) = (exceed.as_slice()?, tried.as_slice()?);
    let n1 = lab.iter().take(n).map(|&v| v as usize).sum::<usize>();
    if !labels.is_c_contiguous() || labels.as_array().ncols() != n || hits.len() != genes
        || draws.len() != genes || n1 == 0 || n1 == n
        || lab.chunks(n).any(|row| row.iter().any(|&v| v > 1) || row.iter().map(|&v| v as usize).sum::<usize>() != n1)
    {
        return Err(PyValueError::new_err("invalid shuffles or records"));
    }
    let grid = Grid::new(n1, n - n1);
    let mut records: Vec<(u64, u64)> = hits.iter().copied().zip(draws.iter().copied()).collect();
    py.detach(|| {
        records.par_iter_mut().enumerate().for_each(|(g, record)| {
            if record.0 >= stop || record.1 >= maximum {
                return;
            }
            let row = g * n..(g + 1) * n;
            let mut walker = StudyWalker::new(&grid, &values[row.clone()], &at[row], shift[g], linear, relative, side);
            let bar = observed[g] - TIE_RTOL * observed[g].abs();
            for labelling in lab.chunks_exact(n) {
                record.1 += 1;
                if walker.score(labelling) >= bar {
                    record.0 += 1;
                }
                if record.0 >= stop || record.1 >= maximum {
                    break;
                }
            }
        });
    });
    let (e, t): (Vec<u64>, Vec<u64>) = records.into_iter().unzip();
    Ok((e.into_pyarray(py), t.into_pyarray(py)))
}

/// Brunner and Munzel's studentized rank statistic, cases minus controls, for one gene under one
/// labelling. `at` lists the samples in increasing order of their (untied) values. A case's
/// placement is the number of controls below it, a control's the number of cases below it; U,
/// the cases' total placement, is centred at n1 n0 / 2 and divided by the square root of
/// n1 var(case placements) + n0 var(control placements). Complete separation, where both
/// variances vanish, gives an infinite statistic of U's sign. One pass, integer sums.
fn brunner_munzel(at: &[i64], labels: &[u8], n1: usize) -> f64 {
    let n0 = at.len() - n1;
    let (mut cases, mut p, mut p2, mut q, mut q2) = (0u64, 0u64, 0u64, 0u64, 0u64);
    for (k, &i) in at.iter().enumerate() {
        if labels[i as usize] == 1 {
            let placement = k as u64 - cases;
            p += placement;
            p2 += placement * placement;
            cases += 1;
        } else {
            q += cases;
            q2 += cases * cases;
        }
    }
    let (m1, m0) = (n1 as u128, n0 as u128);
    let (p, p2, q, q2) = (p as u128, p2 as u128, q as u128, q2 as u128);
    let spread = (m1 * p2 - p * p) as f64 / (n1 - 1) as f64 + (m0 * q2 - q * q) as f64 / (n0 - 1) as f64;
    let centred = (2 * p) as f64 - (m1 * m0) as f64;                 // 2U - n1 n0, exact
    if spread > 0.0 {
        centred / (2.0 * spread.sqrt())
    } else if centred == 0.0 {
        0.0
    } else {
        centred.signum() * f64::INFINITY
    }
}

fn directed(side: Side, statistic: f64) -> f64 {
    match side {
        Side::Both => statistic.abs(),
        Side::Up => statistic,
        Side::Down => -statistic,
    }
}

fn rank_order<'a>(order: &'a PyReadonlyArray2<'a, i64>) -> PyResult<(usize, usize, &'a [i64])> {
    let (genes, n) = order.as_array().dim();
    let at = order.as_slice()?;
    if n == 0 || at.iter().any(|&i| i < 0 || i as usize >= n) {
        return Err(PyValueError::new_err("order must be C-ordered genes x samples of sample indices"));
    }
    Ok((genes, n, at))
}

/// Each gene's rank statistic under one labelling, for `direction` (both: its size).
#[pyfunction]
pub fn rank_score_study<'py>(
    py: Python<'py>, order: PyReadonlyArray2<'py, i64>, direction: &str, labels: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let side = Side::parse(direction)?;
    let (genes, n, at) = rank_order(&order)?;
    let lab = labels.as_slice()?;
    if lab.len() != n {
        return Err(PyValueError::new_err("one label per sample"));
    }
    let n1 = cases_in(lab, n)?;
    if n1 < 2 || n - n1 < 2 {
        return Err(PyValueError::new_err("the rank statistic needs two samples in each group"));
    }
    let scores: Vec<f64> = py.detach(|| {
        (0..genes).into_par_iter().map(|g| directed(side, brunner_munzel(&at[g * n..(g + 1) * n], lab, n1))).collect()
    });
    Ok(scores.into_pyarray(py))
}

/// The rank statistic's shuffles, counted as `tally` counts the walk's.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn rank_tally_study<'py>(
    py: Python<'py>, order: PyReadonlyArray2<'py, i64>, direction: &str, labels: PyReadonlyArray2<'py, u8>,
    observed: PyReadonlyArray1<'py, f64>, exceed: PyReadonlyArray1<'py, u64>, tried: PyReadonlyArray1<'py, u64>,
    stop: u64, maximum: u64,
) -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u64>>)> {
    let side = Side::parse(direction)?;
    let (genes, n, at) = rank_order(&order)?;
    let (lab, observed) = (labels.as_slice()?, observed.as_slice()?);
    let (hits, draws) = (exceed.as_slice()?, tried.as_slice()?);
    if !labels.is_c_contiguous() || labels.as_array().ncols() != n || observed.len() != genes
        || hits.len() != genes || draws.len() != genes
    {
        return Err(PyValueError::new_err("invalid shuffles or records"));
    }
    let n1 = cases_in(lab, n)?;
    if n1 < 2 || n - n1 < 2 {
        return Err(PyValueError::new_err("the rank statistic needs two samples in each group"));
    }
    let mut records: Vec<(u64, u64)> = hits.iter().copied().zip(draws.iter().copied()).collect();
    py.detach(|| {
        records.par_iter_mut().enumerate().for_each(|(g, record)| {
            if record.0 >= stop || record.1 >= maximum {
                return;
            }
            let row = &at[g * n..(g + 1) * n];
            let o = observed[g];
            let bar = if o.is_finite() { o - TIE_RTOL * o.abs() } else { o };   // this close ties with it
            for labelling in lab.chunks_exact(n) {
                record.1 += 1;
                if directed(side, brunner_munzel(row, labelling, n1)) >= bar {
                    record.0 += 1;
                }
                if record.0 >= stop || record.1 >= maximum {
                    break;
                }
            }
        });
    });
    let (e, t): (Vec<u64>, Vec<u64>) = records.into_iter().unzip();
    Ok((e.into_pyarray(py), t.into_pyarray(py)))
}
