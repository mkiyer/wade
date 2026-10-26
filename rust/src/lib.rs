//! The subset walk, its registration, and the shuffles of its permutation test.
//!
//! For sorted cases and controls, the mean shift predicts case quantile u at
//! fold * (Q_control(u) + 1) - 1, where fold = (case mean + 1) / (control mean + 1). The running
//! departure R(k), the integral of Q_case minus that prediction from 0 to k/n1, is a prefix sum
//! at each boundary between the k-th and (k+1)-th case, so one walk costs O(n).
//!
//! With z = case + 1 and w = control + 1, R(k) = mean(z) (L_z(k/n1) - L_w(k/n1)), L being a
//! group's Lorenz curve of count + 1. The walk's depth divided by the case mean plus one is
//! therefore the largest amount by which the cases' Lorenz curve falls below the controls': how
//! much more concentrated expression is among the cases. The gap is unchanged by a fold on
//! count + 1 applied to every case, so a shuffle walks registered values as they are.
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Values within 100 ulps are equal: no boundary falls between them, and a shuffle this close ties.
const TIE_RTOL: f64 = 100.0 * f64::EPSILON;

/// Where each case boundary k/n1 falls among the controls: whole controls below it, and the
/// fraction of the next one.
struct Grid {
    n1: usize,
    n0: usize,
    whole: Vec<usize>,
    part: Vec<f64>,
}

impl Grid {
    fn new(n1: usize, n0: usize) -> Self {
        let whole = (0..n1).map(|k| k * n0 / n1).collect();
        let part = (0..n1).map(|k| (k * n0 % n1) as f64 / n1 as f64).collect();
        Grid { n1, n0, whole, part }
    }

    /// Fill `running` with R(k) for k = 1..n1-1, infinite where equal cases forbid a boundary,
    /// and return its lowest value, the rounding scale of those sums, and the case mean.
    fn walk(&self, case: &[f64], control: &[f64], running: &mut [f64]) -> (f64, f64, f64) {
        let (by1, by0) = (1.0 / self.n1 as f64, 1.0 / self.n0 as f64);
        let ((sum1, size1), (sum0, size0)) = (totals(case), totals(control));
        let (mean1, size1, mean0, size0) = (sum1 * by1, size1 * by1, sum0 * by0, size0 * by0);
        let fold = (mean1 + 1.0) / (mean0 + 1.0);
        let (mut cases, mut controls, mut j, mut low) = (0.0, 0.0, 0, f64::INFINITY);
        for k in 1..self.n1 {
            cases += case[k - 1];
            while j < self.whole[k] {
                controls += control[j];
                j += 1;
            }
            let t = k as f64 * by1;
            let below = (controls + self.part[k] * control[j]) * by0; // Q_control integrated to t
            let due = fold * (below + t) - t;
            let r = if case[k] - case[k - 1] > TIE_RTOL * case[k].abs() { cases * by1 - due } else { f64::INFINITY };
            running[k - 1] = r;
            low = low.min(r);
        }
        let predicted = fold.abs() * (size0 + 1.0) + 1.0;
        (low, TIE_RTOL * (self.n1 + self.n0) as f64 * (size1 + predicted), mean1)
    }

    /// The controls' quantile function integrated from 0 to k/n1, for k = 0..=n1.
    fn below(&self, control: &[f64], k: usize) -> f64 {
        if k == self.n1 {
            return control.iter().sum::<f64>() / self.n0 as f64;
        }
        let j = self.whole[k];
        (control[..j].iter().sum::<f64>() + self.part[k] * control[j]) / self.n0 as f64
    }
}

/// The sum of the values and of their magnitudes, in four independent lanes so the additions
/// need not wait on one another.
fn totals(values: &[f64]) -> (f64, f64) {
    let (mut sum, mut size) = ([0.0; 4], [0.0; 4]);
    let (lanes, rest) = values.as_chunks::<4>();
    for lane in lanes {
        for i in 0..4 {
            sum[i] += lane[i];
            size[i] += lane[i].abs();
        }
    }
    let (mut sum, mut size) = ((sum[0] + sum[1]) + (sum[2] + sum[3]), (size[0] + size[1]) + (size[2] + size[3]));
    for &x in rest {
        sum += x;
        size += x.abs();
    }
    (sum, size)
}

/// The walk's cut: the highest boundary among its minima (those within rounding of the lowest).
fn cut(running: &[f64], low: f64, rounding: f64) -> usize {
    running.iter().rposition(|&r| r <= low + rounding).map_or(0, |k| k + 1)
}

/// The walk's gap: its depth over the case mean + 1, the Lorenz gap.
fn gap(low: f64, mean: f64) -> f64 {
    -low / (mean + 1.0)
}

/// Scratch for registering one gene.
struct Registration {
    case_log: Vec<f64>,
    control_log: Vec<f64>,
    segments: Vec<(f64, u64)>,
}

impl Registration {
    fn new() -> Self {
        Registration { case_log: Vec::new(), control_log: Vec::new(), segments: Vec::new() }
    }

    /// The fold on count + 1 that minimizes the area between the groups' quantile functions of
    /// log(count + 1): the weighted median of their difference over the merged breakpoints,
    /// weights in units of 1/(n1 n0). Where the weights below a value make exactly half, every
    /// point up to the next value is a median; the midpoint is taken, so that swapping the groups
    /// inverts the fold.
    fn fit(&mut self, case: &[f64], control: &[f64]) -> f64 {
        let (n1, n0) = (case.len(), control.len());
        self.case_log.clear();
        self.case_log.extend(case.iter().map(|&v| v.ln_1p()));
        self.control_log.clear();
        self.control_log.extend(control.iter().map(|&v| v.ln_1p()));
        self.segments.clear();
        let (mut i, mut j, mut at) = (0usize, 0usize, 0u64);
        while i < n1 && j < n0 {
            let (ci, cj) = ((i as u64 + 1) * n0 as u64, (j as u64 + 1) * n1 as u64);
            let next = ci.min(cj);
            self.segments.push((self.case_log[i] - self.control_log[j], next - at));
            at = next;
            if ci == next {
                i += 1;
            }
            if cj == next {
                j += 1;
            }
        }
        self.segments.sort_unstable_by(|p, q| p.0.total_cmp(&q.0));
        let total = n1 as u64 * n0 as u64;
        let (mut below, mut shift) = (0u64, 0.0);
        for (k, &(value, weight)) in self.segments.iter().enumerate() {
            below += weight;
            if 2 * below >= total {
                shift = match self.segments.get(k + 1) {
                    Some(&(next, _)) if 2 * below == total => 0.5 * (value + next),
                    _ => value,
                };
                break;
            }
        }
        shift.exp()
    }
}

/// A walk's subset: its size, whether it lies above the rest of the cases, its log2 fold over
/// what the registration predicts for its quantiles, and the walk's gap.
struct Found {
    size: usize,
    up: bool,
    fold: f64,
    gap: f64,
}

/// Walk the cases against the controls and describe the subset. `shift` carries the controls onto
/// the cases: a fold on count + 1.
fn describe(grid: &Grid, case: &[f64], control: &[f64], shift: f64, running: &mut [f64]) -> Option<Found> {
    let n1 = grid.n1;
    let (low, rounding, mean1) = grid.walk(case, control, running);
    if -low <= rounding {
        return None;
    }
    // The smaller side of the cut departs further per sample: it is the subset.
    let k = cut(running, low, rounding);
    let up = n1 - k <= k;
    let (from, to) = if up { (k, n1) } else { (0, k) };
    let size = to - from;
    let seen = case[from..to].iter().sum::<f64>() / size as f64;
    let other = (grid.below(control, to) - grid.below(control, from)) * n1 as f64 / size as f64;
    let fold = ((seen + 1.0) / (shift * (other + 1.0))).log2();
    Some(Found { size, up, fold, gap: gap(low, mean1) })
}

/// Which way a test looks. Up asks whether the cases are higher: a subset of cases above the rest
/// of the cases. Down asks the reverse, Both either.
#[derive(Clone, Copy)]
enum Side {
    Both,
    Up,
    Down,
}

impl Side {
    fn parse(direction: &str) -> PyResult<Self> {
        match direction {
            "both" => Ok(Side::Both),
            "up" => Ok(Side::Up),
            "down" => Ok(Side::Down),
            _ => Err(PyValueError::new_err("direction must be 'both', 'up' or 'down'")),
        }
    }

    /// A walk's score for this test: its gap, negative when the subset lies the other way.
    fn score(self, up: bool, gap: f64) -> f64 {
        match self {
            Side::Both => gap,
            Side::Up => if up { gap } else { -gap },
            Side::Down => if up { -gap } else { gap },
        }
    }
}

/// Each gene's subset from rows of sorted cases and controls: its size (0 for none), its side
/// of the cases (1 above, -1 below, 0 none), its log2 fold over what the registration predicts
/// for it (NaN for none), the walk's gap (0 for none), and the gene's registration.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn split<'py>(
    py: Python<'py>, case: PyReadonlyArray2<'py, f64>, control: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<f64>>,
               Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (genes, n1) = case.as_array().dim();
    let n0 = control.as_array().ncols();
    if n1 == 0 || n0 == 0 || control.as_array().nrows() != genes
        || !case.is_c_contiguous() || !control.is_c_contiguous()
    {
        return Err(PyValueError::new_err("cases and controls need one nonempty C-ordered row per gene"));
    }
    let (c, d) = (case.as_slice()?, control.as_slice()?);
    let grid = Grid::new(n1, n0);
    let found: Vec<(i64, i8, f64, f64, f64)> = py.detach(|| {
        (0..genes).into_par_iter().map_init(
            || (Registration::new(), vec![0.0; n1]),
            |(registration, running), g| {
                let (case, control) = (&c[g * n1..(g + 1) * n1], &d[g * n0..(g + 1) * n0]);
                let shift = registration.fit(case, control);
                match describe(&grid, case, control, shift, &mut running[..n1 - 1]) {
                    None => (0, 0, f64::NAN, 0.0, shift),
                    Some(f) => (f.size as i64, if f.up { 1 } else { -1 }, f.fold, f.gap, shift),
                }
            },
        ).collect()
    });
    let mut out = (Vec::with_capacity(genes), Vec::with_capacity(genes), Vec::with_capacity(genes),
                   Vec::with_capacity(genes), Vec::with_capacity(genes));
    for (size, above, fold, gap, shift) in found {
        out.0.push(size);
        out.1.push(above);
        out.2.push(fold);
        out.3.push(gap);
        out.4.push(shift);
    }
    Ok((out.0.into_pyarray(py), out.1.into_pyarray(py), out.2.into_pyarray(py), out.3.into_pyarray(py),
        out.4.into_pyarray(py)))
}

/// Each gene's walk as a gap at every case boundary: how far the cases' Lorenz curve of count + 1
/// falls below the controls' at k/n1, for k = 1..n1-1, or -inf where equal cases forbid a
/// boundary. Its largest value is the gene's gap, at the gene's cut; at any one cut, the gaps of
/// different genes compare subsets of the same size and side. Rows of sorted cases and controls;
/// the curves are returned row after row (genes x (n1 - 1)).
#[pyfunction]
fn gaps<'py>(
    py: Python<'py>, case: PyReadonlyArray2<'py, f64>, control: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (genes, n1) = case.as_array().dim();
    let n0 = control.as_array().ncols();
    if n1 < 2 || n0 == 0 || control.as_array().nrows() != genes || !case.is_c_contiguous() || !control.is_c_contiguous() {
        return Err(PyValueError::new_err("cases (at least two) and controls need one C-ordered row per gene"));
    }
    let (c, d) = (case.as_slice()?, control.as_slice()?);
    let grid = Grid::new(n1, n0);
    let mut curves = vec![0.0; genes * (n1 - 1)];
    py.detach(|| {
        curves.par_chunks_mut(n1 - 1).enumerate().for_each(|(g, row)| {
            let (_, _, mean) = grid.walk(&c[g * n1..(g + 1) * n1], &d[g * n0..(g + 1) * n0], row);
            for r in row.iter_mut() {
                *r = if r.is_finite() { -*r / (mean + 1.0) } else { f64::NEG_INFINITY };
            }
        });
    });
    Ok(curves.into_pyarray(py))
}

/// One gene, ready to be walked under any labelling: `values` are its registered values, sorted,
/// and `at` gives each value's sample, so both groups stay sorted as the values are read in order.
struct Walker<'a> {
    grid: &'a Grid,
    values: &'a [f64],
    at: &'a [i64],
    side: Side,
    case: Vec<f64>,
    control: Vec<f64>,
    running: Vec<f64>,
}

impl<'a> Walker<'a> {
    fn new(grid: &'a Grid, values: &'a [f64], at: &'a [i64], side: Side) -> Self {
        let (n1, n0) = (grid.n1, grid.n0);
        Walker { grid, values, at, side, case: vec![0.0; n1 + 1], control: vec![0.0; n0 + 1], running: vec![0.0; n1] }
    }

    /// The walk of this labelling's cases against its controls, scored for the test (0 without a subset).
    fn score(&mut self, labels: &[u8]) -> f64 {
        let (n1, n0) = (self.grid.n1, self.grid.n0);
        let (mut a, mut b) = (0, 0);
        for (&v, &i) in self.values.iter().zip(self.at) {
            // Write both, keep one: no branch on a random label.
            let is_case = labels[i as usize] as usize;
            self.case[a] = v;
            self.control[b] = v;
            a += is_case;
            b += 1 - is_case;
        }
        let running = &mut self.running[..n1 - 1];
        let (low, rounding, mean) = self.grid.walk(&self.case[..n1], &self.control[..n0], running);
        if -low <= rounding {
            return 0.0;
        }
        let k = cut(running, low, rounding);
        self.side.score(n1 - k <= k, gap(low, mean))
    }
}

/// Validated views of the pooled genes to walk: each gene's draws are consecutive rows.
struct Pool<'a> {
    genes: usize,
    draws: usize,
    n: usize,
    values: &'a [f64],
    at: &'a [i64],
}

impl<'a> Pool<'a> {
    fn new(pooled: &'a PyReadonlyArray2<'a, f64>, order: &'a PyReadonlyArray2<'a, i64>, draws: usize) -> PyResult<Self> {
        let (rows, n) = pooled.as_array().dim();
        if !pooled.is_c_contiguous() || !order.is_c_contiguous() || order.as_array().dim() != (rows, n) {
            return Err(PyValueError::new_err("pooled values and order must be C-ordered rows x samples"));
        }
        if draws == 0 || !rows.is_multiple_of(draws) {
            return Err(PyValueError::new_err("the rows must hold the same number of draws for every gene"));
        }
        let (values, at) = (pooled.as_slice()?, order.as_slice()?);
        if n == 0 || at.iter().any(|&i| i < 0 || i as usize >= n) {
            return Err(PyValueError::new_err("invalid walk inputs: shapes or sample order"));
        }
        Ok(Pool { genes: rows / draws, draws, n, values, at })
    }

    /// Gene g's walkers, one for each of its draws.
    fn walkers<'g>(&'g self, grid: &'g Grid, g: usize, side: Side) -> Draws<'g> {
        Draws((0..self.draws).map(|d| {
            let row = (g * self.draws + d) * self.n..(g * self.draws + d + 1) * self.n;
            Walker::new(grid, &self.values[row.clone()], &self.at[row], side)
        }).collect())
    }
}

/// One gene's walkers, one per thinning draw. Its score under a labelling is the mean of theirs, so
/// the gene's one stream of shuffles is compared with one observed mean.
struct Draws<'a>(Vec<Walker<'a>>);

impl Draws<'_> {
    fn score(&mut self, labels: &[u8]) -> f64 {
        let total: f64 = self.0.iter_mut().map(|walker| walker.score(labels)).sum();
        total / self.0.len() as f64
    }
}

/// The number of cases in a labelling, or an error unless every row of `labels` is a 0/1
/// labelling of n samples with that many cases and at least one of each.
fn cases_in(labels: &[u8], n: usize) -> PyResult<usize> {
    let n1 = labels.iter().take(n).map(|&v| v as usize).sum::<usize>();
    if labels.is_empty() || !labels.len().is_multiple_of(n) || n1 == 0 || n1 == n
        || labels.chunks(n).any(|row| row.iter().any(|&v| v > 1) || row.iter().map(|&v| v as usize).sum::<usize>() != n1)
    {
        return Err(PyValueError::new_err("invalid labellings"));
    }
    Ok(n1)
}

/// Each gene's score under one labelling, by the walk the shuffles use: the observed statistic. With
/// several draws per gene (consecutive rows), the mean of the draws' scores.
#[pyfunction]
#[pyo3(signature = (pooled, order, direction, labels, draws=1))]
fn score<'py>(
    py: Python<'py>, pooled: PyReadonlyArray2<'py, f64>, order: PyReadonlyArray2<'py, i64>, direction: &str,
    labels: PyReadonlyArray1<'py, u8>, draws: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let side = Side::parse(direction)?;
    let pool = Pool::new(&pooled, &order, draws)?;
    let lab = labels.as_slice()?;
    if lab.len() != pool.n {
        return Err(PyValueError::new_err("one label per sample"));
    }
    let n1 = cases_in(lab, pool.n)?;
    let grid = Grid::new(n1, pool.n - n1);
    let scores: Vec<f64> = py.detach(|| {
        (0..pool.genes).into_par_iter().map(|g| pool.walkers(&grid, g, side).score(lab)).collect()
    });
    Ok(scores.into_pyarray(py))
}

/// The Besag–Clifford p-value under the most conservative future: an upper bound, on every path,
/// on the p-value a gene with `exceed` exceedances in `tried` shuffles will finish with, and equal to
/// it once the gene stops (at `stop` exceedances or `maximum` shuffles). It never rises as shuffles
/// are added. Python's `pvalues.bound` computes the same value in the same arithmetic.
fn bound(exceed: u64, tried: u64, stop: u64, maximum: u64) -> f64 {
    if exceed >= stop {
        stop as f64 / tried as f64
    } else if tried + stop - exceed <= maximum {
        stop as f64 / (tried + stop - exceed) as f64
    } else {
        (exceed + maximum - tried + 1) as f64 / (maximum + 1) as f64
    }
}

/// Whether a gene needs no more shuffles: it has finished its Besag–Clifford run, or its bound has
/// reached `threshold` (a BH threshold; -inf runs every gene to the end).
fn settled(exceed: u64, tried: u64, stop: u64, maximum: u64, threshold: f64) -> bool {
    exceed >= stop || tried >= maximum || bound(exceed, tried, stop, maximum) <= threshold
}

/// Walk the shuffles in `labels`, which hold positions `first`, `first` + 1, ... of the run's one
/// shuffle stream, for every gene not yet settled. Each gene resumes at its own position (the
/// shuffles it has tried), counts shuffles and those scoring at least the observed score, and
/// stops once settled. A gene whose position lies beyond this block waits for its own. With several
/// draws per gene (consecutive rows), each shuffle's score is the mean of the draws'.
#[pyfunction]
#[pyo3(signature = (pooled, order, direction, labels, observed, exceed, tried, stop, maximum, first, threshold, draws=1))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn tally<'py>(
    py: Python<'py>, pooled: PyReadonlyArray2<'py, f64>, order: PyReadonlyArray2<'py, i64>, direction: &str,
    labels: PyReadonlyArray2<'py, u8>, observed: PyReadonlyArray1<'py, f64>, exceed: PyReadonlyArray1<'py, u64>,
    tried: PyReadonlyArray1<'py, u64>, stop: u64, maximum: u64, first: u64, threshold: f64, draws: usize,
) -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u64>>)> {
    let side = Side::parse(direction)?;
    let pool = Pool::new(&pooled, &order, draws)?;
    let n = pool.n;
    let (lab, observed) = (labels.as_slice()?, observed.as_slice()?);
    let (hits, draws) = (exceed.as_slice()?, tried.as_slice()?);
    if !labels.is_c_contiguous() || labels.as_array().ncols() != n || observed.len() != pool.genes
        || hits.len() != pool.genes || draws.len() != pool.genes || stop == 0 || stop > maximum
        || threshold.is_nan()
    {
        return Err(PyValueError::new_err("invalid shuffles or records"));
    }
    if hits.iter().zip(draws).any(|(&k, &t)| k > t || t > maximum || (!settled(k, t, stop, maximum, threshold) && t < first)) {
        return Err(PyValueError::new_err("a running gene's next shuffle precedes this block"));
    }
    let n1 = cases_in(lab, n)?;
    let grid = Grid::new(n1, n - n1);
    let end = first + (lab.len() / n) as u64;
    let mut records: Vec<(u64, u64)> = hits.iter().copied().zip(draws.iter().copied()).collect();
    py.detach(|| {
        records.par_iter_mut().enumerate().for_each(|(g, record)| {
            if settled(record.0, record.1, stop, maximum, threshold) || record.1 >= end {
                return;
            }
            let mut walkers = pool.walkers(&grid, g, side);
            let bar = observed[g] - TIE_RTOL * observed[g].abs();       // this close ties with it
            for row in lab.chunks_exact(n).skip((record.1 - first) as usize) {
                record.1 += 1;
                if walkers.score(row) >= bar {
                    record.0 += 1;
                }
                if settled(record.0, record.1, stop, maximum, threshold) {
                    break;
                }
            }
        });
    });
    let (e, t): (Vec<u64>, Vec<u64>) = records.into_iter().unzip();
    Ok((e.into_pyarray(py), t.into_pyarray(py)))
}

#[pymodule]
fn _kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(split, m)?)?;
    m.add_function(wrap_pyfunction!(gaps, m)?)?;
    m.add_function(wrap_pyfunction!(score, m)?)?;
    m.add_function(wrap_pyfunction!(tally, m)?)?;
    Ok(())
}
