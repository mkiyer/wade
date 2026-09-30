//! The subset walk, its registration, and the shuffles of its permutation test.
//!
//! For sorted cases and controls, the mean shift predicts case quantile u at
//! fold * (Q_control(u) + 1) - 1 on the linear scale, where fold = (case mean + 1) / (control
//! mean + 1), and at Q_control(u) + (case mean - control mean) on the log2 scale. The running
//! departure R(k), the integral of Q_case minus that prediction from 0 to k/n1, is a prefix sum
//! at each boundary between the k-th and (k+1)-th case, so one walk costs O(n).
//!
//! On the linear scale, with z = case + 1 and w = control + 1, R(k) = mean(z) (L_z(k/n1) -
//! L_w(k/n1)), L being a group's Lorenz curve of count + 1. The walk's depth divided by the
//! case mean plus one is therefore the largest amount by which the cases' Lorenz curve falls
//! below the controls': how much more concentrated expression is among the cases. On the log2
//! scale R is the difference of the two absolute Lorenz curves, A(t) = the integral from 0 to t
//! of (Q - mean), and the depth is that gap itself. Either gap is unchanged by a fold on
//! count + 1 applied to every case (a shift on log2), so a shuffle walks registered values as
//! they are.
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

mod study;

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
    fn walk(&self, case: &[f64], control: &[f64], linear: bool, running: &mut [f64]) -> (f64, f64, f64) {
        let (by1, by0) = (1.0 / self.n1 as f64, 1.0 / self.n0 as f64);
        let ((sum1, size1), (sum0, size0)) = (totals(case), totals(control));
        let (mean1, size1, mean0, size0) = (sum1 * by1, size1 * by1, sum0 * by0, size0 * by0);
        let (fold, difference) = ((mean1 + 1.0) / (mean0 + 1.0), mean1 - mean0);
        let (mut cases, mut controls, mut j, mut low) = (0.0, 0.0, 0, f64::INFINITY);
        for k in 1..self.n1 {
            cases += case[k - 1];
            while j < self.whole[k] {
                controls += control[j];
                j += 1;
            }
            let t = k as f64 * by1;
            let below = (controls + self.part[k] * control[j]) * by0; // Q_control integrated to t
            let due = if linear { fold * (below + t) - t } else { below + t * difference };
            let r = if case[k] - case[k - 1] > TIE_RTOL * case[k].abs() { cases * by1 - due } else { f64::INFINITY };
            running[k - 1] = r;
            low = low.min(r);
        }
        let predicted = if linear { fold.abs() * (size0 + 1.0) + 1.0 } else { size0 + difference.abs() };
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
    let lanes = values.chunks_exact(4);
    let rest = lanes.remainder();
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

/// The walk's gap: its depth over the walked group's mean + 1 on the linear scale (the Lorenz
/// gap), the depth itself on the log2 scale (the absolute Lorenz gap).
fn gap(low: f64, mean: f64, linear: bool) -> f64 {
    if linear { -low / (mean + 1.0) } else { -low }
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

    /// The fold on count + 1 (linear) or the log2 difference that minimizes the area between the
    /// groups' quantile functions of log(count + 1) (on log2, of the values as they are): the
    /// weighted median of their difference over the merged breakpoints, weights in units of
    /// 1/(n1 n0). Where the weights below a value make exactly half, every point up to the next
    /// value is a median; the midpoint is taken, so that swapping the groups inverts the fold.
    fn fit(&mut self, case: &[f64], control: &[f64], linear: bool) -> f64 {
        let (n1, n0) = (case.len(), control.len());
        let log = |v: f64| if linear { v.ln_1p() } else { v };
        self.case_log.clear();
        self.case_log.extend(case.iter().map(|&v| log(v)));
        self.control_log.clear();
        self.control_log.extend(control.iter().map(|&v| log(v)));
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
        if linear { shift.exp() } else { shift }
    }
}

/// A walk's subset: its size, whether it lies above the rest of its group, its log2 fold over
/// what the registration predicts for its quantiles, and the walk's gap.
struct Found {
    size: usize,
    up: bool,
    fold: f64,
    gap: f64,
}

/// Walk `case` against `control` and describe the subset. `shift` carries the other group onto
/// the walked one: a fold on count + 1, or a log2 difference.
fn describe(grid: &Grid, case: &[f64], control: &[f64], linear: bool, shift: f64, running: &mut [f64]) -> Option<Found> {
    let n1 = grid.n1;
    let (low, rounding, mean1) = grid.walk(case, control, linear, running);
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
    let fold = if linear { ((seen + 1.0) / (shift * (other + 1.0))).log2() } else { seen - (other + shift) };
    Some(Found { size, up, fold, gap: gap(low, mean1, linear) })
}

/// Which way a test looks. Up asks whether the cases are higher: a mean shift up, and a subset
/// of cases above the rest of the cases or, when both groups are walked, a subset of controls
/// below the rest of the controls. Down asks the reverse, Both either.
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
    fn score(self, up: bool, cases: bool, gap: f64) -> f64 {
        let higher = up == cases;
        match self {
            Side::Both => gap,
            Side::Up => if higher { gap } else { -gap },
            Side::Down => if higher { -gap } else { gap },
        }
    }
}

/// Each gene's subset from rows of sorted cases and controls: its size (0 for none), its side
/// of its group (1 above, -1 below, 0 none), its group (1 cases, 0 controls), its log2 fold over
/// what the registration predicts for it (NaN for none), the walk's gap (0 for none), and the
/// gene's registration. With `both_groups` the controls are walked against the cases too, and
/// the subset with the higher score for `direction` is kept (the cases' on a tie).
#[pyfunction]
#[allow(clippy::type_complexity)]
fn split<'py>(
    py: Python<'py>, case: PyReadonlyArray2<'py, f64>, control: PyReadonlyArray2<'py, f64>, linear: bool,
    both_groups: bool, direction: &str,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<u8>>,
               Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let side = Side::parse(direction)?;
    let (genes, n1) = case.as_array().dim();
    let n0 = control.as_array().ncols();
    if n1 == 0 || n0 == 0 || control.as_array().nrows() != genes
        || !case.is_c_contiguous() || !control.is_c_contiguous()
    {
        return Err(PyValueError::new_err("cases and controls need one nonempty C-ordered row per gene"));
    }
    let (c, d) = (case.as_slice()?, control.as_slice()?);
    let (grid, swapped) = (Grid::new(n1, n0), Grid::new(n0, n1));
    let found: Vec<(i64, i8, u8, f64, f64, f64)> = py.detach(|| {
        (0..genes).into_par_iter().map_init(
            || (Registration::new(), vec![0.0; n1], vec![0.0; n0]),
            |(registration, running1, running0), g| {
                let (case, control) = (&c[g * n1..(g + 1) * n1], &d[g * n0..(g + 1) * n0]);
                let shift = registration.fit(case, control, linear);
                let inverse = if linear { 1.0 / shift } else { -shift };
                let mut best = describe(&grid, case, control, linear, shift, &mut running1[..n1 - 1])
                    .map(|f| (side.score(f.up, true, f.gap), true, f));
                if both_groups {
                    if let Some(f) = describe(&swapped, control, case, linear, inverse, &mut running0[..n0 - 1]) {
                        let score = side.score(f.up, false, f.gap);
                        if best.as_ref().is_none_or(|(b, _, _)| score > *b) {
                            best = Some((score, false, f));
                        }
                    }
                }
                match best {
                    None => (0, 0, 1, f64::NAN, 0.0, shift),
                    Some((_, cases, f)) => (f.size as i64, if f.up { 1 } else { -1 }, cases as u8, f.fold, f.gap, shift),
                }
            },
        ).collect()
    });
    let mut out = (Vec::with_capacity(genes), Vec::with_capacity(genes), Vec::with_capacity(genes),
                   Vec::with_capacity(genes), Vec::with_capacity(genes), Vec::with_capacity(genes));
    for (size, above, group, fold, gap, shift) in found {
        out.0.push(size);
        out.1.push(above);
        out.2.push(group);
        out.3.push(fold);
        out.4.push(gap);
        out.5.push(shift);
    }
    Ok((out.0.into_pyarray(py), out.1.into_pyarray(py), out.2.into_pyarray(py), out.3.into_pyarray(py),
        out.4.into_pyarray(py), out.5.into_pyarray(py)))
}

/// One gene, ready to be walked under any labelling: `values` are its registered values, sorted,
/// and `at` gives each value's sample, so both groups stay sorted as the values are read in order.
struct Walker<'a> {
    grid: &'a Grid,
    swapped: Option<&'a Grid>,
    values: &'a [f64],
    at: &'a [i64],
    linear: bool,
    side: Side,
    case: Vec<f64>,
    control: Vec<f64>,
    running: Vec<f64>,
}

impl<'a> Walker<'a> {
    fn new(grid: &'a Grid, swapped: Option<&'a Grid>, values: &'a [f64], at: &'a [i64], linear: bool, side: Side) -> Self {
        let (n1, n0) = (grid.n1, grid.n0);
        Walker { grid, swapped, values, at, linear, side,
                 case: vec![0.0; n1 + 1], control: vec![0.0; n0 + 1], running: vec![0.0; n1.max(n0)] }
    }

    /// One group's walk against the other, scored for the test (0 without a subset).
    fn walked(grid: &Grid, walked: &[f64], other: &[f64], linear: bool, side: Side, cases: bool, running: &mut [f64]) -> f64 {
        let n = grid.n1;
        let (low, rounding, mean) = grid.walk(walked, other, linear, &mut running[..n - 1]);
        if -low <= rounding {
            return 0.0;
        }
        let k = cut(&running[..n - 1], low, rounding);
        side.score(n - k <= k, cases, gap(low, mean, linear))
    }

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
        let (case, control) = (&self.case[..n1], &self.control[..n0]);
        let mut score = Self::walked(self.grid, case, control, self.linear, self.side, true, &mut self.running);
        if let Some(swapped) = self.swapped {
            score = score.max(Self::walked(swapped, control, case, self.linear, self.side, false, &mut self.running));
        }
        score
    }
}

/// Validated views of the pooled genes to walk.
struct Pool<'a> {
    genes: usize,
    n: usize,
    values: &'a [f64],
    at: &'a [i64],
}

impl<'a> Pool<'a> {
    fn new(pooled: &'a PyReadonlyArray2<'a, f64>, order: &'a PyReadonlyArray2<'a, i64>) -> PyResult<Self> {
        let (genes, n) = pooled.as_array().dim();
        if !pooled.is_c_contiguous() || !order.is_c_contiguous() || order.as_array().dim() != (genes, n) {
            return Err(PyValueError::new_err("pooled values and order must be C-ordered genes x samples"));
        }
        let (values, at) = (pooled.as_slice()?, order.as_slice()?);
        if n == 0 || at.iter().any(|&i| i < 0 || i as usize >= n) {
            return Err(PyValueError::new_err("invalid walk inputs: shapes or sample order"));
        }
        Ok(Pool { genes, n, values, at })
    }

    fn walker<'g>(&'g self, grid: &'g Grid, swapped: Option<&'g Grid>, g: usize, linear: bool, side: Side) -> Walker<'g> {
        let row = g * self.n..(g + 1) * self.n;
        Walker::new(grid, swapped, &self.values[row.clone()], &self.at[row], linear, side)
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

/// Each gene's score under one labelling, by the walk the shuffles use: the observed statistic.
#[pyfunction]
fn score<'py>(
    py: Python<'py>, pooled: PyReadonlyArray2<'py, f64>, order: PyReadonlyArray2<'py, i64>, linear: bool,
    direction: &str, both_groups: bool, labels: PyReadonlyArray1<'py, u8>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let side = Side::parse(direction)?;
    let pool = Pool::new(&pooled, &order)?;
    let lab = labels.as_slice()?;
    if lab.len() != pool.n {
        return Err(PyValueError::new_err("one label per sample"));
    }
    let n1 = cases_in(lab, pool.n)?;
    let (grid, swapped) = (Grid::new(n1, pool.n - n1), Grid::new(pool.n - n1, n1));
    let scores: Vec<f64> = py.detach(|| {
        (0..pool.genes).into_par_iter()
            .map(|g| pool.walker(&grid, both_groups.then_some(&swapped), g, linear, side).score(lab))
            .collect()
    });
    Ok(scores.into_pyarray(py))
}

/// Walk each shuffle for every gene still running, counting shuffles and those scoring at
/// least the observed score; a gene stops at `stop` exceedances or `maximum` shuffles.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn tally<'py>(
    py: Python<'py>, pooled: PyReadonlyArray2<'py, f64>, order: PyReadonlyArray2<'py, i64>, linear: bool,
    direction: &str, both_groups: bool, labels: PyReadonlyArray2<'py, u8>, observed: PyReadonlyArray1<'py, f64>,
    exceed: PyReadonlyArray1<'py, u64>, tried: PyReadonlyArray1<'py, u64>, stop: u64, maximum: u64,
) -> PyResult<(Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<u64>>)> {
    let side = Side::parse(direction)?;
    let pool = Pool::new(&pooled, &order)?;
    let n = pool.n;
    let (lab, observed) = (labels.as_slice()?, observed.as_slice()?);
    let (hits, draws) = (exceed.as_slice()?, tried.as_slice()?);
    if !labels.is_c_contiguous() || labels.as_array().ncols() != n || observed.len() != pool.genes
        || hits.len() != pool.genes || draws.len() != pool.genes
    {
        return Err(PyValueError::new_err("invalid shuffles or records"));
    }
    let n1 = cases_in(lab, n)?;
    let (grid, swapped) = (Grid::new(n1, n - n1), Grid::new(n - n1, n1));
    let mut records: Vec<(u64, u64)> = hits.iter().copied().zip(draws.iter().copied()).collect();
    py.detach(|| {
        records.par_iter_mut().enumerate().for_each(|(g, record)| {
            if record.0 >= stop || record.1 >= maximum {
                return;
            }
            let mut walker = pool.walker(&grid, both_groups.then_some(&swapped), g, linear, side);
            let bar = observed[g] - TIE_RTOL * observed[g].abs();       // this close ties with it
            for row in lab.chunks_exact(n) {
                record.1 += 1;
                if walker.score(row) >= bar {
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

#[pymodule]
fn _kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(split, m)?)?;
    m.add_function(wrap_pyfunction!(score, m)?)?;
    m.add_function(wrap_pyfunction!(tally, m)?)?;
    m.add_function(wrap_pyfunction!(study::split_study, m)?)?;
    m.add_function(wrap_pyfunction!(study::tally_study, m)?)?;
    m.add_function(wrap_pyfunction!(study::rank_score_study, m)?)?;
    m.add_function(wrap_pyfunction!(study::rank_tally_study, m)?)?;
    Ok(())
}
