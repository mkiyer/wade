//! Benchmark-only loop. The Python wrapper validates all shapes and values.
//! Six ordered sums: linear up/down/unsigned, then log up/down/unsigned.
//! No RNG, approximation, parallel reduction or change to public WADE.
use std::slice;

#[no_mangle]
pub unsafe extern "C" fn update_points(
    genes: usize, samples: usize, permutations: usize, cases: usize, stop: u64,
    order: *const usize, gaps: *const f64, labels: *const u8,
    thresholds: *const f64, exceed: *mut u64, tried: *mut u64,
) {
    let order = slice::from_raw_parts(order, genes * samples);
    let gaps = slice::from_raw_parts(gaps, genes * 2 * (samples - 1));
    let labels = slice::from_raw_parts(labels, permutations * samples);
    let thresholds = slice::from_raw_parts(thresholds, genes * 6);
    let exceed = slice::from_raw_parts_mut(exceed, genes * 6);
    let tried = slice::from_raw_parts_mut(tried, genes * 6);
    for gene in 0..genes {
        let k = &mut exceed[gene*6..(gene+1)*6];
        let t = &mut tried[gene*6..(gene+1)*6];
        let threshold = &thresholds[gene*6..(gene+1)*6];
        if k.iter().all(|&v| v == stop) { continue; }
        let indices = &order[gene*samples..(gene+1)*samples];
        let linear = &gaps[gene*2*(samples-1)..(gene*2+1)*(samples-1)];
        let log = &gaps[(gene*2+1)*(samples-1)..(gene*2+2)*(samples-1)];
        for row in labels.chunks(samples) {
            let mut prefix = 0i64;
            let mut area = [0.0; 6];
            for j in 0..samples-1 {
                prefix += row[indices[j]] as i64;
                let mass = (cases*(j+1)) as i64 - (samples as i64)*prefix;
                let up = mass.max(0) as f64;
                let down = (-mass).max(0) as f64;
                let absolute = mass.abs() as f64;
                area[0] += up*linear[j];
                area[1] += down*linear[j];
                area[2] += absolute*linear[j];
                area[3] += up*log[j];
                area[4] += down*log[j];
                area[5] += absolute*log[j];
            }
            for d in 0..6 {
                if k[d] < stop {
                    t[d] += 1;
                    if area[d] >= threshold[d] { k[d] += 1; }
                }
            }
            if k.iter().all(|&v| v == stop) { break; }
        }
    }
}
