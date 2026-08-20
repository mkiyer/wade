"""The scale question (docs/method.md section 10), pinned.

* Stage 1's level is exact under any monotone transform — permutation does not
  care what statistic it is given — and stage 1 on the linear scale is far
  more powerful than on the log scale for a rare strong subset at low counts.
  This is why stage 1 is linear.
* Stage 2 is silent on a genuine global fold change in NB counts at every
  expression level — low counts included — now that its null is built by
  binomial thinning (10.3), and the log-ratio curve carries a one-count
  pseudocount (10.4) so a single zero cannot halve ``affected_fraction``.
  Before these landed the false-subset rate was 0.95 at 2 counts and a global
  2x read 0.16; those numbers are in 10.2 and the tests below were strict
  xfails.

Counts are NB, and the matrices are run with **unit library sizes** on
purpose: a matrix whose every gene is shifted 2x would otherwise normalize to
no shift at all. With unit libraries one count is ``norm_factor`` TPM-like
units, which is exactly what the pseudocount becomes.
"""

from __future__ import annotations

import numpy as np
import pytest

import wade
from wade.permutation import draw_perms, null_statistics
from wade.pvalues import perm_pvalues

pytestmark = pytest.mark.validation

N1 = N0 = 200
COND = np.r_[np.ones(N1, int), np.zeros(N0, int)]
PHI = 0.1
B = 200
REPS = 80


def _nb(rng, mu, size):
    return rng.poisson(rng.gamma(1 / PHI, PHI * mu, size=size)).astype(float)


def _counts(rng, mu, alt, fold=2.0, frac=0.05):
    case = _nb(rng, mu, (REPS, N1))
    ctrl = _nb(rng, mu, (REPS, N0))
    if alt == "global":
        case = _nb(rng, mu * fold, (REPS, N1))
    elif alt == "subset":
        k = round(frac * N1)
        for i in range(REPS):
            case[i, rng.choice(N1, k, replace=False)] = _nb(rng, mu * fold, k)
    return np.c_[case, ctrl] + rng.uniform(0, 0.01, (REPS, N1 + N0))   # the jitter, by hand


def _stage1_rate(x, perms, alpha=0.05):
    obs = wade.wade_stats(x, COND).mean_shift
    p, _, _ = perm_pvalues(obs, null_statistics(x, perms), alternative="two-sided")
    return float((p < alpha).mean())


def _raw_counts(rng, mu, alt, fold=2.0, frac=0.05):
    """Integer NB counts, no jitter: the raw-count entry point adds its own."""
    case = _nb(rng, mu, (REPS, N1))
    ctrl = _nb(rng, mu, (REPS, N0))
    if alt == "global":
        case = _nb(rng, mu * fold, (REPS, N1))
    elif alt == "subset":
        k = round(frac * N1)
        for i in range(REPS):
            case[i, rng.choice(N1, k, replace=False)] = _nb(rng, mu * fold, k)
    return np.c_[case, ctrl]


def _stage2_rate(counts, alpha=0.05, **kw):
    """Through the raw-count entry point with unit library sizes."""
    res = wade.wade(counts, np.ones(counts.shape[0]), COND, lib_sizes=np.ones(N1 + N0),
                    nperms=B, seed=2, **kw)
    return float((res.p_subset < alpha).mean()), res


@pytest.fixture(scope="module")
def perms():
    return draw_perms(COND, B, seed=1)


# ---------------------------------------------------------------------------
# Stage 1


@pytest.mark.parametrize("g", ["linear", "log2", "sqrt"])
def test_stage1_level_is_exact_under_any_transform(perms, g):
    """Permutation is exact for any statistic, so the scale cannot move the
    false-positive rate — only the power. Measured 0.03-0.07 everywhere."""
    rng = np.random.default_rng(31)
    x = _counts(rng, 2.0, "null")
    gx = {"linear": x, "log2": np.log2(x), "sqrt": np.sqrt(x)}[g]
    rate = _stage1_rate(gx, perms)
    assert rate <= 0.12, f"{g}: null rejection {rate:.2f}"     # 3 SE above 0.05 at 80 genes


def test_stage1_linear_is_far_more_powerful_than_log_for_a_rare_subset_at_low_counts(perms):
    """The reason stage 1 is linear (method.md 10.1). Measured 0.73 vs 0.07 at
    200 v 200, B = 500; asserted with margin at B = 200."""
    rng = np.random.default_rng(32)
    x = _counts(rng, 2.0, "subset", fold=8.0, frac=0.05)
    linear = _stage1_rate(x, perms)
    log = _stage1_rate(np.log2(x), perms)
    assert linear >= 0.50, f"linear power {linear:.2f}"
    assert log <= 0.30, f"log power {log:.2f}"
    assert linear - log >= 0.30


# ---------------------------------------------------------------------------
# Stage 2: the regime the original validation covered, and the one it did not


def test_stage2_is_silent_on_a_global_nb_fold_change_at_high_counts():
    rng = np.random.default_rng(33)
    rate, res = _stage2_rate(_raw_counts(rng, 500.0, "global"))
    assert rate <= 0.12, f"false-subset rate at 500 counts {rate:.2f}"
    assert np.median(res.affected_fraction) >= 0.9
    assert res.subset.correction == "thinning"


@pytest.mark.parametrize("mu", [2.0, 20.0])
def test_stage2_is_silent_on_a_global_nb_fold_change_at_low_counts(mu):
    """method.md 10.2-10.3. Before thinning: 0.95 at 2 counts (the zero floor)
    and 0.17 at 20 counts (NB's Poisson component). Measured after: 0.05 at both."""
    rng = np.random.default_rng(34)
    rate, res = _stage2_rate(_raw_counts(rng, mu, "global"))
    assert rate <= 0.12, f"false-subset rate at {mu:g} counts {rate:.2f}"
    # The fitted fold change is what the null was built under, and it is unbiased.
    assert abs(np.median(res.fitted_fold_change) - 2.0) < 0.15


def test_stage2_division_correction_is_what_it_was_and_still_fires_at_low_counts():
    """The continuous-data correction is kept for wade_from_matrix and thin=False;
    on counts at 2 it fires on genuine global shifts (10.2). Pinned so the
    difference between the two corrections stays visible."""
    rng = np.random.default_rng(34)
    counts = _raw_counts(rng, 2.0, "global")
    rate_div, res = _stage2_rate(counts, thin=False)
    assert res.subset.correction == "division"
    assert rate_div >= 0.5, f"expected the division correction to fire at 2 counts; got {rate_div:.2f}"


def test_stage2_keeps_its_power_on_subsets_at_low_counts():
    """10.3/10.4: 5% at 8x at 2 counts, measured 0.87 with thinning and the
    pseudocount (0.64 before)."""
    rng = np.random.default_rng(37)
    rate, _ = _stage2_rate(_raw_counts(rng, 2.0, "subset", fold=8.0, frac=0.05))
    assert rate >= 0.6, f"power on a 5% subset at 8x, 2 counts: {rate:.2f}"


@pytest.mark.parametrize("mu, floor", [(2.0, 0.5), (5.0, 0.7), (20.0, 0.85)])
def test_affected_fraction_anchors_a_global_nb_fold_change_with_the_pseudocount(mu, floor):
    """10.4: with log(x + one count) a global 2x reads 0.70 at 2 counts, 0.86
    at 5, 0.96 at 20 (0.16, 0.07, 0.95 without). Below ~5 counts the log-scale
    characterization is noise-dominated and the anchor is approximate."""
    rng = np.random.default_rng(35)
    _, res = _stage2_rate(_raw_counts(rng, mu, "global"))
    med = float(np.median(res.affected_fraction))
    assert med >= floor, f"global 2x at {mu:g} counts read {med:.2f}"


def test_affected_fraction_is_not_halved_by_a_single_zero_at_large_n():
    """10.4: three of six seeds at 20,000 v 20,000 carry a control zero; under
    the jitter alone that node is a log-ratio of ~10 and the estimate read
    0.39-0.70 on those seeds. With one count added before the log it does not
    depend on that."""
    from wade.quantiles import probability_grid, type7_quantiles
    from wade.subset import affected_fraction, log_ratio_curve
    worst_with, worst_without = 1.0, 1.0
    for seed in range(6):
        rng = np.random.default_rng(seed)
        m = 20000
        case = _nb(rng, 40, (1, m)) + rng.uniform(0, 0.01, (1, m))
        ctrl = _nb(rng, 20, (1, m)) + rng.uniform(0, 0.01, (1, m))
        q = probability_grid(m)
        r_with = log_ratio_curve(type7_quantiles(case + 1, q), type7_quantiles(ctrl + 1, q))
        r_without = log_ratio_curve(type7_quantiles(case, q), type7_quantiles(ctrl, q))
        worst_with = min(worst_with, float(affected_fraction(r_with)[0]))
        worst_without = min(worst_without, float(affected_fraction(r_without)[0]))
    assert worst_with >= 0.9, f"a global 2x read {worst_with:.2f} on some seed"
    assert worst_without < 0.8, "the mechanism this guards against no longer reproduces"


def test_stage2_one_sided_greater_is_clean_at_low_counts():
    """The cancer-outlier direction; 0.03-0.04 before thinning too, because
    the old artifact was bottom-heavy (10.2). Still clean."""
    rng = np.random.default_rng(36)
    rate, _ = _stage2_rate(_raw_counts(rng, 2.0, "global"), alternative="greater")
    assert rate <= 0.12
