"""Method validation — not parity.

These validate the **method** rather than the port, which is why they belong in
the repository rather than only in a notebook. Rewritten around the two-stage
design; the originals validated the retired tail window.

Three claims, each with its own simulation:

(a) **Calibration.** Neither stage manufactures significance.
(b) **Discrimination.** A gene altered in a subset and a gene shifted globally
    can produce the same mean difference; the subset stage and
    ``affected_fraction`` must separate them. This is the entire reason WADE
    exists rather than an ordinary DE test.
(c) **Power against the combinatorial floor**, which is a property of the
    design that no detector, effect size or permutation count can move.
"""

from __future__ import annotations

from math import lgamma

import numpy as np
import pytest

import wade

pytestmark = pytest.mark.validation

NPERMS = 300


def _lchoose(n, r):
    return lgamma(n + 1) - lgamma(r + 1) - lgamma(n - r + 1)


def comb_floor(n1: int, n: int, k: int) -> float:
    """P(a random relabelling puts all k signal-carrying samples in one group).

    The scale of the smallest p-value a label-permutation test can resolve for
    a k-sample subset. A property of the design, not of the implementation —
    and a scale rather than a bound: see ``docs/limits.md`` §4.1, where 21-42%
    of planted genes came in under it.
    """
    if k < 1 or k > n1:
        return 1.0
    return float(np.exp(_lchoose(n1, k) - _lchoose(n, k)))


def _matrix(rng, spec, n1, n0, sdlog=0.6):
    rows, lab = [], []
    for name, kind, arg, count in spec:
        for _ in range(count):
            ctrl = rng.lognormal(3, sdlog, n0)
            case = rng.lognormal(3, sdlog, n1)
            if kind == "global":
                case = case * arg
            elif kind == "subset":
                k = max(1, round(arg * n1))
                case[rng.choice(n1, k, replace=False)] *= 8.0
            rows.append(np.r_[case, ctrl])
            lab.append(name)
    return np.array(rows), np.array(lab)


@pytest.fixture(scope="module")
def calibration():
    n1, n0 = 77, 18
    rng = np.random.default_rng(2024)
    counts = rng.poisson(30, size=(800, n1 + n0)).astype(float)
    normalizer = rng.integers(1, 13, size=800).astype(float)
    cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
    return wade.wade(counts, normalizer, cond, nperms=NPERMS, seed=1)


def test_a_neither_stage_manufactures_significance(calibration):
    """Poisson counts, labels carrying no signal by construction.

    Any departure from Uniform(0, 1) here is the method inventing signal. The
    KS test warns about ties because permutation p-values are discrete — a
    property of any permutation test's calibration check — so the realised
    type-I error, which assumes no continuity, is reported alongside.
    """
    from scipy import stats

    res = calibration
    print(f"\n(a) calibration, 800 genes at 77 v 18, nperms = {NPERMS}")
    for name, p in (("mean_shift", res.p_mean_shift), ("subset", res.p_subset)):
        ks = stats.kstest(p, "uniform").pvalue
        t1 = float(np.mean(p <= 0.05))
        print(f"    {name:<12s} KS vs Uniform(0,1) p = {ks:.4f}    type-I at 0.05 = {t1:.4f}")
        assert ks > 0.01, f"{name} departs from uniform (KS p = {ks:.4g})"
        assert abs(t1 - 0.05) < 0.025, f"{name} type-I error {t1:.4f} off nominal"


def test_a_neither_stage_is_anticonservative(calibration):
    res = calibration
    g = res.p_mean_shift.size
    for name, p in (("mean_shift", res.p_mean_shift), ("subset", res.p_subset)):
        for alpha in (0.01, 0.05, 0.10, 0.25):
            realised = float(np.mean(p <= alpha))
            bound = alpha + 3 * np.sqrt(alpha * (1 - alpha) / g) + 0.01
            assert realised < bound, f"{name} anticonservative at {alpha}: {realised:.4f}"


@pytest.fixture(scope="module")
def discrimination():
    n1 = n0 = 200
    rng = np.random.default_rng(11)
    # A realistic signal fraction: library-size normalization couples genes, so
    # a signal-saturated matrix would shift the characterization of every gene.
    spec = [("null", None, None, 600),
            ("global", "global", 2.0, 60),
            ("subset 5%", "subset", 0.05, 60),
            ("subset 20%", "subset", 0.20, 60)]
    x, lab = _matrix(rng, spec, n1, n0)
    cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]
    return wade.wade(x, np.full(x.shape[0], 4.0), cond, nperms=NPERMS, seed=1), lab


def test_b_the_subset_stage_separates_concentrated_from_global(discrimination):
    """The claim WADE exists to make.

    A global shift must be found by the mean-shift stage and left alone by the
    subset stage; a concentrated difference must fire the subset stage. Without
    the second half, "subset" would mean nothing.
    """
    res, lab = discrimination
    print(f"\n(b) discrimination, {lab.size} genes at 200 v 200")
    print(f"    {'class':<12s} {'p_mean<=.05':>12s} {'p_subset<=.05':>14s} "
          f"{'affected_fraction':>18s} {'direction':>10s}")
    rate = {}
    for c in ("null", "global", "subset 5%", "subset 20%"):
        s = lab == c
        rate[c] = (float(np.mean(res.p_mean_shift[s] <= 0.05)),
                   float(np.mean(res.p_subset[s] <= 0.05)))
        print(f"    {c:<12s} {rate[c][0]:>12.3f} {rate[c][1]:>14.3f} "
              f"{np.median(res.affected_fraction[s]):>18.3f} "
              f"{np.median(res.direction[s]):>+10.3f}")

    assert rate["global"][0] > 0.9, "a 2x global shift must be detected"
    assert rate["global"][1] < 0.15, (
        "a genuine global shift must NOT fire the subset stage — without this, "
        "'a subset explains it better' is not a claim about anything"
    )
    assert rate["subset 5%"][1] > 0.6, "a 5% subset must fire the subset stage"
    assert rate["subset 20%"][1] > 0.8


def test_b_affected_fraction_orders_the_classes(discrimination):
    """The characterization must recover *how much* of the group differs."""
    res, lab = discrimination
    med = {c: float(np.median(res.affected_fraction[lab == c]))
           for c in ("global", "subset 5%", "subset 20%")}
    assert med["subset 5%"] < med["subset 20%"] < med["global"]
    assert med["global"] > 0.8, "a global change should read near 1"
    assert med["subset 5%"] < 0.2, "a 5% subset should read as a small fraction"


def test_c_power_is_bounded_by_the_combinatorial_floor():
    """No detector can beat the design.

    If ``k`` samples carry the signal, label shuffling puts all of them in the
    case group with probability ``C(n1,k)/C(n,k)``. Where that exceeds alpha, no
    effect size and no permutation count can separate the signal — so the test
    asserts the *mechanism*, not a memorized curve.
    """
    n1 = n0 = 60
    n = n1 + n0
    rng = np.random.default_rng(200)
    cond = np.r_[np.ones(n1, int), np.zeros(n0, int)]

    print(f"\n(c) power vs affected fraction at {n1} v {n0}, alpha = 0.05")
    print(f"    {'frac':>6s} {'k':>4s} {'floor':>11s} {'p_subset power':>15s}")
    for frac in (0.02, 0.05, 0.10, 0.25):
        k = max(1, round(frac * n1))
        floor = comb_floor(n1, n, k)
        x, lab = _matrix(rng, [("null", None, None, 200),
                               ("planted", "subset", frac, 60)], n1, n0)
        res = wade.wade(x, np.full(x.shape[0], 4.0), cond, nperms=NPERMS, seed=1)
        power = float(np.mean(res.p_subset[lab == "planted"] <= 0.05))
        print(f"    {frac:>6.0%} {k:>4d} {floor:>11.3g} {power:>15.3f}")
        if floor > 0.05:
            assert power < 0.25, (
                f"at {frac:.0%} the floor is {floor:.3g}, above alpha — nothing "
                f"in this family can detect it, yet power was {power:.3f}"
            )
        elif floor < 1e-4:
            assert power > 0.5, f"well below the floor, power should be real; got {power:.3f}"


def test_c_the_floor_is_arithmetic_not_simulation():
    """Worked directly, so the limit is checkable without running anything."""
    assert comb_floor(77, 95, 4) == pytest.approx(0.425, abs=0.01)
    assert comb_floor(77, 95, 15) == pytest.approx(0.0320, abs=1e-3)
    assert comb_floor(200, 400, 10) == pytest.approx(0.00087, rel=0.05)
    ks = [1, 2, 4, 8, 16]
    floors = [comb_floor(77, 95, k) for k in ks]
    assert all(a > b for a, b in zip(floors, floors[1:])), "monotone decreasing in k"
