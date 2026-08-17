"""Layer 9 — method validation. Not parity.

The three simulations from ``reference/R/validation_sims_v7.R``, ported.
They validate the **method** rather than the port, which is why they
belong in the repository at all rather than only in a notebook.

**These cannot match the R to the digit and are not supposed to.** The
generators are different (hazard 2), so the synthetic data differ and so
do the permutation nulls. What must match is the *behaviour*: calibration
at nominal, subset-versus-bulk separation on the tail axis, and a power
curve that steps where the combinatorial floor crosses the BH threshold.
The R's measured values are quoted at each assertion as the target, and
the port's own values are printed so the two can be read side by side.

Run with ``-m validation -s`` to see the tables.
"""

from __future__ import annotations

import numpy as np
import pytest

import wade

pytestmark = [pytest.mark.validation, pytest.mark.slow]

# The cfRNA v7 primary contrast's own geometry, so the simulation inherits
# a real design rather than a round number. min(n0, n1) IS the quantile
# grid, so the 10% tail window is 2 order statistics here.
N_CASES = 77
N_CONTROLS = 18
NPERMS = 1000
G_NULL = 800
G_BACKGROUND = 600
N_PLANTED = 25
N_SUBSET_PWR = 80
SUBSET_FRAC = 0.08
SPIKE_MU = 6.2
LEN_CONST = 4.0
FRACS = (0.03, 0.05, 0.08, 0.12, 0.20, 0.35, 0.50)

NN = N_CASES + N_CONTROLS
COND = np.r_[np.ones(N_CASES, int), np.zeros(N_CONTROLS, int)]


def _mk_subset(rng, frac, spike_mu=SPIKE_MU):
    """Case group at the null, with a random fraction replaced by a spike."""
    x = np.r_[rng.lognormal(3, 0.6, N_CASES), rng.lognormal(3, 0.6, N_CONTROLS)]
    k = max(1, round(frac * N_CASES))
    sp = rng.choice(N_CASES, size=k, replace=False)
    x[sp] = rng.lognormal(spike_mu, 0.3, k)
    return x


def _mk_bulk(rng, shift):
    """The whole case group's meanlog shifted."""
    return np.r_[rng.lognormal(3 + shift, 0.6, N_CASES),
                 rng.lognormal(3, 0.6, N_CONTROLS)]


@pytest.fixture(scope="module")
def null_calibration():
    rng = np.random.default_rng(2024)
    counts = rng.poisson(30, size=(G_NULL, NN)).astype(float)
    normalizer = rng.integers(1, 13, size=G_NULL).astype(float)
    return wade.wade(counts, normalizer, COND, nperms=NPERMS, seed=1)


def test_a_null_calibration_pvalues_are_uniform(null_calibration):
    """Poisson counts, random per-gene normalizers, labels with no signal.

    Any departure from Uniform(0, 1) here is the method inventing
    significance.

    The KS test warns about ties because permutation p-values are
    **discrete** — the unrefined ones are multiples of 1/(B+1) — so its
    p-value is conservative rather than exact. That is a property of any
    permutation test's calibration check and it appears in any language.
    The realised type-I error, which assumes no continuity, is reported
    alongside for that reason.

    R sandbox targets: KS p = 0.6554 (bulk) and 0.2327 (subset); type-I
    at nominal 0.05 of 0.0512 and 0.0488.
    """
    from scipy import stats

    res = null_calibration
    ks_diff = stats.kstest(res.p_diff, "uniform").pvalue
    ks_tail = stats.kstest(res.p_tail, "uniform").pvalue
    t1_diff = float(np.mean(res.p_diff < 0.05))
    t1_tail = float(np.mean(res.p_tail < 0.05))

    print(f"\n(a) null calibration, {G_NULL} genes at {N_CASES} vs {N_CONTROLS}, "
          f"nperms = {NPERMS}")
    print(f"    KS vs Uniform(0,1):  diff.mean p = {ks_diff:.4f}   "
          f"tail.mean p = {ks_tail:.4f}      [R: 0.6554 / 0.2327]")
    print(f"    type-I error @0.05:  diff.mean = {t1_diff:.4f}     "
          f"tail.mean = {t1_tail:.4f}        [R: 0.0512 / 0.0488]")

    assert ks_diff > 0.01, f"bulk axis departs from uniform (KS p = {ks_diff:.4g})"
    assert ks_tail > 0.01, f"subset axis departs from uniform (KS p = {ks_tail:.4g})"
    # Binomial s.e. at 800 genes is 0.0077, so +/- 0.02 is about 2.6 s.e.
    assert abs(t1_diff - 0.05) < 0.02, f"bulk type-I error {t1_diff:.4f} off nominal"
    assert abs(t1_tail - 0.05) < 0.02, f"subset type-I error {t1_tail:.4f} off nominal"


def test_a_null_calibration_is_not_anticonservative_on_either_axis(null_calibration):
    """Both axes independently, since neither gates the other."""
    res = null_calibration
    for axis, p in (("bulk", res.p_diff), ("subset", res.p_tail)):
        for alpha in (0.01, 0.05, 0.10, 0.25):
            realised = float(np.mean(p < alpha))
            assert realised < alpha + 3 * np.sqrt(alpha * (1 - alpha) / G_NULL) + 0.01, (
                f"{axis} axis anticonservative at alpha={alpha}: {realised:.4f}"
            )


@pytest.fixture(scope="module")
def discrimination():
    rng = np.random.default_rng(11)
    bg = np.array([np.r_[rng.lognormal(3, 0.6, N_CASES),
                         rng.lognormal(3, 0.6, N_CONTROLS)]
                   for _ in range(G_BACKGROUND)])
    bulk = np.array([_mk_bulk(rng, rng.uniform(0.5, 0.9)) for _ in range(N_PLANTED)])
    sub = np.array([_mk_subset(rng, SUBSET_FRAC) for _ in range(N_PLANTED)])
    mm = np.vstack([bg, bulk, sub])
    ll = np.full(mm.shape[0], LEN_CONST)
    label = np.array(["null"] * G_BACKGROUND + ["bulk"] * N_PLANTED
                     + ["subset"] * N_PLANTED)
    return wade.wade(mm, ll, COND, nperms=NPERMS, seed=1), label, bg


def test_b_subset_genes_separate_from_bulk_shifts_on_the_tail_axis(discrimination):
    """The entire reason the tail statistic exists.

    ``diff_mean`` alone collapses a rare-subset signal and a whole-group
    shift to the same number. The subset axis must not.

    R sandbox targets: planted subset genes at median ``tail_mean``
    27,537 against ``diff_mean`` 3,046, a ratio of 8.9; planted bulk
    shifts at 4,064 against 1,348, a ratio of 3.3; the null background at
    535 against -46.
    """
    res, label, _ = discrimination
    print(f"\n(b) subset vs bulk discrimination, {res.diff_mean.size} genes"
          f"                    [R medians]")
    print(f"    {'class':<8s} {'n':>4s} {'median diff.mean':>18s} "
          f"{'median tail.mean':>18s} {'ratio':>8s}")
    ratios = {}
    for cls, r_dm, r_tm, r_ratio in (("null", 535, -46, None),
                                     ("bulk", 1348, 4064, 3.3),
                                     ("subset", 3046, 27537, 8.9)):
        m = label == cls
        dm = float(np.median(res.diff_mean[m]))
        tm = float(np.median(res.tail_mean[m]))
        ratios[cls] = float(np.median(res.tail_mean[m] / res.diff_mean[m]))
        print(f"    {cls:<8s} {m.sum():4d} {dm:18.1f} {tm:18.1f} "
              f"{ratios[cls]:8.2f}   [R: {r_dm}, {r_tm}"
              + (f", {r_ratio}]" if r_ratio else "]"))

    assert ratios["subset"] > ratios["bulk"], (
        "planted subset genes must carry more of their signal in the tail "
        "than planted bulk shifts do"
    )
    assert ratios["subset"] > 2 * ratios["bulk"], (
        f"separation too weak: subset {ratios['subset']:.2f} vs bulk {ratios['bulk']:.2f}"
    )

    # And the tail axis must actually rank the subset genes above the bulk ones,
    # which is the operational claim rather than a summary statistic.
    med_tail_sub = np.median(res.tail_mean[label == "subset"])
    med_tail_bulk = np.median(res.tail_mean[label == "bulk"])
    assert med_tail_sub > med_tail_bulk


def test_b_the_bulk_axis_alone_would_not_separate_them(discrimination):
    """The negative control for the claim above.

    If ``diff_mean`` separated the two planted classes as cleanly as
    ``tail_mean`` does, the subset axis would be redundant. Measured in
    the R, the subset genes' bulk medians (3,046) sit close to the bulk
    shifts' (1,348) — same order of magnitude — while their tail medians
    (27,537 vs 4,064) differ by nearly a factor of 7.
    """
    res, label, _ = discrimination
    sub_dm = np.median(res.diff_mean[label == "subset"])
    bulk_dm = np.median(res.diff_mean[label == "bulk"])
    sub_tm = np.median(res.tail_mean[label == "subset"])
    bulk_tm = np.median(res.tail_mean[label == "bulk"])

    bulk_axis_separation = sub_dm / bulk_dm
    tail_axis_separation = sub_tm / bulk_tm
    print(f"    separation on the bulk axis  : {bulk_axis_separation:.2f}x")
    print(f"    separation on the subset axis: {tail_axis_separation:.2f}x")
    assert tail_axis_separation > bulk_axis_separation


def _pfloor(k):
    """P(a random relabelling puts all k signal-carrying samples in the case group).

    The smallest p-value ANY label-permutation test can return for a
    k-sample subset, so it is a property of the design. No number of
    permutations and no better tail fit moves it.
    """
    from math import lgamma

    def lchoose(n, r):
        return lgamma(n + 1) - lgamma(r + 1) - lgamma(n - r + 1)

    return np.exp(lchoose(N_CASES, k) - lchoose(NN, k))


def test_c_power_steps_where_the_floor_crosses_the_BH_threshold(discrimination):
    """Power against the exact combinatorial floor.

    The 0 -> 1 transition is **not** a simulation artefact and not a limit
    of the permutation count. With ``N_SUBSET_PWR`` true positives among
    ``G_BACKGROUND + N_SUBSET_PWR`` genes, BH at q = 0.10 can only declare
    p-values at or below ``0.10 * 80/680 = 0.0118``; the floor column
    crosses that threshold between 20% and 35% of cases. Raising nperms
    cannot move it.

    R sandbox target: power 0 at every fraction through 20%, and 1 from
    35% on.
    """
    _, _, bg = discrimination
    bh_threshold = 0.10 * N_SUBSET_PWR / (G_BACKGROUND + N_SUBSET_PWR)

    print(f"\n(c) power vs subset fraction (BH q = 0.10, declarable below "
          f"{bh_threshold:.4f})")
    print(f"    {'frac':>6s} {'k':>4s} {'floor':>12s} {'power':>7s}   "
          f"{'floor <= thr':>12s}   [R power]")

    r_power = {0.03: 0, 0.05: 0, 0.08: 0, 0.12: 0, 0.20: 0, 0.35: 1, 0.50: 1}
    rows = []
    for frac in FRACS:
        k = max(1, round(frac * N_CASES))
        rng = np.random.default_rng(200 + k)
        s = np.array([_mk_subset(rng, frac) for _ in range(N_SUBSET_PWR)])
        mm = np.vstack([bg, s])
        ll = np.full(mm.shape[0], LEN_CONST)
        rr = wade.wade(mm, ll, COND, nperms=NPERMS, seed=1)
        is_sub = np.r_[np.zeros(G_BACKGROUND, bool), np.ones(N_SUBSET_PWR, bool)]
        power = float(np.mean(rr.padj_tail[is_sub] < 0.10))
        floor = _pfloor(k)
        rows.append((frac, k, floor, power))
        print(f"    {frac:6.2f} {k:4d} {floor:12.3e} {power:7.2f}   "
              f"{str(floor <= bh_threshold):>12s}   [{r_power[frac]}]")

    for frac, k, floor, power in rows:
        if floor > bh_threshold:
            assert power == 0.0, (
                f"at {frac:.0%} the floor {floor:.3e} exceeds the BH threshold "
                f"{bh_threshold:.4f}, so no gene can be declared — got power {power}"
            )

    # The step must actually happen, and in the documented interval.
    powers = {frac: p for frac, _, _, p in rows}
    assert powers[0.20] < 0.5 <= powers[0.35], (
        f"power should step between 20% and 35% of cases; "
        f"got {powers[0.20]:.2f} -> {powers[0.35]:.2f}"
    )
    assert powers[0.50] >= 0.9


def test_c_the_floor_is_a_property_of_the_design_not_the_implementation():
    """Worked directly, without running the method.

    This is the arithmetic behind ``docs/limits.md``: at the v7 design the
    floor is 0.43 at 5% of cases, which is why the source documents'
    "below roughly 5% of cases" understates the constraint badly.
    """
    assert _pfloor(max(1, round(0.05 * N_CASES))) == pytest.approx(0.425, abs=0.01)
    assert _pfloor(max(1, round(0.03 * N_CASES))) == pytest.approx(0.655, abs=0.01)
    assert _pfloor(max(1, round(0.20 * N_CASES))) == pytest.approx(0.0320, abs=1e-3)
    assert _pfloor(max(1, round(0.35 * N_CASES))) == pytest.approx(1.15e-3, rel=0.05)

    # Monotone decreasing in k, and it reaches below the BH threshold only
    # somewhere between 20% and 35% of cases.
    ks = [max(1, round(f * N_CASES)) for f in FRACS]
    floors = [_pfloor(k) for k in ks]
    assert all(a > b for a, b in zip(floors, floors[1:]))
