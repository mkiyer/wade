"""The subset test and the characterization statistics — docs/method.md sections 3-4.

These are new work with no R counterpart, so there is nothing to be in parity
*with*. They are validated on their own terms, against planted ground truth,
and the assertions are the properties the design claims rather than remembered
numbers.

**On negative-binomial counts**, which is WADE's scope (``docs/method.md``
S10): dispersion 0.1, mean 50 counts unless a test says otherwise, and the
default correction — binomial thinning of the counts, with a one-count
pseudocount on the log-ratio curve. An earlier version of this file generated
continuous lognormal values, a pure multiplicative model in which the
division correction is exact and the count regime is never exercised; that is
how the low-expression failure in S10.2 went unnoticed. ``test_scale.py``
covers the expression levels below a few counts, this file the ordinary one.

Library sizes are fixed at 1 and ``norm_factor`` at 1, so a normalized value
**is** a count and the pseudocount is exactly +1. That also decouples the
genes, which is deliberate: composition is tested on purpose in
``test_composition_shifts_the_characterization_and_this_is_real`` and would
otherwise contaminate every other assertion here.
"""

from __future__ import annotations

import numpy as np
import pytest

import wade
from wade.subset import bridge, log_ratio_curve, subset_test

N1 = N0 = 400
COND = np.r_[np.ones(N1, int), np.zeros(N0, int)]
PHI = 0.1            # NB dispersion; biological CV ~ 0.32
MU = 50.0            # counts per sample, unless a test overrides it
PSEUDO = 1.0         # one count, since values are counts here


def _nb(rng, mu, size):
    """Negative binomial as a Poisson-Gamma mixture: Var = mu + PHI mu^2."""
    return rng.poisson(rng.gamma(1 / PHI, PHI * mu, size=size)).astype(float)


def _mk(rng, kind, arg=None, n1=N1, n0=N0, mu=MU):
    """One gene's raw counts, cases first. ``kind`` plants the ground truth."""
    ctrl = _nb(rng, mu, n0)
    case = _nb(rng, mu, n1)
    if kind == "global":
        case = _nb(rng, mu * arg, n1)
    elif kind == "var":
        # Wider spread, same *median*: half the cases up by arg, half down by
        # arg. Not the same thing as raising the NB dispersion at a fixed mean,
        # which also skews the distribution — measured, that reads direction
        # -0.4 to -0.65 rather than 0, because the median falls while the mean
        # is held. This is the symmetric-spread scenario direction is for.
        hi, lo = _nb(rng, mu * arg, n1), _nb(rng, mu / arg, n1)
        case = np.where(rng.random(n1) < 0.5, hi, lo)
    elif kind in ("up", "down"):
        k = max(1, round(arg * n1))
        idx = rng.choice(n1, k, replace=False)
        case[idx] = _nb(rng, mu * 8.0 if kind == "up" else mu / 8.0, k)
    elif kind == "both":
        k = max(1, round(arg * n1))
        idx = rng.choice(n1, 2 * k, replace=False)
        case[idx[:k]] = _nb(rng, mu * 8.0, k)
        case[idx[k:]] = _nb(rng, mu / 8.0, k)
    return np.r_[case, ctrl]


def _run(x, nperms=300, **kw):
    """Through the raw-count entry point, with counts as the normalized unit."""
    return wade.wade(x, np.ones(x.shape[0]), COND, lib_sizes=np.ones(x.shape[1]),
                     norm_factor=1.0, nperms=nperms, seed=1, **kw)


def _curve(rng, kind, arg=None, rep=80, cond=None, **kw):
    """The log-ratio curve as the tool computes it: counts, plus one count."""
    x = np.array([_mk(rng, kind, arg, **kw) for _ in range(rep)])
    st = wade.wade_stats(x + PSEUDO, COND if cond is None else cond)
    return x, log_ratio_curve(st.Q1, st.Q0)


# ---------------------------------------------------------------------
# The bridge
# ---------------------------------------------------------------------

def test_bridge_is_zero_for_a_flat_curve():
    """A pure global fold change is a flat R, so the bridge vanishes."""
    r = np.full((5, 40), 1.7)
    assert np.allclose(bridge(r), 0.0, atol=1e-12)


def test_bridge_is_exactly_invariant_to_a_global_shift():
    """The property the whole two-stage design rests on.

    ``R -> R - c`` sends ``S_k -> S_k - kc`` and leaves ``B_k`` untouched. It is
    what lets the observed statistic stay fixed while only its null is rebuilt
    under the shift model.
    """
    rng = np.random.default_rng(0)
    r = rng.normal(size=(20, 60))
    for c in (-3.5, -0.1, 0.0, 0.75, 12.0):
        assert np.allclose(bridge(r + c), bridge(r), rtol=0, atol=1e-12)


def test_bridge_last_position_is_identically_zero():
    rng = np.random.default_rng(1)
    assert np.allclose(bridge(rng.normal(size=(9, 30)))[:, -1], 0.0, atol=1e-12)


def test_bridge_is_positive_when_the_difference_is_front_loaded():
    r = np.zeros((1, 50))
    r[0, :5] = 3.0                      # top 10% of quantiles carry everything
    assert bridge(r)[0, :40].max() > 0


# ---------------------------------------------------------------------
# affected_fraction -- the effective affected fraction
# ---------------------------------------------------------------------

@pytest.mark.parametrize("frac", [0.02, 0.05, 0.10, 0.25, 0.50])
def test_affected_fraction_recovers_the_planted_fraction(frac):
    rng = np.random.default_rng(11)
    _, r = _curve(rng, "up", frac)
    got = np.median(wade.affected_fraction(r))
    assert abs(got - frac) < max(0.04, 0.25 * frac), (
        f"planted {frac:.0%}, estimated {got:.3f}"
    )


@pytest.mark.parametrize("fc", [1.5, 2.0, 4.0, 8.0])
def test_affected_fraction_is_one_for_a_global_fold_change_of_any_size(fc):
    """The anchor. This is what the log scale buys and the raw scale cannot."""
    rng = np.random.default_rng(12)
    _, r = _curve(rng, "global", fc)
    assert np.median(wade.affected_fraction(r)) > 0.93


def test_affected_fraction_on_the_raw_scale_would_lose_the_anchor():
    """Why section 4 specifies the log-ratio curve, asserted rather than asserted-to.

    On the absolute scale a global fold change produces a difference curve
    proportional to Q(p), which is itself concentrated at high quantiles, so the
    same statistic reads a global change as partial — and the reading drifts with
    each gene's own dispersion instead of anchoring at 1.
    """
    rng = np.random.default_rng(13)
    x = np.array([_mk(rng, "global", 2.0) for _ in range(80)])
    st = wade.wade_stats(x + PSEUDO, COND)
    on_log = np.median(wade.affected_fraction(log_ratio_curve(st.Q1, st.Q0)))
    on_raw = np.median(wade.affected_fraction(st.D))
    assert on_log > 0.93
    assert on_raw < 0.80, f"raw scale should NOT anchor at 1; got {on_raw:.3f}"


def test_affected_fraction_is_bounded():
    rng = np.random.default_rng(14)
    for kind, arg in (("up", 0.05), ("global", 3.0), ("var", 2.0), ("down", 0.2)):
        _, r = _curve(rng, kind, arg, rep=40)
        v = wade.affected_fraction(r)
        assert np.all((v > 0) & (v <= 1.0 + 1e-12))


def test_affected_fraction_resolution_limit_is_one_over_m():
    """Below m ~ 50 it stops being a fraction estimate and stays qualitative.

    Stated openly rather than hidden: the grid has m points, so no finer
    fraction is resolvable. Global still separates from concentrated.
    """
    rng = np.random.default_rng(15)
    for n in (20, 500):
        cond = np.r_[np.ones(n, int), np.zeros(n, int)]
        _, r_sub = _curve(rng, "up", 0.05, rep=60, n1=n, n0=n, cond=cond)
        _, r_glob = _curve(rng, "global", 3.0, rep=60, n1=n, n0=n, cond=cond)
        sub, glob = np.median(wade.affected_fraction(r_sub)), np.median(wade.affected_fraction(r_glob))
        assert glob > sub, f"m={n}: global {glob:.3f} must exceed concentrated {sub:.3f}"
        if n == 500:
            assert abs(sub - 0.05) < 0.02, f"at m=500 the fraction should be quantitative, got {sub:.3f}"


# ---------------------------------------------------------------------
# direction -- direction
# ---------------------------------------------------------------------

def test_direction_separates_orientation_and_symmetry():
    """The pair (affected_fraction, direction) is the whole characterization."""
    rng = np.random.default_rng(16)
    expect = {("global", 2.0): +1.00, ("global", 0.5): -1.00,
              ("up", 0.05): +0.92, ("down", 0.05): -0.93,
              ("var", 1.6): 0.00, ("both", 0.05): 0.00}
    for (kind, arg), want in expect.items():
        _, r = _curve(rng, kind, arg)
        got = float(np.median(wade.direction(r)))
        assert abs(got - want) < 0.12, (
            f"{kind}({arg}): direction {got:+.3f}, expected ~{want:+.2f}"
        )


def test_direction_is_bounded():
    rng = np.random.default_rng(17)
    for kind, arg in (("up", 0.1), ("down", 0.1), ("var", 2.0), ("global", 0.25)):
        _, r = _curve(rng, kind, arg, rep=40)
        v = wade.direction(r)
        assert np.all((v >= -1.0 - 1e-12) & (v <= 1.0 + 1e-12))


def test_variance_change_and_one_sided_subset_are_distinguishable():
    """Both give affected_fraction ~ 0.3; only direction tells them apart.

    This is the caveat that made direction necessary: the subset test's claim is
    'a global shift does not explain this', which a variance change satisfies.
    """
    rng = np.random.default_rng(18)
    # 1.3 is the spread that puts this scenario at the same affected_fraction
    # as a 30% subset on counts (0.30 against 0.31); the point of the test is
    # that the two are indistinguishable on that statistic alone.
    _, r_var = _curve(rng, "var", 1.3)
    _, r_sub = _curve(rng, "up", 0.30)
    pi_var, pi_sub = np.median(wade.affected_fraction(r_var)), np.median(wade.affected_fraction(r_sub))
    assert abs(pi_var - pi_sub) < 0.15, "the two should be similar on affected_fraction alone"
    assert abs(np.median(wade.direction(r_var))) < 0.15
    assert np.median(wade.direction(r_sub)) > 0.80


# ---------------------------------------------------------------------
# The subset test
# ---------------------------------------------------------------------

def _rates(rng, kind, arg, rep=120, n_perms=300, alpha=0.05, **kw):
    """Rate at which the subset stage fires, through the public entry point so
    the shipped correction (thinning) and pseudocount are what is tested."""
    x = np.array([_mk(rng, kind, arg) for _ in range(rep)])
    res = _run(x, nperms=n_perms, **kw)
    return float(np.mean(res.p_subset <= alpha)), res.subset


def test_shape_test_holds_nominal_level_under_the_null():
    rng = np.random.default_rng(21)
    rate, _ = _rates(rng, "null", None)
    assert rate < 0.12, f"type-I error {rate:.3f}"


@pytest.mark.parametrize("fc", [1.5, 2.0, 8.0])
def test_shape_test_is_SILENT_on_a_genuine_global_fold_change(fc):
    """The property the whole test depends on.

    Without this, "a subset explains it better" means nothing. It is also the
    thing that fails loudly if the null is built by permuting the raw data
    instead of the shift-corrected data — measured, 14-19% instead of 5%.
    """
    rng = np.random.default_rng(22)
    rate, _ = _rates(rng, "global", fc)
    assert rate < 0.12, f"fold change {fc}x fires the subset test at {rate:.3f}"


def test_shape_test_detects_a_small_subset():
    rng = np.random.default_rng(23)
    rate, _ = _rates(rng, "up", 0.05)
    assert rate > 0.70, f"power on a 5% subset only {rate:.3f}"


def test_shape_test_sees_downward_subsets_the_mean_test_cannot():
    """Documented as in-scope: WADE was one-sided-up and blind to these."""
    rng = np.random.default_rng(24)
    rate, res = _rates(rng, "down", 0.10)
    assert rate > 0.70
    assert np.median(res.direction) < -0.7, "direction must report these as downward"


def test_the_shift_correction_is_what_delivers_specificity():
    """Assert the mechanism, not just the outcome — and rank the three of them.

    Stage 2's null is *a global fold change*, not no-difference, and how that
    null is built is the whole design (docs/method.md S3, S10.3). Three ways to
    build it, measured here on a genuine 2x at 50 counts:

    * **no correction** — permute the raw matrix, which tests against
      no-difference and fires on real fold changes;
    * **division** — the continuous-data correction, better but not exact for
      counts, because a fold change in an NB mean is not a multiplicative
      shift of the NB distribution;
    * **thinning** — the count-native one, which is what wade() does.

    The ordering is the claim. Removing the correction, or reverting thinning
    to division, must fail here.
    """
    from wade.permutation import _subset_null_numpy
    from wade.quantiles import probability_grid
    from wade.thinning import fit_fold_change, thin_counts

    rng = np.random.default_rng(25)
    counts = np.array([_mk(rng, "global", 2.0) for _ in range(120)])
    x = counts + PSEUDO
    perms = wade.draw_perms(COND, 300, seed=3)
    q = probability_grid(min(N1, N0))
    st = wade.wade_stats(x, COND)
    r_obs = log_ratio_curve(st.Q1, st.Q0)

    def rate(matrix, b):
        stat, null, *_ = _subset_null_numpy(matrix, b, perms, q, "two-sided")
        p = (1 + (null >= stat[:, None]).sum(1)) / 301
        return float(np.mean(p <= 0.05))

    def bridge_of(matrix):
        s = wade.wade_stats(matrix, COND)
        return bridge(log_ratio_curve(s.Q1, s.Q0))

    b_obs = bridge(r_obs)
    none = rate(x, b_obs)
    division = rate(wade.subset.shift_correct(x, COND, r_obs), b_obs)

    normalize = lambda c: c + 0.0                      # counts already are the scale
    fold = fit_fold_change(counts, COND, normalize, seed=0)
    thinned = thin_counts(counts, COND, fold, np.random.default_rng(1)) + PSEUDO
    thinning = rate(thinned, bridge_of(thinned))

    assert abs(np.median(fold) - 2.0) < 0.15, f"f-hat {np.median(fold):.2f}"
    assert thinning < 0.12, f"the count-native null must hold nominal level; got {thinning:.3f}"
    assert thinning < division < none, (
        f"expected thinning < division < no correction; got {thinning:.3f}, "
        f"{division:.3f}, {none:.3f}"
    )


def test_shape_test_refuses_a_grid_too_small_to_have_a_bridge():
    rng = np.random.default_rng(26)
    cond = np.r_[np.ones(6, int), np.zeros(2, int)]
    x = _nb(rng, MU, (4, 8)) + PSEUDO
    with pytest.raises(ValueError, match="at least 3 grid points"):
        subset_test(x, cond, wade.draw_perms(cond, 10, seed=1))


def test_affected_fraction_is_robust_to_signal_shape_and_the_scan_argmax_is_not():
    """Why affected_fraction is the estimator and the scan's argmax is only a diagnostic.

    The argmax tracks the planted fraction when the affected samples are
    shifted multiplicatively, and underestimates it several-fold when their
    values are replaced outright — the log-ratio curve then declines steeply
    across the affected region, so the cumulative departure peaks before it
    ends. affected_fraction is accurate under both, which is the property that matters
    when nobody knows the signal's shape in advance.
    """
    rng = np.random.default_rng(27)
    perms = wade.draw_perms(COND, 150, seed=2)

    def gen(shape, frac):
        case = _nb(rng, MU, N1)
        k = max(1, round(frac * N1))
        idx = rng.choice(N1, k, replace=False)
        case[idx] = _nb(rng, MU * 8.0, k) if shape == "multiply" else _nb(rng, MU * 20.0, k)
        return np.r_[case, _nb(rng, MU, N0)]

    err = {"argmax": 0.0, "affected_fraction": 0.0}
    for shape in ("multiply", "replace"):
        for frac in (0.05, 0.10, 0.25):
            x = np.array([gen(shape, frac) for _ in range(60)])
            res = subset_test(x, COND, perms)
            # argmax_k / m — the width the scan chose, computed here rather than
            # carried as a property nothing else consumed.
            scan_fraction = res.argmax_k / res.r.shape[1]
            err["argmax"] += abs(float(np.median(scan_fraction)) - frac)
            err["affected_fraction"] += abs(float(np.median(res.affected_fraction)) - frac)
            assert abs(float(np.median(res.affected_fraction)) - frac) < max(0.04, 0.2 * frac), (
                f"affected_fraction must track {frac:.0%} under a {shape} signal"
            )
    assert err["affected_fraction"] < 0.5 * err["argmax"], (
        f"affected_fraction error {err['affected_fraction']:.3f} should be well below the argmax's "
        f"{err['argmax']:.3f}; if that stopped holding, revisit which is reported"
    )


# ---------------------------------------------------------------------
# Integration through the public entry point
# ---------------------------------------------------------------------

def test_wade_reports_the_two_stages_and_the_characterization():
    rng = np.random.default_rng(31)
    rows, lab = [], []
    for kind, arg, n in (("null", None, 300), ("global", 2.0, 40), ("up", 0.08, 40)):
        for _ in range(n):
            rows.append(_mk(rng, kind, arg)); lab.append(kind)
    x = np.array(rows); lab = np.array(lab)
    # The real library sizes here, not unit ones: this is the end-to-end path,
    # composition included, which is why the direction assertion below is
    # relative to the null genes rather than absolute.
    res = wade.wade(x, np.ones(x.shape[0]), COND, nperms=300, seed=1)

    for name in ("subset_stat", "p_subset", "padj_subset", "affected_fraction", "direction"):
        assert name in res.columns()
    assert res.mean_shift is not None

    assert np.mean(res.p_subset[lab == "global"] <= 0.05) < 0.15
    assert np.median(res.affected_fraction[lab == "global"]) > 0.85
    assert np.mean(res.p_subset[lab == "up"] <= 0.05) > 0.5
    # A global change and a subset are told apart by the characterization, not
    # by a categorical label: no threshold turns these into classes.
    assert np.median(res.affected_fraction[lab == "global"]) > 0.85
    assert np.median(res.affected_fraction[lab == "up"]) < 0.25
    # Compared against the null genes rather than against an absolute value:
    # this matrix contains 40 globally-up genes, which inflate the case
    # libraries and drag every other gene's direction downward (see
    # test_composition_shifts_the_characterization_and_this_is_real). The
    # ordering survives that; an absolute threshold would not.
    assert (np.median(res.direction[lab == "up"])
            > np.median(res.direction[lab == "null"]) + 0.1)


def test_subset_can_be_switched_off_and_is_skipped_without_permutations():
    rng = np.random.default_rng(32)
    x = _nb(rng, MU, (20, N1 + N0))
    assert wade.wade(x, np.ones(20), COND, nperms=0).subset is None
    assert wade.wade(x, np.ones(20), COND, nperms=50, subset=False).subset is None
    assert wade.wade(x, np.ones(20), COND, nperms=50).subset is not None


def test_composition_shifts_the_characterization_and_this_is_real():
    """A warning encoded as a test rather than a comment.

    Library-size normalization couples genes: a matrix where a large fraction of
    genes are strongly up in cases inflates the case libraries, which pushes
    every *other* gene down. That moves direction for null genes well away from
    0.5 and is a property of normalized data, not a defect — but it will mislead
    anyone reading the characterization off a signal-saturated matrix.
    """
    rng = np.random.default_rng(33)

    def null_gene_direction(n_signal):
        rows = [_mk(rng, "null") for _ in range(200)]
        rows += [_mk(rng, "up", 0.5) for _ in range(n_signal)]
        x = np.array(rows)
        r = wade.wade(x, np.ones(x.shape[0]), COND, nperms=0, subset=False)
        st = wade.wade_stats(r.tpm + r.pseudocount, COND)
        return float(np.median(wade.direction(log_ratio_curve(st.Q1, st.Q0))[:200]))

    none, some, saturated = (null_gene_direction(n) for n in (0, 5, 300))

    # With no signal at all the null genes sit where they should.
    assert abs(none) < 0.5, f"no-signal baseline {none:+.3f} should be near 0"
    # Adding signal drags every OTHER gene the opposite way, monotonically, and
    # at large n it takes very little: the systematic offset quickly dominates
    # the sampling noise that would otherwise keep direction near 0.5.
    assert some < none, f"5 signal genes already shift the nulls ({some:+.3f} vs {none:+.3f})"
    assert saturated < some, f"saturation shifts them further ({saturated:+.3f})"
    assert saturated < -0.6
