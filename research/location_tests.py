"""Which location test can WADE's differential-expression test calibrate? (Review 3, point 3.)

The mean test's null is equal arithmetic means. Its unstudentized permutation test is exact when
the groups share a distribution, but with unequal spreads in unbalanced designs it rejects equal
means far too often, and Welch studentization only partly repairs that (docs/subset-null-plan.md,
section 2.7). No test of arithmetic means is uniformly valid over distributions with arbitrarily
large rare tails (Bahadur and Savage, 1956), so the choice is between declaring a regularity class
for the mean and testing a robust location instead. This compares four permutation tests, each
under its own null:

  mean      the difference in means, unstudentized (the present mean test)
  Welch     the difference in means over its unpooled standard error
  Wilcoxon  the rank sum, unstudentized: exact when the groups share a distribution
  BM        Brunner and Munzel's studentized rank statistic for the relative effect
            P(case > control) + P(tie)/2 = 1/2 (Neubert and Brunner, 2007)

Scenarios, lognormal expression (continuous, so no count model is involved):

  equal means     both groups' means 50, the cases' CV^2 against the controls'
  equal medians   both groups' medians 50 (so the relative effect is exactly 1/2), CVs as above
  identical       both CV^2 1: every test's null
  subset          identical CV^2 0.5, except a tenth of the cases at 8x: the subset test's target
  shift           every case 1.5x, CV^2 0.5 both: power

    PYTHONPATH=src:research python research/location_tests.py
"""
import sys

import numpy as np

TESTS = ("mean", "Welch", "Wilcoxon", "BM")


def draw(rng, genes, n1, n0, scenario, cv_case=2.0, cv_control=0.25):
    s_case, s_control = np.sqrt(np.log1p(cv_case)), np.sqrt(np.log1p(cv_control))
    z = rng.normal(size=(genes, n1 + n0))
    x = np.empty_like(z)
    if scenario == "equal means":
        x[:, :n1] = 50 * np.exp(s_case * z[:, :n1] - s_case ** 2 / 2)
        x[:, n1:] = 50 * np.exp(s_control * z[:, n1:] - s_control ** 2 / 2)
    elif scenario == "equal medians":
        x[:, :n1] = 50 * np.exp(s_case * z[:, :n1])
        x[:, n1:] = 50 * np.exp(s_control * z[:, n1:])
    elif scenario == "identical":
        x = 50 * np.exp(np.sqrt(np.log(2)) * z)
    elif scenario == "subset":
        x = 50 * np.exp(np.sqrt(np.log(1.5)) * z)
        k = max(1, round(0.1 * n1))
        x[:, :k] *= 8
    elif scenario == "shift":
        x = 50 * np.exp(np.sqrt(np.log(1.5)) * z)
        x[:, :n1] *= 1.5
    return x


def _ranks(a):
    """Row-wise ranks, 1-based (continuous values: no ties)."""
    r = np.empty_like(a)
    order = np.argsort(a, axis=1)
    np.put_along_axis(r, order, np.arange(1, a.shape[1] + 1, dtype=float)[None].repeat(len(a), 0), axis=1)
    return r


def statistics(x, pooled_ranks, labels):
    """Each test's statistic for one labelling (positive: cases higher)."""
    cases, controls = labels == 1, labels == 0
    n1, n0 = int(cases.sum()), int(controls.sum())
    a, b = x[:, cases], x[:, controls]
    diff = a.mean(1) - b.mean(1)
    welch = diff / np.sqrt(a.var(1, ddof=1) / n1 + b.var(1, ddof=1) / n0)
    ra, rb = pooled_ranks[:, cases], pooled_ranks[:, controls]
    ranksum = ra.sum(1)
    within_a, within_b = _ranks(ra), _ranks(rb)
    sa = ((ra - within_a - ra.mean(1, keepdims=True) + within_a.mean(1, keepdims=True)) ** 2).sum(1) / (n1 - 1)
    sb = ((rb - within_b - rb.mean(1, keepdims=True) + within_b.mean(1, keepdims=True)) ** 2).sum(1) / (n0 - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        bm = n1 * n0 * (ra.mean(1) - rb.mean(1)) / ((n1 + n0) * np.sqrt(n1 * sa + n0 * sb))
    return np.stack([diff, welch, ranksum - n1 * (n1 + n0 + 1) / 2, bm])


def pvalues(x, n1, rng, replicates):
    """Two-sided permutation p-values for every test, all from the same shuffles."""
    n = x.shape[1]
    labels = np.r_[np.ones(n1, int), np.zeros(n - n1, int)]
    ranks = _ranks(x)
    observed = np.abs(statistics(x, ranks, labels))
    exceed = np.zeros_like(observed)
    for _ in range(replicates):
        star = np.abs(statistics(x, ranks, rng.permutation(labels)))
        exceed += star >= observed * (1 - 1e-12)
    return (exceed + 1) / (replicates + 1)


def calibration(genes=2000, replicates=999, seed=1):
    print(f"\n== calibration: {genes:,} genes per cell, {replicates:,} shuffles; two-sided share at p <= 0.05 / 0.01 "
          f"for {' | '.join(TESTS)}")
    rng = np.random.default_rng(seed)
    cells = [("identical", None), ("equal means", (2.0, 0.25)), ("equal means", (0.25, 2.0)),
             ("equal medians", (2.0, 0.25)), ("equal medians", (0.25, 2.0)), ("subset", None), ("shift", None)]
    for n1, n0 in ((50, 500), (500, 50), (250, 250)):
        for scenario, cvs in cells:
            x = draw(rng, genes, n1, n0, scenario, *(cvs or (2.0, 0.25)))
            p = pvalues(x, n1, rng, replicates)
            label = scenario if cvs is None else f"{scenario} (case CV^2 {cvs[0]:g}, control {cvs[1]:g})"
            print(f"{n1:3d}:{n0:<3d} {label:42s} " + " | ".join(
                f"{t} {np.mean(p[k] <= .05):.1%} / {np.mean(p[k] <= .01):.1%}" for k, t in enumerate(TESTS)), flush=True)


SECTIONS = dict(calibration=calibration)

if __name__ == "__main__":
    for section in sys.argv[1:] or list(SECTIONS):
        SECTIONS[section]()
