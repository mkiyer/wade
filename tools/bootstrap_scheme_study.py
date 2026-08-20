"""Bootstrap schemes for the characterization CIs — scaling.md §6's standing
question plus the user's Poisson-resampling proposal, measured.

Schemes, per replicate (all on raw counts, unit libraries, jitter U(0,0.01)
and a one-count pseudocount exactly as wade() applies them):

  resample  within-group column resampling with replacement (what ships)
  poisson   keep every sample, redraw each count as Poisson(count)
  hybrid    resample columns with replacement, then Poisson-redraw the
            drawn counts (de-duplicates the ties resampling creates)
  m-of-n    subsample m = n/2 without replacement, percentile CI rescaled
            by sqrt(m/n) around the point estimate

Calibration target: the TRUE sampling distribution of the estimator at this
design size (NTRUE independent datasets). Metrics per scheme:

  cover   fraction of NSIM datasets whose 95% CI contains the true sampling
          distribution's median (a bias-free target: a CI can only be asked
          to cover what the estimator estimates)
  sdr     median over datasets of (bootstrap SD / true sampling SD) — the
          direct calibration measure; 1.0 is perfect, <1 anti-conservative
  width   median 95% CI width
"""
import numpy as np

from wade.quantiles import probability_grid, type7_quantiles
from wade.subset import affected_fraction, direction, log_ratio_curve

N1 = N0 = 200
PHI = 0.1
NOISE = 0.01
NSIM = 200          # datasets per scenario
NBOOT = 400         # replicates per dataset
NTRUE = 2000        # datasets for the true sampling distribution
rng = np.random.default_rng(0)


def nb(mu, size, phi=PHI):
    if phi == 0.0:
        return rng.poisson(mu, size=size).astype(float)
    return rng.poisson(rng.gamma(1 / phi, phi * mu, size=size)).astype(float)


def draw_counts(scenario, ndata):
    """(ndata, N1+N0) integer counts; each row one dataset's single gene."""
    mu, phi = scenario["mu"], scenario.get("phi", PHI)
    case = nb(mu * scenario.get("fold", 1.0) if scenario.get("kind") == "global" else mu,
              (ndata, N1), phi)
    ctrl = nb(mu, (ndata, N0), phi)
    if scenario.get("kind") == "subset":
        k = round(scenario["frac"] * N1)
        hot = nb(mu * scenario["fold"], (ndata, k), phi)
        case[:, :k] = hot                      # position is irrelevant to quantiles
    return np.c_[case, ctrl]


def estimate(case_counts, ctrl_counts, q=None):
    """aff, dir, lfc rows from count blocks, with jitter and the pseudocount."""
    n1 = case_counts.shape[1]
    x1 = case_counts + rng.uniform(0, NOISE, case_counts.shape)
    x0 = ctrl_counts + rng.uniform(0, NOISE, ctrl_counts.shape)
    if q is None:
        q = probability_grid(min(n1, x0.shape[1]))
    r = log_ratio_curve(type7_quantiles(x1 + 1.0, q), type7_quantiles(x0 + 1.0, q))
    with np.errstate(divide="ignore", invalid="ignore"):
        lfc = np.log2(x1.mean(axis=1) / x0.mean(axis=1))
    return affected_fraction(r), direction(r), lfc


SCENARIOS = [
    dict(name="null nb(50)", kind="null", mu=50.0),
    dict(name="global 2x nb(50)", kind="global", mu=50.0, fold=2.0),
    dict(name="subset 5% 8x nb(50)", kind="subset", mu=50.0, fold=8.0, frac=0.05),
    dict(name="global 2x nb(5)", kind="global", mu=5.0, fold=2.0),
    dict(name="global 2x pois(50)", kind="global", mu=50.0, fold=2.0, phi=0.0),
]

print(f"{'scenario':<22}{'estimand':<7}{'truth':>7} | " +
      " | ".join(f"{s:^24}" for s in ("resample", "poisson", "hybrid", "m-of-n")))
print(f"{'':<22}{'':<7}{'':>7} | " +
      " | ".join(f"{'cov':>5} {'sdr':>5} {'width':>7}" for _ in range(4)))

for sc in SCENARIOS:
    # The true sampling distribution at this design size.
    tc = draw_counts(sc, NTRUE)
    t_aff, t_dir, t_lfc = estimate(tc[:, :N1], tc[:, N1:])
    truth = dict(aff=np.median(t_aff), dir=np.median(t_dir), lfc=np.median(t_lfc))
    true_sd = dict(aff=t_aff.std(), dir=t_dir.std(), lfc=t_lfc.std())

    data = draw_counts(sc, NSIM)
    stats = {k: {s: np.empty((NSIM, 3)) for s in ("resample", "poisson", "hybrid", "m-of-n")}
             for k in ("aff", "dir", "lfc")}   # per dataset: lo, hi, boot-sd

    for i in range(NSIM):
        c1, c0 = data[i, :N1], data[i, N1:]
        # the dataset's own jittered values, fixed across resample replicates
        x1 = c1 + rng.uniform(0, NOISE, N1)
        x0 = c0 + rng.uniform(0, NOISE, N0)
        point = estimate(c1[None, :], c0[None, :])   # for m-of-n rescaling

        for scheme in ("resample", "poisson", "hybrid", "m-of-n"):
            if scheme == "resample":
                j1 = rng.integers(0, N1, (NBOOT, N1))
                j0 = rng.integers(0, N0, (NBOOT, N0))
                b1, b0 = x1[j1], x0[j0]
                q = probability_grid(min(N1, N0))
                r = log_ratio_curve(type7_quantiles(b1 + 1.0, q), type7_quantiles(b0 + 1.0, q))
                with np.errstate(divide="ignore", invalid="ignore"):
                    vals = (affected_fraction(r), direction(r),
                            np.log2(b1.mean(axis=1) / b0.mean(axis=1)))
            elif scheme == "poisson":
                b1 = rng.poisson(np.broadcast_to(c1, (NBOOT, N1))).astype(float)
                b0 = rng.poisson(np.broadcast_to(c0, (NBOOT, N0))).astype(float)
                vals = estimate(b1, b0)
            elif scheme == "hybrid":
                j1 = rng.integers(0, N1, (NBOOT, N1))
                j0 = rng.integers(0, N0, (NBOOT, N0))
                b1 = rng.poisson(c1[j1]).astype(float)
                b0 = rng.poisson(c0[j0]).astype(float)
                vals = estimate(b1, b0)
            else:                                    # m-of-n
                m1, m0 = N1 // 2, N0 // 2
                j1 = np.argsort(rng.random((NBOOT, N1)), axis=1)[:, :m1]
                j0 = np.argsort(rng.random((NBOOT, N0)), axis=1)[:, :m0]
                vals = estimate(np.take_along_axis(np.broadcast_to(c1, (NBOOT, N1)), j1, 1),
                                np.take_along_axis(np.broadcast_to(c0, (NBOOT, N0)), j0, 1))

            for k, v, pt in zip(("aff", "dir", "lfc"), vals, point):
                lo, hi = np.nanpercentile(v, [2.5, 97.5])
                if scheme == "m-of-n":               # rescale around the point estimate
                    scale = np.sqrt((N1 // 2) / N1)
                    lo = pt[0] + scale * (lo - pt[0])
                    hi = pt[0] + scale * (hi - pt[0])
                stats[k][scheme][i] = (lo, hi, np.nanstd(v))

    for k in ("aff", "dir", "lfc"):
        cells = []
        for scheme in ("resample", "poisson", "hybrid", "m-of-n"):
            s = stats[k][scheme]
            cover = float(np.mean((s[:, 0] <= truth[k]) & (truth[k] <= s[:, 1])))
            sdr = float(np.median(s[:, 2]) / true_sd[k])
            width = float(np.median(s[:, 1] - s[:, 0]))
            cells.append(f"{cover:5.2f} {sdr:5.2f} {width:7.3f}")
        print(f"{sc['name']:<22}{k:<7}{truth[k]:>7.3f} | " + " | ".join(cells))
    print()
