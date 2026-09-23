"""Research instruments for observable effects, localization and latent truth.

Every patient has unit mass. These are descriptive diagnostics, not a new
inferential procedure or a fitted mixture/component estimator.
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import betainc, betaincinv


@dataclass
class PatientDistribution:
    nodes: np.ndarray                 # patients x equal-probability nodes
    means: np.ndarray                 # exact means (not quadrature estimates)
    w1_error: np.ndarray              # transport bound including numerical allowance
    intervals: np.ndarray | None = None
    quantile_repairs: int = 0
    max_cdf_residual: float = 0.
    roundoff_bound: np.ndarray | None = None

    def __post_init__(self):
        self.nodes = np.asarray(self.nodes, float)
        self.means = np.asarray(self.means, float)
        self.w1_error = np.asarray(self.w1_error, float)
        n = len(self.means)
        if (self.nodes.ndim != 2 or self.nodes.shape[0] != n or not self.nodes.shape[1]
                or self.means.shape != (n,) or self.w1_error.shape != (n,)
                or not all(np.isfinite(x).all() and np.all(x >= 0)
                           for x in (self.nodes, self.means, self.w1_error))):
            raise ValueError("Require finite nonnegative patient nodes, means and errors")
        if self.intervals is not None:
            self.intervals = np.asarray(self.intervals, float)
            if (self.intervals.shape != (n, 2) or not np.isfinite(self.intervals).all()
                    or np.any(self.intervals[:, 0] < 0)
                    or np.any(self.intervals[:, 1] < self.intervals[:, 0])):
                raise ValueError("Require ordered nonnegative patient intervals")


def points(values):
    x = np.asarray(values, float)
    if x.ndim != 1 or not len(x):
        raise ValueError("Require a nonempty patient vector")
    return PatientDistribution(x[:, None], x, np.zeros_like(x))


def _beta_quantiles(a, b, probabilities):
    """Verify inverse-CDF requests; bisect the CDF if a library result fails.

    Large, asymmetric shapes exposed inaccurate inverse-beta values in the
    development environment. Check the defining identity rather than accepting
    finite but wrong quantiles or clipping a negative transport cost.
    """
    a, b, u = np.broadcast_arrays(a, b, probabilities)
    x = betaincinv(a, b, u)
    residual = abs(betainc(a, b, x)-u)
    # At N around 1e8 the CDF itself can step by a few 1e-9 between
    # adjacent floats. 1e-7 is still far below the 1/128 node mass.
    bad = ~np.isfinite(x) | ~np.isfinite(residual) | (residual > 1e-7)
    repairs = int(bad.sum())
    if repairs:
        lo, hi = np.zeros(repairs), np.ones(repairs)
        for _ in range(100):
            mid = (lo+hi)/2
            lower = betainc(a[bad], b[bad], mid) < u[bad]
            lo, hi = np.where(lower, mid, lo), np.where(lower, hi, mid)
        x[bad] = (lo+hi)/2
        residual = abs(betainc(a, b, x)-u)
    if not np.isfinite(residual).all() or residual.max() > 1e-7:
        raise ArithmeticError("Beta quantile inversion failed its CDF check")
    return x, repairs, float(residual.max())


def beta_distribution(engine, nodes):
    """Midpoint quantile representation with an analytic W1 error bound.

    For bin [l,r] and midpoint m, integral |Q(u)-Q(m)| du equals
    M(r)-2M(m)+M(l), where M(u)=integral_0^u Q(v)dv. Averaging patient
    costs bounds the mixture's W1 error. The two group costs bound area error.
    No random draws or rank jitter enter these descriptions.
    """
    if not isinstance(nodes, int) or nodes < 2:
        raise ValueError("At least two quadrature nodes are required")
    u = np.arange(2 * nodes + 1) / (2 * nodes)
    a, b = engine.a[:, None], engine.b[:, None]
    fraction, repairs, residual = _beta_quantiles(a, b, u[None, :])
    moment = engine.mean[:, None] * betainc(a + 1, b, fraction)
    cost = (moment[:, 2::2] - 2 * moment[:, 1::2] + moment[:, :-2:2]).sum(axis=1)
    if np.any(cost < -1e-10 * np.maximum(1, engine.mean)):
        raise ArithmeticError("Negative quantile approximation cost")
    ids = engine.patient_state
    limits, more_repairs, limit_residual = _beta_quantiles(a, b, [.025, .975])
    intervals = engine.scale[:, None] * limits
    # Interior probability errors perturb each truncated first moment by at
    # most ceiling * probability_error, provided the ceiling brackets them.
    # Include an outward allowance for the incomplete-beta moment evaluation.
    interior = u[1:-1].max() + residual
    if np.any(betainc(engine.a, engine.b, np.minimum(1, engine.ceiling/engine.scale)) < interior):
        raise ArithmeticError("Kernel ceiling does not bracket quadrature nodes")
    padding = 4*nodes*(engine.ceiling*residual + engine.mean*1e-7)
    return PatientDistribution(
        (engine.scale[:, None] * fraction[:, 1::2])[ids],
        engine.mean[ids], (np.maximum(cost, 0)+padding)[ids], intervals[ids],
        repairs+more_repairs, max(residual, limit_residual), padding[ids],
    )


def paired_quantiles(case, control):
    """Exact step-quantile alignment for two equal-weight atomic distributions."""
    a, b = np.sort(np.asarray(case, float).ravel()), np.sort(np.asarray(control, float).ravel())
    if not len(a) or not len(b) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Two nonempty finite distributions are required")
    ea, eb = np.arange(1, len(a)+1)/len(a), np.arange(1, len(b)+1)/len(b)
    edges = np.unique(np.r_[0., ea, eb])
    mid = (edges[1:] + edges[:-1])/2
    return edges, a[np.searchsorted(ea, mid)], b[np.searchsorted(eb, mid)]


def log2_ratio(numerator, denominator):
    """Undefined zero contrasts remain missing; no hidden pseudocount."""
    if numerator <= 0 or denominator <= 0:
        return float('nan')
    return float(np.log2(numerator / denominator))


def describe(distribution, labels):
    labels = np.asarray(labels)
    if (labels.shape != distribution.means.shape or not np.isin(labels, [0, 1]).all()
            or labels.sum() in (0, len(labels))):
        raise ValueError("Require aligned binary labels and both groups")
    case = labels.astype(bool)
    edges, qa, qb = paired_quantiles(distribution.nodes[case], distribution.nodes[~case])
    width = np.diff(edges)
    increments = (qa - qb) * width
    walk = np.r_[np.cumsum(increments[::-1])[::-1], 0.]
    tolerance = 1e-12 * max(float(np.max(abs(qa))), float(np.max(abs(qb))), 1e-300)
    # The highest cutoff wins a plateau. Identical groups therefore select no tail.
    index = np.flatnonzero(walk >= walk.max() - tolerance)[-1]
    fraction = float(1 - edges[index])
    selected = np.zeros(case.sum())
    tail_mean = float('nan')
    if fraction > 0:
        nodes = distribution.nodes[case]
        sorted_nodes = np.sort(nodes.ravel())
        mass = fraction * sorted_nodes.size
        cutoff = sorted_nodes[int(np.clip(np.floor(sorted_nodes.size - mass), 0, sorted_nodes.size-1))]
        greater, tied = nodes > cutoff, nodes == cutoff
        tie_weight = np.clip((mass - greater.sum()) / tied.sum(), 0, 1)
        selected = (greater + tied * tie_weight).mean(axis=1)
        tail_mean = float(np.sum(qa[index:] * width[index:]) / fraction)
    summary = dict(
        area=float(np.maximum(increments, 0).sum()),
        down_area=float(np.maximum(-increments, 0).sum()),
        mean_log2=log2_ratio(distribution.means[case].mean(), distribution.means[~case].mean()),
        tail_fraction=fraction,
        tail_vs_control_log2=log2_ratio(tail_mean, distribution.means[~case].mean()),
        area_numerical_bound=float(distribution.w1_error[case].mean()+distribution.w1_error[~case].mean()),
    )
    return summary, selected


def average_precision(truth, scores):
    """Tie-aware step PR integral (average precision); no positives => undefined."""
    truth, scores = np.asarray(truth, bool), np.asarray(scores, float)
    if truth.shape != scores.shape or truth.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Require aligned truth and finite patient scores")
    if not truth.sum():
        return float('nan')
    order = np.argsort(-scores, kind='stable')
    y, s = truth[order], scores[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s)-1]
    hits = np.cumsum(y)[ends]
    return float(np.sum(np.diff(np.r_[0, hits]) / truth.sum() * hits / (ends+1)))


def membership_metrics(truth, selected, scores):
    truth, selected = np.asarray(truth, bool), np.asarray(selected, float)
    if truth.shape != selected.shape or np.any((selected < 0) | (selected > 1)):
        raise ValueError("Require aligned membership probabilities in [0,1]")
    hits = selected[truth].sum()
    return dict(
        selected_fraction=float(selected.mean()),
        fraction_error=float(selected.mean()-truth.mean()),
        soft_precision=float(hits/selected.sum()) if selected.sum() else float('nan'),
        soft_recall=float(hits/truth.sum()) if truth.sum() else float('nan'),
        membership_brier=float(np.mean((selected-truth)**2)),
        average_precision=average_precision(truth, scores),
    )


def group_w1(distribution, truth, labels):
    """Mean of case and control W1 errors against finite-cohort latent values."""
    errors = []
    for group in (0, 1):
        mask = np.asarray(labels) == group
        edges, estimate, target = paired_quantiles(distribution.nodes[mask], np.asarray(truth)[mask])
        errors.append(float(np.sum(np.diff(edges)*abs(estimate-target))))
    return float(np.mean(errors))
