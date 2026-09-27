"""Measurement diagnostics for the existing binomial–Beta candidates.

These condition on known exposures. They do not fit normalization, a prior,
technical dispersion, a patient population, or a gene-level null.
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import betainc, betaln
from scipy.stats import binom, betabinom

from .assessment import _beta_quantiles


@dataclass
class BinomialBeta:
    counts: np.ndarray
    totals: np.ndarray
    size_factors: np.ndarray
    prior_shape: float

    def __post_init__(self):
        c, n, s = np.broadcast_arrays(np.asarray(self.counts, float),
                                     np.asarray(self.totals, float),
                                     np.asarray(self.size_factors, float))
        if (self.prior_shape not in (1., .5) or isinstance(self.prior_shape, bool)
                or not all(np.isfinite(v).all() for v in (c, n, s))
                or np.any((c < 0) | (c > n) | (n <= 0) | (n > 2**53) | (s <= 0))
                or np.any(c != np.floor(c)) or np.any(n != np.floor(n))):
            raise ValueError('Require integer 0 <= C <= N, positive exposure, and explicit h=1 or h=0.5')
        self.counts, self.totals, self.size_factors = c, n, s
        self.a, self.b = c+self.prior_shape, n-c+self.prior_shape
        self.scale = n/s
        self.mean = self.scale*self.a/(self.a+self.b)
        self.variance = self.scale**2*self.a*self.b/((self.a+self.b)**2*(self.a+self.b+1))

    def cdf(self, abundance):
        return betainc(self.a, self.b, np.clip(np.asarray(abundance)/self.scale, 0, 1))

    def quantiles(self, probabilities):
        q, repairs, residual = _beta_quantiles(self.a[..., None], self.b[..., None], probabilities)
        return self.scale[..., None]*q, repairs, residual

    def predictive_logpmf(self, future_counts, future_total):
        """New binomial sampling from the SAME underlying fraction, integrated over p."""
        return betabinom.logpmf(future_counts, future_total, self.a, self.b)

    def predictive_zero(self, future_total):
        return np.exp(betaln(self.a, self.b+future_total)-betaln(self.a, self.b))


def binomial_support(total, fraction, tail_probability=1e-11):
    """A finite summation range with explicit omitted probability, not simulation."""
    lo, hi = binom.ppf([tail_probability, 1-tail_probability], total, fraction).astype(int)
    counts = np.arange(lo, hi+1)
    weights = binom.pmf(counts, total, fraction)
    omitted = float(binom.cdf(lo-1, total, fraction)+binom.sf(hi, total, fraction))
    if omitted > 2*tail_probability+1e-14 or abs(weights.sum()+omitted-1) > 5e-8:
        raise ArithmeticError('Binomial support failed probability accounting')
    return counts, weights, omitted


def beta_marginal_mode(a, b):
    """Mode in fraction space; NaN denotes nonunique modes, never an arbitrary tie."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    if not np.isfinite(a+b).all() or np.any((a <= 0) | (b <= 0)):
        raise ValueError('Beta shapes must be finite and positive')
    mode = np.full(a.shape, np.nan)
    interior = (a > 1) & (b > 1)
    np.divide(a-1, a+b-2, out=mode, where=interior)
    mode[(a <= 1) & (b > 1)] = 0
    mode[(a > 1) & (b <= 1)] = 1
    # Beta(a,1), a<1 and Beta(1,b), b<1 have unique endpoint modes too.
    mode[(a < 1) & (b == 1)] = 0
    mode[(a == 1) & (b < 1)] = 1
    return mode


def uniform_dirichlet_summary(counts):
    """Constraint audit of Dirichlet(C+1), NOT a selected measurement kernel.

    The joint mode preserves C/N. Its one-gene marginal modes generally do not.
    Inputs are observations, never overwritten with inferred means or shapes.
    """
    c = np.asarray(counts, float)
    if (c.ndim != 1 or len(c) < 2 or not np.isfinite(c).all()
            or np.any(c < 0) or np.any(c != np.floor(c)) or c.sum() <= 0
            or c.sum() > 2**53):
        raise ValueError('Require at least two complete integer count categories and positive total')
    n, g = c.sum(), len(c)
    a, b = c+1, n-c+g-1
    return dict(a=a, b=b, mean=a/(n+g), variance=a*b/((n+g)**2*(n+g+1)),
                joint_mode=c/n, marginal_mode=beta_marginal_mode(a, b))
