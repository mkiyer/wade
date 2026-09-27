"""Mode-preserving constructions for a bounded measurement audit, not WADE APIs.

The discrete object is a smoothed multinomial bootstrap. The continuous object
is a uniform contraction of a Dirichlet posterior toward the observations.
Neither is automatically a calibrated distribution for latent abundance.
All calculations here use physical fraction/count units; normalization is a
pushforward, never replacement of the input observations.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import betainc


def complete_counts(counts):
    """Validate an exhaustive, disjoint count universe without editing it."""
    c = np.asarray(counts, dtype=float)
    if (c.ndim != 1 or len(c) < 2 or not np.isfinite(c).all()
            or np.any(c < 0) or np.any(c != np.floor(c))
            or not 0 < c.sum() <= 2**53):
        raise ValueError('Require a complete integer count vector, G>=2 and 0<N<=2**53')
    return c, float(c.sum())


def composition_weights(weights, genes):
    w = np.asarray(weights, dtype=float)
    if (w.shape != (genes,) or not np.isfinite(w).all()
            or np.any((w <= 0) | (w >= 1))
            or not np.isclose(w.sum(), 1., rtol=0, atol=1e-13)):
        raise ValueError('Require explicit positive weights summing to one')
    return w


def smoothed_binomial(counts, total, weight):
    """Marginal parameters; a joint model additionally requires sum(weight)=1.

For q=(C+w)/(N+1), the PMF ratio proves the unique Binomial(N,q) mode
is C when 0<w<1. The mean is N*q, not C, and data are not modified.
"""
    c, n, w = np.broadcast_arrays(np.asarray(counts, float),
                                 np.asarray(total, float), np.asarray(weight, float))
    if (not all(np.isfinite(v).all() for v in (c, n, w))
            or np.any((c < 0) | (c > n) | (n <= 0) | (n > 2**53))
            or np.any(c != np.floor(c)) or np.any(n != np.floor(n))
            or np.any((w <= 0) | (w >= 1))):
        raise ValueError('Require integer 0<=C<=N and 0<w<1')
    q = (c+w)/(n+1)
    if np.any((q <= 0) | (q >= 1)):
        raise ArithmeticError('Insufficient floating-point precision for the specified weight/depth')
    return dict(q=q, mean=n*q, variance=n*q*(1-q))


def multinomial_parameters(counts, weights):
    c, n = complete_counts(counts)
    w = composition_weights(weights, len(c))
    result = smoothed_binomial(c, n, w)
    result['total'] = int(n)
    return result


def contraction_moments(mode, a, b):
    """P=m+U*(Q-m), Q~Beta(a,b), independent U~Uniform(0,1)."""
    m, a, b = np.broadcast_arrays(np.asarray(mode, float),
                                 np.asarray(a, float), np.asarray(b, float))
    if (not all(np.isfinite(v).all() for v in (m, a, b))
            or np.any((m < 0) | (m > 1) | (a <= 0) | (b <= 0))):
        raise ValueError('Require a fraction mode and positive finite Beta shapes')
    mean_q = a/(a+b)
    var_q = mean_q*(1-mean_q)/(a+b+1)
    return dict(mean=(m+mean_q)/2, variance=var_q/3+(mean_q-m)**2/12,
                base_mean=mean_q, base_variance=var_q)


def contracted_dirichlet_parameters(counts, prior):
    c, n = complete_counts(counts)
    alpha = np.asarray(prior, float)
    if (alpha.shape != c.shape or not np.isfinite(alpha).all()
            or np.any(alpha <= 0) or not np.isfinite(alpha.sum())):
        raise ValueError('Require an explicit positive prior for every category')
    a = c+alpha
    b = (n-c)+(alpha.sum()-alpha)
    result = contraction_moments(c/n, a, b)
    return dict(**result, a=a, b=b, observed_fraction=c/n)


def contracted_beta_cdf(fraction, mode, a, b, *, tolerance=1e-9):
    """Integrate over U using a bounded integrand, not a singular density.

    Returns the CDF and the adaptive quadrature's absolute error estimate.
    This numerical error estimate is not a statistical confidence bound.
    At x=m the integral equals the base Beta CDF exactly. Tiny zero-count
    shapes can have unrepresentably small quantiles; evaluate probabilities
    directly instead of clamping those quantiles and claiming zero variance.
    """
    x, m, a, b = np.broadcast_arrays(np.asarray(fraction, float),
                                    np.asarray(mode, float), np.asarray(a, float),
                                    np.asarray(b, float))
    contraction_moments(m, a, b)  # Shared parameter validation.
    if np.isnan(x).any() or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Require non-NaN fractions and a positive numerical tolerance')
    result = np.zeros(x.shape)
    result[x >= 1] = 1
    equal = (x == m) & (x > 0) & (x < 1)
    result[equal] = betainc(a[equal], b[equal], m[equal])
    active = (x > 0) & (x < 1) & ~equal
    if not np.any(active):
        return result, 0.
    center, delta = m[active], x[active]-m[active]
    aa, bb = a[active], b[active]

    def integrand(u):
        return betainc(aa, bb, np.clip(center+delta/u, 0, 1))

    values, error, info = quad_vec(integrand, 0., 1., epsabs=tolerance,
                                   epsrel=0., norm='max', full_output=True)
    if not info.success or not np.isfinite(values).all() or error > tolerance:
        raise ArithmeticError(f'Contraction CDF integration failed: {info.message}; error={error}')
    result[active] = np.clip(values, 0, 1)
    return result, float(error)


def draw_contracted_dirichlet(counts, prior, draws, rng):
    """Joint witness: ONE shared U per whole-library draw conserves the sum.

    Very small Dirichlet shapes can underflow in generic random samplers.
    This routine is for small audit examples, not transcriptome simulation.
    """
    c, n = complete_counts(counts)
    params = contracted_dirichlet_parameters(c, prior)
    q = rng.dirichlet(params['a'], size=draws)
    u = rng.uniform(size=(draws, 1))
    return (1-u)*(c/n)+u*q
