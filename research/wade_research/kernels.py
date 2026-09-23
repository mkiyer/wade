"""Research-only normalized Beta mixtures and bounded directional integration.

Extracted from the audited September 2026 adapter/reference prototypes.
This module is not imported or distributed by production WADE.
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import betainc, betaincc
from scipy.stats import beta

from ._cache import KernelNodeCache


@dataclass
class AreaBounds:
    lower: np.ndarray
    upper: np.ndarray
    signed: np.ndarray
    nodes: int
    refinements: int
    peak_array_bytes: int

    @property
    def area(self):
        return (self.lower + self.upper) / 2

    def permutation_p_bounds(self, coefficients, tie_tolerance=0.):
        """Observed row first. Count inclusive ties without optimistic rounding.

        This is a numerical interval for a fixed-budget randomization p-value,
        conditional on a valid permutation design, not a confidence interval.
        """
        same = np.all(coefficients[1:] == coefficients[0], axis=1)
        definite = self.lower[1:] >= self.upper[0] - tie_tolerance
        possible = self.upper[1:] >= self.lower[0] - tie_tolerance
        definite[same] = possible[same] = True
        return ((1 + definite.sum()) / len(self.lower),
                (1 + possible.sum()) / len(self.lower))


class BetaKernelArea:
    """Research binomial–Beta kernels on the counts / size_factor axis.

    Both size_factors and prior_shape are explicit: there is no implicit CPM
    scaling or selected production prior. Only the two previously studied
    shapes (1 and 0.5) are supported. Each patient retains one unit of mass.
    """

    def __init__(self, counts, totals, *, size_factors, prior_shape,
                 memory_mb=192, node_cache_mb=0):
        c, n = np.asarray(counts), np.asarray(totals)
        if (c.ndim != 1 or n.shape != c.shape or not len(c)
                or not np.isfinite(c).all() or not np.isfinite(n).all()
                or np.any(c != np.floor(c)) or np.any(n != np.floor(n))
                or np.any(c < 0) or np.any(n <= 0) or np.any(c > n)
                or np.any(n > 2**53)):
            raise ValueError("Require aligned integer 0 <= C <= N, 0 < N <= 2**53")
        sf = np.asarray(size_factors, float)
        if sf.shape != c.shape or not np.isfinite(sf).all() or np.any(sf <= 0):
            raise ValueError("Require a finite positive size factor for every patient")
        if isinstance(prior_shape, (bool, np.bool_)) or prior_shape not in (1., .5):
            raise ValueError("Use an explicit studied prior shape: 1 or 0.5")
        if not np.isfinite(memory_mb) or memory_mb <= 0:
            raise ValueError("memory_mb must be finite and positive")
        if not np.isfinite(node_cache_mb) or node_cache_mb < 0:
            raise ValueError("node_cache_mb must be finite and nonnegative")
        scale = n / sf
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Normalized abundance scale must be finite and positive")
        states, self.patient_state = np.unique(np.c_[n, c, scale], axis=0,
                                               return_inverse=True)
        self.npatients = len(c)
        self.prior_shape = float(prior_shape)
        self.a = states[:, 1] + prior_shape
        self.b = states[:, 0] - states[:, 1] + prior_shape
        self.scale = states[:, 2]
        self.mean = self.scale * self.a / (self.a + self.b)
        self.unit = float(np.median(scale) / 1e6)
        self.memory_bytes = int(memory_mb * 1024**2)
        geometry = np.vstack([
            beta.ppf(np.array([.01, .5, .99])[:, None], self.a, self.b),
            beta.isf(1e-12, self.a, self.b),
        ])
        anchors = geometry[:3] * self.scale
        self.ceiling = float(np.max(geometry[3] * self.scale))
        self.initial_nodes = np.unique(np.r_[0., anchors.ravel(), self.ceiling])
        self.evaluated_patient_nodes = 0
        self.tail_call = self.evaluate(np.array([self.ceiling]))[2][:, 0]
        self.node_cache = (KernelNodeCache(self.initial_nodes, len(self.a),
                                         int(node_cache_mb * 1024**2)) if node_cache_mb else None)
        for array in [self.a, self.b, self.scale, self.mean, self.patient_state,
                      self.initial_nodes, self.tail_call]:
            array.flags.writeable = False

    def compress(self, coefficients):
        w = np.atleast_2d(np.asarray(coefficients, float))
        if (w.shape[1] != self.npatients or not np.isfinite(w).all()
                or np.any(abs(w.sum(axis=1)) > 2e-12)
                or np.any(abs(w).sum(axis=1) > 2 + 2e-12)):
            raise ValueError("Need aligned finite group-probability differences")
        result = np.array([np.bincount(self.patient_state, weights=row,
                                      minlength=len(self.a)) for row in w])
        # Exactly identical patient kernels cancel before numerical integration.
        result[abs(result) < 4 * np.finfo(float).eps] = 0.
        return result

    def evaluate(self, x):
        """CDF, E[(x-X)+] and E[(X-x)+], using incomplete Beta moments."""
        x = np.asarray(x, float)
        self.evaluated_patient_nodes += len(self.a) * len(x)
        z = np.clip(x[None, :] / self.scale[:, None], 0, 1)
        a, b = self.a[:, None], self.b[:, None]
        mean = self.mean[:, None]
        f = betainc(a, b, z)
        put = x * f - mean * betainc(a + 1, b, z)
        call = mean * betaincc(a + 1, b, z) - x * betaincc(a, b, z)
        # Use put-call parity on the upper side to avoid subtracting CDFs
        # close to one. On the lower side put is the smaller-tail expression.
        put = np.where(x >= mean, x - mean + call, put)
        rounding = 64 * np.finfo(float).eps * np.maximum(x, mean)
        if np.any(put < -rounding) or np.any(call < -rounding):
            raise ArithmeticError("Integrated Beta CDF has a negative tail moment")
        return f, np.maximum(put, 0), np.maximum(call, 0)

    def _group_nodes(self, x, plus, minus):
        """Evaluate in bounded patient-by-node tiles; retain group curves only."""
        groups = len(plus)
        output = np.empty((3, groups, len(x)))
        block = max(1, min(128, self.memory_bytes // max(1, 128 * len(self.a))))
        for start in range(0, len(x), block):
            end = min(start + block, len(x))
            if self.node_cache is None:
                f, put, _ = self.evaluate(x[start:end])
            else:
                f, put = self.node_cache.get(x[start:end], self.evaluate)
            output[0, :, start:end] = plus @ f
            output[1, :, start:end] = minus @ f
            # Center the antiderivative to reduce common large terms. The
            # original coefficients sum to zero, so this preserves the target.
            put -= put[:1]
            output[2, :, start:end] = (plus - minus) @ put
        return output

    @staticmethod
    def _intervals(x, curves):
        width = np.diff(x)
        positive, negative, primitive = curves
        lo = positive[:, :-1] - negative[:, 1:]
        hi = positive[:, 1:] - negative[:, :-1]
        signed = np.diff(primitive, axis=1)
        lower = np.maximum(signed, 0.)
        upper = lower.copy()
        crossing = (lo < 0) & (hi > 0)
        # On [lo,hi], the chord of max(d,0) bounds it from above.
        upper[crossing] = (hi[crossing]
            * (signed - lo * width)[crossing] / (hi - lo)[crossing])
        upper = np.maximum(upper, lower)
        return lower, upper

    def scores(self, coefficients, *, atol=None, rtol=2e-4, max_refinements=24):
        """Bracket each linear upward area; numerical bounds are not error bars.

        Intervals are exact in real arithmetic given exact special functions.
        Independent quadrature checks are still required for floating-point
        roundoff. The raw/normalized measurement model is unchanged.
        """
        atol = 2e-5 * self.unit if atol is None else float(atol)
        if atol <= 0 or rtol < 0:
            raise ValueError("Need positive absolute and nonnegative relative tolerance")
        w = self.compress(coefficients)
        plus, minus = np.maximum(w, 0), np.maximum(-w, 0)
        zero = np.all(w == 0, axis=1)
        x = self.initial_nodes.copy()
        # Memory budget covers the persistent group arrays and workspaces;
        # the runner also measures actual process resident memory.
        tile_bytes = 128 * len(self.a) * min(128, len(x))
        max_nodes = max(1, (self.memory_bytes - tile_bytes) // max(1, 160 * len(w)))
        if len(x) > max_nodes:
            raise MemoryError("Split the permutation bank into smaller batches")
        curves = self._group_nodes(x, plus, minus)
        tail = minus @ self.tail_call
        peak = 0
        for iteration in range(max_refinements + 1):
            lower, upper = self._intervals(x, curves)
            lo = lower.sum(axis=1)
            hi = upper.sum(axis=1) + tail
            lo[zero] = hi[zero] = 0.
            error = hi - lo
            tolerance = atol + rtol * lo
            peak = max(peak, 160 * len(w) * len(x)
                       + 128 * len(self.a) * min(128, len(x)))
            if np.all(error <= tolerance):
                return AreaBounds(lo, hi, -w @ self.mean, len(x), iteration, peak)
            gap = upper - lower
            # Refine all cells contributing substantially to a row's error.
            active = error > tolerance
            threshold = .25 * tolerance[active, None] / max(1, len(x) - 1)
            split = np.any(gap[active] > threshold, axis=0)
            if not split.any():
                raise ArithmeticError("Tail or roundoff exceeds area tolerance")
            midpoint = (x[:-1][split] + x[1:][split]) / 2
            if len(x) + len(midpoint) > max_nodes:
                raise MemoryError("Area refinement exceeds bounded bank memory")
            added = self._group_nodes(midpoint, plus, minus)
            order = np.argsort(np.r_[x, midpoint])
            x = np.r_[x, midpoint][order]
            curves = np.concatenate([curves, added], axis=2)[:, :, order]
        raise ArithmeticError("Crossing-area brackets did not converge")
