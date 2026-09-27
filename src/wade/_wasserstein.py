"""Exact one-sided quantile areas, with a readable NumPy resampling oracle."""
import numpy as np

from .permutation import draw_perms, validate_labels
from .pvalues import sequential_p

try:
    from . import _kernel
except ImportError:
    _kernel = None

HAVE_RUST_KERNEL = _kernel is not None and hasattr(_kernel, "update_areas")
DIRECTIONS = ("up", "down")
SCALES = ("linear", "log")
STOP_EXCEEDANCES = 20
TIE_RTOL = 100 * np.finfo(float).eps


def resolve_backend(backend):
    if backend not in ("auto", "numpy", "rust"):
        raise ValueError("backend must be 'auto', 'numpy' or 'rust'")
    if backend == "rust" and not HAVE_RUST_KERNEL:
        raise RuntimeError("the compiled area kernel is unavailable; build/install the current wheel")
    return "rust" if backend != "numpy" and HAVE_RUST_KERNEL else "numpy"


def expression(counts, factors, scale):
    """Normalized counts on the test scale: count/size_factor, or log2 of that plus one."""
    if scale not in SCALES:
        raise ValueError("scale must be 'linear' or 'log'")
    # Keep the audited evaluation order, including the offset.
    with np.errstate(over="ignore", invalid="ignore"):
        x = counts / factors
        if scale == "log":
            x = np.log2(x + 1)
    if not np.isfinite(x).all():
        raise ValueError("counts / size_factors exceeds the finite expression range")
    return x


def _areas(prefix, grid, samples, gaps):
    """Ordered sums of the positive and negative CDF-difference mass, as in Rust."""
    mass = grid - samples * prefix
    return np.stack([np.cumsum(np.maximum(mass, 0) * gaps, axis=-1)[..., -1],
                     np.cumsum(np.maximum(-mass, 0) * gaps, axis=-1)[..., -1]], axis=-1)


class AreaState:
    """Upward and downward areas between case and control quantile functions.

    Up integrates where case quantiles exceed control quantiles, down the
    reverse; up + down is W1 and up - down the difference in means. Each
    direction keeps its own sequential stopping record under shared draws.
    """

    def __init__(self, expression, cond, *, active=(True, True), stop=STOP_EXCEEDANCES):
        x = np.asarray(expression, dtype=float)
        if x.ndim != 2 or x.shape[1] < 2 or not np.isfinite(x).all():
            raise ValueError("expression must be a finite genes x samples matrix")
        labels = validate_labels(cond, x.shape[1])
        self.stop, self.cases = int(stop), int(labels.sum())
        if self.stop < 1 or self.stop != stop:
            raise ValueError("stop must be a positive integer")
        self.active = tuple(bool(a) for a in active)
        if len(self.active) != 2 or not any(self.active):
            raise ValueError("at least one direction must be tested")
        self.genes, self.samples = x.shape
        self.grid = self.cases * np.arange(1, self.samples)
        self.order = np.ascontiguousarray(np.argsort(x, axis=1, kind="stable"), dtype=np.uint64)
        self.gaps = np.ascontiguousarray(np.diff(np.take_along_axis(x, self.order.astype(np.intp), axis=1), axis=1))
        prefix = np.cumsum(labels[self.order.astype(np.intp)], axis=1, dtype=np.int64)[:, :-1]
        area = _areas(prefix, self.grid, self.samples, self.gaps)
        self.observed = area / (self.cases * (self.samples - self.cases))
        self.thresholds = np.ascontiguousarray(area * (1 - TIE_RTOL))
        self.exceed = np.zeros((self.genes, 2), np.uint64)
        self.tried = np.zeros((self.genes, 2), np.uint64)

    def live(self):
        return (self.exceed < self.stop) & np.array(self.active)

    def update(self, labels, *, backend):
        labels = np.asarray(labels)
        if (labels.ndim != 2 or labels.shape[1] != self.samples
                or not np.isin(labels, (0, 1)).all()
                or np.any(labels.sum(axis=1) != self.cases)):
            raise ValueError("every permutation must preserve the binary group sizes")
        labels = np.ascontiguousarray(labels, dtype=np.uint8)
        if backend == "rust":
            self.exceed, self.tried = _kernel.update_areas(
                self.order, self.gaps, labels, self.thresholds,
                self.exceed, self.tried, self.active, self.cases, self.stop)
            return
        # One gene at a time bounds scratch memory. Ordered cumsum matches Rust.
        live = self.live()
        for gene in np.flatnonzero(live.any(axis=1)):
            prefix = np.cumsum(labels[:, self.order[gene].astype(np.intp)], axis=1, dtype=np.int64)[:, :-1]
            hits = _areas(prefix, self.grid, self.samples, self.gaps[gene]) >= self.thresholds[gene]
            for d in np.flatnonzero(live[gene]):
                crossings = np.flatnonzero(hits[:, d])
                need = self.stop - int(self.exceed[gene, d])
                used = int(crossings[need - 1]) + 1 if len(crossings) >= need else len(labels)
                self.tried[gene, d] += used
                self.exceed[gene, d] += int(hits[:used, d].sum())


def infer(expression, cond, *, active, maximum, seed, strata, backend, batch=4096):
    """A chunk uses the same random stream as every other chunk.

    Returns observed areas and p-values as (genes, 2) arrays, columns ordered
    as DIRECTIONS; an untested direction has NaN p-values and empty records.
    """
    state = AreaState(expression, cond, active=active, stop=min(STOP_EXCEEDANCES, maximum))
    rng = np.random.default_rng(seed)
    generated = 0
    while generated < maximum and state.live().any():
        labels = draw_perms(cond, min(batch, maximum - generated), rng=rng, strata=strata)
        state.update(labels, backend=backend)
        generated += len(labels)
    pvalue = np.full(state.exceed.shape, np.nan)
    for d in np.flatnonzero(state.active):
        pvalue[:, d] = sequential_p(state.exceed[:, d], state.tried[:, d], stop=state.stop, maximum=maximum)
    return state.observed, pvalue, state.exceed, state.tried
