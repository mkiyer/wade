"""Exact log-expression W1, with a readable NumPy resampling oracle."""
import numpy as np

from .permutation import draw_perms, validate_labels
from .pvalues import sequential_p

try:
    from . import _kernel
except ImportError:
    _kernel = None

HAVE_RUST_KERNEL = _kernel is not None and hasattr(_kernel, "update_w1")
STOP_EXCEEDANCES = 20
TIE_RTOL = 100 * np.finfo(float).eps


def resolve_backend(backend):
    if backend not in ("auto", "numpy", "rust"):
        raise ValueError("backend must be 'auto', 'numpy' or 'rust'")
    if backend == "rust" and not HAVE_RUST_KERNEL:
        raise RuntimeError("the compiled W1 kernel is unavailable; build/install the current wheel")
    return "rust" if backend != "numpy" and HAVE_RUST_KERNEL else "numpy"


def log_expression(counts, factors):
    # Keep the audited expression and evaluation order, including the offset.
    with np.errstate(over="ignore", invalid="ignore"):
        x = np.log2(counts / factors + 1)
    if not np.isfinite(x).all():
        raise ValueError("counts / size_factors exceeds the finite expression range")
    return x


class W1State:
    def __init__(self, expression, cond, *, stop=STOP_EXCEEDANCES):
        x = np.asarray(expression, dtype=float)
        if x.ndim != 2 or x.shape[1] < 2 or not np.isfinite(x).all():
            raise ValueError("expression must be a finite genes x samples matrix")
        labels = validate_labels(cond, x.shape[1])
        self.stop, self.cases = int(stop), int(labels.sum())
        if self.stop < 1 or self.stop != stop:
            raise ValueError("stop must be a positive integer")
        self.genes, self.samples = x.shape
        self.order = np.ascontiguousarray(np.argsort(x, axis=1, kind="stable"), dtype=np.uint64)
        self.gaps = np.ascontiguousarray(np.diff(np.take_along_axis(x, self.order.astype(np.intp), axis=1), axis=1))
        nc = np.cumsum(labels[self.order.astype(np.intp)], axis=1, dtype=np.int64)[:, :-1]
        mass = np.abs(self.samples * nc - self.cases * np.arange(1, self.samples))
        area = np.cumsum(mass * self.gaps, axis=1)[:, -1]
        self.observed = area / (self.cases * (self.samples - self.cases))
        self.thresholds = np.ascontiguousarray(area * (1 - TIE_RTOL))
        self.exceed = np.zeros(self.genes, np.uint64)
        self.tried = np.zeros(self.genes, np.uint64)

    def update(self, labels, *, backend):
        labels = np.asarray(labels)
        if (labels.ndim != 2 or labels.shape[1] != self.samples
                or not np.isin(labels, (0, 1)).all()
                or np.any(labels.sum(axis=1) != self.cases)):
            raise ValueError("every permutation must preserve the binary group sizes")
        labels = np.ascontiguousarray(labels, dtype=np.uint8)
        if backend == "rust":
            self.exceed, self.tried = _kernel.update_w1(
                self.order, self.gaps, labels, self.thresholds,
                self.exceed, self.tried, self.cases, self.stop)
            return
        # One gene at a time bounds scratch memory. Ordered cumsum matches Rust.
        for gene in np.flatnonzero(self.exceed < self.stop):
            nc = np.cumsum(labels[:, self.order[gene].astype(np.intp)], axis=1, dtype=np.int64)[:, :-1]
            mass = np.abs(self.samples * nc - self.cases * np.arange(1, self.samples))
            area = np.cumsum(mass * self.gaps[gene], axis=1)[:, -1]
            hits = area >= self.thresholds[gene]
            crossings = np.flatnonzero(hits)
            need = self.stop - int(self.exceed[gene])
            used = int(crossings[need - 1]) + 1 if len(crossings) >= need else len(labels)
            self.tried[gene] += used
            self.exceed[gene] += int(hits[:used].sum())


def infer(expression, cond, *, maximum, seed, strata, backend, batch=4096):
    """A chunk uses the same random stream as every other chunk."""
    state = W1State(expression, cond, stop=min(STOP_EXCEEDANCES, maximum))
    rng = np.random.default_rng(seed)
    generated = 0
    while generated < maximum and np.any(state.exceed < state.stop):
        labels = draw_perms(cond, min(batch, maximum - generated), rng=rng, strata=strata)
        state.update(labels, backend=backend)
        generated += len(labels)
    return (state.observed, sequential_p(state.exceed, state.tried, stop=state.stop,
                                        maximum=maximum), state.exceed, state.tried)
