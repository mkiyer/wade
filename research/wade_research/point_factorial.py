"""Exact paired direction/scale experiment, with checked optional acceleration.

The benchmark-only Rust loop has no RNG. A NumPy oracle and the existing public
unsigned baseline check its statistics, exceedances and stopping records.
"""
import ctypes
from pathlib import Path
import subprocess
import sys

import numpy as np

from wade._wasserstein import TIE_RTOL, STOP_EXCEEDANCES
from wade.permutation import draw_perms, validate_labels
from wade.pvalues import bh_adjust, sequential_p

METHODS = ("directional_linear", "unsigned_linear", "directional_log", "unsigned_log")
LANES = {"directional_linear": [0, 1], "unsigned_linear": [2],
         "directional_log": [3, 4], "unsigned_log": [5]}


def build_backend(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parents[1]/"point_permutations.rs"
    target = destination/("point_permutations.dylib" if sys.platform == "darwin" else "point_permutations.so")
    subprocess.run(["rustc", "--edition=2021", "--crate-type", "cdylib", "-C", "opt-level=3",
                    str(source), "-o", str(target)], check=True, capture_output=True, text=True)
    return target


def load_backend(path):
    library = ctypes.CDLL(str(Path(path).resolve()))
    update = library.update_points
    pointer = np.ctypeslib.ndpointer
    update.argtypes = [ctypes.c_size_t]*4 + [ctypes.c_uint64] + [
        pointer(dtype=np.uintp, ndim=2, flags="C_CONTIGUOUS"),
        pointer(dtype=np.float64, ndim=3, flags="C_CONTIGUOUS"),
        pointer(dtype=np.uint8, ndim=2, flags="C_CONTIGUOUS"),
        pointer(dtype=np.float64, ndim=2, flags="C_CONTIGUOUS"),
        pointer(dtype=np.uint64, ndim=2, flags=("C_CONTIGUOUS", "WRITEABLE")),
        pointer(dtype=np.uint64, ndim=2, flags=("C_CONTIGUOUS", "WRITEABLE"))]
    update.restype = None
    return update


class FactorialState:
    def __init__(self, normalized, cond, *, stop=STOP_EXCEEDANCES):
        x = np.asarray(normalized, float)
        if x.ndim != 2 or x.shape[1] < 2 or not np.isfinite(x).all() or np.any(x < 0):
            raise ValueError("normalized expression must be finite nonnegative genes x samples")
        self.cond = validate_labels(cond, x.shape[1])
        if isinstance(stop, bool) or not isinstance(stop, (int, np.integer)) or stop < 1:
            raise ValueError("stop must be a positive integer")
        self.stop, self.cases = int(stop), int(self.cond.sum())
        self.genes, self.samples = x.shape
        self.denominator = self.cases*(self.samples-self.cases)
        self.grid = self.cases*np.arange(1, self.samples)
        self.order = np.ascontiguousarray(np.argsort(x, axis=1, kind="stable"), dtype=np.uintp)
        sorted_x = np.take_along_axis(x, self.order, axis=1)
        self.gaps = np.ascontiguousarray(np.stack((np.diff(sorted_x, axis=1),
                                                   np.diff(np.log2(sorted_x+1), axis=1)), axis=1))
        area = np.array([self.areas(gene, self.cond[None])[0] for gene in range(self.genes)]).reshape(-1, 6)
        if not np.isfinite(area).all():
            raise ValueError("expression exceeds the finite area range")
        self.observed = area/self.denominator
        self.threshold = np.ascontiguousarray(area*(1-TIE_RTOL))
        self.exceed = np.zeros(area.shape, np.uint64)
        self.tried = np.zeros(area.shape, np.uint64)

    def areas(self, gene, labels):
        mass = self.grid-self.samples*np.cumsum(labels[:, self.order[gene]], axis=1, dtype=np.int64)[:, :-1]
        weights = (np.maximum(mass, 0), np.maximum(-mass, 0), np.abs(mass))
        return np.column_stack([np.cumsum(weight*gap, axis=1)[:, -1]
                                for gap in self.gaps[gene] for weight in weights])

    def update(self, labels, native=None):
        labels = np.asarray(labels)
        if (labels.ndim != 2 or labels.shape[1] != self.samples
                or not np.isin(labels, (0, 1)).all()
                or np.any(labels.sum(axis=1) != self.cases)):
            raise ValueError("permutations must preserve binary group sizes")
        labels = np.ascontiguousarray(labels, dtype=np.uint8)
        if not len(labels) or not self.genes:
            return
        if native is not None:
            # Dimensions and ranges are checked before crossing the private FFI.
            if (self.order.shape != (self.genes, self.samples)
                    or self.gaps.shape != (self.genes, 2, self.samples-1)
                    or any(a.shape != (self.genes, 6) for a in (self.threshold, self.exceed, self.tried))
                    or np.any(self.order >= self.samples) or not np.isfinite(self.gaps).all()
                    or np.any(self.gaps < 0) or not np.isfinite(self.threshold).all()
                    or np.any(self.threshold < 0) or np.any(self.exceed > self.stop)
                    or np.any(self.tried < self.exceed)):
                raise ValueError("invalid native state")
            native(self.genes, self.samples, len(labels), self.cases, self.stop,
                   self.order, self.gaps, labels, self.threshold, self.exceed, self.tried)
            return
        for gene in np.flatnonzero(np.any(self.exceed < self.stop, axis=1)):
            hits = self.areas(gene, labels) >= self.threshold[gene]
            for lane in np.flatnonzero(self.exceed[gene] < self.stop):
                crossings = np.flatnonzero(hits[:, lane])
                need = self.stop-int(self.exceed[gene, lane])
                used = int(crossings[need-1])+1 if len(crossings) >= need else len(labels)
                self.tried[gene, lane] += used
                self.exceed[gene, lane] += hits[:used, lane].sum()


def factorial_infer(normalized, cond, *, maximum, seed, native=None, batch=4096):
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1 for v in (maximum, batch)):
        raise ValueError("maximum and batch must be positive integers")
    state = FactorialState(normalized, cond, stop=min(STOP_EXCEEDANCES, maximum))
    rng, generated = np.random.default_rng(seed), 0
    while generated < maximum and np.any(state.exceed < state.stop):
        labels = draw_perms(cond, min(batch, maximum-generated), rng=rng)
        state.update(labels, native)
        generated += len(labels)
    pvalue = sequential_p(state.exceed, state.tried, stop=state.stop, maximum=maximum)
    qvalue = np.full(pvalue.shape, np.nan)
    for lanes in LANES.values():
        # Four distinct benchmark procedures, not one analysis choosing a best q.
        qvalue[:, lanes] = bh_adjust(pvalue[:, lanes].ravel()).reshape(-1, len(lanes))
    return state.observed, pvalue, qvalue, state.exceed, state.tried
