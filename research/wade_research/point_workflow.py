"""Candidate directional point workflow; not the released public API.

Both directions share sorting and label draws. BH covers every tested
gene-direction pair, not two separately corrected gene lists. The null is
exchangeability/equality of distributions, not a fitted count likelihood.
"""
from dataclasses import dataclass

import numpy as np

from wade._wasserstein import STOP_EXCEEDANCES, TIE_RTOL
from wade.io import Condition, as_counts
from wade.normalize import Normalization, _indices, _named, fit_normalization
from wade.permutation import draw_perms, permutation_space, validate_labels
from wade.pvalues import bh_adjust, sequential_p


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class DirectionalState:
    """Exact upward/downward CDF areas with independent stopping records."""

    def __init__(self, expression, cond, *, stop=STOP_EXCEEDANCES):
        x = np.asarray(expression, float)
        if x.ndim != 2 or x.shape[1] < 2 or not np.isfinite(x).all():
            raise ValueError("expression must be a finite genes x samples matrix")
        labels = validate_labels(cond, x.shape[1])
        self.stop = _positive_int(stop, "stop")
        self.samples, self.cases = x.shape[1], int(labels.sum())
        self.order = np.argsort(x, axis=1, kind="stable")
        self.gaps = np.diff(np.take_along_axis(x, self.order, axis=1), axis=1)
        self.grid = self.cases * np.arange(1, self.samples)
        self.denominator = self.cases * (self.samples - self.cases)
        self.area = np.array([self._areas(i, labels[None])[0] for i in range(len(x))]).reshape(-1, 2)
        self.observed = self.area / self.denominator
        if not np.isfinite(self.area).all():
            raise ValueError("expression exceeds the finite area range")
        self.threshold = self.area * (1 - TIE_RTOL)
        self.exceed = np.zeros(self.area.shape, np.uint64)
        self.tried = np.zeros(self.area.shape, np.uint64)

    def _areas(self, gene, labels):
        prefix = np.cumsum(labels[:, self.order[gene]], axis=1, dtype=np.int64)[:, :-1]
        # Control CDF minus case CDF; positive mass denotes upward enrichment.
        mass = self.grid - self.samples * prefix
        return np.column_stack([
            np.cumsum(np.maximum(sign * mass, 0) * self.gaps[gene], axis=1)[:, -1]
            for sign in (1, -1)
        ])

    def update(self, labels):
        labels = np.asarray(labels)
        if (labels.ndim != 2 or labels.shape[1] != self.samples
                or not np.isin(labels, (0, 1)).all()
                or np.any(labels.sum(axis=1) != self.cases)):
            raise ValueError("permutations must preserve binary group sizes")
        for gene in np.flatnonzero(np.any(self.exceed < self.stop, axis=1)):
            hits = self._areas(gene, labels) >= self.threshold[gene]
            for direction in np.flatnonzero(self.exceed[gene] < self.stop):
                crossings = np.flatnonzero(hits[:, direction])
                need = self.stop - int(self.exceed[gene, direction])
                used = int(crossings[need - 1]) + 1 if len(crossings) >= need else len(labels)
                self.tried[gene, direction] += used
                self.exceed[gene, direction] += np.count_nonzero(hits[:used, direction])


@dataclass
class PointResult:
    """Columns of each inference array are up, down; exclusions remain NaN."""

    gene: np.ndarray
    statistic: np.ndarray
    pvalue: np.ndarray
    padj: np.ndarray
    status: np.ndarray
    n_permutations: np.ndarray
    n_exceedances: np.ndarray
    expression: np.ndarray
    normalization: Normalization
    params: dict

    @property
    def tested(self):
        return self.status == "tested"


def point_workflow(counts, cond, *, normalization="rle", test_genes=None,
                   scale="linear", max_permutations=999_999, seed=1,
                   strata=None, batch=64):
    """Evaluate a frozen two-direction candidate using existing WADE preprocessing.

    This NumPy reference intentionally precedes native optimization and public
    API migration. It retains all observed zeros and adds no jitter. Scale is
    prespecified; searching both scales requires a larger declared test family.
    """
    maximum = _positive_int(max_permutations, "max_permutations")
    batch = _positive_int(batch, "batch")
    if scale not in ("linear", "log"):
        raise ValueError("scale must be 'linear' or 'log'")
    data = as_counts(counts)
    labels = cond.vector(data.sample_names) if isinstance(cond, Condition) else cond
    labels = validate_labels(labels, data.shape[1])
    if min(int(labels.sum()), int((labels == 0).sum())) < 2:
        raise ValueError("at least two independent samples per group are required")
    space = permutation_space(labels, strata)
    if space["log10_space"] == 0:
        raise ValueError("the specified strata permit no label exchanges")
    if isinstance(normalization, str):
        fitted = fit_normalization(data, method=normalization)
    elif isinstance(normalization, Normalization):
        fitted = normalization._align(data, named_samples=_named(counts, None))
    else:
        raise TypeError("normalization must be a method name or fitted Normalization")
    if not np.all(data.values.sum(axis=0) > 0):
        raise ValueError("remove empty samples before analysis")
    selected = np.ones(data.shape[0], bool)
    if test_genes is not None:
        selected[:] = False
        selected[_indices(test_genes, data.gene_names, "test genes")] = True
    nonzero = np.any(data.values > 0, axis=1)
    tested = selected & nonzero
    status = np.where(~nonzero, "all_zero", np.where(selected, "tested", "filtered"))
    x = fitted.transform(data)
    if scale == "log":
        x = np.log2(x + 1)
    if seed is None:
        seed = np.random.SeedSequence().entropy
    if isinstance(seed, (np.random.Generator, np.random.BitGenerator, np.random.SeedSequence)):
        raise TypeError("seed must be an integer, an integer sequence, or None")
    rng = np.random.default_rng(seed)
    state = DirectionalState(x[tested], labels, stop=min(STOP_EXCEEDANCES, maximum))
    generated = 0
    while generated < maximum and np.any(state.exceed < state.stop):
        bank = draw_perms(labels, min(batch, maximum-generated), rng=rng, strata=strata)
        state.update(bank)
        generated += len(bank)
    shape = (len(tested), 2)
    statistic, pvalue = np.full(shape, np.nan), np.full(shape, np.nan)
    exceed, tried = np.zeros(shape, np.uint64), np.zeros(shape, np.uint64)
    statistic[tested] = state.observed
    exceed[tested], tried[tested] = state.exceed, state.tried
    pvalue[tested] = sequential_p(state.exceed, state.tried, stop=state.stop, maximum=maximum)
    params = dict(algorithm="directional-point-candidate-v1", directions=["up", "down"],
                  scale=scale, family="all tested gene-direction pairs", hypotheses=int(2*tested.sum()),
                  max_permutations=maximum, stop_exceedances=state.stop, minimum_pvalue=1/(maximum+1),
                  seed=np.asarray(seed).tolist(), batch=batch, permutation_space=space,
                  strata=None if strata is None else np.asarray(strata).tolist(),
                  singleton_bh_floor=float(2*tested.sum()/(maximum+1)))
    return PointResult(data.gene_names.copy(), statistic, pvalue,
                       bh_adjust(pvalue.ravel()).reshape(shape), status, tried, exceed, x, fitted, params)
