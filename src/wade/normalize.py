"""Fit a normalization once; reuse its factors across testing and display filters."""
from dataclasses import dataclass, replace
import hashlib
import json
import numpy as np
from .io import Counts, as_counts


def _positive(values, shape, what):
    a = np.asarray(values, dtype=float)
    if a.shape != shape or not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError(f"{what} must be finite, positive, and have shape {shape}")
    return a


def _readonly(a, dtype=float):
    a = np.array(a, dtype=dtype, copy=True)
    a.flags.writeable = False
    return a


def _indices(selection, names, what):
    """Rows or columns chosen by a Boolean mask or by unique labels; never silently drop a name."""
    sel = np.asarray(selection)
    if sel.ndim != 1:
        raise ValueError(f"{what} must be one-dimensional")
    if sel.dtype.kind == "b":
        if len(sel) != len(names):
            raise ValueError(f"{what} mask must have {len(names)} entries")
        idx = np.flatnonzero(sel)
    else:
        lookup = {name: i for i, name in enumerate(names)}
        if len(lookup) != len(names):
            raise ValueError(f"{what} requires unique identifiers")
        missing = [v for v in sel.tolist() if v not in lookup]
        if missing:
            raise ValueError(f"unknown {what}: {missing[:8]}")
        idx = np.array([lookup[v] for v in sel.tolist()], dtype=int)
    if len(np.unique(idx)) != len(idx):
        raise ValueError(f"duplicate {what}")
    return idx


def _fit_input(counts, gene_names, sample_names):
    data = as_counts(counts, gene_names=gene_names, sample_names=sample_names)
    totals = data.values.sum(axis=0)
    if not data.shape[1] or np.any(~np.isfinite(totals)) or np.any(totals <= 0):
        bad = data.sample_names[~(np.isfinite(totals) & (totals > 0))].tolist()
        raise ValueError(f"zero library size or invalid total for samples {bad[:8]}; remove empty samples first")
    return data, totals


def _fingerprint(data):
    h = hashlib.sha256()
    h.update(str(data.shape).encode())
    h.update(
        json.dumps(
            [np.asarray(data.gene_names).tolist(), np.asarray(data.sample_names).tolist()],
            ensure_ascii=False,
        ).encode()
    )
    # Canonical little-endian floats, bounded to one row at a time.
    for row in data.values:
        h.update(np.asarray(row, dtype="<f8").tobytes())
    return h.hexdigest()


@dataclass(frozen=True, eq=False)
class Normalization:
    """Frozen sample size factors and their reference provenance.

    Factors have geometric mean one over the fit cohort. Selecting samples
    never refits or recenters them. transform() returns counts / size_factor,
    the normalized scale WADE reports; its tests run on counts thinned by
    these factors. External factors must include sampling depth, not just a
    composition correction.
    """
    method: str
    sample_names: tuple
    size_factors: np.ndarray
    library_totals: np.ndarray
    reference_genes: tuple
    used_reference_genes: tuple
    fit_sample_names: tuple
    counts_sha256: str
    source: str = "WADE"

    def __post_init__(self):
        for name in ("sample_names", "reference_genes", "used_reference_genes", "fit_sample_names"):
            values = tuple(v.item() if isinstance(v, np.generic) else v for v in getattr(self, name))
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique identifiers")
            object.__setattr__(self, name, values)
        if not set(self.used_reference_genes) <= set(self.reference_genes):
            raise ValueError("used references must belong to the candidate references")
        if not set(self.sample_names) <= set(self.fit_sample_names):
            raise ValueError("selected samples must belong to the fit cohort")
        if self.method not in ("rle", "library", "supplied") or not self.source.strip():
            raise ValueError("invalid normalization method or source")
        for name in ("size_factors", "library_totals"):
            a = _positive(getattr(self, name), (len(self.sample_names),), name)
            object.__setattr__(self, name, _readonly(a))

    def __repr__(self):
        return (f"Normalization(method={self.method!r}, samples={len(self.sample_names)}, "
                f"references={len(self.used_reference_genes)}/{len(self.reference_genes)})")

    @classmethod
    def from_size_factors(cls, counts, factors, *, source, gene_names=None, sample_names=None):
        """Import complete exposures, positionally or keyed by sample ID.

        For scran, use its positive deconvolution factors. For edgeR, multiply
        composition factors by library sizes first. WADE centers to geometric
        mean one once, on this fit cohort; no sparse-assay calibration is implied.
        """
        data, totals = _fit_input(counts, gene_names, sample_names)
        if hasattr(factors, "keys"):
            if set(factors.keys()) != set(data.sample_names):
                raise ValueError("size factor keys must match the full sample set exactly")
            factors = [factors[name] for name in data.sample_names]
        factors = _positive(factors, (data.shape[1],), "size factors")
        logs = np.log(factors)
        return _make(data, totals, "supplied", np.exp(logs - logs.mean()), source=source)

    def select_samples(self, selection):
        idx = _indices(selection, self.sample_names, "samples")
        return replace(self, sample_names=tuple(self.sample_names[i] for i in idx),
                       size_factors=self.size_factors[idx], library_totals=self.library_totals[idx])

    def _align(self, data, *, named_samples):
        if named_samples:
            return self.select_samples(data.sample_names)
        if data.shape[1] != len(self.sample_names):
            raise ValueError("unnamed counts must retain all fitted samples in order; supply sample_names")
        return self

    def transform(self, counts, *, gene_names=None, sample_names=None):
        """Return linear size-factor-normalized counts; never refit on a subset."""
        data = as_counts(counts, gene_names=gene_names, sample_names=sample_names)
        fitted = self._align(data, named_samples=_named(counts, sample_names))
        out = data.values / fitted.size_factors
        if not np.isfinite(out).all():
            raise ValueError("normalized counts exceed the finite numerical range")
        return out

    def to_dict(self):
        return dict(algorithm_version=2, expression="counts / size_factor",
                    method=self.method, source=self.source,
                    sample_names=list(self.sample_names), fit_sample_names=list(self.fit_sample_names),
                    size_factors=self.size_factors.tolist(), library_totals=self.library_totals.tolist(),
                    reference_genes=list(self.reference_genes), used_reference_genes=list(self.used_reference_genes),
                    counts_sha256=self.counts_sha256)

    @classmethod
    def from_dict(cls, record):
        record = dict(record)
        if record.pop("algorithm_version", None) != 2 or record.pop("expression", None) != "counts / size_factor":
            raise ValueError("unsupported normalization record")
        return cls(**record)


def _named(counts, names):
    return names is not None or isinstance(counts, Counts) or hasattr(counts, "columns")


def _make(data, totals, method, factors, *, candidates=(), used=(), source="WADE"):
    return Normalization(method, tuple(data.sample_names), factors, totals,
                         tuple(candidates), tuple(used), tuple(data.sample_names), _fingerprint(data), source)


def fit_normalization(counts, *, method="rle", reference_genes=None, gene_names=None, sample_names=None):
    """Fit RLE on a declared count universe, optionally using internal controls.

    RLE uses positive-in-every-sample references and median log ratios. It
    assumes an adequately stable reference population; a low-variance panel
    is not automatically valid. Initial QC may precede fitting. Later test
    filtering must reuse these factors.

    method='library' uses total counts as exposures and assumes comparable
    RNA composition. It requires no other samples to calculate each total;
    geometric centering merely fixes the cohort's count unit. Sparse data
    with complex composition should use externally justified factors.
    """
    if method not in ("rle", "library"):
        raise ValueError("method must be 'rle' or 'library'; supply external factors for other estimators")
    data, totals = _fit_input(counts, gene_names, sample_names)
    if method == "library":
        if reference_genes is not None:
            raise ValueError("reference_genes applies to RLE, not library normalization")
        logs = np.log(totals)
        return _make(data, totals, method, np.exp(logs - logs.mean()))
    c = data.values
    idx = np.arange(c.shape[0]) if reference_genes is None else _indices(reference_genes, data.gene_names, "reference genes")
    reference, keep = np.empty(len(idx)), np.empty(len(idx), bool)
    for start in range(0, len(idx), 512):
        rows = slice(start, start + 512)
        x = c[idx[rows]]
        keep[rows] = np.all(x > 0, axis=1)
        with np.errstate(divide="ignore"):
            logs = np.log(x)
        logs[~np.isfinite(logs)] = 0
        reference[rows] = logs.sum(axis=1) / c.shape[1]
    if not keep.any():
        raise ValueError(f"rle: no usable all-positive reference genes among {len(idx)} candidates; "
                         "provide validated references or externally fitted factors")
    used, reference = idx[keep], reference[keep]
    ratios = np.array([np.median(np.log(c[used, j]) - reference) for j in range(c.shape[1])])
    factors = np.exp(ratios - ratios.mean())
    return _make(data, totals, method, factors, candidates=data.gene_names[idx], used=data.gene_names[used])


def cpm(counts):
    """Counts per million of each sample's total; every sample must have reads.

    A descriptive abundance conversion, not WADE's inferential expression scale.
    All measured genes should contribute to the library denominator.
    """
    data, totals = _fit_input(counts, None, None)
    return data.values / totals * 1e6


def tpm(counts, lengths):
    """Transcripts per million from positive per-gene lengths in a common unit.

    Uses length-adjusted rates and a separate denominator for each sample.
    """
    data, _ = _fit_input(counts, None, None)
    lengths = _positive(lengths, (data.shape[0],), "gene lengths")
    rates = data.values / lengths[:, None]
    totals = rates.sum(axis=0)
    if not np.isfinite(totals).all() or np.any(totals <= 0):
        raise ValueError("length-adjusted totals must be finite and positive")
    return rates / totals * 1e6
