"""Data in, results out — ``ROADMAP.md`` §1.

**WADE does not own I/O.** There are no file readers here. polars and pandas
already read CSV, TSV, Parquet, Arrow, Excel and gzip better than this package
would, and every reader written here would be surface area with nothing to do
with the statistic. What this module does instead:

* :func:`as_counts` — **accept** a count matrix in whatever shape it arrives
  in (NumPy array, polars or pandas DataFrame, any sparse matrix with
  ``.toarray()``) and turn it into the one shape the statistic uses: a dense
  ``genes x samples`` float64 array with gene and sample labels beside it.
  Labels are optional; the orientation is not guessed.
* :func:`condition` — turn a **sample metadata** table into a case/control
  assignment, and align it to the matrix's samples *strictly*. This is the
  one genuinely error-prone step in the whole area and the reason the helper
  exists: a silently dropped sample is a silently different analysis, and
  because library sizes are computed on the samples handed in, a silently
  different normalization too.
* :func:`to_frame` and :func:`write_results` — the results, which **are**
  WADE's to own, plus a JSON manifest of what produced them.

Neither polars nor pandas is a dependency: frames are duck-typed on
``.columns`` and ``.to_numpy()``, and only :func:`to_frame` and
:func:`write_results` need polars at all, imported inside the call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "Counts",
    "Condition",
    "as_counts",
    "condition",
    "to_frame",
    "write_results",
    "manifest",
    "RESULT_COLUMNS",
]


# ---------------------------------------------------------------------------
# Accepting a count matrix


@dataclass(frozen=True)
class Counts:
    """A count matrix in WADE's canonical shape, with whatever labels came with it.

    ``values`` is always dense ``genes x samples`` float64: every WADE statistic
    is a quantile over all the samples of one gene, so there is no sparse path
    through the arithmetic and a sparse input is materialized on entry.
    """

    values: np.ndarray                  # (genes, samples) float64
    gene_names: np.ndarray              # (genes,) object
    sample_names: np.ndarray            # (samples,) object
    #: Numeric non-sample columns of an input frame, per gene — where
    #: ``normalizer="Length"`` is looked up.
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    def normalizer(self, column: str) -> np.ndarray:
        """The per-gene vector held in ``meta`` under ``column``."""
        if column not in self.meta:
            raise KeyError(
                f"no column {column!r} to use as a normalizer; the matrix carries "
                f"{sorted(self.meta) or 'no metadata columns'}. Pass the values as an "
                f"array instead, or name a column that is in the counts frame."
            )
        return self.meta[column]


def _is_frame(obj) -> bool:
    """polars and pandas, without importing either."""
    return (hasattr(obj, "columns") and hasattr(obj, "to_numpy")
            and not isinstance(obj, np.ndarray))


def _column(frame, name) -> np.ndarray:
    """One column of a duck-typed frame, as an array."""
    col = frame[name]
    return np.asarray(col.to_numpy() if hasattr(col, "to_numpy") else col)


def _positional(prefix: str, n: int) -> np.ndarray:
    return np.array([f"{prefix}{i}" for i in range(n)], dtype=object)


def _numeric(a: np.ndarray) -> bool:
    if a.dtype.kind in "iufb":
        return True
    return a.dtype == object and all(isinstance(v, (int, float, np.number)) and
                                     not isinstance(v, bool) for v in a[:32])


def _from_frame(frame, id_column, sample_columns, exclude=()):
    """(values, gene_names, meta) from a duck-typed DataFrame."""
    cols = list(frame.columns)
    if not cols:
        raise ValueError("the counts frame has no columns")

    if id_column == "auto":
        first = _column(frame, cols[0])
        id_column = cols[0] if not _numeric(first) else None
    elif isinstance(id_column, (int, np.integer)) and not isinstance(id_column, bool):
        id_column = cols[int(id_column)]
    if id_column is not None and id_column not in cols:
        raise KeyError(f"no column {id_column!r} in the counts frame; it has {cols}")

    gene_names = (np.asarray(_column(frame, id_column), dtype=object)
                  if id_column is not None else None)

    rest = [c for c in cols if c != id_column]
    if sample_columns is None:
        # A column named as the normalizer is metadata, not a sample: without
        # this a featureCounts-lite frame (id, Length, samples) would quietly
        # analyse the gene lengths as an extra sample.
        sample_columns = [c for c in rest if c not in set(exclude)]
        # Everything after the id column is a sample by default. A non-numeric
        # leftover means the frame is not a plain matrix — a featureCounts file
        # has Chr and Strand in there — so refuse rather than analyse a
        # chromosome name as expression.
        bad = [c for c in sample_columns if not _numeric(_column(frame, c))]
        if bad:
            raise ValueError(
                f"columns {bad} are not numeric, so they cannot be samples. Pass "
                f"sample_columns=[...] to say which columns hold counts; the others "
                f"are kept as per-gene metadata. (A featureCounts file needs this: "
                f"its Chr/Start/End/Strand/Length columns sit between the gene id "
                f"and the samples.)"
            )
    else:
        sample_columns = list(sample_columns)
        missing = [c for c in sample_columns if c not in cols]
        if missing:
            raise KeyError(f"sample_columns not in the counts frame: {missing}")
    if not sample_columns:
        raise ValueError("no sample columns were found in the counts frame")

    try:                                        # polars and pandas both slice by list
        values = np.asarray(frame[sample_columns].to_numpy(), dtype=np.float64)
    except (TypeError, KeyError, AttributeError):
        values = np.column_stack([_column(frame, c) for c in sample_columns]).astype(np.float64)
    meta = {c: _column(frame, c) for c in rest
            if c not in sample_columns and _numeric(_column(frame, c))}
    return values, gene_names, np.asarray(sample_columns, dtype=object), meta


def as_counts(
    obj,
    *,
    genes: str = "rows",
    id_column="auto",
    sample_columns=None,
    exclude=(),
    gene_names=None,
    sample_names=None,
) -> Counts:
    """Accept a count matrix in any of the supported shapes.

    Parameters
    ----------
    obj
        A 2-D NumPy array; a polars or pandas DataFrame (duck-typed on
        ``.columns`` and ``.to_numpy()``, so neither is a dependency); any
        sparse matrix exposing ``.toarray()``; or an existing :class:`Counts`,
        which is returned unchanged.
    genes
        ``"rows"`` (the default and WADE's canonical orientation) or
        ``"columns"``, which **transposes**. Never inferred: a square matrix
        would have no tell, and getting it wrong silently compares the wrong
        things.
    id_column
        For a frame: ``"auto"`` takes the first column when it is non-numeric,
        a name or integer position pins one, ``None`` means the frame is all
        samples and gene names are positional.
    sample_columns
        For a frame: which columns hold counts. By default every column after
        the id column, and a **non-numeric** leftover raises rather than being
        read as a sample. Numeric leftovers become :attr:`Counts.meta`, which
        is where a ``normalizer="Length"`` is looked up.
    exclude
        Column names that are **not** samples, kept as per-gene metadata.
        :func:`wade.wade` passes the normalizer's column name here, so
        ``normalizer="Length"`` never doubles as an extra sample.
    gene_names, sample_names
        Explicit labels, overriding anything found on the object. Positional
        names (``gene0``, ``sample0``, ...) are synthesized when neither is
        available — labels are optional throughout.
    """
    if isinstance(obj, Counts):
        return obj
    if genes not in ("rows", "columns"):
        raise ValueError(f"genes must be 'rows' or 'columns'; got {genes!r}")

    meta: dict = {}
    found_genes = found_samples = None

    if _is_frame(obj):
        # A frame's columns are the axis with labels, so read it in its own
        # orientation and transpose afterwards if the caller says so.
        values, found_genes, found_samples, meta = _from_frame(
            obj, id_column, sample_columns, exclude)
    elif hasattr(obj, "toarray"):                     # scipy.sparse and friends
        values = np.asarray(obj.toarray(), dtype=np.float64)
    else:
        values = np.asarray(obj, dtype=np.float64)

    if values.ndim != 2:
        raise ValueError(f"expected a 2-D genes x samples matrix, got shape {values.shape}")

    if genes == "columns":
        # The frame's rows were samples, so its leftover columns described
        # samples, not genes: they are not gene metadata and are dropped.
        values = values.T
        found_genes, found_samples = found_samples, found_genes
        meta = {}
    values = np.ascontiguousarray(values, dtype=np.float64)
    g, n = values.shape

    if gene_names is not None:
        found_genes = np.asarray(gene_names, dtype=object)
    if sample_names is not None:
        found_samples = np.asarray(sample_names, dtype=object)
    if found_genes is None:
        found_genes = _positional("gene", g)
    if found_samples is None:
        found_samples = _positional("sample", n)

    for label, arr, want in (("gene", found_genes, g), ("sample", found_samples, n)):
        if arr.shape != (want,):
            raise ValueError(
                f"{label}_names must have one entry per {label}: expected {want}, "
                f"got {arr.shape[0] if arr.ndim else arr.shape}"
            )
        dup = _duplicates(arr)
        if dup:
            raise ValueError(f"duplicate {label} names: {dup[:8]}"
                             + (f" and {len(dup) - 8} more" if len(dup) > 8 else ""))

    if not np.all(np.isfinite(values)):
        raise ValueError("the count matrix contains non-finite values")
    if np.any(values < 0):
        raise ValueError("counts must be non-negative")

    meta = {k: np.asarray(v) for k, v in meta.items() if len(v) == g}
    return Counts(values=values, gene_names=found_genes, sample_names=found_samples, meta=meta)


def _duplicates(arr: np.ndarray) -> list:
    vals, counts = np.unique(arr.astype(object), return_counts=True)
    return [v for v, c in zip(vals, counts) if c > 1]


# ---------------------------------------------------------------------------
# The condition, from sample metadata


@dataclass(frozen=True)
class Condition:
    """A case/control assignment by sample name, plus what the labels were.

    Produced by :func:`condition`. :meth:`vector` aligns it to a matrix's
    samples; :func:`wade.wade` does that for you when handed one of these.
    """

    labels: dict                 # sample name -> 1 (case) or 0 (control)
    case: object                 # the value of `column` that meant case
    control: object              # ... and control
    column: str | None = None
    dropped: tuple = ()          # samples at some other level, when drop_other

    def __len__(self) -> int:
        return len(self.labels)

    def vector(self, sample_names, *, allow_extra: bool = False) -> np.ndarray:
        """The 0/1 vector for ``sample_names``, in that order. Strict.

        Raises, naming the offenders, when a sample has no assignment or an
        assignment matches no sample. Both are usually a mis-joined sample
        sheet, and both would otherwise change the analysis silently.
        """
        names = np.asarray(sample_names, dtype=object)
        missing = [s for s in names.tolist() if s not in self.labels]
        if missing:
            raise ValueError(
                f"{len(missing)} sample(s) in the count matrix have no entry in the "
                f"sample metadata: {missing[:8]}"
                + (f" and {len(missing) - 8} more. " if len(missing) > 8 else ". ")
                + (f"{len(self.dropped)} sample(s) were dropped as neither "
                   f"{self.case!r} nor {self.control!r}: {list(self.dropped)[:8]}. "
                   if self.dropped else "")
                + "Fix the sheet, or subset the matrix to the samples you mean."
            )
        extra = [s for s in self.labels if s not in set(names.tolist())]
        if extra and not allow_extra:
            raise ValueError(
                f"{len(extra)} sample(s) in the sample metadata are not columns of the "
                f"count matrix: {extra[:8]}"
                + (f" and {len(extra) - 8} more. " if len(extra) > 8 else ". ")
                + "Pass allow_extra=True if the sheet legitimately covers other runs."
            )
        return np.array([self.labels[s] for s in names.tolist()], dtype=int)


def condition(
    sample_meta,
    *,
    key,
    column: str,
    case,
    control,
    drop_other: bool = False,
) -> Condition:
    """Read a case/control assignment out of a sample metadata table.

    ``sample_meta`` is a duck-typed DataFrame or a mapping of column name to
    sequence — whatever your reader produced. ``key`` names the column holding
    sample identifiers, which must match the count matrix's sample names;
    ``column`` names the condition; ``case`` and ``control`` are the two values
    within it.

    A third level in ``column`` **raises**, naming it, unless
    ``drop_other=True``: quietly discarding a group is quietly running a
    different contrast. Duplicate sample ids always raise.
    """
    if _is_frame(sample_meta):
        cols = list(sample_meta.columns)
        for name in (key, column):
            if name not in cols:
                raise KeyError(f"no column {name!r} in the sample metadata; it has {cols}")
        ids = np.asarray(_column(sample_meta, key), dtype=object)
        values = np.asarray(_column(sample_meta, column), dtype=object)
    else:
        try:
            ids = np.asarray(list(sample_meta[key]), dtype=object)
            values = np.asarray(list(sample_meta[column]), dtype=object)
        except (KeyError, TypeError) as e:
            raise TypeError(
                "sample_meta must be a DataFrame (polars/pandas) or a mapping of "
                f"column name to sequence; {e}"
            ) from e
    if ids.shape != values.shape:
        raise ValueError(f"columns {key!r} and {column!r} have different lengths")

    dup = _duplicates(ids)
    if dup:
        raise ValueError(f"duplicate sample ids in the sample metadata: {dup[:8]}")

    is_case = values == case
    is_ctrl = values == control
    if case == control:
        raise ValueError(f"case and control are the same value ({case!r})")
    other = ~(is_case | is_ctrl)
    if other.any() and not drop_other:
        levels = sorted({str(v) for v in values[other].tolist()})
        raise ValueError(
            f"column {column!r} has {int(other.sum())} sample(s) at "
            f"{len(levels)} other level(s) than {case!r} and {control!r}: {levels[:8]}. "
            f"Pass drop_other=True to exclude them, or name the two levels you mean."
        )
    if not is_case.any() or not is_ctrl.any():
        raise ValueError(
            f"column {column!r} must contain both {case!r} ({int(is_case.sum())} samples) "
            f"and {control!r} ({int(is_ctrl.sum())} samples)"
        )

    labels = {s: int(c) for s, c in zip(ids[is_case | is_ctrl].tolist(),
                                        is_case[is_case | is_ctrl].tolist())}
    return Condition(labels=labels, case=case, control=control, column=column,
                     dropped=tuple(ids[other].tolist()))


# ---------------------------------------------------------------------------
# Results out


#: The written column order. ``neglog10_p_*`` is what a volcano plots; ``p_*``
#: and ``padj_*`` stay beside it because significance is read off BH, not off
#: the raw p-value.
RESULT_COLUMNS = (
    "gene", "case_mean", "ctrl_mean", "mean_shift", "log2_fc",
    "p_mean_shift", "padj_mean_shift", "neglog10_p_mean_shift",
    "subset_stat", "p_subset", "padj_subset", "neglog10_p_subset",
    "affected_fraction", "direction", "w1",
)


def _neglog10(p):
    p = np.asarray(p, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log10(p)


def result_columns(res) -> dict:
    """The result as a plain dict of arrays, in :data:`RESULT_COLUMNS` order.

    Stage-2 columns are omitted when the subset stage did not run; bootstrap
    interval columns are appended when ``n_boot`` produced them.
    """
    cols = {
        "gene": res.gene,
        "case_mean": res.case_mean,
        "ctrl_mean": res.ctrl_mean,
        "mean_shift": res.mean_shift,
        "log2_fc": res.log2_fc,
        "p_mean_shift": res.p_mean_shift,
        "padj_mean_shift": res.padj_mean_shift,
        "neglog10_p_mean_shift": _neglog10(res.p_mean_shift),
    }
    if res.subset is not None:
        cols.update({
            "subset_stat": res.subset.statistic,
            "p_subset": res.p_subset,
            "padj_subset": res.padj_subset,
            "neglog10_p_subset": _neglog10(res.p_subset),
            "affected_fraction": res.affected_fraction,
            "direction": res.direction,
        })
    cols["w1"] = res.w1
    ordered = {k: cols[k] for k in RESULT_COLUMNS if k in cols}
    for name, ci in (("affected_fraction", res.ci_affected_fraction),
                     ("direction", res.ci_direction), ("log2_fc", res.ci_log2_fc)):
        if ci is not None:
            ordered[f"{name}_lo"] = ci[0]
            ordered[f"{name}_hi"] = ci[1]
    return ordered


def to_frame(res):
    """The result as a polars DataFrame, in :data:`RESULT_COLUMNS` order."""
    import polars as pl

    return pl.DataFrame(result_columns(res))


def manifest(res, *, alpha: float = 0.05) -> dict:
    """What produced this result — parameters, shapes, versions, timestamp.

    The timestamp makes the manifest deliberately **not** byte-reproducible.
    Provenance is what it is for; the result table beside it is reproducible
    from the seed.
    """
    from datetime import datetime, timezone

    from . import __version__

    p = dict(res.params)
    out = {
        "wade_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "numpy_version": np.__version__,
        "run": {
            "nperms": p.get("nperms"),
            "seed": p.get("seed"),
            "alternative": p.get("alternative"),
            "correction": p.get("correction"),
            "n_exc_min": p.get("n_exc_min"),
            "n_tail": p.get("n_tail"),
            "n_boot": p.get("n_boot"),
            # The two opt-in fast paths change reported numbers, so a
            # manifest without them could not reproduce the run; gene_chunk
            # is recorded for completeness even though it is bit-identical.
            "stage1": p.get("stage1", "grid"),
            "fit_backend": p.get("fit_backend", "numpy"),
            "gene_chunk": p.get("gene_chunk"),
            "pseudocount": None if res.pseudocount is None else float(np.median(res.pseudocount)),
        },
        "design": {
            "n_genes": int(res.gene.shape[0]),
            "n_case": p.get("n1"),
            "n_ctrl": p.get("n0"),
            "nprobs": p.get("nprobs"),
            "max_probs": p.get("max_probs"),
            "case_label": p.get("case_label"),
            "control_label": p.get("control_label"),
            "condition_column": p.get("condition_column"),
        },
        "results": {
            "alpha": alpha,
            "n_significant_mean_shift": int(np.sum(np.nan_to_num(res.padj_mean_shift, nan=np.inf) <= alpha)),
        },
    }
    if res.padj_subset is not None:
        out["results"]["n_significant_subset"] = int(
            np.sum(np.nan_to_num(res.padj_subset, nan=np.inf) <= alpha))
    try:
        import polars as pl
        out["polars_version"] = pl.__version__
    except ImportError:                                     # pragma: no cover
        pass
    return out


_manifest = manifest


_WRITERS = {
    ".tsv": lambda df, p: df.write_csv(p, separator="\t"),
    ".txt": lambda df, p: df.write_csv(p, separator="\t"),
    ".csv": lambda df, p: df.write_csv(p),
    ".parquet": lambda df, p: df.write_parquet(p),
    ".arrow": lambda df, p: df.write_ipc(p),
    ".ipc": lambda df, p: df.write_ipc(p),
    ".feather": lambda df, p: df.write_ipc(p),
}


def write_results(res, path, *, manifest: bool = True, alpha: float = 0.05) -> Path:
    """Write the result table, and beside it a JSON manifest of the run.

    Format follows the extension: ``.tsv`` / ``.txt``, ``.csv``, ``.parquet``,
    ``.arrow`` / ``.ipc`` / ``.feather``. The manifest is written to
    ``<stem>.manifest.json`` unless ``manifest=False``.
    """
    path = Path(path)
    writer = _WRITERS.get(path.suffix.lower())
    if writer is None:
        raise ValueError(
            f"unknown result format {path.suffix!r}; use one of "
            f"{sorted(_WRITERS)}. (WADE writes its own results but does not read "
            f"data files — see ROADMAP.md §1.)"
        )
    frame = to_frame(res)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer(frame, path)
    if manifest:
        man = _manifest(res, alpha=alpha)
        path.with_suffix("").with_suffix(".manifest.json").write_text(
            json.dumps(man, indent=2, sort_keys=False) + "\n")
    return path
