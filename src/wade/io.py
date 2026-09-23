"""Data in, results out.

**WADE reads no files.** polars and pandas already read CSV, TSV, Parquet,
Arrow, Excel and gzip better than this package would. What this module does
instead:

* :func:`as_counts` — accept a count matrix in whatever shape it arrives in
  (NumPy array, polars or pandas DataFrame, any sparse matrix with
  ``.toarray()``) and turn it into the one shape the statistic uses: a dense
  ``genes x samples`` float64 array with gene and sample labels beside it.
  Labels are optional; the orientation is never guessed. Other per-gene
  columns are carried in ``Counts.meta`` and read by no statistic.
* :func:`condition` — turn a sample metadata table into a case/control
  assignment and align it to the matrix's samples *strictly*: a silently
  dropped sample is a silently different analysis, and because library sizes
  are computed on the samples handed in, a silently different normalization.
* :func:`write_results` and :func:`manifest` — the results, plus a JSON
  record of what produced them.

Neither polars nor pandas is a dependency: frames are duck-typed on
``.columns`` and ``.to_numpy()``, and only writing results needs polars,
imported inside the call.
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
    #: Every non-sample column of an input frame, per gene — a symbol, a
    #: biotype, a length. Read by no statistic; figures can show it and
    #: ``normalizer="Length"`` is looked up here.
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    def normalizer(self, column: str) -> np.ndarray:
        """The per-gene **numeric** vector held in ``meta`` under ``column``.

        ``meta`` carries every per-gene column the input had, including string
        ones (a symbol, a biotype); only a numeric column can be a normalizer,
        so naming a non-numeric one raises here rather than failing later
        inside the arithmetic.
        """
        if column not in self.meta:
            raise KeyError(
                f"no column {column!r} to use as a normalizer; the matrix carries "
                f"{sorted(self.meta) or 'no metadata columns'}. Pass the values as an "
                f"array instead, or name a column that is in the counts frame."
            )
        col = np.asarray(self.meta[column])
        if not _numeric(col):
            raise TypeError(
                f"column {column!r} is not numeric, so it cannot be a normalizer; "
                f"it is carried as per-gene metadata instead."
            )
        return col


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


def _labelled_index(frame):
    """A pandas index carrying gene ids (``read_csv(..., index_col=0)``), or
    ``None`` for a default positional index and for polars."""
    index = getattr(frame, "index", None)
    if index is None or type(index).__name__ == "RangeIndex":
        return None
    values = np.asarray(index, dtype=object)
    return None if _numeric(values) else values


def _from_frame(frame, id_column, sample_columns, exclude=()):
    """(values, gene_names, sample_names, meta) from a duck-typed DataFrame."""
    cols = list(frame.columns)
    if not cols:
        raise ValueError("the counts frame has no columns")

    indexed = None
    if id_column == "auto":
        indexed = _labelled_index(frame)
        first = _column(frame, cols[0])
        id_column = cols[0] if indexed is None and not _numeric(first) else None
    elif isinstance(id_column, (int, np.integer)) and not isinstance(id_column, bool):
        id_column = cols[int(id_column)]
    if id_column is not None and id_column not in cols:
        raise KeyError(f"no column {id_column!r} in the counts frame; it has {cols}")

    gene_names = (np.asarray(_column(frame, id_column), dtype=object)
                  if id_column is not None else indexed)

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

    values = np.asarray(frame[sample_columns].to_numpy(), dtype=np.float64)
    # Every leftover column is carried, string ones included: a symbol is what
    # makes a ranked table readable. Only a numeric one can be a normalizer.
    meta = {c: _column(frame, c) for c in rest if c not in sample_columns}
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
        For a frame: ``"auto"`` takes a labelled pandas index if there is one
        (``read_csv(..., index_col=0)``), else the first column when it is
        non-numeric; a name or integer position pins one; ``None`` means the
        frame is all samples and gene names are positional. **An integer id
        column (Entrez ids) is numeric and must be named**, or it is read as
        a sample.
    sample_columns
        For a frame: which columns hold counts. By default every column after
        the id column, so **every numeric column is a sample** unless it is
        named in ``exclude`` or left out of ``sample_columns``; a non-numeric
        leftover raises rather than being read as a sample. Columns not in
        ``sample_columns`` become :attr:`Counts.meta`.
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
        if gene_names is not None or sample_names is not None or genes != "rows":
            raise ValueError("Counts already carries labels and orientation; change them explicitly before analysis")
        if obj.values.ndim != 2:
            raise ValueError("Counts.values must be genes x samples")
        for labels, size in ((obj.gene_names, obj.shape[0]), (obj.sample_names, obj.shape[1])):
            if np.asarray(labels).shape != (size,) or _duplicates(np.asarray(labels)):
                raise ValueError("Counts identifiers must be unique and match the matrix shape")
        if any(np.asarray(v).shape != (obj.shape[0],) for v in obj.meta.values()):
            raise ValueError("Counts metadata must contain one value per gene")
        _locate_bad_counts(obj.values, obj.gene_names, obj.sample_names)
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
                f"{label}_names must have one entry per {label}: expected shape "
                f"({want},), got {arr.shape}"
            )
        dup = _duplicates(arr)
        if dup:
            raise ValueError(f"duplicate {label} names: {dup[:8]}"
                             + (f" and {len(dup) - 8} more" if len(dup) > 8 else ""))

    _locate_bad_counts(values, found_genes, found_samples)
    return Counts(values=values, gene_names=found_genes, sample_names=found_samples,
                  meta={k: np.asarray(v) for k, v in meta.items()})


def _locate_bad_counts(values, gene_names, sample_names) -> None:
    """Refuse a non-finite or negative count, **and say which cell**.

    A blank in a large counts file is the ordinary cause, and "the matrix
    contains non-finite values" leaves the caller to find it. There is no
    handling policy on offer because there is no good default: a missing count
    is not a zero (a zero is an observation), so substituting one would invent
    data. Naming the cell is the help WADE can honestly give.
    """
    bad = ~np.isfinite(values)
    what = "non-finite (NaN or inf)"
    if not bad.any():
        bad = values < 0
        what = "negative"
        if not bad.any():
            return
    rows, cols = np.nonzero(bad)
    where = ", ".join(f"{gene_names[i]}/{sample_names[j]}"
                      for i, j in zip(rows[:5].tolist(), cols[:5].tolist()))
    raise ValueError(
        f"{rows.size} count(s) are {what}: {where}"
        + (f" and {rows.size - 5} more." if rows.size > 5 else ".")
        + " A missing count is not a zero — a zero is an observation — so WADE "
          "will not substitute one. Drop the gene, drop the sample, or decide "
          "what the value should be."
    )


def _duplicates(arr: np.ndarray) -> list:
    seen, duplicates = set(), []
    for value in arr.tolist():
        if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
            raise ValueError("identifiers cannot be missing or non-finite")
        try:
            if value in seen:
                duplicates.append(value)
            seen.add(value)
        except TypeError as e:
            raise ValueError("identifiers must be hashable") from e
    return duplicates


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
        have = set(names.tolist())
        extra = [s for s in self.labels if s not in have]
        if extra and not allow_extra:
            raise ValueError(
                f"{len(extra)} sample(s) in the sample metadata are not columns of the "
                f"count matrix: {extra[:8]}"
                + (f" and {len(extra) - 8} more. " if len(extra) > 8 else ". ")
                + "If the sheet legitimately covers other runs, resolve it yourself "
                  "and pass the vector: cond.vector(counts.sample_names, "
                  "allow_extra=True)."
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

    if case == control:
        raise ValueError(f"case and control are the same value ({case!r})")
    is_case = values == case
    is_ctrl = values == control
    other = ~(is_case | is_ctrl)
    if other.any() and not drop_other:
        levels = sorted({repr(v) for v in values[other].tolist()})

        raise ValueError(
            f"column {column!r} has {int(other.sum())} sample(s) at "
            f"{len(levels)} other level(s) than {case!r} and {control!r}: "
            f"[{', '.join(levels[:8])}]. Pass drop_other=True to exclude them, "
            f"or name the two levels you mean."
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


RESULT_COLUMNS = (
    "gene", "status", "statistic", "pvalue", "padj", "bulk_shift", "upper_tail", "lower_tail",
    "n_permutations", "n_exceedances",
)


def result_columns(res):
    return {name: getattr(res, name) for name in RESULT_COLUMNS}


def manifest(res, *, alpha=0.05):
    """The complete tested family, preprocessing, sample design and MC records."""
    from datetime import datetime, timezone
    from . import __version__
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    genes = res.gene.tolist()
    references = set(res.normalization.used_reference_genes)
    tested = res.tested
    return dict(
        wade_version=__version__, created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        numpy_version=np.__version__, run=dict(res.params), normalization=res.normalization.to_dict(),
        design=dict(genes=genes, sample_names=res.sample_names.tolist(), condition=res.cond.tolist(),
                    n_case=int(res.cond.sum()), n_control=int((res.cond == 0).sum()),
                    n_tested=int(tested.sum()), status=res.status.tolist(),
                    tested_reference_genes=[g for g, keep in zip(genes, tested)
                                            if keep and g in references]),
        resampling=dict(n_permutations=res.n_permutations.tolist(), n_exceedances=res.n_exceedances.tolist(),
                        rule="h/T at the h-th inclusive exceedance; otherwise (K+1)/(B+1) at cap B"),
        results=dict(alpha=alpha, n_significant=int(np.sum(res.padj <= alpha)),
                     minimum_discovery_rank=int(np.ceil(tested.sum() * res.params["minimum_pvalue"] / alpha))))


_WRITERS = {
    ".tsv": lambda df, p: df.write_csv(p, separator="\t"),
    ".txt": lambda df, p: df.write_csv(p, separator="\t"),
    ".csv": lambda df, p: df.write_csv(p),
    ".parquet": lambda df, p: df.write_parquet(p),
    ".arrow": lambda df, p: df.write_ipc(p),
    ".ipc": lambda df, p: df.write_ipc(p),
    ".feather": lambda df, p: df.write_ipc(p),
}


def _manifest_path(path: Path) -> Path:
    """``<stem>.manifest.json`` beside the table; ``stem`` keeps every dotted
    component, so ``plasma.v1.tsv`` and ``plasma.v2.tsv`` get their own."""
    return path.with_name(path.stem + ".manifest.json")


def write_results(res, path, *, alpha: float = 0.05) -> Path:
    """Write the result table, and beside it a JSON manifest of the run.

    Format follows the extension: ``.tsv`` / ``.txt``, ``.csv``, ``.parquet``,
    ``.arrow`` / ``.ipc`` / ``.feather``. The manifest is written to
    ``<stem>.manifest.json``; ``alpha`` sets the significance counts it
    records.
    """
    path = Path(path)
    writer = _WRITERS.get(path.suffix.lower())
    if writer is None:
        raise ValueError(
            f"unknown result format {path.suffix!r}; use one of {sorted(_WRITERS)}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    writer(res.to_frame(), path)
    _manifest_path(path).write_text(json.dumps(manifest(res, alpha=alpha), indent=2) + "\n")
    return path
