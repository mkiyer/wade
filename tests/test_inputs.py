"""Data in, results out — :mod:`wade.io`.

WADE reads no files, so there is nothing here about parsing. What is tested is
the boundary it *does* own: accepting a count matrix in the shapes it arrives
in, refusing input it cannot safely interpret, and aligning a sample sheet by
name without ever dropping a sample quietly. Writing results is tested in
test_results.py.

polars is used to build the frames because it is what the environment has; the
frame support itself is duck-typed and imports neither polars nor pandas.
"""

from __future__ import annotations

import numpy as np
import pytest

from wade.io import Condition, Counts, as_counts, condition

pl = pytest.importorskip("polars")

G, N = 24, 12
SAMPLES = [f"S{j}" for j in range(N)]
GENES = [f"g{i}" for i in range(G)]


@pytest.fixture(scope="module")
def matrix():
    rng = np.random.default_rng(4)
    return rng.poisson(40, size=(G, N)).astype(float)


@pytest.fixture(scope="module")
def frame(matrix):
    return pl.DataFrame({"gene_id": GENES, **{s: matrix[:, j] for j, s in enumerate(SAMPLES)}})


@pytest.fixture(scope="module")
def sheet():
    return pl.DataFrame({"sample_id": SAMPLES,
                         "condition": ["tumor"] * (N // 2) + ["normal"] * (N // 2),
                         "batch": ["a", "b"] * (N // 2)})


# ---------------------------------------------------------------------------
# as_counts


def test_as_counts_from_an_array_synthesizes_positional_labels(matrix):
    c = as_counts(matrix)
    assert isinstance(c, Counts) and c.shape == (G, N)
    assert c.gene_names[0] == "gene0" and c.sample_names[0] == "sample0"
    assert c.values.dtype == np.float64
    np.testing.assert_array_equal(c.values, matrix)
    again = as_counts(c)                           # a Counts is checked like any input, and kept as it is
    assert again.values is c.values and list(again.gene_names) == list(c.gene_names)


def test_as_counts_from_a_frame_takes_labels_from_it(frame, matrix):
    c = as_counts(frame)
    assert c.shape == (G, N)
    assert list(c.gene_names) == GENES
    assert list(c.sample_names) == SAMPLES
    np.testing.assert_allclose(c.values, matrix)


def test_as_counts_id_column_auto_name_index_or_none(frame, matrix):
    assert list(as_counts(frame, id_column="gene_id").gene_names) == GENES
    assert list(as_counts(frame, id_column=0).gene_names) == GENES
    # No id column: the first column becomes a sample and names are positional.
    c = as_counts(frame, id_column=None, sample_columns=SAMPLES)
    assert c.gene_names[0] == "gene0" and list(c.sample_names) == SAMPLES
    # A frame that is all numeric has no id column to find.
    plain = pl.DataFrame({s: matrix[:, j] for j, s in enumerate(SAMPLES)})
    assert as_counts(plain).gene_names[0] == "gene0"
    with pytest.raises(KeyError, match="no column 'nope'"):
        as_counts(frame, id_column="nope")


def test_as_counts_refuses_a_non_numeric_leftover(matrix):
    """The featureCounts layout: Chr/Strand must not be read as expression, nor a
    Boolean flag; once ``sample_columns`` says which columns are counts, the
    others are not read."""
    fc = pl.DataFrame({
        "Geneid": GENES, "Chr": ["chr1"] * G, "Start": np.arange(G),
        "Strand": ["+"] * G, "Length": np.arange(100, 100 + G, dtype=float),
        **{s: matrix[:, j] for j, s in enumerate(SAMPLES)},
    })
    with pytest.raises(ValueError, match=r"not numeric.*sample_columns.*featureCounts"):
        as_counts(fc)
    c = as_counts(fc, sample_columns=SAMPLES)
    assert c.shape == (G, N) and list(c.sample_names) == SAMPLES
    flagged = pl.DataFrame({"gene_id": GENES, "coding": [True] * G,
                            **{s: matrix[:, j] for j, s in enumerate(SAMPLES)}})
    with pytest.raises(ValueError, match=r"\['coding'\] are not numeric"):
        as_counts(flagged)
    with pytest.raises(KeyError, match="sample_columns not in"):
        as_counts(fc, sample_columns=["S0", "nope"])


def test_as_counts_reads_a_pandas_frame_with_a_labelled_index(matrix):
    pd = pytest.importorskip("pandas")
    c = as_counts(pd.DataFrame(matrix, index=GENES, columns=SAMPLES))
    assert list(c.gene_names) == GENES and list(c.sample_names) == SAMPLES
    np.testing.assert_array_equal(c.values, matrix)


def test_as_counts_transposes_when_genes_are_columns(matrix):
    t = pl.DataFrame({"sample_id": SAMPLES,
                      **{g: matrix[i, :] for i, g in enumerate(GENES)}})
    c = as_counts(t, genes="columns")
    assert c.shape == (G, N)
    assert list(c.gene_names) == GENES and list(c.sample_names) == SAMPLES
    np.testing.assert_allclose(c.values, matrix)
    with pytest.raises(ValueError, match="genes must be"):
        as_counts(matrix, genes="down")


def test_as_counts_accepts_a_sparse_matrix():
    sp = pytest.importorskip("scipy.sparse")
    dense = np.zeros((5, 4)); dense[1, 2] = 7.0; dense[4, 0] = 3.0
    c = as_counts(sp.csr_matrix(dense))
    assert c.shape == (5, 4)
    np.testing.assert_array_equal(c.values, dense)


def test_as_counts_validates_labels_and_values(matrix):
    with pytest.raises(ValueError, match="duplicate gene names"):
        as_counts(matrix, gene_names=["x"] * G)
    with pytest.raises(ValueError, match="duplicate sample names"):
        as_counts(matrix, sample_names=["x"] * N)
    with pytest.raises(ValueError, match="one entry per gene"):
        as_counts(matrix, gene_names=GENES[:-1])
    # A bad count is located, not merely reported: a blank in a big file is
    # the ordinary cause and "contains non-finite values" leaves you hunting.
    # ... and no value is substituted, because a missing count is not a zero.
    with pytest.raises(ValueError, match=r"non-finite.*gene0/sample1.*not a zero"):
        as_counts(np.array([[1.0, np.nan]]))
    with pytest.raises(ValueError, match=r"negative.*g1/S0"):
        as_counts(np.array([[1.0, 2.0], [-2.0, 3.0]]),
                  gene_names=["g0", "g1"], sample_names=["S0", "S1"])
    with pytest.raises(ValueError, match="2-D"):
        as_counts(np.arange(5.0))


# ---------------------------------------------------------------------------
# condition


def test_condition_reads_the_two_levels_and_records_them(sheet):
    c = condition(sheet, key="sample_id", column="condition", case="tumor", control="normal")
    assert isinstance(c, Condition) and len(c.labels) == N
    assert c.case == "tumor" and c.control == "normal" and c.column == "condition"
    np.testing.assert_array_equal(c.vector(SAMPLES), [1] * (N // 2) + [0] * (N // 2))
    # Order follows the samples asked for, not the sheet.
    np.testing.assert_array_equal(c.vector(SAMPLES[::-1]), [0] * (N // 2) + [1] * (N // 2))


def test_condition_accepts_a_plain_mapping_too():
    c = condition({"id": ["a", "b"], "grp": ["x", "y"]}, key="id", column="grp",
                  case="x", control="y")
    np.testing.assert_array_equal(c.vector(["b", "a"]), [0, 1])
    with pytest.raises(TypeError, match="DataFrame"):
        condition(object(), key="id", column="grp", case="x", control="y")


def test_condition_refuses_a_third_level_unless_asked(sheet):
    three = sheet.with_columns(
        pl.when(pl.col("sample_id") == "S0").then(pl.lit("adjacent"))
        .otherwise(pl.col("condition")).alias("condition"))
    with pytest.raises(ValueError, match=r"other level.*adjacent"):
        condition(three, key="sample_id", column="condition", case="tumor", control="normal")
    c = condition(three, key="sample_id", column="condition", case="tumor",
                  control="normal", drop_other=True)
    assert c.dropped == ("S0",) and len(c.labels) == N - 1
    # ... and the dropped sample is then reported as unassigned, not ignored.
    with pytest.raises(ValueError, match="no entry in the sample metadata"):
        c.vector(SAMPLES)
    np.testing.assert_array_equal(c.vector(SAMPLES[1:]), [1] * (N // 2 - 1) + [0] * (N // 2))


def test_condition_validates_its_inputs(sheet):
    with pytest.raises(KeyError, match="no column 'nope'"):
        condition(sheet, key="nope", column="condition", case="tumor", control="normal")
    # A named level that no sample carries: reachable once the other samples
    # are explicitly dropped (otherwise the third-level check fires first).
    with pytest.raises(ValueError, match=r"other level.*normal"):
        condition(sheet, key="sample_id", column="condition", case="tumor", control="absent")
    with pytest.raises(ValueError, match="must contain both"):
        condition(sheet, key="sample_id", column="condition", case="tumor",
                  control="absent", drop_other=True)
    with pytest.raises(ValueError, match="same value"):
        condition(sheet, key="sample_id", column="condition", case="tumor", control="tumor")
    dup = pl.DataFrame({"sample_id": ["S0", "S0"], "condition": ["tumor", "normal"]})
    with pytest.raises(ValueError, match="duplicate sample ids"):
        condition(dup, key="sample_id", column="condition", case="tumor", control="normal")


def test_condition_alignment_is_strict_in_both_directions(sheet):
    c = condition(sheet, key="sample_id", column="condition", case="tumor", control="normal")
    with pytest.raises(ValueError, match="no entry in the sample metadata"):
        c.vector(SAMPLES + ["S99"])
    with pytest.raises(ValueError, match="not columns of the count matrix"):
        c.vector(SAMPLES[:-1])
    np.testing.assert_array_equal(c.vector(SAMPLES[:-1], allow_extra=True),
                                  [1] * (N // 2) + [0] * (N // 2 - 1))


def test_condition_refuses_missing_strata_in_every_form():
    pd = pytest.importorskip("pandas")
    for missing in (None, float("nan"), np.float32("nan"), pd.NA):
        sheet = {"id": list("abcd"), "group": ["x", "x", "y", "y"], "batch": ["p", missing, "p", "q"]}
        with pytest.raises(ValueError, match="no value"):
            condition(sheet, key="id", column="group", case="x", control="y", strata="batch")
