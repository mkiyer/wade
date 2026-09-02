"""Data in, results out — ``ROADMAP.md`` §1, :mod:`wade.io`.

WADE reads no files, so there is nothing here about parsing. What is tested is
the boundary it *does* own: accepting a count matrix in the shapes it arrives
in, refusing input it cannot safely interpret, aligning a sample sheet by name
without ever dropping a sample quietly, and writing the result with a manifest.

polars is used to build the frames because it is what the environment has; the
frame support itself is duck-typed and imports neither polars nor pandas.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import wade
from wade.io import RESULT_COLUMNS, Condition, Counts, as_counts, condition

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
# The optional-dependency contract


def test_importing_wade_imports_neither_frame_library():
    """The statistic keeps its NumPy-only surface: polars is imported inside
    to_frame/write_results, and frames are duck-typed, never imported."""
    import subprocess
    import sys

    code = ("import sys, wade; "
            "bad = [m for m in ('polars', 'pandas') if m in sys.modules]; "
            "sys.exit(1 if bad else 0)")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# as_counts


def test_as_counts_from_an_array_synthesizes_positional_labels(matrix):
    c = as_counts(matrix)
    assert isinstance(c, Counts) and c.shape == (G, N)
    assert c.gene_names[0] == "gene0" and c.sample_names[0] == "sample0"
    assert c.values.dtype == np.float64 and c.meta == {}
    np.testing.assert_array_equal(c.values, matrix)
    assert as_counts(c) is c                       # idempotent


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


def test_as_counts_refuses_a_non_numeric_leftover_and_carries_every_column(matrix):
    """The featureCounts layout: Chr/Strand must not be read as expression, and
    Length must land where the normalizer is looked up.

    Once ``sample_columns`` says which columns are counts, **every** leftover
    per-gene column is carried — string ones included, because a symbol is what
    makes a ranked table readable. No statistic reads any of it; only a numeric
    column can serve as the normalizer.
    """
    fc = pl.DataFrame({
        "Geneid": GENES, "Chr": ["chr1"] * G, "Start": np.arange(G),
        "Strand": ["+"] * G, "Length": np.arange(100, 100 + G, dtype=float),
        **{s: matrix[:, j] for j, s in enumerate(SAMPLES)},
    })
    with pytest.raises(ValueError, match=r"not numeric.*featureCounts"):
        as_counts(fc)
    c = as_counts(fc, sample_columns=SAMPLES)
    assert c.shape == (G, N)
    assert set(c.meta) == {"Chr", "Start", "Strand", "Length"}   # all of them
    assert list(c.meta["Chr"][:2]) == ["chr1", "chr1"]
    with pytest.raises(TypeError, match="not numeric"):
        c.normalizer("Chr")
    np.testing.assert_allclose(c.normalizer("Length"), np.arange(100, 100 + G))
    with pytest.raises(KeyError, match="no column 'Width'"):
        c.normalizer("Width")
    with pytest.raises(KeyError, match="sample_columns not in"):
        as_counts(fc, sample_columns=["S0", "nope"])


def test_as_counts_transposes_when_genes_are_columns(matrix):
    t = pl.DataFrame({"sample_id": SAMPLES,
                      **{g: matrix[i, :] for i, g in enumerate(GENES)}})
    c = as_counts(t, genes="columns")
    assert c.shape == (G, N)
    assert list(c.gene_names) == GENES and list(c.sample_names) == SAMPLES
    np.testing.assert_allclose(c.values, matrix)
    # The leftover columns of a samples-as-rows frame describe samples, not
    # genes, so they are not offered as a gene normalizer.
    assert c.meta == {}
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
    with pytest.raises(ValueError, match=r"non-finite.*gene0/sample1"):
        as_counts(np.array([[1.0, np.nan]]))
    with pytest.raises(ValueError, match=r"negative.*g1/S0"):
        as_counts(np.array([[1.0, 2.0], [-2.0, 3.0]]),
                  gene_names=["g0", "g1"], sample_names=["S0", "S1"])
    # ... and no value is substituted, because a missing count is not a zero
    with pytest.raises(ValueError, match="not a zero"):
        as_counts(np.array([[1.0, np.nan]]))
    with pytest.raises(ValueError, match="2-D"):
        as_counts(np.arange(5.0))


# ---------------------------------------------------------------------------
# condition


def test_condition_reads_the_two_levels_and_records_them(sheet):
    c = condition(sheet, key="sample_id", column="condition", case="tumor", control="normal")
    assert isinstance(c, Condition) and len(c) == N
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
    assert c.dropped == ("S0",) and len(c) == N - 1
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


# ---------------------------------------------------------------------------
# Through wade()


def test_wade_accepts_a_frame_a_condition_and_a_named_normalizer(matrix):
    fc = pl.DataFrame({"Geneid": GENES, "Length": np.full(G, 2.0),
                       **{s: matrix[:, j] for j, s in enumerate(SAMPLES)}})
    sheet = pl.DataFrame({"sample_id": SAMPLES,
                          "condition": ["tumor"] * (N // 2) + ["normal"] * (N // 2)})
    cond = condition(sheet, key="sample_id", column="condition",
                     case="tumor", control="normal")
    # Naming a column as the normalizer keeps it out of the samples, so an
    # (id, Length, samples) frame is a one-call run.
    res = wade.wade(fc, "Length", cond, nperms=50, seed=1)
    assert res.tpm.shape == (G, N)
    assert list(res.gene) == GENES
    assert list(res.sample_names) == SAMPLES
    assert res.params["case_label"] == "tumor" and res.params["control_label"] == "normal"
    assert res.params["condition_column"] == "condition"
    # A full featureCounts frame still needs as_counts, because of Chr/Strand.
    full = fc.with_columns(pl.lit("chr1").alias("Chr"))
    with pytest.raises(ValueError, match="not numeric"):
        wade.wade(full, "Length", cond, nperms=10)
    named = wade.wade(as_counts(full, sample_columns=SAMPLES), "Length", cond,
                      nperms=50, seed=1)
    np.testing.assert_array_equal(named.p_mean_shift, res.p_mean_shift)

    # Identical to the array path with the same inputs spelled out.
    plain = wade.wade(matrix, np.full(G, 2.0),
                      np.r_[np.ones(N // 2, int), np.zeros(N // 2, int)], nperms=50, seed=1)
    np.testing.assert_array_equal(res.p_mean_shift, plain.p_mean_shift)


def test_wade_accepts_a_scalar_normalizer(matrix):
    cond = np.r_[np.ones(N // 2, int), np.zeros(N // 2, int)]
    a = wade.wade(matrix, 4.0, cond, nperms=50, seed=1)
    b = wade.wade(matrix, np.full(G, 4.0), cond, nperms=50, seed=1)
    np.testing.assert_array_equal(a.p_mean_shift, b.p_mean_shift)


def test_wade_contrast_by_name_and_by_index(frame, matrix):
    case, ctrl = SAMPLES[:5], SAMPLES[6:11]
    by_name = wade.wade_contrast(frame, 1.0, case, ctrl, nperms=50, seed=1)
    assert by_name.params["n_case"] == 5 and by_name.params["n_ctrl"] == 5
    assert list(by_name.sample_names) == case + ctrl
    by_index = wade.wade_contrast(matrix, 1.0, list(range(5)), list(range(6, 11)),
                                  nperms=50, seed=1)
    np.testing.assert_array_equal(by_name.p_mean_shift, by_index.p_mean_shift)


def test_wade_contrast_refuses_unknown_names_and_overlaps(frame):
    with pytest.raises(KeyError, match="not samples of this matrix"):
        wade.wade_contrast(frame, 1.0, ["S0", "nope"], ["S5"], nperms=10)
    with pytest.raises(ValueError, match="both groups"):
        wade.wade_contrast(frame, 1.0, ["S0", "S1"], ["S1", "S2"], nperms=10)
    with pytest.raises(ValueError, match="is empty"):
        wade.wade_contrast(frame, 1.0, [], ["S1"], nperms=10)


def test_sample_names_that_are_integers_still_resolve_by_name(matrix):
    """A matrix whose samples are literally named 0..n-1 must not be read as
    positional selection."""
    named = as_counts(matrix, sample_names=list(range(N)))
    res = wade.wade_contrast(named, 1.0, list(range(5)), list(range(6, 11)), nperms=20, seed=1)
    assert list(res.sample_names) == list(range(5)) + list(range(6, 11))


# ---------------------------------------------------------------------------
# Results out


@pytest.fixture(scope="module")
def result(matrix):
    cond = np.r_[np.ones(N // 2, int), np.zeros(N // 2, int)]
    return wade.wade(matrix, np.full(G, 2.0), cond, nperms=100, seed=1,
                     gene_names=GENES, sample_names=SAMPLES, n_boot=20)


def test_report_columns_are_in_the_agreed_order_and_neglog10_is_right(result):
    cols = list(result.report())
    assert cols[:len(RESULT_COLUMNS)] == list(RESULT_COLUMNS)
    assert cols[len(RESULT_COLUMNS):] == [
        "affected_fraction_lo", "affected_fraction_hi",
        "direction_lo", "direction_hi",
        "subset_log2_fc_lo", "subset_log2_fc_hi",
        "log2_fc_lo", "log2_fc_hi",
        "mean_shift_lo", "mean_shift_hi"]
    rep = result.report()
    np.testing.assert_allclose(rep["neglog10_p_mean_shift"], -np.log10(result.p_mean_shift))
    np.testing.assert_allclose(rep["neglog10_p_subset"], -np.log10(result.p_subset))
    assert result.to_frame().columns == cols


def test_report_omits_the_subset_stage_when_it_did_not_run(matrix):
    cond = np.r_[np.ones(N // 2, int), np.zeros(N // 2, int)]
    res = wade.wade(matrix, 1.0, cond, nperms=50, subset=False, seed=1)
    cols = list(res.report())
    assert "p_subset" not in cols and "affected_fraction" not in cols
    assert cols[:5] == list(RESULT_COLUMNS[:5])


@pytest.mark.parametrize("ext", [".tsv", ".csv", ".parquet", ".arrow"])
def test_write_results_round_trips_every_format(result, tmp_path, ext):
    path = wade.write_results(result, tmp_path / f"r{ext}")
    assert path.exists()
    read = {".tsv": lambda p: pl.read_csv(p, separator="\t"), ".csv": pl.read_csv,
            ".parquet": pl.read_parquet, ".arrow": pl.read_ipc}[ext](path)
    assert read.columns == list(result.report())
    assert read.height == G
    np.testing.assert_allclose(read["log2_fc"].to_numpy(), result.log2_fc, rtol=1e-6)


def test_write_results_writes_a_manifest_beside_the_table(result, tmp_path):
    wade.write_results(result, tmp_path / "run.tsv")
    man = json.loads((tmp_path / "run.manifest.json").read_text(encoding="utf-8"))
    assert man["wade_version"] == wade.__version__
    assert man["run"]["nperms"] == 100 and man["run"]["seed"] == 1
    assert man["run"]["correction"] == "thinning" and man["run"]["n_boot"] == 20
    design = man["design"]
    space = design.pop("permutation_space")
    assert design == {"n_genes": G, "n_case": N // 2, "n_ctrl": N // 2,
                      "nprobs": N // 2, "max_probs": wade.DEFAULT_MAX_PROBS,
                      "case_label": None,
                      "control_label": None, "condition_column": None}
    # An unrestricted run records the unrestricted permutation space, so a
    # reader can see what resolution the design could support at all.
    assert space["restricted"] is False and space["n_strata"] == 1
    assert space["log10_space"] > 0 and 0 < space["p_floor"] < 1
    assert man["results"]["alpha"] == 0.05
    assert "n_significant_subset" in man["results"]
    assert "created_utc" in man and man["numpy_version"] == np.__version__

    wade.write_results(result, tmp_path / "quiet.tsv", manifest=False)
    assert not (tmp_path / "quiet.manifest.json").exists()


def test_manifest_records_the_condition_labels_when_there_were_any(matrix, sheet):
    cond = condition(sheet, key="sample_id", column="condition",
                     case="tumor", control="normal")
    res = wade.wade(as_counts(matrix, sample_names=SAMPLES), 1.0, cond, nperms=50, seed=1)
    man = wade.manifest(res)
    assert man["design"]["case_label"] == "tumor"
    assert man["design"]["control_label"] == "normal"
    assert man["design"]["condition_column"] == "condition"


def test_write_results_refuses_an_unknown_extension(result, tmp_path):
    with pytest.raises(ValueError, match="unknown result format"):
        wade.write_results(result, tmp_path / "r.xlsx")


def test_manifest_path_keeps_one_manifest_per_table(result, tmp_path):
    """``with_suffix("").with_suffix(...)`` strips every dotted component, so
    ``plasma.v1.tsv`` and ``plasma.v2.tsv`` both wrote ``plasma.manifest.json``
    — the second run silently overwrote the first run's provenance and the
    survivor described the wrong table (package audit, 2026-08-20)."""
    for name in ("plasma.v1.tsv", "plasma.v2.tsv", "run.2026-08-20.csv", "plain.tsv"):
        wade.write_results(result, tmp_path / name)
    manifests = sorted(p.name for p in tmp_path.iterdir() if p.suffix == ".json")
    assert manifests == ["plain.manifest.json", "plasma.v1.manifest.json",
                         "plasma.v2.manifest.json", "run.2026-08-20.manifest.json"]


def test_zero_library_samples_are_refused_by_name(matrix):
    """Every gene in an all-zero library normalizes to the norm_factor
    constant and then takes part in every quantile — the silent-wrong-answer
    shape, so it raises and names the samples."""
    counts = matrix.copy()
    counts[:, 2] = 0.0
    cond = np.r_[np.ones(N // 2, int), np.zeros(N // 2, int)]
    names = np.asarray(SAMPLES, dtype=object)
    with pytest.raises(ValueError, match=r"zero library size.*S2"):
        wade.wade(counts, np.ones(G), cond, nperms=10, sample_names=names)
    res = wade.wade(counts, np.ones(G), cond, nperms=10, sample_names=names,
                    allow_empty_samples=True)
    assert np.all(np.isfinite(res.mean_shift))
    qc = wade.library_qc(counts)
    assert qc["depth"][2] == 0.0
