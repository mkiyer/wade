"""Reproducible gene-count inputs and analyses for the TCGA notebooks.

This is example code, not WADE API. See docs/tcga-examples.md for the
prespecified design and the distinction between labels and validation.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

import numpy as np
import polars as pl
import wade

from notebook_support import array_hash, export_table, sha256, inference_fingerprint, cached_computation

SUBTYPES = ["Luminal A", "Luminal B", "HER2-enriched", "Basal-like"]
PROSTATE_PANEL = ["ERG", "ETV1", "ETV4", "FLI1", "SPINK1", "AMACR", "KRT5", "KRT15", "TP63"]
BREAST_PANEL = ["ERBB2", "GRB7", "STARD3", "PGAP3", "ESR1", "PGR", "GATA3", "FOXA1",
                "KRT5", "KRT14", "KRT17", "MKI67"]
PARAMS = dict(seed=1)


def clinical_wide(path, kind):
    rows = pl.DataFrame(json.loads(Path(path).read_text()))
    keys = ["patientId"] if kind == "patient" else ["sampleId", "patientId"]
    if rows.select(keys + ["clinicalAttributeId"]).is_duplicated().any():
        raise ValueError(f"Duplicate clinical keys in {path}")
    return rows.pivot(on="clinicalAttributeId", index=keys, values="value")


def checked_join(left, right, *, on):
    """A left join may have missing annotations, never duplicate identifiers."""
    if right[on].null_count() or right[on].n_unique() != right.height:
        raise ValueError(f"Annotation key {on!r} is missing or nonunique")
    before = left["library"].to_list()
    joined = left.join(right, on=on, how="left", validate="m:1", maintain_order="left")
    if joined["library"].to_list() != before:
        raise ValueError("Annotation join changed sample order")
    return joined


def annotations(meta, cancer, directory):
    directory = Path(directory)
    sources = json.loads((directory / "sources.json").read_text())
    used = [f"{cancer}_sample.json", f"{cancer}_patient.json"]
    if cancer == "breast":
        used += ["breast_clinical_patient.json", "breast_erbb2_cna.json"]
    for name in used:
        if sha256(directory / name) != sources[name]["sha256"]:
            raise ValueError(f"Annotation checksum mismatch: {name}")
    sample = clinical_wide(directory / f"{cancer}_sample.json", "sample")
    patient = clinical_wide(directory / f"{cancer}_patient.json", "patient")
    # Sample IDs must agree with patient IDs, before any biological join.
    if sample.filter(pl.col("sampleId").str.slice(0, 12) != pl.col("patientId")).height:
        raise ValueError("Sample/patient mismatch in annotation source")
    if cancer == "prostate":
        sample_cols = ["ETV1_STATUS", "ETV4_STATUS", "FLI1_STATUS", "SPINK1_HIGH"]
        patient_cols = ["ERG_STATUS"]
    else:
        sample_cols = ["CANCER_TYPE_DETAILED", "TISSUE_SOURCE_SITE_CODE"]
        patient_cols = ["SUBTYPE", "SEX", "ICD_O_3_HISTOLOGY", "HISTORY_NEOADJUVANT_TRTYN"]
    meta = checked_join(meta, sample.select("sampleId", *sample_cols)
                        .rename({"sampleId": "tcga_sample"}), on="tcga_sample")
    meta = checked_join(meta, patient.select("patientId", *patient_cols)
                        .rename({"patientId": "patient"}), on="patient")
    if cancer == "breast":
        clinical = clinical_wide(directory / "breast_clinical_patient.json", "patient")
        columns = ["ER_STATUS_BY_IHC", "PR_STATUS_BY_IHC", "IHC_HER2", "HER2_FISH_STATUS", "SEX"]
        clinical = clinical.select("patientId", *columns).rename(
            {"patientId": "patient", "SEX": "clinical_sex"})
        meta = checked_join(meta, clinical, on="patient")
        if meta.filter(pl.col("SEX").is_not_null() & pl.col("clinical_sex").is_not_null()
                       & (pl.col("SEX") != pl.col("clinical_sex"))).height:
            raise ValueError("Conflicting patient sex across TCGA annotation sources")
        cna = pl.DataFrame(json.loads((directory / "breast_erbb2_cna.json").read_text()))
        meta = checked_join(meta, cna.select("sampleId", "value").rename(
            {"sampleId": "tcga_sample", "value": "ERBB2_CNA"}), on="tcga_sample")
        subtype_map = dict(zip(["BRCA_LumA", "BRCA_LumB", "BRCA_Her2", "BRCA_Basal", "BRCA_Normal"],
                               SUBTYPES + ["Normal-like"]))
        unexpected = set(meta["SUBTYPE"].drop_nulls().to_list()) - set(subtype_map)
        if unexpected:
            raise ValueError(f"Unrecognized subtype labels: {unexpected}")
        meta = meta.with_columns(
            subtype=pl.col("SUBTYPE").replace_strict(subtype_map, default="Unknown"),
            annotated_sex=pl.coalesce("SEX", "clinical_sex").fill_null("Unknown"))
    return meta, {name: sources[name] for name in used}


def independent_indices(meta):
    normal = meta["group"].to_numpy() == "normal"
    paired_patients = set(meta.filter(pl.col("group") == "normal")["patient"].to_list())
    tumour = meta["group"].to_numpy() == "tumor"
    matched = np.array([p in paired_patients for p in meta["patient"]])
    independent = np.flatnonzero(normal | (tumour & ~matched))
    paired = np.flatnonzero(normal | (tumour & matched))
    patients = meta[independent.tolist()]["patient"].to_list()
    if len(patients) != len(set(patients)):
        raise ValueError("The independent design repeats a patient")
    pairs = meta[paired.tolist()].group_by("patient").agg(
        pl.len().alias("n"), pl.col("group").n_unique().alias("groups"))
    if pairs.filter((pl.col("n") != 2) | (pl.col("groups") != 2)).height:
        raise ValueError("Paired design requires one tumour and one normal per patient")
    return independent, paired


@dataclass
class Cohort:
    cancer: str
    counts: np.ndarray
    genes: pl.DataFrame
    meta: pl.DataFrame
    normalization: wade.Normalization
    normalizations: dict
    independent: np.ndarray
    paired: np.ndarray
    provenance: dict
    test_genes: np.ndarray


def load_cohort(cancer, out, config_path=None):
    """Read an rna100k cohort bundle, join the frozen TCGA annotations, fix the design.

    The configuration names one bundle per cancer (written by `rna100k extract`)
    and the annotation snapshot from tools/fetch_tcga_annotations.py; relative
    paths resolve against the configuration's folder.
    """
    config_path = config_path or os.environ.get("WADE_TCGA_CONFIG")
    if not config_path:
        raise ValueError("Set WADE_TCGA_CONFIG; see docs/tcga-examples.md")
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    def path(value):
        value = Path(value).expanduser()
        return value if value.is_absolute() else (config_path.parent / value).resolve()
    bundle = path(config["bundles"][cancer])
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha256(bundle / name) != digest:
            raise ValueError(f"Bundle file changed since extraction: {name}")
    samples, genes = pl.read_parquet(bundle / "samples.parquet"), pl.read_parquet(bundle / "genes.parquet")
    required = ["library", "sample", "patient", "study", "tissue_of_origin", "sample_class",
                "library_protocol", "preservation", "oncotree_code"]
    tissue = {"prostate": "Prostate", "breast": "Breast"}[cancer]
    export_table(samples.select(required), out, "input-library-inventory")
    eligible = samples.select(required).filter((pl.col("study") == "TCGA") & (pl.col("tissue_of_origin") == tissue)
                                               & pl.col("sample_class").is_in(["tumor", "normal_adjacent"]))
    for column in ("library", "sample", "patient"):
        if eligible[column].null_count():
            raise ValueError(f"Missing {column} in eligible samples")
    meta = eligible.with_columns(
        group=pl.when(pl.col("sample_class") == "tumor").then(pl.lit("tumor")).otherwise(pl.lit("normal")),
        tcga_sample=pl.col("sample").str.slice(0, 15))
    if meta.filter(pl.col("sample").str.slice(0, 12) != pl.col("patient")).height:
        raise ValueError("Local sample/patient IDs disagree")
    if meta.select("patient", "group").is_duplicated().any():
        raise ValueError("The bundle must hold one library per patient and sample class")
    export_table(pl.read_parquet(bundle / "excluded.parquet"), out, "duplicate-libraries-excluded")
    meta, sources = annotations(meta, cancer, path(config["annotations"]))
    if cancer == "breast":
        # Freeze the target population before fitting: female primary tumours
        # and their normal-adjacent references. Report all excluded records.
        export_table(meta.filter(pl.col("annotated_sex") != "Female"), out, "sex-eligibility-exclusions")
        meta = meta.filter(pl.col("annotated_sex") == "Female")
    independent, paired = independent_indices(meta)
    frame = pl.read_ipc(bundle / "counts.feather", memory_map=False)
    if not frame["gene_id"].equals(genes["gene_id"]):
        raise ValueError("Bundle counts and genes.parquet rows differ")
    counts = frame.select(meta["library"].to_list()).to_numpy().astype(np.float64)
    lib_sizes = counts.sum(axis=0)
    if np.any(lib_sizes <= 0):
        raise ValueError("Empty gene-count library")
    names = genes["label"].to_numpy()
    fitting_cohorts = {"primary": independent, "paired": paired,
                       "descriptive": np.arange(meta.height)}
    if cancer == "breast":
        fitting_cohorts["subtypes"] = np.flatnonzero(
            ((meta["group"] == "tumor") & meta["subtype"].is_in(SUBTYPES)).to_numpy())
    normalizations = {name: wade.fit_normalization(counts[:, idx], method="rle",
        gene_names=names, sample_names=meta["library"].to_numpy()[idx])
        for name, idx in fitting_cohorts.items()}
    normalization = normalizations["descriptive"]
    cpm = counts / lib_sizes * 1e6
    keep = ((counts >= 10).sum(axis=1) >= 5) & (cpm.mean(axis=1) >= 0.1)
    filter_table = genes.with_columns(tested=pl.Series(keep),
        samples_ge_10=pl.Series((counts >= 10).sum(axis=1)), pooled_mean_cpm=pl.Series(cpm.mean(axis=1)))
    export_table(filter_table, out, "gene-filter")
    qc = wade.library_qc(counts)
    meta = meta.with_columns(**{k: pl.Series(v) for k, v in qc.items()})
    export_table(meta, out, "samples")
    provenance = dict(assay="gene counts", normalizations={k: v.to_dict() for k, v in normalizations.items()},
        normalization_cohort="sample eligibility first within each analysis family, then full genes; nested contrasts reuse exposures",
        filter="raw counts >= 10 in >= 5 pooled samples and pooled mean CPM >= 0.1",
        breast_target="female primary tumour and normal-adjacent" if cancer == "breast" else None,
        bundle=dict(path=str(bundle), release=manifest["release"]["name"], query=manifest["query"],
                    files=manifest["files"], rna100k_version=manifest["rna100k_version"]),
        count_slice_sha256=array_hash(counts), sample_order_sha256=array_hash(meta["library"].to_numpy()),
        annotation_sources=sources, config_sha256=sha256(config_path),
        source_fingerprint=os.environ.get("WADE_SOURCE_FINGERPRINT"),
        genes_supplied=genes.height, genes_tested=int(keep.sum()), samples=meta.height)
    (Path(out) / "inputs.json").write_text(json.dumps(provenance, indent=2) + "\n")
    genes = genes.with_columns(gene_name=pl.col("label"))    # unique names for WADE and plots
    return Cohort(cancer, counts, genes, meta, normalization, normalizations,
                  independent, paired, provenance, keep)


def run(cohort, indices, out, name, *, max_permutations=wade.DEFAULT_MAX_PERMUTATIONS, condition=None,
        reference=None):
    out = Path(out) / name
    out.mkdir(parents=True, exist_ok=True)
    meta = cohort.meta[indices.tolist()]
    cond = (meta["group"].to_numpy() == "tumor").astype(int) if condition is None else np.asarray(condition)
    if len(cond) != len(indices):
        raise ValueError("Condition is not aligned to selected samples")
    reference = reference or "primary"
    normalization = cohort.normalizations[reference]
    settings = dict(PARAMS, max_permutations=max_permutations)
    export_table(meta.with_columns(condition=pl.Series(cond)), out, "sample-order")
    (out / "design.json").write_text(json.dumps(dict(settings=settings,
        sample_order_sha256=array_hash(meta["library"].to_numpy()), inputs=cohort.provenance), indent=2) + "\n")
    start = time.perf_counter()
    print(f"Running {cohort.cancer}/{name}: {int(cohort.test_genes.sum())} selected genes, "
          f"{int(cond.sum())}:{int((cond == 0).sum())}, cap={max_permutations}", flush=True)
    key = dict(inference=inference_fingerprint(), settings=settings,
        counts=array_hash(cohort.counts[:, indices]),
        normalization=normalization.select_samples(meta["library"].to_numpy()).to_dict(),
        genes=array_hash(cohort.genes["gene_name"].to_numpy()), test_genes=array_hash(cohort.test_genes),
        samples=array_hash(meta["library"].to_numpy()), cond=array_hash(cond))
    result = cached_computation(key, lambda: wade.wade(
        cohort.counts[:, indices], cond, normalization=normalization, test_genes=cohort.test_genes,
        gene_names=cohort.genes["gene_name"].to_numpy(), sample_names=meta["library"].to_numpy(),
        **settings), namespace="tcga")
    wade.write_results(result, out / "all-genes.tsv")
    (out / "timing.json").write_text(json.dumps(dict(seconds=time.perf_counter()-start,
        shuffled_mean_genes=int(np.sum(result.mean_permutations > 0)),
        mean_permutations=int(result.mean_permutations.sum()),
        subset_permutations=int(result.subset_permutations.sum())), indent=2)+"\n")
    print(f"Finished {name} in {time.perf_counter() - start:.1f}s", flush=True)
    return result


def panel_table(result, panel):
    order = pl.DataFrame(dict(gene=panel, panel_order=list(range(len(panel)))))
    return order.join(result.to_frame(), on="gene", how="left", validate="1:1").sort("panel_order").drop("panel_order")


def annotation_expression(cohort, result, panel):
    """Every sample, with its supplied annotation; no inferred memberships."""
    meta = cohort.meta.filter(pl.col("library").is_in(result.sample_names.tolist()))
    lookup = {sample: j for j, sample in enumerate(result.sample_names)}
    columns = np.array([lookup[s] for s in meta["library"]])
    frames = []
    for gene in panel:
        if gene in result.gene:
            frames.append(meta.with_columns(gene=pl.lit(gene),
                log_expression=pl.Series(log2_expression(result, gene)[columns])))
    return pl.concat(frames)


def log2_expression(result, gene):
    """log2(count / size factor + 1) for display, whatever scale was tested."""
    x = result.expression[result.gene_index(gene)]
    return x if result.params["scale"] == "log2" else np.log2(x + 1)


def annotated_subsets(cohort, result, labels):
    """Each gene's subset against an annotation WADE never sees.

    `labels` maps gene -> (column, positive value, negative values). Only tumours
    annotated positive or negative enter the denominators; other and missing
    annotations are never counted as negatives.
    """
    tumours = cohort.meta.filter(pl.col("library").is_in(result.sample_names.tolist()) & (pl.col("group") == "tumor"))
    rows = []
    for gene, (column, positive, negatives) in labels.items():
        status = dict(zip(tumours["library"].to_list(), tumours[column].to_list()))
        found = result.subset(gene)
        members = set(found.samples.tolist())
        pos = {lib for lib, v in status.items() if v == positive}
        neg = {lib for lib, v in status.items() if v in negatives}
        rows.append(dict(gene=gene, side=found.side, subset=len(members), annotated_positive=len(pos),
                         positive_in_subset=len(members & pos), negative_in_subset=len(members & neg),
                         precision=len(members & pos) / max(1, len(members & (pos | neg))),
                         recall=len(members & pos) / max(1, len(pos))))
    return pl.DataFrame(rows)


def plot_subsets(result, panel):
    return wade.plot_patterns(result, label=[g for g in panel if g in result.gene])


def plot_qc(cohort, result):
    import matplotlib.pyplot as plt
    selected = cohort.meta.filter(pl.col("library").is_in(result.sample_names.tolist()))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for group, color, label in [("tumor", "#477caa", "Tumour"), ("normal", "#dd9454", "Normal-adjacent")]:
        part = selected.filter(pl.col("group") == group)
        axes[0].scatter(np.log10(part["depth"]), part["n_detected"], s=14, alpha=.5, color=color, label=label)
        axes[1].scatter(np.log10(part["depth"]), part["top_share"], s=14, alpha=.5, color=color)
    axes[0].set(ylabel="Detected genes", xlabel="Library gene counts, log10")
    axes[1].set(ylabel="Share in top 10 genes", xlabel="Library gene counts, log10")
    axes[0].legend(frameon=False, fontsize=9)
    return fig
