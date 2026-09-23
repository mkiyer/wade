"""Score the outlier pilot without treating unannotated real cells as true nulls."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control


def average_precision(truth, score):
    truth, score = np.asarray(truth).ravel(), np.asarray(score).ravel()
    if not np.isfinite(score).all():
        raise ValueError("Nonfinite score; report a method failure instead of dropping cells")
    if not truth.any():
        return float("nan")
    order = np.argsort(-score, kind="stable")
    truth, score = truth[order], score[order]
    ends = np.r_[np.flatnonzero(np.diff(score)), len(score)-1]
    tp = np.cumsum(truth)[ends]
    return float(np.sum(np.diff(np.r_[0,tp/truth.sum()]) * tp / (ends+1)))


def native(path, method, genes, samples):
    if method == "OutSingle":
        score = pd.read_csv(path/method/"counts-fzse-zs-svd-optht-zs.csv",sep="\t",index_col=0)
        p, q = None, None
    else:
        p = pd.read_csv(path/method/"p.tsv.gz",sep="\t",index_col=0)
        q = pd.read_csv(path/method/"q-native.tsv.gz",sep="\t",index_col=0)
        assert p.index.equals(genes) and p.columns.tolist() == list(samples)
        assert q.index.equals(genes) and q.columns.tolist() == list(samples)
        p, q = p.to_numpy(), q.to_numpy()
        assert np.isfinite(p).all() and ((p >= 0)&(p <= 1)).all()
        assert np.isfinite(q).all() and ((q >= 0)&(q <= 1)).all()
        score = pd.DataFrame(-np.log10(np.maximum(p, np.nextafter(0.,1.))),index=genes,columns=samples)
    assert score.index.equals(genes) and score.columns.tolist() == list(samples)
    return np.abs(score.to_numpy()), p, q


def score(root):
    native_rows, group_rows, burden_rows = [], [], []
    for label in ("kremer", "gtex-skin"):
        base, injected = root/f"{label}-baseline", root/f"{label}-injected"
        counts = pd.read_csv(base/"counts.tsv",sep="\t",index_col=0)
        genes, samples = counts.index, counts.columns
        with np.load(injected/"injection.npz",allow_pickle=False) as arrays:
            cell_truth = arrays["changed"]
            intended = arrays["intended"]
        truth = pd.read_csv(injected/"gene-truth.tsv",sep="\t",index_col=0)
        membership = pd.read_csv(injected/"samples.tsv",sep="\t")
        assert membership["sample"].tolist() == samples.tolist()
        cases = membership.condition.eq("case").to_numpy()
        # A selected zero count can remain zero after multiplication. Retain
        # that target in accounting, but do not invent an altered-gene truth.
        gene_truth = cell_truth.any(axis=1)
        realized_gene = pd.Series(gene_truth,index=genes)
        # Global shifts and mixed bulk mechanisms belong to the group task.
        # Counting every shifted cell as a native outlier would penalize the
        # latent-factor methods for explaining a shared expression pattern.
        complex_genes = truth.index[~truth.pattern.isin(["up","down"])]
        native_rows_mask = ~genes.isin(complex_genes)
        evaluation = np.broadcast_to(native_rows_mask[:,None], cell_truth.shape).copy()
        evaluation &= ~(intended & ~cell_truth)
        gene_predictions = {}
        for method in ("OUTRIDER-AE", "OUTRIDER-PCA", "saseR", "OutSingle"):
            before_score, before_p, before_q = native(base, method, genes, samples)
            after_score, after_p, after_q = native(injected, method, genes, samples)
            record = dict(background=label,method=method,genes=len(genes),samples=len(samples),
                changed_cells=int((cell_truth&evaluation).sum()),
                cell_evaluation_genes=int(native_rows_mask.sum()),
                cell_evaluation_count=int(evaluation.sum()),
                apparent_cell_average_precision=average_precision(cell_truth[evaluation],after_score[evaluation]),
                apparent_gene_average_precision=average_precision(gene_truth[native_rows_mask],after_score[native_rows_mask][:,cases].max(axis=1)))
            if after_q is not None:
                called = after_q <= .05
                called_before = before_q <= .05
                record.update(native_calls=int(called.sum()),
                    native_evaluated_calls=int((called&evaluation).sum()),
                    changed_cells_called=int((called&cell_truth&evaluation).sum()),
                    changed_cell_recall=float((called&cell_truth&evaluation).sum()/(cell_truth&evaluation).sum()),
                    apparent_precision=float((called&cell_truth&evaluation).sum()/max(1,(called&evaluation).sum())),
                    native_baseline_calls=int(called_before.sum()),
                    unmodified_cells_called_before_and_after=int((called&called_before&~cell_truth).sum()))
                gene_p = np.minimum(1, cases.sum()*after_p[:,cases].min(axis=1))
                gene_q = false_discovery_control(gene_p)
                gene_predictions[method+" case-minP"] = pd.Series(gene_q,index=genes)
                pd.DataFrame(dict(gene=genes,p_case_min=gene_p,q_case_min=gene_q)).to_csv(
                    injected/method/"gene-case-minP.tsv",sep="\t",index=False)
                for j,sample in enumerate(samples):
                    burden_rows.append(dict(background=label,method=method,sample=sample,
                        group="case" if cases[j] else "control",before=int(called_before[:,j].sum()),
                        after=int(called[:,j].sum()),changed_called=int((called[:,j]&cell_truth[:,j]).sum())))
            native_rows.append(record)
        for method in ("WADE",):
            frame = pd.read_csv(injected/(method+".tsv"),sep="\t",index_col="gene")
            gene_predictions[method] = frame["padj"].reindex(genes).fillna(1)
        deseq = pd.read_csv(injected/"DESeq2-alpha-0.05.tsv",sep="\t",index_col="gene")
        gene_predictions["DESeq2"] = deseq.padj.reindex(genes).fillna(1)
        edger = pd.read_csv(injected/"edgeR-QL.tsv",sep="\t",index_col="gene")
        gene_predictions["edgeR-QL"] = edger.FDR.reindex(genes).fillna(1)
        for method,predictions in gene_predictions.items():
            for (pattern, prevalence), selected in truth.groupby(["pattern","prevalence"]):
                realized = selected.index[realized_gene.reindex(selected.index)]
                calls = predictions.reindex(realized).le(.05)
                group_rows.append(dict(background=label,method=method,pattern=pattern,
                    prevalence=prevalence,realized_prevalence=selected.realized_prevalence.iloc[0],
                    targeted_genes=len(selected),unmodified_target_genes=len(selected)-len(realized),
                    planted_genes=len(realized),recovered=int(calls.sum()),recall=float(calls.mean()),
                    total_calls=int(predictions.le(.05).sum()),
                    calls_outside_planted=int(predictions.loc[~gene_truth].le(.05).sum())))
    pd.DataFrame(native_rows).to_csv(root/"native-scores.tsv",sep="\t",index=False)
    pd.DataFrame(group_rows).to_csv(root/"group-scores.tsv",sep="\t",index=False)
    pd.DataFrame(burden_rows).to_csv(root/"patient-burden.tsv",sep="\t",index=False)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    score(parser.parse_args().root)
