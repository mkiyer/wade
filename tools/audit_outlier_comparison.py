"""Independently check injection identity, native scoring and group recall.

Uses a grouped-score implementation of AP, not the production scorer's rank
endpoint code. These are provenance/metric checks, not biological calibration.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from notebook_support import sha256


def grouped_ap(truth, score):
    values, inverse, sizes = np.unique(score, return_inverse=True, return_counts=True)
    assert np.isfinite(values).all()
    positives = np.bincount(inverse, weights=np.asarray(truth, int), minlength=len(values))[::-1]
    if not positives.sum():
        return np.nan
    precision = positives.cumsum() / sizes[::-1].cumsum()
    return float(np.sum(positives / positives.sum() * precision))


def check(root):
    root = Path(root)
    metrics = pd.read_csv(root/'native-scores.tsv', sep='\t').set_index(['background','method'])
    group = pd.read_csv(root/'group-scores.tsv', sep='\t')
    group_checks = 0
    native_checks = 0
    for background in ('kremer','gtex-skin'):
        before, after = root/f'{background}-baseline', root/f'{background}-injected'
        base = pd.read_csv(before/'counts.tsv', sep='\t', index_col=0)
        counts = pd.read_csv(after/'counts.tsv', sep='\t', index_col=0)
        assert base.index.equals(counts.index) and base.columns.equals(counts.columns)
        arrays = np.load(after/'injection.npz')
        changed, intended = arrays['changed'], arrays['intended']
        np.testing.assert_array_equal(changed, base.to_numpy() != counts.to_numpy())
        np.testing.assert_array_equal(counts.to_numpy(), np.rint(base.to_numpy()*arrays['multiplier']))
        truth = pd.read_csv(after/'gene-truth.tsv', sep='\t', index_col='gene')
        assert truth.index.is_unique and truth.index.isin(counts.index).all()
        samples = pd.read_csv(after/'samples.tsv', sep='\t')
        assert samples['sample'].tolist() == counts.columns.tolist()
        cases = samples.condition.eq('case').to_numpy()
        assert not changed[:,~cases].any()
        pure = ~counts.index.isin(truth.index[~truth.pattern.isin(['up','down'])])
        evaluate = pure[:,None] & ~(intended & ~changed)
        assert not np.any(changed & ~intended)
        truth_gene = changed.any(axis=1)
        predictions = {}
        for method in ('OUTRIDER-AE','OUTRIDER-PCA','saseR','OutSingle'):
            expected = metrics.loc[(background,method)]
            if method == 'OutSingle':
                scored = pd.read_csv(after/method/'counts-fzse-zs-svd-optht-zs.csv', sep='\t', index_col=0)
                score = np.abs(scored.to_numpy())
            else:
                scored = pd.read_csv(after/method/'p.tsv.gz', sep='\t', index_col=0)
                p = scored.to_numpy()
                q = pd.read_csv(after/method/'q-native.tsv.gz', sep='\t', index_col=0)
                assert q.index.equals(counts.index) and q.columns.equals(counts.columns)
                called = q.to_numpy() <= .05
                target_calls = int(np.sum(called & changed & evaluate))
                assert target_calls == expected.changed_cells_called
                assert abs(target_calls / np.sum(changed & evaluate) - expected.changed_cell_recall) < 1e-12
                aggregated = np.minimum(1, cases.sum() * p[:,cases].min(axis=1))
                adjusted = false_discovery_control(aggregated)
                saved = pd.read_csv(after/method/'gene-case-minP.tsv', sep='\t', index_col='gene')
                assert saved.index.equals(counts.index)
                np.testing.assert_allclose(saved.p_case_min, aggregated, rtol=3e-12, atol=1e-16)
                np.testing.assert_allclose(saved.q_case_min, adjusted, rtol=3e-12, atol=1e-16)
                predictions[method+' case-minP'] = adjusted
                score = -np.log10(np.maximum(p, np.nextafter(0.,1.)))
            assert scored.index.equals(counts.index) and scored.columns.equals(counts.columns)
            assert expected.changed_cells == int(np.sum(changed & evaluate))
            assert expected.cell_evaluation_genes == int(pure.sum())
            assert expected.cell_evaluation_count == int(evaluate.sum())
            assert abs(grouped_ap(changed[evaluate],score[evaluate])-expected.apparent_cell_average_precision) < 2e-12
            assert abs(grouped_ap(truth_gene[pure],score[pure][:,cases].max(axis=1))-expected.apparent_gene_average_precision) < 2e-12
            native_checks += 1
        for method,filename,column in [('WADE','WADE.tsv','padj'),('DESeq2','DESeq2-alpha-0.05.tsv','padj'),('edgeR-QL','edgeR-QL.tsv','FDR')]:
            table = pd.read_csv(after/filename,sep='\t',index_col='gene').reindex(counts.index)
            predictions[method] = table[column].fillna(1).to_numpy()
        for row in group[group.background.eq(background)].itertuples():
            ids = truth.index[(truth.pattern==row.pattern)&(truth.prevalence==row.prevalence)]
            selected = counts.index.isin(ids)
            called = predictions[row.method] <= .05
            real = selected & truth_gene
            assert row.targeted_genes == selected.sum()
            assert row.planted_genes == real.sum()
            assert row.unmodified_target_genes == (selected & ~truth_gene).sum()
            assert row.recovered == (called & real).sum()
            assert abs(row.recall - np.sum(called & real)/real.sum()) < 1e-12
            assert row.total_calls == called.sum()
            assert row.calls_outside_planted == (called & ~truth_gene).sum()
            group_checks += 1
        print(background,'injection and scoring PASS',flush=True)
    result = dict(status='PASS',native_metric_rows=native_checks,group_metric_rows=group_checks,
                  source_checksums_sha256=sha256(root/'checksums.json'),script_sha256=sha256(__file__))
    (root.parent/'independent-outlier-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result,flush=True)


if __name__ == '__main__':
    check('.release/0.2.0/production-comparisons-20260919/outlier')
