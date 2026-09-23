"""Independent finite-p/BH, metric and native-copy audit of migrated comparisons."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
from notebook_support import sha256


def audit(root):
    checks=0;genes=0;metrics=0;copies=0
    for kind in ('large','reference','outlier'):
        folder=root/kind
        files=json.loads((folder/'checksums.json').read_text())['files']
        for name,digest in files.items():
            assert sha256(folder/name)==digest
            checks+=1
        for path in sorted(folder.iterdir()):
            if not (path/'WADE.tsv').exists():continue
            frame=pd.read_csv(path/'WADE.tsv',sep='\t',index_col='gene')
            record=json.loads((path/'WADE.manifest.json').read_text())
            refreshed=json.loads((path/'refresh.json').read_text())
            source=Path(refreshed['source'])
            for name,digest in refreshed['inputs'].items():
                assert sha256(source/name)==digest
                if (path/name).exists():
                    assert sha256(path/name)==digest
                    copies+=1
            keep=frame.status.eq('tested').to_numpy()
            k=frame.n_exceedances.to_numpy();t=frame.n_permutations.to_numpy()
            h=record['run']['stop_exceedances'];cap=record['run']['max_permutations']
            assert np.all((k[keep]==h)|(t[keep]==cap))
            p=np.where(k[keep]==h,h/t[keep],(k[keep]+1)/(cap+1))
            np.testing.assert_allclose(frame.pvalue[keep],p,rtol=3e-12,atol=1e-16)
            np.testing.assert_allclose(frame.padj[keep],false_discovery_control(p),rtol=3e-12,atol=1e-16)
            assert frame.pvalue[~keep].isna().all() and frame.padj[~keep].isna().all()
            assert frame.index.tolist()==record['design']['genes']
            assert int(keep.sum())==record['design']['n_tested']
            assert t.tolist()==record['resampling']['n_permutations']
            assert k.tolist()==record['resampling']['n_exceedances']
            genes+=int(keep.sum())
            scores=pd.read_csv(path/'scores.tsv',sep='\t')
            for row in scores.itertuples():
                name={'WADE':('WADE.tsv','padj'),'DESeq2':('DESeq2-alpha-0.05.tsv','padj'),
                      'DESeq2-no-Cooks':('DESeq2-no-Cooks-alpha-0.05.tsv','padj'),'edgeR-QL':('edgeR-QL.tsv','FDR')}[row.method]
                table=pd.read_csv(path/name[0],sep='\t',index_col='gene').reindex(frame.index)
                called=table[name[1]].le(.05).to_numpy()
                assert row.calls==int(called.sum())
                if (path/'truth.tsv').exists():
                    truth=pd.read_csv(path/'truth.tsv',sep='\t',index_col='gene').reindex(frame.index)
                    alt=(truth.mean_alternative | truth.subset_alternative).to_numpy() if kind=='large' else truth['differential.expression'].to_numpy(bool)
                    selected=np.ones(len(alt),bool) if row.category=='all' else truth.category.eq(row.category).to_numpy()
                    recovered=int(np.sum(called & alt & selected))
                    assert row.recovered==recovered
                    expected=np.sum(called & ~alt)/max(1,called.sum())
                    assert abs(row.fdp_distribution-expected)<1e-12
                    if row.altered:
                        assert abs(row.power-recovered/row.altered)<1e-12
                metrics+=1
    result=dict(status='PASS',file_checks=checks,native_and_input_copies=copies,tested_wade_genes=genes,metric_rows=metrics)
    (root/'independent-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__=='__main__':
    audit(Path('.release/0.2.0/production-comparisons-20260919'))
