"""Known-exposure diagnostic on existing high-alteration benchmark matrices.

This isolates normalization after the RLE refresh showed inflation; it is not
a fresh confirmation or an automatically available normalization estimator.
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import wade
from notebook_support import sha256, inference_fingerprint


def run():
    root=Path('.release/0.2.0')
    output=root/'production-known-reference-20260919'
    output.mkdir(exist_ok=False)
    fingerprint=inference_fingerprint()
    (output/'protocol.json').write_text(json.dumps(dict(
        purpose="Diagnose the observed 500:500 high-alteration RLE inflation using the true equal exposures",
        matrices="20 existing mixed-500v500 development matrices; no new draws, no parameter changes",
        maximum=49_999,seed=20260919,post_refresh_diagnostic=True,inference=fingerprint),indent=2)+'\n')
    rows=[]
    for source in sorted((root/'large-cohort-development-20260914/datasets').glob('mixed-500v500-*')):
        started=time.monotonic()
        registry=json.loads((source/'checksums.json').read_text())['files']
        for name in ('counts.tsv','truth.tsv','eligibility.tsv','samples.tsv'):
            assert sha256(source/name)==registry[name]
        frame=pd.read_csv(source/'counts.tsv',sep='\t',index_col=0)
        truth=pd.read_csv(source/'truth.tsv',sep='\t',index_col='gene').reindex(frame.index)
        mask=pd.read_csv(source/'eligibility.tsv',sep='\t',index_col='gene').reindex(frame.index).kept.to_numpy(bool)
        samples=pd.read_csv(source/'samples.tsv',sep='\t')
        assert frame.columns.tolist()==samples['sample'].tolist()
        cond=samples.condition.eq('case').to_numpy(int)
        norm=wade.Normalization.from_size_factors(frame,np.ones(frame.shape[1]),source='Known equal generating exposures')
        res=wade.wade(frame,cond,normalization=norm,test_genes=mask,max_permutations=49_999,seed=20260919,backend='rust')
        out=output/source.name;out.mkdir()
        wade.write_results(res,out/'WADE-known.tsv')
        baseline=pd.read_csv(root/'production-comparisons-20260919/large'/source.name/'WADE.tsv',sep='\t',index_col='gene').reindex(frame.index)
        alternative=(truth.mean_alternative | truth.subset_alternative).to_numpy()
        for method,q in [('Known exposures',res.padj),('RLE',baseline.padj.to_numpy())]:
            calls=q<=.05
            rows.append(dict(dataset=source.name,method=method,calls=int(calls.sum()),
                fdp=float(np.sum(calls & ~alternative)/max(1,calls.sum())),
                power=float(np.sum(calls & alternative)/alternative.sum()),
                source_counts_sha256=registry['counts.tsv']))
        print(source.name,round(time.monotonic()-started,1),'s',flush=True)
    assert fingerprint==inference_fingerprint()
    pd.DataFrame(rows).to_csv(output/'metrics.tsv',sep='\t',index=False)
    from comparison_support import summary
    result=summary(pd.DataFrame(rows),['method'],'fdp')
    result.to_csv(output/'fdp-summary.tsv',sep='\t',index=False)
    print(result.to_string(index=False),flush=True)
    (output/'check_production_reference.py').write_bytes(Path(__file__).read_bytes())
    files={str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file()}
    (output/'checksums.json').write_text(json.dumps(dict(files=files,user_approval='pending'),indent=2)+'\n')


if __name__=='__main__':
    run()
