"""Current public WADE on frozen comparator counts; no native method refitting.

Prespecified refresh budget: 49,999 shuffles, seed 20260919, q=0.05.
The 2,000-gene development screen can resolve an isolated 5% discovery at
this cap. Larger reference families retain the explicit resolution limitation.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import shutil
import time
import numpy as np
import pandas as pd
import wade
from notebook_support import inference_fingerprint, sha256


def refresh(source, target, kind):
    source, target = Path(source), Path(target)
    target.mkdir(parents=True)
    started = time.monotonic()
    registry_path = source/"checksums.json" if kind=="large" else source.parent/"checksums.json"
    registry = json.loads(registry_path.read_text())["files"]
    validated = {}
    def take(name, copy=False):
        path = source/name
        key = name if kind=="large" else source.name+"/"+name
        digest = sha256(path)
        if registry.get(key)!=digest:
            raise ValueError(f"Changed or uninventoried comparator input: {path}")
        validated[name]=digest
        if copy:
            destination=target/name; destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,destination)
        return path
    frame = pd.read_csv(take("counts.tsv"),sep="\t",index_col=0)
    samples = pd.read_csv(take("samples.tsv",True),sep="\t",dtype={"sample":str})
    assert frame.columns.tolist()==samples["sample"].tolist()
    cond = samples.condition.eq("case").to_numpy(int)
    if kind=="large":
        truth=pd.read_csv(take("truth.tsv",True),sep="\t",index_col="gene")
        eligible=pd.read_csv(take("eligibility.tsv",True),sep="\t",index_col="gene")
        settings=json.loads(take("settings.json",True).read_text())
        mask=eligible.reindex(frame.index)["kept"].to_numpy(bool)
    else:
        eligible=pd.read_csv(take("feature-filter.tsv",True),sep="\t",index_col="gene")
        mask=eligible.reindex(frame.index)["kept"].to_numpy(bool)
        settings=dict(dataset=source.name,n_case=int(cond.sum()),n_control=int((cond==0).sum()))
        truth=pd.read_csv(take("truth.tsv",True),sep="\t",index_col="gene") if (source/"truth.tsv").exists() else None
        take("generation.json",True)
    norm=wade.fit_normalization(frame,reference_genes=mask)
    if kind=="large":
        native_norm=pd.read_csv(take("DESeq2-normalization.tsv",True),sep="\t")
        factors=native_norm["size_factor"].to_numpy() if "size_factor" in native_norm else native_norm["sizeFactor"].to_numpy()
    else:
        native_norm=pd.read_csv(take("normalization.tsv",True),sep="\t")
        factors=native_norm.deseq2_size_factor.to_numpy()
    factors=factors/np.exp(np.log(factors).mean())
    np.testing.assert_allclose(norm.size_factors,factors,rtol=2e-12,atol=0)
    result=wade.wade(frame,cond,normalization=norm,test_genes=mask,max_permutations=49_999,
                     seed=20260919,strata=samples.block.to_numpy() if "block" in samples else None,
                     backend="rust",gene_chunk=1024)
    wade.write_results(result,target/"WADE.tsv")
    native={"DESeq2":("DESeq2-alpha-0.05.tsv","padj"),"edgeR-QL":("edgeR-QL.tsv","FDR")}
    if kind=="large":
        native["DESeq2-no-Cooks"]=("DESeq2-no-Cooks-alpha-0.05.tsv","padj")
    predictions={"WADE":pd.Series(result.padj,index=frame.index)}
    for method,(name,qcol) in native.items():
        data=pd.read_csv(take(name,True),sep="\t",index_col="gene")
        assert set(data.index)==set(frame.index[mask])
        predictions[method]=data[qcol].reindex(frame.index)
    rows=[]
    if truth is not None:
        truth=truth.reindex(frame.index)
        if kind=="large":
            distribution=truth.mean_alternative | truth.subset_alternative
            mean=truth.mean_alternative
            groups={"all":np.ones(len(truth),bool),**{c:(truth.category==c).to_numpy() for c in truth.category.unique()}}
            requested=int(distribution.sum())
        else:
            distribution=truth["differential.expression"].astype(bool)
            mean=distribution
            groups={"all":np.ones(len(truth),bool)}
            generation=json.loads((source/"generation.json").read_text())
            requested=int(generation["parameters"]["n.diffexp"])
        for method,q in predictions.items():
            called=q.le(.05)
            native_truth=distribution if method=="WADE" else mean
            for category,selected in groups.items():
                alt=distribution & selected
                tp=int((called & alt).sum())
                denominator=requested if category=="all" else int(alt.sum())
                rows.append(dict(**settings,method=method,category=category,alpha=.05,
                    altered=denominator,recovered=tp,power=tp/denominator if denominator else np.nan,
                    calls=int(called.sum()),false_distribution=int((called & ~distribution).sum()),
                    fdp_distribution=float((called & ~distribution).sum()/max(1,called.sum())),
                    fdp_native_target=float((called & ~native_truth).sum()/max(1,called.sum()))))
    else:
        rows=[dict(**settings,method=m,alpha=.05,calls=int(q.le(.05).sum()),category="all") for m,q in predictions.items()]
    if kind=="outlier":
        # Native cell-level and gene-aggregate outputs keep their own inferential unit.
        for folder in ("OUTRIDER-AE","OUTRIDER-PCA","saseR","OutSingle"):
            for p in (source/folder).rglob('*'):
                if p.is_file() and (p.name in {"p.tsv.gz","q-native.tsv.gz","counts-fzse-zs-svd-optht-zs.csv"}):
                    take(str(p.relative_to(source)),True)
        for name in ("injection.npz","gene-truth.tsv"):
            if (source/name).exists(): take(name,True)
        # The scorer needs the complete gene/sample identity, not the old WADE fit.
        take("counts.tsv",True)
    pd.DataFrame(rows).to_csv(target/"scores.tsv",sep="\t",index=False)
    record=dict(source=str(source.resolve()),source_registry_sha256=sha256(registry_path),inputs=validated,
                settings=settings,seconds=time.monotonic()-started,normalization_parity="PASS",
                maximum=49_999,seed=20260919,gene_permutations=int(result.n_permutations.sum()))
    (target/"refresh.json").write_text(json.dumps(record,indent=2)+"\n")
    return rows,record


def main(args):
    root=args.source/"datasets" if args.kind=="large" else args.source
    paths=[p for p in sorted(root.iterdir()) if (p/"counts.tsv").exists()]
    if args.limit: paths=paths[:args.limit]
    args.output.mkdir(parents=True,exist_ok=False)
    fingerprint=inference_fingerprint()
    (args.output/"protocol.json").write_text(json.dumps(dict(kind=args.kind,source=str(args.source.resolve()),
        datasets=len(paths),max_permutations=49_999,seed=20260919,alpha=.05,
        normalization="RLE on the original native-eligible reference; test mask unchanged",
        inference=fingerprint,native_policy="Reuse checked native fit tables without modification"),indent=2)+"\n")
    shutil.copy2(__file__,args.output/Path(__file__).name)
    shutil.copytree(Path(wade.__file__).parent,args.output/"package-source",ignore=shutil.ignore_patterns('__pycache__'))
    rows=[]
    with ProcessPoolExecutor(args.workers) as pool:
        jobs=[pool.submit(refresh,p,args.output/p.name,args.kind) for p in paths]
        for i,future in enumerate(as_completed(jobs),1):
            batch,record=future.result();rows.extend(batch)
            print(f"{i}/{len(paths)} {record['settings']['dataset']}: {record['seconds']:.1f}s",flush=True)
    assert fingerprint==inference_fingerprint(),"inference changed during benchmark"
    pd.DataFrame(rows).to_csv(args.output/"scores.tsv",sep="\t",index=False)
    if args.kind=="outlier":
        from score_outlier_reference import score
        score(args.output)
    hashes={str(p.relative_to(args.output)):sha256(p) for p in args.output.rglob('*') if p.is_file()}
    (args.output/"checksums.json").write_text(json.dumps(dict(files=hashes,user_approval="pending"),indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kind',choices=['large','reference','outlier'],required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--limit',type=int)
    main(p.parse_args())
