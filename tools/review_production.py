"""Replay frozen transcriptome inputs through the public single-test API.

Compare every retained gene's statistic, stopping records, p-value, references,
and all 18 predeclared BH families. No new simulated data or tuning.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import shutil
import time
import numpy as np
import wade
from notebook_support import inference_fingerprint, sha256


def replay(path, output):
    started = time.monotonic()
    source = Path(path)
    stamp = json.loads((source / "setting.json").read_text())
    arrays = np.load(source / "inputs.npz")
    counts, cond = arrays["counts"], arrays["cond"]
    names = np.array([f"g{j:05}" for j in range(len(counts))])
    frozen = np.load(source / "results.npz")
    factors = np.load(source / "factors.npz")
    used = np.load(source / "used-references.npz")
    cfg = json.loads((source.parent.parent / "protocol.json").read_text())
    tested = (counts > 0).any(axis=1)
    rows = []
    for arm in ("known", "rle", "controls"):
        norm = (wade.Normalization.from_size_factors(counts, arrays["known"], source="known exposure", gene_names=names)
                if arm == "known" else wade.fit_normalization(counts, gene_names=names,
                    reference_genes=None if arm == "rle" else names[:cfg["designated_controls"]]))
        np.testing.assert_allclose(norm.size_factors, factors[arm], rtol=2e-14, atol=0)
        np.testing.assert_array_equal(np.isin(names,norm.used_reference_genes),used[arm])
        result = wade.wade(counts, cond, gene_names=names, normalization=norm,
                           max_permutations=cfg["maximum_permutations"],
                           seed=stamp["permutation_seed"], backend="rust", gene_chunk=len(counts))
        np.testing.assert_array_equal(result.tested,tested)
        error = np.abs(result.statistic[tested]-frozen[arm+"__observed"][tested])
        scale = max(float(np.max(np.abs(frozen[arm+"__observed"][tested]))),1)
        assert np.max(error)/scale < 5e-14
        for current, previous in (("pvalue","p"),("n_exceedances","exceed"),("n_permutations","tried")):
            np.testing.assert_array_equal(getattr(result,current)[tested],frozen[arm+"__"+previous][tested])
        n_families = 0
        for key in frozen.files:
            prefix=arm+"__mask__"
            if not key.startswith(prefix):
                continue
            family=key[len(prefix):]
            mask=frozen[key]
            q=wade.bh_adjust(np.where(mask,result.pvalue,np.nan))
            np.testing.assert_allclose(q,frozen[arm+"__q__"+family],rtol=2e-14,atol=0,equal_nan=True)
            n_families += 1
        rows.append(dict(arm=arm,genes=int(tested.sum()),families=n_families,
                         scaled_statistic_error=float(np.max(error)/scale),
                         gene_permutations=int(result.n_permutations.sum())))
    record=dict(dataset=source.name,source=str(source),seconds=time.monotonic()-started,
                rows=rows,status="PASS",inputs_sha256=sha256(source/"inputs.npz"),
                frozen_results_sha256=sha256(source/"results.npz"))
    destination=Path(output)/source.parent.parent.name/source.name
    destination.mkdir(parents=True,exist_ok=True)
    (destination/"review.json").write_text(json.dumps(record,indent=2)+"\n")
    return record


def main(args):
    paths=[p for root in args.source for p in sorted((root/"datasets").iterdir()) if (p/"results.npz").exists()]
    if args.limit:
        paths=paths[:args.limit]
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    shutil.copytree(root/"src/wade",args.output/"package-source",ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(__file__,args.output/Path(__file__).name)
    fingerprint=inference_fingerprint()
    (args.output/"run.json").write_text(json.dumps(dict(sources=list(map(str,args.source)),datasets=len(paths),
                                                        workers=args.workers,inference=fingerprint),indent=2)+"\n")
    results=[]
    with ProcessPoolExecutor(args.workers) as pool:
        jobs={pool.submit(replay,p,args.output):p for p in paths}
        for future in as_completed(jobs):
            record=future.result(); results.append(record)
            print(f"{len(results)}/{len(paths)} {record['dataset']}: PASS ({record['seconds']:.1f}s)",flush=True)
    assert fingerprint==inference_fingerprint(),"production inference changed during replay"
    summary=dict(status="PASS",datasets=len(results),families=sum(r["families"] for d in results for r in d["rows"]),
                 gene_permutations=sum(r["gene_permutations"] for d in results for r in d["rows"]),
                 max_scaled_statistic_error=max(r["scaled_statistic_error"] for d in results for r in d["rows"]))
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    hashes={str(p.relative_to(args.output)):sha256(p) for p in args.output.rglob('*') if p.is_file()}
    (args.output/"checksums.json").write_text(json.dumps(hashes,indent=2)+"\n")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,action="append",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--limit",type=int)
    main(parser.parse_args())
