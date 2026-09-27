"""Replicated full-transcriptome comparison, frozen before seeing outcomes.

Native DESeq2/edgeR run on the exact saved integer matrices. This executable
does not import archived code or change the public WADE package.
"""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import scipy
import wade

from wade_research.point_factorial import FactorialState, factorial_infer, build_backend, load_backend, LANES
from wade_research.point_simulation import datasets, generate

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def mean_ratio(x, cond):
    a, b = x[:, cond == 1].mean(axis=1), x[:, cond == 0].mean(axis=1)
    valid = (a > 0) & (b > 0)
    result = np.full(len(x), np.nan)
    result[valid] = np.log2(a[valid]/b[valid])
    return result


def operating_point(called, truth):
    called, truth = np.asarray(called, bool), np.asarray(truth, bool)
    tp, fp = int((called & truth).sum()), int((called & ~truth).sum())
    tn, fn = int((~called & ~truth).sum()), int((~called & truth).sum())
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, calls=tp+fp,
                sensitivity=tp/(tp+fn) if tp+fn else np.nan,
                specificity=tn/(tn+fp) if tn+fp else np.nan,
                fdp=fp/max(1, tp+fp), precision=tp/(tp+fp) if tp+fp else np.nan)


def effect_summary(estimate, target, mask):
    finite = mask & np.isfinite(estimate) & np.isfinite(target)
    error = estimate[finite]-target[finite]
    return dict(n=int(mask.sum()), n_finite=int(finite.sum()), n_missing=int((mask & ~finite).sum()),
                rmse=float(np.sqrt(np.mean(error**2))) if len(error) else np.nan,
                mae=float(np.mean(np.abs(error))) if len(error) else np.nan,
                bias=float(np.mean(error)) if len(error) else np.nan)


def score(cfg, setting, data, keep, x, statistics, pvalues, qvalues, folder):
    g = len(keep)
    genes = np.array([f'g{i:05d}' for i in range(g)])
    cond = data['cond']
    profiles = data['profile_indices']
    native = {name: pd.read_csv(folder/f'{name}.tsv', sep='\t', index_col='gene').reindex(genes)
              for name in ('DESeq2', 'edgeR-TMM', 'edgeR-RLE')}
    truth = {'distribution': data['altered'], 'mean': data['mean_altered']}
    estimates = {'empirical_RLE_mean': mean_ratio(x, cond), 'DESeq2': native['DESeq2'].log2FoldChange.to_numpy(),
                 'edgeR-TMM': native['edgeR-TMM'].logFC.to_numpy(), 'edgeR-RLE': native['edgeR-RLE'].logFC.to_numpy()}
    latent_ratio = mean_ratio(data['latent'], cond)
    latent_area = FactorialState(data['latent'], cond).observed
    observed_area = FactorialState(x, cond).observed
    np.testing.assert_array_equal(observed_area[keep], statistics[keep])
    masks = {'all': np.ones(g, bool), 'eligible': keep}
    common = {k: setting[k] for k in ('id', 'panel', 'replicate', 'biology', 'n_case', 'n_control')}
    detection, patterns, effect_rows, area_rows, profile_rows, exclusions = [], [], [], [], [], []
    q_arrays = {}
    for alpha in cfg['workflow']['q_levels']:
        q = {name: np.min(qvalues[:, lanes], axis=1) for name, lanes in LANES.items()}
        q['DESeq2'] = native['DESeq2'][f'q_{alpha:.2f}'].to_numpy()
        q.update({name: native[name].FDR.to_numpy() for name in ('edgeR-TMM', 'edgeR-RLE')})
        q_arrays[f'DESeq2_q_{alpha:.2f}'] = q['DESeq2']
        for method, adjusted in q.items():
            called = adjusted <= alpha
            for target, alternative in truth.items():
                for population, mask in masks.items():
                    detection.append(dict(**common, method=method, alpha=alpha, target=target,
                                          unit='gene', population=population, **operating_point(called[mask], alternative[mask])))
            if alpha == cfg['workflow']['primary_q']:
                for pattern in cfg['generator']['profiles'] + ['shared_tail_null']:
                    for abundance in cfg['generator']['profile_means']:
                        mask = (data['profile'] == pattern) & (data['baseline_mean'] == abundance)
                        patterns.append(dict(**common, method=method, pattern=pattern, abundance=abundance,
                                             total=int(mask.sum()), eligible=int((mask & keep).sum()),
                                             recovered=int((mask & called).sum()),
                                             power=float(called[mask].mean()) if mask.any() else np.nan))
                for gene in profiles:
                    profile_rows.append(dict(**common, method=method, gene=genes[gene], profile=data['profile'][gene],
                                             abundance=data['baseline_mean'][gene], tested=bool(keep[gene]),
                                             q=adjusted[gene], called=bool(called[gene])))
                if method == 'DESeq2':
                    p = native[method].pvalue.to_numpy()
                elif method in native:
                    p = native[method].PValue.to_numpy()
                else:
                    p = np.min(pvalues[:, LANES[method]], axis=1)
                exclusions.append(dict(**common, method=method, input_genes=g, eligible=int(keep.sum()),
                                       filtered_alternatives=int((data['altered'] & ~keep).sum()),
                                       native_p_na=int((keep & ~np.isfinite(p)).sum()),
                                       finite_p_no_q=int((keep & np.isfinite(p) & ~np.isfinite(adjusted)).sum()),
                                       replaced=int(native['DESeq2']['replaced'].eq(True).sum()) if method == 'DESeq2' else 0))
        for method in ('directional_linear', 'directional_log'):
            called = qvalues[:, LANES[method]] <= alpha
            pair_truth = np.repeat(data['altered'][:, None], 2, axis=1)
            detection.append(dict(**common, method=method, alpha=alpha, target='distribution', unit='gene-direction',
                                  population='eligible', **operating_point(called[keep], pair_truth[keep])))
            wrong = ((data['ordered_direction'] == 1) & called[:, 1]
                     | (data['ordered_direction'] == -1) & called[:, 0])
            detection[-1]['wrong_direction_bulk_calls'] = int(wrong.sum())
    selection = {'eligible': keep, 'altered': keep & data['altered'], 'unchanged': keep & ~data['altered'],
                 'ordered_bulk': keep & (data['ordered_direction'] != 0),
                 'distribution_only': keep & data['altered'] & ~data['mean_altered']}
    for estimator, estimate in estimates.items():
        for target, value in [('population', data['population_log2_ratio']), ('realized_latent', latent_ratio)]:
            for population, mask in selection.items():
                effect_rows.append(dict(**common, estimator=estimator, target=target, population=population,
                                        **effect_summary(estimate, value, mask)))
    for scale, transformed, latent in [('linear', x, data['latent']), ('log', np.log2(x+1), np.log2(data['latent']+1))]:
        control_mean = latent[:, cond == 0].mean(axis=1)
        valid = control_mean > 0
        curve_error = np.full(g, np.nan)
        curve_error[valid] = sum(np.mean(np.abs(np.sort(transformed[:, cond == label], axis=1)
                                               - np.sort(latent[:, cond == label], axis=1)), axis=1)[valid]
                                for label in (0, 1))/(2*control_mean[valid])
        for population, mask in selection.items():
            area_rows.append(dict(**common, scale=scale, description='relative_group_curve_error', population=population,
                                  **effect_summary(curve_error, np.zeros(g), mask)))
        offset = 0 if scale == 'linear' else 3
        for lane, description in enumerate(('up_area', 'down_area', 'unsigned_area')):
            for population, mask in selection.items():
                area_rows.append(dict(**common, scale=scale, description=description, population=population,
                                      **effect_summary(observed_area[:, offset+lane], latent_area[:, offset+lane], mask)))
    effect_profiles = []
    for gene in profiles:
        row = dict(gene=genes[gene], profile=data['profile'][gene], abundance=data['baseline_mean'][gene],
                   population_log2=data['population_log2_ratio'][gene], latent_log2=latent_ratio[gene],
                   perturbed_fraction=float(data['perturbed'][gene].mean()), enriched_fraction=float(data['enriched'][gene].mean()))
        row.update({method: estimate[gene] for method, estimate in estimates.items()})
        for lane, name in enumerate(('linear_up', 'linear_down', 'linear_unsigned', 'log_up', 'log_down', 'log_unsigned')):
            row[f'observed_{name}'] = observed_area[gene, lane]
            row[f'latent_{name}'] = latent_area[gene, lane]
        effect_profiles.append(dict(**common, **row))
    tables = dict(detection=detection, patterns=patterns, effects=effect_rows, areas=area_rows,
                  profiles=profile_rows, effect_profiles=effect_profiles, exclusions=exclusions)
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(folder/f'{name}.tsv', sep='\t', index=False)
    np.savez_compressed(folder/'effects.npz', latent_area=latent_area, observed_area=observed_area,
                        latent_mean_log2=latent_ratio, **estimates, **q_arrays)


def run_dataset(cfg, setting, out, library, r_library):
    folder = out/setting['id']
    if (folder/'checksums.json').exists():
        old = json.loads((folder/'checksums.json').read_text())
        for name, value in old['files'].items():
            if sha256(folder/name) != value:
                raise ValueError(f'Changed completed input/output: {folder/name}')
        return json.loads((folder/'audit.json').read_text())
    folder.mkdir(exist_ok=False)
    start = time.monotonic()
    data = generate(cfg, setting)
    counts, cond = data['counts'], data['cond']
    np.savez_compressed(folder/'input.npz', **data)
    norm = wade.fit_normalization(counts)
    x = norm.transform(counts)
    keep = wade.filter_counts(counts, **cfg['workflow']['filter']) & np.any(counts > 0, axis=1)
    genes, samples = [f'g{i:05d}' for i in range(len(counts))], [f's{i:04d}' for i in range(len(cond))]
    pd.DataFrame(counts, index=genes, columns=samples).to_csv(folder/'counts.tsv.gz', sep='\t', index_label='gene')
    pd.DataFrame(dict(sample=samples, condition=np.where(cond, 'case', 'control'), rle_factor=norm.size_factors)).to_csv(folder/'samples.tsv', sep='\t', index=False)
    pd.DataFrame(dict(gene=genes, tested=keep.astype(int))).to_csv(folder/'features.tsv', sep='\t', index=False)
    write_json(folder/'protocol.json', cfg)
    write_json(folder/'settings.json', setting)
    write_json(folder/'normalization.json', norm.to_dict())
    env = os.environ.copy()
    env.update(R_LIBS_USER=str(r_library), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    with (folder/'native.log').open('w') as log:
        proc = subprocess.Popen(['Rscript', '--vanilla', str(out/'source/research/point_comparison_native.R'), str(folder)],
                                stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            tick = time.monotonic()
            answer = factorial_infer(x[keep], cond, maximum=cfg['workflow']['maximum_permutations'],
                                     seed=setting['permutation_seed'], native=load_backend(library),
                                     batch=cfg['workflow']['permutation_batch'])
            point_seconds = time.monotonic()-tick
            statistics, pvalues, qvalues = [np.full((len(keep), 6), np.nan) for _ in range(3)]
            exceed, tried = np.zeros((len(keep), 6), np.uint64), np.zeros((len(keep), 6), np.uint64)
            for full, part in zip((statistics, pvalues, qvalues, exceed, tried), answer):
                full[keep] = part
            tick = time.monotonic()
            baseline = wade.wade(counts, cond, normalization=norm, test_genes=keep,
                                 max_permutations=cfg['workflow']['maximum_permutations'], seed=setting['permutation_seed'], backend='rust')
            baseline_seconds = time.monotonic()-tick
            for full, target in ((statistics, baseline.statistic), (pvalues, baseline.pvalue), (qvalues, baseline.padj),
                                 (exceed, baseline.n_exceedances), (tried, baseline.n_permutations)):
                np.testing.assert_array_equal(full[:, 5], target)
            np.savez_compressed(folder/'point-results.npz', statistic=statistics, pvalue=pvalues, padj=qvalues,
                                exceed=exceed, tried=tried, tested=keep, factors=norm.size_factors)
            if proc.wait() != 0:
                raise RuntimeError(f'Native comparator failed; inspect {folder / "native.log"}')
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
    score(cfg, setting, data, keep, x, statistics, pvalues, qvalues, folder)
    result = dict(**setting, seconds=time.monotonic()-start, point_seconds=point_seconds,
                  public_baseline_seconds=baseline_seconds, native=json.loads((folder/'native.json').read_text()),
                  genes=len(counts), tested=int(keep.sum()), alternatives=int(data['altered'].sum()),
                  filtered_alternatives=int((data['altered'] & ~keep).sum()), baseline_parity='PASS',
                  size_factor_log2_rmse=float(np.sqrt(np.mean(np.log2(norm.size_factors/data['exposure'])**2))),
                  gene_statistic_permutations=int(tried.sum()),
                  joint_singleton_bh_floor=2*int(keep.sum())/(cfg['workflow']['maximum_permutations']+1))
    write_json(folder/'audit.json', result)
    write_json(folder/'checksums.json', dict(files={p.name: sha256(p) for p in folder.iterdir() if p.is_file()}))
    return result


def main(args):
    cfg = json.loads(args.protocol.read_text())
    out = args.output.resolve()
    source = [Path(__file__).resolve(), ROOT/'research/point_comparison_native.R', ROOT/'research/point_permutations.rs',
              ROOT/'research/wade_research/point_factorial.py', ROOT/'research/wade_research/point_simulation.py',
              ROOT/'research/tests/test_point_factorial.py', ROOT/'research/tests/test_point_simulation.py',
              ROOT/'research/tests/test_point_comparison.py',
              *sorted((ROOT/'src/wade').glob('*.py')), *sorted((ROOT/'src/wade').glob('*.so'))]
    fingerprint = {str(p.relative_to(ROOT)): sha256(p) for p in source}
    if args.resume:
        manifest = json.loads((out/'manifest.json').read_text())
        if manifest['sources'] != fingerprint or manifest['protocol_sha256'] != sha256(args.protocol):
            raise ValueError('Cannot resume with changed sources or protocol')
        library = Path(manifest['compiled_backend'])
        if sha256(library) != manifest['compiled_sha256']:
            raise ValueError('Compiled backend changed')
    else:
        out.mkdir(parents=True, exist_ok=False)
        for path in source:
            target = out/'source'/path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        shutil.copy2(args.protocol, out/'protocol.json')
        library = build_backend(out/'compiled')
        manifest = dict(started=datetime.now(timezone.utc).isoformat(), protocol_sha256=sha256(args.protocol), sources=fingerprint,
                        compiled_backend=str(library), compiled_sha256=sha256(library),
                        rustc=subprocess.check_output(['rustc', '--version']).decode().strip(),
                        numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__,
                        r_library=str(args.r_library.resolve()), workers=args.workers,
                        phase=cfg['phase'], user_approval='pending')
        write_json(out/'manifest.json', manifest)
        write_json(out/'registry.json', list(datasets(cfg)))
    jobs = iter(list(datasets(cfg)))
    audits, pending = [], {}
    pool = ProcessPoolExecutor(max_workers=args.workers)
    try:
        def submit():
            job = next(jobs, None)
            if job is not None:
                pending[pool.submit(run_dataset, cfg, job, out, library, args.r_library.resolve())] = job
        for _ in range(args.workers):
            submit()
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                setting = pending.pop(future)
                result = future.result()
                audits.append(result)
                print(f'{len(audits)}/{len(list(datasets(cfg)))} {setting["id"]} {setting["biology"]}: '
                      f'{result["seconds"]:.1f}s, {result["tested"]} eligible, baseline parity PASS', flush=True)
                submit()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    if fingerprint != {str(p.relative_to(ROOT)): sha256(p) for p in source}:
        raise RuntimeError('Source changed during the run')
    names = ('detection', 'patterns', 'effects', 'areas', 'profiles', 'effect_profiles', 'exclusions')
    for name in names:
        frames = [pd.read_csv(out/job['id']/f'{name}.tsv', sep='\t') for job in datasets(cfg)]
        pd.concat(frames, ignore_index=True).to_csv(out/f'{name}.tsv', sep='\t', index=False)
    write_json(out/'audit.json', dict(complete=True, datasets=sorted(audits, key=lambda r: r['id']), user_approval='pending'))
    write_json(out/'checksums.json', dict(files={str(p.relative_to(out)): sha256(p) for p in out.rglob('*')
                                               if p.is_file() and p != out/'checksums.json'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=ROOT/'research/protocols/point-comparison.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--r-library', type=Path, default=ROOT/'.release/0.2.0/r-library')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    main(p.parse_args())
