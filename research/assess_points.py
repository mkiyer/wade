"""Frozen raw-count directional workflow audit, with no historical data dependency.

This first run checks the instruments, not the release's scientific performance.
Use --output to create a new immutable run; reruns refuse existing destinations.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time

import numpy as np
import pandas as pd
import wade

from wade_research.point_workflow import DirectionalState, point_workflow

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, record):
    Path(path).write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')


def generate(settings, panel):
    """Known biological abundances and raw integer observations, with saved truth."""
    rng = np.random.default_rng(settings['seeds'][panel])
    genes, nc, n0 = settings['genes'], settings['n_case'], settings['n_control']
    cond = np.r_[np.ones(nc, int), np.zeros(n0, int)]
    exposure = np.exp(rng.normal(0, .35, len(cond)))
    exposure /= np.exp(np.log(exposure).mean())
    mean = np.exp(rng.uniform(np.log(.05), np.log(1000), genes))
    mean[:64] = 100
    cv2 = rng.uniform(.2, 1, genes)
    patterns = np.full(genes, 'null', dtype='<U48')
    if panel == 'mixed':
        profile_means = np.repeat(settings['profile_means'], len(settings['mixed_profiles']))
        mean[64:64+len(profile_means)] = profile_means
        patterns[64:64+len(profile_means)] = np.tile(settings['mixed_profiles'], len(settings['profile_means']))
    latent = rng.gamma(1/cv2[:, None], mean[:, None]*cv2[:, None], (genes, len(cond)))
    altered = patterns != 'null'
    affected = np.zeros((genes, nc), bool)
    population_mean_ratio = np.ones(genes)
    ordered_direction = np.zeros(genes, int)
    for gene in np.flatnonzero(altered):
        pattern = patterns[gene]
        members = rng.choice(nc, max(1, round(.1*nc)), replace=False)
        multiplier = np.ones(nc)
        if pattern == 'bulk_up_2x':
            multiplier[:] = 2
            ordered_direction[gene] = 1
        elif pattern == 'bulk_down_half':
            multiplier[:] = .5
            ordered_direction[gene] = -1
        elif pattern == 'subset_up_10pct_8x':
            multiplier[members] = 8
        elif pattern == 'subset_down_10pct_eighth':
            multiplier[members] = .125
        elif pattern == 'bulk_half_subset_up_10pct_4x':
            multiplier[:] = .5
            multiplier[members] = 4
        elif pattern == 'bulk_2x_subset_down_10pct_quarter':
            multiplier[:] = 2
            multiplier[members] = .25
        elif pattern == 'same_mean_wider':
            latent[gene, :nc] = rng.gamma(1/(4*cv2[gene]), mean[gene]*4*cv2[gene], nc)
        elif pattern == 'singleton_up_16x':
            multiplier[rng.integers(nc)] = 16
        else:
            raise ValueError(f'Unknown profile: {pattern}')
        latent[gene, :nc] *= multiplier
        affected[gene] = multiplier != 1
        population_mean_ratio[gene] = multiplier.mean()
    latent[-1] = 0
    counts = rng.poisson(latent*exposure)
    return dict(counts=counts, latent=latent, cond=cond, exposure=exposure,
                altered=altered, patterns=patterns, affected=affected,
                population_mean_ratio=population_mean_ratio, ordered_direction=ordered_direction)


def operating_point(called, truth):
    called, truth = np.asarray(called, bool), np.asarray(truth, bool)
    tp, fp = int((called & truth).sum()), int((called & ~truth).sum())
    tn, fn = int((~called & ~truth).sum()), int((~called & truth).sum())
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, calls=tp+fp,
                sensitivity=tp/(tp+fn) if tp+fn else None,
                specificity=tn/(tn+fp) if tn+fp else None, fdp=fp/max(1, tp+fp))


def mean_ratio(x, cond):
    case, control = x[:, cond == 1].mean(axis=1), x[:, cond == 0].mean(axis=1)
    valid = (case > 0) & (control > 0)
    result = np.full(len(x), np.nan)
    result[valid] = np.log2(case[valid]/control[valid])
    return result


def assess(cfg, panel, out):
    start = time.monotonic()
    folder = out/panel
    folder.mkdir()
    data = generate(cfg['engineering'], panel)
    np.savez_compressed(folder/'input.npz', **data)
    counts, cond, truth = data['counts'], data['cond'], data['altered']
    settings = cfg['candidate']
    norm = wade.fit_normalization(counts)
    keep = wade.filter_counts(counts, **settings['filter'])
    write_json(folder/'normalization.json', norm.to_dict())
    common = dict(normalization=norm, test_genes=keep,
                  max_permutations=settings['maximum_permutations'],
                  seed=cfg['engineering']['permutation_seed'])
    candidate = point_workflow(counts, cond, scale=settings['scale'], **common)
    print(f'{panel}: directional candidate complete in {time.monotonic()-start:.1f}s', flush=True)
    baseline = wade.wade(counts, cond, **common)
    np.testing.assert_array_equal(candidate.tested, baseline.tested)
    np.testing.assert_array_equal(candidate.normalization.size_factors, baseline.normalization.size_factors)
    # Realized finite-cohort targets include ordinary sampling differences in null genes.
    true_area = DirectionalState(data['latent'], cond).observed
    true_ratio = mean_ratio(data['latent'], cond)
    estimate_ratio = mean_ratio(candidate.expression, cond)
    np.savez_compressed(folder/'results.npz',
                        point_statistic=candidate.statistic, point_pvalue=candidate.pvalue,
                        point_padj=candidate.padj, point_tried=candidate.n_permutations,
                        point_exceed=candidate.n_exceedances, status=candidate.status,
                        baseline_statistic=baseline.statistic, baseline_pvalue=baseline.pvalue,
                        baseline_padj=baseline.padj, baseline_tried=baseline.n_permutations,
                        baseline_exceed=baseline.n_exceedances,
                        latent_area=true_area, latent_mean_log2_ratio=true_ratio,
                        point_mean_log2_ratio=estimate_ratio, factors=norm.size_factors)
    alpha = settings['alpha']
    directional_calls = candidate.padj <= alpha
    point_called = directional_calls.any(axis=1)
    baseline_called = baseline.padj <= alpha
    rows = []
    for name, calls in [('directional_linear', point_called), ('unsigned_log_baseline', baseline_called)]:
        # Include excluded genes in the primary intention-to-test denominator.
        # Also publish eligible-only counts, so filtering cannot disguise losses.
        rows.append(dict(panel=panel, method=name, unit='gene', population='all',
                         **operating_point(calls, truth)))
        rows.append(dict(panel=panel, method=name, unit='gene', population='eligible',
                         **operating_point(calls[keep], truth[keep])))
    rows.append(dict(panel=panel, method='directional_linear', unit='gene-direction', population='eligible',
                     **operating_point(directional_calls[keep], np.repeat(truth[keep, None], 2, axis=1))))
    np.testing.assert_array_equal(keep, candidate.tested)
    selected = np.r_[np.arange(64, 64+len(cfg['engineering']['mixed_profiles'])*len(cfg['engineering']['profile_means'])),
                     np.arange(16)]
    profile = pd.DataFrame(dict(gene=selected, pattern=data['patterns'][selected],
                                population_mean_ratio=data['population_mean_ratio'][selected],
                                latent_mean_log2=true_ratio[selected], observed_mean_log2=estimate_ratio[selected],
                                mean_log2_error=(estimate_ratio-true_ratio)[selected],
                                area_up=candidate.statistic[selected, 0], area_down=candidate.statistic[selected, 1],
                                latent_area_up=true_area[selected, 0], latent_area_down=true_area[selected, 1],
                                q_up=candidate.padj[selected, 0], q_down=candidate.padj[selected, 1],
                                q_unsigned_log=baseline.padj[selected], tested=candidate.tested[selected]))
    profile.to_csv(folder/'profiles.tsv', sep='\t', index=False)
    wrong = ((data['ordered_direction'] == 1) & directional_calls[:, 1]
             | (data['ordered_direction'] == -1) & directional_calls[:, 0])
    audit = dict(panel=panel, elapsed_seconds=time.monotonic()-start,
                 input_sha256=sha256(folder/'input.npz'), genes=len(counts), tested=int(keep.sum()),
                 altered=int(truth.sum()), filtered_alternatives=int((truth & ~keep).sum()),
                 size_factor_log2_rmse=float(np.sqrt(np.mean(np.log2(norm.size_factors/data['exposure'])**2))),
                 wrong_direction_bulk_calls=int(wrong.sum()), ordered_bulk_genes=int(np.count_nonzero(data['ordered_direction'])),
                 candidate=candidate.params, baseline=baseline.params,
                 point_gene_direction_permutations=int(candidate.n_permutations.sum()),
                 baseline_gene_permutations=int(baseline.n_permutations.sum()))
    write_json(folder/'audit.json', audit)
    write_json(folder/'operating-points.json', rows)
    print(f'{panel}: finished {audit["tested"]} tested genes in {audit["elapsed_seconds"]:.1f}s', flush=True)
    return rows, audit


def main(args):
    cfg = json.loads(args.protocol.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.protocol, args.output/'protocol.json')
    sources = [Path(__file__), ROOT/'research/wade_research/point_workflow.py',
               ROOT/'research/tests/test_point_workflow.py', ROOT/'research/tests/test_point_assessment.py',
               *sorted((ROOT/'src/wade').glob('*.py')),
               *sorted((ROOT/'src/wade').glob('*.so'))]
    for path in sources:
        target = args.output/'source'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    before = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
    write_json(args.output/'manifest.json', dict(started=datetime.now(timezone.utc).isoformat(),
               protocol_sha256=sha256(args.protocol), sources=before, numpy=np.__version__, pandas=pd.__version__,
               interpretation='Engineering audit, two matrices; no FDR or comparative-performance certification'))
    rows, audits = [], []
    for panel in cfg['engineering']['seeds']:
        print(f'{panel}: starting raw-count workflow', flush=True)
        result, audit = assess(cfg, panel, args.output)
        rows.extend(result)
        audits.append(audit)
    if before != {str(p.relative_to(ROOT)): sha256(p) for p in sources}:
        raise RuntimeError('Evaluated sources changed during execution')
    pd.DataFrame(rows).to_csv(args.output/'operating-points.tsv', sep='\t', index=False)
    write_json(args.output/'audit.json', dict(complete=True, datasets=audits, user_approval='pending'))
    files = {str(p.relative_to(args.output)): sha256(p) for p in args.output.rglob('*') if p.is_file()}
    write_json(args.output/'checksums.json', dict(files=files))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'research/protocols/point-release.json')
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
