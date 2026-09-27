"""Audit a saved subset failure without changing the method or its truth labels.

Exploratory diagnostics, not a power study: independently integrate all 80
balanced depressed-bulk profiles, replay two original count generators, use
fixed-budget direct-sort permutations on four disclosed profiles, and enumerate
a three-level toy's entire label distribution by category counts.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import shutil

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import beta

from wade_research.point_simulation import generate


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def areas(case, control):
    """Equal-size empirical quantile integration, independent of the CDF engine."""
    difference = np.sort(case, axis=-1)-np.sort(control, axis=-1)
    return np.stack((np.maximum(difference, 0).mean(axis=-1),
                     np.maximum(-difference, 0).mean(axis=-1),
                     np.abs(difference).mean(axis=-1)), axis=-1)


def three_level(n, spikes, fold, scale):
    """Enumerate the multivariate-hypergeometric distribution of label counts.

Observed: n controls at 100; n-spikes cases at 50; spikes cases at 100*fold.
This is a deterministic algebra check, not simulated RNA-seq or a power estimate.
"""
    levels = np.array([50., 100., 100.*fold])
    if scale == 'log':
        levels = np.log2(levels+1)
    gap = np.diff(levels)
    observed = spikes*gap[1]/n

    def logchoose(total, chosen):
        return gammaln(total+1)-gammaln(chosen+1)-gammaln(total-chosen+1)

    probability = total_mass = 0.
    for high_case in range(spikes+1):
        for low_case in range(n-spikes+1):
            middle_case = n-low_case-high_case
            if not 0 <= middle_case <= n:
                continue
            mass = np.exp(logchoose(spikes, high_case)+logchoose(n-spikes, low_case)
                          +logchoose(n, middle_case)-logchoose(2*n, n))
            score = (max(n-spikes-2*low_case, 0)*gap[0]
                     +max(2*high_case-spikes, 0)*gap[1])/n
            probability += mass*(score >= observed*(1-100*np.finfo(float).eps))
            total_mass += mass
    assert abs(total_mass-1) < 2e-11
    return observed, float(probability)


def check_toy():
    for scale in ('linear', 'log'):
        x = np.array([50., 400., 100., 100.])
        if scale == 'log':
            x = np.log2(x+1)
        observed, expected = three_level(2, 1, 4, scale)
        hits = []
        for indices in itertools.combinations(range(4), 2):
            mask = np.zeros(4, bool)
            mask[list(indices)] = True
            value = areas(x[mask], x[~mask])[0]
            hits.append(value >= observed*(1-100*np.finfo(float).eps))
        np.testing.assert_allclose(expected, np.mean(hits), atol=1e-14)


def main(args):
    source, out = args.source.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out/'audit_subset_failure.py')
    cfg = json.loads((source/'protocol.json').read_text())
    registry = json.loads((source/'registry.json').read_text())
    hashes = json.loads((source/'checksums.json').read_text())['files']
    selected = [r for r in registry if r['panel'] == 'mixed' and r['n_case'] == r['n_control'] == 250]
    # r00 is the first gamma replicate; r15 contains the smallest saved linear
    # upward p-value for this pattern. Deliberate diagnostic selection, not random.
    replay = {'mixed-250v250-r00', 'mixed-250v250-r15'}
    protocol = dict(source=str(source), pattern='bulk_half_subset_up_10pct_4x',
                    population='All 80 balanced profiles for recount; four selected profiles for fixed permutations',
                    chosen_datasets=sorted(replay), chosen_means=[.5, 500],
                    views=['RLE counts', 'true-exposure counts', 'latent biological abundance'],
                    permutation_draws=args.permutations, seed=2026092603,
                    selection='First gamma replicate and replicate with smallest saved upward p, disclosed post hoc',
                    interpretation='Exploratory diagnostic, not independent confirmation or a new production method')
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    profiles, factors, fixed_inputs, replays = [], [], [], []
    max_difference = 0.
    for setting in selected:
        folder = source/setting['id']
        for name in ('input.npz', 'point-results.npz'):
            assert sha256(folder/name) == hashes[f'{setting["id"]}/{name}']
        with np.load(folder/'input.npz') as data, np.load(folder/'point-results.npz') as result:
            cond = data['cond']
            np.testing.assert_array_equal(cond, np.r_[np.ones(250, int), np.zeros(250, int)])
            indices = np.flatnonzero(data['profile'] == protocol['pattern'])
            counts, latent = data['counts'][indices], data['latent'][indices]
            x = counts/result['factors']
            for offset, values in ((0, x), (3, np.log2(x+1))):
                computed = areas(values[:, :250], values[:, 250:])
                saved = result['statistic'][indices, offset:offset+3]
                np.testing.assert_allclose(computed, saved, atol=1e-11, rtol=2e-12)
                max_difference = max(max_difference, float(np.abs(computed-saved).max()))
            factor_error = np.log2(result['factors']/data['exposure'])
            factors.append(dict(dataset=setting['id'], biology=setting['biology'],
                                log2_factor_rmse=float(np.sqrt(np.mean(factor_error**2))),
                                case_minus_control_error=float(factor_error[:250].mean()-factor_error[250:].mean())))
            for j, gene in enumerate(indices):
                enriched = data['enriched'][gene]
                assert enriched.sum() == 25 and data['perturbed'][gene].sum() == 250
                profiles.append(dict(dataset=setting['id'], gene=int(gene), biology=setting['biology'],
                                     mean=float(data['baseline_mean'][gene]), cv2=float(data['cv2'][gene]),
                                     tested=bool(result['tested'][gene]),
                                     p_up_linear=float(result['pvalue'][gene, 0]),
                                     q_up_linear=float(result['padj'][gene, 0]),
                                     p_up_log=float(result['pvalue'][gene, 3]),
                                     q_up_log=float(result['padj'][gene, 3]),
                                     latent_mean_ratio=float(latent[j, :250].mean()/latent[j, 250:].mean()),
                                     enriched_above_control_max=float((latent[j, :250][enriched] > latent[j, 250:].max()).mean())))
                if setting['id'] in replay and data['baseline_mean'][gene] in (.5, 500):
                    for view, values in [('RLE counts', x[j]), ('true-exposure counts', counts[j]/data['exposure']),
                                         ('latent biological abundance', latent[j])]:
                        fixed_inputs.append((dict(dataset=setting['id'], gene=int(gene), mean=float(data['baseline_mean'][gene]),
                                                  view=view), values.copy()))
            if setting['id'] in replay:
                regenerated = generate(cfg, setting)
                for key in ('counts', 'latent', 'cond', 'exposure', 'profile', 'enriched', 'perturbed'):
                    np.testing.assert_array_equal(regenerated[key], data[key])
                replays.append(setting['id'])
                del regenerated
        print(f'Audited {setting["id"]}', flush=True)
    pd.DataFrame(profiles).to_csv(out/'profiles.tsv', sep='\t', index=False)
    pd.DataFrame(factors).to_csv(out/'normalization.tsv', sep='\t', index=False)

    rng = np.random.default_rng(protocol['seed'])
    observed = np.array([[areas(values[:250], values[250:])[0],
                          areas(np.log2(values[:250]+1), np.log2(values[250:]+1))[0]]
                         for _, values in fixed_inputs])
    hits = np.zeros(observed.shape, int)
    for start in range(0, args.permutations, 256):
        order = np.argsort(rng.random((min(256, args.permutations-start), 500)), axis=1)
        for i, (_, values) in enumerate(fixed_inputs):
            case, control = np.sort(values[order[:, :250]], axis=1), np.sort(values[order[:, 250:]], axis=1)
            scores = np.column_stack((np.maximum(case-control, 0).mean(axis=1),
                                      np.maximum(np.log2(case+1)-np.log2(control+1), 0).mean(axis=1)))
            hits[i] += (scores >= observed[i]*(1-100*np.finfo(float).eps)).sum(axis=0)
    results = []
    for i, (identity, _) in enumerate(fixed_inputs):
        for lane, scale in enumerate(('linear', 'log')):
            k, b = int(hits[i, lane]), args.permutations
            results.append(dict(**identity, scale=scale, area=float(observed[i, lane]), hits=k, permutations=b,
                                pvalue=(k+1)/(b+1),
                                probability_low=0 if k == 0 else beta.ppf(.025, k, b-k+1),
                                probability_high=1 if k == b else beta.ppf(.975, k+1, b-k)))
    pd.DataFrame(results).to_csv(out/'fixed-permutations.tsv', sep='\t', index=False)
    check_toy()
    toy = []
    for fold in (4, 16, 32, 1024):
        for scale in ('linear', 'log'):
            area, pvalue = three_level(250, 25, fold, scale)
            toy.append(dict(cases=250, controls=250, spikes=25, fold=fold, scale=scale, area=area, pvalue=pvalue))
    pd.DataFrame(toy).to_csv(out/'three-level-toy.tsv', sep='\t', index=False)
    audit = dict(complete=True, checked_source_files=2*len(selected), source_checksums_sha256=sha256(source/'checksums.json'),
                 regenerated_exact=replays, independent_area_max_difference=max_difference,
                 fixed_permutation_implementation='Directly sort each randomized group; independent RNG and fixed budget; no production scoring/stopping code',
                 exact_toy_check='Category-count enumeration checked against all six assignments in a 2:2 example',
                 user_approval='pending', interpretation=protocol['interpretation'])
    (out/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    (out/'checksums.json').write_text(json.dumps({p.name: sha256(p) for p in out.iterdir() if p.is_file()}, indent=2)+'\n')
    print('Subset failure diagnostic complete.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--permutations', type=int, default=49999)
    args = parser.parse_args()
    if args.permutations < 1:
        parser.error('--permutations must be positive')
    main(args)
