"""Run gate-B truth instruments and bounded historical replay, without archive imports.

Usage from the repository root (requires the optional test dependencies):
  PYTHONPATH=src:research python research/assess_kernels.py --output .release/0.2.0/truth-recovery
Historical input artifacts are local, optional research data, not package assets.
The analytic instrument tests run in a clean checkout without those artifacts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import scipy
from scipy.stats import binomtest

from wade_research import BetaKernelArea, label_coefficients, point_scores, permutation_bank
from wade_research.assessment import (
    beta_distribution, describe, group_w1, log2_ratio, membership_metrics, points,
)

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def pvalue(observed, null, tolerance=1e-14):
    return float((1 + np.count_nonzero(null >= observed-tolerance))/(1 + len(null)))


def input_paths(cfg):
    source = ROOT / cfg['source']
    yield source / 'registry.json'
    registry = json.loads((source/'registry.json').read_text())
    for base in registry:
        if base['id'] not in cfg['cohorts']:
            continue
        yield Path(base['input'])
        yield source / 'inputs' / f"{base['id']}-latent.npz"
        yield Path(base['labels'])
        for scenario in cfg['scenarios']:
            folder = source/'datasets'/f"{base['id']}-r00-{scenario}"
            for name in ('inputs.npz', 'scores.npz', 'panel.tsv'):
                yield folder/name
    failure = ROOT / cfg['failure_replay']['source']
    yield failure/'results.json'
    for row in json.loads((failure/'results.json').read_text())['rows']:
        if row['biological_reference'] == 'point' and row['cases'] == 250:
            yield failure/'data'/f"{row['index']:04d}.npz"
    yield ROOT/'.release/0.2.0/count-kernel-boundaries-20260921/results.json'
    yield ROOT/'.release/0.2.0/kernel-location-diagnostic-20260922/results.json'


def freeze(cfg, protocol, out):
    # Verify availability before creating a run; refuse overwriting earlier evidence.
    inputs = list(input_paths(cfg))
    missing = [str(p) for p in inputs if not p.is_file()]
    if missing:
        raise FileNotFoundError('Local historical inputs required for replay:\n'+'\n'.join(missing))
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(protocol, out/'protocol.json')
    source_files = [*sorted((ROOT/'research').glob('*.py')), *sorted((ROOT/'research/wade_research').glob('*.py')),
                    *sorted((ROOT/'research/tests').glob('*.py'))]
    for path in source_files:
        target = out/'source'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    manifest = dict(started=datetime.now(timezone.utc).isoformat(),
                    checkpoint=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
                    protocol_sha256=sha256(protocol),
                    inputs={str(p.relative_to(ROOT)): sha256(p) for p in inputs},
                    sources={str(p.relative_to(ROOT)): sha256(p) for p in source_files},
                    numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__)
    write(out/'manifest.json', manifest)
    return manifest


def transcriptome_replay(cfg, out):
    source = ROOT/cfg['source']
    registry = json.loads((source/'registry.json').read_text())
    rows, coverage, normalization = [], [], []
    legacy = {'point': 'point', 'beta_h1_mean': 'uniform_mean',
              'beta_h1_kernel': 'uniform', 'beta_h0.5_mean': 'jeffreys_mean',
              'beta_h0.5_kernel': 'jeffreys'}
    for base in registry:
        if base['id'] not in cfg['cohorts']:
            continue
        with np.load(base['input']) as original:
            membership, known = original['membership'], original['known_factors']
            original_labels, original_totals = original['cond'], original['totals']
            original_latent = original['panel_latent_abundance']
        labels = np.load(base['labels'])['labels']
        np.testing.assert_array_equal(labels[0], original_labels)
        case = labels[0].astype(bool)
        truth = np.load(source/'inputs'/f"{base['id']}-latent.npz")['abundance']
        latent_units = truth[:cfg['profiles']]/original_latent[:cfg['profiles']]
        np.testing.assert_allclose(latent_units, latent_units[0, 0], rtol=3e-15)
        for scenario in cfg['scenarios']:
            folder = source/'datasets'/f"{base['id']}-r00-{scenario}"
            data = np.load(folder/'inputs.npz')
            counts, total, factors = data['counts'], data['totals'], data['size_factors']
            unit, shallow, keep = float(data['unit_ratio']), data['shallow'], data['keep']
            np.testing.assert_array_equal(counts.sum(axis=0), total)
            np.testing.assert_allclose(total/original_totals, data['retained_fractions'], rtol=1e-15)
            np.testing.assert_allclose(unit, np.exp(np.log(data['retained_fractions']).mean()), rtol=1e-15)
            # Using s*unit aligns both points and the ENTIRE kernel to original units.
            aligned_factors = factors*unit
            expected = known*data['retained_fractions']/unit
            errors = np.log2(factors/expected)
            normalization.append(dict(cohort=base['id'], scenario=scenario,
                log2_factor_rmse=float(np.sqrt(np.mean(errors**2))),
                case_minus_control_log2_error=float(errors[case].mean()-errors[~case].mean())))
            saved = np.load(folder/'scores.npz')
            table = pd.read_csv(folder/'panel.tsv', sep='\t').set_index(['gene', 'method'])
            for gi, info in enumerate(base['panel'][:cfg['profiles']]):
                y = truth[gi]
                scale = info['baseline_abundance_in_deep_units']
                latent, latent_selected = describe(points(y), labels[0])
                np.testing.assert_allclose(latent['area'], info['true_patient_area'], atol=2e-9)
                target = membership[gi, case].copy()
                if info['pattern'] == 'bulk_1.5':
                    target[:] = True
                elif not info['pattern'].startswith('subset'):
                    target[:] = False
                affected_log2 = log2_ratio(y[case][target].mean(), y[~case].mean()) if target.any() else np.nan
                common = dict(cohort=base['id'], allocation=base['cohort'], biology=base['biology'],
                    scenario=scenario, gene=info['gene'], pattern=info['pattern'], baseline=info['baseline_mean'],
                    eligible=bool(keep[gi]), null=not info['distribution_alternative'],
                    upward=bool(info['upward_biomarker']), planted_fraction=float(target.mean()),
                    true_mean_log2=latent['mean_log2'], population_mean_log2=float(np.log2(info['mean_ratio'])),
                    true_affected_vs_control_log2=affected_log2,
                    true_area=latent['area'], latent_tail_fraction=latent['tail_fraction'])
                latent_members = membership_metrics(target, latent_selected, y[case])
                rows.append(dict(**common, method='latent', **latent, **latent_members,
                                 group_w1_scaled=0., patient_rmse_scaled=0.,
                                 mean_log2_error=0., area_error_scaled=0.,
                                 tail_fold_error=latent['tail_vs_control_log2']-affected_log2))
                representations = [('point', points(counts[gi]/aligned_factors), None)]
                kernel_engines = {}
                for h in (1., .5):
                    engine = BetaKernelArea(counts[gi], total, size_factors=aligned_factors, prior_shape=h)
                    coarse = beta_distribution(engine, cfg['kernel_nodes_per_patient'][0])
                    fine = beta_distribution(engine, cfg['kernel_nodes_per_patient'][1])
                    prefix = f'beta_h{h:g}'
                    kernel_engines[prefix+'_kernel'] = engine
                    representations.extend([(prefix+'_mean', points(fine.means), None),
                                            (prefix+'_kernel', fine, coarse)])
                for method, distribution, coarse in representations:
                    description, selected = describe(distribution, labels[0])
                    record = dict(**common, method=method, **description,
                        **membership_metrics(target, selected, distribution.means[case]),
                        group_w1_scaled=group_w1(distribution, y, labels[0])/scale,
                        patient_rmse_scaled=float(np.sqrt(np.mean((distribution.means-y)**2))/scale),
                        mean_log2_error=description['mean_log2']-latent['mean_log2'],
                        area_error_scaled=(description['area']-latent['area'])/scale,
                        tail_fold_error=description['tail_vs_control_log2']-affected_log2,
                        quantile_repairs=distribution.quantile_repairs,
                        max_cdf_residual=distribution.max_cdf_residual)
                    if coarse is not None:
                        coarse_description, _ = describe(coarse, labels[0])
                        record.update(node_doubling_tail_fraction_change=description['tail_fraction']-coarse_description['tail_fraction'],
                                      node_doubling_area_change_scaled=(description['area']-coarse_description['area'])/scale,
                                      area_numerical_bound_scaled=description['area_numerical_bound']/scale)
                    name = legacy[method]
                    suffix = '_kernel_linear_up' if method.endswith('_kernel') else '_linear_up'
                    old_name = name+suffix
                    stored = table.loc[(info['gene'], old_name)]
                    record.update(pvalue=float(stored.p_upper), called=bool(keep[gi] and stored.p_upper <= .05))
                    # Area error uses the original bounded CDF integral, not the
                    # finite-node approximation needed for curve descriptions.
                    area_lo = area_hi = description['area']
                    if keep[gi]:
                        if coarse is None:
                            replay = point_scores(distribution.means, labels)
                            reference = saved[f'{gi}_{old_name}']/unit
                            np.testing.assert_allclose(replay, reference, atol=2e-8, rtol=2e-10)
                            record['point_bank_max_error'] = float(np.max(abs(replay-reference)))
                        else:
                            lo, hi = saved[f'{gi}_{name}_lower'][0]/unit, saved[f'{gi}_{name}_upper'][0]/unit
                            area_lo, area_hi = lo, hi
                            gap = max(0., lo-description['area'], description['area']-hi)
                            assert gap <= description['area_numerical_bound']+2e-8
                            record['saved_area_bracket_gap'] = gap
                    elif coarse is not None:
                        bounds = kernel_engines[method].scores(label_coefficients(labels[0]))
                        area_lo, area_hi = bounds.lower[0], bounds.upper[0]
                    record.update(cdf_area_lower=float(area_lo), cdf_area_upper=float(area_hi),
                                  area_error_scaled=((area_lo+area_hi)/2-latent['area'])/scale)
                    if distribution.intervals is not None:
                        ci = distribution.intervals
                        covered = (ci[:, 0] <= y) & (y <= ci[:, 1])
                        for stratum, mask in [('all', np.ones(len(y), bool)), ('shallow', shallow),
                                              ('not_shallow', ~shallow), ('observed_zero', counts[gi] == 0)]:
                            if mask.any():
                                coverage.append(dict(cohort=base['id'], scenario=scenario, gene=info['gene'],
                                    baseline=info['baseline_mean'], method=method, stratum=stratum,
                                    patients=int(mask.sum()), covered=int(covered[mask].sum()),
                                    mean_width_scaled=float(np.mean(ci[mask, 1]-ci[mask, 0])/scale)))
                    rows.append(record)
            pd.DataFrame(rows).to_csv(out/'profiles.tsv', sep='\t', index=False)
            print(f"Truth recovery: {base['id']} {scenario}, {len(rows)} rows", flush=True)
    pd.DataFrame(coverage).to_csv(out/'coverage.tsv', sep='\t', index=False)
    pd.DataFrame(normalization).to_csv(out/'normalization.tsv', sep='\t', index=False)


def failure_replay(cfg, out):
    source = ROOT/cfg['failure_replay']['source']
    original = json.loads((source/'results.json').read_text())['rows']
    results = []
    for row in original:
        if row['biological_reference'] != 'point' or row['cases'] != 250:
            continue
        data = np.load(source/'data'/f"{row['index']:04d}.npz")
        labels, total = data['case'], data['total']
        x = data['count']/total
        observed = point_scores(x, labels[None])[0]
        np.testing.assert_allclose(observed, data['observed'][0], atol=1e-14)
        values = {}
        for bank, saved_name, key in [('bootstrap_counts', 'bootstrap_statistics', 'raw'),
                                      ('oracle_counts', 'oracle_area', 'oracle_raw')]:
            current = np.array([point_scores(c/total, labels[None])[0] for c in data[bank]])
            old = data[saved_name][:, 0] if bank == 'bootstrap_counts' else data[saved_name]
            np.testing.assert_allclose(current, old, atol=1e-14)
            values[key] = pvalue(observed, current)
        values['ordinary_permutation'] = pvalue(observed, data['ordinary_area'])
        for key, value in values.items():
            assert value == row['pvalues'][key]
        bank = permutation_bank(labels, 127, np.random.SeedSequence([20260923, row['index']]))
        fresh = point_scores(x, bank)
        values['current_permutation'] = pvalue(fresh[0], fresh[1:])
        for method, value in values.items():
            results.append(dict(index=row['index'], depth_ratio=row['case_depth_ratio'],
                                method=method, pvalue=value, called=value <= .05))
    assert len(results) == 72*4
    frame = pd.DataFrame(results)
    frame.to_csv(out/'failure-replay.tsv', sep='\t', index=False)
    summary = []
    for (ratio, method), cell in frame.groupby(['depth_ratio', 'method']):
        k, n = int(cell.called.sum()), len(cell)
        ci = binomtest(k, n).proportion_ci(method='exact')
        summary.append(dict(depth_ratio=ratio, method=method, rejections=k, datasets=n,
                            rate=k/n, interval95=[ci.low, ci.high]))
    write(out/'failure-summary.json', summary)
    # Keep adverse historical sentinels visible, without presenting them as new fits.
    paths = {'singleton_collapse': ROOT/'.release/0.2.0/count-kernel-boundaries-20260921/results.json',
             'weak_subset': ROOT/'.release/0.2.0/kernel-location-diagnostic-20260922/results.json'}
    write(out/'historical-sentinels.json', {name: dict(status='historical record only; not refitted',
          source=str(path.relative_to(ROOT)), sha256=sha256(path), record=json.loads(path.read_text()))
          for name, path in paths.items()})
    print('Current point statistic reproduces all 72 saved null experiments.', flush=True)


def summarize(out):
    frame = pd.read_csv(out/'profiles.tsv', sep='\t', keep_default_na=False, na_values=[''])
    rows = []
    # Aggregate cohort summaries, not patient-level binomial pseudo-replicates.
    for keys, cell in frame.groupby(['cohort', 'scenario', 'method']):
        row = dict(zip(['cohort', 'scenario', 'method'], keys))
        subset = cell[cell.pattern.str.startswith('subset')]
        called = cell.called.eq(True)
        row.update(profiles=len(cell), true_positive=int((called & cell.upward).sum()),
            positive=int(cell.upward.sum()), false_positive=int((called & cell.null).sum()), nulls=int(cell.null.sum()),
            mean_fold_mae=float(cell.mean_log2_error.abs().mean()),
            mean_fold_bias=float(cell.mean_log2_error.mean()),
            area_rmse_scaled=float(np.sqrt(np.mean(cell.area_error_scaled**2))),
            group_w1_scaled=float(cell.group_w1_scaled.mean()),
            subset_fraction_mae=float(subset.fraction_error.abs().mean()),
            subset_fold_mae=float(subset.tail_fold_error.abs().mean()),
            subset_average_precision=float(subset.average_precision.mean()),
            null_selected_fraction=float(cell.loc[cell.null, 'selected_fraction'].mean()))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'cohort-scorecard.tsv', sep='\t', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'research/protocols/truth-recovery.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.protocol.read_text())
    manifest = freeze(cfg, args.protocol, args.output)
    transcriptome_replay(cfg, args.output)
    failure_replay(cfg, args.output)
    summarize(args.output)
    # Detect concurrent edits rather than mixing executable versions in one record.
    assert all(sha256(ROOT/p) == digest for p, digest in manifest['sources'].items())
    write(args.output/'completion.json', dict(status='INSTRUMENTS_AND_REPLAY_ONLY',
        finished=datetime.now(timezone.utc).isoformat(),
        claims='Reused-data diagnostics; no model selected, no FDR certification, no production changes',
        outputs={p.name: sha256(p) for p in args.output.glob('*') if p.is_file()}))


if __name__ == '__main__':
    main()
