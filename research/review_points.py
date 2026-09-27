"""Independent result audit, matrix-level summaries and scientific figures.

Run only after compare_points.py completes. Figures and claims remain pending
user review. The source inference outputs are read-only.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
from scipy.stats import beta, false_discovery_control, hypergeom

from wade.diagnostics import gene_detail

METHODS = ['unsigned_log', 'unsigned_linear', 'directional_log', 'directional_linear', 'DESeq2', 'edgeR-TMM', 'edgeR-RLE']
LABELS = {'unsigned_log': 'WADE unsigned · log (baseline)', 'unsigned_linear': 'WADE unsigned · linear',
          'directional_log': 'WADE directional · log', 'directional_linear': 'WADE directional · linear',
          'DESeq2': 'DESeq2', 'edgeR-TMM': 'edgeR QL · TMM', 'edgeR-RLE': 'edgeR QL · matched RLE',
          'empirical_RLE_mean': 'Empirical RLE group means'}
COLORS = dict(zip(METHODS, ['#234c78', '#8ba8be', '#825798', '#188477', '#bf6829', '#a34247', '#6d6d6d']))
COLORS['empirical_RLE_mean'] = '#188477'
PATTERNS = {'bulk_up_1.5x': 'Bulk up\n1.5×', 'bulk_down_two_thirds': 'Bulk down\n2/3×',
            'subset_up_5pct_4x': '5% up\n4×', 'subset_up_10pct_4x': '10% up\n4×',
            'subset_up_20pct_4x': '20% up\n4×', 'subset_up_10pct_16x': '10% up\n16×',
            'subset_down_10pct_quarter': '10% down\n1/4×',
            'bulk_half_subset_up_10pct_4x': 'Bulk half +\n10% up 4×',
            'same_mean_wider': 'Same mean,\nwider',
            'mean_preserving_subset_10pct_4x': 'Mean-neutral\n10% up 4×',
            'opposing_20pct': 'Opposing\n20% tails', 'singleton_up_16x': 'One case\nup 16×'}
LANES = {'directional_linear': [0, 1], 'unsigned_linear': [2], 'directional_log': [3, 4], 'unsigned_log': [5]}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def two_level_tail(ncase, ncontrol, spikes):
    """Exact toy probabilities: identical baseline, K higher cases, no high controls.

    Conditional on the observed two-valued sample, H high values assigned to
    controls is hypergeometric. Integer arithmetic identifies inclusive W1 ties.
    This illustrates a finite-control limitation; it is not a bound for every
    continuous gene profile, and it is separate from simulated-study outcomes.
    """
    if not (ncase > 0 and ncontrol > 0 and 0 < spikes <= ncase):
        raise ValueError('Require two groups and one or more spikes among cases')
    high_controls = np.arange(min(spikes, ncontrol)+1)
    probabilities = hypergeom.pmf(high_controls, ncase+ncontrol, spikes, ncontrol)
    mass = ncontrol*spikes-(ncase+ncontrol)*high_controls
    return dict(p_up=float(probabilities[0]),
                p_unsigned=float(probabilities[np.abs(mass) >= ncontrol*spikes].sum()))


def interval(values, strata, *, binary=False, bounded=True, seed=2026092602, draws=10000):
    """Stratified matrix bootstrap; exact per-stratum bounds for null events.

    Pooled binary bounds average Bonferroni-adjusted stratum bounds. They are
    conservative, not an assertion that different generators share a rate.
    All-zero/all-one bounded samples use the same conservative boundary envelope.
    Other bootstrap intervals describe finite-replicate uncertainty, not a proof
    of nominal FDR or an equivalence test.
    """
    values, strata = np.asarray(values, float), np.asarray(strata)
    valid = np.isfinite(values)
    values, strata = values[valid], strata[valid]
    if not len(values):
        return dict(n=0, mean=np.nan, low=np.nan, high=np.nan)
    groups = [values[strata == value] for value in sorted(set(strata))]
    weights = np.array([len(x) for x in groups], float)/len(values)
    average = float(values.mean())
    if binary or (bounded and (np.all(values == 0) or np.all(values == 1))):
        lower, upper = [], []
        for group in groups:
            n, k = len(group), int(group.sum())
            alpha = .05/len(groups)
            lower.append(0 if k == 0 else beta.ppf(alpha/2, k, n-k+1))
            upper.append(1 if k == n else beta.ppf(1-alpha/2, k+1, n-k))
        lo, hi = float(weights@lower), float(weights@upper)
    else:
        rng = np.random.default_rng(seed)
        resampled = sum(weight*rng.choice(group, size=(draws, len(group)), replace=True).mean(axis=1)
                        for weight, group in zip(weights, groups))
        lo, hi = np.quantile(resampled, [.025, .975])
        if bounded:
            lo, hi = max(0, lo), min(1, hi)
    return dict(n=len(values), mean=average, low=float(lo), high=float(hi))


def summarize(frame, keys, value, *, binary=False, bounded=True):
    rows = []
    # Keep model-specific results as well as the prespecified equal-weight mixture.
    for group_key, group in frame.groupby(keys, dropna=False, sort=True):
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        for biology in ['pooled', *sorted(group.biology.unique())]:
            selected = group if biology == 'pooled' else group[group.biology == biology]
            result = interval(selected[value], selected.biology, binary=binary, bounded=bounded)
            rows.append(dict(zip(keys, group_key), biology=biology, metric=value, **result))
    return pd.DataFrame(rows)


def audit_dataset(folder):
    folder = Path(folder)
    cfg = json.loads((folder/'protocol.json').read_text())
    maximum = cfg['workflow']['maximum_permutations']
    with np.load(folder/'input.npz') as data, np.load(folder/'point-results.npz') as result, np.load(folder/'effects.npz') as effect:
        counts, cond, factors = data['counts'], data['cond'], result['factors']
        meta = pd.read_csv(folder/'samples.tsv', sep='\t')
        native_counts = pd.read_csv(folder/'counts.tsv.gz', sep='\t', index_col=0)
        np.testing.assert_array_equal(native_counts.to_numpy(), counts)
        assert native_counts.columns.tolist() == meta['sample'].tolist()
        np.testing.assert_array_equal(meta.condition.eq('case').to_numpy(), cond == 1)
        features = pd.read_csv(folder/'features.tsv', sep='\t')
        assert native_counts.index.tolist() == features.gene.tolist()
        keep = (counts >= cfg['workflow']['filter']['min_count']).sum(axis=1) >= cfg['workflow']['filter']['min_samples']
        np.testing.assert_array_equal(keep, result['tested'])
        np.testing.assert_array_equal(features.tested, keep.astype(int))
        k, t = result['exceed'][keep], result['tried'][keep]
        assert np.all((k == 20) | (t == maximum))
        assert np.all(k <= 20) and np.all(k <= t) and np.all(t <= maximum)
        p = np.where(k == 20, 20/np.maximum(t, 1), (k+1)/(maximum+1))
        np.testing.assert_array_equal(p, result['pvalue'][keep])
        assert np.all(p >= 1/(maximum+1))
        for lanes in LANES.values():
            q = false_discovery_control(p[:, lanes].ravel()).reshape(-1, len(lanes))
            np.testing.assert_allclose(q, result['padj'][keep][:, lanes], rtol=2e-15, atol=2e-16)
        assert np.isnan(result['pvalue'][~keep]).all()
        assert np.all(result['tried'][~keep] == 0)
        x = counts/factors
        largest = 0.
        # Independent quantile integration, not the CDF/native code used to fit.
        for gene in data['profile_indices']:
            for offset, transformed in ((0, x[gene]), (3, np.log2(x[gene]+1))):
                detail = gene_detail(transformed, cond)
                weight = np.diff(detail.probability)
                want = np.array([np.sum(np.maximum(detail.effect, 0)*weight),
                                 np.sum(np.maximum(-detail.effect, 0)*weight), np.sum(np.abs(detail.effect)*weight)])
                got = effect['observed_area'][gene, offset:offset+3]
                np.testing.assert_allclose(want, got, rtol=2e-12, atol=2e-11)
                largest = max(largest, float(np.max(np.abs(want-got))))
        # Recount every procedure's primary gene calls from raw q tables and truth.
        scores = pd.read_csv(folder/'detection.tsv', sep='\t')
        primary = scores[(scores.alpha == .05) & (scores.unit == 'gene') & (scores.target == 'distribution') & (scores.population == 'all')]
        for row in primary.itertuples():
            if row.method in LANES:
                calls = (result['padj'][:, LANES[row.method]] <= .05).any(axis=1)
            else:
                table = pd.read_csv(folder/f'{row.method}.tsv', sep='\t', index_col='gene').reindex(features.gene)
                calls = table['q_0.05' if row.method == 'DESeq2' else 'FDR'].le(.05).to_numpy()
            truth = data['altered']
            assert row.tp == np.count_nonzero(calls & truth)
            assert row.fp == np.count_nonzero(calls & ~truth)
            assert row.tn == np.count_nonzero(~calls & ~truth)
            assert row.fn == np.count_nonzero(~calls & truth)
    return dict(dataset=folder.name, genes=counts.shape[0], samples=counts.shape[1],
                tested=int(keep.sum()), max_quantile_area_difference=largest, status='PASS')


def tables(source, out):
    detection = pd.read_csv(source/'detection.tsv', sep='\t')
    mixed = detection[(detection.panel == 'mixed') & (detection.unit == 'gene')]
    keys = ['n_case', 'n_control', 'method', 'alpha', 'target', 'population']
    summaries = [summarize(mixed, keys, metric) for metric in ('sensitivity', 'specificity', 'fdp', 'precision')]
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out/'detection-summary.tsv', sep='\t', index=False)
    null = detection[(detection.panel == 'complete_null') & (detection.unit == 'gene')
                     & (detection.target == 'distribution') & (detection.population == 'eligible')].copy()
    null['any_discovery'] = (null.calls > 0).astype(int)
    null_summary = summarize(null, ['n_case', 'n_control', 'method', 'alpha'], 'any_discovery', binary=True)
    null_summary.to_csv(out/'complete-null-summary.tsv', sep='\t', index=False)
    pair_rows = []
    pairs = [('directional_linear', 'unsigned_log'), ('directional_log', 'unsigned_log'),
             ('directional_linear', 'unsigned_linear'), ('unsigned_linear', 'unsigned_log'),
             ('directional_linear', 'directional_log'), ('directional_linear', 'DESeq2'),
             ('directional_linear', 'edgeR-TMM'), ('edgeR-RLE', 'edgeR-TMM')]
    primary = mixed[(mixed.alpha == .05) & (mixed.target == 'distribution') & (mixed.population == 'all')]
    for allocation, group in primary.groupby(['n_case', 'n_control']):
        for first, second in pairs:
            a = group[group.method == first].set_index('id').sort_index()
            b = group[group.method == second].set_index('id').reindex(a.index)
            assert a.biology.tolist() == b.biology.tolist()
            for metric in ('sensitivity', 'fdp', 'specificity'):
                for biology in ['pooled', *sorted(a.biology.unique())]:
                    use = np.ones(len(a), bool) if biology == 'pooled' else (a.biology == biology).to_numpy()
                    delta = (a[metric]-b[metric]).to_numpy()[use]
                    result = interval(delta, a.biology.to_numpy()[use], bounded=False)
                    pair_rows.append(dict(n_case=allocation[0], n_control=allocation[1], first=first,
                                          second=second, metric=metric, biology=biology, **result))
    pd.DataFrame(pair_rows).to_csv(out/'paired-differences.tsv', sep='\t', index=False)
    patterns = pd.read_csv(source/'patterns.tsv', sep='\t')
    patterns = patterns[(patterns.panel == 'mixed') & (patterns.pattern != 'shared_tail_null')]
    detail = summarize(patterns, ['n_case', 'n_control', 'method', 'pattern', 'abundance'], 'power')
    detail.to_csv(out/'pattern-abundance-summary.tsv', sep='\t', index=False)
    # Average abundances within each matrix before resampling matrices.
    by_matrix = patterns.groupby(['id', 'n_case', 'n_control', 'biology', 'method', 'pattern'], as_index=False).agg(
        recovered=('recovered', 'sum'), total=('total', 'sum'))
    by_matrix['power'] = by_matrix.recovered/by_matrix.total
    pattern_summary = summarize(by_matrix, ['n_case', 'n_control', 'method', 'pattern'], 'power')
    pattern_summary.to_csv(out/'pattern-summary.tsv', sep='\t', index=False)
    effects = pd.read_csv(source/'effects.tsv', sep='\t')
    effects = effects[effects.panel == 'mixed']
    effect_keys = ['n_case', 'n_control', 'estimator', 'target', 'population']
    effect_summary = pd.concat([summarize(effects, effect_keys, metric, bounded=False)
                               for metric in ('rmse', 'mae', 'bias', 'n_missing')], ignore_index=True)
    effect_summary.to_csv(out/'effect-summary.tsv', sep='\t', index=False)
    areas = pd.read_csv(source/'areas.tsv', sep='\t')
    areas = areas[areas.panel == 'mixed']
    area_summary = summarize(areas, ['n_case', 'n_control', 'scale', 'description', 'population'], 'rmse', bounded=False)
    area_summary.to_csv(out/'area-summary.tsv', sep='\t', index=False)
    # Explain which prespecified direction carried a discovery. This changes
    # no test, family or threshold and is not a new patient-membership estimate.
    direction_rows = []
    for setting in json.loads((source/'registry.json').read_text()):
        if setting['panel'] != 'mixed':
            continue
        folder = source/setting['id']
        with np.load(folder/'input.npz') as data, np.load(folder/'point-results.npz') as results:
            for method in ('directional_linear', 'directional_log'):
                call = results['padj'][:, LANES[method]] <= .05
                for pattern in PATTERNS:
                    select = data['profile'] == pattern
                    up, down = call[select, 0], call[select, 1]
                    direction_rows.append(dict(id=setting['id'], n_case=setting['n_case'], n_control=setting['n_control'],
                                               biology=setting['biology'], method=method, pattern=pattern,
                                               total=int(select.sum()), up=int(up.sum()), down=int(down.sum()),
                                               both=int((up & down).sum()), either=int((up | down).sum())))
    pd.DataFrame(direction_rows).to_csv(out/'directional-pattern-counts.tsv', sep='\t', index=False)
    cfg = json.loads((source/'protocol.json').read_text())
    toy = []
    for nc, n0 in cfg['allocations']:
        for fraction in (.05, .1, .2):
            spikes = max(1, round(nc*fraction))
            toy.append(dict(n_case=nc, n_control=n0, spikes=spikes, actual_fraction=spikes/nc,
                            **two_level_tail(nc, n0, spikes)))
    pd.DataFrame(toy).to_csv(out/'analytic-tail-example.tsv', sep='\t', index=False)
    return summary, pattern_summary, effect_summary, null_summary


def figures(source, out, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    summary, patterns, effects, null = summaries
    cfg = json.loads((source/'protocol.json').read_text())
    allocations = [tuple(a) for a in cfg['allocations']]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 180})
    saved = []
    def save(fig, name):
        for suffix in ('png', 'pdf'):
            path = out/f'{name}.{suffix}'
            fig.savefig(path, bbox_inches='tight', facecolor='white')
            saved.append(path)
        plt.close(fig)
    nalt = len(cfg['generator']['profiles'])*len(cfg['generator']['profile_means'])
    biology_titles = {'pooled': 'Equal-weight average of both biological models',
                      'gamma': 'Independent Gamma–Poisson biology',
                      'correlated_lognormal': 'Correlated lognormal–Poisson biology'}
    # Show the prespecified biological strata beside their average. The pooled
    # native-method results alone would hide substantial generator dependence.
    for biology, biology_title in biology_titles.items():
        primary = summary[(summary.biology == biology) & (summary.alpha == .05)
                          & (summary.target == 'distribution') & (summary.population == 'all')]
        if primary.empty:
            continue
        fig, axes = plt.subplots(len(allocations), 2, figsize=(12.8, 3.5*len(allocations)), layout='constrained', squeeze=False)
        for row, (nc, n0) in enumerate(allocations):
            for col, (metric, title) in enumerate((('sensitivity', 'Recovery of altered genes'), ('fdp', 'Mean false discovery proportion'))):
                ax = axes[row, col]
                part = primary[(primary.n_case == nc) & (primary.n_control == n0) & (primary.metric == metric)].set_index('method')
                for y, method in enumerate(METHODS):
                    value = part.loc[method]
                    ax.errorbar(value['mean'], y, xerr=[[max(0, value['mean']-value.low)], [max(0, value.high-value['mean'])]],
                                fmt='o', color=COLORS[method], ms=6, capsize=3, lw=1.4)
                ax.set_yticks(range(len(METHODS)), [LABELS[m] for m in METHODS] if col == 0 else [])
                ax.invert_yaxis()
                ax.set_title(f'{nc} cases : {n0} controls — {title}', fontsize=11)
                ax.xaxis.set_major_formatter(PercentFormatter(1))
                ax.grid(axis='x', alpha=.18)
                if metric == 'fdp':
                    ax.axvline(.05, color='#444444', lw=1, ls='--')
                    ax.set_xlim(0, min(1, max(.1, float(part.high.max())*1.08)))
                else:
                    ax.set_xlim(0, 1)
        # Precision is undefined for matrices with no calls, whereas both
        # displayed metrics retain every mixed matrix in their denominators.
        replicates = int(primary.loc[primary.metric == 'sensitivity', 'n'].min())
        fig.suptitle(f'{biology_title} · q ≤ 0.05\n{replicates} independent mixed matrices per cohort; '
                     f'{nalt} altered genes in {cfg["genes"]:,}', fontsize=14)
        save(fig, 'detection-fdr' if biology == 'pooled' else f'detection-{biology}')

    fig, axes = plt.subplots(len(allocations), 1, figsize=(15.5, 3.33*len(allocations)), layout='constrained')
    axes = np.atleast_1d(axes)
    for ax, (nc, n0) in zip(axes, allocations):
        part = patterns[(patterns.biology == 'pooled') & (patterns.n_case == nc) & (patterns.n_control == n0)]
        values = part.pivot(index='method', columns='pattern', values='mean').reindex(index=METHODS, columns=list(PATTERNS)).to_numpy()
        im = ax.imshow(values, vmin=0, vmax=1, cmap='Blues', aspect='auto')
        for (y, x), value in np.ndenumerate(values):
            ax.text(x, y, f'{100*value:.0f}', ha='center', va='center', fontsize=9,
                    color='white' if value > .6 else '#1e2730')
        ax.set_yticks(range(len(METHODS)), [LABELS[m] for m in METHODS])
        ax.set_xticks(range(len(PATTERNS)), list(PATTERNS.values()), fontsize=9)
        ax.set_title(f'{nc} cases : {n0} controls', loc='left')
        ax.tick_params(length=0)
    fig.colorbar(im, ax=axes, label='Altered genes recovered', shrink=.75, format=PercentFormatter(1))
    fig.suptitle('Sensitivity by planted pattern at q ≤ 0.05\nNumbers are percentages; four abundance levels averaged within each matrix', fontsize=14)
    save(fig, 'pattern-recovery')

    estimators = ['empirical_RLE_mean', 'DESeq2', 'edgeR-TMM', 'edgeR-RLE']
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), layout='constrained', sharey=True)
    for ax, target in zip(axes, ('population', 'realized_latent')):
        part = effects[(effects.biology == 'pooled') & (effects.metric == 'rmse')
                       & (effects.population == 'altered') & (effects.target == target)]
        for index, estimator in enumerate(estimators):
            rows = part[part.estimator == estimator].set_index(['n_case', 'n_control']).reindex(allocations)
            x = np.arange(len(allocations)) + (index-1.5)*.15
            ax.errorbar(x, rows['mean'], yerr=np.stack((rows['mean']-rows.low, rows.high-rows['mean'])).clip(0),
                        fmt='o', capsize=3, color=COLORS[estimator], label=LABELS[estimator])
        ax.set_xticks(range(len(allocations)), [f'{nc}:{n0}' for nc, n0 in allocations])
        ax.set_xlabel('Cases : controls')
        ax.set_title('Population mean-ratio target' if target == 'population' else 'Realized latent mean-ratio target')
        ax.grid(axis='y', alpha=.18)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel('Mean per-matrix RMSE (log2 fold change)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=2, frameon=False)
    save(fig, 'mean-effect-recovery')

    fig, axes = plt.subplots(1, len(allocations), figsize=(4.95*len(allocations), 4.6), layout='constrained', sharey=True)
    axes = np.atleast_1d(axes)
    selected = summary[(summary.biology == 'pooled') & (summary.target == 'distribution') & (summary.population == 'all')]
    for ax, (nc, n0) in zip(axes, allocations):
        part = selected[(selected.n_case == nc) & (selected.n_control == n0)]
        for method in METHODS:
            values = part[part.method == method].pivot(index='alpha', columns='metric', values='mean').sort_index()
            ax.plot(values.fdp, values.sensitivity, 'o-', color=COLORS[method], ms=4, lw=1.2, label=LABELS[method])
            ax.plot(values.loc[.05, 'fdp'], values.loc[.05, 'sensitivity'], 'o', ms=8, mfc='none', mec=COLORS[method])
        ax.axvline(.05, color='#555555', lw=1, ls='--')
        ax.set_title(f'{nc} cases : {n0} controls')
        ax.set_xlabel('Mean realized FDP')
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.16)
    axes[0].set_ylabel('Altered genes recovered')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=4, frameon=False, fontsize=9)
    fig.supxlabel('Prespecified q thresholds: 0.01, 0.02, 0.05, 0.10, 0.20; open rings mark q = 0.05', fontsize=9)
    save(fig, 'power-fdp-curves')
    return saved


def main(args):
    source, out = args.source.resolve(), args.output.resolve()
    record = json.loads((source/'audit.json').read_text())
    if not record['complete']:
        raise ValueError('Require the complete frozen experiment before review')
    expected = json.loads((source/'registry.json').read_text())
    assert len(record['datasets']) == len(expected)
    assert all(row['baseline_parity'] == 'PASS' for row in record['datasets'])
    registry = json.loads((source/'checksums.json').read_text())['files']
    for name, digest in registry.items():
        if sha256(source/name) != digest:
            raise ValueError(f'Changed sealed input/output: {source/name}')
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out/'review_points.py')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        audits = list(pool.map(audit_dataset, [source/row['id'] for row in expected]))
    pd.DataFrame(audits).to_csv(out/'independent-audit.tsv', sep='\t', index=False)
    result = tables(source, out)
    plots = figures(source, out, result)
    write_json(out/'audit.json', dict(complete=True, source=str(source), source_registry_sha256=sha256(source/'checksums.json'),
                                    checked_files=len(registry), checked_datasets=len(audits),
                                    exact_native_input_identity='PASS', stopping_and_BH='PASS', primary_call_recount='PASS',
                                    independent_quantile_areas='PASS',
                                    largest_area_difference=max(row['max_quantile_area_difference'] for row in audits),
                                    interpretation='Replicated development comparison; fresh confirmation and final release review pending',
                                    intervals='Matrix bootstrap stratified by generator; exact per-stratum binomial null bounds, conservative Bonferroni pooling; boundary envelopes for all-zero/all-one bounded observations',
                                    figures=[p.name for p in plots], user_approval='pending'))
    write_json(out/'checksums.json', dict(files={p.name: sha256(p) for p in out.iterdir() if p.is_file()}))
    print(f'Review complete: {len(audits)} datasets, {len(registry)} source files checked, {len(plots)} figure files. User approval pending.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    main(p.parse_args())
