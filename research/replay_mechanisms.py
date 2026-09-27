"""Replay normalization and noise failures through current code, without archive imports."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import wade
from wade._wasserstein import W1State, resolve_backend
from wade.permutation import draw_perms
from wade.pvalues import bh_adjust
from wade_research import point_scores

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def backend_provenance(backend):
    if backend != 'rust':
        return dict(backend=backend, numpy=np.__version__)
    binary = Path(importlib.import_module('wade._kernel').__file__)
    sources = [ROOT/'Cargo.toml', ROOT/'Cargo.lock', *sorted((ROOT/'rust/src').rglob('*.rs'))]
    return dict(backend=backend, numpy=np.__version__, binary=str(binary),
                binary_sha256=digest(binary),
                rust_sources={str(p.relative_to(ROOT)):digest(p) for p in sources})


def inputs(cfg):
    for setting in cfg['normalization']['settings']:
        for rep in cfg['normalization']['replicates']:
            folder = ROOT/cfg['normalization']['source']/f'{setting}-r{rep:03d}'
            yield folder/'data-source.json'
            source = ROOT/json.loads((folder/'data-source.json').read_text())['path']
            yield source/'inputs.npz'
            yield source/'reference.npz'
            for name in ['selection.npz', 'factors.npz', 'statistics.npz', 'results.tsv']:
                yield folder/name
    for setting in cfg['noise']['settings']:
        for rep in cfg['noise']['replicates']:
            folder = ROOT/cfg['noise']['source']/f'ladder-{setting}-bankNone-rep{rep:03d}'
            for name in ['inputs.npz', 'input.json']:
                yield folder/name
            for arm in cfg['noise']['arms']:
                yield folder/f'{arm}.json'
                yield folder/f'{arm}.npz'


def normalization(cfg, out, backend):
    rows, profiles = [], []
    for setting in cfg['settings']:
        for rep in cfg['replicates']:
            folder = ROOT/cfg['source']/f'{setting}-r{rep:03d}'
            source = ROOT/json.loads((folder/'data-source.json').read_text())['path']
            data, reference = np.load(source/'inputs.npz'), np.load(source/'reference.npz')
            counts, labels, condition = data['counts'], data['labels'], data['cond']
            ref_counts, ref_fold = reference['counts'], reference['fold']
            np.testing.assert_array_equal(condition, labels[0])
            selection, saved_factors = np.load(folder/'selection.npz'), np.load(folder/'factors.npz')
            stored = np.load(folder/'statistics.npz')
            table = pd.read_csv(folder/'results.tsv', sep='\t', keep_default_na=False)
            alternative = table.alternative.to_numpy(bool)
            for method in cfg['methods']:
                ids = selection[method]
                if method == 'known':
                    sf = data['known_factors']
                else:
                    mask = np.zeros(len(ref_counts), bool)
                    mask[ids] = True
                    sf = wade.fit_normalization(ref_counts, reference_genes=mask).size_factors
                np.testing.assert_allclose(sf, saved_factors[method], rtol=5e-13, atol=1e-14)
                err = np.log2(sf/data['known_factors'])
                for scale in cfg['scales']:
                    x = counts/sf if scale == 'linear' else np.log2(counts/sf+1)
                    state = W1State(x, condition, stop=len(labels))
                    state.update(labels[1:], backend=backend)
                    key = f'{method}__{scale}'
                    np.testing.assert_allclose(state.observed, stored[key+'__observed'], rtol=3e-12, atol=1e-12)
                    expected = np.unpackbits(stored[key+'__exceed'], axis=1, bitorder='little')[:, :len(labels)-1].sum(axis=1)
                    np.testing.assert_array_equal(state.exceed, expected)
                    np.testing.assert_array_equal(state.tried, len(labels)-1)
                    p = (1+state.exceed)/len(labels)
                    q = bh_adjust(p)
                    np.testing.assert_allclose(p, table['p_'+key], rtol=0, atol=1e-14)
                    np.testing.assert_allclose(q, table['q_'+key], rtol=0, atol=1e-14)
                    called = q <= .05
                    tp, fp = int((called & alternative).sum()), int((called & ~alternative).sum())
                    rows.append(dict(setting=setting, replicate=rep, method=method, scale=scale,
                        factor_group_log2_error=float(err[condition.astype(bool)].mean()-err[~condition.astype(bool)].mean()),
                        selected_changed=float(np.mean(ref_fold[ids] != 1)) if len(ids) else None,
                        true_positive=tp, false_positive=fp, positives=int(alternative.sum()),
                        nulls=int((~alternative).sum()), calls=int(called.sum()), fdp=fp/max(1,called.sum()),
                        bank_parity=True, permutations=len(labels)-1))
                    profiles.extend(dict(setting=setting, replicate=rep, method=method, scale=scale,
                        gene=gene, alternative=bool(alt), pvalue=float(pi), qvalue=float(qi))
                        for gene, alt, pi, qi in zip(table.gene, alternative, p, q))
            print(f'Normalization replay: {setting} r{rep:02d}', flush=True)
            pd.DataFrame(rows).to_csv(out/'normalization.tsv', sep='\t', index=False)
    pd.DataFrame(profiles).to_csv(out/'normalization-profiles.tsv', sep='\t', index=False)


def perturb(counts, factors, record):
    """Exact old interventions, including their intentionally frozen exposure."""
    arm, amplitude, repeats = record['arm'], record['amplitude'], record['repeats']
    if arm == 'none':
        return (counts/factors)[None]
    rng = np.random.default_rng(record['noise_seed'])
    shape = (repeats, *counts.shape)
    if arm == 'poisson_offset':
        return rng.poisson(counts[None]+amplitude, size=shape)/factors
    u = rng.random(shape)
    if arm == 'raw_uniform':
        return (counts[None]+amplitude*u)/factors
    if arm == 'post_uniform':
        return counts[None]/factors+amplitude*u
    raise ValueError(f'Unregistered perturbation: {arm}')


def noise(cfg, out):
    rows, profiles = [], []
    for setting in cfg['settings']:
        for rep in cfg['replicates']:
            folder = ROOT/cfg['source']/f'ladder-{setting}-bankNone-rep{rep:03d}'
            data = np.load(folder/'inputs.npz')
            mask = np.array(json.loads((folder/'input.json').read_text())['mask'])
            counts, factors, cond = data['counts'][mask], data['factors'], data['condition']
            np.testing.assert_array_equal(mask, data['counts'].any(axis=1))
            null = data['biological_null'][mask]
            for arm in cfg['arms']:
                record = json.loads((folder/f'{arm}.json').read_text())
                x = perturb(counts, factors, record)
                labels = np.r_[cond[None], draw_perms(cond, 255, rng=np.random.default_rng(record['permutation_seed']))]
                score = np.array([np.mean([point_scores(xi, labels) for xi in x[:, gi]], axis=0)
                                  for gi in range(len(counts))])
                saved = np.load(folder/f'{arm}.npz')
                np.testing.assert_allclose(score[:, 0], saved['observed'][:, 0], rtol=3e-12, atol=1e-12)
                log_observed = np.array([np.mean([point_scores(np.log2(xi+1), labels[:1])[0]
                                                  for xi in x[:, gi]]) for gi in range(len(counts))])
                np.testing.assert_allclose(log_observed, saved['observed'][:, 3], rtol=3e-12, atol=1e-12)
                # Same conservative arithmetic tie allowance as the old recipe.
                tolerance = 100*np.finfo(float).eps*saved['observed'][:, 1]
                p = (1+(score[:, 1:] >= score[:, :1]-tolerance[:, None]).sum(axis=1))/len(labels)
                called = p <= .05
                historic = saved['q'][:, 0] <= .05
                rows.append(dict(setting=setting, replicate=rep, arm=arm, scale='linear', direction='up',
                    nulls=int(null.sum()), false_positive=int((called & null).sum()),
                    alternatives=int((~null).sum()), alternative_calls=int((called & ~null).sum()),
                    historical_bh_calls=int(historic.sum()), historical_bh_false=int((historic & null).sum()),
                    permutations=255, observed_parity=True,
                    mean_background_case=float((record['amplitude']/factors[cond.astype(bool)]).mean())
                        if record['arm']=='poisson_offset' else None,
                    mean_background_control=float((record['amplitude']/factors[~cond.astype(bool)]).mean())
                        if record['arm']=='poisson_offset' else None))
                profiles.extend(dict(setting=setting, replicate=rep, arm=arm, gene=int(g),
                    null=bool(n), pvalue=float(pv)) for g,n,pv in zip(np.flatnonzero(mask),null,p))
            print(f'Noise replay: {setting} r{rep:02d}', flush=True)
            pd.DataFrame(rows).to_csv(out/'noise.tsv', sep='\t', index=False)
    pd.DataFrame(profiles).to_csv(out/'noise-profiles.tsv', sep='\t', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'research/protocols/mechanism-replay.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.protocol.read_text())
    paths = list(inputs(cfg))
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError('Required local study artifacts are missing')
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.protocol, args.output/'protocol.json')
    sources = [Path(__file__), *sorted((ROOT/'src/wade').rglob('*.py')),
               ROOT/'research/wade_research/points.py', ROOT/'research/wade_research/resampling.py']
    for p in sources:
        dest = args.output/'source'/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    backend = resolve_backend('auto')
    native = backend_provenance(backend)
    source_hashes = {str(p.relative_to(ROOT)):digest(p) for p in sources}
    write(args.output/'manifest.json', dict(started=datetime.now(timezone.utc).isoformat(), backend=backend,
        protocol_sha256=digest(args.protocol), inputs={str(p.relative_to(ROOT)):digest(p) for p in paths},
        sources=source_hashes, backend_provenance=native))
    normalization(cfg['normalization'], args.output, backend)
    noise(cfg['noise'], args.output)
    assert native == backend_provenance(backend), 'Backend changed during execution'
    assert source_hashes == {str(p.relative_to(ROOT)):digest(p) for p in sources}, 'Sources changed during execution'
    write(args.output/'completion.json', dict(status='CURRENT_CODE_REPLAY_ONLY',
        finished=datetime.now(timezone.utc).isoformat(),
        outputs={p.name:digest(p) for p in args.output.glob('*') if p.is_file()}))


if __name__ == '__main__':
    main()
