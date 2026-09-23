"""Check rendered-notebook exports against full results and preserved raw counts."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, false_discovery_control
from notebook_support import sha256


class Resources(HTMLParser):
    def __init__(self):
        super().__init__(); self.images = []; self.local = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'img':
            self.images.append(attrs.get('src',''))
        url = attrs.get('src') if tag in ('img','script','iframe') else None
        if tag == 'link' and attrs.get('rel') == 'stylesheet':
            url = attrs.get('href')
        if url and not urlsplit(url).scheme and not url.startswith('#'):
            self.local.append(unquote(urlsplit(url).path))


def decile(values, upper):
    values = np.sort(values)
    if upper:
        values = values[::-1]
    mass = len(values)/10
    whole = int(np.floor(mass)); remainder = mass-whole
    return (values[:whole].sum() + (remainder*values[whole] if remainder else 0))/mass


def table_audit(path):
    frame = pd.read_csv(path,sep='\t',index_col='gene')
    record = json.loads(path.with_suffix('.manifest.json').read_text())
    assert frame.index.is_unique and frame.index.tolist() == record['design']['genes']
    mask = frame.status.eq('tested').to_numpy()
    assert mask.sum() == record['design']['n_tested']
    assert frame.status.tolist() == record['design']['status']
    h,cap = record['run']['stop_exceedances'],record['run']['max_permutations']
    k,t = frame.n_exceedances.to_numpy(),frame.n_permutations.to_numpy()
    assert t.tolist() == record['resampling']['n_permutations']
    assert k.tolist() == record['resampling']['n_exceedances']
    assert np.all((k[mask]==h)|(t[mask]==cap))
    p = np.where(k[mask]==h,h/t[mask],(k[mask]+1)/(cap+1))
    np.testing.assert_allclose(frame.pvalue[mask],p,rtol=3e-12,atol=1e-16)
    np.testing.assert_allclose(frame.padj[mask],false_discovery_control(p),rtol=3e-12,atol=1e-16)
    assert frame.pvalue[~mask].isna().all() and frame.padj[~mask].isna().all()
    return frame,record


def panel_audit(folder, frame, record, input_path, *, demo=False):
    raw = np.load(input_path)
    genes = raw['genes'] if demo else raw['gene_name']
    lookup = {g:j for j,g in enumerate(genes)}
    if demo:
        cols = np.arange(raw['counts'].shape[1])
        panel = pd.read_csv(folder/'marker-results.tsv',sep='\t',index_col='gene')
    else:
        sample_idx = {s:j for j,s in enumerate(raw['sample_names'])}
        cols = [sample_idx[s] for s in record['design']['sample_names']]
        panel = pd.read_csv(folder/'primary-panel.tsv',sep='\t',index_col='gene')
    factors = np.asarray(record['normalization']['size_factors'])
    labels = np.asarray(record['design']['condition'],bool)
    for gene,row in panel.iterrows():
        expected = frame.loc[gene]
        for key in panel.columns:
            if key == 'status':
                assert row[key] == expected[key]
            else:
                np.testing.assert_allclose(row[key],expected[key],rtol=3e-12,atol=1e-14,equal_nan=True)
        x = np.log2(raw['counts'][lookup[gene],cols]/factors+1)
        case,control = x[labels],x[~labels]
        bulk = np.median(case)-np.median(control)
        assert abs(wasserstein_distance(case,control)-expected.statistic) < 2e-12
        assert abs(bulk-expected.bulk_shift) < 2e-12
        assert abs(decile(case,True)-decile(control,True)-bulk-expected.upper_tail) < 2e-12
        assert abs(decile(case,False)-decile(control,False)-bulk-expected.lower_tail) < 2e-12
    if not demo:
        samples = pd.read_csv(folder/'primary/sample-order.tsv',sep='\t')
        assert samples.library.tolist() == record['design']['sample_names']
        assert samples.patient.is_unique
        assert samples.condition.tolist() == record['design']['condition']
        calls = pd.read_csv(folder/'transcriptome-calls.tsv',sep='\t').iloc[0]
        assert calls['Tested genes'] == frame.status.eq('tested').sum()
        assert calls['Distribution calls'] == frame.padj.le(.05).sum()
        assert abs(calls['Called fraction']-frame.padj.le(.05).sum()/frame.status.eq('tested').sum()) < 1e-12
    return len(panel)


def audit(build):
    build = Path(build)
    info = json.loads((build/'build.json').read_text()); name = info['notebook']
    assert info['returncode'] == 0
    parsed = Resources(); parsed.feed((build/f'source/notebooks/{name}.html').read_text())
    assert parsed.images
    for relative in parsed.local:
        assert (build/'source/notebooks'/relative).is_file(),relative
    genes = 0; panel_genes = 0; combined_tests = 0
    key = {'demo':'demo','examples':'prostate','examples-breast':'breast','rna100k':'rna100k'}.get(name)
    if key:
        folder = build/'results'/key
        paths=list(folder.rglob('*all-genes.tsv'))
        assert paths, 'No current result table in notebook exports'
        for path in paths:
            frame,record = table_audit(path); genes += record['design']['n_tested']
            if path.parent.name == 'primary' or name == 'demo':
                panel_genes += panel_audit(folder,frame,record,
                    folder/('simulation.npz' if name=='demo' else 'selected-gene-counts.npz'),demo=name=='demo')
        if name == 'examples-breast':
            combined = pd.read_csv(folder/'all-subtype-contrasts.tsv',sep='\t')
            keep = combined.pvalue.notna()
            expected = false_discovery_control(combined.loc[keep,'pvalue'].to_numpy())
            np.testing.assert_allclose(combined.loc[keep,'q_four_contrasts'],expected,rtol=3e-12,atol=1e-16)
            panel = pd.read_csv(folder/'subtype-panel.tsv',sep='\t').set_index(['subtype','gene'])
            full = combined.set_index(['subtype','gene']).loc[panel.index,panel.columns]
            np.testing.assert_allclose(panel,full,rtol=3e-12,atol=1e-14,equal_nan=True)
            assert combined.subtype.nunique() == 4
            combined_tests = int(keep.sum())
    else:
        # The independent study audit checks all per-gene/native results.
        # Check that plotted replicate means retain that exact score table.
        folder = build/'results'/name
        if name == 'benchmark':
            scores = pd.read_csv(folder/'large/scores.tsv',sep='\t')
            for filename,geometry in [('pattern-power-summary.tsv','balanced'),('limited-control-summary.tsv','fixed_control')]:
                summary = pd.read_csv(folder/filename,sep='\t')
                for row in summary.itertuples():
                    selected = scores[(scores.panel=='mixed')&(scores.geometry==geometry)&
                        (scores.method==row.method)&(scores.category==row.category)&(scores.n_case==row.n_case)]
                    assert len(selected) == row.n
                    assert abs(selected.power.mean()-row.mean) < 1e-12
        else:
            scores = pd.read_csv(folder/'reference/scores.tsv',sep='\t')
            for row in pd.read_csv(folder/'reference-power-summary.tsv',sep='\t').itertuples():
                selected = scores[scores.dataset.str.startswith('compcodeR')&(scores.method==row.method)&(scores.n_case==row.n_case)]
                assert len(selected) == row.n and abs(selected.power.mean()-row.mean) < 1e-12
    output = dict(status='PASS',notebook=name,registered_html_images=len(parsed.images),
                  local_html_resources=len(parsed.local),gene_stopping_and_bh_checks=genes,
                  independent_marker_distance_and_description_checks=panel_genes,combined_subtype_tests=combined_tests,
                  build_record_sha256=sha256(build/'build.json'),script_sha256=sha256(__file__),
                  browser_review='Unavailable: no browser connected; HTML resources and PDF/figure rendering are checked separately')
    (build/'review/numerical-audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(name,output,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('builds',type=Path,nargs='+')
    for build in parser.parse_args().builds:
        audit(build)
