"""Render frozen measurement results for review; no recomputation or selection."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(source):
    names = ['observed-counts', 'repeated-counts', 'composition', 'technical-overdispersion']
    tables = {name:pd.read_csv(source/f'{name}.tsv', sep='\t') for name in names}
    out = source/'review'
    out.mkdir(exist_ok=True)
    files = [out/f'measurement-audit.{ext}' for ext in ('png', 'pdf', 'json')]
    if any(p.exists() for p in files):
        raise FileExistsError('Refusing to overwrite a reviewed artifact')
    colors = {1.:'#b67826', .5:'#187c80'}
    with plt.rc_context({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False}):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        ax = axes[0, 0]
        data = tables['observed-counts'].query('count==0 and relative_exposure==1')
        for h, group in data.groupby('h'):
            ax.plot(group.depth, group['mean'], 'o-', color=colors[h], label=f'h={h:g}: mean and central 95% interval')
            ax.fill_between(group.depth, group.lower95, group.upper95, color=colors[h], alpha=.13)
        ax.set(xscale='log', yscale='log', xlabel='Library total counts', ylabel='Normalized abundance (CPM)',
               title='A. After observing zero counts')
        ax.legend(frameon=False, fontsize=8, loc='upper right')
        ax = axes[0, 1]
        data = tables['repeated-counts'].query('true_abundance==1')
        for h, group in data.groupby('h'):
            ax.plot(group.depth, 100*group.central95_coverage, 'o-', color=colors[h], label=f'h={h:g}')
        ax.axhline(95, linestyle=':', color='#52647a', label='95% reference')
        ax.set(xscale='log', xlabel='Library total counts', ylabel='Coverage (%)', ylim=(-3, 105),
               title='B. Repeated counts from a fixed 1 CPM truth')
        ax.legend(frameon=False, loc='lower right')
        ax = axes[1, 0]
        data = tables['composition'].query('genes==16384')
        for h, group in data.groupby('h'):
            ax.plot(group.depth, group.summed_fraction_mean, 'o-', color=colors[h], label=f'h={h:g}')
        ax.axhline(1, linestyle=':', color='#52647a', label='Required total = 1')
        ax.set(xscale='log', yscale='log', xlabel='Library total counts', ylabel='Sum of posterior gene-fraction means',
               title='C. Complete count universe: 16,384 genes', ylim=(.8, 25), yticks=[1, 2, 5, 10, 20])
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.legend(frameon=False, loc='upper right')
        ax = axes[1, 1]
        data = tables['technical-overdispersion'].query('true_abundance==1000 and technical_cv==.1 and h==.5')
        for target, color, label in [('source', '#963d4c', 'Source abundance'),
                                     ('preparation', '#187c80', 'Prepared-library abundance')]:
            group = data[data.target == target]
            ax.errorbar(group.depth, 100*group.central95_coverage, yerr=1.96*100*group.coverage_mc_se,
                        fmt='o-', color=color, capsize=3, label=label)
        ax.axhline(95, linestyle=':', color='#52647a')
        ax.set(xscale='log', xlabel='Library total counts', ylabel='Coverage (%)', ylim=(-3, 105),
               title='D. Added 10% technical CV; 1,000 CPM; h=0.5')
        ax.legend(frameon=False, loc='lower left')
        fig.suptitle('Measurement audit of the existing binomial–Beta kernels\n'
                     'Known exposures; linear CPM units; h values are diagnostic priors', fontsize=14)
        fig.savefig(files[0], dpi=180)
        fig.savefig(files[1])
        plt.close(fig)
    metadata = dict(user_approval='pending',
        target='Measurement diagnostics, not gene detection or a selected new model',
        source_sha256=digest(Path(__file__)),
        inputs={name:digest(source/f'{name}.tsv') for name in names},
        artifacts={p.name:digest(p) for p in files[:2]},
        uncertainty='Panel B: deterministic binomial sums. Panel D: 20,000 technical replicates; bars are 1.96 Monte Carlo SE.')
    files[2].write_text(json.dumps(metadata, indent=2)+'\n')
    print(files[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    render(parser.parse_args().source)
