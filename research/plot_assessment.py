"""Render the analytic localization check; illustration requires user review.

PYTHONPATH=research python research/plot_assessment.py --output <artifact-directory>
This figure uses exact latent abundances, without observations or kernel fitting.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from wade_research.assessment import describe, paired_quantiles, points


def render(output):
    output.mkdir(parents=True, exist_ok=True)
    files = [output/'walk-limit.png', output/'walk-limit.pdf', output/'walk-limit.json']
    if any(path.exists() for path in files):
        raise FileExistsError('Refusing to overwrite an existing illustration')
    baseline = np.arange(1, 101)/100
    case = np.r_[np.repeat(baseline, 9), np.full(100, 10.)]
    control = np.repeat(baseline, 10)
    labels = np.r_[np.ones(len(case)), np.zeros(len(control))]
    description, _ = describe(points(np.r_[case, control]), labels)
    edges, qa, qb = paired_quantiles(case, control)
    walk = np.r_[np.cumsum(((qa-qb)*np.diff(edges))[::-1])[::-1], 0.]
    with plt.rc_context({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        ax = axes[0]
        ax.stairs(qa, edges, baseline=None, color='#187c80', linewidth=2, label='Cases: 10% at abundance 10')
        ax.stairs(qb, edges, baseline=None, color='#52647a', linewidth=2, label='Controls: abundance 0.01–1')
        ax.axvline(.9, color='#ba7521', linestyle=':', linewidth=1.5)
        ax.set(xlabel='Quantile', ylabel='Latent abundance', title='Only 10% of cases were changed',
               xlim=(0, 1), ylim=(0, 11))
        ax.legend(loc='upper left', frameon=False, fontsize=9)
        ax = axes[1]
        ax.plot((1-edges)[::-1], walk[::-1], color='#187c80', linewidth=2)
        ax.axvline(.1, color='#ba7521', linestyle=':', label='Planted fraction: 10%')
        fraction = description['tail_fraction']
        ax.scatter([fraction], [walk.max()], color='#963d4c', zorder=3)
        ax.annotate(f'Maximum walk: {100*fraction:.1f}% selected',
                    xy=(fraction, walk.max()), xytext=(.32, .6),
                    arrowprops={'arrowstyle': '->', 'color': '#963d4c'}, color='#963d4c', fontsize=10)
        ax.set(xlabel='Fraction included from the upper tail', ylabel='Cumulative signed area',
               title='The walk also includes unchanged cases', xlim=(0, 1.02), ylim=(0, 1.05))
        ax.legend(loc='upper left', frameon=False, fontsize=9)
        fig.suptitle('A cutoff limitation that exists before count noise or kernels', fontsize=14)
        fig.savefig(files[0], dpi=180)
        fig.savefig(files[1])
        plt.close(fig)
    metadata = dict(
        target='Analytic finite-cohort example, not a performance or significance estimate',
        cases=1000, controls=1000, planted_fraction=.1,
        selected_fraction=fraction, equal_unaffected_and_control_distributions=True,
        user_approval='pending',
        sources={str(Path(__file__).name): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'assessment.py': hashlib.sha256((Path(__file__).parent/'wade_research/assessment.py').read_bytes()).hexdigest()},
        artifacts={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files[:2]},
    )
    files[2].write_text(json.dumps(metadata, indent=2)+'\n')
    print(files[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    render(parser.parse_args().output)
