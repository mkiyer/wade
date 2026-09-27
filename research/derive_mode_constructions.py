"""Freeze and audit alternative mode-preserving uncertainty constructions.

PYTHONPATH=src:research python research/derive_mode_constructions.py --output <new-directory>
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import brentq
from scipy.special import betainc, exp1, ndtr
from scipy.stats import binom

from assess_measurement import ROOT, digest, save, write
from wade_research.measurement import binomial_support
from wade_research.mode_constructions import (
    contracted_beta_cdf, contracted_dirichlet_parameters, contraction_moments,
    draw_contracted_dirichlet, multinomial_parameters, smoothed_binomial,
)


def examples(cfg, out):
    rows, checks = [], []
    rng = np.random.default_rng(cfg['seed'])
    for index, values in enumerate(cfg['examples']):
        counts = np.asarray(values, dtype=np.int64)
        original = counts.copy()
        counts.flags.writeable = False
        n, g = int(counts.sum()), len(counts)
        model = multinomial_parameters(counts, np.full(g, 1/g))
        support = np.arange(n+1)
        pmf = binom.pmf(support[:, None], n, model['q'])
        np.testing.assert_array_equal(np.argmax(pmf, axis=0), counts)
        assert np.all((pmf == pmf.max(axis=0)).sum(axis=0) == 1)
        draws = rng.multinomial(n, model['q'], size=cfg['joint_draws'])
        np.testing.assert_array_equal(draws.sum(axis=1), n)
        models = [('multinomial_bootstrap', model['mean']/n, model['variance']/n**2, draws/n)]
        for budget in (1, g):
            prior = np.full(g, budget/g)
            posterior = contracted_dirichlet_parameters(counts, prior)
            draws = draw_contracted_dirichlet(counts, prior, cfg['joint_draws'], rng)
            models.append((f'contraction_budget{budget}', posterior['mean'], posterior['variance'], draws))
        for method, mean, variance, draws in models:
            sum_error = float(abs(draws.sum(axis=1)-1).max())
            assert sum_error < 2e-15 and np.all(variance > 0)
            np.testing.assert_allclose(mean.sum(), 1, atol=1e-15, rtol=0)
            mean_se = np.sqrt(variance/len(draws))
            maximum_z = float(np.max(abs(draws.mean(axis=0)-mean)/mean_se))
            assert maximum_z < 6
            checks.append(dict(example=index, method=method, total=n, genes=g,
                maximum_draw_sum_error=sum_error, maximum_sample_mean_z=maximum_z,
                sum_of_means=float(mean.sum()), positive_variance=bool(np.all(variance > 0))))
            for gene in range(g):
                rows.append(dict(example=index, method=method, total=n, genes=g, gene=gene,
                    observed_count=int(counts[gene]), mean_count=n*mean[gene], variance_count=n*n*variance[gene]))
        np.testing.assert_array_equal(counts, original)
        assert counts.dtype == original.dtype
    save(out, 'worked-examples', rows)
    save(out, 'joint-checks', checks)


def observed(cfg, out):
    rows = []
    for n in cfg['depths']:
        c = np.asarray(cfg['observed_counts'])
        for g in cfg['gene_universes']:
            model = smoothed_binomial(c, n, 1/g)
            quantiles = binom.ppf(np.array([.025, .975])[:, None], n, model['q'])
            for j, count in enumerate(c):
                rows.append(dict(depth=n, genes=g, observed_count=int(count), method='multinomial_bootstrap',
                    mean_count=model['mean'][j], variance_count=model['variance'][j],
                    mean_cpm=model['mean'][j]*cfg['units']/n,
                    probability_at_zero=float(binom.pmf(0, n, model['q'][j])),
                    probability_above_one_read=float(binom.sf(1, n, model['q'][j])),
                    lower95_count=quantiles[0, j], upper95_count=quantiles[1, j], cdf_error_estimate=0.))
            for budget in (1, g):
                a, b = c+budget/g, n-c+budget-budget/g
                model = contraction_moments(c/n, a, b)
                cdf, error = contracted_beta_cdf(1/n, c/n, a, b, tolerance=cfg['cdf_tolerance'])
                for j, count in enumerate(c):
                    rows.append(dict(depth=n, genes=g, observed_count=int(count),
                        method=f'contraction_budget{budget}',
                        mean_count=n*model['mean'][j], variance_count=n*n*model['variance'][j],
                        mean_cpm=cfg['units']*model['mean'][j], probability_at_zero=0.,
                        probability_above_one_read=1-cdf[j], lower95_count=np.nan, upper95_count=np.nan,
                        cdf_error_estimate=error))
    save(out, 'observed-counts', rows)


def moment_score(n, g, truth, *, budget=None, contract=False):
    """Exact repeated-sampling moments; no tail truncation in these endpoints."""
    var_c = truth*(1-truth/n)
    if budget is None:
        slope, offset = n/(n+1), n/g/(n+1)
        mean = slope*truth+offset
        average_variance = mean-((slope*slope*var_c)+mean*mean)/n
    else:
        total, alpha = n+budget, budget/g
        slope, offset = n/total, n*alpha/total
        average_variance = n*n*((truth+alpha)*(n-truth+budget-alpha)-var_c)/(total*total*(total+1))
        if contract:
            average_variance = average_variance/3+((slope-1)**2*var_c+((slope-1)*truth+offset)**2)/12
            slope, offset = (1+slope)/2, offset/2
        mean = slope*truth+offset
    mean_variance = slope*slope*var_c
    return dict(mean_bias_count=mean-truth, mean_rmse_count=np.sqrt(mean_variance+(mean-truth)**2),
        point_rmse_count=np.sqrt(var_c), average_kernel_variance_count=average_variance,
        mixture_variance_count=mean_variance+average_variance,
        mixture_variance_over_point=(mean_variance+average_variance)/var_c if var_c else np.nan)


def repeated_counts(cfg, out):
    rows = []
    for n in cfg['depths']:
        for g in cfg['gene_universes']:
            for truth in cfg['expected_counts']:
                p = truth/n
                c, weights, omitted = binomial_support(n, p, cfg['support_tail_probability'])
                smooth = smoothed_binomial(c, n, 1/g)
                interval = binom.ppf(np.array([.025, .95, .975])[:, None], n, smooth['q'])
                central = (interval[0] <= truth) & (truth <= interval[2])
                upper = truth <= interval[1]
                common = dict(depth=n, genes=g, true_fraction=p, expected_count=truth,
                              omitted_probability=omitted, support_size=len(c))
                rows.append(dict(**common, method='multinomial_bootstrap',
                    central95_coverage=float(weights @ central), upper95_coverage=float(weights @ upper),
                    cdf_error_estimate=0., tighter_cdf_change=0., threshold_ambiguous_probability=0.,
                    **moment_score(n, g, truth)))
                for budget in (1, g):
                    a, b = c+budget/g, n-c+budget-budget/g
                    base_cdf = betainc(a, b, p)
                    contracted_cdf, error = contracted_beta_cdf(p, c/n, a, b,
                                                               tolerance=cfg['cdf_tolerance'])
                    tight, tight_error = contracted_beta_cdf(p, c/n, a, b,
                                                           tolerance=cfg['cdf_tolerance']/20)
                    change = float(np.max(abs(tight-contracted_cdf)))
                    assert change < 2*cfg['cdf_tolerance']
                    margin = max(error, tight_error, change, 1e-12)
                    ambiguous = np.any(abs(tight[:, None]-[.025, .95, .975]) <= margin, axis=1)
                    for contract, cdf in ((False, base_cdf), (True, tight)):
                        central = (cdf >= .025) & (cdf <= .975)
                        upper = cdf <= .95
                        rows.append(dict(**common, method=f'{"contraction" if contract else "dirichlet"}_budget{budget}',
                            central95_coverage=float(weights @ central), upper95_coverage=float(weights @ upper),
                            cdf_error_estimate=tight_error if contract else 0.,
                            tighter_cdf_change=change if contract else 0.,
                            threshold_ambiguous_probability=float(weights @ ambiguous) if contract else 0.,
                            **moment_score(n, g, truth, budget=budget, contract=contract)))
        print(f'Exact measurement checks completed depth {n}', flush=True)
    save(out, 'repeated-counts', rows)


def analytic_limits(cfg, out):
    def product_cdf(x):
        return ndtr(x)+x*exp1(x*x/2)/(2*np.sqrt(2*np.pi))
    half_width = brentq(lambda x: product_cdf(x)-.975, .01, 5)
    write(out/'analytic-limits.json', dict(
        contracted_gaussian_variance_ratio=1/3,
        contracted_gaussian_975_quantile=half_width,
        high_information_coverage_of_central95=float(2*ndtr(half_width)-1),
        multinomial_family='Every unique-mode-preserving multinomial has q_g=(C_g+w_g)/(N+1), with 0<w_g<1 and sum(w)=1; strict form excludes degenerate boundary kernels.',
        unseen_gene_total_expected_resampled_counts='N/(N+1) * sum(w_g over C_g=0) < 1',
        continuous_mean='(m + E[Q])/2',
        continuous_covariance='Cov(Q)/3 + (E[Q]-m)(E[Q]-m)^T/12',
        interpretation='The constraints are compatible. These constructions are not automatically calibrated latent-abundance posteriors.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'research/protocols/mode-constructions.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.protocol.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.protocol, args.output/'protocol.json')
    sources = [Path(__file__), ROOT/'research/assess_measurement.py',
               ROOT/'research/tests/test_mode_constructions.py',
               *sorted((ROOT/'research/wade_research').glob('*.py'))]
    hashes = {str(p.relative_to(ROOT)):digest(p) for p in sources}
    for path in sources:
        dest = args.output/'source'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    write(args.output/'manifest.json', dict(started=datetime.now(timezone.utc).isoformat(),
        checkpoint=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT).decode().strip(),
        protocol_sha256=digest(args.protocol), sources=hashes, python=sys.version,
        numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__,
        claims='Derivation and known-exposure measurement audit only; no WADE tests or FDR claims'))
    for study in (analytic_limits, examples, observed, repeated_counts):
        study(cfg, args.output)
    assert hashes == {str(p.relative_to(ROOT)):digest(p) for p in sources}, 'Sources changed during execution'
    write(args.output/'decision.json', dict(
        structural_status='COMPATIBLE_CONSTRAINTS_DEMONSTRATED',
        measurement_status='NEITHER_CONSTRUCTION_SELECTED',
        discrete_limit='Exact modes restrict unseen-gene expected mass to less than one read in total; uniform allocation effectively collapses zero percentile intervals.',
        continuous_limit='Uniform contraction preserves marginal modes but reduces variance and yields about 79.17% asymptotic coverage for central 95% intervals. It is not the original posterior.',
        next='Retain measurement-likelihood and calibration gates. Resolve the allocation of unseen mass and uncertainty width before a new performance sweep; do not tune a contraction to these results.'))
    write(args.output/'completion.json', dict(status='DERIVATION_AUDIT_COMPLETE',
        finished=datetime.now(timezone.utc).isoformat(),
        outputs={p.name:digest(p) for p in args.output.glob('*') if p.is_file()}))


if __name__ == '__main__':
    main()
