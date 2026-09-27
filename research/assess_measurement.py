"""Frozen measurement-level diagnostics; no gene tests or model selection.

PYTHONPATH=src:research python research/assess_measurement.py --output <new-directory>
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy
from scipy.special import betainc
from scipy.stats import binom, gamma

from wade_research import BetaKernelArea
from wade_research.measurement import (
    BinomialBeta, beta_marginal_mode, binomial_support, uniform_dirichlet_summary,
)
from wade_research.assessment import _beta_quantiles

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def save(out, name, rows):
    pd.DataFrame(rows).to_csv(out/f'{name}.tsv', sep='\t', index=False)
    print(f'{name}: {len(rows)} rows', flush=True)


def observed(cfg, out):
    rows = []
    for n in cfg['depths']:
        c = np.array([v for v in cfg['observed_counts'] if v <= n])
        for h in cfg['prior_shapes']:
            for exposure in cfg['relative_exposures']:
                sf = n/cfg['units']*exposure
                model = BinomialBeta(c, n, sf, h)
                q, repairs, residual = model.quantiles([.025, .5, .95, .975])
                engine = BetaKernelArea(c, np.full(len(c), n),
                                       size_factors=np.full(len(c), sf), prior_shape=h)
                np.testing.assert_allclose(model.mean, engine.mean[engine.patient_state], rtol=2e-15)
                np.testing.assert_allclose(model.cdf(q[:, 1]), .5, atol=1e-7, rtol=0)
                log_means = []
                for nodes in (128, 256):
                    z, w = np.polynomial.legendre.leggauss(nodes)
                    quantiles, added, error = model.quantiles((z+1)/2)
                    log_means.append(np.log2(1+quantiles) @ (w/2))
                    repairs += added
                    residual = max(residual, error)
                mode = model.scale*np.where(c == 0, 0, np.where(c == n, 1,
                        (model.a-1)/(model.a+model.b-2)))
                poisson_q = gamma.ppf([.025, .5, .975], c[:, None]+h)/sf
                for i, count in enumerate(c):
                    rows.append(dict(depth=n, count=int(count), h=h, relative_exposure=exposure,
                        size_factor=sf, point=count/sf, mean=model.mean[i], median=q[i, 1], mode=mode[i],
                        variance=model.variance[i], lower95=q[i, 0], upper95=q[i, 3], upper_one_sided95=q[i, 2],
                        shift_from_point=model.mean[i]-count/sf, low_count_shift=h/sf,
                        poisson_mean=(count+h)/sf, poisson_variance=(count+h)/sf**2,
                        poisson_quantile_max_relative_difference=float(np.max(abs(q[i, [0, 1, 3]]-poisson_q[i])/poisson_q[i])),
                        expected_log2=log_means[1][i], log2_of_mean=np.log2(1+model.mean[i]),
                        log_quadrature_change=log_means[1][i]-log_means[0][i],
                        quantile_repairs_in_batch=repairs, max_cdf_residual_in_batch=residual))
    save(out, 'observed-counts', rows)


def exact_repeated(cfg, out):
    rows, strata = [], []
    for n in cfg['depths']:
        sf = n/cfg['units']
        for abundance in cfg['true_abundances']:
            p = abundance/cfg['units']
            c, w, omitted = binomial_support(n, p, cfg['support_tail_probability'])
            for h in cfg['prior_shapes']:
                model = BinomialBeta(c, n, sf, h)
                q, repairs, residual = model.quantiles([.025, .975])
                cdf = model.cdf(abundance)
                cover = (cdf >= .025) & (cdf <= .975)
                upper = cdf <= .95
                # Exact repeated-sampling moments, independent of truncation.
                total = n+2*h
                bias = cfg['units']*(n*p+h)/total-abundance
                variance_mean = cfg['units']**2*n*p*(1-p)/total**2
                mse = variance_mean+bias**2
                average_variance = cfg['units']**2*((n*p+h)*(n-n*p+h)-n*p*(1-p))/(total**2*(total+1))
                point_mse = cfg['units']**2*p*(1-p)/n
                common = dict(depth=n, true_abundance=abundance, expected_count=n*p, h=h)
                rows.append(dict(**common, mean_bias=bias, mean_rmse=np.sqrt(mse), point_rmse=np.sqrt(point_mse),
                    central95_coverage=float(w @ cover), upper95_coverage=float(w @ upper),
                    expected_width95=float(w @ (q[:, 1]-q[:, 0])),
                    average_kernel_variance=average_variance,
                    mixture_variance=average_variance+variance_mean,
                    mixture_variance_over_point= (average_variance+variance_mean)/point_mse if point_mse else np.nan,
                    expected_kernel_squared_error=average_variance+mse,
                    probability_zero=float(binom.pmf(0, n, p)), omitted_probability=omitted,
                    support_size=len(c), quantile_repairs=repairs, max_cdf_residual=residual))
                for name, mask in [('zero', c == 0), ('one', c == 1), ('2-9', (c >= 2) & (c < 10)),
                                   ('10-99', (c >= 10) & (c < 100)), ('100+', c >= 100)]:
                    probability = w[mask].sum()
                    if probability:
                        weight = w[mask]/probability
                        strata.append(dict(**common, count_stratum=name, stratum_probability=probability,
                            central95_coverage=float(weight @ cover[mask]), upper95_coverage=float(weight @ upper[mask]),
                            mean_bias=float(weight @ (model.mean[mask]-abundance)),
                            expected_width95=float(weight @ (q[mask, 1]-q[mask, 0]))))
        print(f'Exact repeated sampling finished depth {n}', flush=True)
    save(out, 'repeated-counts', rows)
    save(out, 'count-strata', strata)


def mean_se(values):
    values = np.asarray(values, float)
    return float(values.mean()), float(values.std(ddof=1)/np.sqrt(len(values))) if len(values) > 1 else np.nan


def predictive(cfg, out):
    study, rows = cfg['predictive'], []
    for ni, n in enumerate(study['depths']):
        for ai, abundance in enumerate(study['true_abundances']):
            p = abundance/cfg['units']
            for mi, multiplier in enumerate(study['future_depth_multipliers']):
                rng = np.random.default_rng([cfg['seed'], 1, ni, ai, mi])
                size, future_n = study['replicates'], n*multiplier
                count, future = rng.binomial(n, p, size), rng.binomial(future_n, p, size)
                oracle_log = binom.logpmf(future, future_n, p)
                methods = [('oracle', oracle_log, np.full(size, binom.pmf(0, future_n, p))),
                           ('point_plugin', binom.logpmf(future, future_n, count/n), binom.pmf(0, future_n, count/n))]
                for h in cfg['prior_shapes']:
                    model = BinomialBeta(count, n, n/cfg['units'], h)
                    methods.append((f'beta_h{h:g}', model.predictive_logpmf(future, future_n),
                                    model.predictive_zero(future_n)))
                for method, logp, pzero in methods:
                    assert np.all(logp <= 1e-6) and not np.isnan(logp).any()
                    assert np.isfinite(pzero).all() and np.all((pzero >= 0) & (pzero <= 1+1e-12))
                    for name, selected in [('all', np.ones(size, bool)), ('observed_zero', count == 0)]:
                        if not selected.any():
                            continue
                        loss = -logp[selected]
                        impossible = int(np.isinf(loss).sum())
                        mean, se = mean_se(loss) if not impossible else (np.inf, np.nan)
                        brier, brier_se = mean_se((pzero[selected]-(future[selected] == 0))**2)
                        regret = loss+oracle_log[selected]
                        excess, excess_se = mean_se(regret) if not impossible else (np.inf, np.nan)
                        rows.append(dict(depth=n, true_abundance=abundance, expected_count=n*p,
                            future_depth=future_n, method=method, stratum=name, draws=int(selected.sum()),
                            log_loss=mean, log_loss_mc_se=se, impossible_outcomes=impossible,
                            excess_log_loss_vs_oracle=excess, excess_mc_se=excess_se,
                            zero_brier=brier, zero_brier_mc_se=brier_se,
                            predicted_nonzero=float(np.mean(1-pzero[selected])),
                            observed_nonzero=float(np.mean(future[selected] > 0))))
    save(out, 'predictive', rows)


def technical_overdispersion(cfg, out):
    study, rows = cfg['technical_overdispersion'], []
    for ni, n in enumerate(study['depths']):
        for ai, abundance in enumerate(study['true_abundances']):
            p = abundance/cfg['units']
            for vi, cv in enumerate(study['technical_cv']):
                rng = np.random.default_rng([cfg['seed'], 2, ni, ai, vi])
                size = study['replicates']
                concentration = (1-p)/(p*cv**2)-1
                preparation_p = rng.beta(p*concentration, (1-p)*concentration, size)
                counts = rng.binomial(n, preparation_p)
                unique, inverse = np.unique(counts, return_inverse=True)
                for h in cfg['prior_shapes']:
                    model = BinomialBeta(unique, n, n/cfg['units'], h)
                    q, repairs, residual = model.quantiles([.025, .975])
                    for target, truth in [('source', np.full(size, abundance)),
                                          ('preparation', preparation_p*cfg['units'])]:
                        covered = (q[inverse, 0] <= truth) & (truth <= q[inverse, 1])
                        coverage, se = mean_se(covered)
                        rows.append(dict(depth=n, true_abundance=abundance, expected_count=n*p,
                            technical_cv=cv, h=h, target=target, draws=size, central95_coverage=coverage,
                            coverage_mc_se=se, mean_bias=float(np.mean(model.mean[inverse]-truth)),
                            mean_rmse=float(np.sqrt(np.mean((model.mean[inverse]-truth)**2))),
                            expected_width95=float(np.mean(q[inverse, 1]-q[inverse, 0])),
                            quantile_repairs=repairs, max_cdf_residual=residual))
    save(out, 'technical-overdispersion', rows)


def self_consistency(cfg, out):
    rows, bins = [], []
    study = cfg['prior_self_consistency']
    for ni, n in enumerate(study['depths']):
        for hi, h in enumerate(cfg['prior_shapes']):
            rng = np.random.default_rng([cfg['seed'], 3, ni, hi])
            size = study['replicates']
            p = rng.beta(h, h, size)
            count = rng.binomial(n, p)
            rank = betainc(count+h, n-count+h, p)
            cover, se = mean_se((rank >= .025) & (rank <= .975))
            rows.append(dict(depth=n, h=h, draws=size, central95_coverage=cover,
                             coverage_mc_se=se, mean_posterior_cdf=float(rank.mean())))
            hist, edges = np.histogram(rank, bins=np.linspace(0, 1, 11))
            bins.extend(dict(depth=n, h=h, lower=lo, upper=hi, count=int(c))
                        for c,lo,hi in zip(hist, edges[:-1], edges[1:]))
    save(out, 'prior-self-consistency', rows)
    save(out, 'prior-cdf-ranks', bins)


def composition(cfg, out):
    rows = []
    for n in cfg['depths']:
        for g in cfg['gene_universes']:
            # Any complete count allocation has the same sum. Use a deterministic
            # nonuniform integer allocation, including zero categories when shallow.
            counts = np.zeros(g, dtype=int)
            counts[:min(n, g)] = 1
            counts[0] += n-counts.sum()
            for h in cfg['prior_shapes']:
                model = BinomialBeta(counts, n, n, h)
                expected = (n+g*h)/(n+2*h)
                np.testing.assert_allclose(model.mean.sum(), expected, rtol=1e-14)
                rows.append(dict(depth=n, genes=g, h=h, summed_fraction_mean=float(model.mean.sum()),
                    excess_fraction=expected-1, per_gene_prior_pseudocounts=g*h,
                    denominator_prior_pseudocounts=2*h))
    save(out, 'composition', rows)


def mode_contract(cfg, out):
    """Reject a candidate on a failed design constraint before performance testing."""
    rows, libraries, depth_rows = [], [], []
    for index, values in enumerate(cfg['examples']):
        counts = np.asarray(values, dtype=np.int64)
        original = counts.copy()
        counts.flags.writeable = False
        n, g = int(counts.sum()), len(counts)
        joint = uniform_dirichlet_summary(counts)
        representations = [('uniform_dirichlet', joint['a'], joint['b'], joint['joint_mode'])]
        representations += [(f'independent_beta_h{h:g}', counts+h, n-counts+h, None) for h in (1., .5)]
        for name, a, b, joint_mode in representations:
            marginal_mode = beta_marginal_mode(a, b)
            mean = a/(a+b)
            variance = a*b/((a+b)**2*(a+b+1))
            q, repairs, residual = _beta_quantiles(a[:, None], b[:, None], [.025, .975])
            libraries.append(dict(example=index, method=name, genes=g, observed_total=n,
                expected_total_in_count_units=float(n*mean.sum()),
                marginal_mode_sum_in_count_units=float(n*marginal_mode.sum()),
                mean_total_conserved=bool(np.isclose(mean.sum(), 1, atol=1e-12, rtol=0)),
                every_marginal_mode_preserved=bool(np.allclose(n*marginal_mode, counts, rtol=1e-12, atol=1e-12)),
                joint_mode_preserved=None if joint_mode is None else bool(np.allclose(n*joint_mode, counts)),
                positive_zero_variance=bool(np.all(variance[counts == 0] > 0)),
                quantile_repairs=repairs, max_cdf_residual=residual))
            for gene in range(g):
                rows.append(dict(example=index, method=name, gene=gene, observed_count=int(counts[gene]),
                    observed_total=n, joint_mode_in_count_units=np.nan if joint_mode is None else n*joint_mode[gene],
                    marginal_mode_in_count_units=n*marginal_mode[gene], mean_in_count_units=n*mean[gene],
                    variance_in_count_units=n*n*variance[gene], lower95=n*q[gene, 0], upper95=n*q[gene, 1]))
        np.testing.assert_array_equal(counts, original)
        assert counts.dtype == original.dtype and counts.sum() == original.sum()
    for n in cfg['depths']:
        for g in cfg['gene_universes']:
            # A complete deterministic allocation. The mode-retention factor is
            # independent of allocation, so no gene-selection simulation is needed.
            counts = np.zeros(g, dtype=np.int64)
            counts[:min(g, n)] = 1
            counts[0] += n-counts.sum()
            original = counts.copy()
            model = uniform_dirichlet_summary(counts)
            fraction = n/(n+g-2)
            np.testing.assert_allclose(n*model['marginal_mode'], counts*fraction, rtol=2e-15)
            np.testing.assert_allclose(model['mean'].sum(), 1, rtol=2e-15)
            np.testing.assert_array_equal(counts, original)
            depth_rows.append(dict(depth=n, genes=g, marginal_mode_retention=fraction,
                marginal_mode_sum_in_count_units=float(n*model['marginal_mode'].sum()),
                joint_mode_sum_in_count_units=float(n*model['joint_mode'].sum()),
                zero_kernel_mean_cpm=cfg['units']/(n+g)))
    save(out, 'mode-examples', rows)
    save(out, 'mode-contract', libraries)
    save(out, 'mode-depth', depth_rows)
    write(out/'decision.json', dict(candidate='uniform Dirichlet prior with multinomial counts',
        status='REJECTED_FOR_EXACT_MARGINAL_MODE_CONTRACT',
        reason='Joint mode and means conserve the total, but individual marginal modes shift for G>2.',
        scope='Dirichlet family incompatibility; not a general impossibility result.',
        observations='Raw counts, integer dtype and observed totals remain unchanged.',
        next_step='Derive a different uncertainty construction while preserving the exact marginal-mode requirement.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT/'research/protocols/measurement.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.protocol.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.protocol, args.output/'protocol.json')
    sources = [Path(__file__), ROOT/'research/tests/test_measurement.py',
               *sorted((ROOT/'research/wade_research').glob('*.py'))]
    hashes = {str(p.relative_to(ROOT)):digest(p) for p in sources}
    for path in sources:
        dest = args.output/'source'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    write(args.output/'manifest.json', dict(started=datetime.now(timezone.utc).isoformat(),
        checkpoint=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT).decode().strip(),
        sources=hashes, protocol_sha256=digest(args.protocol), python=sys.version,
        numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__,
        input_data=('None: deterministic mode-contract examples and depth grid' if cfg.get('study') == 'mode_contract'
                    else 'None: frozen count grid, analytic probability sums, and new seeded draws')))
    studies = ((mode_contract,) if cfg.get('study') == 'mode_contract' else
               (observed, exact_repeated, predictive, technical_overdispersion, self_consistency, composition))
    for study in studies:
        study(cfg, args.output)
    assert hashes == {str(p.relative_to(ROOT)):digest(p) for p in sources}, 'Sources changed during execution'
    write(args.output/'completion.json', dict(status='MEASUREMENT_DIAGNOSTIC_ONLY',
        finished=datetime.now(timezone.utc).isoformat(),
        claims='No new candidate selected; known exposures; no gene testing or FDR claim',
        outputs={p.name:digest(p) for p in args.output.glob('*') if p.is_file()}))


if __name__ == '__main__':
    main()
